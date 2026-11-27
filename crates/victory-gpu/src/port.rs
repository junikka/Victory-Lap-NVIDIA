//! Renderer port #1: the title's 32-bit graphics heap as the card's own
//! memory (`VICTORY_PORT_CONSTANTS`, off by default).
//!
//! What the title does (game_scarlett_master.exe, the routine at image
//! +0x1e86440, reached at its start): one `XMemVirtualAlloc` of the five
//! sizes it is given together (kind 0x42000: a reservation of 64 KiB
//! pages, flags 7: graphics memory below 4 GiB, 185 MB at 0x34310000 in
//! every run so far), split into five placeholders (`VirtualFree` with
//! `MEM_PRESERVE_PLACEHOLDER`), each turned into graphics memory of its own
//! protection (`XMemVirtualAlloc`, kind 0x20406000, flags 0x27); then ONE
//! `XMemAllocatePhysicalPages` for all their pages and five
//! `XMemMapPhysicalPages`, in order, each from the next page of that one
//! array. `xmem` makes one section of all the pages, so the whole
//! reservation is one stretch of one section: views that follow one
//! another in memory and in the section alike. The title's 174 MB buffer at
//! 0x34e10000 (where its ring of constants lies, and what the skinning
//! passes write through one view of all of it) is in it; it is never
//! tile-mapped (`CopyPageMappings` maps the GPU heap's pages, not these).
//!
//! What the layer did with it: a host copy in the card's memory, the title's
//! writes found and sent up (17 MB a frame in ~600 pieces), what shaders
//! wrote fetched back and merged, and every constant buffer read out of it
//! (~6,000 fills a frame) either waited for flights or done on the card
//! from the copy.
//!
//! What is done here instead: the stretch is mapped once more, at an
//! address of ours (a view of the same section: the same physical pages),
//! and opened as a host heap there (`OpenExistingHeapFromAddress1`); title
//! buffers and windows of title memory in it are buffers placed in that
//! heap ([`crate::shadow::Shadow`]'s in-place form): the card reads and
//! writes the title's bytes where the title wrote them. Nothing is tracked,
//! sent up, fetched back or compared for it, and no compare or hash pass
//! of ours ever runs over it (they are for copies in the card's memory:
//! one such pass over memory shared with the PC removed the device on
//! another tree). No tile mapping is ever made into this heap (that removes
//! the device).
//!
//! Why our own view: a heap opened from an address holds the pages it was
//! opened on: the title could not unmap, decommit or protect them while it
//! stands. Opened on our view, the title's own views stay free to change.
//! When the title maps other pages at such an address, what is in place
//! there is let go (`crate::inplace::title_memory_freeing`) and the region
//! forgotten; the view of ours stays (shortcut: it is never unmapped, a
//! few hundred MB of address space at most).
//!
//! What it assumes: the CPU steps of ours that read this memory at
//! submission (fills of constant buffers, the title's buffer copies done on
//! the CPU) read what the title's CPU wrote, not what a submission still in
//! the air writes there: in place nothing tells the flights' writes, so
//! nothing waits for them. The title's constants are its CPU's; how many
//! such steps there are is on the frame-rate line ([`log`]).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Memory as mem;

use crate::inplace::{Placed, cover, stretch};

const K64: u64 = 64 << 10;

/// The ported memory lies below this: the title's graphics memory for
/// 32-bit addresses, which only that heap is.
const LOW: u64 = 1 << 32;

/// The most bytes of one region: its pages are locked into the PC's memory
/// while the heap stands.
const MOST: u64 = 512 << 20;

/// `VICTORY_PORT_CONSTANTS`: `1` turns renderer port #1 on (title buffers
/// and windows of title memory in the title's 32-bit graphics memory, view
/// memory that is one stretch of a section, used in place); unset or
/// anything else: off, everything as before.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Read once, here: the switch of renderer port #1.
    *ON.get_or_init(|| std::env::var("VICTORY_PORT_CONSTANTS").is_ok_and(|v| v.trim() == "1"))
}

/// A stretch of title memory opened as a heap through a view of ours.
struct Region {
    start: u64,
    end: u64,
    heap: ID3D12Heap,
    /// Our view the heap was opened on.
    view: usize,
}

// SAFETY: D3D12 heaps are free-threaded.
unsafe impl Send for Region {}

static REGIONS: Mutex<Vec<Region>> = Mutex::new(Vec::new());

/// 64-bit regions forgotten ([`forget`]) whose heap something may still
/// hold (a list in the air, a slot): their view is unmapped once nothing
/// does ([`sweep_retired`]).
static RETIRED: Mutex<Vec<Region>> = Mutex::new(Vec::new());

/// The lowest start and highest end of every region (both 0: none), for a
/// quick look on the paths that run thousands of times a frame.
static BOUNDS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Since the start: regions opened, their bytes; title buffers and windows
/// placed; refused because the memory is not one stretch of a section,
/// not committed, refused by the host; regions forgotten because the title
/// mapped other pages there. Then, since the last line: fills read out of
/// ported memory on the CPU, their bytes, buffer copies on the CPU that
/// touch it, their bytes. Then, since the start, 64-bit regions whose view
/// was unmapped ([`sweep_retired`]). (8: refused over
/// `VICTORY_PORT_SECTIONS_MB`.)
static STATS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];

/// `VICTORY_PORT_SECTIONS` (after Voltexs/VIctory-RTXNvidia 295fd537): title
/// buffers in the title's 64-bit memory, where it is one stretch of a
/// section (its page pool, view memory), used in place as port 1 uses the
/// 32-bit heap: a view of ours of the same pages, opened as a host heap,
/// and the buffer placed in it. No host copy, nothing sent up, compared or
/// fetched back for it.
/// - unset, `0` or anything else: off, as before (only the 32-bit heap,
///   with `VICTORY_PORT_CONSTANTS=1`);
/// - `noncard`: only the buffers that would not get a copy in the card's
///   own memory (`crate::shadow`'s card limits); the big ones stay there;
/// - `1` or `all`: every such buffer, big or not (the card then reads the
///   big ones over the bus too).
///
/// Windows outside buffers above 4 GiB are not placed by this switch (only
/// port 1's below; [`windows_high`] is theirs). At most [`sections_budget`] bytes are open at once, each region
/// [`HIGH_MOST`] at most; a region the title maps other pages over is
/// forgotten and its view of ours unmapped once nothing holds its heap
/// ([`sweep_retired`]; upstream kept such views mapped for good).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Sections {
    Off,
    All,
    NonCard,
}

/// [`Sections`] for a setting.
fn sections_value(value: Option<&str>) -> Sections {
    match value.map(str::trim) {
        Some("1") | Some("all") => Sections::All,
        Some("noncard") => Sections::NonCard,
        _ => Sections::Off,
    }
}

/// The `VICTORY_PORT_SECTIONS` setting, read once.
pub(crate) fn sections() -> Sections {
    static MODE: std::sync::OnceLock<Sections> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| sections_value(std::env::var("VICTORY_PORT_SECTIONS").ok().as_deref()))
}

/// The most bytes of 64-bit regions open at once (`VICTORY_PORT_SECTIONS_MB`,
/// 1024): opening locks their pages into the PC's memory.
fn sections_budget() -> u64 {
    static BUDGET: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("VICTORY_PORT_SECTIONS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(1024) << 20
    })
}

/// The most bytes of one 64-bit region: a longer stretch opens only what
/// was asked for, not hundreds of megabytes the title gave no buffer.
const HIGH_MOST: u64 = 64 << 20;

/// Bytes of 64-bit regions open now (retired ones included until their
/// view is unmapped).
static HIGH_BYTES: AtomicU64 = AtomicU64::new(0);

/// `VICTORY_PORT_WINDOWS=1` (off unless `1`): windows of title memory
/// outside buffers above 4 GiB are used in place too, where the window is
/// one stretch of a section, as [`Sections`] does for title buffers (it
/// leaves windows out, as its upstream does). A window is the PC's memory
/// whether it is a copy or the title's own pages, and the card reads it
/// over the bus either way; in place the copy is gone (2.9 GB of windows
/// lay over 2.8 GB of title memory on the test path) and so is the
/// copying into it. A window that is not one stretch, not committed whole
/// or over [`sections_budget`] gets its copy as before.
///
/// What it assumes is what port 1 assumes below 4 GiB: a submission in the
/// air reads the title's bytes as they are when the card gets to them, not
/// as they were when the list was prepared (the console's own behaviour).
/// With the title's streaming remapping pages in a window's cell, the
/// window and its region go and are made again ([`forget`]): the 64-bit
/// line counts the regions retired.
pub(crate) fn windows_high() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| windows_value(std::env::var("VICTORY_PORT_WINDOWS").ok().as_deref()))
}

/// [`windows_high`] for a setting.
fn windows_value(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

/// Since the start: windows above 4 GiB placed ([`windows_high`]), their
/// bytes.
static HIGH_WINDOWS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether `[at, at + len)` may be ported: in the title's 32-bit graphics
/// memory.
#[cfg(test)]
pub(crate) fn wanted(at: u64, len: u64) -> bool {
    wanted_for(at, len, false, false, on(), Sections::Off, false)
}

/// Whether `[at, at + len)` may be ported, with port 1 on (`constants`) and
/// `sections`: below 4 GiB with port 1, as before; above, a title buffer
/// with [`Sections`], `card` saying whether it would get a copy in the
/// card's memory, and a window only with `windows` ([`windows_high`]).
fn wanted_for(at: u64, len: u64, window: bool, card: bool, constants: bool, sections: Sections, windows: bool) -> bool {
    let Some(end) = at.checked_add(len).filter(|_| len > 0) else {
        return false;
    };
    if end <= LOW {
        return constants;
    }
    if at < LOW {
        return false;
    }
    if window {
        return windows;
    }
    match sections {
        Sections::Off => false,
        Sections::All => true,
        Sections::NonCard => !card,
    }
}

/// Title memory `[start, start + len)` in place, if [`wanted`] and the
/// memory is one stretch of a section: a buffer placed in the region's
/// heap. `None` leaves the caller to do what it did before.
pub(crate) fn place_for(start: u64, len: u64, window: bool) -> Option<Placed> {
    place_for_kind(start, len, window, false)
}

/// [`place_for`] of a title buffer that would get a copy in the card's
/// memory if `card` ([`Sections::NonCard`] leaves those alone).
pub(crate) fn place_for_kind(start: u64, len: u64, window: bool, card: bool) -> Option<Placed> {
    if !wanted_for(start, len, window, card, on(), sections(), windows_high()) {
        return None;
    }
    // Above 4 GiB most memory is not one stretch of a section (private
    // memory, scattered pages): asked here, without the regions' lock, and
    // not counted as refused.
    if start >= LOW && !matches!(cover(start as usize, len as usize), crate::inplace::Cover::Run(..)) {
        return None;
    }
    let host = crate::host::host()?;
    match place_in(&host.device, start, len) {
        Ok(placed) => {
            crate::inplace::counted(window, len);
            if window && start >= LOW {
                HIGH_WINDOWS[0].fetch_add(1, Ordering::Relaxed);
                HIGH_WINDOWS[1].fetch_add(len, Ordering::Relaxed);
            }
            let n = STATS[2].fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 32 || n.is_power_of_two() {
                tracing::info!(
                    start = format_args!("{start:#x}"),
                    len,
                    window,
                    "renderer port 1: title memory in place, read and written by the card where the title has it"
                );
            }
            Some(placed)
        }
        Err((at, why)) => {
            let n = STATS[at].fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 8 || n.is_power_of_two() {
                tracing::warn!(start = format_args!("{start:#x}"), len, window, why, "renderer port 1: title memory not in place, a copy as before");
            }
            None
        }
    }
}

/// [`place_for`]'s work: the region holding the pages of `[start, start +
/// len)`, opened if none does, and a buffer placed in it. `Err` with the
/// statistic to count and why.
pub(crate) fn place_in(device: &ID3D12Device, start: u64, len: u64) -> Result<Placed, (usize, String)> {
    let at = start & !(K64 - 1);
    let end = (start + len.max(1)).next_multiple_of(K64);
    let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    let i = match regions.iter().position(|r| r.start <= at && end <= r.end) {
        Some(i) => i,
        None => {
            let region = open_region(device, at, end)?;
            regions.push(region);
            let n = regions.len() - 1;
            let r = &regions[n];
            STATS[0].fetch_add(1, Ordering::Relaxed);
            STATS[1].fetch_add(r.end - r.start, Ordering::Relaxed);
            bounds(&regions);
            n
        }
    };
    let region = &regions[i];
    let buffer = crate::inplace::place(device, &region.heap, at - region.start, end - at).map_err(|e| (6, format!("no buffer placed in the region: {e}")))?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (start - at);
    Ok(Placed { buffer, heap: region.heap.clone(), gpu, bias: start - at })
}

/// Opens the stretch around `[at, end)` (64 KiB aligned) as a heap through
/// a view of ours of the same pages.
fn open_region(device: &ID3D12Device, at: u64, end: u64) -> Result<Region, (usize, String)> {
    let Some((first, bytes, section, offset)) = stretch(at as usize, (end - at) as usize) else {
        return Err((3, format!("not one stretch of a section: {:?}", cover(at as usize, (end - at) as usize))));
    };
    // The whole stretch, kept to what may be ported: the next buffer in it
    // finds the region made.
    let (mut first, mut bytes, mut offset) = (first as u64, bytes as u64, offset);
    if first < LOW && first + bytes > LOW {
        bytes = LOW.saturating_sub(first);
    }
    let high = at >= LOW;
    if bytes > region_most(high) {
        // Only what was asked for.
        offset += at - first;
        (first, bytes) = (at, end - at);
    }
    if bytes > MOST {
        return Err((3, format!("{bytes:#x} bytes, over the most of a region")));
    }
    if high && !high_fits(HIGH_BYTES.load(Ordering::Relaxed), bytes, sections_budget()) {
        return Err((8, format!("{bytes:#x} bytes over VICTORY_PORT_SECTIONS_MB")));
    }
    if first > at || first + bytes < end || offset % K64 != 0 {
        return Err((3, format!("the stretch {first:#x}+{bytes:#x} does not hold it")));
    }
    // SAFETY: the section of a view the title has; the runtime keeps its
    // handle open while a view of it is there, and the range is inside it.
    let view = crate::remap::count::kernel(crate::remap::count::Kernel::Map, || unsafe {
        mem::MapViewOfFile(section as _, mem::FILE_MAP_WRITE, (offset >> 32) as u32, offset as u32, bytes as usize)
    })
    .Value as usize;
    if view == 0 {
        return Err((5, format!("no view of the stretch: {}", std::io::Error::last_os_error())));
    }
    let unmap = || {
        // SAFETY: our own view, nothing opened on it.
        unsafe { mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
    };
    if !committed(view, bytes as usize) {
        unmap();
        return Err((4, "pages of the stretch are not committed".to_owned()));
    }
    match crate::inplace::open(device, crate::inplace::From::Part(view, bytes as usize)) {
        Ok(heap) => {
            tracing::info!(
                start = format_args!("{first:#x}"),
                bytes,
                section = format_args!("{section:#x}"),
                offset = format_args!("{offset:#x}"),
                view = format_args!("{view:#x}"),
                "renderer port 1: a stretch of the title's graphics memory opened as a host heap through a view of ours"
            );
            // shortcut: below 4 GiB the view stays mapped for good (the
            // heap holds its pages while anything names it); above, it is
            // unmapped once the region is forgotten and nothing holds the
            // heap ([`sweep_retired`]).
            if high {
                HIGH_BYTES.fetch_add(bytes, Ordering::Relaxed);
            }
            Ok(Region { start: first, end: first + bytes, heap, view })
        }
        Err(e) => {
            unmap();
            Err((5, format!("refused by the host: {e}")))
        }
    }
}

/// The most bytes a region opens of a longer stretch: below 4 GiB the
/// whole 32-bit heap ([`MOST`]); above, [`HIGH_MOST`].
fn region_most(high: bool) -> u64 {
    if high { HIGH_MOST } else { MOST }
}

/// Whether a 64-bit region of `bytes` may open with `open` bytes of them
/// open already and `budget` the most.
fn high_fits(open: u64, bytes: u64, budget: u64) -> bool {
    open.saturating_add(bytes) <= budget
}

/// Whether every page of `[at, at + len)` is committed.
fn committed(at: usize, len: usize) -> bool {
    let mut here = at;
    while here < at + len {
        // SAFETY: plain data filled in by the query.
        let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { mem::VirtualQuery(here as *const _, &mut info, size_of_val(&info)) } == 0 || info.State != mem::MEM_COMMIT {
            return false;
        }
        here = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    }
    true
}

fn bounds(regions: &[Region]) {
    let lo = regions.iter().map(|r| r.start).min().unwrap_or(0);
    let hi = regions.iter().map(|r| r.end).max().unwrap_or(0);
    BOUNDS[0].store(lo, Ordering::Relaxed);
    BOUNDS[1].store(hi, Ordering::Relaxed);
}

/// The title maps other pages at `[at, at + len)`: the regions over it are
/// forgotten (what is placed in them is let go by the caller).
pub(crate) fn forget(at: usize, len: usize) {
    let (at, end) = (at as u64, (at as u64).saturating_add(len as u64));
    if BOUNDS[1].load(Ordering::Relaxed) <= at || BOUNDS[0].load(Ordering::Relaxed) >= end {
        return;
    }
    let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    let (kept, gone): (Vec<Region>, Vec<Region>) =
        std::mem::take(&mut *regions).into_iter().partition(|r| r.end <= at || r.start >= end);
    *regions = kept;
    STATS[7].fetch_add(gone.len() as u64, Ordering::Relaxed);
    bounds(&regions);
    drop(regions);
    // Below 4 GiB as before: the region goes, its view stays.
    let high: Vec<Region> = gone.into_iter().filter(|r| r.start >= LOW).collect();
    if !high.is_empty() {
        RETIRED.lock().unwrap_or_else(|e| e.into_inner()).extend(high);
    }
    sweep_retired();
}

/// Unmaps the views of the retired 64-bit regions whose heap nothing but
/// the region holds any more (the buffers placed in it are gone: no list,
/// no slot), and takes their bytes off [`HIGH_BYTES`]. Called when a
/// region is forgotten and once a second ([`log`]).
fn sweep_retired() {
    let free: Vec<Region> = {
        let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
        if retired.is_empty() {
            return;
        }
        let (free, held): (Vec<Region>, Vec<Region>) =
            std::mem::take(&mut *retired).into_iter().partition(|r| crate::host::references(&r.heap) <= 1);
        *retired = held;
        free
    };
    for Region { start, end, heap, view } in free {
        drop(heap);
        // SAFETY: our own view; the heap opened on it went with its last
        // reference just now, so nothing is opened on it any more.
        crate::remap::count::kernel(crate::remap::count::Kernel::Unmap, || unsafe {
            mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ })
        });
        let _ = HIGH_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |b| Some(b.saturating_sub(end - start)));
        STATS[13].fetch_add(1, Ordering::Relaxed);
    }
}

/// Whether `[at, at + len)` touches ported memory (a quick look).
fn touches(at: usize, len: usize) -> bool {
    let hi = BOUNDS[1].load(Ordering::Relaxed);
    hi != 0 && (at as u64) < hi && (at as u64).saturating_add(len as u64) > BOUNDS[0].load(Ordering::Relaxed)
}

/// A fill of upload memory read `len` bytes at `src` on the CPU.
pub(crate) fn filled(src: usize, len: usize) {
    if touches(src, len) {
        STATS[9].fetch_add(1, Ordering::Relaxed);
        STATS[10].fetch_add(len as u64, Ordering::Relaxed);
    }
}

/// A title buffer copy was done on the CPU.
pub(crate) fn copied(dst: usize, src: usize, len: usize) {
    if touches(dst, len) || touches(src, len) {
        STATS[11].fetch_add(1, Ordering::Relaxed);
        STATS[12].fetch_add(len as u64, Ordering::Relaxed);
    }
}

/// For the frame-rate line: what is in place, and what of ours read or
/// wrote it on the CPU per frame.
pub(crate) fn log(frames: f64) {
    let sections = sections();
    let windows = windows_high();
    if sections != Sections::Off || windows {
        sweep_retired();
        let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
        let high = [
            HIGH_BYTES.load(Ordering::Relaxed),
            sections_budget(),
            RETIRED.lock().unwrap_or_else(|e| e.into_inner()).len() as u64,
            s[13],
            s[8],
        ];
        let placed = [HIGH_WINDOWS[0].load(Ordering::Relaxed), HIGH_WINDOWS[1].load(Ordering::Relaxed)];
        tracing::info!("{}", high_line(sections, windows, high, placed));
    }
    if !on() && sections == Sections::Off && !windows {
        return;
    }
    let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    let per: Vec<f64> = STATS[9..13].iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "renderer port 1 (VICTORY_PORT_CONSTANTS={}): {} regions of the title's graphics memory in place ({} MB), {} title buffers and windows placed in them \
         (no host copy, nothing sent up or fetched back for them); per frame on the CPU: {:.0} fills read out of them ({:.1} KB), {:.1} buffer copies \
         touching them ({:.1} KB); since the start refused: {} not one stretch of a section, {} not committed, {} by the host, {} no buffer; {} regions \
         forgotten (other pages mapped there)",
        u8::from(on()),
        s[0],
        s[1] >> 20,
        s[2],
        per[0],
        per[1] / 1024.0,
        per[2],
        per[3] / 1024.0,
        s[3],
        s[4],
        s[5],
        s[6],
        s[7],
    );
}

/// The 64-bit line. `high`: bytes of regions open, the budget, retired
/// regions, views unmapped, refusals over the budget; `placed`: windows
/// above 4 GiB placed since the start and their bytes. It reads as before
/// while `VICTORY_PORT_WINDOWS` is unset.
fn high_line(sections: Sections, windows: bool, high: [u64; 5], placed: [u64; 2]) -> String {
    let mut line = format!(
        "renderer port 1, 64-bit sections (VICTORY_PORT_SECTIONS={sections:?}): {} MB of 64-bit regions open (at most {} MB), {} retired \
         regions waiting for their heap to go, {} views unmapped; since the start refused over the budget {}",
        high[0] >> 20,
        high[1] >> 20,
        high[2],
        high[3],
        high[4],
    );
    if windows {
        line.push_str(&format!(
            "; windows above 4 GiB in place (VICTORY_PORT_WINDOWS=1), since the start: {} placed, {} MB with no copy in the PC's memory",
            placed[0],
            placed[1] >> 20,
        ));
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the title's 32-bit memory is asked for.
    #[test]
    fn only_memory_below_4_gib_is_wanted_with_the_switch() {
        // The switch is read once from the environment; tests do not set
        // it, so nothing is wanted.
        if !on() {
            assert!(!wanted(0x3431_0000, 0x1000));
        }
        let low = |at: u64, len: u64| at.checked_add(len).is_some_and(|end| end <= LOW && len > 0);
        assert!(low(0x34e1_0000, 0xae0_0000));
        assert!(!low(0xffff_0000, 0x2_0000));
        assert!(!low(0xe_1c80_0000, 0x1000));
        assert!(!low(u64::MAX, 2));
    }

    /// `VICTORY_PORT_SECTIONS`: unset is off and wants nothing above 4 GiB;
    /// below 4 GiB only port 1 decides, as before; windows above are never
    /// wanted; `noncard` leaves the buffers the card would take.
    #[test]
    fn sections_want_only_64_bit_title_buffers_by_their_mode() {
        assert_eq!(sections_value(None), Sections::Off);
        assert_eq!(sections_value(Some("0")), Sections::Off);
        assert_eq!(sections_value(Some("yes")), Sections::Off);
        assert_eq!(sections_value(Some(" 1 ")), Sections::All);
        assert_eq!(sections_value(Some("all")), Sections::All);
        assert_eq!(sections_value(Some("noncard")), Sections::NonCard);
        let high = 0xe_1c80_0000u64;
        for mode in [Sections::Off, Sections::All, Sections::NonCard] {
            // Below 4 GiB: port 1 alone, whatever the mode.
            assert!(wanted_for(0x34e1_0000, 0x1000, false, false, true, mode, false));
            assert!(wanted_for(0x34e1_0000, 0x1000, true, true, true, mode, false));
            assert!(!wanted_for(0x34e1_0000, 0x1000, false, false, false, mode, false));
            // Across 4 GiB: never.
            assert!(!wanted_for(0xffff_0000, 0x2_0000, false, false, true, mode, false));
            // A window above 4 GiB: never.
            assert!(!wanted_for(high, 0x1000, true, false, true, mode, false));
            assert!(!wanted_for(high, 0, false, false, true, mode, false));
        }
        assert!(!wanted_for(high, 0x1000, false, false, true, Sections::Off, false));
        assert!(wanted_for(high, 0x1000, false, false, false, Sections::All, false));
        assert!(wanted_for(high, 0x1000, false, true, false, Sections::All, false));
        assert!(wanted_for(high, 0x1000, false, false, false, Sections::NonCard, false));
        assert!(!wanted_for(high, 0x1000, false, true, false, Sections::NonCard, false));
        assert!(!wanted_for(u64::MAX - 1, 4, false, false, true, Sections::All, false));
    }

    /// Windows above 4 GiB are wanted only with `VICTORY_PORT_WINDOWS=1`,
    /// whatever the sections' mode; buffers and everything below 4 GiB are
    /// as without it.
    #[test]
    fn windows_above_4_gib_are_wanted_only_with_their_switch() {
        assert!(!windows_value(None));
        assert!(!windows_value(Some("0")));
        assert!(!windows_value(Some("yes")));
        assert!(!windows_value(Some("all")));
        assert!(windows_value(Some(" 1 ")));
        let high = 0xe_1c80_0000u64;
        for mode in [Sections::Off, Sections::All, Sections::NonCard] {
            for card in [false, true] {
                for constants in [false, true] {
                    // A window above: by the switch alone.
                    assert!(!wanted_for(high, 4 << 20, true, card, constants, mode, false));
                    assert!(wanted_for(high, 4 << 20, true, card, constants, mode, true));
                    // A buffer above, and anything below: the switch changes nothing.
                    for window in [false, true] {
                        assert_eq!(
                            wanted_for(0x34e1_0000, 0x1000, window, card, constants, mode, true),
                            wanted_for(0x34e1_0000, 0x1000, window, card, constants, mode, false)
                        );
                    }
                    assert_eq!(
                        wanted_for(high, 0x1000, false, card, constants, mode, true),
                        wanted_for(high, 0x1000, false, card, constants, mode, false)
                    );
                }
            }
            // Nothing across 4 GiB, nothing empty, nothing that wraps.
            assert!(!wanted_for(0xffff_0000, 0x2_0000, true, false, true, mode, true));
            assert!(!wanted_for(high, 0, true, false, true, mode, true));
            assert!(!wanted_for(u64::MAX - 1, 4, true, false, true, mode, true));
        }
    }

    /// The 64-bit line reads as before without the windows' switch and
    /// names the windows placed with it.
    #[test]
    fn the_64_bit_line_names_windows_only_with_their_switch() {
        let high = [96 << 20, 1024 << 20, 2, 5, 7];
        let before = high_line(Sections::All, false, high, [0, 0]);
        assert_eq!(
            before,
            "renderer port 1, 64-bit sections (VICTORY_PORT_SECTIONS=All): 96 MB of 64-bit regions open (at most 1024 MB), 2 retired \
             regions waiting for their heap to go, 5 views unmapped; since the start refused over the budget 7"
        );
        let with = high_line(Sections::Off, true, high, [31, 124 << 20]);
        assert!(with.starts_with("renderer port 1, 64-bit sections (VICTORY_PORT_SECTIONS=Off): 96 MB"), "{with}");
        assert!(with.ends_with("VICTORY_PORT_WINDOWS=1), since the start: 31 placed, 124 MB with no copy in the PC's memory"), "{with}");
    }

    /// A 64-bit region opens 64 MiB of a stretch at most and only within
    /// the budget; the 32-bit heap keeps its 512 MiB.
    #[test]
    fn sections_regions_keep_to_64_mib_and_the_budget() {
        assert_eq!(region_most(false), 512 << 20);
        assert_eq!(region_most(true), 64 << 20);
        assert!(high_fits(0, 64 << 20, 1024 << 20));
        assert!(high_fits(960 << 20, 64 << 20, 1024 << 20));
        assert!(!high_fits(961 << 20, 64 << 20, 1024 << 20));
        assert!(!high_fits(0, 1, 0));
        // No wrap past the top: a sum that would overflow stays over a budget below it.
        assert!(!high_fits(u64::MAX, 1, u64::MAX - 1));
    }

    /// A stretch of a reserved section, mapped as `xmem` maps physical
    /// pages (views of consecutive pages in turn, committed), is opened
    /// through a view of ours: the card reads the bytes the title's views
    /// show, at the buffer's offset, and its writes show in them; a second
    /// buffer in the stretch is placed in the same region.
    #[test]
    fn warp_a_stretch_of_a_reserved_section_is_used_in_place_through_a_view_of_ours() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = crate::inplace::tests::warp() else { return };
        if windows::core::Interface::cast::<ID3D12Device13>(&device).is_err() {
            return;
        }
        const K: usize = 64 << 10;
        // SAFETY (the whole test): the section, placeholder and views made
        // here.
        unsafe {
            let process = windows_sys::Win32::System::Threading::GetCurrentProcess();
            let section = mem::CreateFileMappingW(
                windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                std::ptr::null(),
                mem::PAGE_READWRITE | mem::SEC_RESERVE,
                0,
                8 * K as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let hole = mem::VirtualAlloc2(process, std::ptr::null(), 8 * K, mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER, mem::PAGE_NOACCESS, std::ptr::null_mut(), 0)
                .cast::<u8>();
            assert!(!hole.is_null());
            // Two views, pages 0..3 and 3..8 of the section, in turn.
            mem::VirtualFree(hole.cast(), 3 * K, mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER);
            let mut views = Vec::new();
            for (page, n) in [(0usize, 3usize), (3, 5)] {
                let view = mem::MapViewOfFile3(section, process, hole.add(page * K).cast(), (page * K) as u64, n * K, mem::MEM_REPLACE_PLACEHOLDER, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
                assert!(!view.Value.is_null(), "{}", std::io::Error::last_os_error());
                assert!(!mem::VirtualAlloc(view.Value, n * K, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());
                crate::inplace::title_memory_view(view.Value as usize, n * K, section as usize, (page * K) as u64);
                views.push(view);
            }
            let base = hole as u64;
            let (start, len) = (base + K as u64 + 0x40, 5 * K as u64);
            let placed = place_in(&device, start, len).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert_eq!(placed.bias, 0x40);
            let mut rig = crate::inplace::Rig::new(&device).unwrap();
            for (i, b) in std::slice::from_raw_parts_mut(start as *mut u8, len as usize).iter_mut().enumerate() {
                *b = (i * 7 + 3) as u8;
            }
            let got = rig.fetch(&placed.buffer, placed.bias, len as usize).unwrap();
            assert!(got == std::slice::from_raw_parts(start as *const u8, len as usize), "the card reads the title's bytes, across both views");
            rig.store(&placed.buffer, placed.bias + 3 * K as u64, &[0x5a; 256]).unwrap();
            assert!(std::slice::from_raw_parts((start + 3 * K as u64) as *const u8, 256).iter().all(|&b| b == 0x5a), "the card's write shows in the title's view");
            // Another buffer in the same stretch: the same region.
            let regions = REGIONS.lock().unwrap().len();
            let second = place_in(&device, base + 6 * K as u64, 100).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert_eq!(REGIONS.lock().unwrap().len(), regions, "one region for the stretch");
            assert_eq!(second.bias, 0);
            // The title's views may change under the heap.
            drop((placed, second));
            forget(base as usize, 8 * K);
            for view in views {
                assert!(mem::UnmapViewOfFile2(process, view, mem::MEM_PRESERVE_PLACEHOLDER) != 0, "the title's own view can be unmapped");
            }
            crate::inplace::views_gone(base as usize, 8 * K);
            mem::VirtualFree(hole.cast(), 0, mem::MEM_RELEASE);
            mem::VirtualFree(hole.add(3 * K).cast(), 0, mem::MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }
}
