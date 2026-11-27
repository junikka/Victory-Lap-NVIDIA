//! Lets the pages of mapped files go from the process's memory, again and
//! again: `VICTORY_MAPPED_TRIM=<seconds>` (unset or `0`: off; `1`: every
//! 60 s; 10 at least).
//!
//! The memory map of two sessions shows "files and other mappings at host
//! addresses 7,386 MB (3,817 MB in memory)" beside a working set of
//! 21.5 GB on a PC of 32 GB with 2.5 to 3.7 GB free: the driver's own cache
//! files, the pipeline library's file, and what else the host maps. The
//! library's pages are let go once, after the early loads
//! (`pipeline::library::trim`); the others stay until Windows takes them,
//! and Windows takes pages by age, the title's own among them.
//!
//! A page of a file that is let go stays in Windows' file cache while
//! nothing else wants the memory (it then counts as available), and comes
//! back without the disk on the next read. Only views of files at host
//! addresses go: a section without a file (the title's pages, a shared
//! heap) would have to be written to the paging file first, and programs
//! and libraries are left to Windows.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

/// Console addresses lie below this (the title's own mappings: not ours
/// to judge).
const HOST: usize = 1 << 40;

/// What `VICTORY_MAPPED_TRIM` says: seconds between passes, 0 for none.
pub(crate) fn period(value: Option<&str>) -> u64 {
    match value.map(str::trim) {
        None | Some("") | Some("0") => 0,
        Some(v) => match v.parse::<u64>() {
            Ok(1) => 60,
            Ok(n) => n.max(10),
            Err(_) => 0,
        },
    }
}

fn every() -> u64 {
    static EVERY: OnceLock<u64> = OnceLock::new();
    *EVERY.get_or_init(|| {
        let every = period(std::env::var("VICTORY_MAPPED_TRIM").ok().as_deref());
        if every > 0 {
            tracing::info!(every, "mapped files' pages are let go from memory every {every} s (VICTORY_MAPPED_TRIM)");
        }
        every
    })
}

/// Whether a region's pages are let go: committed, a mapping (no program
/// or library, no private memory), of a file. (The walk begins at the
/// host's addresses, [`HOST`]: the title's own mappings lie below.)
pub(crate) fn goes(committed: bool, mapping: bool, file: bool) -> bool {
    committed && mapping && file
}

/// Passes made, and bytes of views let go in the last pass and in all.
static DONE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// For the memory lines: passes, MB of views in the last pass.
#[cfg(feature = "diagnostics")]
pub(crate) fn done() -> [u64; 2] {
    [DONE[0].load(Relaxed), DONE[1].load(Relaxed) >> 20]
}

/// One pass over the address space from `from` on: the views of files
/// there are let go from memory. Says how many views and bytes.
pub(crate) fn pass(from: usize) -> (u64, u64) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery, VirtualUnlock,
    };
    use windows_sys::Win32::System::ProcessStatus::K32GetMappedFileNameW;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let (mut views, mut bytes) = (0u64, 0u64);
    // The last allocation asked about, and whether it is a file's.
    let mut last = (usize::MAX, false);
    let mut at = from;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
            break;
        }
        let base = m.AllocationBase as usize;
        let (committed, mapping) = (m.State == MEM_COMMIT, m.Type == MEM_MAPPED);
        if committed && mapping {
            if last.0 != base {
                let mut name = [0u16; 520];
                // SAFETY: our own process and a buffer of that many
                // characters (a section without a file has no name).
                let n = unsafe {
                    K32GetMappedFileNameW(GetCurrentProcess(), base as *const _, name.as_mut_ptr(), name.len() as u32)
                };
                last = (base, n != 0);
            }
            if goes(committed, mapping, last.1) {
                // SAFETY: a committed region of a view, as just asked.
                // Unlocking pages that are not locked fails, as expected,
                // and takes them out of the working set: the documented
                // way to let pages of a mapped file go. A view unmapped
                // meanwhile makes the call fail and nothing else.
                unsafe { VirtualUnlock(m.BaseAddress, m.RegionSize) };
                views += 1;
                bytes += m.RegionSize as u64;
            }
        }
        let next = (m.BaseAddress as usize).saturating_add(m.RegionSize);
        if next <= at {
            break;
        }
        at = next;
    }
    (views, bytes)
}

/// Called once a second by the host's sweep: starts a pass on a thread of
/// its own when one is due.
pub(crate) fn tick() {
    let every = every();
    if every == 0 {
        return;
    }
    static NEXT: AtomicU64 = AtomicU64::new(0);
    static BUSY: AtomicBool = AtomicBool::new(false);
    let now = crate::host::now_ms();
    let next = NEXT.load(Relaxed);
    // (The first pass after the first period: the start reads its files.)
    if next == 0 {
        NEXT.store(now + every * 1000, Relaxed);
        return;
    }
    if now < next || BUSY.swap(true, Relaxed) {
        return;
    }
    NEXT.store(now + every * 1000, Relaxed);
    let work = || {
        let started = std::time::Instant::now();
        let (views, bytes) = pass(HOST);
        DONE[0].fetch_add(1, Relaxed);
        DONE[1].store(bytes, Relaxed);
        DONE[2].fetch_add(bytes, Relaxed);
        tracing::info!(
            "mapped files let go from memory (VICTORY_MAPPED_TRIM): {views} views of {} MB in {} ms",
            bytes >> 20,
            started.elapsed().as_millis()
        );
        BUSY.store(false, Relaxed);
    };
    if std::thread::Builder::new().name("victory-mapped-trim".into()).spawn(work).is_err() {
        BUSY.store(false, Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_gives_a_period_and_unset_is_off() {
        assert_eq!(period(None), 0);
        assert_eq!(period(Some("0")), 0);
        assert_eq!(period(Some("off")), 0);
        assert_eq!(period(Some("1")), 60);
        assert_eq!(period(Some("5")), 10);
        assert_eq!(period(Some(" 120 ")), 120);
    }

    #[test]
    fn only_views_of_files_go() {
        assert!(goes(true, true, true));
        assert!(!goes(false, true, true), "reserved");
        assert!(!goes(true, false, true), "private memory, programs and libraries");
        assert!(!goes(true, true, false), "a section without a file");
    }

    /// A file's view that was read is in memory; after a pass it is not,
    /// and reads the same again. A section without a file stays.
    #[test]
    fn a_pass_lets_a_file_s_pages_go_and_they_read_the_same() {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, FILE_MAP_WRITE, MapViewOfFile, PAGE_READONLY, PAGE_READWRITE,
        };
        use windows_sys::Win32::System::ProcessStatus::K32QueryWorkingSetEx;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const LEN: usize = 1 << 20;
        let path = std::env::temp_dir().join(format!("victory-mapped-trim-{}.bin", std::process::id()));
        let bytes: Vec<u8> = (0..LEN).map(|i| (i / 4096) as u8 ^ 0x5a).collect();
        std::fs::write(&path, &bytes).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let in_memory = |at: usize| {
            let mut asked = [at, 0usize];
            // SAFETY: our own process and one entry of two words.
            assert_ne!(unsafe { K32QueryWorkingSetEx(GetCurrentProcess(), asked.as_mut_ptr().cast(), 16) }, 0);
            asked[1] & 1 != 0
        };
        // SAFETY: a view of a file of ours and one of a section of ours,
        // read and written inside their lengths.
        unsafe {
            let section = CreateFileMappingW(file.as_raw_handle() as _, std::ptr::null(), PAGE_READONLY, 0, 0, std::ptr::null());
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_READ, 0, 0, LEN).Value as usize;
            assert_ne!(view, 0);
            let plain = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, LEN as u32, std::ptr::null());
            let other = MapViewOfFile(plain, FILE_MAP_WRITE, 0, 0, LEN).Value as usize;
            assert_ne!(other, 0);
            (other as *mut u8).write_volatile(7);
            let read = || std::slice::from_raw_parts(view as *const u8, LEN).to_vec();
            assert_eq!(read(), bytes);
            assert!(in_memory(view) && in_memory(view + LEN - 4096) && in_memory(other));
            let (views, let_go) = pass(0);
            assert!(views >= 1 && let_go >= LEN as u64, "{views} views, {let_go} bytes");
            assert!(!in_memory(view) && !in_memory(view + LEN - 4096), "the file's pages are let go");
            assert!(in_memory(other), "a section without a file stays");
            assert_eq!(read(), bytes);
            assert_eq!((other as *const u8).read_volatile(), 7);
        }
        drop(file);
        let _ = std::fs::remove_file(&path);
    }
}
