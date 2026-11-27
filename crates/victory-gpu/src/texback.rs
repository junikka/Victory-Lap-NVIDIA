//! Textures shaders write that the title's own code then reads on the CPU.
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it stays
//! in the texture's host copy. The title's ocean is one: its wave passes write
//! a 256x256 height map (R32_FLOAT) and a displacement map (R32G32_FLOAT), and
//! its CPU code samples both, which is what boats float on; it read memory
//! nothing had written (run rw1, `VICTORY_READ_WATCH=1`: about 30 places in
//! the title read those two kinds), and boats sank.
//!
//! For textures of the kinds in `VICTORY_TEXTURE_BACK` (comma-separated
//! `<width>x<height>f<format>`; unset or `1`: the four water formats found by
//! the upstream trace; `0`: none), a list in which shaders write one copies it
//! into a read-back buffer of ours when it closes, and once the list has run
//! the texels are tiled into the texture's title memory in the console's
//! layout, as the console's GPU would have left them.
//!
//! shortcut: by kind, not by texture: textures of those kinds the title never
//! reads are copied back too (about a megabyte per such list). The copy shows
//! what the list's last write left; a title read in between sees the frame
//! before, as it would before the console's GPU has run.
//!
//! `VICTORY_TEXBACK_ASYNC=1` writes the texels on a thread of their own.
//! Unset, they are written on the landing thread, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

/// The kinds copied back: (width, height, format).
fn kinds() -> &'static [(u64, u32, u32)] {
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        const DEFAULT: &str = "256x256f41,256x256f16,64x64f16,128x128f36";
        let list = match std::env::var("VICTORY_TEXTURE_BACK").as_deref() {
            Ok("1") | Err(_) => DEFAULT.to_owned(),
            Ok(spec) => spec.to_owned(),
        };
        let kinds: Vec<(u64, u32, u32)> = list
            .split(',')
            .filter_map(|k| {
                let (size, format) = k.trim().split_once('f')?;
                let (w, h) = size.split_once('x')?;
                Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
            })
            .collect();
        if !kinds.is_empty() {
            tracing::info!(?kinds, "textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK)");
        }
        kinds
    })
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty()
        && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// An outstanding write to a specific allocation. Mapping and resource
/// changes invalidate only the writes that could touch the replaced memory.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());
static TRACKING: AtomicBool = AtomicBool::new(false);

fn track(owner: usize, at: usize, end: usize) -> Arc<Ticket> {
    let ticket = Arc::new(Ticket { owner, at, end, valid: AtomicBool::new(true) });
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| weak.strong_count() != 0);
    tickets.push(Arc::downgrade(&ticket));
    TRACKING.store(true, Ordering::Relaxed);
    ticket
}

fn invalidate(matches: impl Fn(&Ticket) -> bool) {
    if !TRACKING.load(Ordering::Relaxed) {
        return;
    }
    TICKETS.lock().unwrap_or_else(|e| e.into_inner()).retain(|weak| {
        let Some(ticket) = weak.upgrade() else { return false };
        if matches(&ticket) {
            ticket.valid.store(false, Ordering::Release);
        }
        true
    });
}

pub(crate) fn released(owner: usize) {
    invalidate(|ticket| ticket.owner == owner);
}

pub(crate) fn replaced(at: usize) {
    invalidate(|ticket| ticket.at == at);
}

pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    invalidate(|ticket| at < ticket.end && ticket.at < end);
}

pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Target {
    pub(crate) fn new(resource: This) -> Option<Self> {
        // Keep the host copy independently of the title's resource lifetime.
        let copy = crate::resource::host_copy(resource)?;
        let _mapping = crate::watch::title_memory_reading();
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        let end = at.checked_add(size).filter(|&end| at != 0 && end > at)?;
        let ticket = track(resource as usize, at, end);
        Some(Self { copy, layout, ticket })
    }

    pub(crate) fn owner(&self) -> usize { self.ticket.owner }
}

/// The readback resources and the allocation the pixels belong to.
pub(crate) struct Back {
    buffer: ID3D12Resource,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Back {
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }

    /// The read-back buffer, once the list that copied into it has run and
    /// landed (`crate::host`'s spare lists).
    pub(crate) fn into_buffer(self) -> ID3D12Resource {
        self.buffer
    }
}

/// Records a host copy from a texture already in COPY_SOURCE.
pub(crate) fn record(list: &ID3D12GraphicsCommandList, target: Target) -> Option<Back> {
    if !target.ticket.valid.load(Ordering::Acquire) {
        return None;
    }
    let host = crate::host::host()?;
    let texture = &target.copy.texture;
    // SAFETY: a live texture kept through the command list's state tracking.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels.max(1));
    let subresources = mips * u32::from(desc.DepthOrArraySize.max(1));
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out arrays of `n` entries.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        )
    };
    // The last footprint's advertised extent can omit row-end padding.
    // The tiler reads a pitched slice, so the mapped buffer must include it.
    for (footprint, &rows) in footprints.iter().zip(&rows) {
        let format = footprint.Footprint;
        let pitched = u64::from(format.RowPitch)
            .checked_mul(u64::from(rows))?
            .checked_mul(u64::from(format.Depth))?;
        total = total.max(footprint.Offset.checked_add(pitched)?);
    }
    let buffer = host.read_back_buffer(total)?;
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
        };
        // SAFETY: an open list; the texture is in the copy-source state.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    Some(Back { buffer, footprints, rows, mips, layout: target.layout, ticket: target.ticket })
}

/// Once the list has run, tile its texels back into the same title allocation.
///
/// `VICTORY_TEXBACK_ASYNC=1` copies the texels out here and lets another
/// thread wait, tile and write. Unset, [`land_here`] does all of that on
/// this thread.
pub(crate) fn land(back: &Back) {
    if asynchronous() {
        land_apart(back);
    } else {
        land_here(back);
    }
}

/// `VICTORY_TEXBACK_ASYNC=1`. Anything else, including unset, is off.
fn asynchronous_value(value: Option<&str>) -> bool {
    matches!(value.map(str::trim), Some("1"))
}

/// Whether texture read-backs are written on a thread of their own.
pub(crate) fn asynchronous() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let value = std::env::var("VICTORY_TEXBACK_ASYNC").ok();
        asynchronous_value(value.as_deref())
    })
}

/// Tile and write on the thread that landed the list.
fn land_here(back: &Back) {
    static COPIED: AtomicU64 = AtomicU64::new(0);
    let _mapping = crate::watch::title_memory_reading();
    let _ticket_guard = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    if !back.ticket.valid.load(Ordering::Acquire) {
        stale();
        return;
    }
    let (at, end) = back.range();
    let size = end - at;
    let mut tiled = vec![0u8; size];
    // Keep title bytes the subresources do not cover.
    if !crate::host::read_into(at as *const u8, &mut tiled) {
        return;
    }
    let mut data = std::ptr::null_mut();
    // SAFETY: our read-back buffer, written by the list that has run.
    unsafe {
        if back.buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
            return;
        }
    }
    for (i, (footprint, &rows)) in back.footprints.iter().zip(&back.rows).enumerate() {
        let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let len = pitch * rows as usize * f.Depth as usize;
        // SAFETY: the subresource's part of the mapped buffer.
        let linear = unsafe { std::slice::from_raw_parts(data.cast::<u8>().add(footprint.Offset as usize), len) };
        if let Err(e) = back.layout.tile(&mut tiled, mip, slice, linear, pitch, pitch * rows as usize) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled"));
            // SAFETY: mapped above.
            unsafe { back.buffer.Unmap(0, None) };
            return;
        }
    }
    // SAFETY: mapped above.
    unsafe { back.buffer.Unmap(0, None) };
    crate::host::copy_mapped(at as *mut u8, tiled.as_ptr(), size, true);
    let copied = COPIED.fetch_add(1, Ordering::Relaxed) + 1;
    if copied == 1 || copied.is_multiple_of(128) {
        tracing::info!(copied, bytes = size, "texture readback landed");
    }
}

/// A read-back's texels, copied out of its buffer, to be tiled into title memory.
struct Landing {
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
    /// The read-back buffer's bytes, from 0 through its last subresource.
    data: Vec<u8>,
}

/// Copy the texels out, then tile and write them on [`writer`]'s thread.
fn land_apart(back: &Back) {
    if !back.ticket.valid.load(Ordering::Acquire) {
        stale();
        return;
    }
    let len = back
        .footprints
        .iter()
        .zip(&back.rows)
        .map(|(f, &rows)| {
            f.Offset as usize + f.Footprint.RowPitch as usize * rows as usize * f.Footprint.Depth as usize
        })
        .max()
        .unwrap_or(0);
    let mut mapped = std::ptr::null_mut();
    // SAFETY: our read-back buffer, written by the list that has run; `len`
    // ends with its last subresource, which the buffer holds.
    let data = unsafe {
        if back.buffer.Map(0, None, Some(&mut mapped)).is_err() || mapped.is_null() {
            return;
        }
        let data = std::slice::from_raw_parts(mapped.cast::<u8>().cast_const(), len).to_vec();
        back.buffer.Unmap(0, None);
        data
    };
    let landing = Landing {
        footprints: back.footprints.clone(),
        rows: back.rows.clone(),
        mips: back.mips,
        layout: back.layout.clone(),
        ticket: back.ticket.clone(),
        data,
    };
    match writer() {
        Some(writer) => writer.send(landing),
        None => write(&landing),
    }
}

fn stale() {
    static STALE: AtomicU64 = AtomicU64::new(0);
    let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
    if skipped.is_power_of_two() {
        tracing::info!(skipped, "skipping texture readbacks whose title allocation changed");
    }
}

/// Tiles a read-back's texels into its title allocation.
fn write(landing: &Landing) {
    use crate::land::{count, spent, stat};
    static COPIED: AtomicU64 = AtomicU64::new(0);
    let asked = std::time::Instant::now();
    let _mapping = crate::watch::title_memory_reading();
    let _ticket_guard = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    let working = spent(stat::TB_WAIT, asked);
    if !landing.ticket.valid.load(Ordering::Acquire) {
        stale();
        return;
    }
    let (at, end) = (landing.ticket.at, landing.ticket.end);
    let size = end - at;
    let mut tiled = vec![0u8; size];
    // Keep title bytes the subresources do not cover.
    if !crate::host::read_into(at as *const u8, &mut tiled) {
        return;
    }
    for (i, (footprint, &rows)) in landing.footprints.iter().zip(&landing.rows).enumerate() {
        let (mip, slice) = (i as u32 % landing.mips, i as u32 / landing.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let from = footprint.Offset as usize;
        let len = pitch * rows as usize * f.Depth as usize;
        let Some(linear) = landing.data.get(from..from + len) else {
            return;
        };
        if let Err(e) = landing.layout.tile(&mut tiled, mip, slice, linear, pitch, pitch * rows as usize) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled"));
            return;
        }
    }
    crate::host::copy_mapped(at as *mut u8, tiled.as_ptr(), size, true);
    spent(stat::TB_WORK, working);
    count(stat::TB_WRITTEN, 1);
    let copied = COPIED.fetch_add(1, Ordering::Relaxed) + 1;
    if copied == 1 || copied.is_multiple_of(128) {
        tracing::info!(copied, bytes = size, "texture readback landed");
    }
}

/// Keeps `landing` as the newest copy of its title allocation.
fn note_at(waiting: &mut HashMap<usize, Landing>, landing: Landing) -> bool {
    waiting.insert(landing.ticket.at, landing).is_some()
}

/// The thread that writes read-backs into title memory, and what waits for it.
struct Writer {
    /// By title address: the newest landing for that allocation.
    waiting: Mutex<HashMap<usize, Landing>>,
    more: Condvar,
}

impl Writer {
    fn send(&self, landing: Landing) {
        let mut waiting = self.waiting.lock().unwrap_or_else(|e| e.into_inner());
        if note_at(&mut waiting, landing) {
            crate::land::count(crate::land::stat::TB_REPLACED, 1);
        }
        drop(waiting);
        self.more.notify_one();
    }
}

/// The writer, if its thread could be started. If not, landings are written
/// where they are made.
fn writer() -> Option<&'static Writer> {
    static WRITER: OnceLock<Option<&'static Writer>> = OnceLock::new();
    *WRITER.get_or_init(|| {
        let writer: &'static Writer = Box::leak(Box::new(Writer {
            waiting: Mutex::new(HashMap::new()),
            more: Condvar::new(),
        }));
        let started = std::thread::Builder::new().name("texture read-backs".into()).spawn(move || loop {
            let batch: Vec<Landing> = {
                let mut waiting = writer.waiting.lock().unwrap_or_else(|e| e.into_inner());
                while waiting.is_empty() {
                    waiting = writer.more.wait(waiting).unwrap_or_else(|e| e.into_inner());
                }
                waiting.drain().map(|(_, landing)| landing).collect()
            };
            for landing in &batch {
                write(landing);
            }
        });
        match started {
            Ok(_) => Some(writer),
            Err(e) => {
                tracing::warn!(error = %e, "no thread for texture read-backs: they are written where they land");
                None
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_readbacks_for_changed_allocations_are_invalidated() {
        let first = track(91, 0x1000, 0x2000);
        let next = track(92, 0x2000, 0x3000);
        let far = track(93, 0x4000, 0x5000);

        remapped(0x0f00, 0x200);
        assert!(!first.valid.load(Ordering::Acquire));
        assert!(next.valid.load(Ordering::Acquire));
        replaced(0x2000);
        assert!(!next.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        released(93);
        assert!(!far.valid.load(Ordering::Acquire));
        assert!(track(94, 0x2000, 0x3000).valid.load(Ordering::Acquire));
    }

    #[test]
    fn async_readbacks_are_off_unless_asked() {
        assert!(!asynchronous_value(None));
        assert!(!asynchronous_value(Some("")));
        assert!(!asynchronous_value(Some("0")));
        assert!(!asynchronous_value(Some("2")));
        assert!(!asynchronous_value(Some("off")));
        assert!(asynchronous_value(Some("1")));
        assert!(asynchronous_value(Some(" 1 ")));
    }

    #[test]
    fn only_the_newest_landing_of_one_allocation_is_kept() {
        let layout = victory_formats::texture::Layout::new(&victory_formats::texture::Desc {
            dimension: victory_formats::texture::Dimension::Tex2d,
            width: 4,
            height: 4,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: victory_formats::texture::Swizzle::LINEAR,
        })
        .expect("a 4x4 linear layout");
        let landing = |at: usize, byte: u8| Landing {
            footprints: Vec::new(),
            rows: Vec::new(),
            mips: 1,
            layout: layout.clone(),
            ticket: Arc::new(Ticket { owner: 1, at, end: at + 16, valid: AtomicBool::new(true) }),
            data: vec![byte],
        };
        let mut waiting = HashMap::new();
        assert!(!note_at(&mut waiting, landing(0x1000, 1)));
        assert!(note_at(&mut waiting, landing(0x1000, 2)));
        assert!(!note_at(&mut waiting, landing(0x2000, 3)));
        assert_eq!(waiting.len(), 2);
        assert_eq!(waiting[&0x1000].data, vec![2]);
        assert_eq!(waiting[&0x2000].data, vec![3]);
    }
}
