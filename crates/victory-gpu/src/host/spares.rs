//! Spare read-back and upload buffers (`VICTORY_BUFFER_SPARE=1`, off by
//! default).
//!
//! A play session (no window bar) made some 20 read-back buffers (5.3 MB) and 6-8
//! upload buffers (6-7 MB) a frame, each a committed resource made, mapped,
//! zeroed by the kernel and freed again within the frame; in the world's
//! first minutes 66 upload buffers and 75 MB a frame. With the switch the
//! buffers a host list holds until it is reset (its read-backs and its own
//! upload buffers) go back to a spare list by size class once the list's
//! flight has landed, and [`super::Host::read_back_buffer`] and
//! [`super::Host::upload_buffer`] take one of the class before they make a new one.
//!
//! What makes a given buffer safe to hand out again:
//! - it is given back only where its list's last submission has run (the
//!   list is reset or dropped, as the scratch chunks are);
//! - only when that reference is the last one ([`sole`]): a descriptor slot
//!   or anything else that still holds the buffer keeps it out;
//! - only a buffer of a class's exact width, in the heap of its kind;
//! - an upload buffer taken again is zeroed where it was asked for, as a new
//!   one is ([`super::Host::upload_buffer`]: "all zeros" is what the staging pool
//!   and the strides rely on).
//!
//! The lists are emptied with the spare scratch chunks when the system is
//! short of memory ([`trim`]).

use super::Spare;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

/// The smallest class: a committed buffer takes 64 KiB of memory anyway.
const SMALLEST: u64 = 64 << 10;
/// Past this a buffer is made and freed as before: a large one is rare, and
/// a spare one would hold much memory for little.
const LARGEST: u64 = 16 << 20;
/// The most bytes kept, upload and read-back.
const MOST: [u64; 2] = [96 << 20, 32 << 20];

/// Which list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Upload = 0,
    ReadBack = 1,
}

/// `VICTORY_BUFFER_SPARE=1`. Anything else, including unset, is off.
fn on_value(value: Option<&str>) -> bool {
    matches!(value.map(str::trim), Some("1"))
}

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| on_value(std::env::var("VICTORY_BUFFER_SPARE").ok().as_deref()))
}

/// The class of a buffer of `size` bytes: the power of two that holds it,
/// 64 KiB at least; `None` past [`LARGEST`].
pub(super) fn class(size: u64) -> Option<u64> {
    let class = size.max(SMALLEST).checked_next_power_of_two()?;
    (class <= LARGEST).then_some(class)
}

/// The width to make a buffer of `size` bytes with: its class with the
/// switch, else `size`.
pub(super) fn width(size: u64) -> u64 {
    if on() { class(size).unwrap_or(size) } else { size }
}

/// An upload buffer kept, with its mapping.
struct Upload {
    buffer: ID3D12Resource,
    data: usize,
}

static UPLOADS: Mutex<Spare<Upload>> = Mutex::new(Spare::new());
static READ_BACKS: Mutex<Spare<ID3D12Resource>> = Mutex::new(Spare::new());

/// Since the start, by [`Kind`]: taken from the list, given to it, and
/// refused by it (held elsewhere, not a class, or past [`MOST`]).
static COUNTS: [[AtomicU64; 3]; 2] = [const { [const { AtomicU64::new(0) }; 3] }; 2];

fn count(kind: Kind, what: usize) {
    COUNTS[kind as usize][what].fetch_add(1, Ordering::Relaxed);
}

/// A kept upload buffer for `size` bytes, its first `size` bytes zeroed.
pub(super) fn take_upload(size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    if !on() {
        return None;
    }
    let class = class(size)?;
    let kept = UPLOADS.lock().unwrap_or_else(|e| e.into_inner()).take(class)?;
    count(Kind::Upload, 0);
    let data = kept.data as *mut u8;
    // SAFETY: the buffer's mapping holds `class` bytes, and nothing else
    // holds the buffer ([`sole`] when it was given).
    unsafe { std::ptr::write_bytes(data, 0, size as usize) };
    Some((kept.buffer, data))
}

/// A kept read-back buffer for `size` bytes.
pub(super) fn take_read_back(size: u64) -> Option<ID3D12Resource> {
    if !on() {
        return None;
    }
    let class = class(size)?;
    let kept = READ_BACKS.lock().unwrap_or_else(|e| e.into_inner()).take(class)?;
    count(Kind::ReadBack, 0);
    Some(kept)
}

/// Whether the caller's reference to `buffer` is its only one.
fn sole(buffer: &ID3D12Resource) -> bool {
    // SAFETY: a live interface; every COM interface's table begins with
    // IUnknown's, and a reference taken and given back at once leaves the
    // count as it was, which the giving back returns.
    unsafe {
        let unknown = &Interface::vtable(buffer).base__.base__.base__.base__;
        (unknown.AddRef)(buffer.as_raw());
        (unknown.Release)(buffer.as_raw()) == 1
    }
}

/// The class `buffer` was made with, if it is a buffer of a class's width
/// in the heap of `kind`.
fn class_of(buffer: &ID3D12Resource, kind: Kind) -> Option<u64> {
    // SAFETY: a live resource.
    let desc = unsafe { buffer.GetDesc() };
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || class(desc.Width) != Some(desc.Width) {
        return None;
    }
    let mut heap = D3D12_HEAP_PROPERTIES::default();
    // SAFETY: a committed resource answers its heap's properties.
    unsafe { buffer.GetHeapProperties(Some(&raw mut heap), None) }.ok()?;
    let wanted = match kind {
        Kind::Upload => D3D12_HEAP_TYPE_UPLOAD,
        Kind::ReadBack => D3D12_HEAP_TYPE_READBACK,
    };
    (heap.Type == wanted).then_some(desc.Width)
}

/// After the last submission of the list that held it: an upload buffer
/// mapped at `data`, kept for the next ask of its class or let go.
pub(super) fn give_upload(buffer: ID3D12Resource, data: usize) {
    if !on() {
        return;
    }
    let Some(class) = class_of(&buffer, Kind::Upload).filter(|_| sole(&buffer) && data != 0) else {
        count(Kind::Upload, 2);
        return;
    };
    let back = UPLOADS.lock().unwrap_or_else(|e| e.into_inner()).give(Upload { buffer, data }, class, MOST[0]);
    count(Kind::Upload, if back.is_none() { 1 } else { 2 });
    // (One handed back is freed outside the lock.)
    drop(back);
}

/// After the last submission of the list that held it: a read-back buffer
/// (not mapped), kept for the next ask of its class or let go.
pub(super) fn give_read_back(buffer: ID3D12Resource) {
    if !on() {
        return;
    }
    let Some(class) = class_of(&buffer, Kind::ReadBack).filter(|_| sole(&buffer)) else {
        count(Kind::ReadBack, 2);
        return;
    };
    let back = READ_BACKS.lock().unwrap_or_else(|e| e.into_inner()).give(buffer, class, MOST[1]);
    count(Kind::ReadBack, if back.is_none() { 1 } else { 2 });
    drop(back);
}

/// Lets go of every kept buffer (the system is short of memory).
pub(super) fn trim() {
    // (Freed outside the locks.)
    let uploads = std::mem::replace(&mut *UPLOADS.lock().unwrap_or_else(|e| e.into_inner()), Spare::new());
    let backs = std::mem::replace(&mut *READ_BACKS.lock().unwrap_or_else(|e| e.into_inner()), Spare::new());
    drop((uploads, backs));
}

/// For the frame line: by [`Kind`], taken, given and refused since the
/// last call, and the bytes kept now.
#[cfg(feature = "diagnostics")]
pub(super) fn line() -> [[u64; 4]; 2] {
    let kept = [
        UPLOADS.lock().unwrap_or_else(|e| e.into_inner()).bytes,
        READ_BACKS.lock().unwrap_or_else(|e| e.into_inner()).bytes,
    ];
    std::array::from_fn(|k| {
        let c = &COUNTS[k];
        [c[0].swap(0, Ordering::Relaxed), c[1].swap(0, Ordering::Relaxed), c[2].swap(0, Ordering::Relaxed), kept[k]]
    })
}

/// With diagnostics, a buffer of `kind` made new, counted by the line of
/// code that asked for it (through [`super::Host::upload_buffer`] and
/// [`super::Host::read_back_buffer`], which pass their caller on).
#[track_caller]
pub(super) fn made(kind: Kind) {
    #[cfg(feature = "diagnostics")]
    made_at(kind, std::panic::Location::caller());
    #[cfg(not(feature = "diagnostics"))]
    let _ = kind;
}

/// Buffers made new since the last frame line, by kind and asker.
#[cfg(feature = "diagnostics")]
static MADE_AT: Mutex<Vec<(Kind, &'static std::panic::Location<'static>, u64)>> = Mutex::new(Vec::new());

#[cfg(feature = "diagnostics")]
fn made_at(kind: Kind, at: &'static std::panic::Location<'static>) {
    let mut made = MADE_AT.lock().unwrap_or_else(|e| e.into_inner());
    match made.iter_mut().find(|(k, a, _)| *k == kind && **a == *at) {
        Some((.., n)) => *n += 1,
        None => made.push((kind, at, 1)),
    }
}

/// The askers of buffers made since the last call, most first: kind,
/// `file:line`, how many.
#[cfg(feature = "diagnostics")]
pub(super) fn made_since() -> Vec<(Kind, String, u64)> {
    let mut made: Vec<_> = std::mem::take(&mut *MADE_AT.lock().unwrap_or_else(|e| e.into_inner()))
        .into_iter()
        .map(|(kind, at, n)| (kind, format!("{}:{}", at.file().rsplit(['/', '\\']).next().unwrap_or(""), at.line()), n))
        .collect();
    made.sort_by_key(|m| std::cmp::Reverse(m.2));
    made
}

#[cfg(test)]
mod tests {
    use super::{Spare, class, on_value};

    #[test]
    fn the_switch_is_on_only_at_one() {
        assert!(!on_value(None));
        assert!(!on_value(Some("0")));
        assert!(!on_value(Some("yes")));
        assert!(on_value(Some("1")));
        assert!(on_value(Some(" 1 ")));
    }

    /// Sizes go to the power of two that holds them, 64 KiB at least, and
    /// none past 16 MiB.
    #[test]
    fn sizes_go_to_their_class() {
        assert_eq!(class(0), Some(64 << 10));
        assert_eq!(class(1), Some(64 << 10));
        assert_eq!(class(64 << 10), Some(64 << 10));
        assert_eq!(class((64 << 10) + 1), Some(128 << 10));
        assert_eq!(class(265_000), Some(512 << 10));
        assert_eq!(class(16 << 20), Some(16 << 20));
        assert_eq!(class((16 << 20) + 1), None);
        assert_eq!(class(u64::MAX), None);
    }

    /// The lists as the buffers use them: a buffer given back under its
    /// class comes back for any size of that class, never for another, and
    /// nothing is kept past the bytes allowed.
    #[test]
    fn a_spare_comes_back_for_its_class_within_the_limit() {
        let most = 1 << 20;
        let mut spare: Spare<&str> = Spare::new();
        let asked = |size: u64| class(size).unwrap_or(size);
        assert_eq!(spare.take(asked(300_000)), None, "nothing kept yet");
        assert_eq!(spare.give("a", asked(300_000), most), None);
        assert_eq!(spare.give("b", asked(70_000), most), None);
        assert_eq!(spare.bytes, (512 << 10) + (128 << 10));
        // Past the limit it is handed back to be let go.
        assert_eq!(spare.give("c", asked(400_000), most), Some("c"));
        assert_eq!(spare.take(asked(200_000)), None, "256 KiB: no buffer of that class");
        assert_eq!(spare.take(asked(500_000)), Some("a"), "any size of the 512 KiB class");
        assert_eq!(spare.take(asked(500_000)), None);
        assert_eq!(spare.take(asked(65_537)), Some("b"));
        assert_eq!(spare.bytes, 0);
    }
}
