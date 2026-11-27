//! `VICTORY_MEMORY_BY_USE=1` (off by default): every host copy of title
//! memory (a title buffer's copy, a window over memory outside buffers, a
//! copy of one view's bytes, title memory in place) by what is done with
//! it. Logging only: nothing of what is copied, when or where changes, but
//! that one fetched piece in [`SAMPLE_EVERY`] is fetched a frame later.
//!
//! Four facts per copy, each kept since the copy was made and as the frame
//! it was last seen in (for the last [`RECENT`] frames):
//! - [`CPU_WROTE`]: a refresh took pages the copy held before that the
//!   write watch says were written since (`Shadow::copy_stale`), or a copy
//!   without a watch was taken whole again after its first fill. Bytes a
//!   landing of shader writes put into title memory count here too where
//!   the watch tells of them (the upload echo, `crate::census`).
//! - [`GPU_READ`]: a draw or dispatch was given a range of it
//!   (`Shadow::refresh`); for a window, also which of its 64 KiB pieces.
//! - [`SHADER_WROTE`]: a piece the card's compare flagged was fetched and
//!   merged (a copy in the card's memory), or the compare on the CPU found
//!   bytes shaders wrote and put them into title memory (a copy in the
//!   PC's memory).
//! - [`CPU_READ`]: the title's CPU read a page a shader wrote. Sampled: one
//!   fetched piece in [`SAMPLE_EVERY`] is held on the card through the lazy
//!   copy-back's path (`crate::lazy_back::defer_sampled`): its title pages
//!   are made unreadable for one frame, and a read of the title's there
//!   lands it at once and is noted. Copies none of whose pieces was ever
//!   held so are told apart ("not sampled").
//!
//! Every [`period`] (10 s) and at exit: a table of MB by class
//! ([`Class`], the classes of the alignment report's table of 2026-11-26)
//! and by where the bytes lie now ([`Place`]); the bytes of the windows no
//! draw read; and the copies let go since the last table.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The facts, as bits of [`Uses::flags`] and indices of [`Uses::last`].
pub(crate) const CPU_WROTE: usize = 0;
pub(crate) const GPU_READ: usize = 1;
pub(crate) const SHADER_WROTE: usize = 2;
pub(crate) const CPU_READ: usize = 3;
/// A piece of the copy was held to sample [`CPU_READ`].
const SAMPLED: u8 = 1 << 4;
/// The copy was filled once (for copies without a write watch).
const FILLED: u8 = 1 << 5;
/// Frames that make "lately".
pub(crate) const RECENT: u64 = 600;
/// One fetched piece in this many is held to sample [`CPU_READ`].
pub(crate) const SAMPLE_EVERY: u64 = 64;
/// A window's pieces, as the card's compare cuts copies.
const PIECE: u64 = 64 << 10;

/// Whether the copies are classed (`VICTORY_MEMORY_BY_USE=1`).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_MEMORY_BY_USE").is_ok_and(|v| v.trim() == "1");
        if on {
            tracing::warn!(
                "VICTORY_MEMORY_BY_USE=1: host copies classed by use (logging only); one fetched piece in {SAMPLE_EVERY} is \
                 held on the card for a frame to see whether the title's CPU reads it"
            );
        }
        on
    })
}

fn period() -> Duration {
    Duration::from_secs(10)
}

/// What a copy is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    Buffer,
    Window,
    View,
}

/// Where a copy's bytes lie beside title memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Place {
    /// A buffer in the PC's memory that the card reads over the bus.
    Pc,
    /// Only in the card's memory (a window the CPU writes through the bar).
    Card,
    /// The CPU's side in the PC's memory and the card's side.
    Both,
    /// No copy: the card reads title memory itself.
    InPlace,
}

const PLACES: [&str; 5] = ["PC mirror", "card", "both", "in place", "not asked yet"];

impl Place {
    fn index(self) -> usize {
        self as usize
    }
}

/// A copy's class by use, from its facts ([`class`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Class {
    /// The CPU writes it again and again, the card reads it, shaders never
    /// write it (vertices, indices, constants).
    CpuWrittenGpuRead,
    /// Written once when it was filled, read by the card since.
    FilledGpuRead,
    /// Shaders write it, and the title's CPU was seen reading what they wrote.
    ShaderWrittenCpuRead,
    /// Shaders write it; its pieces were held and the title's CPU never read
    /// them (the card-only class of ticket 115).
    ShaderWrittenNoCpuRead,
    /// Shaders write it; none of its pieces was held to see.
    ShaderWrittenNotSampled,
    /// No draw or dispatch read it: CPU-only memory under a copy.
    NotGpuRead,
}

const CLASSES: [Class; 6] = [
    Class::CpuWrittenGpuRead,
    Class::FilledGpuRead,
    Class::ShaderWrittenCpuRead,
    Class::ShaderWrittenNoCpuRead,
    Class::ShaderWrittenNotSampled,
    Class::NotGpuRead,
];

impl Class {
    fn name(self) -> &'static str {
        match self {
            Class::CpuWrittenGpuRead => "CPU-written, GPU-read, never shader-written",
            Class::FilledGpuRead => "CPU-written at fill only, GPU-read, never shader-written",
            Class::ShaderWrittenCpuRead => "GPU-written, CPU-read",
            Class::ShaderWrittenNoCpuRead => "GPU-written, no CPU read seen",
            Class::ShaderWrittenNotSampled => "GPU-written, CPU read not sampled",
            Class::NotGpuRead => "CPU-only (no draw or dispatch read it)",
        }
    }

    fn index(self) -> usize {
        CLASSES.iter().position(|&c| c == self).unwrap_or(0)
    }
}

/// The class of a copy whose facts are `flags` (bits [`CPU_WROTE`] to
/// [`CPU_READ`], and [`SAMPLED`]). What shaders write decides first: such a
/// copy is GPU-accessed whatever else is known of it.
pub(crate) fn class(flags: u8) -> Class {
    let has = |bit: usize| flags & 1 << bit != 0;
    if has(SHADER_WROTE) {
        if has(CPU_READ) {
            Class::ShaderWrittenCpuRead
        } else if flags & SAMPLED != 0 {
            Class::ShaderWrittenNoCpuRead
        } else {
            Class::ShaderWrittenNotSampled
        }
    } else if !has(GPU_READ) {
        Class::NotGpuRead
    } else if has(CPU_WROTE) {
        Class::CpuWrittenGpuRead
    } else {
        Class::FilledGpuRead
    }
}

/// Of `flags`, the facts last seen within [`RECENT`] frames of `now`
/// (`last`: the frame each was last seen in, plus one; 0: never). Whether
/// a piece was sampled is kept as it is.
fn lately(flags: u8, last: [u64; 4], now: u64) -> u8 {
    let mut out = flags & !0xf;
    for (bit, &seen) in last.iter().enumerate() {
        if flags & 1 << bit != 0 && seen != 0 && now < (seen - 1) + RECENT {
            out |= 1 << bit;
        }
    }
    out
}

/// One copy's facts. Shared by the copy (`Shadow::uses`) and the table.
pub(crate) struct Uses {
    start: u64,
    len: u64,
    /// 0 until a hook tells: then 1 + kind * 4 + place ([`Uses::shape`]).
    shape: AtomicU8,
    view: AtomicBool,
    flags: AtomicU8,
    last: [AtomicU64; 4],
    /// For a window: which of its pieces a draw or dispatch was given.
    read: OnceLock<Box<[AtomicU64]>>,
}

impl Uses {
    fn new(start: u64, len: u64) -> Uses {
        Uses {
            start,
            len,
            shape: AtomicU8::new(0),
            view: AtomicBool::new(false),
            flags: AtomicU8::new(0),
            last: [const { AtomicU64::new(0) }; 4],
            read: OnceLock::new(),
        }
    }

    /// The copy is one view's own bytes (`crate::shadow::exact`).
    pub(crate) fn mark_view(&self) {
        self.view.store(true, Ordering::Relaxed);
    }

    /// What the copy is and where it lies, told by every hook (known only
    /// once the copy is made in full).
    fn shape(&self, window: bool, place: Place) {
        if self.shape.load(Ordering::Relaxed) == 0 {
            let kind = if self.view.load(Ordering::Relaxed) {
                Kind::View
            } else if window {
                Kind::Window
            } else {
                Kind::Buffer
            };
            self.shape.store(1 + kind as u8 * 4 + place as u8, Ordering::Relaxed);
        }
    }

    fn shaped(&self) -> Option<(Kind, Place)> {
        let s = self.shape.load(Ordering::Relaxed);
        let kind = match s.checked_sub(1)? / 4 {
            0 => Kind::Buffer,
            1 => Kind::Window,
            _ => Kind::View,
        };
        let place = match (s - 1) % 4 {
            0 => Place::Pc,
            1 => Place::Card,
            2 => Place::Both,
            _ => Place::InPlace,
        };
        Some((kind, place))
    }

    /// Fact `bit` was seen now.
    pub(crate) fn note(&self, bit: usize, window: bool, place: Place) {
        self.shape(window, place);
        if self.flags.load(Ordering::Relaxed) & 1 << bit == 0 {
            self.flags.fetch_or(1 << bit, Ordering::Relaxed);
        }
        self.last[bit].store(crate::frame::current() + 1, Ordering::Relaxed);
    }

    /// A draw or dispatch was given title memory `[from, to)` of the copy.
    pub(crate) fn read(&self, from: u64, to: u64, window: bool, place: Place) {
        self.note(GPU_READ, window, place);
        if !window || from >= to || self.len == 0 {
            return;
        }
        let pieces = self.len.div_ceil(PIECE) as usize;
        let bits = self.read.get_or_init(|| (0..pieces.div_ceil(64)).map(|_| AtomicU64::new(0)).collect());
        let first = (from.max(self.start) - self.start) / PIECE;
        let last = (to.min(self.start + self.len).max(self.start + 1) - 1 - self.start) / PIECE;
        for piece in first..=last.min(pieces as u64 - 1) {
            let (word, bit) = (piece as usize / 64, piece % 64);
            if bits[word].load(Ordering::Relaxed) & 1 << bit == 0 {
                bits[word].fetch_or(1 << bit, Ordering::Relaxed);
            }
        }
    }

    /// A copy without a write watch was taken whole: after its first fill
    /// that is the CPU's writing as far as can be told.
    pub(crate) fn taken_whole(&self, window: bool, place: Place) {
        if self.flags.fetch_or(FILLED, Ordering::Relaxed) & FILLED != 0 {
            self.note(CPU_WROTE, window, place);
        }
    }

    /// A piece of the copy is held to see whether the title reads it.
    pub(crate) fn sampled(&self) {
        self.flags.fetch_or(SAMPLED, Ordering::Relaxed);
        count(n::HELD, 1);
    }

    /// The bytes of a window no draw or dispatch was given.
    fn unread(&self) -> u64 {
        let Some(bits) = self.read.get() else {
            return self.len;
        };
        let read: u64 = bits.iter().map(|w| u64::from(w.load(Ordering::Relaxed).count_ones())).sum();
        self.len.saturating_sub(read * PIECE)
    }

    fn facts(&self) -> (u8, [u64; 4]) {
        (self.flags.load(Ordering::Relaxed), std::array::from_fn(|i| self.last[i].load(Ordering::Relaxed)))
    }
}

/// The live copies, by their number.
static LIVE: Mutex<BTreeMap<u64, Arc<Uses>>> = Mutex::new(BTreeMap::new());

/// Counters since the start.
mod n {
    /// Pieces asked to be sampled, pieces held for it, reads of the title's
    /// that landed one.
    pub(super) const ASKED: usize = 0;
    pub(super) const HELD: usize = 1;
    pub(super) const READS: usize = 2;
    /// Copies let go: windows and their bytes, others and their bytes.
    pub(super) const WINDOWS_GONE: usize = 3;
    pub(super) const WINDOW_BYTES_GONE: usize = 4;
    pub(super) const OTHERS_GONE: usize = 5;
    pub(super) const OTHER_BYTES_GONE: usize = 6;
    pub(super) const MADE: usize = 7;
    pub(super) const MADE_BYTES: usize = 8;
    pub(super) const COUNT: usize = 9;
}

static COUNTS: [AtomicU64; n::COUNT] = [const { AtomicU64::new(0) }; n::COUNT];

fn count(i: usize, by: u64) {
    COUNTS[i].fetch_add(by, Ordering::Relaxed);
}

/// Bytes let go since the start by class (since made), windows apart.
static GONE_BY_CLASS: [[AtomicU64; 6]; 2] = [const { [const { AtomicU64::new(0) }; 6] }; 2];

/// A copy of `[start, start + len)` numbered `id` is made: its facts, with
/// the switch on.
pub(crate) fn made(id: u64, start: u64, len: u64) -> Option<Arc<Uses>> {
    if !on() {
        return None;
    }
    static EXIT: std::sync::Once = std::sync::Once::new();
    // SAFETY: registers a function with the host CRT's exit table.
    EXIT.call_once(|| unsafe {
        atexit(at_exit);
    });
    let uses = Arc::new(Uses::new(start, len));
    LIVE.lock().unwrap_or_else(|e| e.into_inner()).insert(id, uses.clone());
    count(n::MADE, 1);
    count(n::MADE_BYTES, len);
    Some(uses)
}

/// Copy `id` is gone.
pub(crate) fn gone(id: u64) {
    let Some(uses) = LIVE.lock().unwrap_or_else(|e| e.into_inner()).remove(&id) else {
        return;
    };
    let window = uses.shaped().is_some_and(|(kind, _)| kind == Kind::Window);
    let (flags, _) = uses.facts();
    GONE_BY_CLASS[usize::from(window)][class(flags).index()].fetch_add(uses.len, Ordering::Relaxed);
    if window {
        count(n::WINDOWS_GONE, 1);
        count(n::WINDOW_BYTES_GONE, uses.len);
    } else {
        count(n::OTHERS_GONE, 1);
        count(n::OTHER_BYTES_GONE, uses.len);
    }
}

/// Whether the fetched piece now looked at is the one in [`SAMPLE_EVERY`]
/// to hold (with the switch on).
pub(crate) fn sample_due() -> bool {
    on() && {
        static PIECES: AtomicU64 = AtomicU64::new(0);
        PIECES.fetch_add(1, Ordering::Relaxed) % SAMPLE_EVERY == 0
    }
}

/// [`sample_due`] said to hold a piece; whether it could be held is told by
/// [`Uses::sampled`].
pub(crate) fn sample_asked() {
    count(n::ASKED, 1);
}

/// A read of the title's landed a held piece of the copy `uses`.
pub(crate) fn title_read(uses: &Uses, window: bool, place: Place) {
    count(n::READS, 1);
    uses.note(CPU_READ, window, place);
}

unsafe extern "C" {
    fn atexit(f: extern "C" fn()) -> i32;
}

extern "C" fn at_exit() {
    report(true);
}

/// With the frame line: the table, if [`period`] has gone by.
pub(crate) fn report_if_due() {
    if on() {
        report(false);
    }
}

/// Per class and place: copies and bytes; and of windows the bytes.
#[derive(Default)]
struct Sums {
    copies: [[u64; 5]; 6],
    bytes: [[u64; 5]; 6],
    windows: [u64; 6],
    views: [u64; 6],
}

impl Sums {
    fn add(&mut self, class: Class, kind: Option<Kind>, place: usize, len: u64) {
        let c = class.index();
        self.copies[c][place] += 1;
        self.bytes[c][place] += len;
        match kind {
            Some(Kind::Window) => self.windows[c] += len,
            Some(Kind::View) => self.views[c] += len,
            _ => {}
        }
    }

    fn lines(&self, scope: &str) {
        let mb = |b: u64| b as f64 / 1e6;
        for class in CLASSES {
            let c = class.index();
            let copies: u64 = self.copies[c].iter().sum();
            if copies == 0 {
                continue;
            }
            let places: Vec<String> = PLACES
                .iter()
                .enumerate()
                .filter(|&(p, _)| self.copies[c][p] > 0)
                .map(|(p, name)| format!("{name} {:.0} MB ({})", mb(self.bytes[c][p]), self.copies[c][p]))
                .collect();
            tracing::info!(
                "memory by use, {scope}: {}: {copies} copies {:.0} MB: {}; of them windows {:.0} MB, views' own copies {:.0} MB",
                class.name(),
                mb(self.bytes[c].iter().sum()),
                places.join(", "),
                mb(self.windows[c]),
                mb(self.views[c]),
            );
        }
    }
}

/// When the last table was written.
static LAST: Mutex<Option<Instant>> = Mutex::new(None);

fn report(exiting: bool) {
    let now = Instant::now();
    {
        let Ok(mut last) = LAST.try_lock() else {
            return;
        };
        match *last {
            Some(at) if !exiting && now.duration_since(at) < period() => return,
            None if !exiting => {
                *last = Some(now);
                return;
            }
            _ => *last = Some(now),
        }
    }
    // At exit a thread that ended can hold the table: then nothing.
    let live: Vec<Arc<Uses>> = match LIVE.try_lock() {
        Ok(live) => live.values().cloned().collect(),
        Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner().values().cloned().collect(),
        Err(std::sync::TryLockError::WouldBlock) => return,
    };
    let frame = crate::frame::current();
    let mb = |b: u64| b as f64 / 1e6;
    let (mut ever, mut lately_) = (Sums::default(), Sums::default());
    // Host copies of title memory as the memory line counts them (not in
    // place), and of them those whose place is known.
    let (mut copies, mut named) = (0u64, 0u64);
    // Windows: bytes, never read by a draw, and by the part of each unread.
    let (mut window_bytes, mut window_unread) = (0u64, 0u64);
    let mut unread_parts = [0u64; 5];
    let mut most_unread: Vec<(u64, u64, u64)> = Vec::new();
    for uses in &live {
        let shaped = uses.shaped();
        let place = shaped.map_or(4, |(_, p)| p.index());
        let kind = shaped.map(|(k, _)| k);
        let (flags, last) = uses.facts();
        ever.add(class(flags), kind, place, uses.len);
        lately_.add(class(lately(flags, last, frame)), kind, place, uses.len);
        if shaped.is_none_or(|(_, p)| p != Place::InPlace) {
            copies += uses.len;
            if shaped.is_some() {
                named += uses.len;
            }
        }
        if kind == Some(Kind::Window) && place != Place::InPlace.index() {
            let unread = uses.unread();
            window_bytes += uses.len;
            window_unread += unread;
            let part = if unread == 0 {
                0
            } else if unread == uses.len {
                4
            } else {
                1 + (unread * 3 / uses.len.max(1)) as usize
            };
            unread_parts[part.min(4)] += 1;
            most_unread.push((unread, uses.start, uses.len));
        }
    }
    let c: [u64; n::COUNT] = std::array::from_fn(|i| COUNTS[i].load(Ordering::Relaxed));
    tracing::info!(
        "memory by use (VICTORY_MEMORY_BY_USE=1){} at frame {frame}: {} copies alive; host copies of title memory (not in place) \
         {:.0} MB, {:.0} MB of them named by class and place ({:.0} %); classes since each copy was made, then over the last \
         {RECENT} frames",
        if exiting { " at exit" } else { "" },
        live.len(),
        mb(copies),
        mb(named),
        if copies == 0 { 100.0 } else { named as f64 * 100.0 / copies as f64 },
    );
    ever.lines("since made");
    lately_.lines(&format!("last {RECENT} frames"));
    most_unread.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let most: Vec<String> =
        most_unread.iter().take(8).map(|&(unread, start, len)| format!("{start:#x}+{:.1}MB {:.1} MB", mb(len), mb(unread))).collect();
    tracing::info!(
        "memory by use, windows outside buffers: {} {:.0} MB, {:.0} MB of them in pieces no draw or dispatch was given; windows by \
         the part unread: none {}, under a third {}, a third to two {}, over two thirds {}, all {}; the most unread: {}",
        most_unread.len(),
        mb(window_bytes),
        mb(window_unread),
        unread_parts[0],
        unread_parts[1],
        unread_parts[2],
        unread_parts[3],
        unread_parts[4],
        most.join(", "),
    );
    let gone = |w: usize| -> String {
        CLASSES
            .iter()
            .map(|class| (class, GONE_BY_CLASS[w][class.index()].load(Ordering::Relaxed)))
            .filter(|&(_, b)| b > 0)
            .map(|(class, b)| format!("{} {:.0} MB", class.name(), mb(b)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    tracing::info!(
        "memory by use since the start: {} copies made {:.0} MB; let go {} windows {:.0} MB ({}) and {} other copies {:.0} MB ({}); \
         CPU reads sampled: {} pieces asked, {} held for a frame, {} landed by a read of the title's",
        c[n::MADE],
        mb(c[n::MADE_BYTES]),
        c[n::WINDOWS_GONE],
        mb(c[n::WINDOW_BYTES_GONE]),
        gone(1),
        c[n::OTHERS_GONE],
        mb(c[n::OTHER_BYTES_GONE]),
        gone(0),
        c[n::ASKED],
        c[n::HELD],
        c[n::READS],
    );
}

#[cfg(test)]
mod tests {
    use super::{CPU_READ, CPU_WROTE, Class, GPU_READ, PIECE, Place, RECENT, SAMPLED, SHADER_WROTE, Uses, class, lately};

    const fn bits(of: &[usize]) -> u8 {
        let mut out = 0;
        let mut i = 0;
        while i < of.len() {
            out |= 1 << of[i];
            i += 1;
        }
        out
    }

    #[test]
    fn the_class_follows_the_four_facts() {
        // Vertices, indices, constants: the CPU writes, the card reads.
        assert_eq!(class(bits(&[CPU_WROTE, GPU_READ])), Class::CpuWrittenGpuRead);
        // Filled once and read since.
        assert_eq!(class(bits(&[GPU_READ])), Class::FilledGpuRead);
        // Nothing read it: CPU-only, written or not.
        assert_eq!(class(bits(&[CPU_WROTE])), Class::NotGpuRead);
        assert_eq!(class(0), Class::NotGpuRead);
        // What shaders write decides first, read by a draw or not.
        assert_eq!(class(bits(&[SHADER_WROTE])), Class::ShaderWrittenNotSampled);
        assert_eq!(class(bits(&[CPU_WROTE, GPU_READ, SHADER_WROTE])), Class::ShaderWrittenNotSampled);
        assert_eq!(class(bits(&[GPU_READ, SHADER_WROTE]) | SAMPLED), Class::ShaderWrittenNoCpuRead);
        assert_eq!(class(bits(&[GPU_READ, SHADER_WROTE, CPU_READ]) | SAMPLED), Class::ShaderWrittenCpuRead);
        // A sample of a copy no shader wrote says nothing of its class.
        assert_eq!(class(bits(&[CPU_WROTE, GPU_READ]) | SAMPLED), Class::CpuWrittenGpuRead);
    }

    #[test]
    fn lately_keeps_only_facts_of_the_last_frames() {
        let flags = bits(&[CPU_WROTE, GPU_READ, SHADER_WROTE]) | SAMPLED;
        // Written by shaders at frame 10, by the CPU and read at frame 300.
        let last = [301, 301, 11, 0];
        let now = 10 + RECENT + 90;
        let late = lately(flags, last, now);
        assert_eq!(late, bits(&[CPU_WROTE, GPU_READ]) | SAMPLED);
        assert_eq!(class(late), Class::CpuWrittenGpuRead);
        // Within the frames, everything holds.
        assert_eq!(lately(flags, last, 500), flags);
        assert_eq!(class(lately(flags, last, 500)), Class::ShaderWrittenNoCpuRead);
    }

    #[test]
    fn a_window_counts_the_pieces_no_draw_was_given() {
        let start = 0x4000_0000u64;
        let uses = Uses::new(start, 64 * PIECE);
        assert_eq!(uses.unread(), 64 * PIECE);
        uses.read(start + 10, start + 20, true, Place::Pc);
        uses.read(start + 3 * PIECE - 1, start + 3 * PIECE + 1, true, Place::Pc);
        // Past the end it is cut to the window.
        uses.read(start + 63 * PIECE, start + 80 * PIECE, true, Place::Pc);
        assert_eq!(uses.unread(), (64 - 4) * PIECE);
        assert_eq!(uses.shaped(), Some((super::Kind::Window, Place::Pc)));
        assert_eq!(class(uses.facts().0), Class::FilledGpuRead);
    }
}
