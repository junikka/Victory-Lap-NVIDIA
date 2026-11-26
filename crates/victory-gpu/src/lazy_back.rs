//! What shaders wrote to a copy in the card's memory stays on the card until
//! the title looks (`VICTORY_LAZY_BACK=1`, off by default; ported from
//! jman's demand-paged copy-back, 17590808..1a7aa791, onto this tree's
//! landing and guards).
//!
//! As it was: every piece of 64 KiB a flight's compare flagged was fetched
//! when the flight landed and merged into title memory, read by the title
//! or not: 98 MB a frame over the bus in the world (`crate::land`).
//!
//! With the switch a flagged piece whose block of title memory is a view of
//! the title's pages, guarded at every address the title sees it at, is not
//! fetched. Its views are made unreadable instead ([`defer`],
//! `crate::watch::lazy_hold`), and the piece is remembered. A later flight
//! that writes it again finds it remembered and fetches nothing either.
//! The piece is fetched and merged ("landed") when
//!
//! - a thread of the title's, or code of ours that reads title memory
//!   where it lies, touches the block: the fault handler lands it and the
//!   access is made again ([`faulted`]);
//! - we are about to write title memory there, or have the kernel write it,
//!   or the runtime changes what the addresses show ([`land_range`]);
//! - another copy landed bytes in the block, or a copy of ours read it
//!   through our own view meanwhile ([`want`], [`read_held`]): at the next
//!   landing ([`tend`]);
//! - no flight wrote it for [`idle`] frames ([`tend`]).
//!
//! A landing here is the landing there: `Shadow::land_card` through our own
//! view of the title's pages (`crate::inplace::beside`), noted in the
//! journal when the bytes are in. The title's views stay unreadable until
//! then, so no thread of the title's reads the block between "the card has
//! it" and "title memory has it": it faults and waits. Our own copies do
//! not wait (they may hold locks a landing takes): a read of theirs that
//! faults there takes the title's bytes as they are through our view
//! ([`read_held`]), as a read just ahead of a landing always did, and the
//! journal's note of the landing has them take the block again.
//!
//! While a piece is held its copy's refresh leaves it alone, as it leaves a
//! range shaders are writing ([`held`], `Shadow::update`): the CPU's side
//! there is what the card held before the shaders wrote, which the merge
//! compares with.
//!
//! A block the title touches in frame after frame is not worth a fault each
//! time: after [`promote`] frames with one it is fetched at once again
//! ("hot"), for as long as it is written, and tried anew after
//! [`REPROBE`] frames.
//!
//! Locks: the table's is taken before the guards' (`crate::watch::GUARDS`,
//! held alone to hold and release views) and before the views'
//! (`crate::inplace`), never after either, and is not held while a piece is
//! fetched or merged.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, OnceLock};

use crate::land::BLOCK;
use crate::shadow::Shadow;

/// See the file. Only with what it rests on: flights landed on their own
/// thread (`crate::shadow::relay`), merged in one pass (`crate::land`) and
/// put through our own view of the title's pages (`VICTORY_LAND_BESIDE`).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if !std::env::var("VICTORY_LAZY_BACK").is_ok_and(|v| v.trim() == "1") {
            return false;
        }
        let can = crate::shadow::relay() && crate::land::on() && crate::land::beside_on();
        if !can {
            tracing::warn!(
                "VICTORY_LAZY_BACK=1 is left off: it needs VICTORY_RELAY, VICTORY_LAND and VICTORY_LAND_BESIDE as they are by default"
            );
        }
        can
    })
}

/// Frames with a fault after which a block is hot
/// (`VICTORY_LAZY_BACK_PROMOTE`, 2 as jman's).
fn promote() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_LAZY_BACK_PROMOTE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2).max(1))
}

/// Frames without a flight writing a held piece after which it is landed
/// anyway (`VICTORY_LAZY_BACK_IDLE`, 120; 0: never). A piece nobody writes
/// any more costs one fetch then, and its copy is not kept alive for it.
fn idle() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_LAZY_BACK_IDLE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(120))
}

/// Frames without a fault after which a block is no longer hot (jman's).
const DECAY: u64 = 120;
/// Frames after which a hot block that is still written is held once more,
/// to see whether the title still reads it; doubled each time it did, up to
/// sixteen times (jman's).
const REPROBE: u64 = 1800;
/// Idle pieces landed by one [`tend`] at most.
const TEND_MOST: usize = 64;

/// A held block: pieces of one copy in one 64 KiB block of title memory.
struct Held {
    copy: Arc<Shadow>,
    /// Offsets in the copy, in order, none touching another.
    ranges: Vec<(usize, usize)>,
    /// The newest send of title bytes that a flight which wrote the pieces
    /// ran behind (`crate::host`'s order of the queue).
    after: u64,
    /// Every address the title sees the block at, made unreadable.
    views: Vec<usize>,
    /// The frame a flight last wrote it in.
    frame: u64,
    /// A thread is landing it.
    landing: bool,
    /// To be landed at the next [`tend`].
    wanted: bool,
}

/// How often the title touched a block that was held (jman's hot blocks).
#[derive(Clone, Copy)]
struct Hot {
    faults: u32,
    last: u64,
    since: u64,
    probes: u32,
    probing: bool,
}

#[derive(Default)]
struct Table {
    /// By the block (address / 64 KiB) of the copy's own address.
    held: BTreeMap<usize, Held>,
    /// Every view's block, to the block it is held by.
    views: BTreeMap<usize, usize>,
    hot: HashMap<usize, Hot>,
}

static TABLE: LazyLock<Mutex<Table>> = LazyLock::new(Mutex::default);
/// A landing ended.
static LANDED: Condvar = Condvar::new();
/// Blocks held now.
static HELD: AtomicU64 = AtomicU64::new(0);

std::thread_local! {
    /// This thread is landing held pieces: what it does to title memory
    /// meanwhile lands nothing.
    static LANDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn table() -> MutexGuard<'static, Table> {
    TABLE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Since the last frame-rate line.
mod stat {
    /// Pieces held anew, pieces found held, their bytes (not fetched).
    pub(super) const NEW: usize = 0;
    pub(super) const AGAIN: usize = 1;
    pub(super) const BYTES: usize = 2;
    /// Pieces fetched at once after all: the block is no view of ours to
    /// write through (or shown at too many addresses, or a mapping was
    /// changing); not guarded at every view, or the kernel is writing it;
    /// hot; held by another copy or being landed.
    pub(super) const NO_VIEW: usize = 3;
    pub(super) const NOT_GUARDED: usize = 4;
    pub(super) const HOT: usize = 5;
    pub(super) const TAKEN: usize = 6;
    /// Blocks landed: for a read of the title's, a write of the title's, a
    /// write of ours or the kernel's or a change of mapping, wanted, idle.
    pub(super) const READ: usize = 7;
    pub(super) const WRITE: usize = 8;
    pub(super) const OURS: usize = 9;
    pub(super) const WANTED: usize = 10;
    pub(super) const IDLE: usize = 11;
    /// Bytes fetched for them, fetches, and microseconds the title's
    /// threads waited in faults.
    pub(super) const LANDED_BYTES: usize = 12;
    pub(super) const FETCHES: usize = 13;
    pub(super) const FAULT_US: usize = 14;
    /// Reads of ours that took a held block through our own view; blocks
    /// that became hot.
    pub(super) const BESIDE: usize = 15;
    pub(super) const PROMOTED: usize = 16;
    pub(super) const COUNT: usize = 17;
}

static STATS: [AtomicU64; stat::COUNT] = [const { AtomicU64::new(0) }; stat::COUNT];

fn count(which: usize, n: u64) {
    STATS[which].fetch_add(n, Ordering::Relaxed);
}

/// Whether anything is held (one load: asked on every path that reads or
/// writes title memory).
pub(crate) fn active() -> bool {
    HELD.load(Ordering::Acquire) != 0
}

/// The frame-rate line's.
pub(crate) fn log(frames: f64) {
    if !enabled() {
        return;
    }
    let t: Vec<f64> = STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "lazy copy-back (VICTORY_LAZY_BACK=1) per frame: pieces held {:.1} new + {:.1} again ({:.2} MB not fetched); \
         fetched at once: {:.1} no view of ours, {:.1} not guarded, {:.1} hot, {:.1} held otherwise; \
         blocks landed: {:.1} title reads, {:.1} title writes, {:.1} our writes and mappings, {:.1} wanted, {:.1} idle \
         ({:.2} MB in {:.1} fetches, {:.2} ms of title threads waiting); {:.1} reads of ours beside; {:.2} blocks made hot; \
         {} blocks held now",
        t[stat::NEW],
        t[stat::AGAIN],
        t[stat::BYTES] / 1048576.0,
        t[stat::NO_VIEW],
        t[stat::NOT_GUARDED],
        t[stat::HOT],
        t[stat::TAKEN],
        t[stat::READ],
        t[stat::WRITE],
        t[stat::OURS],
        t[stat::WANTED],
        t[stat::IDLE],
        t[stat::LANDED_BYTES] / 1048576.0,
        t[stat::FETCHES],
        t[stat::FAULT_US] / 1000.0,
        t[stat::BESIDE],
        t[stat::PROMOTED],
        HELD.load(Ordering::Relaxed),
    );
}

/// `ranges` with `[lo, hi)`: in order, what overlaps or touches as one.
fn joined(ranges: &mut Vec<(usize, usize)>, lo: usize, hi: usize) {
    ranges.push((lo, hi));
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(a, b) in ranges.iter() {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    *ranges = out;
}

/// Whether a hot block is fetched at once at `frame` (jman's rule: hot
/// while it is written, tried anew after [`REPROBE`]).
fn stays_hot(table: &mut Table, block: usize, frame: u64) -> bool {
    let Some(hot) = table.hot.get_mut(&block) else {
        return false;
    };
    if frame > hot.last.saturating_add(DECAY) {
        table.hot.remove(&block);
        return false;
    }
    let threshold = promote();
    if hot.faults < threshold {
        return false;
    }
    if frame > hot.since.saturating_add(REPROBE << hot.probes.min(4)) {
        hot.faults = threshold - 1;
        hot.last = frame;
        hot.since = frame;
        hot.probes = hot.probes.saturating_add(1);
        hot.probing = true;
        return false;
    }
    hot.last = frame;
    true
}

/// The title touched `block` while it was held, in `frame`.
fn touched(table: &mut Table, block: usize, frame: u64) {
    let threshold = promote();
    let hot = table.hot.entry(block).or_insert(Hot { faults: 0, last: frame, since: frame, probes: 0, probing: false });
    let before = hot.faults;
    let counted = if hot.faults == 0 || !hot.probing && frame > hot.last.saturating_add(DECAY) {
        hot.faults = 1;
        true
    } else if hot.probing || frame != hot.last {
        hot.faults = hot.faults.saturating_add(1);
        true
    } else {
        false
    };
    hot.last = frame;
    hot.probing = false;
    if counted && hot.faults >= threshold && (hot.faults == 1 || before < threshold) {
        hot.since = frame;
        count(stat::PROMOTED, 1);
    }
}

/// A flight that ran behind send `after` wrote offsets `[lo, hi)` of `copy`
/// (a piece: within one 64 KiB block of title memory), and the flight has
/// landed. Whether the piece is held here, not to be fetched now.
pub(crate) fn defer(copy: &Arc<Shadow>, lo: usize, hi: usize, after: u64) -> bool {
    let title = copy.span().0 as usize + lo;
    let block = title / BLOCK;
    if lo >= hi || (title + (hi - lo) - 1) / BLOCK != block {
        return false;
    }
    let frame = crate::frame::current();
    let mut table = table();
    if let Some(&at) = table.views.get(&block) {
        let same = table.held.get_mut(&at).filter(|h| at == block && !h.landing && Arc::ptr_eq(&h.copy, copy));
        let Some(held) = same else {
            // Another copy's, or by another address, or on its way into
            // title memory with a fetch that may be older than this flight:
            // fetched now, and what is held follows.
            if let Some(held) = table.held.get_mut(&at) {
                held.wanted = true;
            }
            count(stat::TAKEN, 1);
            return false;
        };
        joined(&mut held.ranges, lo, hi);
        held.after = held.after.max(after);
        held.frame = frame;
        count(stat::AGAIN, 1);
        count(stat::BYTES, (hi - lo) as u64);
        return true;
    }
    if stays_hot(&mut table, block, frame) {
        count(stat::HOT, 1);
        return false;
    }
    // No mapping changes until the views are held: all of them, as the
    // record has them now.
    let views: Vec<usize> = {
        let Some(_reading) = crate::watch::title_memory_reading_now() else {
            count(stat::NO_VIEW, 1);
            return false;
        };
        let Some((_, shown)) = crate::inplace::beside_page(block * BLOCK) else {
            count(stat::NO_VIEW, 1);
            return false;
        };
        let views: Vec<usize> = shown.into_iter().filter(|&view| view != 0).collect();
        if views.iter().any(|view| table.views.contains_key(&(view / BLOCK))) {
            count(stat::TAKEN, 1);
            return false;
        }
        if !crate::watch::lazy_hold(&views) {
            count(stat::NOT_GUARDED, 1);
            return false;
        }
        views
    };
    for view in &views {
        table.views.insert(view / BLOCK, block);
    }
    table.held.insert(
        block,
        Held { copy: copy.clone(), ranges: vec![(lo, hi)], after, views, frame, landing: false, wanted: false },
    );
    HELD.fetch_add(1, Ordering::AcqRel);
    count(stat::NEW, 1);
    count(stat::BYTES, (hi - lo) as u64);
    true
}

/// The pieces of `copy` held in title memory `[from, to)`, as addresses:
/// its refresh leaves them alone (`Shadow::update`).
pub(crate) fn held(copy: &Shadow, from: usize, to: usize) -> Vec<(usize, usize)> {
    let start = copy.span().0 as usize;
    let table = table();
    let mut out = Vec::new();
    for (_, held) in table.held.range(from / BLOCK..to.div_ceil(BLOCK)) {
        if std::ptr::eq(Arc::as_ptr(&held.copy), copy) {
            out.extend(held.ranges.iter().map(|&(lo, hi)| (start + lo, start + hi)).filter(|&(a, e)| a < to && e > from));
        }
    }
    out
}

/// What is held of title memory `[at, at + len)` is landed at the next
/// [`tend`]: bytes went into it beside the guards, or a copy of ours read
/// it as it is.
pub(crate) fn want(at: usize, len: usize) {
    if len == 0 || !active() || LANDING.get() {
        return;
    }
    let mut table = table();
    let blocks: Vec<usize> = table.views.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).map(|(_, &b)| b).collect();
    for block in blocks {
        if let Some(held) = table.held.get_mut(&block) {
            held.wanted = true;
        }
    }
}

/// A read of ours of title memory `[src, src + dst.len())` failed. If
/// blocks of it are held: the title's bytes as they are now, those of held
/// blocks through our own view, and the held blocks are wanted. `false` if
/// none is held, or the bytes are not there.
///
/// # Safety
/// `dst` is ours for `len` bytes, and no Rust reference covers it.
pub(crate) unsafe fn read_held(src: usize, dst: *mut u8, len: usize) -> bool {
    if !crate::watch::lazy_any(src, len) {
        return false;
    }
    let _reading = crate::watch::title_memory_reading_now();
    let (mut at, end) = (src, src + len);
    while at < end {
        let n = (BLOCK - at % BLOCK).min(end - at);
        let from = match crate::watch::lazy_any(at, n) {
            true => match crate::inplace::beside(at, n) {
                Some(own) => own,
                None => return false,
            },
            false => at,
        };
        // SAFETY: the caller's; a source that is not there fails the copy.
        if !unsafe { crate::safecopy::copy(dst.add(at - src), from as *const u8, n) } {
            return false;
        }
        at += n;
    }
    count(stat::BESIDE, 1);
    want(src, len);
    true
}

/// What one thread lands of the table.
struct Claim {
    block: usize,
    copy: Arc<Shadow>,
    ranges: Vec<(usize, usize)>,
    after: u64,
    views: Vec<usize>,
}

/// The claims of a landing, done with when it goes: the views are readable
/// again and the blocks are held no more, whether the landing ended or
/// unwound.
struct Landing(Vec<Claim>);

impl Drop for Landing {
    fn drop(&mut self) {
        for claim in &self.0 {
            crate::watch::lazy_release(&claim.views);
        }
        {
            let mut table = table();
            for claim in &self.0 {
                if table.held.remove(&claim.block).is_some() {
                    HELD.fetch_sub(1, Ordering::AcqRel);
                }
                for view in &claim.views {
                    table.views.remove(&(view / BLOCK));
                }
            }
        }
        LANDING.set(false);
        LANDED.notify_all();
    }
}

/// The bytes the card holds of `pieces` (a copy and offsets), one after the
/// other; and the send of title bytes the fetch ran behind, if it is known
/// better than by the flights that wrote them.
fn fetch(pieces: &[(Arc<Shadow>, usize, usize)]) -> (Vec<u8>, Option<u64>) {
    #[cfg(test)]
    if let Some(card) = tests::card(pieces) {
        return (card, None);
    }
    crate::host::lazy_fetch(pieces)
}

/// Lands `claims`: one fetch for all, each merged as a flight's landing
/// merges it, and noted.
fn land(claims: Vec<Claim>, why: usize) {
    LANDING.set(true);
    let landing = Landing(claims);
    let pieces: Vec<(Arc<Shadow>, usize, usize)> =
        landing.0.iter().flat_map(|c| c.ranges.iter().map(move |&(lo, hi)| (c.copy.clone(), lo, hi))).collect();
    let (bytes, ran_behind) = fetch(&pieces);
    count(why, landing.0.len() as u64);
    count(stat::FETCHES, 1);
    count(stat::LANDED_BYTES, bytes.len() as u64);
    let mut notes = crate::land::Notes::default();
    let mut at = 0;
    for claim in &landing.0 {
        let addressed: Vec<(usize, usize, usize)> = claim
            .ranges
            .iter()
            .map(|&(lo, hi)| {
                let piece = (lo, hi, bytes.as_ptr() as usize + at);
                at += hi - lo;
                piece
            })
            .collect();
        if at > bytes.len() {
            break;
        }
        // SAFETY: the fetched bytes, which stay as they are; each piece is
        // of its length.
        unsafe { claim.copy.land_card(&addressed, ran_behind.unwrap_or(claim.after), &mut notes, false) };
    }
    // The copies are the claims', alive until the landing goes.
    notes.flush();
}

/// Lands the blocks `pick` names (by their own block) that nobody is
/// landing, and waits for those someone is.
fn land_where(pick: impl Fn(&Table) -> Vec<usize>, why: usize) {
    if LANDING.get() {
        return;
    }
    loop {
        let mut table = table();
        let all = pick(&table);
        let frame = crate::frame::current();
        let mut claims = Vec::new();
        for &block in &all {
            let Some(held) = table.held.get_mut(&block).filter(|h| !h.landing) else {
                continue;
            };
            held.landing = true;
            claims.push(Claim {
                block,
                copy: held.copy.clone(),
                ranges: held.ranges.clone(),
                after: held.after,
                views: held.views.clone(),
            });
            if matches!(why, stat::READ | stat::WRITE | stat::WANTED) {
                touched(&mut table, block, frame);
            }
        }
        let others = all.len() - claims.len();
        if claims.is_empty() {
            if others == 0 {
                return;
            }
            drop(LANDED.wait(table).unwrap_or_else(|e| e.into_inner()));
            continue;
        }
        drop(table);
        land(claims, why);
        if others == 0 {
            return;
        }
    }
}

/// The blocks that hold views in `[first, last)`, each once.
fn holders(table: &Table, first: usize, last: usize) -> Vec<usize> {
    let mut blocks: Vec<usize> = table.views.range(first..last).map(|(_, &b)| b).collect();
    blocks.sort_unstable();
    blocks.dedup();
    blocks
}

/// A thread's access to `at` (not one of our fault-taking copies) faulted
/// in a held block: what shaders wrote there is landed, or waited for if
/// another thread is landing it, and the access is to be made again. Not
/// (`false`) if the thread is the one landing it: that access is a fault
/// like any other, nobody else's to wait for.
pub(crate) fn faulted(at: usize, write: bool) -> bool {
    if LANDING.get() {
        return false;
    }
    let began = std::time::Instant::now();
    let block = at / BLOCK;
    land_where(|table| holders(table, block, block + 1), if write { stat::WRITE } else { stat::READ });
    count(stat::FAULT_US, began.elapsed().as_micros() as u64);
    true
}

/// Before title memory `[at, at + len)` is written by us or by the kernel,
/// or shows something else: what is held of it is landed first.
pub(crate) fn land_range(at: usize, len: usize) {
    if len == 0 || !active() {
        return;
    }
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    land_where(|table| holders(table, first, last), stat::OURS);
}

/// On the thread that lands flights, before a landing: what is wanted is
/// landed, and what no flight wrote for [`idle`] frames.
pub(crate) fn tend() {
    if !active() {
        return;
    }
    let frame = crate::frame::current();
    land_where(|table| table.held.iter().filter(|(_, h)| h.wanted).map(|(&b, _)| b).collect(), stat::WANTED);
    let idle = idle();
    if idle != 0 {
        land_where(
            |table| {
                table.held.iter().filter(|(_, h)| h.frame.saturating_add(idle) <= frame).map(|(&b, _)| b).take(TEND_MOST).collect()
            },
            stat::IDLE,
        );
    }
}

#[cfg(test)]
mod tests;
