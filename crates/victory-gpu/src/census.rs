//! `VICTORY_CENSUS=1` (off by default): what of the copy-back of shader
//! writes comes from memory the title reserved write-combining, and who
//! takes those bytes afterwards. Written to the log every
//! `VICTORY_CENSUS_SECONDS` (10). The census is logging only: it changes
//! nothing of what is copied, when, or where.
//!
//! From fork PR #868 (Chipsraine; the per-range classification and the
//! in-flight count by iGoldenNinja): the `XMemVirtualAlloc` reservation each
//! copy in the card's memory lies in, with the flags the title asked for
//! (the runtime drops write-combining, uncached and the GPU access bits;
//! they are kept here); per copy the bytes sent up, the pieces compared and
//! the bytes fetched back; the host resources made per frame by kind.
//!
//! Ours: this tree keeps many copies of the same title memory (a buffer
//! inside a bigger one with a copy of its own, windows over memory outside
//! buffers, copies in place, textures). A byte a shader wrote through one
//! copy reaches the others only through title memory: the write-back puts
//! it there, the write watch tells the others, and they take it again when
//! next asked. So the census notes, per 64 KiB page of write-combining
//! memory (WC), which copy in the card's memory last wrote bytes back
//! there, and counts every later taker of those bytes:
//! - another copy brought up to date over them (a buffer's copy, a window,
//!   a copy in place) or a texture over them: such a page cannot be left on
//!   the card, the other copy would read old bytes;
//! - the same copy taking the page again (the upload echo: the write-back
//!   wrote title memory, the write watch tells of the page, and the copy's
//!   next refresh takes it in whole and sends it up again);
//!
//! and, of the WC bytes fetched, how many lie in pages no other live copy
//! covers (textures read from title memory count as copies), and how many
//! in pages no other copy was ever seen taking after a card write: what
//! [`skip_wc_back`] would leave on the card with `1` and with `2`. What the
//! title's own CPU reads is not seen: nothing watches reads. Copies in the
//! PC's memory, merged on the CPU, are counted on a line of their own.
//!
//! The same notes serve [`skip_wc_back`] (`VICTORY_SKIP_WC_BACK`), which
//! keeps them whether or not the census logs.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Title memory is noted by 64 KiB page, as the card's compare cuts it.
const PAGE: usize = 64 << 10;

/// Whether the census is on (`VICTORY_CENSUS=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CENSUS").is_some_and(|v| v == "1"))
}

/// Whether the pages and the reservations are kept: with the census or
/// with [`skip_wc_back`].
pub(crate) fn tracking() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on() || skip_wc_back() != Skip::Off)
}

/// `VICTORY_SKIP_WC_BACK` (off by default), an experiment after fork PR
/// #868 (Chipsraine): what shaders wrote to a title buffer's copy in the
/// card's memory, in memory the title reserved write-combining, is left on
/// the card instead of being fetched back into title memory. The console's
/// CPU writes such memory and does not read it back (#868's census: 97-99 %
/// of some 100 MB fetched back a frame came from it); fetched back, a page
/// also went up again whole with the copy's next refresh (the echo).
///
/// What it assumes, and what guards it here:
/// - The title's CPU does not read those bytes. Not seen: nothing watches
///   reads. With `VICTORY_SKIP_WC_BACK_CHECK=N` ([`check_every`]), every N
///   frames one landing leaves nothing on the card and asks for the pieces
///   left before with the next flight that declares them: a reader then
///   sees bytes some N frames old at worst, and the census counts what that
///   landing wrote into title memory (what had differed).
/// - No other copy reads them. Another copy of the same memory (a buffer
///   inside this one with a copy of its own, a window, a copy in place, a
///   texture) gets a shader's write only through title memory. So a piece
///   is left only in a title buffer's copy (not a window: windows are let
///   go when idle, and what they hold with them), only in a page shown at
///   one address, only where no copy other than the one that wrote it ever
///   wrote it back or took it after a card write, and with `1` only in a
///   page no other live copy covers at all. A copy that takes a page while
///   it holds bytes left on the card is counted, the page is fetched with
///   the next flight that declares it, and it is never left again.
/// - The title's CPU writing part of such a page: the refresh would take
///   the whole page from title memory, old bytes where the shaders wrote,
///   and send it up over the card's. Here only the bytes title memory
///   changed go into the CPU's side and up there ([`stale_pages`],
///   `Shadow::copy`).
///
/// `1`: as above. `2`: also in pages other live copies cover, until one of
/// them is seen taking a card write there (that one reads old bytes once).
/// Any other value: off, and nothing changes. The compare-skip from that
/// experiment is not taken: a device removal and a black world were
/// reported with it.
///
/// `VICTORY_LAZY_BACK` (the lazy copy-back) cuts the same fetch another
/// way: it holds a flagged piece on the card and fetches it when the title
/// looks. With both, a piece this switch leaves is never held there (it is
/// asked first, and the title then reads title memory's bytes as they
/// are); the pieces it fetches, those due back among them, may be held
/// until the title looks. `VICTORY_CARD_BUFFERS_MIN_KB` is not changed
/// here; measured with smaller copies in the card's memory, this switch
/// did not gain, and small copies took bytes left on the card: a run uses
/// one or the other.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Skip {
    Off,
    /// `1`: in pages no other copy covers.
    Alone,
    /// `2`: in pages no other copy took a card write in.
    Untaken,
}

pub(crate) fn skip_wc_back() -> Skip {
    static MODE: std::sync::OnceLock<Skip> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_SKIP_WC_BACK").as_deref().map(str::trim) {
            Ok("1") => Skip::Alone,
            Ok("2") => Skip::Untaken,
            _ => Skip::Off,
        };
        if mode != Skip::Off {
            tracing::warn!(
                ?mode,
                check_every = check_every(),
                "VICTORY_SKIP_WC_BACK: what shaders write to copies of write-combining title memory is left on the card where no \
                 other copy takes it (an experiment after fork PR #868)"
            );
        }
        mode
    })
}

/// `VICTORY_SKIP_WC_BACK_HOLD=1` (off by default; read only with
/// [`skip_wc_back`] on): a piece the rules would leave on the card is held
/// there through `crate::lazy_back` instead (ticket 110): every address the
/// title sees its page at is made unreadable together, so a later read of
/// the title's, through any of them, faults, lands what the shaders wrote
/// and reads it. Nothing is noted stale and nothing is fetched until then
/// (or until lazy_back's own landings: a write of ours there, another copy
/// asking, idle frames). A piece that cannot be held (its block is no view
/// of ours, not guarded at every address, hot, held by another copy) is
/// fetched as without the switch: it is never left unguarded.
///
/// With it, a page shown at more than one address is no longer refused
/// outright: the rules of `1` and `2` are asked of each of its addresses
/// ([`Pages::may_hold`]), and with `1` no live copy may cover any address
/// but the writer's own. Needs what lazy_back rests on (VICTORY_RELAY,
/// VICTORY_LAND, VICTORY_LAND_BESIDE as by default); without that it stays
/// off with a warning, and `1` and `2` leave pieces as before.
pub(crate) fn hold() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        if skip_wc_back() == Skip::Off || !std::env::var("VICTORY_SKIP_WC_BACK_HOLD").is_ok_and(|v| v.trim() == "1") {
            return false;
        }
        let can = crate::lazy_back::can();
        if can {
            tracing::warn!(
                "VICTORY_SKIP_WC_BACK_HOLD=1: pieces VICTORY_SKIP_WC_BACK leaves are held unreadable at every address instead, \
                 and land when the title looks"
            );
        } else {
            tracing::warn!(
                "VICTORY_SKIP_WC_BACK_HOLD=1 is left off: it needs VICTORY_RELAY, VICTORY_LAND and VICTORY_LAND_BESIDE as they \
                 are by default"
            );
        }
        can
    })
}

/// `VICTORY_SKIP_WC_MIN_KB=N` (read only with [`skip_wc_back`] on; unset or
/// 0: every size, as before): copies of title buffers under N KiB always
/// come back, held or not. After grosterr/rtx3060 857308712 and 53a48f6c3
/// (default 1024 there, the name of fxceless/victory-fork#1163): small
/// write-combining buffers hold counts and visibility lists the title's CPU
/// reads; left on the card they gave flashes and half the draws there.
/// Their compare-skip half (`VICTORY_SKIP_WC_COMPARE`) is not here (see
/// [`skip_wc_back`]).
pub(crate) fn skip_wc_min_bytes() -> u64 {
    static MIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("VICTORY_SKIP_WC_MIN_KB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0).saturating_mul(1024)
    })
}

/// What [`skip_back`] decides for a flagged piece.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Back {
    /// Fetched and merged, as without the switch.
    Fetch,
    /// Left on the card, title memory noted stale.
    Leave,
    /// To be held on the card through `crate::lazy_back` ([`hold`]);
    /// fetched if it cannot be. Whether the page is shown at more than one
    /// address.
    Hold(bool),
}

/// `VICTORY_SKIP_WC_BACK_CHECK=N` (N at least 2; off by default): one
/// landing in every N frames leaves nothing on the card ([`skip_wc_back`]).
pub(crate) fn check_every() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_SKIP_WC_BACK_CHECK").ok().and_then(|v| v.trim().parse().ok()).filter(|&n| n >= 2).unwrap_or(0)
    })
}

/// Whether the landing that asks is the check of its N frames
/// ([`check_every`]): the first to ask once a multiple of N frames is reached.
pub(crate) fn check_landing() -> bool {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let every = check_every();
    every != 0 && skip_wc_back() != Skip::Off && check_due(&LAST, crate::frame::current(), every)
}

fn check_due(last: &AtomicU64, frame: u64, every: u64) -> bool {
    let epoch = frame / every + 1;
    let was = last.load(Ordering::Relaxed);
    epoch > was && last.compare_exchange(was, epoch, Ordering::Relaxed, Ordering::Relaxed).is_ok()
}

fn period() -> Duration {
    static S: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    Duration::from_secs(*S.get_or_init(|| {
        std::env::var("VICTORY_CENSUS_SECONDS").ok().and_then(|v| v.trim().parse().ok()).filter(|&s| s > 0).unwrap_or(10)
    }))
}

/// A reservation of `XMemVirtualAlloc`, as the title asked for it.
struct Reservation {
    len: usize,
    kind: u32,
    flags: u64,
    protect: u32,
}

/// The console's write-combining protection bit.
const WRITE_COMBINE: u32 = 0x400;

static RESERVATIONS: Mutex<BTreeMap<usize, Reservation>> = Mutex::new(BTreeMap::new());

fn reservations() -> MutexGuard<'static, BTreeMap<usize, Reservation>> {
    RESERVATIONS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The runtime reserved `[at, at + len)` for the title (`XMemVirtualAlloc`
/// at an address of its choice), with the title's own `kind`, XMem `flags`
/// and console `protect` (before the runtime cleans them).
pub fn title_memory_reserved(at: usize, len: usize, kind: u32, flags: u64, protect: u32) {
    // The account by owner (`VICTORY_MEMORY_ACCOUNT`): the title's own,
    // wherever it lies (a reservation may lie above the console's 1 TiB).
    #[cfg(feature = "diagnostics")]
    crate::account::title(at, len as u64);
    if !tracking() || at == 0 || len == 0 {
        return;
    }
    reservations().insert(at, Reservation { len, kind, flags, protect });
}

/// Whether every title range of `ranges` lies wholly inside a single
/// reservation the title asked write-combining. A copy (a window above
/// all) can cover several reservations, so each range is looked at on its
/// own (iGoldenNinja, fork `f9c43fa`).
pub(crate) fn all_wc(ranges: &[(usize, usize)]) -> bool {
    all_wc_in(&reservations(), ranges)
}

fn all_wc_in(reservations: &BTreeMap<usize, Reservation>, ranges: &[(usize, usize)]) -> bool {
    !ranges.is_empty()
        && ranges.iter().all(|&(from, to)| {
            // Seeking from the exclusive end also finds a reservation that
            // begins inside the range. One ending at `to` holds it.
            from < to
                && reservations.range(..to).next_back().is_some_and(|(&base, r)| {
                    base <= from && r.protect & WRITE_COMBINE != 0 && base.checked_add(r.len).is_some_and(|end| to <= end)
                })
        })
}

/// What a later taker of title memory is ([`asked`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Taker {
    /// A title buffer's copy.
    Copy,
    /// A window over memory outside buffers.
    Window,
    /// A copy in place (the card reads title memory itself).
    InPlace,
    /// A texture read from title memory.
    Texture,
}

/// A copy of title memory, as the census names it.
#[derive(Clone, Copy)]
pub(crate) struct Which {
    pub(crate) id: u64,
    pub(crate) start: u64,
    pub(crate) len: u64,
    pub(crate) window: bool,
}

/// What a texture is named by among the takers: its memory's address with
/// the top bit set (copies are numbered from 1 up).
pub(crate) fn texture_taker(at: usize) -> u64 {
    (1 << 63) | at as u64
}

/// The counters, since the start ([`COUNTS`]).
mod n {
    /// Pieces of copies in the card's memory compared, all and WC.
    pub(super) const COMPARED: usize = 0;
    pub(super) const COMPARED_WC: usize = 1;
    /// Bytes fetched back from the card, all and WC.
    pub(super) const FETCHED: usize = 2;
    pub(super) const FETCHED_WC: usize = 3;
    /// Bytes the merges of those stored into title memory, all and WC.
    pub(super) const WRITTEN: usize = 4;
    pub(super) const WRITTEN_WC: usize = 5;
    /// WC bytes fetched in pages one copy covers, at one address.
    pub(super) const WC_ALONE: usize = 6;
    /// WC bytes fetched in pages no other copy took after a card write so
    /// far, at one address.
    pub(super) const WC_UNTAKEN: usize = 7;
    /// WC bytes fetched in pages shown at more than one address.
    pub(super) const WC_MULTI: usize = 8;
    /// Bytes a copy in the card's memory took again of pages it wrote back.
    pub(super) const ECHO: usize = 9;
    /// Bytes of card-written WC pages taken afterwards by others, by kind.
    pub(super) const TAKEN_COPY: usize = 10;
    pub(super) const TAKEN_WINDOW: usize = 11;
    pub(super) const TAKEN_IN_PLACE: usize = 12;
    pub(super) const TAKEN_TEXTURE: usize = 13;
    /// Pages a first other taker took after a card write.
    pub(super) const TAKEN_PAGES: usize = 14;
    /// Pages written back through a copy after another copy wrote them back.
    pub(super) const TWO_WRITERS: usize = 15;
    /// Copies in the PC's memory: declared bytes compared and bytes stored
    /// into title memory, all and WC.
    pub(super) const PC_COMPARED: usize = 16;
    pub(super) const PC_COMPARED_WC: usize = 17;
    pub(super) const PC_WRITTEN: usize = 18;
    pub(super) const PC_WRITTEN_WC: usize = 19;
    /// Bytes sent up to copies in the card's memory.
    pub(super) const SENT: usize = 20;
    /// WC bytes fetched from windows (never left on the card).
    pub(super) const WC_WINDOW: usize = 21;
    /// `VICTORY_SKIP_WC_BACK`: pieces and bytes left on the card.
    pub(super) const LEFT_PIECES: usize = 22;
    pub(super) const LEFT: usize = 23;
    /// Bytes left on the card that another copy then took (old bytes), and
    /// how many times.
    pub(super) const STALE_TAKEN: usize = 24;
    pub(super) const STALE_TAKEN_PAGES: usize = 25;
    /// Bytes the merges wrote of pieces left before (fetched again).
    pub(super) const BROUGHT_BACK: usize = 26;
    /// Pages holding bytes left on the card that the title's CPU wrote
    /// again, and the bytes of its that went up there.
    pub(super) const TITLE_OVER: usize = 27;
    pub(super) const TITLE_OVER_UP: usize = 28;
    /// Pages holding bytes left on the card whose copy went.
    pub(super) const LOST: usize = 29;
    /// Check landings ([`super::check_every`]): bytes fetched of pieces the
    /// rules would have left, bytes the merges wrote of them, and of those
    /// the bytes of pieces left before.
    pub(super) const CHECK_FETCHED: usize = 30;
    pub(super) const CHECK_WRITTEN: usize = 31;
    pub(super) const CHECK_STALE: usize = 32;
    /// `VICTORY_SKIP_WC_BACK_HOLD`: pieces and bytes held on the card, of
    /// them in pages shown at more than one address, and pieces the rules
    /// allowed that lazy_back could not hold (fetched).
    pub(super) const HELD_PIECES: usize = 33;
    pub(super) const HELD: usize = 34;
    pub(super) const HELD_MULTI: usize = 35;
    pub(super) const HOLD_REFUSED: usize = 36;
    /// WC bytes fetched in pages shown at more than one address that the
    /// rules of [`super::hold`] (with `2`) would allow at every address.
    pub(super) const WC_MULTI_HOLDABLE: usize = 37;
    pub(super) const COUNT: usize = 38;
}

static COUNTS: [AtomicU64; n::COUNT] = [const { AtomicU64::new(0) }; n::COUNT];

fn count(i: usize, by: u64) {
    if by > 0 {
        COUNTS[i].fetch_add(by, Ordering::Relaxed);
    }
}

/// A page of WC title memory shaders wrote through a copy in the card's
/// memory, and the merge stored into title memory.
#[derive(Debug)]
struct Written {
    /// The copy that wrote it back last.
    owner: u64,
    /// The title bytes written back since its last other taker, as the
    /// smallest stretch that holds them.
    lo: usize,
    hi: usize,
    /// Who took them since (copies' numbers, [`texture_taker`]s).
    takers: Vec<u64>,
    /// Whether the owner took the page again since (the echo).
    echoed: bool,
    /// [`skip_wc_back`]: `[lo, hi)` was left on the card, and title memory
    /// lacks what the owner's shaders wrote there.
    stale: bool,
}

impl Written {
    fn new(owner: u64, lo: usize, hi: usize, stale: bool) -> Written {
        Written {
            owner,
            lo,
            hi,
            takers: Vec::new(),
            echoed: false,
            stale,
        }
    }
}

/// The pages noted ([`Written`]), how many live copies cover each page,
/// and the pages another copy took after a card write.
#[derive(Default)]
struct Pages {
    written: BTreeMap<usize, Written>,
    /// Pages noted per owner.
    owners: HashMap<u64, usize>,
    cover: HashMap<usize, u32>,
    /// Pages another copy took after a card write, or two copies wrote
    /// back: never left on the card.
    taken: HashSet<usize>,
    /// Pages noted `stale`, in all and per owner.
    stale: usize,
    stale_by: HashMap<u64, usize>,
    /// Stale pages (owner, page) to fetch with the next flight that
    /// declares them, whatever its compare says.
    refetch: HashSet<(u64, usize)>,
}

/// What [`Pages::took`] found: bytes taken, pages that had no other taker
/// before, and bytes taken that title memory lacked (left on the card).
#[derive(Default, Debug, PartialEq, Eq)]
struct Took {
    bytes: usize,
    pages: usize,
    stale: usize,
}

impl Pages {
    /// The pages of `[from, to)`.
    fn pages(from: usize, to: usize) -> std::ops::Range<usize> {
        if from >= to { 0..0 } else { from / PAGE..(to - 1) / PAGE + 1 }
    }

    /// A copy of `[start, start + len)` was made (`by` 1) or is gone (`-1`).
    fn cover(&mut self, start: usize, len: usize, made: bool) {
        for page in Self::pages(start, start.saturating_add(len)) {
            let c = self.cover.entry(page).or_insert(0);
            if made {
                *c += 1;
            } else {
                *c = c.saturating_sub(1);
                if *c == 0 {
                    self.cover.remove(&page);
                }
            }
        }
    }

    /// How many live copies cover `page`.
    fn covered(&self, page: usize) -> u32 {
        self.cover.get(&page).copied().unwrap_or(0)
    }

    /// Copy `owner` wrote title bytes `[lo, hi)` (inside one page) back.
    /// True if another copy had written the page back before.
    fn wrote(&mut self, owner: u64, lo: usize, hi: usize) -> bool {
        if lo >= hi {
            return false;
        }
        let page = lo / PAGE;
        let mut two = false;
        match self.written.get_mut(&page) {
            Some(w) if w.owner != owner && w.stale => {
                // The first copy's bytes left on the card are still not in
                // title memory: the page stays noted under it.
                self.taken.insert(page);
                (w.lo, w.hi) = (w.lo.min(lo), w.hi.max(hi));
                return true;
            }
            Some(w) if w.owner == owner && w.stale => {
                // Bytes left before, beside this write.
                (w.lo, w.hi) = (w.lo.min(lo), w.hi.max(hi));
                w.takers.clear();
                w.echoed = false;
            }
            Some(w) => {
                if w.owner != owner {
                    two = true;
                    self.taken.insert(page);
                    if let Some(n) = self.owners.get_mut(&w.owner) {
                        *n -= 1;
                        if *n == 0 {
                            self.owners.remove(&w.owner);
                        }
                    }
                    *self.owners.entry(owner).or_insert(0) += 1;
                    w.owner = owner;
                    (w.lo, w.hi) = (lo, hi);
                } else if w.takers.is_empty() && !w.echoed {
                    // Nobody took the earlier bytes yet: they are all new.
                    (w.lo, w.hi) = (w.lo.min(lo), w.hi.max(hi));
                } else {
                    (w.lo, w.hi) = (lo, hi);
                }
                w.takers.clear();
                w.echoed = false;
            }
            None => {
                self.written.insert(page, Written::new(owner, lo, hi, false));
                *self.owners.entry(owner).or_insert(0) += 1;
            }
        }
        two
    }

    /// `taker` (not `owner` of a page) brought `[from, to)` up to date:
    /// what of the noted pages' written bytes it took for the first time
    /// since their write. A stale page so taken is to be fetched again.
    fn took(&mut self, taker: u64, from: usize, to: usize) -> Took {
        let mut took = Took::default();
        let mut first = Vec::new();
        let mut again = Vec::new();
        for (&page, w) in self.written.range_mut(Self::pages(from, to)) {
            if w.owner == taker || w.takers.contains(&taker) {
                continue;
            }
            let (a, b) = (w.lo.max(from), w.hi.min(to));
            if a >= b {
                continue;
            }
            if w.takers.is_empty() {
                took.pages += 1;
                first.push(page);
            }
            if w.stale {
                took.stale += b - a;
                again.push((w.owner, page));
            }
            w.takers.push(taker);
            took.bytes += b - a;
        }
        self.taken.extend(first);
        self.refetch.extend(again);
        took
    }

    /// Whether copy `owner` may leave a piece of `page` on the card under
    /// `mode` ([`skip_wc_back`]): no other copy took a card write there or
    /// wrote it back, and with [`Skip::Alone`] no other live copy covers it.
    fn may_skip(&self, mode: Skip, owner: u64, page: usize) -> bool {
        mode != Skip::Off
            && !self.taken.contains(&page)
            && self.written.get(&page).is_none_or(|w| w.owner == owner)
            && (mode != Skip::Alone || self.covered(page) <= 1)
    }

    /// [`super::hold`]: whether copy `owner` may hold a piece of `page` on
    /// the card when the title also shows that page at `aliases` (pages,
    /// not `page`): [`Pages::may_skip`] at `page`, and at every alias no
    /// other copy took a card write or wrote it back, and with
    /// [`Skip::Alone`] no live copy covers it (not even `owner`: its bytes
    /// at the alias come from title memory too). A hold makes every address
    /// unreadable together, so a read through any of them lands it.
    fn may_hold(&self, mode: Skip, owner: u64, page: usize, aliases: &[usize]) -> bool {
        self.may_skip(mode, owner, page)
            && aliases.iter().all(|&alias| {
                !self.taken.contains(&alias)
                    && self.written.get(&alias).is_none_or(|w| w.owner == owner && !w.stale)
                    && (mode != Skip::Alone || self.covered(alias) == 0)
            })
    }

    /// Copy `owner` left title bytes `[lo, hi)` (inside one page) on the
    /// card ([`Pages::may_skip`] said it may).
    fn left(&mut self, owner: u64, lo: usize, hi: usize) {
        let page = lo / PAGE;
        match self.written.get_mut(&page) {
            Some(w) => {
                if w.stale {
                    (w.lo, w.hi) = (w.lo.min(lo), w.hi.max(hi));
                } else {
                    (w.lo, w.hi, w.stale) = (lo, hi, true);
                    self.stale += 1;
                    *self.stale_by.entry(owner).or_insert(0) += 1;
                }
                w.takers.clear();
                w.echoed = false;
            }
            None => {
                self.written.insert(page, Written::new(owner, lo, hi, true));
                *self.owners.entry(owner).or_insert(0) += 1;
                self.stale += 1;
                *self.stale_by.entry(owner).or_insert(0) += 1;
            }
        }
    }

    /// `n` of `owner`'s stale pages are not any more.
    fn stale_gone(&mut self, owner: u64, n: usize) {
        self.stale -= n;
        if let Some(by) = self.stale_by.get_mut(&owner) {
            *by = by.saturating_sub(n);
            if *by == 0 {
                self.stale_by.remove(&owner);
            }
        }
    }

    /// Copy `owner`'s piece `[lo, hi)` came back and was merged (`wrote`:
    /// it changed title memory). Returns whether another copy had written
    /// the page back, and whether it brought back all that was left there.
    fn merged(&mut self, owner: u64, lo: usize, hi: usize, wrote: bool) -> (bool, bool) {
        let page = lo / PAGE;
        let mut back = false;
        if let Some(w) = self.written.get_mut(&page)
            && w.stale
            && w.owner == owner
            && lo <= w.lo
            && w.hi <= hi
        {
            w.stale = false;
            back = true;
        }
        if back {
            self.stale_gone(owner, 1);
            self.refetch.remove(&(owner, page));
        }
        (wrote && self.wrote(owner, lo, hi), back)
    }

    /// Copy `id` is gone: its pages left on the card are lost (title memory
    /// keeps the old bytes) and never left again. Returns how many.
    fn gone(&mut self, id: u64) -> usize {
        if !self.owners.contains_key(&id) {
            return 0;
        }
        let mut lost = Vec::new();
        for (&page, w) in self.written.iter_mut() {
            if w.owner == id && w.stale {
                w.stale = false;
                lost.push(page);
            }
        }
        self.stale_gone(id, lost.len());
        self.refetch.retain(|r| r.0 != id);
        let n = lost.len();
        self.taken.extend(lost);
        n
    }

    /// Copy `id`'s stale pages in `[from, to)`.
    fn stale_in(&self, id: u64, from: usize, to: usize) -> Vec<usize> {
        self.written.range(Self::pages(from, to)).filter(|(_, w)| w.stale && w.owner == id).map(|(&page, _)| page).collect()
    }

    /// Copy `id` took title memory `[from, to)` again: the bytes of pages it
    /// wrote back that it takes for the first time since (whole pages' parts,
    /// as they go up again).
    fn recopied(&mut self, id: u64, from: usize, to: usize) -> usize {
        if !self.owners.contains_key(&id) {
            return 0;
        }
        let mut bytes = 0;
        for (&page, w) in self.written.range_mut(Self::pages(from, to)) {
            if w.owner != id || w.echoed {
                continue;
            }
            w.echoed = true;
            bytes += ((page + 1) * PAGE).min(to) - (page * PAGE).max(from);
        }
        bytes
    }
}

static PAGES: LazyLock<Mutex<Pages>> = LazyLock::new(|| Mutex::new(Pages::default()));

fn pages() -> MutexGuard<'static, Pages> {
    PAGES.lock().unwrap_or_else(|e| e.into_inner())
}

/// How many copies own noted pages, and the one when it is one: an ask of
/// that copy has nothing to look for.
static OWNERS: AtomicUsize = AtomicUsize::new(0);
static SOLE_OWNER: AtomicU64 = AtomicU64::new(0);
/// Pages noted stale, and pages to fetch again ([`Pages`]).
static STALE: AtomicUsize = AtomicUsize::new(0);
static REFETCH: AtomicUsize = AtomicUsize::new(0);
/// The copies with stale pages, as bit `id % 64`: a refresh of any other
/// copy has nothing to look for ([`stale_pages`]).
static STALE_OWNERS: AtomicU64 = AtomicU64::new(0);

/// After a change of the pages: what the hooks look at without the lock.
fn noted(p: &Pages) {
    OWNERS.store(p.owners.len(), Ordering::Relaxed);
    SOLE_OWNER.store(p.owners.keys().next().copied().unwrap_or(0), Ordering::Relaxed);
    STALE.store(p.stale, Ordering::Relaxed);
    REFETCH.store(p.refetch.len(), Ordering::Relaxed);
    STALE_OWNERS.store(p.stale_by.keys().fold(0, |bits, id| bits | 1 << (id % 64)), Ordering::Relaxed);
}

/// A copy of title memory `[start, start + len)` was made (any kind: in the
/// card's memory, in the PC's, in place; a buffer's, a window).
pub(crate) fn copy_made(start: u64, len: u64) {
    if tracking() {
        pages().cover(start as usize, len as usize, true);
    }
}

/// Copy `id` of `[start, start + len)` is gone.
pub(crate) fn copy_gone(id: u64, start: u64, len: u64) {
    if !tracking() {
        return;
    }
    let mut p = pages();
    p.cover(start as usize, len as usize, false);
    let lost = p.gone(id);
    if lost > 0 {
        noted(&p);
        drop(p);
        count(n::LOST, lost as u64);
        tracing::warn!(
            copy = id,
            start = format_args!("{start:#x}"),
            pages = lost,
            "VICTORY_SKIP_WC_BACK: a copy went with pages it had left on the card; title memory keeps older bytes there"
        );
    }
}

/// The other pages the title shows title page `page` (an index) at.
fn aliases(page: usize) -> Vec<usize> {
    crate::inplace::pages_of(page * PAGE).into_iter().map(|at| at / PAGE).filter(|&other| other != page).collect()
}

/// [`skip_wc_back`]: whether piece `[lo, hi)` (offsets) of copy `which`,
/// flagged by the card's compare under the declared title range
/// `declared`, is left on the card instead of being fetched back, or (with
/// [`hold`]) is to be held there. Not in a check landing (`check`,
/// [`check_landing`]). See [`skip_wc_back`] for the rules.
pub(crate) fn skip_back(which: Which, declared: (usize, usize), lo: usize, hi: usize, check: bool) -> Back {
    let mode = skip_wc_back();
    if mode == Skip::Off || check || which.window || lo >= hi || which.len < skip_wc_min_bytes() {
        return Back::Fetch;
    }
    let (a, b) = (which.start as usize + lo, which.start as usize + hi);
    if !all_wc(&[declared, (a, b)]) {
        return Back::Fetch;
    }
    let multi = crate::inplace::view_census(a, b - a).1 > 0;
    if hold() {
        // Asked before the pages' lock: the views' is not taken under it.
        let aliases = if multi { aliases(a / PAGE) } else { Vec::new() };
        let p = pages();
        if !p.may_hold(mode, which.id, a / PAGE, &aliases) || p.refetch.contains(&(which.id, a / PAGE)) {
            return Back::Fetch;
        }
        return Back::Hold(multi);
    }
    if multi {
        return Back::Fetch;
    }
    let mut p = pages();
    // A page due to come back ([`refetch_due`]) comes back.
    if !p.may_skip(mode, which.id, a / PAGE) || p.refetch.contains(&(which.id, a / PAGE)) {
        return Back::Fetch;
    }
    p.left(which.id, a, b);
    noted(&p);
    drop(p);
    count(n::LEFT, (b - a) as u64);
    let pieces = COUNTS[n::LEFT_PIECES].fetch_add(1, Ordering::Relaxed) + 1;
    if pieces.is_power_of_two() {
        tracing::info!(
            "VICTORY_SKIP_WC_BACK: {pieces} pieces ({} MB) of write-combining memory left on the card so far",
            COUNTS[n::LEFT].load(Ordering::Relaxed) >> 20
        );
    }
    Back::Leave
}

/// [`hold`]: a piece [`skip_back`] said to hold, `bytes` long (in a page
/// shown at more than one address: `multi`), was held (`held`) or could
/// not be, and is fetched.
pub(crate) fn hold_done(bytes: usize, multi: bool, held: bool) {
    if !held {
        count(n::HOLD_REFUSED, 1);
        return;
    }
    count(n::HELD, bytes as u64);
    if multi {
        count(n::HELD_MULTI, bytes as u64);
    }
    let pieces = COUNTS[n::HELD_PIECES].fetch_add(1, Ordering::Relaxed) + 1;
    if pieces.is_power_of_two() {
        tracing::info!(
            "VICTORY_SKIP_WC_BACK_HOLD: {pieces} pieces ({} MB) of write-combining memory held on the card so far",
            COUNTS[n::HELD].load(Ordering::Relaxed) >> 20
        );
    }
}

/// Copy `id`'s pages in title memory `[from, to)` that hold bytes left on
/// the card: a refresh there must not send title memory's old bytes up
/// over them (`Shadow::copy`).
pub(crate) fn stale_pages(id: u64, from: usize, to: usize) -> Vec<usize> {
    if STALE_OWNERS.load(Ordering::Relaxed) & 1 << (id % 64) == 0 || from >= to {
        return Vec::new();
    }
    pages().stale_in(id, from, to)
}

/// The title wrote a page holding bytes left on the card: `up` bytes of its
/// went into the copy's CPU side and up.
pub(crate) fn title_over(up: usize) {
    count(n::TITLE_OVER, 1);
    count(n::TITLE_OVER_UP, up as u64);
}

/// For a landing ([`skip_wc_back`]): the stale pages (copy, page) to fetch
/// whatever the compares say, all of them in a check landing (`check`).
/// `None` if there are none.
pub(crate) fn refetch_due(check: bool) -> Option<HashSet<(u64, usize)>> {
    if skip_wc_back() == Skip::Off || (!check && REFETCH.load(Ordering::Relaxed) == 0) {
        return None;
    }
    let mut p = pages();
    if check {
        let stale: Vec<(u64, usize)> = p.written.iter().filter(|(_, w)| w.stale).map(|(&page, w)| (w.owner, page)).collect();
        p.refetch.extend(stale);
        noted(&p);
    }
    (!p.refetch.is_empty()).then(|| p.refetch.clone())
}

/// Tests: copy `id` left title bytes `[a, b)` on the card, or is done with.
#[cfg(test)]
pub(crate) fn left_for_test(id: u64, range: Option<(usize, usize)>) {
    let mut p = pages();
    match range {
        Some((a, b)) => p.left(id, a, b),
        None => {
            p.gone(id);
        }
    }
    noted(&p);
}

/// Whether the piece at offset `lo` of copy `which` is in `due`
/// ([`refetch_due`]).
pub(crate) fn due(due: &HashSet<(u64, usize)>, which: Which, lo: usize) -> bool {
    due.contains(&(which.id, (which.start as usize + lo) / PAGE))
}

/// `taker` (a copy's number or a [`texture_taker`]) is brought up to date
/// over title memory `[from, to)`, or reads it in place.
pub(crate) fn asked(taker: u64, kind: Taker, from: usize, to: usize) {
    if !tracking() || from >= to {
        return;
    }
    match OWNERS.load(Ordering::Relaxed) {
        0 => return,
        1 if SOLE_OWNER.load(Ordering::Relaxed) == taker => return,
        _ => {}
    }
    let took = {
        let mut p = pages();
        let took = p.took(taker, from, to);
        if took.stale > 0 {
            noted(&p);
        }
        took
    };
    if took.bytes == 0 {
        return;
    }
    let i = match kind {
        Taker::Copy => n::TAKEN_COPY,
        Taker::Window => n::TAKEN_WINDOW,
        Taker::InPlace => n::TAKEN_IN_PLACE,
        Taker::Texture => n::TAKEN_TEXTURE,
    };
    count(i, took.bytes as u64);
    count(n::TAKEN_PAGES, took.pages as u64);
    if took.stale > 0 {
        count(n::STALE_TAKEN, took.stale as u64);
        let times = COUNTS[n::STALE_TAKEN_PAGES].fetch_add(1, Ordering::Relaxed) + 1;
        if times.is_power_of_two() {
            tracing::warn!(
                taker = format_args!("{taker:#x}"),
                ?kind,
                at = format_args!("{from:#x}..{to:#x}"),
                bytes = took.stale,
                "VICTORY_SKIP_WC_BACK: another copy took bytes left on the card from title memory (older ones); {times} times so \
                 far. The page comes back with its next flight and is not left again"
            );
        }
    }
}

/// Copy `id`, in the card's memory, took title memory `[from, to)` into its
/// CPU side again (and up to the card).
pub(crate) fn recopied(id: u64, from: usize, to: usize) {
    // Counted only for the census: a refresh's lock less with the switch alone.
    if !on() || from >= to || OWNERS.load(Ordering::Relaxed) == 0 {
        return;
    }
    count(n::ECHO, pages().recopied(id, from, to) as u64);
}

/// Counters of one copy in the card's memory, since the start.
#[derive(Default, Clone, Copy)]
struct Row {
    start: u64,
    len: u64,
    window: bool,
    sent: u64,
    sent_ranges: u64,
    compared: u64,
    fetched: u64,
    fetched_pieces: u64,
    written: u64,
}

static ROWS: Mutex<BTreeMap<u64, Row>> = Mutex::new(BTreeMap::new());

fn row(which: Which, f: impl FnOnce(&mut Row)) {
    let mut rows = ROWS.lock().unwrap_or_else(|e| e.into_inner());
    let r = rows.entry(which.id).or_default();
    (r.start, r.len, r.window) = (which.start, which.len, which.window);
    f(r);
}

/// `bytes` of copy `which` were sent up to the card in `ranges` ranges.
pub(crate) fn sent(which: Which, bytes: u64, ranges: u64) {
    if !on() {
        return;
    }
    count(n::SENT, bytes);
    row(which, |r| {
        r.sent += bytes;
        r.sent_ranges += ranges;
    });
    report_if_due();
}

/// `pieces` 64 KiB pieces of copy `which` were compared on the card for the
/// declared title range `declared`.
pub(crate) fn compared(which: Which, declared: (usize, usize), pieces: u64) {
    if !on() {
        return;
    }
    count(n::COMPARED, pieces);
    if all_wc(&[declared]) {
        count(n::COMPARED_WC, pieces);
    }
    row(which, |r| r.compared += pieces);
}

/// Offsets `[lo, hi)` of copy `which` came back from the card and its
/// merge stored `written` bytes of them into title memory. For a piece
/// taken from an in-flight slot, `[lo, hi)` is what of the slot is merged
/// (bytes a newer fetch brought are left out). `check`: in a check landing
/// ([`check_landing`]).
pub(crate) fn merged(which: Which, lo: usize, hi: usize, written: usize, check: bool) {
    if !tracking() || lo >= hi {
        return;
    }
    let (a, b) = (which.start as usize + lo, which.start as usize + hi);
    let bytes = (hi - lo) as u64;
    count(n::FETCHED, bytes);
    count(n::WRITTEN, written as u64);
    if on() {
        row(which, |r| {
            r.fetched += bytes;
            r.fetched_pieces += 1;
            r.written += written as u64;
        });
    }
    if all_wc(&[(a, b)]) {
        count(n::FETCHED_WC, bytes);
        count(n::WRITTEN_WC, written as u64);
        let multi = crate::inplace::view_census(a, b - a).1 > 0;
        let page = a / PAGE;
        // Asked before the pages' lock, as in `skip_back`.
        let aliases = if multi && !which.window && (on() || hold()) { aliases(page) } else { Vec::new() };
        let mut p = pages();
        // What the rules of `skip_wc_back` would leave of the bytes fetched.
        let mut would = false;
        if which.window {
            count(n::WC_WINDOW, bytes);
        } else if multi {
            count(n::WC_MULTI, bytes);
            if !aliases.is_empty() && p.may_hold(Skip::Untaken, which.id, page, &aliases) {
                count(n::WC_MULTI_HOLDABLE, bytes);
            }
            if hold() {
                would = p.may_hold(skip_wc_back(), which.id, page, &aliases);
            }
        } else {
            if p.may_skip(Skip::Alone, which.id, page) {
                count(n::WC_ALONE, bytes);
            }
            if p.may_skip(Skip::Untaken, which.id, page) {
                count(n::WC_UNTAKEN, bytes);
            }
            would = p.may_skip(skip_wc_back(), which.id, page);
        }
        let (two, back) = p.merged(which.id, a, b, written > 0);
        noted(&p);
        drop(p);
        if two {
            count(n::TWO_WRITERS, 1);
        }
        if back {
            count(n::BROUGHT_BACK, written as u64);
        }
        if check && (would || back) {
            count(n::CHECK_FETCHED, bytes);
            count(n::CHECK_WRITTEN, written as u64);
            if back {
                count(n::CHECK_STALE, written as u64);
            }
        }
    }
    if on() {
        report_if_due();
    }
}

/// A copy in the PC's memory: shaders' declared title range `[from, to)`
/// was compared and `written` bytes went into title memory.
pub(crate) fn pc_back(from: usize, to: usize, written: usize) {
    if !on() || from >= to {
        return;
    }
    count(n::PC_COMPARED, (to - from) as u64);
    count(n::PC_WRITTEN, written as u64);
    if all_wc(&[(from, to)]) {
        count(n::PC_COMPARED_WC, (to - from) as u64);
        count(n::PC_WRITTEN_WC, written as u64);
    }
}

/// Host resources made, by where: count and bytes ([`made`]).
const KINDS: [&str; 9] =
    ["read-back", "upload", "shared", "uav", "texture", "reserved texture", "depth twin", "placed in title memory", "host list"];
static MADE: [AtomicU64; 18] = [const { AtomicU64::new(0) }; 18];

/// With the census on: a host resource of kind `kind` ([`KINDS`]) and
/// `bytes` (0 when not known) was asked of the device.
pub(crate) fn made(kind: usize, bytes: u64) {
    if on() && kind < KINDS.len() {
        MADE[kind * 2].fetch_add(1, Ordering::Relaxed);
        MADE[kind * 2 + 1].fetch_add(bytes, Ordering::Relaxed);
    }
}

/// The console protection bits the runtime drops, spelled out.
fn protect_words(p: u32) -> String {
    let mut w = Vec::new();
    if p & WRITE_COMBINE != 0 {
        w.push("WC");
    }
    if p & 0x200 != 0 {
        w.push("uncached");
    }
    if p & 0x100 != 0 {
        w.push("guard");
    }
    let gpu = p & 0x7_f800;
    let mut s = w.join("+");
    if gpu != 0 {
        if !s.is_empty() {
            s.push('+');
        }
        s.push_str(&format!("gpu{gpu:#x}"));
    }
    if s.is_empty() { "-".to_owned() } else { s }
}

/// The last report: when, at which frame, and the counters then.
struct Last {
    at: Instant,
    frame: u64,
    counts: [u64; n::COUNT],
    made: [u64; 18],
    rows: BTreeMap<u64, Row>,
}

static LAST: Mutex<Option<Last>> = Mutex::new(None);

fn report_if_due() {
    let frame = crate::frame::current();
    let now = Instant::now();
    let Ok(mut last) = LAST.try_lock() else {
        return;
    };
    let counts: [u64; n::COUNT] = std::array::from_fn(|i| COUNTS[i].load(Ordering::Relaxed));
    let made: [u64; 18] = std::array::from_fn(|i| MADE[i].load(Ordering::Relaxed));
    let Some(was) = last.as_mut() else {
        let rows = ROWS.lock().unwrap_or_else(|e| e.into_inner()).clone();
        *last = Some(Last { at: now, frame, counts, made, rows });
        return;
    };
    if now.duration_since(was.at) < period() {
        return;
    }
    let rows = ROWS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let frames = frame.saturating_sub(was.frame).max(1) as f64;
    let secs = now.duration_since(was.at).as_secs_f64();
    let d: [f64; n::COUNT] = std::array::from_fn(|i| counts[i].saturating_sub(was.counts[i]) as f64 / frames);
    let mb = |b: f64| b / 1e6;
    let (taken_since_start, covered_pages, noted) = {
        let p = pages();
        (p.taken.len(), p.cover.len(), p.written.len())
    };
    tracing::info!(
        target: "census",
        "census over {secs:.1} s, {frames:.0} frames, copies in the card's memory, per frame: {:.1} MB sent up, {:.0} pieces compared \
         ({:.0} of write-combining memory), {:.1} MB fetched back ({:.1} MB WC), {:.1} MB of it written into title memory ({:.1} MB WC)",
        mb(d[n::SENT]),
        d[n::COMPARED],
        d[n::COMPARED_WC],
        mb(d[n::FETCHED]),
        mb(d[n::FETCHED_WC]),
        mb(d[n::WRITTEN]),
        mb(d[n::WRITTEN_WC]),
    );
    tracing::info!(
        target: "census",
        "census of the WC bytes fetched, per frame: of title buffers' copies, at one address, {:.1} MB in pages no other live copy \
         covers (what VICTORY_SKIP_WC_BACK=1 leaves on the card), {:.1} MB in pages no other copy took after a card write so far \
         (=2); {:.1} MB in pages shown at more than one address ({:.1} MB of it holdable by =2 + HOLD=1), {:.1} MB \
         of windows (never left). Taken again after their \
         write-back: by the copy that wrote them (the upload echo) {:.1} MB, by other copies {:.2} MB, windows {:.2} MB, copies in \
         place {:.2} MB, textures {:.2} MB; {:.1} pages a frame taken by another for the first time, {taken_since_start} pages \
         since the start, {:.2} pages a frame written back through two copies in turn ({noted} WC pages noted, {covered_pages} \
         pages under copies). The title's CPU reads are not seen",
        mb(d[n::WC_ALONE]),
        mb(d[n::WC_UNTAKEN]),
        mb(d[n::WC_MULTI]),
        mb(d[n::WC_MULTI_HOLDABLE]),
        mb(d[n::WC_WINDOW]),
        mb(d[n::ECHO]),
        mb(d[n::TAKEN_COPY]),
        mb(d[n::TAKEN_WINDOW]),
        mb(d[n::TAKEN_IN_PLACE]),
        mb(d[n::TAKEN_TEXTURE]),
        d[n::TAKEN_PAGES],
        d[n::TWO_WRITERS],
    );
    tracing::info!(
        target: "census",
        "census of copies in the PC's memory, per frame: {:.1} MB of declared ranges compared ({:.1} MB WC), {:.1} MB written into \
         title memory ({:.1} MB WC)",
        mb(d[n::PC_COMPARED]),
        mb(d[n::PC_COMPARED_WC]),
        mb(d[n::PC_WRITTEN]),
        mb(d[n::PC_WRITTEN_WC]),
    );
    if skip_wc_back() != Skip::Off {
        tracing::info!(
            target: "census",
            "census of VICTORY_SKIP_WC_BACK={:?}, per frame: {:.1} pieces ({:.1} MB) left on the card; {} pages hold bytes title \
             memory lacks now; other copies took {:.3} MB of such bytes ({} times since the start); {:.3} MB a frame brought back \
             by later fetches; the title wrote such pages again {:.2} times a frame ({:.3} MB of its bytes went up there, the \
             card's kept beside them); {} pages lost with their copy since the start. Check landings (VICTORY_SKIP_WC_BACK_CHECK={}, \
             0: none): {:.2} MB a frame fetched of pieces the rules leave, {:.3} MB written into title memory, {:.3} MB of it left by \
             earlier landings",
            skip_wc_back(),
            d[n::LEFT_PIECES],
            mb(d[n::LEFT]),
            STALE.load(Ordering::Relaxed),
            mb(d[n::STALE_TAKEN]),
            counts[n::STALE_TAKEN_PAGES],
            mb(d[n::BROUGHT_BACK]),
            d[n::TITLE_OVER],
            mb(d[n::TITLE_OVER_UP]),
            counts[n::LOST],
            check_every(),
            mb(d[n::CHECK_FETCHED]),
            mb(d[n::CHECK_WRITTEN]),
            mb(d[n::CHECK_STALE]),
        );
    }
    if hold() {
        tracing::info!(
            target: "census",
            "census of VICTORY_SKIP_WC_BACK_HOLD=1, per frame: {:.1} pieces ({:.1} MB) held on the card through the lazy \
             copy-back, {:.1} MB of them in pages shown at more than one address; {:.1} pieces the rules allowed were fetched, \
             not holdable (see the lazy copy-back line for why and for what landed)",
            d[n::HELD_PIECES],
            mb(d[n::HELD]),
            mb(d[n::HELD_MULTI]),
            d[n::HOLD_REFUSED],
        );
    }
    {
        let parts: Vec<String> = KINDS
            .iter()
            .enumerate()
            .map(|(k, name)| {
                let count = made[k * 2].saturating_sub(was.made[k * 2]) as f64 / frames;
                let bytes = made[k * 2 + 1].saturating_sub(was.made[k * 2 + 1]) as f64 / frames;
                format!("{name} {count:.1} ({:.1} MB)", mb(bytes))
            })
            .collect();
        let all: u64 = (0..KINDS.len()).map(|k| made[k * 2].saturating_sub(was.made[k * 2])).sum();
        tracing::info!(target: "census", "census resources made per frame: {:.1} in all; {}", all as f64 / frames, parts.join(", "));
    }
    {
        let reservations = reservations();
        let mut lines: Vec<(f64, String)> = rows
            .iter()
            .map(|(id, now_)| {
                let was_ = was.rows.get(id).copied().unwrap_or_default();
                let per = |a: u64, b: u64| a.saturating_sub(b) as f64 / frames;
                let (sent, fetched) = (per(now_.sent, was_.sent), per(now_.fetched, was_.fetched));
                let start = now_.start as usize;
                let res = reservations.range(..=start).next_back().filter(|(base, r)| *base + r.len > start);
                let (res, words) = match res {
                    Some((base, r)) => (
                        format!("{base:#x}+{}MB kind {:#x} flags {:#x} protect {:#x}", r.len >> 20, r.kind, r.flags, r.protect),
                        protect_words(r.protect),
                    ),
                    None => ("not in a recorded reservation".to_owned(), "?".to_owned()),
                };
                let line = format!(
                    "census copy #{id} {start:#x}+{:.1}MB{}: per frame sent {:.2} MB in {:.1} ranges, compared {:.0} pieces, fetched \
                     {:.2} MB in {:.1} pieces, written {:.2} MB; {words} [{res}]",
                    mb(now_.len as f64),
                    if now_.window { " (window)" } else { "" },
                    mb(sent),
                    per(now_.sent_ranges, was_.sent_ranges),
                    per(now_.compared, was_.compared),
                    mb(fetched),
                    per(now_.fetched_pieces, was_.fetched_pieces),
                    mb(per(now_.written, was_.written)),
                );
                (sent + fetched, line)
            })
            .filter(|(traffic, _)| *traffic > 0.0)
            .collect();
        drop(reservations);
        lines.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (_, line) in lines.iter().take(24) {
            tracing::info!(target: "census", "{line}");
        }
    }
    *was = Last { at: now, frame, counts, made, rows };
}

#[cfg(test)]
mod tests {
    use super::{
        Back, PAGE, Pages, Reservation, Skip, Took, Which, all_wc_in, check_due, check_landing, on, refetch_due, skip_back, skip_wc_back,
    };
    use std::collections::BTreeMap;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn a_page_is_left_only_where_the_rules_allow() {
        let mut p = Pages::default();
        let (alone, shared, taken, other) = (1, 2, 3, 4);
        p.cover(alone * PAGE, PAGE, true);
        p.cover(shared * PAGE, PAGE, true);
        p.cover(shared * PAGE, 2 * PAGE, true);
        p.cover(taken * PAGE, PAGE, true);
        p.cover(other * PAGE, PAGE, true);
        // Taken by another copy after a write-back of copy 1.
        p.wrote(1, taken * PAGE, taken * PAGE + 8);
        p.took(9, taken * PAGE, taken * PAGE + PAGE);
        // Written back by copy 2 before.
        p.wrote(2, other * PAGE, other * PAGE + 8);
        for mode in [Skip::Alone, Skip::Untaken] {
            assert!(p.may_skip(mode, 1, alone));
            assert!(!p.may_skip(mode, 1, taken));
            assert!(!p.may_skip(mode, 1, other), "another copy wrote it back");
            assert!(p.may_skip(mode, 2, other));
        }
        assert!(!p.may_skip(Skip::Alone, 1, shared), "another copy covers it");
        assert!(p.may_skip(Skip::Untaken, 1, shared));
        assert!(!p.may_skip(Skip::Off, 1, alone));
    }

    #[test]
    fn a_stale_page_another_copy_takes_is_counted_fetched_again_and_never_left_again() {
        let mut p = Pages::default();
        let page = 5 * PAGE;
        p.left(1, page + 100, page + 200);
        p.left(1, page, page + 50);
        assert_eq!(p.stale, 1);
        assert_eq!(p.stale_in(1, page, page + PAGE), vec![5]);
        assert!(p.stale_in(2, page, page + PAGE).is_empty());
        // The owner asking takes nothing; another copy takes old bytes.
        assert_eq!(p.took(1, page, page + PAGE), Took::default());
        assert_eq!(p.took(2, page + 40, page + 120), Took { bytes: 80, pages: 1, stale: 80 });
        assert!(p.refetch.contains(&(1, 5)));
        assert!(!p.may_skip(Skip::Untaken, 1, 5));
        // A piece that does not hold all that was left brings back part of it.
        assert_eq!(p.merged(1, page + 100, page + 200, true), (false, false));
        assert_eq!(p.stale, 1);
        // The whole piece: all of it.
        assert_eq!(p.merged(1, page, page + PAGE, true), (false, true));
        assert_eq!((p.stale, p.refetch.len()), (0, 0));
        assert!(p.stale_in(1, page, page + PAGE).is_empty());
    }

    #[test]
    fn a_write_back_through_another_copy_keeps_the_first_copys_stale_bytes_noted() {
        let mut p = Pages::default();
        let page = 6 * PAGE;
        p.left(1, page, page + 64);
        assert!(p.wrote(2, page + 32, page + 96));
        assert_eq!(p.stale_in(1, page, page + PAGE), vec![6]);
        assert!(p.taken.contains(&6));
    }

    #[test]
    fn a_copy_that_goes_loses_what_it_left_and_the_pages_are_never_left_again() {
        let mut p = Pages::default();
        p.left(1, 2 * PAGE, 2 * PAGE + 8);
        p.left(1, 3 * PAGE, 3 * PAGE + 8);
        p.left(2, 4 * PAGE, 4 * PAGE + 8);
        p.took(7, 3 * PAGE, 4 * PAGE);
        assert_eq!(p.gone(1), 2);
        assert_eq!(p.gone(1), 0);
        assert_eq!(p.stale, 1);
        assert!(p.refetch.iter().all(|r| r.0 != 1));
        assert!(!p.may_skip(Skip::Untaken, 1, 2));
        assert!(p.may_skip(Skip::Untaken, 2, 4));
    }

    #[test]
    fn one_check_landing_per_n_frames() {
        let last = AtomicU64::new(0);
        assert!(check_due(&last, 0, 30));
        assert!(!check_due(&last, 0, 30));
        assert!(!check_due(&last, 29, 30));
        assert!(check_due(&last, 30, 30));
        assert!(!check_due(&last, 45, 30));
        // Frames skipped over: one check, not one per period missed.
        assert!(check_due(&last, 200, 30));
        assert!(!check_due(&last, 209, 30));
    }

    fn reservation(len: usize, protect: u32) -> Reservation {
        Reservation { len, kind: 0, flags: 0, protect }
    }

    // iGoldenNinja's (fork `f9c43fa`).
    #[test]
    fn a_write_in_the_ordinary_half_of_a_card_window_still_comes_back() {
        const MIB: usize = 1 << 20;
        let window = 8 * MIB;
        let middle = window + 2 * MIB;
        let map = BTreeMap::from([(window, reservation(2 * MIB, 0x404)), (middle, reservation(2 * MIB, 0x04))]);
        let wc = (middle - 65536, middle);
        let ordinary = (middle, middle + 65536);
        assert!(all_wc_in(&map, &[wc]));
        assert!(!all_wc_in(&map, &[ordinary]));
        assert!(!all_wc_in(&map, &[wc, ordinary]));
        assert!(!all_wc_in(&map, &[(wc.0, ordinary.1)]));
    }

    // iGoldenNinja's (fork `f9c43fa`).
    #[test]
    fn a_declaration_must_fit_one_known_reservation() {
        let map = BTreeMap::from([
            (0x10000, reservation(0x10000, 0x404)),
            (0x20000, reservation(0x10000, 0x404)),
            (usize::MAX - 8, reservation(16, 0x404)),
        ]);
        assert!(all_wc_in(&map, &[(0x10000, 0x20000), (0x20000, 0x30000)]));
        assert!(!all_wc_in(&map, &[(0x1ffff, 0x20001)]));
        assert!(!all_wc_in(&map, &[(0x30000, 0x30001)]));
        assert!(!all_wc_in(&map, &[(0x10000, 0x10000)]));
        assert!(!all_wc_in(&map, &[(usize::MAX - 8, 1)]));
        assert!(!all_wc_in(&map, &[(usize::MAX - 8, usize::MAX)]));
    }

    #[test]
    fn another_copy_takes_a_written_page_once_per_write_and_the_owner_never() {
        let mut p = Pages::default();
        let page = 7 * PAGE;
        assert!(!p.wrote(1, page + 16, page + 64));
        // The owner asking is no other taker.
        assert_eq!(p.took(1, page, page + PAGE), Took::default());
        // Another copy over part of the written bytes: those bytes, once.
        assert_eq!(p.took(2, page + 32, page + PAGE), Took { bytes: 32, pages: 1, stale: 0 });
        assert_eq!(p.took(2, page, page + PAGE), Took::default());
        // A third taker counts its bytes, not the page again.
        assert_eq!(p.took(3, page, page + PAGE), Took { bytes: 48, pages: 0, stale: 0 });
        // Asks beside the written bytes take nothing.
        assert_eq!(p.took(4, page + 64, page + PAGE), Took::default());
        assert!(p.taken.contains(&7));
        // A new write: taken again.
        assert!(!p.wrote(1, page, page + 8));
        assert_eq!(p.took(2, page, page + PAGE), Took { bytes: 8, pages: 1, stale: 0 });
    }

    #[test]
    fn writes_nobody_took_yet_add_up_and_a_second_writer_is_told() {
        let mut p = Pages::default();
        let page = 3 * PAGE;
        p.wrote(1, page, page + 8);
        p.wrote(1, page + 100, page + 108);
        assert_eq!(p.took(2, page, page + PAGE).bytes, 108);
        assert!(p.wrote(5, page + 4, page + 12));
        assert_eq!(p.owners.get(&1), None);
        assert_eq!(p.owners.get(&5), Some(&1));
        // The first owner is now another taker.
        assert_eq!(p.took(1, page, page + PAGE).bytes, 8);
    }

    #[test]
    fn the_echo_is_the_owners_page_taken_again_once_per_write() {
        let mut p = Pages::default();
        let page = 9 * PAGE;
        p.wrote(1, page + 10, page + 20);
        assert_eq!(p.recopied(2, page, page + PAGE), 0);
        assert_eq!(p.recopied(1, page, page + PAGE), PAGE);
        assert_eq!(p.recopied(1, page, page + PAGE), 0);
        p.wrote(1, page + 10, page + 20);
        // Part of the page taken: that part goes up again.
        assert_eq!(p.recopied(1, page + PAGE / 2, page + PAGE), PAGE / 2);
    }

    /// `VICTORY_SKIP_WC_BACK_HOLD`: a page the title shows at two addresses
    /// is held only if the rules hold at both; with `1` no copy may cover
    /// the other address, the writer's own included.
    #[test]
    fn a_page_at_two_addresses_is_held_only_where_both_allow() {
        let (page, alias) = (11, 17);
        let mut p = Pages::default();
        p.cover(page * PAGE, PAGE, true);
        for mode in [Skip::Alone, Skip::Untaken] {
            assert!(p.may_hold(mode, 1, page, &[alias]));
            assert!(p.may_hold(mode, 1, page, &[]), "at one address: as may_skip");
        }
        assert!(!p.may_hold(Skip::Off, 1, page, &[alias]));
        // Another copy at the other address.
        p.cover(alias * PAGE, PAGE, true);
        assert!(!p.may_hold(Skip::Alone, 1, page, &[alias]), "another copy covers the other address");
        assert!(p.may_hold(Skip::Untaken, 1, page, &[alias]));
        p.cover(alias * PAGE, PAGE, false);
        // The writer's own copy over both addresses.
        p.cover(page * PAGE, (alias - page + 1) * PAGE, true);
        assert!(!p.may_hold(Skip::Alone, 1, page, &[alias]), "its bytes there come from title memory");
        p.cover(page * PAGE, (alias - page + 1) * PAGE, false);
        assert!(p.may_hold(Skip::Alone, 1, page, &[alias]));
        // Another copy took a card write at the other address.
        let mut q = Pages::default();
        q.wrote(1, alias * PAGE, alias * PAGE + 8);
        assert!(q.may_hold(Skip::Untaken, 1, page, &[alias]), "the writer's own write-back there");
        q.took(9, alias * PAGE, alias * PAGE + PAGE);
        assert!(!q.may_hold(Skip::Untaken, 1, page, &[alias]));
        assert!(q.may_skip(Skip::Untaken, 1, page), "at its own address alone it would be left");
        // Another copy wrote the other address back.
        let mut r = Pages::default();
        r.wrote(2, alias * PAGE, alias * PAGE + 8);
        assert!(!r.may_hold(Skip::Untaken, 1, page, &[alias]));
        assert!(r.may_hold(Skip::Untaken, 2, page, &[alias]));
    }

    #[test]
    fn copies_cover_their_pages_until_they_go() {
        let mut p = Pages::default();
        p.cover(PAGE - 1, 2, true);
        p.cover(PAGE, 4 * PAGE, true);
        assert_eq!((p.covered(0), p.covered(1), p.covered(4), p.covered(5)), (1, 2, 1, 0));
        p.cover(PAGE - 1, 2, false);
        assert_eq!((p.covered(0), p.covered(1)), (0, 1));
        assert_eq!(Pages::pages(5, 5), 0..0);
    }

    /// Unset, or any value but 1 and 2: a piece is fetched, as before.
    #[test]
    fn nothing_is_left_on_the_card_unless_the_switch_asks() {
        let asked = matches!(std::env::var("VICTORY_SKIP_WC_BACK").as_deref().map(str::trim), Ok("1") | Ok("2"));
        assert_eq!(skip_wc_back() != Skip::Off, asked);
        let census = std::env::var_os("VICTORY_CENSUS").is_some_and(|v| v == "1");
        assert_eq!(on(), census);
        if asked {
            return;
        }
        assert_eq!(super::check_every(), 0);
        assert!(!check_landing());
        assert!(refetch_due(false).is_none());
        let which = Which { id: 1, start: 0, len: PAGE as u64, window: false };
        assert!(!super::hold());
        if std::env::var_os("VICTORY_SKIP_WC_MIN_KB").is_none() {
            assert_eq!(super::skip_wc_min_bytes(), 0, "every size, unless asked");
        }
        assert_eq!(skip_back(which, (0, 64), 0, 64, false), Back::Fetch);
        assert_eq!(skip_back(Which { id: 1, start: 0, len: PAGE as u64, window: true }, (0, 64), 0, 64, false), Back::Fetch);
    }
}
