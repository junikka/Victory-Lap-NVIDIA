//! Residency by use (ticket 063): which waiting textures the texture pass
//! serves first, and which of a texture's levels take the card's memory.
//!
//! The title fills a table of texture views that shaders pick by an index
//! they compute. Every slot it writes is queued (`descriptor/host.rs`
//! `TextureQueue`) and served the newest write first, whether or not
//! anything samples it; and a host copy either has room on the card for
//! every level the title holds, or for none. Four switches, all off unless
//! set, change that:
//!
//! - `VICTORY_SERVE_BY_USE=1` (`alt`: 20 s off, 20 s on): the due slots are
//!   served by what is known of their use ([`order`]): slots draws reach
//!   through a table first, then slots that show a texture now (it is on
//!   the card: the title raises or changes something it shows), then the
//!   rest; every eighth one is the oldest waiting, so that none starves.
//!   Which slot a shader picks by index is known to the card alone: these
//!   are the signs the layer has.
//! - `VICTORY_LEVELS_PAST_POOL=1`: with `VICTORY_LEVELS_ON_DEMAND=1`, a
//!   texture made level by level is a reserved one and not a piece of a
//!   block. (`resource.rs` `host_texture_as` asked the pool first: no
//!   texture was ever made level by level while the pool was on, and each
//!   try took a piece of the texture's full size and gave it back.)
//! - `VICTORY_LEVELS_BY_ROOM=1` (`alt`): small levels before large
//!   ([`levels_allowed`]). With room every held level gets its tiles, as
//!   before; without room the small ones still do, so a new texture shows
//!   its small levels instead of nothing (the texture pass makes such a
//!   texture although the card has no room: `draw.rs` `heap_textures`);
//!   over the card's line ([`is_over`]) none does. Every texture of a
//!   megabyte or more that could be made level by level is. Sets the
//!   switch above.
//! - `VICTORY_LEVELS_DROP=1` (`alt`): while the card is over its line the
//!   largest levels with tiles are hidden, and their tiles given back once
//!   nothing can read them (`tiles.rs` `Tiled::hide_largest`, `resource.rs`
//!   `drop_levels`): a texture loses a level, not its host copy. Sets the
//!   second switch too.
//!
//! For the last two the card's memory is counted without what the tile
//! heaps hold free ([`effective`]), and no tile heap is made while the
//! card itself is over its line (`tiles.rs` `take`).
//!
//! shortcut: a level's tiles go back to their heap, and a heap to the card
//! only when all of it is free. What is dropped is room for the next
//! levels at once; the card's own count falls as heaps empty.
//!
//! [`log_line`] says what they did, with the residency line.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};

const OFF: u8 = 0;
const ON: u8 = 1;
const ALT: u8 = 2;

/// What a switch's value asks for: `1` on, `alt` in turn, anything else off.
fn mode_of(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        Some("1") => ON,
        Some("alt") => ALT,
        _ => OFF,
    }
}

/// Whether a switch in turn is on `secs` seconds after the start: off for
/// 20 s, on for 20 s.
fn turn(secs: u64) -> bool {
    (secs / 20) % 2 == 1
}

/// A switch that is off, on, or on and off in turn.
struct Switch {
    name: &'static str,
    what: &'static str,
    mode: OnceLock<u8>,
    /// What it was when last asked, in turn (2: not asked yet).
    last: AtomicU8,
}

impl Switch {
    const fn new(name: &'static str, what: &'static str) -> Switch {
        Switch { name, what, mode: OnceLock::new(), last: AtomicU8::new(2) }
    }

    fn mode(&self) -> u8 {
        *self.mode.get_or_init(|| {
            let mode = mode_of(std::env::var(self.name).ok().as_deref());
            if mode != OFF {
                tracing::info!("{} ({}={})", self.what, self.name, if mode == ALT { "alt: 20 s off, 20 s on" } else { "1" });
            }
            mode
        })
    }

    /// Whether it is set at all (on, or in turn).
    fn set(&self) -> bool {
        self.mode() != OFF
    }

    fn on(&self) -> bool {
        match self.mode() {
            OFF => false,
            ON => true,
            _ => {
                let on = turn(crate::host::coarse_ms() / 1000);
                if self.last.swap(u8::from(on), Relaxed) != u8::from(on) {
                    tracing::info!("{} from now ({}=alt): {}", self.what, self.name, if on { "ON" } else { "off" });
                }
                on
            }
        }
    }
}

static BY_USE: Switch = Switch::new("VICTORY_SERVE_BY_USE", "textures picked by index are served by what is known of their use");
static BY_ROOM: Switch =
    Switch::new("VICTORY_LEVELS_BY_ROOM", "a texture's small levels get the card's memory before its large ones");
static DROP: Switch =
    Switch::new("VICTORY_LEVELS_DROP", "the largest levels give their memory back while the card is over its line");

/// Whether the due slots are served by use now.
pub(crate) fn by_use() -> bool {
    BY_USE.on()
}

/// Whether the due slots are counted by use (the switch is set: in turn,
/// both turns count).
pub(crate) fn by_use_set() -> bool {
    BY_USE.set()
}

/// Whether levels get tiles by the card's room now ([`levels_allowed`]).
pub(crate) fn by_room() -> bool {
    BY_ROOM.on()
}

/// Whether every texture that can be is made level by level (the switch is
/// set: how a texture is made is decided once).
pub(crate) fn by_room_set() -> bool {
    BY_ROOM.set()
}

/// Whether the largest levels are hidden now while the card is over its line.
pub(crate) fn drops() -> bool {
    DROP.on()
}

/// Whether a texture made level by level passes the pool by
/// (`VICTORY_LEVELS_PAST_POOL=1`, or one of the switches that need it).
pub(crate) fn past_pool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var_os("VICTORY_LEVELS_PAST_POOL").is_some_and(|v| v == "1");
        let on = asked || BY_ROOM.set() || DROP.set();
        if on {
            tracing::info!(
                "textures made level by level are reserved ones, not pieces of blocks (VICTORY_LEVELS_PAST_POOL=1; \
                 it takes VICTORY_LEVELS_ON_DEMAND=1 to make any)"
            );
        }
        on
    })
}

/// A waiting slot's use, the surest first: draws reach it through a table
/// (lately); its host slot shows a texture now; nothing is known.
pub(crate) const READ: u8 = 0;
pub(crate) const SHOWN: u8 = 1;
pub(crate) const UNKNOWN: u8 = 2;

/// One in so many slots handed out is the oldest waiting, whatever its use.
const AGED_EVERY: usize = 8;

/// The order to serve `fresh` in (slot and write number, the newest write
/// first), `class[i]` being the use of `fresh[i]`: by use, the newest first
/// within one, and every [`AGED_EVERY`]th the oldest not handed out yet. At
/// most `max`.
pub(crate) fn order(fresh: &[(u32, u64)], class: &[u8], max: usize) -> Vec<(u32, u64)> {
    let most = max.min(fresh.len());
    let mut ranked: Vec<usize> = Vec::with_capacity(fresh.len());
    for wanted in [READ, SHOWN] {
        ranked.extend((0..fresh.len()).filter(|&i| class.get(i) == Some(&wanted)));
    }
    ranked.extend((0..fresh.len()).filter(|&i| !matches!(class.get(i), Some(&READ) | Some(&SHOWN))));
    let mut taken = vec![false; fresh.len()];
    let mut out = Vec::with_capacity(most);
    let (mut next, mut oldest, mut since_aged) = (0, fresh.len(), 0);
    while out.len() < most {
        if since_aged + 1 == AGED_EVERY {
            since_aged = 0;
            while oldest > 0 && taken[oldest - 1] {
                oldest -= 1;
            }
            if oldest > 0 {
                oldest -= 1;
                taken[oldest] = true;
                out.push(fresh[oldest]);
            }
            continue;
        }
        while next < ranked.len() && taken[ranked[next]] {
            next += 1;
        }
        let Some(&i) = ranked.get(next) else {
            break;
        };
        taken[i] = true;
        out.push(fresh[i]);
        next += 1;
        since_aged += 1;
    }
    out
}

/// The share of the card's budget, in percent, from which the card is over
/// its line: nothing new takes its memory, and levels are dropped.
const OVER_PERCENT: u128 = 97;

/// Whether `used` bytes of a budget of `budget` are over the card's line.
pub(crate) fn is_over(budget: u64, used: u64) -> bool {
    budget != 0 && u128::from(used) * 100 >= u128::from(budget) * OVER_PERCENT
}

/// How far over its line the card is, in bytes (0: it is not).
pub(crate) fn over_by(budget: u64, used: u64) -> u64 {
    let line = (u128::from(budget) * OVER_PERCENT / 100) as u64;
    if is_over(budget, used) { used - line.min(used) } else { 0 }
}

/// [`is_over`] of the card now (not known: not over).
pub(crate) fn over(memory: Option<[u64; 4]>) -> bool {
    memory.is_some_and(|[budget, used, ..]| is_over(budget, used))
}

/// The card's memory as the levels count it: what is in use less what the
/// tile heaps hold free (`crate::tiles::free_bytes`), which a level takes
/// without taking more of the card. Levels dropped ([`drops`]) free tiles
/// before they free the card's memory (a heap goes back to the card only
/// when every tile of it is free): by the card's own count nothing would
/// have been gained, and the next look would drop more.
pub(crate) fn effective(memory: Option<[u64; 4]>) -> Option<[u64; 4]> {
    memory.map(|[budget, used, a, b]| [budget, used.saturating_sub(crate::tiles::free_bytes()), a, b])
}

/// The bytes of levels to hide now: what the card is over its line by
/// ([`over_by`]; `used` as [`effective`] counts it) less the `pending`
/// bytes of levels hidden whose tiles are still to go back, and, if that
/// is anything, a hundredth of the budget more, so that the card is not
/// back at its line with the next texture.
pub(crate) fn to_drop(budget: u64, used: u64, pending: u64) -> u64 {
    match over_by(budget, used).saturating_sub(pending) {
        0 => 0,
        over => over + budget / 100,
    }
}

/// [`over`] of the card as the levels count it ([`effective`]), as it was
/// at most a tenth of a second ago, for a question asked of many slots a
/// pass.
pub(crate) fn over_lately() -> bool {
    // The tenth of a second of the last asking, counted from 1, and its
    // answer in the lowest bit.
    static LAST: AtomicU64 = AtomicU64::new(0);
    let tenth = crate::host::coarse_ms() / 100 + 1;
    let last = LAST.load(Relaxed);
    if last >> 1 == tenth {
        return last & 1 != 0;
    }
    let over = over(effective(crate::host::host().and_then(|h| h.video_memory())));
    LAST.store((tenth << 1) | u64::from(over), Relaxed);
    over
}

/// The most tiles (of 64 KiB) of a level that still counts as small
/// (`VICTORY_LEVELS_SMALL_TILES`, 4: 256 KiB, a 512x512 level of one byte a
/// texel).
pub(crate) fn small_tiles() -> u32 {
    static TILES: OnceLock<u32> = OnceLock::new();
    *TILES.get_or_init(|| {
        std::env::var("VICTORY_LEVELS_SMALL_TILES").ok().and_then(|v| v.trim().parse::<u32>().ok()).map_or(4, |n| n.min(1024))
    })
}

/// The levels of a texture that may take tiles now, a bit a level (level 0
/// the lowest bit). `tiles`: the tiles of each level that has tiles of its
/// own, the largest level first; the levels after them share tiles and
/// count as small. Over the card's line none; with room all; else the ones
/// of at most `small` tiles.
pub(crate) fn levels_allowed(tiles: &[u32], room: bool, over: bool, small: u32) -> u32 {
    if over {
        return 0;
    }
    if room {
        return u32::MAX;
    }
    let own = tiles.len().min(32);
    let mut may = if own >= 32 { 0 } else { u32::MAX << own };
    for (level, &n) in tiles.iter().enumerate().take(32) {
        if n <= small {
            may |= 1 << level;
        }
    }
    may
}

/// What the line counts, since the start.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Count {
    /// Textures the pass uploaded, by their slot's use: read through a
    /// table, shown, unknown (counted while [`by_use_set`]).
    ServedRead,
    ServedShown,
    ServedUnknown,
    /// Uploads that gave tiles to small levels only, for want of room.
    SmallOnly,
    /// Slots whose texture was made although the card had no room (it is
    /// made level by level).
    PastFull,
    /// Levels hidden to bring the card under its line, and their bytes.
    Hidden,
    HiddenBytes,
    /// Levels whose tiles went back, and their bytes.
    Released,
    ReleasedBytes,
}

static COUNTS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

pub(crate) fn note(what: Count, n: u64) {
    COUNTS[what as usize].fetch_add(n, Relaxed);
}

/// The pass uploaded the texture of a slot of use `class`.
pub(crate) fn served(class: u8) {
    note(
        match class {
            READ => Count::ServedRead,
            SHOWN => Count::ServedShown,
            _ => Count::ServedUnknown,
        },
        1,
    );
}

/// The queue as the last pass found it: due slots read through a table,
/// shown, unknown (0 unless [`by_use_set`]); due slots looked at before;
/// slots held back for want of room.
static QUEUE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Notes the queue's side of the line ([`QUEUE`]).
pub(crate) fn queue_now(due_by_use: [u64; 3], again: u64, held: u64) {
    for (slot, now) in QUEUE.iter().zip([due_by_use[0], due_by_use[1], due_by_use[2], again, held]) {
        slot.store(now, Relaxed);
    }
}

fn state(switch: &Switch) -> &'static str {
    match switch.mode() {
        OFF => "off",
        ON => "on",
        _ if turn(crate::host::coarse_ms() / 1000) => "alt, on now",
        _ => "alt, off now",
    }
}

/// The line's text. `queue`: [`QUEUE`]; `census`: the queue's own counts
/// (slots, due, waiting for their memory, uploaded in part); `tiles`:
/// textures made level by level alive, the bytes of their tiles, levels
/// given since the start, uploads that left a held level without tiles;
/// `counts`: [`COUNTS`].
fn line(states: [&str; 3], queue: [u64; 5], census: [u64; 4], tiles: [u64; 4], counts: [u64; 9]) -> String {
    let [by_use, by_room, drop] = states;
    let [read, shown, unknown, again, held] = queue;
    let [slots, due, no_memory, in_part] = census;
    let [alive, tile_bytes, given, withheld] = tiles;
    format!(
        "serve: by use {by_use}; waiting {slots}: due {due} (read through a table {read}, shown {shown}, unknown {unknown}, \
         looked at before {again}), held for room {held}, on the title {} (no memory {no_memory}, in part {in_part}); \
         uploaded read {} shown {} unknown {}; levels by room {by_room}: {alive} textures level by level with {} MB of \
         tiles, levels given {given}, uploads that left a level waiting {withheld}, small levels only {}, made without \
         room {}; drop {drop}: levels hidden {} ({} MB), given back {} ({} MB) (counts since the start)",
        no_memory + in_part,
        counts[Count::ServedRead as usize],
        counts[Count::ServedShown as usize],
        counts[Count::ServedUnknown as usize],
        tile_bytes >> 20,
        counts[Count::SmallOnly as usize],
        counts[Count::PastFull as usize],
        counts[Count::Hidden as usize],
        counts[Count::HiddenBytes as usize] >> 20,
        counts[Count::Released as usize],
        counts[Count::ReleasedBytes as usize] >> 20,
    )
}

/// Logs the line, with the residency line, if one of the switches is set.
pub(crate) fn log_line() {
    if !(BY_USE.set() || BY_ROOM.set() || DROP.set() || past_pool()) {
        return;
    }
    let queue = QUEUE.each_ref().map(|q| q.load(Relaxed));
    let census = [0, 1, 2, 3].map(|i| crate::draw::HEAP_TEXTURES_QUEUE[i].load(Relaxed));
    let tiles = [0, 2, 5, 9].map(|i| crate::tiles::STATS[i].load(Relaxed));
    let counts = COUNTS.each_ref().map(|c| c.load(Relaxed));
    tracing::info!("{}", line([state(&BY_USE), state(&BY_ROOM), state(&DROP)], queue, census, tiles, counts));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_one_and_alt_set_a_switch() {
        assert_eq!(mode_of(None), OFF);
        assert_eq!(mode_of(Some("0")), OFF);
        assert_eq!(mode_of(Some("")), OFF);
        assert_eq!(mode_of(Some("true")), OFF);
        assert_eq!(mode_of(Some("1")), ON);
        assert_eq!(mode_of(Some(" alt ")), ALT);
    }

    #[test]
    fn a_switch_in_turn_is_off_for_twenty_seconds_and_on_for_twenty() {
        assert!(!turn(0) && !turn(19));
        assert!(turn(20) && turn(39));
        assert!(!turn(40) && turn(60));
    }

    #[test]
    fn slots_are_handed_out_by_use_and_the_newest_first_within_one() {
        // Newest write first, as the queue hands them over.
        let fresh: Vec<(u32, u64)> = (0..6).map(|i| (100 + i, 60 - u64::from(i))).collect();
        let class = [UNKNOWN, SHOWN, READ, UNKNOWN, READ, SHOWN];
        let slots: Vec<u32> = order(&fresh, &class, 6).iter().map(|s| s.0).collect();
        assert_eq!(slots, [102, 104, 101, 105, 100, 103]);
        // The write numbers stay with their slots.
        assert_eq!(order(&fresh, &class, 1), [(102, 58)]);
        assert!(order(&[], &[], 4).is_empty());
        assert!(order(&fresh, &class, 0).is_empty());
    }

    #[test]
    fn every_eighth_slot_handed_out_is_the_oldest_waiting() {
        let fresh: Vec<(u32, u64)> = (0..40).map(|i| (i, 1000 - u64::from(i))).collect();
        // The 20 oldest are the ones draws read: they come first, and the
        // oldest of all is slot 39.
        let class: Vec<u8> = (0..40).map(|i| if i >= 20 { READ } else { UNKNOWN }).collect();
        let slots: Vec<u32> = order(&fresh, &class, 40).iter().map(|s| s.0).collect();
        assert_eq!(&slots[..9], [20, 21, 22, 23, 24, 25, 26, 39, 27]);
        // Each slot once, all of them.
        let mut all = slots.clone();
        all.sort_unstable();
        assert_eq!(all, (0..40).collect::<Vec<u32>>());
        // A class missing for a slot counts as unknown.
        assert_eq!(order(&fresh[..3], &[SHOWN], 3).iter().map(|s| s.0).collect::<Vec<_>>(), [0, 1, 2]);
        // Unknown ones alone: the newest first, the oldest every eighth.
        let plain: Vec<u32> = order(&fresh, &[], 9).iter().map(|s| s.0).collect();
        assert_eq!(plain, [0, 1, 2, 3, 4, 5, 6, 39, 7]);
    }

    #[test]
    fn the_card_is_over_its_line_from_ninety_seven_percent() {
        assert!(!is_over(10_000, 9_699));
        assert!(is_over(10_000, 9_700));
        assert!(is_over(9058 << 20, 9541 << 20));
        assert!(!is_over(0, 5));
        assert_eq!(over_by(10_000, 9_699), 0);
        assert_eq!(over_by(10_000, 9_900), 200);
        assert!(!over(None));
        assert!(over(Some([100, 99, 0, 0])));
    }

    #[test]
    fn levels_are_dropped_for_what_the_card_is_over_by_and_a_hundredth_more() {
        // Under the line: nothing.
        assert_eq!(to_drop(10_000, 9_699, 0), 0);
        // 200 over: those and a hundredth of the budget.
        assert_eq!(to_drop(10_000, 9_900, 0), 300);
        // Levels hidden whose tiles are still to go back count as gone.
        assert_eq!(to_drop(10_000, 9_900, 150), 150);
        assert_eq!(to_drop(10_000, 9_900, 200), 0);
        assert_eq!(to_drop(10_000, 9_900, 5_000), 0);
        // No budget known: nothing.
        assert_eq!(to_drop(0, 9_900, 0), 0);
        // Session 0152's peak, 9,541 of 9,058 MB: 755 MB over the line,
        // and 90 MB more.
        assert_eq!(to_drop(9058 << 20, 9541 << 20, 0) >> 20, 845);
    }

    #[test]
    fn small_levels_come_before_large_ones() {
        // A 4096x4096 texture of a byte a texel: 256, 64, 16, 4 and 1 tiles,
        // and the levels after them in shared tiles.
        let tiles = [256, 64, 16, 4, 1];
        assert_eq!(levels_allowed(&tiles, true, false, 4), u32::MAX);
        // Without room: the levels of 4 tiles and less, and the shared ones.
        assert_eq!(levels_allowed(&tiles, false, false, 4), !0b111);
        assert_eq!(levels_allowed(&tiles, false, false, 0), u32::MAX << 5);
        // Over the line nothing, room or not.
        assert_eq!(levels_allowed(&tiles, true, true, 4), 0);
        assert_eq!(levels_allowed(&tiles, false, true, 4), 0);
        // No level with tiles of its own: all of them are small.
        assert_eq!(levels_allowed(&[], false, false, 4), u32::MAX);
    }

    #[test]
    fn the_line_holds_every_count() {
        let l = line(["on", "alt, off now", "off"], [1, 2, 3, 4, 5], [100, 10, 30, 40], [7, 64 << 20, 50, 6], [
            11,
            12,
            13,
            14,
            15,
            16,
            32 << 20,
            17,
            16 << 20,
        ]);
        assert_eq!(
            l,
            "serve: by use on; waiting 100: due 10 (read through a table 1, shown 2, unknown 3, looked at before 4), held \
             for room 5, on the title 70 (no memory 30, in part 40); uploaded read 11 shown 12 unknown 13; levels by \
             room alt, off now: 7 textures level by level with 64 MB of tiles, levels given 50, uploads that left a level \
             waiting 6, small levels only 14, made without room 15; drop off: levels hidden 16 (32 MB), given back 17 (16 \
             MB) (counts since the start)"
        );
    }
}
