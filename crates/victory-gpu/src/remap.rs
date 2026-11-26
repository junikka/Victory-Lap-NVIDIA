//! Page-mapping meter (`VICTORY_REMAP_METER`).
//!
//! Off unless the variable is set to something other than `0`. The base
//! mapping path counts nothing and reads no cycle counters, and this tree
//! has no frame log to pour counts into, so an unset variable keeps that
//! path. Set, the system's mapping calls and the queue's page-mapping
//! copies are counted and a line is written about once a second.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether the meter runs. Unset and `0` are off.
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REMAP_METER").is_some_and(|v| v != "0"))
}

/// `VICTORY_REMAP_SPLIT`: 0 unset (split a placeholder first, as before),
/// 1 `late` (map first, split only if that failed), 2 `alt`.
pub fn split_mode(value: Option<&str>) -> u8 {
    match value {
        Some("late") => 1,
        Some("alt") => 2,
        _ => 0,
    }
}

/// Whether page `at` of `section` continues the page mapped just before
/// (`prev_after` is the address after that page). A run is the next index
/// of the same section; the same index is that page again.
pub fn follows(
    prev_after: u64,
    prev_section: u64,
    prev_index: u64,
    at: u64,
    section: u64,
    index: u64,
) -> (bool, bool) {
    if prev_after == at && prev_section == section {
        if index == prev_index.wrapping_add(1) {
            return (true, false);
        }
        if index == prev_index {
            return (false, true);
        }
    }
    (false, false)
}

const PAGE: u64 = 64 << 10;

/// Calls, microseconds, thread cycles and timestamp-counter ticks, four
/// system calls (unmap, split, map, commit).
static SYSCALL: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
/// Queue page-mapping copies: calls, then ranges and pages of kinds 0, 1
/// and 2.
static PAGES: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
/// Alias pages, those that continued a run, those that were the same page,
/// and splits tried only because a mapping failed first.
static ALIAS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn note() {
    static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
    let Ok(mut last) = LAST.lock() else {
        return;
    };
    let now = std::time::Instant::now();
    if last.is_some_and(|t| now.duration_since(t).as_secs() < 1) {
        return;
    }
    *last = Some(now);
    let take = |c: &[AtomicU64]| -> Vec<u64> { c.iter().map(|n| n.swap(0, Relaxed)).collect() };
    let sys = take(&SYSCALL);
    let pages = take(&PAGES);
    let alias = take(&ALIAS);
    if sys.iter().chain(&pages).chain(&alias).all(|&n| n == 0) {
        return;
    }
    tracing::info!(
        "remapping, metered (VICTORY_REMAP_METER): unmap {} calls {} us, split {} {}, map {} {}, commit {} {}; \
         page copies {} (kind 0: {} ranges {} pages, kind 1: {} {}, kind 2: {} {}); \
         alias pages {} ({} continued, {} the same page), splits after a failed map {}",
        sys[0], sys[1], sys[4], sys[5], sys[8], sys[9], sys[12], sys[13],
        pages[0], pages[1], pages[2], pages[3], pages[4], pages[5], pages[6],
        alias[0], alias[1], alias[2], alias[3],
    );
}

/// One system call that changes a mapping. `kind` is [`crate`] xmem's
/// `Call` discriminant (unmap 7, split 9, map 11, commit 13). `cycles` is
/// the calling thread's processor cycles over the call, 0 when they were
/// not read.
pub fn syscall(kind: u8, us: u64, cycles: u64, tsc: u64) {
    if !on() {
        return;
    }
    let slot = match kind {
        7 => 0,
        9 => 4,
        11 => 8,
        13 => 12,
        _ => return,
    };
    SYSCALL[slot].fetch_add(1, Relaxed);
    SYSCALL[slot + 1].fetch_add(us, Relaxed);
    SYSCALL[slot + 2].fetch_add(cycles, Relaxed);
    SYSCALL[slot + 3].fetch_add(tsc, Relaxed);
    note();
}

/// One range of a queue page-mapping copy. Kinds 0, 1 and 2 are counted.
pub fn page_range(kind: u32, pages: u32) {
    if !on() {
        return;
    }
    PAGES[0].fetch_add(1, Relaxed);
    if let Some(slot) = match kind {
        0 => Some(1),
        1 => Some(3),
        2 => Some(5),
        _ => None,
    } {
        PAGES[slot].fetch_add(1, Relaxed);
        PAGES[slot + 1].fetch_add(u64::from(pages), Relaxed);
    }
    note();
}

thread_local! {
    static PREV: std::cell::Cell<(u64, u64, u64)> = const { std::cell::Cell::new((0, 0, 0)) };
}

/// One page mapped through [`crate`] xmem's `map_alias`.
pub fn alias_page(at: u64, section: u64, index: u64) {
    if !on() {
        return;
    }
    let (after, sec, idx) = PREV.with(|p| p.get());
    let (run, same) = follows(after, sec, idx, at, section, index);
    ALIAS[0].fetch_add(1, Relaxed);
    if run {
        ALIAS[1].fetch_add(1, Relaxed);
    }
    if same {
        ALIAS[2].fetch_add(1, Relaxed);
    }
    PREV.with(|p| p.set((at.wrapping_add(PAGE), section, index)));
    note();
}

/// A split made only after a mapping into the placeholder failed.
pub fn late_split() {
    if !on() {
        return;
    }
    ALIAS[3].fetch_add(1, Relaxed);
    note();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_page_run_is_the_next_index_of_the_same_section() {
        assert_eq!(follows(0, 0, 0, 0x1000, 1, 0), (false, false));
        assert_eq!(follows(0x1000, 7, 3, 0x1000, 7, 4), (true, false));
        assert_eq!(follows(0x1000, 7, 3, 0x1000, 7, 3), (false, true));
        assert_eq!(follows(0x1000, 7, 3, 0x1000, 8, 4), (false, false));
        assert_eq!(follows(0x2000, 7, 3, 0x1000, 7, 4), (false, false));
    }

    #[test]
    fn split_late_is_off_unless_asked() {
        assert_eq!(split_mode(None), 0);
        assert_eq!(split_mode(Some("")), 0);
        assert_eq!(split_mode(Some("0")), 0);
        assert_eq!(split_mode(Some("late")), 1);
        assert_eq!(split_mode(Some("alt")), 2);
    }
}
