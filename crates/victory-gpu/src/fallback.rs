//! Does the title fall back when a precompiled pipeline fails? (Ticket P1.)
//!
//! Every call to device slots 0x1c8 and 0x218 fails (`device::unknown_pipeline`).
//! Each one is noted here with its root signature (the description's first
//! word) and its code pointers and sizes; the first 20 per slot are logged.
//! Every ordinary creation (0x50, 0x58, 0x1f0, 0x2b0) then looks its root
//! signature up once: failed calls still waiting under that root count as
//! followed, with the delay from the failure. Every 30 s and at exit one
//! line per failed slot says how many were and were not followed, and the
//! median delay. Logging only; nothing here changes what a call answers.
//!
//! The delay alone misleads (ticket 045): one ordinary creation follows
//! every call waiting under its root, and the title's 0x1c8 calls share a
//! few roots, so run fp-1's 1,367 ms median was 47 ordinary creations
//! clearing 25,450 calls made in a burst during loading. Two more figures
//! per slot tell a wait from a burst: `span_ms`, first to last failed call,
//! and `median_gap_ms`, from the newest call waiting under the root (the
//! one the title would have to have been waiting on) to the creation.
//!
//! On by default; `VICTORY_FALLBACK_LOG=0` turns it off.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Failed calls logged one by one per slot.
const LOGGED: u64 = 20;
/// How often the summary is written.
const EVERY: Duration = Duration::from_secs(30);
/// The two failing slots, in index order.
const FAILED_SLOTS: [usize; 2] = [0x1c8, 0x218];
/// The ordinary creation slots, in index order.
const ORDINARY_SLOTS: [usize; 4] = [0x50, 0x58, 0x1f0, 0x2b0];

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FALLBACK_LOG").is_none_or(|v| v != "0"))
}

fn failed_index(slot: usize) -> Option<usize> {
    FAILED_SLOTS.iter().position(|&s| s == slot)
}

fn ordinary_index(slot: usize) -> Option<usize> {
    ORDINARY_SLOTS.iter().position(|&s| s == slot)
}

/// One failing slot's tally.
#[derive(Default, Clone)]
struct Tally {
    calls: u64,
    /// Calls whose description could not be read (no root to match).
    unread: u64,
    /// Delays in ms from a failed call to the first ordinary creation under
    /// its root signature, one per followed call.
    delays: Vec<u64>,
    /// Of the followed, which ordinary slot came first.
    by: [u64; 4],
    /// ms of the first and last failed call.
    first: Option<u64>,
    last: u64,
    /// Per ordinary creation that followed calls of this slot: ms from the
    /// newest of them.
    gaps: Vec<u64>,
}

/// The matching, apart from time and the log.
#[derive(Default)]
struct Matcher {
    tally: [Tally; 2],
    /// Failed calls not yet followed, per root signature: (failed slot
    /// index, ms of the call).
    waiting: HashMap<usize, Vec<(usize, u64)>>,
    /// Ordinary creations per slot, and those under a root with a failed
    /// call waiting.
    ordinary: [u64; 4],
    matched: [u64; 4],
}

impl Matcher {
    /// A failed call to `failed` (index) at `ms`, under `root` (`None`:
    /// description unreadable). Returns its number for that slot.
    fn failed(&mut self, failed: usize, root: Option<usize>, ms: u64) -> u64 {
        let t = &mut self.tally[failed];
        t.calls += 1;
        t.first.get_or_insert(ms);
        t.last = t.last.max(ms);
        match root {
            Some(root) if root != 0 => self.waiting.entry(root).or_default().push((failed, ms)),
            _ => t.unread += 1,
        }
        t.calls
    }

    /// An ordinary creation through `ordinary` (index) at `ms` under `root`.
    fn ordinary(&mut self, ordinary: usize, root: usize, ms: u64) {
        self.ordinary[ordinary] += 1;
        let Some(waiting) = self.waiting.remove(&root) else {
            return;
        };
        self.matched[ordinary] += 1;
        let mut newest = [None::<u64>; 2];
        for (failed, at) in waiting {
            let t = &mut self.tally[failed];
            t.delays.push(ms.saturating_sub(at));
            t.by[ordinary] += 1;
            newest[failed] = newest[failed].max(Some(at));
        }
        for (t, at) in self.tally.iter_mut().zip(newest) {
            if let Some(at) = at {
                t.gaps.push(ms.saturating_sub(at));
            }
        }
    }

    /// Per failed slot: (ms from first to last failed call, median gap ms).
    fn burst(&self, failed: usize) -> (u64, Option<u64>) {
        let t = &self.tally[failed];
        let span = t.first.map_or(0, |f| t.last - f);
        (span, median(&t.gaps))
    }

    /// Per failed slot: (calls, followed, not followed, median delay ms).
    fn summary(&self, failed: usize) -> (u64, u64, u64, Option<u64>) {
        let t = &self.tally[failed];
        let followed = t.delays.len() as u64;
        (t.calls, followed, t.calls - followed, median(&t.delays))
    }

    fn lines(&self) -> Vec<String> {
        (0..2)
            .map(|i| {
                let (calls, followed, not_followed, median) = self.summary(i);
                let t = &self.tally[i];
                let by: Vec<String> =
                    ORDINARY_SLOTS.iter().zip(t.by).map(|(s, n)| format!("{s:#x}:{n}")).collect();
                let shown = |m: Option<u64>| m.map_or("-".to_owned(), |m| m.to_string());
                let (span, gap) = self.burst(i);
                format!(
                    "pipeline fallback: slot={:#x} failed={calls} followed={followed} not_followed={not_followed} unread={} median_delay_ms={} first_by=[{}] span_ms={span} median_gap_ms={}",
                    FAILED_SLOTS[i],
                    t.unread,
                    shown(median),
                    by.join(" "),
                    shown(gap)
                )
            })
            .collect()
    }

    fn ordinary_line(&self) -> String {
        let per: Vec<String> = ORDINARY_SLOTS
            .iter()
            .enumerate()
            .map(|(i, s)| format!("{s:#x}:{}/{}", self.matched[i], self.ordinary[i]))
            .collect();
        format!(
            "pipeline fallback: ordinary creations under a waiting root/all [{}], roots still waiting={}",
            per.join(" "),
            self.waiting.len()
        )
    }
}

/// The middle value (the upper of two), none of none.
fn median(values: &[u64]) -> Option<u64> {
    let mut d = values.to_vec();
    d.sort_unstable();
    (!d.is_empty()).then(|| d[d.len() / 2])
}

struct State {
    matcher: Matcher,
    start: Instant,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Matcher, u64) -> R) -> R {
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let s = guard.get_or_insert_with(|| State {
        matcher: Matcher::default(),
        start: Instant::now(),
    });
    let ms = s.start.elapsed().as_millis() as u64;
    f(&mut s.matcher, ms)
}

fn write_summary() {
    let lines = {
        let Ok(guard) = STATE.try_lock() else { return };
        let Some(s) = guard.as_ref() else { return };
        let mut l = s.matcher.lines();
        l.push(s.matcher.ordinary_line());
        l
    };
    for line in lines {
        tracing::info!("{line}");
    }
}

unsafe extern "C" {
    fn atexit(f: extern "C" fn()) -> i32;
}

extern "C" fn at_exit() {
    write_summary();
}

/// The summary thread and the exit hook, once, on the first failed call.
fn start() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        // SAFETY: registers a function with the host CRT's exit table.
        unsafe { atexit(at_exit) };
        let _ = std::thread::Builder::new()
            .name("fallback-summary".into())
            .spawn(|| {
                loop {
                    std::thread::sleep(EVERY);
                    write_summary();
                }
            });
    });
}

/// `words` 64-bit words at `at`, read through the kernel: the title's
/// pointer is not known to be mapped that far.
fn words(at: usize, words: usize) -> Option<Vec<u64>> {
    #[cfg(windows)]
    {
        let mut bytes = vec![0u8; words * 8];
        (at != 0 && crate::host::read_into(at as *const u8, &mut bytes)).then(|| {
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| u64::from_le_bytes(*c))
                .collect()
        })
    }
    #[cfg(not(windows))]
    {
        let _ = (at, words);
        None
    }
}

/// The code pairs a description holds: 0x218 has root, dispatch state,
/// an unknown piece, then the code (words 5 and 6); 0x1c8 has root, two
/// state pieces, then stage slots of a code pair and a metadata pair
/// (words 5+4k and 6+4k, four slots looked at). Empty pairs are skipped.
fn code_pairs(slot: usize, w: &[u64]) -> Vec<(u64, u64)> {
    let at: &[usize] = if slot == 0x218 { &[5] } else { &[5, 9, 13, 17] };
    at.iter()
        .filter_map(|&i| Some((*w.get(i)?, *w.get(i + 1)?)))
        .filter(|&(code, size)| code != 0 && size != 0)
        .collect()
}

/// Hook in `device::unknown_pipeline`: a failed call to `slot` with
/// description `desc`.
pub(crate) fn failed(slot: usize, desc: *const u8) {
    if !on() {
        return;
    }
    let Some(index) = failed_index(slot) else {
        return;
    };
    start();
    let w = words(desc as usize, 21);
    let root = w.as_ref().map(|w| w[0] as usize);
    let n = with(|m, ms| m.failed(index, root, ms));
    if n <= LOGGED {
        let code: Vec<String> = w
            .as_deref()
            .map(|w| code_pairs(slot, w))
            .unwrap_or_default()
            .iter()
            .map(|(c, s)| format!("{c:#x}+{s:#x}"))
            .collect();
        tracing::info!(
            "pipeline fallback: failed slot={slot:#x} n={n} root={:#x} code=[{}]",
            root.unwrap_or(0),
            code.join(" ")
        );
    }
}

/// Hook in the ordinary creation methods: a creation through `slot` under
/// root signature `root`. One map lookup.
pub(crate) fn ordinary(slot: usize, root: crate::object::This) {
    if !on() || root.is_null() {
        return;
    }
    let Some(index) = ordinary_index(slot) else {
        return;
    };
    with(|m, ms| m.ordinary(index, root as usize, ms));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_later_creation_under_the_same_root_follows_every_waiting_call() {
        let mut m = Matcher::default();
        m.failed(0, Some(0x100), 10);
        m.failed(0, Some(0x100), 20);
        m.failed(1, Some(0x100), 30);
        m.ordinary(3, 0x100, 50);
        assert_eq!(m.summary(0), (2, 2, 0, Some(40)));
        assert_eq!(m.summary(1), (1, 1, 0, Some(20)));
        assert_eq!(m.tally[0].by, [0, 0, 0, 2]);
        assert_eq!((m.ordinary, m.matched), ([0, 0, 0, 1], [0, 0, 0, 1]));
    }

    #[test]
    fn another_root_or_an_earlier_creation_does_not_follow() {
        let mut m = Matcher::default();
        m.ordinary(0, 0x100, 0);
        m.failed(0, Some(0x100), 10);
        m.ordinary(1, 0x200, 20);
        assert_eq!(m.summary(0), (1, 0, 1, None));
        assert_eq!(m.ordinary, [1, 1, 0, 0]);
        assert_eq!(m.matched, [0, 0, 0, 0]);
    }

    #[test]
    fn a_call_is_followed_once_and_the_first_slot_counts() {
        let mut m = Matcher::default();
        m.failed(1, Some(0x100), 0);
        m.ordinary(1, 0x100, 5);
        m.ordinary(2, 0x100, 9);
        assert_eq!(m.summary(1), (1, 1, 0, Some(5)));
        assert_eq!(m.tally[1].by, [0, 1, 0, 0]);
        assert_eq!(m.matched, [0, 1, 0, 0]);
        // A new failure under the root waits again.
        m.failed(1, Some(0x100), 100);
        assert_eq!(m.summary(1), (2, 1, 1, Some(5)));
    }

    #[test]
    fn unreadable_descriptions_never_match() {
        let mut m = Matcher::default();
        m.failed(0, None, 0);
        m.failed(0, Some(0), 0);
        m.ordinary(0, 0, 1);
        assert_eq!(m.summary(0), (2, 0, 2, None));
        assert_eq!(m.tally[0].unread, 2);
    }

    #[test]
    fn the_median_is_the_middle_delay() {
        let mut m = Matcher::default();
        for (root, at) in [(1, 0), (2, 0), (3, 0)] {
            m.failed(0, Some(root), at);
        }
        m.ordinary(0, 3, 300);
        m.ordinary(0, 1, 100);
        m.ordinary(0, 2, 200);
        assert_eq!(m.summary(0).3, Some(200));
    }

    #[test]
    fn the_summary_line_names_slot_counts_and_median() {
        let mut m = Matcher::default();
        m.failed(0, Some(1), 0);
        m.failed(0, Some(2), 0);
        m.ordinary(2, 1, 7);
        let lines = m.lines();
        assert_eq!(
            lines[0],
            "pipeline fallback: slot=0x1c8 failed=2 followed=1 not_followed=1 unread=0 median_delay_ms=7 first_by=[0x50:0 0x58:0 0x1f0:1 0x2b0:0] span_ms=0 median_gap_ms=7"
        );
        assert!(
            lines[1]
                .starts_with("pipeline fallback: slot=0x218 failed=0 followed=0 not_followed=0")
        );
        assert_eq!(
            m.ordinary_line(),
            "pipeline fallback: ordinary creations under a waiting root/all [0x50:0/0 0x58:0/0 0x1f0:1/1 0x2b0:0/0], roots still waiting=1"
        );
    }

    /// Run fp-1's shape: a burst of calls under one root, then one creation.
    /// The delay is spread over the burst; the gap is from its end.
    #[test]
    fn a_burst_under_one_root_shows_as_span_and_a_short_gap() {
        let mut m = Matcher::default();
        for at in (0..=1400).step_by(100) {
            m.failed(0, Some(0x100), at);
        }
        m.failed(1, Some(0x100), 1000);
        m.ordinary(3, 0x100, 1405);
        let (calls, followed, _, delay) = m.summary(0);
        assert_eq!((calls, followed, delay), (15, 15, Some(705)));
        assert_eq!(m.burst(0), (1400, Some(5)));
        assert_eq!(m.burst(1), (0, Some(405)));
        // Nothing waits any more: a later creation adds no gap.
        m.ordinary(3, 0x100, 2000);
        assert_eq!(m.tally[0].gaps, vec![5]);
    }

    #[test]
    fn no_calls_no_span_no_gap() {
        let m = Matcher::default();
        assert_eq!(m.burst(0), (0, None));
        assert!(m.lines()[1].ends_with("span_ms=0 median_gap_ms=-"));
    }

    #[test]
    fn code_pairs_are_read_per_layout() {
        let mut w = vec![0u64; 21];
        w[5] = 0xa000;
        w[6] = 0x140;
        w[13] = 0xb000;
        w[14] = 0x200;
        assert_eq!(code_pairs(0x218, &w), vec![(0xa000, 0x140)]);
        assert_eq!(
            code_pairs(0x1c8, &w),
            vec![(0xa000, 0x140), (0xb000, 0x200)]
        );
    }
}
