//! Diagnostics: the card's time by kind of list, at play pace
//! (`VICTORY_PASS_TIMES=1`; off by default).
//!
//! The frame line says how long the card had lists of ours queued, and
//! `VICTORY_CARD_SPANS=1` how long it worked on each submission; neither says
//! on what. A timing run (`VICTORY_GPU_TIMING`) does, but sends every list
//! alone and waits for it: about half the frame rate, and the card never
//! works on two lists at a time as it does in play. Here the submissions go
//! out as always, with lists made once, of nothing but a timestamp, between
//! the lists of different kinds: the sync list (changed bytes go up to the
//! copies in the card's memory and are hashed), the runs' preludes, the
//! title's lists, the steps done on the card, the compare behind them. The
//! ticks between two stamps go to the kind between them.
//!
//! What it costs: a list of one timestamp per change of kind, some hundreds a
//! frame, each a few microseconds of the submitting thread. What it cannot
//! say: a stamp is taken when the card reaches it in the stream, and work of
//! the list before may still be under way, so a kind's time may show under
//! the next one's name where the card overlaps them.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::Host;
#[cfg(feature = "diagnostics")]
use crate::copystat::Timed;

/// Stamps in the ring: taken in order, used again once read.
const SLOTS: usize = 4096;

/// The kinds the time is summed by.
pub(super) const KINDS: [&str; 6] = [
    "sync lists (bytes up to the card's copies, hash passes)",
    "preludes",
    "the title's lists",
    "steps done on the card",
    "compare passes",
    "other lists of ours",
];
pub(super) const SYNC: u8 = 0;
pub(super) const COMPARE: u8 = 4;
const OTHER: u8 = 5;

/// The kind a list of a submission is summed under.
#[cfg(feature = "diagnostics")]
pub(super) fn kind(timed: Timed) -> u8 {
    match timed {
        Timed::Prelude => 1,
        Timed::Title(_) => 2,
        Timed::Steps => 3,
        Timed::Compare => COMPARE,
        Timed::Other => OTHER,
    }
}

/// Whether the stamps are asked for.
pub(super) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PASS_TIMES").is_some_and(|v| v == "1"))
}

/// The runs of one kind in a submission of `len` lists: where each begins
/// and its kind. `marks` say where a kind begins (in order; as the timing
/// run's), `shift` how many lists were put in front since they were noted.
/// What lies before the first mark is ours ([`OTHER`]); neighbours of one
/// kind are one run.
pub(super) fn segments(marks: &[(usize, u8)], shift: usize, len: usize) -> Vec<(usize, u8)> {
    let mut out: Vec<(usize, u8)> = Vec::new();
    let put = |at: usize, kind: u8, out: &mut Vec<(usize, u8)>| {
        if at >= len {
            return;
        }
        match out.last_mut() {
            // Nothing between the two marks: the later one names the lists.
            Some(last) if last.0 == at => last.1 = kind,
            _ => out.push((at, kind)),
        }
    };
    put(0, OTHER, &mut out);
    for &(at, kind) in marks {
        put(at + shift, kind, &mut out);
    }
    out.dedup_by(|b, a| a.1 == b.1);
    out
}

/// `lists` with a stamp in front of every run of `segments` and one behind
/// the last, and `cuts` (places where the lists are sent in parts) moved
/// with them. `stamps` are as many as the runs and one more.
pub(super) fn weave<T>(lists: Vec<T>, segments: &[(usize, u8)], cuts: &[usize], stamps: Vec<T>) -> (Vec<T>, Vec<usize>) {
    debug_assert_eq!(stamps.len(), segments.len() + 1);
    let moved = cuts.iter().map(|&c| c + segments.iter().filter(|s| s.0 < c).count()).collect();
    let mut out = Vec::with_capacity(lists.len() + stamps.len());
    let mut stamps = stamps.into_iter();
    let mut next = 0;
    for (i, list) in lists.into_iter().enumerate() {
        if segments.get(next).is_some_and(|s| s.0 == i) {
            out.extend(stamps.next());
            next += 1;
        }
        out.push(list);
    }
    out.extend(stamps);
    (out, moved)
}

/// Which stamps are out, and what was read of those that came back.
#[derive(Default)]
struct Ring {
    next: usize,
    /// Stamps taken and not read yet, in order: the first one's place and the
    /// kinds between it and those behind it, and (with
    /// `VICTORY_KIND_COUNTS=1`) what the lists of each kind hold.
    out: VecDeque<(usize, Vec<u8>, Vec<Counts>)>,
    /// Stamps out.
    held: usize,
    /// Per kind: runs and ticks.
    sums: [(u64, u64); KINDS.len()],
    /// Submissions that went without stamps (the ring was out).
    without: u64,
    /// The time of the title's runs of lists against what they hold.
    fit: super::fit::Fit,
}

/// What the lists between two stamps hold (`super::kinds::PER`).
pub(super) type Counts = [u32; super::kinds::PER];

/// The title's lists' kind.
const TITLE: u8 = 2;

impl Ring {
    /// The place of the first of `kinds.len() + 1` stamps, if the ring has
    /// them (they follow one another, around its end).
    fn take(&mut self, kinds: &[u8]) -> Option<usize> {
        let n = kinds.len() + 1;
        if kinds.is_empty() || self.held + n > SLOTS {
            self.without += 1;
            return None;
        }
        let first = self.next;
        self.next = (first + n) % SLOTS;
        self.held += n;
        self.out.push_back((first, kinds.to_vec(), Vec::new()));
        Some(first)
    }

    /// What the lists between the stamps taken last hold, kind by kind.
    fn holds(&mut self, counts: Vec<Counts>) {
        if let Some(last) = self.out.back_mut() {
            last.2 = counts;
        }
    }

    /// Reads what the card has stamped, in order: `tick` gives a stamp's
    /// ticks (0: not written yet) and is told to clear it.
    fn read(&mut self, mut tick: impl FnMut(usize, bool) -> u64) {
        while let Some((first, kinds, counts)) = self.out.front() {
            let last = (first + kinds.len()) % SLOTS;
            // (The card writes them in order: the last one says all are there.)
            if tick(last, false) == 0 {
                break;
            }
            let mut before = tick(*first, true);
            for (i, &kind) in kinds.iter().enumerate() {
                let now = tick((first + i + 1) % SLOTS, true);
                if before != 0 && now >= before {
                    let sum = &mut self.sums[usize::from(kind).min(KINDS.len() - 1)];
                    sum.0 += 1;
                    sum.1 += now - before;
                    if let (TITLE, Some(counts)) = (kind, counts.get(i)) {
                        self.fit.add(now - before, counts);
                    }
                }
                before = now;
            }
            self.held -= kinds.len() + 1;
            self.out.pop_front();
        }
    }
}

struct Passes {
    _allocator: ID3D12CommandAllocator,
    _heap: ID3D12QueryHeap,
    _back: ID3D12Resource,
    /// The mapped results, a tick count each.
    data: usize,
    lists: Vec<ID3D12CommandList>,
    ticks_per_ms: f64,
    ring: Ring,
}

// SAFETY: free-threaded objects and a mapping only read behind a lock.
unsafe impl Send for Passes {}

static MADE: Mutex<Option<Option<Passes>>> = Mutex::new(None);

/// Submissions stamped since the last line.
static STAMPED: AtomicU64 = AtomicU64::new(0);

impl Passes {
    fn read(&mut self) {
        let data = self.data as *mut u64;
        self.ring.read(|at, clear| {
            // SAFETY: inside the mapping; written by the card at whole words.
            unsafe {
                let p = data.add(at);
                let v = p.read_volatile();
                if clear {
                    p.write_volatile(0);
                }
                v
            }
        });
    }
}

impl Host {
    /// Stamps to stand around and between lists of `kinds` (one more than
    /// the kinds), if they are asked for and the ring has them.
    pub(super) fn pass_stamps(&self, kinds: &[u8]) -> Option<Vec<ID3D12CommandList>> {
        self.pass_stamps_of(kinds, Vec::new())
    }

    /// [`Host::pass_stamps`], with what the lists of each kind hold (as many
    /// as the kinds, or none).
    pub(super) fn pass_stamps_of(&self, kinds: &[u8], counts: Vec<Counts>) -> Option<Vec<ID3D12CommandList>> {
        if !on() {
            return None;
        }
        let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
        let passes = made
            .get_or_insert_with(|| {
                self.make_passes()
                    .inspect_err(|e| tracing::warn!(error = %e, "no stamps between lists (VICTORY_PASS_TIMES)"))
                    .ok()
            })
            .as_mut()?;
        passes.read();
        let first = passes.ring.take(kinds)?;
        passes.ring.holds(counts);
        STAMPED.fetch_add(1, Ordering::Relaxed);
        Some((0..=kinds.len()).map(|i| passes.lists[(first + i) % SLOTS].clone()).collect())
    }

    fn make_passes(&self) -> windows::core::Result<Passes> {
        let device = &self.device;
        // SAFETY: plain object creation; lists of one allocator recorded one
        // after the other.
        unsafe {
            let mut heap: Option<ID3D12QueryHeap> = None;
            device.CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: SLOTS as u32,
                    NodeMask: 0,
                },
                &mut heap,
            )?;
            let heap = heap.ok_or_else(windows::core::Error::empty)?;
            let back = self.read_back_buffer(8 * SLOTS as u64).ok_or_else(windows::core::Error::empty)?;
            let mut data = std::ptr::null_mut();
            back.Map(0, None, Some(&mut data))?;
            std::ptr::write_bytes(data.cast::<u8>(), 0, 8 * SLOTS);
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let mut lists = Vec::with_capacity(SLOTS);
            for i in 0..SLOTS as u32 {
                let list: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, i);
                list.ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, i, 1, &back, u64::from(i) * 8);
                list.Close()?;
                lists.push(list.cast()?);
            }
            Ok(Passes {
                _allocator: allocator,
                _heap: heap,
                _back: back,
                data: data as usize,
                lists,
                ticks_per_ms: self.queue.GetTimestampFrequency()? as f64 / 1000.0,
                ring: Ring::default(),
            })
        }
    }
}

/// The line's text: per kind, runs and milliseconds a frame.
fn text(sums: &[(u64, u64); KINDS.len()], ticks_per_ms: f64, frames: f64) -> String {
    let ms = |ticks: u64| ticks as f64 / ticks_per_ms / frames;
    let all: u64 = sums.iter().map(|s| s.1).sum();
    let parts: Vec<String> = KINDS
        .iter()
        .zip(sums)
        .map(|(name, s)| format!("{name} {:.2} ms in {:.1} runs", ms(s.1), s.0 as f64 / frames))
        .collect();
    format!("{:.1} ms in all: {}", ms(all), parts.join("; "))
}

/// For the frame-rate line: the card's time by kind of list since the last
/// one, per frame.
pub(super) fn line(frames: f64) {
    if !on() {
        return;
    }
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(passes)) = made.as_mut() else {
        return;
    };
    passes.read();
    let sums = std::mem::take(&mut passes.ring.sums);
    let without = std::mem::take(&mut passes.ring.without);
    let fit = std::mem::take(&mut passes.ring.fit);
    if super::kinds::on() {
        tracing::info!(
            "the card's time of the title's lists by what is in them (VICTORY_PASS_TIMES=1 with VICTORY_KIND_COUNTS=1; a fit: \
             what goes together in every run cannot be told apart): {}",
            fit.text(passes.ticks_per_ms, frames)
        );
    }
    tracing::info!(
        "the card's time by kind of list per frame (VICTORY_PASS_TIMES=1; a kind's work may still run under the next \
         one's name): {}; {:.1} submissions stamped, {:.1} without (the ring of {SLOTS} was out)",
        text(&sums, passes.ticks_per_ms, frames),
        STAMPED.swap(0, Ordering::Relaxed) as f64 / frames,
        without as f64 / frames
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_of_one_kind_are_found_from_the_marks() {
        // prelude, title, prelude, title, title, compare
        let marks = [(0, 1), (1, 2), (2, 1), (3, 2), (4, 2), (5, COMPARE)];
        assert_eq!(segments(&marks, 0, 6), [(0, 1), (1, 2), (2, 1), (3, 2), (5, COMPARE)]);
        // A list of ours in front: what is before the first mark is "other".
        assert_eq!(segments(&marks, 1, 7), [(0, OTHER), (1, 1), (2, 2), (3, 1), (4, 2), (6, COMPARE)]);
        // A mark at the end names nothing; no marks: all other; no lists: nothing.
        assert_eq!(segments(&[(0, 2), (2, OTHER)], 0, 2), [(0, 2)]);
        assert_eq!(segments(&[], 0, 3), [(0, OTHER)]);
        assert!(segments(&[(0, 2)], 0, 0).is_empty());
    }

    #[test]
    fn stamps_stand_between_the_runs_and_the_lists_keep_their_order() {
        let lists = vec!["p0", "t0", "t1", "c"];
        let segs = [(0, 1), (1, 2), (3, COMPARE)];
        let (out, cuts) = weave(lists, &segs, &[1, 3], vec!["s0", "s1", "s2", "s3"]);
        assert_eq!(out, ["s0", "p0", "s1", "t0", "t1", "s2", "c", "s3"]);
        // A part still begins with the list it began with, or that list's stamp.
        assert_eq!(cuts, [2, 5]);
        assert_eq!(out[cuts[0]], "s1");
        assert_eq!(out[cuts[1]], "s2");
        // Without the stamps the lists are those sent before.
        let left: Vec<_> = out.into_iter().filter(|l| !l.starts_with('s')).collect();
        assert_eq!(left, ["p0", "t0", "t1", "c"]);
    }

    #[test]
    fn ticks_go_to_the_kind_between_two_stamps_once_the_card_wrote_them() {
        let mut ring = Ring::default();
        let mut ticks = vec![0u64; SLOTS];
        let a = ring.take(&[1, 2, COMPARE]).unwrap();
        let b = ring.take(&[2]).unwrap();
        assert_eq!((a, b, ring.held), (0, 4, 6));
        // Nothing written: nothing read.
        ring.read(|at, _| ticks[at]);
        assert_eq!(ring.held, 6);
        // The first submission ran; of the second only its first stamp.
        ticks[..5].copy_from_slice(&[100, 130, 200, 210, 300]);
        let mut read = |ring: &mut Ring, ticks: &mut Vec<u64>| {
            ring.read(|at, clear| {
                let v = ticks[at];
                if clear {
                    ticks[at] = 0;
                }
                v
            })
        };
        read(&mut ring, &mut ticks);
        assert_eq!(ring.held, 2);
        assert_eq!((ring.sums[1], ring.sums[2], ring.sums[4]), ((1, 30), (1, 70), (1, 10)));
        ticks[5] = 340;
        read(&mut ring, &mut ticks);
        assert_eq!((ring.held, ring.sums[2]), (0, (2, 110)));
        // The idle time between the two submissions (210 to 300) is nobody's.
        assert_eq!(ring.sums.iter().map(|s| s.1).sum::<u64>(), 150);
        assert!(ticks.iter().all(|&t| t == 0), "read stamps are cleared for their next use");
    }

    #[test]
    fn the_titles_runs_go_to_the_fit_with_what_they_hold() {
        let mut ring = Ring::default();
        let mut ticks = vec![0u64; SLOTS];
        // prelude, title, compare; then title, prelude, title.
        ring.take(&[1, TITLE, COMPARE]).unwrap();
        ring.holds(vec![[0; 6], [10, 0, 0, 2, 1, 3], [0, 4, 0, 0, 1, 1]]);
        ring.take(&[TITLE, 1, TITLE]).unwrap();
        ring.holds(vec![[20, 0, 0, 0, 0, 0], [0; 6], [0, 5, 1, 0, 2, 2]]);
        // Without counts: timed, not fitted.
        ring.take(&[TITLE]).unwrap();
        ticks[..10].copy_from_slice(&[100, 110, 200, 230, 300, 360, 365, 400, 500, 520]);
        ring.read(|at, _| ticks[at]);
        assert_eq!(ring.held, 0);
        assert_eq!(ring.sums[usize::from(TITLE)], (4, 90 + 60 + 35 + 20));
        let text = ring.fit.text(1.0, 1.0);
        assert!(text.starts_with("3 runs"), "{text}");
    }

    #[test]
    fn a_ring_that_is_out_leaves_the_submission_without_stamps() {
        let mut ring = Ring::default();
        let kinds = vec![2u8; SLOTS / 2];
        assert!(ring.take(&kinds).is_some());
        assert!(ring.take(&kinds).is_none());
        assert_eq!(ring.without, 1);
        // Read: there is room again, around the ring's end.
        ring.read(|_, _| 7);
        assert_eq!(ring.held, 0);
        let first = ring.take(&kinds).unwrap();
        assert_eq!(first, SLOTS / 2 + 1);
        assert_eq!(ring.next, (first + SLOTS / 2 + 1) % SLOTS);
        assert!(ring.take(&[]).is_none());
    }

    #[test]
    fn the_line_names_every_kind() {
        let mut sums = [(0, 0); KINDS.len()];
        sums[2] = (20, 30_000);
        sums[4] = (10, 10_000);
        let t = text(&sums, 1000.0, 10.0);
        assert!(t.starts_with("4.0 ms in all: "), "{t}");
        assert!(t.contains("the title's lists 3.00 ms in 2.0 runs"), "{t}");
        assert!(t.contains("compare passes 1.00 ms in 1.0 runs"), "{t}");
    }

    /// On the card: stamps around a list of ours come back in order.
    #[test]
    fn the_card_stamps_what_is_sent_between_them() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let Ok(mut passes) = host.make_passes() else {
            return;
        };
        let first = passes.ring.take(&[SYNC, 2]).unwrap();
        let lists: Vec<Option<ID3D12CommandList>> = (0..3).map(|i| Some(passes.lists[first + i].clone())).collect();
        {
            let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            *submitted += 1;
            // SAFETY: closed lists; the fence and event are the host's, under
            // its lock.
            unsafe {
                host.queue.ExecuteCommandLists(&lists);
                host.queue.Signal(&host.fence, *submitted).unwrap();
                host.fence.SetEventOnCompletion(*submitted, host.event).unwrap();
                windows::Win32::System::Threading::WaitForSingleObject(host.event, windows::Win32::System::Threading::INFINITE);
            }
        }
        // SAFETY: the mapping holds `SLOTS` tick counts.
        let ticks = unsafe { std::slice::from_raw_parts(passes.data as *const u64, 3) }.to_vec();
        assert!(ticks[0] != 0 && ticks[0] <= ticks[1] && ticks[1] <= ticks[2], "{ticks:?}");
        passes.read();
        assert_eq!(passes.ring.held, 0);
        assert_eq!((passes.ring.sums[0].0, passes.ring.sums[2].0), (1, 1));
        assert_eq!(passes.ring.sums[0].1 + passes.ring.sums[2].1, ticks[2] - ticks[0]);
    }
}
