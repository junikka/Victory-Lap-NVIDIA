//! Audio numbers for the log: one line every five seconds while any audio
//! work was done (`VICTORY_AUDIO_STATS=0`: no line), written by a thread of
//! its own: the title's mixer and our decoders have no time for the log.
//!
//! Two decoders stand in for console hardware here, and the title's mixer
//! thread (time critical, one block of 256 frames every 5.3 ms) depends on
//! both: the Opus unit (`apu.rs`), which decodes inside the title's own call
//! unless `VICTORY_APU_DECODE=alt`, and the XMA decoder (`audio.rs`), a
//! thread of ours. A mixer that runs out of its block, or finds a ring
//! empty, plays a gap: scratchy audio that cuts in and out. The line says
//! how close each came, and how often the mixer itself came late to our
//! calls, whatever held it up (its own work, a page fault, another thread).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Once, OnceLock};
use std::time::{Duration, Instant};

/// One mixer block: 256 frames at 48 kHz.
pub(crate) const BLOCK_US: u64 = 5_333;
/// Calls of the title's less than this apart belong to one run (one block's
/// work): the mixer sleeps about 4 ms between blocks.
const RUN_GAP_US: u64 = 1_000;
/// An XMA decoding pass this long is slow: an output block lasts 2.7 ms.
const SLOW_PASS_US: u64 = 4_000;
const PERIOD_US: u64 = 5_000_000;
/// The title's audio calls this far apart are late: six blocks, more than
/// a 20 ms Opus packet lasts.
const LATE_US: u64 = 32_000;
/// Calls this far apart are no stall: nothing was playing.
const AWAY_US: u64 = 1_000_000;

fn us(d: Duration) -> u64 {
    d.as_micros() as u64
}

/// Counters since the last line. All relaxed: a line may be a count off.
#[derive(Default)]
pub(crate) struct Meter {
    // The Opus unit.
    packets: AtomicU64,
    decode_us: AtomicU64,
    decode_longest_us: AtomicU64,
    streams: AtomicU64,
    streams_peak: AtomicU64,
    /// Time the title's thread spent in its queueing calls.
    caller_us: AtomicU64,
    caller_longest_us: AtomicU64,
    /// The run of calls going on: its time in our calls, and when its last
    /// call ended (microseconds since `origin`).
    run_us: AtomicU64,
    run_end_us: AtomicU64,
    run_longest_us: AtomicU64,
    runs_over_block: AtomicU64,
    /// When the title last called (queued Opus work or took XMA blocks).
    call_us: AtomicU64,
    call_gap_longest_us: AtomicU64,
    late_calls: AtomicU64,
    /// Results asked for before they were decoded (`VICTORY_APU_DECODE=alt`).
    not_ready: AtomicU64,
    wait_longest_us: AtomicU64,
    // The XMA decoder.
    passes: AtomicU64,
    slow_passes: AtomicU64,
    pass_longest_us: AtomicU64,
    pass_start_us: AtomicU64,
    gap_longest_us: AtomicU64,
    voices_peak: AtomicU64,
    blocks: AtomicU64,
    takes: AtomicU64,
    emptied: AtomicU64,
}

/// What a line shows: the counters of one period.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct Numbers {
    pub packets: u64,
    pub decode_us: u64,
    pub decode_longest_us: u64,
    pub streams_peak: u64,
    pub caller_us: u64,
    pub caller_longest_us: u64,
    pub run_longest_us: u64,
    pub runs_over_block: u64,
    pub call_gap_longest_us: u64,
    pub late_calls: u64,
    pub not_ready: u64,
    pub wait_longest_us: u64,
    pub passes: u64,
    pub slow_passes: u64,
    pub pass_longest_us: u64,
    pub gap_longest_us: u64,
    pub voices_peak: u64,
    pub blocks: u64,
    pub takes: u64,
    pub emptied: u64,
}

pub(crate) static METER: OnceLock<Meter> = OnceLock::new();

pub(crate) fn meter() -> &'static Meter {
    METER.get_or_init(Meter::default)
}

fn origin() -> Instant {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

fn now_us() -> u64 {
    us(origin().elapsed())
}

impl Meter {
    /// One Opus packet decoded, on whichever thread.
    pub(crate) fn decoded(&self, took: Duration) {
        let took = us(took);
        self.packets.fetch_add(1, Relaxed);
        self.decode_us.fetch_add(took, Relaxed);
        self.decode_longest_us.fetch_max(took, Relaxed);
    }

    /// An Opus stream activated (`true`) or deactivated.
    pub(crate) fn stream(&self, on: bool) {
        let now = if on {
            self.streams.fetch_add(1, Relaxed) + 1
        } else {
            // Never below zero, whatever order two threads count in.
            self.streams
                .fetch_update(Relaxed, Relaxed, |n| Some(n.saturating_sub(1)))
                .map_or(0, |n| n.saturating_sub(1))
        };
        self.streams_peak.fetch_max(now, Relaxed);
    }

    /// One queueing call of the title's, which began `start_us` after the
    /// origin and took `took_us`. Calls close together are one run: what
    /// one mixer block spent in our code.
    pub(crate) fn caller_at(&self, start_us: u64, took_us: u64) {
        self.call_at(start_us);
        self.caller_us.fetch_add(took_us, Relaxed);
        self.caller_longest_us.fetch_max(took_us, Relaxed);
        let before = if start_us.saturating_sub(self.run_end_us.load(Relaxed)) < RUN_GAP_US {
            self.run_us.load(Relaxed)
        } else {
            0
        };
        let run = before + took_us;
        self.run_us.store(run, Relaxed);
        self.run_end_us.store(start_us + took_us, Relaxed);
        self.run_longest_us.fetch_max(run, Relaxed);
        // Counted once, by the call that takes the run past a block.
        if before < BLOCK_US && run >= BLOCK_US {
            self.runs_over_block.fetch_add(1, Relaxed);
        }
    }

    pub(crate) fn caller(&self, start: Instant) {
        let took = us(start.elapsed());
        self.caller_at(us(start.saturating_duration_since(origin())).max(1), took);
    }

    /// A call of the title's audio thread at `now_us`: how long it was
    /// away says whether the mixer keeps its blocks' pace.
    pub(crate) fn call_at(&self, now_us: u64) {
        let last = self.call_us.swap(now_us, Relaxed);
        let gap = now_us.saturating_sub(last);
        if last == 0 || gap >= AWAY_US {
            return;
        }
        self.call_gap_longest_us.fetch_max(gap, Relaxed);
        if gap >= LATE_US {
            self.late_calls.fetch_add(1, Relaxed);
        }
    }

    /// The title asked for a result that was still being decoded.
    pub(crate) fn not_ready(&self) {
        self.not_ready.fetch_add(1, Relaxed);
    }

    /// A result became ready this long after its command was queued.
    pub(crate) fn waited(&self, wait: Duration) {
        self.wait_longest_us.fetch_max(us(wait), Relaxed);
    }

    /// An XMA decoding pass that began `start_us` after the origin.
    pub(crate) fn pass_at(&self, start_us: u64, took_us: u64, voices: u64) {
        let last = self.pass_start_us.swap(start_us, Relaxed);
        // The first pass has none before it; nor has one after a line with
        // no voices, when the thread only waited.
        if self.passes.fetch_add(1, Relaxed) > 0 || last > 0 {
            self.gap_longest_us.fetch_max(start_us.saturating_sub(last), Relaxed);
        }
        self.pass_longest_us.fetch_max(took_us, Relaxed);
        if took_us >= SLOW_PASS_US {
            self.slow_passes.fetch_add(1, Relaxed);
        }
        self.voices_peak.fetch_max(voices, Relaxed);
    }

    pub(crate) fn pass(&self, start: Instant, voices: u64) {
        self.pass_at(us(start.saturating_duration_since(origin())), us(start.elapsed()), voices);
    }

    /// An XMA output block written to a ring.
    pub(crate) fn block(&self) {
        self.blocks.fetch_add(1, Relaxed);
    }

    /// The title took blocks from an XMA ring; `emptied` when that left the
    /// ring empty although its stream had more to decode: the title wanted
    /// more than the decoder had written.
    pub(crate) fn take(&self, emptied: bool) {
        self.call_at(now_us().max(1));
        self.takes.fetch_add(1, Relaxed);
        self.emptied.fetch_add(u64::from(emptied), Relaxed);
    }

    /// Reads the period's counters and starts the next period.
    pub(crate) fn drain(&self) -> Numbers {
        let take = |a: &AtomicU64| a.swap(0, Relaxed);
        // The peaks start again from what is active now.
        let streams_peak = self.streams_peak.swap(self.streams.load(Relaxed), Relaxed);
        Numbers {
            packets: take(&self.packets),
            decode_us: take(&self.decode_us),
            decode_longest_us: take(&self.decode_longest_us),
            streams_peak,
            caller_us: take(&self.caller_us),
            caller_longest_us: take(&self.caller_longest_us),
            run_longest_us: take(&self.run_longest_us),
            runs_over_block: take(&self.runs_over_block),
            call_gap_longest_us: take(&self.call_gap_longest_us),
            late_calls: take(&self.late_calls),
            not_ready: take(&self.not_ready),
            wait_longest_us: take(&self.wait_longest_us),
            passes: take(&self.passes),
            slow_passes: take(&self.slow_passes),
            pass_longest_us: take(&self.pass_longest_us),
            gap_longest_us: take(&self.gap_longest_us),
            voices_peak: take(&self.voices_peak),
            blocks: take(&self.blocks),
            takes: take(&self.takes),
            emptied: take(&self.emptied),
        }
    }
}

impl Numbers {
    /// Nothing decoded and nothing asked for: no line.
    pub(crate) fn idle(&self) -> bool {
        self.packets == 0 && self.caller_us == 0 && self.blocks == 0 && self.takes == 0
    }

    /// The line. `workers` is how many threads of ours decode Opus (0: the
    /// title's own thread does), `raised` whether the XMA decoder's thread
    /// runs at a raised priority.
    pub(crate) fn line(&self, seconds: f64, workers: usize, raised: &str) -> String {
        let ms = |us: u64| us as f64 / 1000.0;
        let share = if seconds > 0.0 { self.caller_us as f64 / (seconds * 10_000.0) } else { 0.0 };
        let decode = if workers == 0 {
            "now off, the title's thread decodes".to_string()
        } else {
            format!("now on, {workers} threads")
        };
        format!(
            "audio over the last {seconds:.1} s: Opus {} packets decoded in {:.0} ms (longest {:.2} ms), \
             up to {} streams; the title's thread spent {:.0} ms in its queueing calls ({share:.1} % of \
             the time; longest call {:.2} ms; longest run of calls {:.2} ms; {} runs of a {:.1} ms block \
             or more) (VICTORY_APU_DECODE=alt decodes on threads of ours: {decode}); the title's \
             audio calls came {} times {:.0} ms or more apart (longest {:.1} ms; pauses of 1 s or more \
             not counted); results not ready when asked for {}, longest wait for one {:.2} ms; \
             XMA {} passes for up to {} voices, {} of 4 ms or more (longest {:.2} ms), \
             longest gap between passes {:.2} ms, {} blocks written, \
             {} takes, {} of them emptied a ring whose stream had more (VICTORY_AUDIO_MMCSS=1 \
             or VICTORY_AUDIO_PRIORITY=alt raise the decoder's thread: now {raised})",
            self.packets,
            ms(self.decode_us),
            ms(self.decode_longest_us),
            self.streams_peak,
            ms(self.caller_us),
            ms(self.caller_longest_us),
            ms(self.run_longest_us),
            self.runs_over_block,
            ms(BLOCK_US),
            self.late_calls,
            ms(LATE_US),
            ms(self.call_gap_longest_us),
            self.not_ready,
            ms(self.wait_longest_us),
            self.passes,
            self.voices_peak,
            self.slow_passes,
            ms(self.pass_longest_us),
            ms(self.gap_longest_us),
            self.blocks,
            self.takes,
            self.emptied,
        )
    }
}

/// Starts the thread that writes the lines, when the title first uses a
/// decoder (once; never with `VICTORY_AUDIO_STATS=0`).
pub(crate) fn start() {
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        if std::env::var_os("VICTORY_AUDIO_STATS").is_some_and(|v| v == "0") {
            return;
        }
        let _ = std::thread::Builder::new().name("victory audio meter".into()).spawn(|| {
            let mut last = Instant::now();
            loop {
                std::thread::sleep(Duration::from_micros(PERIOD_US));
                let numbers = meter().drain();
                let seconds = last.elapsed().as_secs_f64();
                last = Instant::now();
                if !numbers.idle() {
                    let raised = super::decoder_raise().name();
                    tracing::info!("{}", numbers.line(seconds, super::opus_workers(), raised));
                }
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calls_close_together_are_one_run_counted_once_past_a_block() {
        let m = Meter::default();
        // Four calls of 2 ms back to back: one run of 8 ms.
        for i in 0..4 {
            m.caller_at(10_000 + i * 2_050, 2_000);
        }
        // The mixer sleeps, then a short block.
        m.caller_at(30_000, 300);
        m.caller_at(30_400, 300);
        let n = m.drain();
        assert_eq!(n.caller_us, 8_600);
        assert_eq!(n.caller_longest_us, 2_000);
        assert_eq!(n.run_longest_us, 8_000);
        assert_eq!(n.runs_over_block, 1);
        // The next period starts from nothing.
        assert_eq!(m.drain(), Numbers::default());
    }

    #[test]
    fn calls_far_apart_are_late_unless_nothing_played() {
        let m = Meter::default();
        // Blocks at their pace, a stall of 45 ms, a pause of 3 s, a stall
        // of 32 ms.
        for at in [1_000, 6_300, 11_600, 56_600, 61_900, 3_061_900, 3_093_900] {
            m.call_at(at);
        }
        let n = m.drain();
        assert_eq!((n.late_calls, n.call_gap_longest_us), (2, 45_000));
        // Opus queueing calls and XMA takes are both the title's calls.
        m.caller_at(3_193_900, 10);
        assert_eq!(m.drain().late_calls, 1);
    }

    #[test]
    fn a_single_long_call_is_a_run_over_a_block() {
        let m = Meter::default();
        m.caller_at(1_000, BLOCK_US);
        m.caller_at(50_000, BLOCK_US - 1);
        let n = m.drain();
        assert_eq!((n.runs_over_block, n.run_longest_us), (1, BLOCK_US));
    }

    #[test]
    fn passes_count_their_gaps_and_the_slow_ones() {
        let m = Meter::default();
        m.pass_at(1_000, 500, 3);
        m.pass_at(3_500, 4_200, 7);
        m.pass_at(12_000, 100, 2);
        m.block();
        m.take(false);
        m.take(true);
        let n = m.drain();
        assert_eq!((n.passes, n.slow_passes, n.voices_peak), (3, 1, 7));
        assert_eq!((n.pass_longest_us, n.gap_longest_us), (4_200, 8_500));
        assert_eq!((n.blocks, n.takes, n.emptied), (1, 2, 1));
        // The gap to the first pass of the next period still counts.
        m.pass_at(40_000, 100, 1);
        assert_eq!(m.drain().gap_longest_us, 28_000);
    }

    #[test]
    fn streams_peak_carries_what_is_still_active() {
        let m = Meter::default();
        m.stream(true);
        m.stream(true);
        m.stream(false);
        assert_eq!(m.drain().streams_peak, 2);
        // One is still active in the next period.
        assert_eq!(m.drain().streams_peak, 1);
        m.stream(false);
        m.stream(false);
        m.stream(true);
        assert_eq!(m.drain().streams_peak, 1);
    }

    #[test]
    fn the_line_names_the_numbers_and_the_switches() {
        let n = Numbers {
            packets: 1200,
            decode_us: 2_100_000,
            decode_longest_us: 950,
            streams_peak: 48,
            caller_us: 2_150_000,
            caller_longest_us: 1_200,
            run_longest_us: 6_100,
            runs_over_block: 37,
            late_calls: 9,
            call_gap_longest_us: 61_400,
            passes: 2400,
            slow_passes: 3,
            pass_longest_us: 7_200,
            gap_longest_us: 9_100,
            voices_peak: 12,
            blocks: 5000,
            takes: 2400,
            emptied: 14,
            ..Numbers::default()
        };
        assert!(!n.idle());
        assert!(Numbers::default().idle());
        let line = n.line(5.0, 0, "off");
        for part in [
            "Opus 1200 packets decoded in 2100 ms (longest 0.95 ms), up to 48 streams",
            "the title's thread spent 2150 ms in its queueing calls (43.0 % of the time",
            "longest run of calls 6.10 ms; 37 runs of a 5.3 ms block or more",
            "now off, the title's thread decodes",
            "the title's audio calls came 9 times 32 ms or more apart (longest 61.4 ms; pauses",
            "XMA 2400 passes for up to 12 voices, 3 of 4 ms or more (longest 7.20 ms)",
            "longest gap between passes 9.10 ms, 5000 blocks written, 2400 takes, 14 of them emptied",
            "raise the decoder's thread: now off)",
        ] {
            assert!(line.contains(part), "{part:?} not in {line:?}");
        }
        assert!(n.line(5.0, 4, "time critical").contains("now on, 4 threads"));
    }
}
