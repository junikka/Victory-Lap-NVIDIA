//! Pipelined submission (`VICTORY_SUBMIT_PIPELINE=1`; off unless set; only
//! with the relay, `crate::host::relay`): a submission sent ahead is
//! prepared in two stages, and the second of submission k runs beside the
//! first of submission k+1.
//!
//! Title memory is read at one point of a submission, under the submission
//! lock: the write watch is asked, the refreshes copy what was written into
//! the host copies, the lists' steps (the title's buffer copies and fills)
//! read and write it, and what goes up to copies in the card's memory is
//! taken out of their CPU sides (`send::take`). Nothing behind that point
//! reads title memory. So the cut is there:
//!
//! - **stage A**, the submitting thread, under the submission lock: steps
//!   given to the card, the asking, bytes under flights' writes, refreshes,
//!   steps, snapshots, the declared writes, the changed ranges taken
//!   (`Host::card_take`). Then the hand-over: the flight joins the line.
//! - **stage B**, one thread behind it, in the order of the hand-overs
//!   (`Host::stage_b`): the list that brings the card's copies up to date
//!   is recorded and sent, the compare is recorded (`Host::card_record`),
//!   the heaps' records are waited for, the slots that waited for a texture
//!   are made, the lists go to the queue, the fence is signalled, and the
//!   flight is marked as on the queue.
//!
//! What orders the two stages, each a rule that stood before in another
//! form:
//!
//! 1. **Sends in title order.** Stage B is one thread and takes its work in
//!    the order it was handed over ([`Behind`]). The number of the sends on
//!    the queue (`QUEUE_ORDER`) is no longer held from the taking to the
//!    send: stage A numbers the taking ([`next_send`]) and stage B says when
//!    it is on the queue ([`queue`]), so a fetch that goes to the queue in
//!    between is known to have run ahead of it. Whatever else must go to the
//!    drawing queue in order with the lists goes through stage B too, or
//!    waits for it: a texture's tile mappings ([`in_order`]), the present
//!    and a synchronous submission ([`wait`]).
//! 2. **A's reads against k's writes.** Submission k is in the air for k+1
//!    from the hand-over, not from its send: the flight joins the line,
//!    with what it declared written, while its lists are still stage B's
//!    (`Flight::sent` is false). The thread that lands flights waits for
//!    that ticket before it looks for a value of the fence ([`landing`]).
//! 3. **Per-copy state.** The dirty lists are stage A's (taken at the
//!    hand-over). The hashes "from before" and the compare's result area
//!    are stage B's alone: stage A makes a copy's hashes when its writes
//!    are first declared and looks at how full the result area is, and does
//!    both only when stage B is done with the submission before
//!    ([`wait`] with [`NEXT`]). So at most one submission is in stage B at
//!    a time.
//! 4. **Settles.** What settles flights (a synchronous submission, the
//!    sweep's moment with nothing in the air, a full result area) first
//!    waits until stage B has nothing left ([`wait`]).
//! 5. **The heaps' records** are begun in stage A and waited for in stage B
//!    (the round goes with the job), as `VICTORY_HEAP_SYNC_FIRST` already
//!    parts them.
//!
//! Not moved: the refreshes themselves. They read title memory, which the
//! title writes until it submits.

use std::collections::VecDeque;
use std::sync::Condvar;
use std::sync::atomic::AtomicU64;

use super::*;

/// What `VICTORY_SUBMIT_PIPELINE` asks for: 0 nothing (unset, or anything
/// not understood), 1 the two stages (`1`), 2 the two stages and the
/// up-copy check (`check`: as `VICTORY_UPCOPY_CHECK=1` beside it).
fn asked_of(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        Some("1") => 1,
        Some("check") => 2,
        _ => 0,
    }
}

fn asked() -> u8 {
    static ASKED: OnceLock<u8> = OnceLock::new();
    *ASKED.get_or_init(|| asked_of(std::env::var("VICTORY_SUBMIT_PIPELINE").ok().as_deref()))
}

/// Whether submissions sent ahead are prepared in two stages.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = asked() > 0 && relay();
        if asked() > 0 && !on {
            tracing::warn!("VICTORY_SUBMIT_PIPELINE is left off: it needs the relay (VICTORY_RELAY) as it is by default");
        }
        if on {
            tracing::info!(
                check = checked(),
                "submissions sent ahead are prepared in two stages, the second on a thread behind the submitting one \
                 (VICTORY_SUBMIT_PIPELINE=1)"
            );
        }
        on
    })
}

/// Whether the switch asks for the up-copy check too (`=check`).
pub(super) fn checked() -> bool {
    asked() == 2
}

/// Work for one thread, done in the order it was handed over.
pub(super) struct Behind<W> {
    state: Mutex<State<W>>,
    /// Told when work is handed over.
    handed: Condvar,
    /// Told when work is done.
    done: Condvar,
}

struct State<W> {
    waiting: VecDeque<W>,
    /// Pieces of work handed over, and those done.
    handed: u64,
    done: u64,
}

impl<W> Behind<W> {
    pub(super) const fn new() -> Behind<W> {
        Behind {
            state: Mutex::new(State {
                waiting: VecDeque::new(),
                handed: 0,
                done: 0,
            }),
            handed: Condvar::new(),
            done: Condvar::new(),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State<W>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `work` goes behind what was handed over before it. Returns how many
    /// pieces are not done with it.
    pub(super) fn hand_over(&self, work: W) -> u64 {
        let mut state = self.state();
        state.waiting.push_back(work);
        state.handed += 1;
        let depth = state.handed - state.done;
        drop(state);
        self.handed.notify_one();
        depth
    }

    /// The thread's: the next piece, in order; waits for one.
    pub(super) fn take(&self) -> W {
        let mut state = self.state();
        loop {
            if let Some(work) = state.waiting.pop_front() {
                return work;
            }
            state = self.handed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// The thread's: the piece it took is done.
    pub(super) fn finished(&self) {
        self.state().done += 1;
        self.done.notify_all();
    }

    /// Waits until everything handed over so far is done. Whether it stood.
    pub(super) fn drain(&self) -> bool {
        let mut state = self.state();
        let all = state.handed;
        let stood = state.done < all;
        while state.done < all {
            state = self.done.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        stood
    }

    /// Pieces handed over and not done.
    #[cfg(test)]
    fn depth(&self) -> u64 {
        let state = self.state();
        state.handed - state.done
    }

    /// `work` in order: `now` does it here if nothing handed over is left
    /// (and nothing is handed over while it runs); otherwise it goes behind
    /// what is. Whether it went behind.
    pub(super) fn in_order(&self, work: W, now: impl FnOnce(W)) -> bool {
        let mut state = self.state();
        if state.done == state.handed {
            now(work);
            return false;
        }
        state.waiting.push_back(work);
        state.handed += 1;
        drop(state);
        self.handed.notify_one();
        true
    }
}

/// A submission as stage A hands it over.
pub(super) struct Job {
    /// Its number among the submissions started: the flight's in the line.
    pub(super) number: u64,
    /// The value stage B signals on the host's fence behind it. Stage A
    /// gives the values out, under the submission lock, in the order of the
    /// hand-overs.
    pub(super) fence: u64,
    /// The number of its taking ([`next_send`]); 0 without copies in the
    /// card's memory.
    pub(super) sent_as: u64,
    /// What it took for copies in the card's memory.
    pub(super) card: Option<CardTaken>,
    pub(super) ahead: Ahead,
    /// The round of the heaps' records begun for it; waited for when dropped.
    pub(super) heaps: crate::descriptor::SyncedHeaps,
    pub(super) heaps_on: bool,
}

// SAFETY: host objects (lists that are closed and nobody records into, the
// copies' buffers) and the submission's own bytes; with one owner, which is
// stage A until the hand-over and stage B's thread from then on.
unsafe impl Send for Job {}

enum Work {
    Job(Box<Job>),
    /// A call that goes to the drawing queue in order with the lists.
    Call(Box<dyn FnOnce(&Host) + Send>),
}

static BEHIND: Behind<Work> = Behind::new();

thread_local! {
    /// Whether this is stage B's thread.
    static STAGE_B: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// For the tests: stage B takes it before each piece of work, so a test that
/// holds it sees a submission handed over and not sent.
#[cfg(test)]
pub(super) static HOLD: Mutex<()> = Mutex::new(());

fn work(host: &Host, work: Work) {
    match work {
        Work::Job(job) => host.stage_b(*job),
        Work::Call(call) => call(host),
    }
}

/// Stage B's thread.
fn thread() {
    crate::cores::keep_on_fast_cores("victory-submit-b");
    let Some(host) = host() else {
        return;
    };
    STAGE_B.with(|b| b.set(true));
    loop {
        let next = BEHIND.take();
        #[cfg(test)]
        let _held = HOLD.lock().unwrap_or_else(|e| e.into_inner());
        // Whoever waits for this thread would wait for ever.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(host, next))).is_err() {
            fail(format_args!("the thread behind the submitting one stopped (VICTORY_SUBMIT_PIPELINE=1)"));
        }
        BEHIND.finished();
    }
}

/// Whether stage B has a thread; started at the first hand-over.
fn threaded() -> bool {
    static MADE: OnceLock<bool> = OnceLock::new();
    *MADE.get_or_init(|| {
        let made = std::thread::Builder::new().name("victory-submit-b".into()).spawn(thread);
        if let Err(e) = &made {
            tracing::warn!(error = %e, "no thread behind the submitting one: stage B runs on the submitting thread (VICTORY_SUBMIT_PIPELINE=1)");
        }
        made.is_ok()
    })
}

/// The hand-over: `job` is stage B's from here on. Its flight is in the
/// line already.
pub(super) fn hand_over(host: &Host, job: Job) {
    STATS[HANDED].fetch_add(1, Ordering::Relaxed);
    if !threaded() {
        host.stage_b(job);
        return;
    }
    let depth = BEHIND.hand_over(Work::Job(Box::new(job)));
    STATS[DEPTH].fetch_add(depth, Ordering::Relaxed);
    MOST.fetch_max(depth, Ordering::Relaxed);
}

/// `call` goes to the host's drawing queue in order with the lists: at once
/// if stage B has nothing, else behind what it has.
pub(super) fn in_order(host: &Host, call: Box<dyn FnOnce(&Host) + Send>) {
    if !on() || !threaded() {
        call(host);
        return;
    }
    if BEHIND.in_order(Work::Call(call), |call| work(host, call)) {
        STATS[CALLS_BEHIND].fetch_add(1, Ordering::Relaxed);
    }
}

/// Why the submitting thread waits for stage B ([`wait`]): before the next
/// submission takes its ranges (rule 3); a settle of flights; the sweep; a
/// present.
pub(super) const NEXT: usize = 0;
pub(super) const SETTLE: usize = 1;
pub(super) const SWEEP: usize = 2;
pub(super) const PRESENT: usize = 3;

/// Waits until stage B has done everything handed over so far. Nothing if
/// the switch is off.
pub(super) fn wait(why: usize) {
    if !on() || STAGE_B.with(std::cell::Cell::get) {
        return;
    }
    STATS[ASKED].fetch_add(1, Ordering::Relaxed);
    let began = std::time::Instant::now();
    if BEHIND.drain() {
        STATS[STOOD + why.min(3)].fetch_add(1, Ordering::Relaxed);
        STATS[STOOD_US].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

/// With the switch: takings of changed ranges for copies in the card's
/// memory so far. `QUEUE_ORDER` then counts those whose list is on the
/// queue.
static TAKEN: AtomicU64 = AtomicU64::new(0);

/// The number of the next taking (stage A, or a synchronous submission).
pub(super) fn next_send() -> u64 {
    TAKEN.fetch_add(1, Ordering::AcqRel) + 1
}

/// `queued` once taking `sent_as` is on the queue: never backwards.
fn reached(queued: u64, sent_as: u64) -> u64 {
    queued.max(sent_as)
}

/// Taking `sent_as` goes to the queue: the order of the queue, held by the
/// caller until its list is there. A fetch sent to the queue before this
/// was taken ran ahead of the taking's bytes, one sent after it is let go
/// behind them.
pub(super) fn queue(sent_as: u64) -> std::sync::MutexGuard<'static, u64> {
    let mut order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
    *order = reached(*order, sent_as);
    order
}

/// What the thread that lands flights does about the first things of the
/// line.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Landing {
    /// That many, from the front, land now.
    Land(usize),
    /// The first is a flight on the queue that the card has not run: the
    /// value of the fence behind it is waited for.
    Card(u64),
    /// The first is a flight handed over and not on the queue yet: no value
    /// of the fence is behind it. Its ticket is waited for (stage B tells).
    Ticket,
}

/// Of the line's things in order (a flight: the value signalled on the
/// host's fence behind it, and whether it is on the queue; anything else
/// `None`), with the card at `completed`: what lands. As [`landable`], and
/// nothing from the first flight that is not on the queue: whatever the
/// fence says, the card has not run it.
pub(super) fn landing(line: &[Option<(u64, bool)>], completed: u64) -> Landing {
    let fences: Vec<Option<u64>> = line.iter().map(|f| f.map(|f| f.0)).collect();
    let unsent = line.iter().position(|f| f.is_some_and(|f| !f.1)).unwrap_or(line.len());
    let n = landable(&fences, completed).min(unsent);
    if n > 0 {
        Landing::Land(n)
    } else if unsent == 0 && !line.is_empty() {
        Landing::Ticket
    } else {
        Landing::Card(fences.first().copied().flatten().unwrap_or(0))
    }
}

const HANDED: usize = 0;
const A_US: usize = 1;
const B_US: usize = 2;
const BOTH_US: usize = 3;
const ASKED: usize = 4;
const STOOD_US: usize = 5;
/// Four places: by why ([`NEXT`]..).
const STOOD: usize = 6;
const SENDS: usize = 10;
const SENT_LISTS: usize = 11;
const SIGNALS: usize = 12;
const CALLS_BEHIND: usize = 13;
const DEPTH: usize = 14;

/// Since the last line: submissions handed over; microseconds of stage A
/// (from the submission lock to the hand-over), of stage B, and of both at
/// the same time; waits of the submitting thread for stage B asked for,
/// microseconds it stood in them, and those in which it stood by why; of
/// stage B's thread: calls of `ExecuteCommandLists`, the lists in them,
/// fence signals; other calls for the queue that went behind its work; the
/// sum of what was in stage B at each hand-over.
static STATS: [AtomicU64; 15] = [const { AtomicU64::new(0) }; 15];

/// Since the last line: the most in stage B at a hand-over.
static MOST: AtomicU64 = AtomicU64::new(0);

/// Whether a stage A and a stage B are under way, and since when both are
/// ([`now_us`]).
static BOTH: Mutex<(bool, bool, u64)> = Mutex::new((false, false, 0));

fn under_way(stage_b: bool, on: bool) {
    let mut both = BOTH.lock().unwrap_or_else(|e| e.into_inner());
    let now = now_us();
    if both.0 && both.1 {
        STATS[BOTH_US].fetch_add(now.saturating_sub(both.2), Ordering::Relaxed);
    }
    if stage_b {
        both.1 = on;
    } else {
        both.0 = on;
    }
    both.2 = now;
}

/// A stage under way: counted when it goes.
pub(super) struct Stage(bool, std::time::Instant);

impl Stage {
    /// Stage A of a submission sent ahead begins (it has the lock).
    pub(super) fn a() -> Stage {
        under_way(false, true);
        Stage(false, std::time::Instant::now())
    }

    /// Stage B of one begins.
    pub(super) fn b() -> Stage {
        under_way(true, true);
        Stage(true, std::time::Instant::now())
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        under_way(self.0, false);
        STATS[if self.0 { B_US } else { A_US }].fetch_add(self.1.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

/// A call of `ExecuteCommandLists` with `lists` lists was made, or
/// (`signal`) a fence signal: counted if stage B's thread made it, to be
/// read against the line of all sends (`crate::host::sent_lists`).
pub(super) fn sent_lists(lists: usize, signal: bool) {
    if !STAGE_B.with(std::cell::Cell::get) {
        return;
    }
    if signal {
        STATS[SIGNALS].fetch_add(1, Ordering::Relaxed);
    } else {
        STATS[SENDS].fetch_add(1, Ordering::Relaxed);
        STATS[SENT_LISTS].fetch_add(lists as u64, Ordering::Relaxed);
    }
}

/// The line's text for `s`, [`STATS`] per frame, and `most`.
fn line_text(s: &[f64], most: u64) -> String {
    let ms = |i: usize| s[i] / 1000.0;
    format!(
        "pipelined submission per frame (VICTORY_SUBMIT_PIPELINE=1): {:.1} submissions handed over; stage A {:.1} ms (the submitting \
         thread, from the submission lock to the hand-over, its waits for flights with it), stage B {:.1} ms (the thread behind it: \
         the list and the compare for copies in the card's memory, the heaps' records, the send, the signal), {:.1} ms of the two at \
         the same time; the submitting thread stood {:.1} ms waiting for stage B in {:.1} of {:.1} waits (before taking its ranges \
         {:.1}, a settle {:.1}, the sweep {:.1}, a present {:.1}); stage B made {:.1} calls of ExecuteCommandLists with {:.1} lists \
         and {:.1} fence signals, and {:.1} other calls for the queue went behind its work; in stage B at a hand-over {:.2} on \
         average, {most} at most",
        s[HANDED],
        ms(A_US),
        ms(B_US),
        ms(BOTH_US),
        ms(STOOD_US),
        s[STOOD..STOOD + 4].iter().sum::<f64>(),
        s[ASKED],
        s[STOOD + NEXT],
        s[STOOD + SETTLE],
        s[STOOD + SWEEP],
        s[STOOD + PRESENT],
        s[SENDS],
        s[SENT_LISTS],
        s[SIGNALS],
        s[CALLS_BEHIND],
        if s[HANDED] > 0.0 { s[DEPTH] / s[HANDED] } else { 0.0 },
    )
}

/// For the frame-rate line, which comes every second: a line every 5 s.
pub(super) fn line(frames: f64) {
    if !on() {
        return;
    }
    static SINCE: Mutex<(f64, Option<std::time::Instant>)> = Mutex::new((0.0, None));
    let frames = {
        let mut since = SINCE.lock().unwrap_or_else(|e| e.into_inner());
        since.0 += frames;
        if since.1.get_or_insert_with(std::time::Instant::now).elapsed() < std::time::Duration::from_secs(5) {
            return;
        }
        since.1 = Some(std::time::Instant::now());
        std::mem::take(&mut since.0).max(1.0)
    };
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!("{}", line_text(&s, MOST.swap(0, Ordering::Relaxed)));
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;

    use super::*;

    /// Unset, or anything but `1` and `check`: off. `check` asks for the
    /// up-copy check with it.
    #[test]
    fn the_switch_is_off_unless_set() {
        assert_eq!(asked_of(None), 0);
        for off in ["", "0", "2", "on", "true", "yes"] {
            assert_eq!(asked_of(Some(off)), 0, "{off:?}");
        }
        assert_eq!(asked_of(Some("1")), 1);
        assert_eq!(asked_of(Some(" 1 ")), 1);
        assert_eq!(asked_of(Some("check")), 2);
        // (The crate's own tests run without the relay, and so without this,
        // unless they ask.)
        if std::env::var_os("VICTORY_SUBMIT_PIPELINE").is_none() {
            assert!(!on() && !checked());
        }
    }

    /// A thread that does the work of `behind` as stage B's does: each piece
    /// goes to `done` once `go` lets it.
    fn worker(behind: &Arc<Behind<u32>>, go: mpsc::Receiver<()>, done: mpsc::Sender<u32>) {
        let behind = behind.clone();
        std::thread::spawn(move || {
            loop {
                let work = behind.take();
                if go.recv().is_err() {
                    return;
                }
                let _ = done.send(work);
                behind.finished();
            }
        });
    }

    /// Rule 1: what is handed over is done by one thread in that order, and a
    /// call for the queue goes behind what waits, or is made at once when
    /// nothing does.
    #[test]
    fn rule_1_work_is_done_in_the_order_of_the_hand_overs() {
        let behind = Arc::new(Behind::<u32>::new());
        let (go, gone) = mpsc::channel();
        let (done, did) = mpsc::channel();
        worker(&behind, gone, done);
        // Nothing waits: a call is made here, at once.
        let mut here = Vec::new();
        assert!(!behind.in_order(7, |call| here.push(call)));
        assert_eq!(here, [7]);
        for n in 0..100 {
            assert_eq!(behind.hand_over(n), u64::from(n) + 1);
        }
        // Something waits: the call goes behind it, and is not made here.
        assert!(behind.in_order(100, |_| panic!("made ahead of what was handed over")));
        for _ in 0..101 {
            go.send(()).unwrap();
        }
        behind.drain();
        let order: Vec<u32> = did.try_iter().collect();
        assert_eq!(order, (0..=100).collect::<Vec<u32>>());
        assert_eq!(behind.depth(), 0);
    }

    /// Rule 1: a taking counts as on the queue when stage B says so, not
    /// when stage A numbers it, and the count never goes back.
    #[test]
    fn rule_1_a_taking_is_on_the_queue_when_its_list_is() {
        // Takings 1 and 2 are numbered; a fetch that goes to the queue now
        // ran behind none of them.
        let mut queued = 0;
        let fetch_before = queued;
        queued = reached(queued, 1);
        let fetch_between = queued;
        queued = reached(queued, 2);
        assert_eq!((fetch_before, fetch_between, queued), (0, 1, 2));
        // A submission without anything to send says its number too; an
        // older number changes nothing.
        assert_eq!(reached(queued, 3), 3);
        assert_eq!(reached(5, 3), 5);
    }

    fn flight(number: u64, fence: u64, sent: bool, writes: (usize, usize)) -> Flight {
        Flight {
            number,
            fence,
            sent,
            lists: Vec::new(),
            order: Vec::new(),
            card: None,
            back: Vec::new(),
            own: None,
            texture_back_ranges: vec![writes],
        }
    }

    /// Rule 2: a submission handed over is in the air at once (what it may
    /// write stands in the way of the next submission's reads) although
    /// nothing of it is on the queue; and it lands only when stage B has
    /// sent it and the card has run it.
    #[test]
    fn rule_2_a_flight_is_in_the_air_from_its_hand_over() {
        let mut line = Line::new();
        // Flight 1 went the old way: on the queue, fence 4 behind it.
        line.join(flight(1, 4, true, (0x1000, 0x2000)), Vec::new(), 1);
        assert_eq!(line.target, 4);
        // Flight 2 is handed over: stage A gave it fence 5, nothing is sent.
        line.join(flight(2, 5, false, (0x8000, 0x9000)), vec![(0x10, crate::queue::Written::Value(1), 8)], 3);
        assert_eq!(line.written, [(0x1000, 0x2000), (0x8000, 0x9000)], "what it may write is in the air");
        assert!(touches(&line.written, 0x8800, 0x8810), "a read of it by the next submission is seen");
        assert_eq!(line.lists, 4);
        assert_eq!(line.target, 4, "no value of the fence is behind it yet");
        let things = |line: &Line| -> Vec<Option<(u64, bool)>> {
            line.line
                .iter()
                .map(|l| match l {
                    Late::Run(f) => Some((f.fence, f.sent)),
                    _ => None,
                })
                .collect()
        };
        // The card has run flight 1: it lands, and nothing of flight 2.
        assert_eq!(landing(&things(&line), 4), Landing::Land(1));
        // Even if the fence said more than was ever signalled.
        assert_eq!(landing(&things(&line), 99), Landing::Land(1));
        line.line.remove(0);
        // Flight 2 is first: its ticket is waited for, not the fence.
        assert_eq!(landing(&things(&line), 4), Landing::Ticket);
        assert_eq!(landing(&things(&line), 99), Landing::Ticket);
        // Stage B has sent it.
        assert!(line.sent(2, 5, None));
        assert_eq!(line.target, 5);
        assert_eq!(landing(&things(&line), 4), Landing::Card(5));
        // And the card has run it: the flight and the write behind it.
        assert_eq!(landing(&things(&line), 5), Landing::Land(2));
        // A flight that is not in the line cannot be marked.
        assert!(!line.sent(3, 6, None));
    }

    /// Rule 2: with every flight on the queue the landing is what it was.
    #[test]
    fn rule_2_flights_on_the_queue_land_as_before() {
        let line = [Some(5), None, None, Some(6), None];
        let sent: Vec<Option<(u64, bool)>> = line.iter().map(|f| f.map(|f| (f, true))).collect();
        for completed in [0, 4, 5, 6, u64::MAX] {
            let n = landable(&line, completed);
            let expected = if n > 0 { Landing::Land(n) } else { Landing::Card(5) };
            assert_eq!(landing(&sent, completed), expected, "the card at {completed}");
        }
        assert_eq!(landing(&[None, None], 0), Landing::Land(2));
        assert_eq!(landing(&[], 3), Landing::Card(0));
        // A signal with no flight ahead lands; behind a flight that is not
        // on the queue it waits with it.
        assert_eq!(landing(&[None, Some((9, false)), None], 9), Landing::Land(1));
        assert_eq!(landing(&[Some((9, false)), None], 9), Landing::Ticket);
    }

    /// Rule 3: stage A takes the next submission's ranges only when stage B
    /// is done with the one before, so the hashes "from before" and the
    /// compare's result area have one user at a time: at most one
    /// submission is in stage B.
    #[test]
    fn rule_3_one_submission_in_stage_b_at_a_time() {
        let behind = Arc::new(Behind::<u32>::new());
        let (go, gone) = mpsc::channel();
        let (done, did) = mpsc::channel();
        worker(&behind, gone, done);
        let mut most = 0;
        for k in 0..50u32 {
            // Stage A of k up to its ranges; stage B of k-1 is let run
            // meanwhile, and is waited for before the ranges are taken.
            if k > 0 {
                go.send(()).unwrap();
            }
            behind.drain();
            assert_eq!(did.try_iter().count(), usize::from(k > 0), "stage B is done with submission {}", k.wrapping_sub(1));
            most = most.max(behind.hand_over(k));
        }
        assert_eq!(most, 1);
        go.send(()).unwrap();
        behind.drain();
    }

    /// Rule 4: a settle waits until stage B has done everything handed over
    /// so far, and does not stand when there is nothing.
    #[test]
    fn rule_4_a_settle_waits_for_what_was_handed_over() {
        let behind = Arc::new(Behind::<u32>::new());
        let (go, gone) = mpsc::channel();
        let (done, did) = mpsc::channel();
        worker(&behind, gone, done);
        assert!(!behind.drain(), "nothing handed over: no wait");
        behind.hand_over(1);
        behind.hand_over(2);
        let settled = Arc::new(AtomicBool::new(false));
        let settling = {
            let (behind, settled) = (behind.clone(), settled.clone());
            std::thread::spawn(move || {
                let stood = behind.drain();
                settled.store(true, Ordering::Release);
                stood
            })
        };
        go.send(()).unwrap();
        assert_eq!(did.recv().unwrap(), 1);
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!settled.load(Ordering::Acquire), "one piece is still stage B's");
        go.send(()).unwrap();
        assert!(settling.join().unwrap(), "it stood");
        assert!(settled.load(Ordering::Acquire));
        assert_eq!(did.try_iter().collect::<Vec<u32>>(), [2]);
        assert!(!behind.drain());
    }

    /// Rule 5: a round of the heaps' records is begun by one thread (stage
    /// A) and waited for by another (stage B) while the next is begun; the
    /// rounds are made one after the other.
    #[test]
    fn rule_5_a_round_of_the_heaps_records_is_waited_for_by_another_thread() {
        let first = crate::descriptor::begin_sync_heaps();
        let second = crate::descriptor::begin_sync_heaps();
        let stage_b = std::thread::spawn(move || drop(first));
        stage_b.join().unwrap();
        let stage_b = std::thread::spawn(move || drop(second));
        stage_b.join().unwrap();
    }

    /// The times of the two stages and of both at once.
    #[test]
    fn the_line_says_what_was_counted() {
        let mut s = vec![0.0; STATS.len()];
        s[HANDED] = 17.0;
        s[A_US] = 41_000.0;
        s[B_US] = 9_500.0;
        s[BOTH_US] = 9_000.0;
        s[ASKED] = 18.0;
        s[STOOD_US] = 500.0;
        s[STOOD + NEXT] = 2.0;
        s[STOOD + PRESENT] = 1.0;
        s[SENDS] = 34.0;
        s[SENT_LISTS] = 150.0;
        s[SIGNALS] = 17.0;
        s[DEPTH] = 17.0;
        let text = line_text(&s, 1);
        for part in [
            "17.0 submissions handed over",
            "stage A 41.0 ms",
            "stage B 9.5 ms",
            "9.0 ms of the two at the same time",
            "stood 0.5 ms waiting for stage B in 3.0 of 18.0 waits",
            "before taking its ranges 2.0",
            "a present 1.0",
            "34.0 calls of ExecuteCommandLists with 150.0 lists and 17.0 fence signals",
            "1.00 on average, 1 at most",
        ] {
            assert!(text.contains(part), "{part:?} in {text:?}");
        }
    }
}
