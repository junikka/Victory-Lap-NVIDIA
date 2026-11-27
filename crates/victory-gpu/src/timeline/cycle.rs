//! Diagnostics: where every frame's time goes on the title's side.
//! `VICTORY_FRAME_TIMELINE=1` prints, every five seconds, the parts of the
//! frame (present to the first list call, to the last submission, to the
//! present call, the present itself) as medians and 95th percentiles, what
//! the presenting thread did between the last submission and the present
//! (waits by the function waited with and the call of ours it was made in,
//! processor time, neither), the waits that took most, and for each of the
//! title's threads its processor time, its waits and what is left: time it
//! could have run and did not, or waits we do not wrap.
//! `VICTORY_FRAME_TIMELINE=<file>.csv` (or `csv`: `victory-frames.csv` where
//! the program was started) also writes a line for each frame, 3,000 at
//! most.
//!
//! A last line puts the frame in one table of means: the title's side by
//! its parts, and the card by timestamps ahead of and behind every
//! submission of ours (which carry the title's lists): running them, with
//! nothing sent, and with the next sent and not begun.
//!
//! All times are the performance counter's ([`ticks`]). Every thread writes
//! what it does into a place of its own, one half for even frames and one
//! for odd; a thread of ours reads the half of the frame that ended. Without
//! the variable nothing is kept and nothing is wrapped.

// The arithmetic and its tests are built with the tests too; what records
// and reports is used by a diagnostics build alone.
#![cfg_attr(not(feature = "diagnostics"), allow(dead_code, unused_imports))]

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::OnceLock;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize};

#[cfg(feature = "diagnostics")]
use crate::host::{card_calibration, ring_stamp, ring_taken, timeline_ready};

/// A build without `diagnostics` (the tests) has no stamps of the card's.
#[cfg(not(feature = "diagnostics"))]
mod no_card {
    pub(super) fn card_calibration() -> Option<(u64, u64, u64)> {
        None
    }
    pub(super) fn ring_stamp(_at: usize) -> Option<(u64, u64)> {
        None
    }
    pub(super) fn ring_taken() -> usize {
        0
    }
    pub(super) fn timeline_ready() {}
}
#[cfg(not(feature = "diagnostics"))]
use no_card::{card_calibration, ring_stamp, ring_taken, timeline_ready};

#[link(name = "kernel32")]
unsafe extern "system" {
    pub(super) fn QueryThreadCycleTime(thread: *mut std::ffi::c_void, cycles: *mut u64) -> i32;
}

#[link(name = "ntdll")]
unsafe extern "system" {
    pub(super) fn NtQueryInformationThread(
        thread: *mut std::ffi::c_void,
        class: i32,
        out: *mut std::ffi::c_void,
        len: u32,
        returned: *mut u32,
    ) -> i32;
}

/// [`ticks`] per second.
pub(super) fn frequency() -> u64 {
    static TICKS: OnceLock<u64> = OnceLock::new();
    *TICKS.get_or_init(|| {
        let mut ticks = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceFrequency(&mut ticks) };
        ticks.max(1) as u64
    })
}

/// The performance counter, as the few-frame timeline reads it.
pub(super) fn ticks() -> u64 {
    let mut ticks = 0;
    // SAFETY: a valid out pointer.
    unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut ticks) };
    ticks as u64
}

/// This thread's number.
fn thread() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

/// The span around the title's submission of lists.
pub(super) const EXECUTE: &str = "the title's ExecuteCommandLists";
/// What the submitting thread's own sends are kept as.
pub(super) const SENT: &str = "lists sent to the card";

/// Threads with a place, and what a thread keeps of a frame.
const THREADS: usize = 128;
const RING: usize = 256;
/// The title's threads whose processor time is read at a submission.
const WATCHED: usize = 48;
/// Lines of the per-frame file.
const ROWS: usize = 3000;

/// Threads the title started, as the loader and the kernel module tell.
static TITLE_IDS: [AtomicU32; 256] = [const { AtomicU32::new(0) }; 256];

/// The thread `id` is one the title started. Atomics alone: it is called
/// before the runtime has set the thread up.
pub(crate) fn title_thread_started(id: u32) {
    if id == 0 {
        return;
    }
    for place in &TITLE_IDS {
        let there = place.load(Relaxed);
        if there == id || (there == 0 && place.compare_exchange(0, id, Relaxed, Relaxed).is_ok()) {
            return;
        }
    }
}

/// The threads the title started.
pub(super) fn title_ids() -> Vec<u32> {
    TITLE_IDS.iter().map(|p| p.load(Relaxed)).take_while(|id| *id != 0).collect()
}

/// What `VICTORY_FRAME_TIMELINE` asks for: nothing, the lines, or the lines
/// and a file (`csv`: `victory-frames.csv` where the program was started).
fn parse(value: &str) -> Option<Option<std::path::PathBuf>> {
    let value = value.trim();
    if value.is_empty() || value == "0" {
        return None;
    }
    if value.eq_ignore_ascii_case("csv") {
        return Some(Some("victory-frames.csv".into()));
    }
    if value.to_ascii_lowercase().ends_with(".csv") {
        return Some(Some(value.into()));
    }
    Some(None)
}

/// The variable, read once.
fn asked() -> Option<&'static Option<std::path::PathBuf>> {
    static ASKED: OnceLock<Option<Option<std::path::PathBuf>>> = OnceLock::new();
    ASKED
        .get_or_init(|| {
            let value = std::env::var("VICTORY_FRAME_TIMELINE").ok()?;
            parse(&value)
        })
        .as_ref()
}

/// Whether the frame timeline was asked for.
pub(super) fn wanted() -> bool {
    asked().is_some()
}

static ON: AtomicBool = AtomicBool::new(false);

/// Whether frames are kept now.
#[inline(always)]
pub(crate) fn on() -> bool {
    ON.load(Relaxed)
}

/// Something a thread did, or waited for.
struct Entry {
    t0: AtomicU64,
    t1: AtomicU64,
    name: AtomicUsize,
    /// The name's length; [`SPAN`] set for a span of ours.
    name_len: AtomicUsize,
    within: AtomicUsize,
    within_len: AtomicUsize,
}

const SPAN: usize = 1 << 62;

/// What a thread did in the frames of one parity.
struct Half {
    /// The frame it holds, plus one.
    frame: AtomicU64,
    len: AtomicU32,
    lost: AtomicU32,
    /// The first list call's beginning and the last one's end.
    first: AtomicU64,
    last: AtomicU64,
    calls: AtomicU64,
    inside: AtomicU64,
    /// Waits made inside list calls.
    inside_waits: AtomicU64,
    /// Waits too short for an entry.
    short: AtomicU64,
    ring: [Entry; RING],
}

struct Slot {
    id: AtomicU32,
    /// All it waited, ever.
    waited: AtomicU64,
    halves: [Half; 2],
}

static SLOTS: AtomicPtr<Slot> = AtomicPtr::new(std::ptr::null_mut());
/// The frame under way.
static FRAME: AtomicU64 = AtomicU64::new(0);

/// A frame's ends.
struct Bound {
    call: AtomicU64,
    ret: AtomicU64,
    thread: AtomicU32,
}

static BOUNDS: [Bound; 8] = [const { Bound { call: AtomicU64::new(0), ret: AtomicU64::new(0), thread: AtomicU32::new(0) } }; 8];
static CALLED: AtomicU64 = AtomicU64::new(0);

/// The processor cycles of the watched threads at a moment.
struct Snap {
    t: AtomicU64,
    cycles: [AtomicU64; WATCHED],
}

const fn snap() -> Snap {
    Snap { t: AtomicU64::new(0), cycles: [const { AtomicU64::new(0) }; WATCHED] }
}

/// At the last submission and at the present call, by the frame's parity.
static AT_SUBMISSION: [Snap; 2] = [snap(), snap()];
static AT_PRESENT: [Snap; 2] = [snap(), snap()];
static WATCH_IDS: [AtomicU32; WATCHED] = [const { AtomicU32::new(0) }; WATCHED];
static WATCH_HANDLES: [AtomicUsize; WATCHED] = [const { AtomicUsize::new(0) }; WATCHED];
static WATCH_LEN: AtomicUsize = AtomicUsize::new(0);
/// Ticks spent reading them, on the title's threads.
static SNAP_TICKS: AtomicU64 = AtomicU64::new(0);

fn take_snap(snap: &Snap, t: u64) {
    let n = WATCH_LEN.load(Acquire).min(WATCHED);
    for i in 0..n {
        let mut cycles = 0u64;
        // SAFETY: a handle the worker opened and keeps; a valid out pointer.
        unsafe { QueryThreadCycleTime(WATCH_HANDLES[i].load(Relaxed) as _, &mut cycles) };
        snap.cycles[i].store(cycles, Relaxed);
    }
    snap.t.store(t, Release);
    SNAP_TICKS.fetch_add(ticks().saturating_sub(t), Relaxed);
}

thread_local! {
    /// This thread's place, plus one (0: not asked yet; `usize::MAX`: none).
    static MINE: Cell<usize> = const { Cell::new(0) };
    /// The innermost span of ours this thread is in.
    static WITHIN: Cell<&'static str> = const { Cell::new("") };
    /// The waits since the last list call returned: when each began, how
    /// long it took.
    static RECENT: Cell<([(u64, u64); 4], usize)> = const { Cell::new(([(0, 0); 4], 0)) };
}

fn slot() -> Option<&'static Slot> {
    let slots = SLOTS.load(Acquire);
    if slots.is_null() {
        return None;
    }
    let at = MINE.with(|mine| {
        if mine.get() == 0 {
            let id = thread();
            let free = (0..THREADS).find(|&i| {
                // SAFETY: within the places made in `start`, which stay.
                unsafe { &*slots.add(i) }.id.compare_exchange(0, id, Relaxed, Relaxed).is_ok()
            });
            mine.set(free.map_or(usize::MAX, |i| i + 1));
        }
        mine.get()
    });
    // SAFETY: as above.
    (at != usize::MAX).then(|| unsafe { &*slots.add(at - 1) })
}

/// The half of `slot` for the frame under way, emptied if it held another.
fn half(slot: &'static Slot) -> &'static Half {
    let frame = FRAME.load(Relaxed);
    let half = &slot.halves[(frame & 1) as usize];
    if half.frame.load(Relaxed) != frame + 1 {
        half.frame.store(frame + 1, Release);
        half.len.store(0, Release);
        for value in [&half.first, &half.last, &half.calls, &half.inside, &half.inside_waits, &half.short] {
            value.store(0, Relaxed);
        }
        half.lost.store(0, Relaxed);
    }
    half
}

fn push(half: &Half, t0: u64, t1: u64, name: &'static str, within: &'static str, span: bool) {
    let at = half.len.load(Relaxed) as usize;
    if at >= RING {
        half.lost.store(half.lost.load(Relaxed) + 1, Relaxed);
        return;
    }
    let entry = &half.ring[at];
    entry.t0.store(t0, Relaxed);
    entry.t1.store(t1, Relaxed);
    entry.name.store(name.as_ptr() as usize, Relaxed);
    entry.name_len.store(name.len() | if span { SPAN } else { 0 }, Relaxed);
    entry.within.store(within.as_ptr() as usize, Relaxed);
    entry.within_len.store(within.len(), Relaxed);
    half.len.store(at as u32 + 1, Release);
}

/// Shorter waits and spans are added up, not kept (20 microseconds).
fn shortest() -> u64 {
    frequency() / 50_000
}

/// This thread waited with `name` since `t0`.
pub(super) fn waited(name: &'static str, t0: u64) {
    let Some(slot) = slot() else {
        return;
    };
    let t1 = ticks();
    let took = t1.saturating_sub(t0);
    slot.waited.store(slot.waited.load(Relaxed) + took, Relaxed);
    RECENT.with(|recent| {
        let (mut waits, n) = recent.get();
        if n < waits.len() {
            waits[n] = (t0, took);
            recent.set((waits, n + 1));
        }
    });
    let half = half(slot);
    if took < shortest() {
        half.short.store(half.short.load(Relaxed) + took, Relaxed);
    } else {
        push(half, t0, t1, name, WITHIN.with(Cell::get), false);
    }
}

/// A list call of the title's that took `inside` ticks has returned.
pub(super) fn called(inside: u64) {
    let Some(slot) = slot() else {
        return;
    };
    let t1 = ticks();
    let t0 = t1.saturating_sub(inside);
    let half = half(slot);
    if half.first.load(Relaxed) == 0 {
        half.first.store(t0, Relaxed);
    }
    half.last.store(t1, Relaxed);
    half.calls.store(half.calls.load(Relaxed) + 1, Relaxed);
    half.inside.store(half.inside.load(Relaxed) + inside, Relaxed);
    RECENT.with(|recent| {
        let (waits, n) = recent.get();
        if n > 0 {
            let waited: u64 = waits[..n].iter().filter(|w| w.0 >= t0).map(|w| w.1).sum();
            half.inside_waits.store(half.inside_waits.load(Relaxed) + waited.min(inside), Relaxed);
            recent.set((waits, 0));
        }
    });
}

/// A span of ours named `name` begins on this thread: what it was in.
pub(super) fn enter(name: &'static str) -> &'static str {
    WITHIN.with(|within| within.replace(name))
}

/// The span `name` that began at `t0` ends at `t1`; the thread is back in
/// `before`.
pub(super) fn left(name: &'static str, t0: u64, t1: u64, before: &'static str) {
    WITHIN.with(|within| within.set(before));
    let submission = name == EXECUTE;
    if t1.saturating_sub(t0) >= shortest() || submission {
        if let Some(slot) = slot() {
            push(half(slot), t0, t1, name, before, true);
        }
    }
    if submission {
        // The watched threads' cycles, at most every half millisecond.
        let snap = &AT_SUBMISSION[(FRAME.load(Relaxed) & 1) as usize];
        if t1.saturating_sub(snap.t.load(Relaxed)) > frequency() / 2000 {
            take_snap(snap, t1);
        }
    }
}

/// Something of ours, or a call of the kernel's, that began at `t0` ends
/// now, on this thread.
pub(super) fn did(name: &'static str, t0: u64) {
    let t1 = ticks();
    if t1.saturating_sub(t0) >= shortest()
        && let Some(slot) = slot()
    {
        push(half(slot), t0, t1, name, WITHIN.with(Cell::get), true);
    }
}

/// When each stamped submission went out, by its place in the ring of the
/// card's stamps (`ring_stamp`).
static SENT_AT: [AtomicU64; 1024] = [const { AtomicU64::new(0) }; 1024];

/// The submission stamped at `stamp` went out at `t0`.
pub(super) fn card_sent(stamp: usize, t0: u64) {
    SENT_AT[stamp % SENT_AT.len()].store(t0, Relaxed);
}

/// What the card did, by the timestamps around every submission: ticks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Card {
    /// Running submissions.
    ran: u64,
    /// Between two submissions, the next one not yet sent: nothing of ours
    /// to run, or what goes out without stamps (the present's copy).
    nothing_sent: u64,
    /// Between two submissions, the next one sent and not yet begun.
    sent_not_begun: u64,
    submissions: u64,
    /// When the last one ended.
    ended: u64,
}

impl Card {
    /// A submission sent at `sent` ran from `from` to `to`.
    pub(super) fn add(&mut self, sent: u64, from: u64, to: u64) {
        self.submissions += 1;
        self.ran += to.saturating_sub(from);
        if self.ended != 0 && from > self.ended {
            let waiting = sent.clamp(self.ended, from) - self.ended;
            self.nothing_sent += waiting;
            self.sent_not_begun += from - self.ended - waiting;
        }
        self.ended = self.ended.max(to);
    }
}

/// The card's clock against the PC's.
#[derive(Clone, Copy)]
struct CardClock {
    card: u64,
    pc: u64,
    /// The PC's ticks a tick of the card's.
    rate: f64,
}

impl CardClock {
    fn now() -> Option<CardClock> {
        let (card, pc, frequency) = card_calibration()?;
        Some(CardClock { card, pc, rate: frequency_f64() / frequency as f64 })
    }

    /// The PC's counter when the card's stood at `ticks`.
    fn pc(self, ticks: u64) -> u64 {
        (self.pc as f64 + (ticks as f64 - self.card as f64) * self.rate).max(0.0) as u64
    }
}

fn frequency_f64() -> f64 {
    frequency() as f64
}

/// Reads the stamps of the submissions from `next` on that the card has
/// run, into `card`; a submission the card has not stamped after 900 more
/// went out is passed over.
fn read_card(next: &mut usize, clock: Option<CardClock>, card: &mut Card, unread: &mut u64) {
    let Some(clock) = clock else {
        return;
    };
    let taken = ring_taken();
    while *next < taken {
        match ring_stamp(*next) {
            Some((from, to)) => card.add(SENT_AT[*next % SENT_AT.len()].load(Relaxed), clock.pc(from), clock.pc(to)),
            None if taken - *next > 900 => *unread += 1,
            None => break,
        }
        *next += 1;
    }
}

/// The title calls its present.
pub(crate) fn present_begins() {
    if !on() {
        return;
    }
    let t = ticks();
    take_snap(&AT_PRESENT[(FRAME.load(Relaxed) & 1) as usize], t);
    CALLED.store(t, Relaxed);
}

/// The title's present came back: a frame ends.
pub(crate) fn presented() {
    if !wanted() {
        return;
    }
    start();
    let t = ticks();
    let frame = FRAME.load(Relaxed);
    let bound = &BOUNDS[(frame & 7) as usize];
    bound.call.store(CALLED.load(Relaxed), Relaxed);
    bound.ret.store(t, Relaxed);
    bound.thread.store(thread(), Relaxed);
    FRAME.store(frame + 1, Release);
}

/// Makes the places and the thread that reads them: once, if asked for.
pub(super) fn start() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    if !wanted() {
        return;
    }
    ONCE.call_once(|| {
        make_room();
        if std::thread::Builder::new().name("victory-cycle".into()).spawn(work).is_err() {
            tracing::warn!("no thread for the frame timeline");
            return;
        }
        ON.store(true, Release);
        tracing::info!("the frame timeline is on (VICTORY_FRAME_TIMELINE)");
    });
}

fn make_room() {
    if !SLOTS.load(Acquire).is_null() {
        return;
    }
    let layout = std::alloc::Layout::array::<Slot>(THREADS).expect("a small number");
    // SAFETY: not of no size; zeroes are valid atomics. Kept for good.
    SLOTS.store(unsafe { std::alloc::alloc_zeroed(layout) }.cast(), Release);
}

/// Something a thread did in a frame, as the worker read it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Item {
    t0: u64,
    t1: u64,
    name: &'static str,
    within: &'static str,
    span: bool,
}

/// A thread's frame.
#[derive(Clone, Debug, Default)]
pub(super) struct ThreadFrame {
    id: u32,
    title: bool,
    first: u64,
    last: u64,
    calls: u64,
    inside: u64,
    inside_waits: u64,
    short: u64,
    lost: u32,
    items: Vec<Item>,
}

/// Reads the half of `frame` of every thread.
fn gather(frame: u64, titles: &[u32]) -> Vec<ThreadFrame> {
    let slots = SLOTS.load(Acquire);
    let mut out = Vec::new();
    if slots.is_null() {
        return out;
    }
    for i in 0..THREADS {
        // SAFETY: within the places, which stay.
        let slot = unsafe { &*slots.add(i) };
        let id = slot.id.load(Relaxed);
        if id == 0 {
            break;
        }
        let half = &slot.halves[(frame & 1) as usize];
        if half.frame.load(Acquire) != frame + 1 {
            continue;
        }
        let n = (half.len.load(Acquire) as usize).min(RING);
        let raw: Vec<[u64; 6]> = half.ring[..n]
            .iter()
            .map(|e| {
                [
                    e.t0.load(Relaxed),
                    e.t1.load(Relaxed),
                    e.name.load(Relaxed) as u64,
                    e.name_len.load(Relaxed) as u64,
                    e.within.load(Relaxed) as u64,
                    e.within_len.load(Relaxed) as u64,
                ]
            })
            .collect();
        let mut thread = ThreadFrame {
            id,
            title: titles.contains(&id),
            first: half.first.load(Relaxed),
            last: half.last.load(Relaxed),
            calls: half.calls.load(Relaxed),
            inside: half.inside.load(Relaxed),
            inside_waits: half.inside_waits.load(Relaxed),
            short: half.short.load(Relaxed),
            lost: half.lost.load(Relaxed),
            items: Vec::with_capacity(n),
        };
        // The half was not emptied for a later frame meanwhile: the first
        // `n` entries are whole, and their names are the statics they were
        // written from.
        if half.frame.load(Acquire) != frame + 1 {
            continue;
        }
        for e in raw {
            let text = |at: u64, len: u64| -> &'static str {
                let len = len as usize & !SPAN;
                if at == 0 || len == 0 || len > 256 {
                    return "";
                }
                // SAFETY: the address and length of a `&'static str`, read
                // whole (see above).
                unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(at as *const u8, len)) }
            };
            thread.items.push(Item {
                t0: e[0],
                t1: e[1],
                name: text(e[2], e[3]),
                within: text(e[4], e[5]),
                span: e[3] as usize & SPAN != 0,
            });
        }
        out.push(thread);
    }
    out
}

/// The parts of a frame, in ticks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Segments {
    frame: u64,
    /// The last present's return to the first list call.
    to_first: u64,
    /// The first list call to the last submission's return.
    recording: u64,
    /// The last submission's return to the present call.
    tail: u64,
    present: u64,
    submissions: u32,
    /// When the last submission returned (0: none).
    last_submission: u64,
    /// How long list calls went on after it.
    lists_after: u64,
    /// How long our own sends to the card went on after it.
    sends_after: u64,
}

/// The parts of the frame from `prev` (the present before returned) over
/// `call` (the present is called) to `ret` (it returns).
pub(super) fn segments(prev: u64, call: u64, ret: u64, threads: &[ThreadFrame]) -> Segments {
    /// When the spans named `name` ended, those that ended in the frame
    /// before its present was called.
    fn ends<'a>(threads: &'a [ThreadFrame], name: &'a str, prev: u64, call: u64) -> impl Iterator<Item = u64> + 'a {
        threads
            .iter()
            .flat_map(|t| t.items.iter())
            .filter(move |i| i.span && i.name == name && i.t1 > prev && i.t1 <= call)
            .map(|i| i.t1)
    }
    let first = threads.iter().filter(|t| t.calls > 0).map(|t| t.first.max(prev)).min().unwrap_or(0);
    let last_list = threads.iter().filter(|t| t.calls > 0).map(|t| t.last.min(ret)).max().unwrap_or(0);
    let submissions = ends(threads, EXECUTE, prev, call).count() as u32;
    let last_submission = ends(threads, EXECUTE, prev, call).max().unwrap_or(0);
    let last_send = ends(threads, SENT, prev, call).max().unwrap_or(0);
    let mut s = Segments {
        frame: ret.saturating_sub(prev),
        present: ret.saturating_sub(call),
        submissions,
        last_submission,
        ..Segments::default()
    };
    if first != 0 {
        s.to_first = first.min(call) - prev;
    }
    if last_submission != 0 {
        s.recording = last_submission.saturating_sub(if first != 0 { first } else { prev });
        s.tail = call - last_submission;
        s.lists_after = last_list.saturating_sub(last_submission);
        s.sends_after = last_send.saturating_sub(last_submission);
    } else if first != 0 {
        s.recording = last_list.min(call).saturating_sub(first);
    }
    s
}

/// How much of `a0..a1` lies in `b0..b1`.
pub(super) fn overlap(a0: u64, a1: u64, b0: u64, b1: u64) -> u64 {
    a1.min(b1).saturating_sub(a0.max(b0))
}

/// The `p`th percentile (0..=100) of `sorted`.
pub(super) fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let at = ((sorted.len() - 1) as f64 * p / 100.0).round() as usize;
    sorted[at.min(sorted.len() - 1)]
}

fn ms(ticks: u64) -> f64 {
    ticks as f64 * 1000.0 / frequency() as f64
}

/// `p50/p95` of `values`.
fn pair(values: &mut [f64]) -> String {
    values.sort_by(f64::total_cmp);
    format!("{:.1}/{:.1}", percentile(values, 50.0), percentile(values, 95.0))
}

/// Whose wait, with what, inside which call of ours.
type WaitKey = (bool, &'static str, &'static str);

/// What a thread added up to in a window.
#[derive(Default)]
struct ThreadSum {
    first: Vec<f64>,
    last: Vec<f64>,
    calls: u64,
    inside: u64,
    inside_waits: u64,
    tail_cpu: f64,
}

/// Five seconds of frames.
#[derive(Default)]
struct Window {
    began: u64,
    frames: u64,
    skipped: u64,
    lost: u64,
    frame: Vec<f64>,
    to_first: Vec<f64>,
    recording: Vec<f64>,
    tail: Vec<f64>,
    present: Vec<f64>,
    lists_after: Vec<f64>,
    sends_after: Vec<f64>,
    submissions: u64,
    /// The presenting thread between the last submission and the present.
    tail_waited: Vec<f64>,
    tail_cpu: Vec<f64>,
    tail_neither: Vec<f64>,
    /// All the title's threads' processor time there.
    tail_title_cpu: Vec<f64>,
    presenter: u32,
    waits: HashMap<WaitKey, (u64, u64)>,
    tail_waits: HashMap<WaitKey, u64>,
    tail_spans: HashMap<&'static str, u64>,
    threads: HashMap<u32, ThreadSum>,
}

/// A frame's numbers beyond its segments, in milliseconds.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Tail {
    waited: f64,
    title_waited: f64,
    cpu: Option<f64>,
    title_cpu: Option<f64>,
}

impl Window {
    /// Adds the frame `s` of `threads`, which began at `prev` and was
    /// presented by `presenter`; `cpu`:
    /// the processor time (ms) of the title's threads in its tail.
    fn add(&mut self, s: &Segments, prev: u64, threads: &[ThreadFrame], presenter: u32, cpu: Option<&HashMap<u32, f64>>) -> Tail {
        self.frames += 1;
        self.presenter = presenter;
        self.frame.push(ms(s.frame));
        self.to_first.push(ms(s.to_first));
        self.recording.push(ms(s.recording));
        self.tail.push(ms(s.tail));
        self.present.push(ms(s.present));
        self.lists_after.push(ms(s.lists_after));
        self.sends_after.push(ms(s.sends_after));
        self.submissions += u64::from(s.submissions);
        let (from, to) = (s.last_submission, s.last_submission + s.tail);
        let mut tail = Tail::default();
        for thread in threads {
            self.lost += u64::from(thread.lost);
            if thread.calls > 0 {
                let sum = self.threads.entry(thread.id).or_default();
                sum.first.push(ms(thread.first.saturating_sub(prev)));
                sum.last.push(ms(thread.last.saturating_sub(prev)));
                sum.calls += thread.calls;
                sum.inside += thread.inside;
                sum.inside_waits += thread.inside_waits;
            }
            if thread.short > 0 {
                let all = self.waits.entry((thread.title, "short waits", "")).or_default();
                all.0 += thread.short;
            }
            for item in &thread.items {
                let inside = if s.last_submission != 0 { overlap(item.t0, item.t1, from, to) } else { 0 };
                if item.span {
                    if inside > 0 && thread.id == presenter {
                        *self.tail_spans.entry(item.name).or_default() += inside;
                    }
                    continue;
                }
                let key = (thread.title, item.name, item.within);
                let all = self.waits.entry(key).or_default();
                all.0 += item.t1.saturating_sub(item.t0);
                all.1 += 1;
                if inside > 0 && (thread.title || thread.id == presenter) {
                    *self.tail_waits.entry(key).or_default() += inside;
                    if thread.id == presenter {
                        tail.waited += ms(inside);
                    }
                    if thread.title {
                        tail.title_waited += ms(inside);
                    }
                }
            }
        }
        if s.last_submission != 0 {
            self.tail_waited.push(tail.waited);
            if let Some(cpu) = cpu {
                let own = cpu.get(&presenter).copied();
                let all: f64 = cpu.values().sum();
                for (id, spent) in cpu {
                    self.threads.entry(*id).or_default().tail_cpu += spent;
                }
                tail.cpu = own;
                tail.title_cpu = Some(all);
                self.tail_title_cpu.push(all);
                if let Some(own) = own {
                    self.tail_cpu.push(own);
                    self.tail_neither.push((ms(s.tail) - tail.waited - own).max(0.0));
                }
            }
        }
        tail
    }
}

/// The `n` largest of `totals`, as milliseconds a frame.
fn top<K: Copy>(totals: impl Iterator<Item = (K, u64)>, n: usize, frames: u64, label: impl Fn(K) -> String) -> String {
    let mut all: Vec<(K, u64)> = totals.collect();
    all.sort_by(|a, b| b.1.cmp(&a.1));
    let mut text = String::new();
    for (key, ticks) in all.into_iter().take(n) {
        let _ = write!(text, "{}{} {:.2}", if text.is_empty() { "" } else { "; " }, label(key), ms(ticks) / frames.max(1) as f64);
    }
    text
}

fn wait_label((title, name, within): WaitKey) -> String {
    let whose = if title { "title" } else { "ours" };
    if within.is_empty() { format!("{whose}: {name}") } else { format!("{whose}: {name} in '{within}'") }
}

/// A thread of the title's the worker follows.
struct Followed {
    id: u32,
    handle: usize,
    mask: u64,
    priority: i32,
    /// Its cycles and its waits when the window began.
    cycles: u64,
    waited: u64,
}

/// A thread's affinity mask and priority.
pub(super) fn mask_and_priority(handle: usize) -> (u64, i32) {
    let mut info = [0u64; 6];
    // SAFETY: writes the 48 bytes of `THREAD_BASIC_INFORMATION` (class 0).
    let status = unsafe { NtQueryInformationThread(handle as _, 0, info.as_mut_ptr().cast(), 48, std::ptr::null_mut()) };
    if status != 0 {
        return (0, 0);
    }
    (info[4], info[5] as u32 as i32)
}

fn cycles_of(handle: usize) -> u64 {
    let mut cycles = 0u64;
    // SAFETY: a handle we opened; a valid out pointer.
    unsafe { QueryThreadCycleTime(handle as _, &mut cycles) };
    cycles
}

/// Opens the title's threads not followed yet; the first [`WATCHED`] of
/// them are read at submissions too.
fn follow(threads: &mut Vec<Followed>) {
    use windows_sys::Win32::System::Threading::{OpenThread, THREAD_QUERY_INFORMATION, THREAD_QUERY_LIMITED_INFORMATION};
    for id in title_ids() {
        if threads.iter().any(|t| t.id == id) {
            continue;
        }
        // SAFETY: a thread of this process, by its number; kept open.
        let handle = unsafe { OpenThread(THREAD_QUERY_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION, 0, id) } as usize;
        if handle == 0 {
            continue;
        }
        let at = WATCH_LEN.load(Relaxed);
        if at < WATCHED {
            WATCH_IDS[at].store(id, Relaxed);
            WATCH_HANDLES[at].store(handle, Relaxed);
            WATCH_LEN.store(at + 1, Release);
        }
        let (mask, priority) = mask_and_priority(handle);
        threads.push(Followed { id, handle, mask, priority, cycles: cycles_of(handle), waited: waited_of(id) });
    }
}

/// All the thread `id` waited so far, as its place has it.
fn waited_of(id: u32) -> u64 {
    let slots = SLOTS.load(Acquire);
    if slots.is_null() {
        return 0;
    }
    (0..THREADS)
        // SAFETY: within the places, which stay.
        .map(|i| unsafe { &*slots.add(i) })
        .take_while(|s| s.id.load(Relaxed) != 0)
        .find(|s| s.id.load(Relaxed) == id)
        .map_or(0, |s| s.waited.load(Relaxed))
}

/// The processor's clock against the performance counter.
#[derive(Clone, Copy)]
struct Clock {
    cycles: u64,
    ticks: u64,
}

impl Clock {
    fn now() -> Clock {
        // SAFETY: reads the time stamp counter.
        Clock { cycles: unsafe { std::arch::x86_64::_rdtsc() }, ticks: ticks() }
    }

    /// Milliseconds a cycle, measured from `self` to `later`.
    fn ms_a_cycle(self, later: Clock) -> f64 {
        let cycles = later.cycles.saturating_sub(self.cycles).max(1) as f64;
        ms(later.ticks.saturating_sub(self.ticks)) / cycles
    }
}

/// The processor time (ms) each watched thread spent between the frame's
/// last submission and its present call, if both were read.
fn tail_cpu(frame: u64, s: &Segments, call: u64, ms_a_cycle: f64) -> Option<HashMap<u32, f64>> {
    let (before, after) = (&AT_SUBMISSION[(frame & 1) as usize], &AT_PRESENT[(frame & 1) as usize]);
    let (t0, t1) = (before.t.load(Acquire), after.t.load(Acquire));
    // The submission's reading may be up to half a millisecond early.
    if s.last_submission == 0 || t1 != call || t0 > s.last_submission || s.last_submission - t0 > frequency() / 1000 {
        return None;
    }
    let n = WATCH_LEN.load(Acquire).min(WATCHED);
    let mut out = HashMap::new();
    for i in 0..n {
        let (c0, c1) = (before.cycles[i].load(Relaxed), after.cycles[i].load(Relaxed));
        if c0 != 0 && c1 >= c0 {
            out.insert(WATCH_IDS[i].load(Relaxed), (c1 - c0) as f64 * ms_a_cycle);
        }
    }
    Some(out)
}

fn report(w: &mut Window, threads: &mut [Followed], ms_a_cycle: f64, wall: f64) {
    let frames = w.frames.max(1);
    let a_frame = |ticks: u64| ms(ticks) / frames as f64;
    tracing::info!(
        frames = w.frames,
        frame = %pair(&mut w.frame),
        present_to_first_list = %pair(&mut w.to_first),
        first_list_to_last_submission = %pair(&mut w.recording),
        last_submission_to_present = %pair(&mut w.tail),
        in_present = %pair(&mut w.present),
        submissions = %format!("{:.1}", w.submissions as f64 / frames as f64),
        lists_after_last_submission = %pair(&mut w.lists_after),
        sends_after_last_submission = %pair(&mut w.sends_after),
        lost = w.lost,
        skipped = w.skipped,
        tool_us = %format!("{:.0}", a_frame(SNAP_TICKS.swap(0, Relaxed)) * 1000.0),
        "the frame timeline, ms as p50/p95 (VICTORY_FRAME_TIMELINE)"
    );
    tracing::info!(
        thread = w.presenter,
        frames = w.tail_waited.len(),
        waited = %pair(&mut w.tail_waited),
        cpu = %pair(&mut w.tail_cpu),
        neither = %pair(&mut w.tail_neither),
        title_threads_cpu = %pair(&mut w.tail_title_cpu),
        waits = %top(w.tail_waits.iter().map(|(k, v)| (*k, *v)), 5, frames, wait_label),
        spans = %top(w.tail_spans.iter().map(|(k, v)| (*k, *v)), 5, frames, |name: &'static str| name.to_string()),
        "the frame timeline: the last submission to the present, the presenting thread, ms as p50/p95 and ms a frame (VICTORY_FRAME_TIMELINE)"
    );
    tracing::info!(
        title = %top(w.waits.iter().filter(|(k, _)| k.0).map(|(k, v)| (*k, v.0)), 5, frames, wait_label),
        ours = %top(w.waits.iter().filter(|(k, _)| !k.0).map(|(k, v)| (*k, v.0)), 5, frames, wait_label),
        "the frame timeline: the waits that took most, ms a frame (VICTORY_FRAME_TIMELINE)"
    );
    // The title's threads: processor time, waits, and what is left of the
    // wall time: able to run and not running, or waits we do not wrap.
    let mut lines: Vec<(f64, String)> = Vec::new();
    for thread in threads.iter_mut() {
        let (cycles, waited) = (cycles_of(thread.handle), waited_of(thread.id));
        let cpu = cycles.saturating_sub(thread.cycles) as f64 * ms_a_cycle;
        let waits = ms(waited.saturating_sub(thread.waited));
        (thread.cycles, thread.waited) = (cycles, waited);
        let (mask, priority) = mask_and_priority(thread.handle);
        if mask != 0 {
            (thread.mask, thread.priority) = (mask, priority);
        }
        let mut text = format!(
            "{} mask {:#x} pri {}: cpu {:.1} waited {:.1} neither {:.1}",
            thread.id,
            thread.mask,
            thread.priority,
            cpu / frames as f64,
            waits / frames as f64,
            (wall - cpu - waits).max(0.0) / frames as f64
        );
        if let Some(sum) = w.threads.get_mut(&thread.id) {
            let _ = write!(text, " tail-cpu {:.1}", sum.tail_cpu / frames as f64);
            if sum.calls > 0 {
                let _ = write!(
                    text,
                    " lists {} first {} last {} inside {:.1} (waits {:.1})",
                    sum.calls / frames,
                    pair(&mut sum.first),
                    pair(&mut sum.last),
                    a_frame(sum.inside),
                    a_frame(sum.inside_waits)
                );
            }
        }
        lines.push((cpu, text));
    }
    lines.sort_by(|a, b| b.0.total_cmp(&a.0));
    let known = lines.len();
    let text: Vec<String> = lines.into_iter().take(12).map(|l| l.1).collect();
    tracing::info!(
        known,
        wall_ms_a_frame = %format!("{:.1}", wall / frames as f64),
        threads = %text.join(" | "),
        "the frame timeline: the title's threads by processor time, ms a frame (VICTORY_FRAME_TIMELINE)"
    );
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

/// The frame as one table, in means (they add up): the title's side by its
/// parts, and the card by the timestamps around every submission.
fn table(w: &Window, card: &Card, unread: u64, wall: f64) {
    let frames = w.frames.max(1) as f64;
    let of_card = |ticks: u64| format!("{:.1}", ms(ticks) / frames);
    let stamped = ms(card.ran + card.nothing_sent + card.sent_not_begun);
    tracing::info!(
        frames = w.frames,
        frame = %format!("{:.1}", mean(&w.frame)),
        game_present_to_first_list = %format!("{:.1}", mean(&w.to_first)),
        game_first_list_to_last_submission = %format!("{:.1}", mean(&w.recording)),
        game_last_submission_to_present = %format!("{:.1}", mean(&w.tail)),
        game_in_present = %format!("{:.1}", mean(&w.present)),
        card_ran_submissions = %of_card(card.ran),
        card_nothing_sent = %of_card(card.nothing_sent),
        card_sent_not_begun = %of_card(card.sent_not_begun),
        card_submissions = %format!("{:.1}", card.submissions as f64 / frames),
        card_not_covered = %format!("{:.1}", (wall - stamped).max(0.0) / frames),
        card_unread = unread,
        "the frame timeline: the frame on the title's side and on the card, mean ms a frame (VICTORY_FRAME_TIMELINE)"
    );
}

/// A line of the per-frame file.
fn row(frame: u64, since: f64, s: &Segments, tail: &Tail, threads: &[ThreadFrame], prev: u64) -> String {
    let mut text = format!(
        "{frame},{since:.1},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.3},{:.3},{:.3},{:.3},{},{},",
        ms(s.frame),
        ms(s.to_first),
        ms(s.recording),
        ms(s.tail),
        ms(s.present),
        s.submissions,
        ms(s.lists_after),
        ms(s.sends_after),
        tail.waited,
        tail.title_waited,
        tail.cpu.map_or(String::new(), |c| format!("{c:.3}")),
        tail.title_cpu.map_or(String::new(), |c| format!("{c:.3}")),
    );
    for thread in threads.iter().filter(|t| t.calls > 0) {
        let waited: u64 = thread.items.iter().filter(|i| !i.span).map(|i| i.t1.saturating_sub(i.t0)).sum();
        let _ = write!(
            text,
            "{}:{:.2}:{:.2}:{}:{:.2}:{:.2} ",
            thread.id,
            ms(thread.first.saturating_sub(prev)),
            ms(thread.last.saturating_sub(prev)),
            thread.calls,
            ms(thread.inside),
            ms(waited + thread.short)
        );
    }
    text.truncate(text.trim_end().len());
    text
}

const HEADER: &str = "frame,ms_since_start,frame_ms,present_to_first_list_ms,first_list_to_last_submission_ms,last_submission_to_present_ms,in_present_ms,submissions,lists_after_last_submission_ms,sends_after_last_submission_ms,tail_presenter_waited_ms,tail_title_waited_ms,tail_presenter_cpu_ms,tail_title_cpu_ms,threads(id:first_ms:last_ms:list_calls:inside_ms:waited_ms)";

/// Reads each frame that ended and prints the window every five seconds.
fn work() {
    use std::io::Write as _;
    let mut file = asked().and_then(|a| a.as_ref()).and_then(|path| match std::fs::File::create(path) {
        Ok(file) => {
            let mut file = std::io::BufWriter::new(file);
            let _ = writeln!(file, "{HEADER}");
            tracing::info!(file = %path.display(), "the frame timeline's file (VICTORY_FRAME_TIMELINE=csv)");
            Some(file)
        }
        Err(error) => {
            tracing::warn!(%error, file = %path.display(), "the frame timeline's file could not be made");
            None
        }
    });
    let started = Clock::now();
    let mut clock = started;
    std::thread::sleep(std::time::Duration::from_millis(50));
    let mut ms_a_cycle = started.ms_a_cycle(Clock::now());
    let mut threads: Vec<Followed> = Vec::new();
    let mut window = Window { began: ticks(), ..Window::default() };
    let mut next = FRAME.load(Acquire);
    let mut rows = 0;
    // The card: its stamps are made once frames go by, then read as it
    // writes them.
    let (mut card, mut card_next, mut card_clock, mut unread) = (Card::default(), 0usize, None, 0u64);
    let mut stamps_made = false;
    loop {
        std::thread::sleep(std::time::Duration::from_millis(3));
        let now = FRAME.load(Acquire);
        if !stamps_made && now > 2 {
            stamps_made = true;
            timeline_ready();
            card_clock = CardClock::now();
        }
        read_card(&mut card_next, card_clock, &mut card, &mut unread);
        while next < now {
            // The half of a frame is used again two frames later.
            if next == 0 || now - next > 1 {
                window.skipped += u64::from(next != 0);
                next += 1;
                continue;
            }
            let (bound, before) = (&BOUNDS[(next & 7) as usize], &BOUNDS[((next - 1) & 7) as usize]);
            let (prev, call, ret) = (before.ret.load(Relaxed), bound.call.load(Relaxed), bound.ret.load(Relaxed));
            let presenter = bound.thread.load(Relaxed);
            let titles = title_ids();
            let of_frame = gather(next, &titles);
            if prev != 0 && call >= prev && ret >= call {
                let s = segments(prev, call, ret, &of_frame);
                let cpu = tail_cpu(next, &s, call, ms_a_cycle);
                let tail = window.add(&s, prev, &of_frame, presenter, cpu.as_ref());
                if rows < ROWS
                    && let Some(file) = file.as_mut()
                {
                    let _ = writeln!(file, "{}", row(next, ms(ret.saturating_sub(started.ticks)), &s, &tail, &of_frame, prev));
                    rows += 1;
                }
            }
            next += 1;
        }
        let t = ticks();
        if t.saturating_sub(window.began) >= 5 * frequency() {
            let later = Clock::now();
            ms_a_cycle = clock.ms_a_cycle(later);
            clock = later;
            if window.frames > 0 {
                let wall = ms(t - window.began);
                report(&mut window, &mut threads, ms_a_cycle, wall);
                table(&window, &card, unread, wall);
            }
            // The card's numbers are a window's; when its last submission
            // ended stays.
            card = Card { ended: card.ended, ..Card::default() };
            unread = 0;
            if stamps_made {
                card_clock = CardClock::now().or(card_clock);
            }
            follow(&mut threads);
            if let Some(file) = file.as_mut() {
                let _ = file.flush();
            }
            window = Window { began: t, ..Window::default() };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(name: &'static str, t0: u64, t1: u64) -> Item {
        Item { t0, t1, name, within: "", span: true }
    }

    fn wait(name: &'static str, within: &'static str, t0: u64, t1: u64) -> Item {
        Item { t0, t1, name, within, span: false }
    }

    fn recorder(id: u32, first: u64, last: u64, items: Vec<Item>) -> ThreadFrame {
        ThreadFrame { id, title: true, first, last, calls: 10, inside: (last - first) / 2, items, ..ThreadFrame::default() }
    }

    #[test]
    fn a_frame_splits_at_the_first_list_call_the_last_submission_and_the_present() {
        let threads = vec![
            recorder(1, 1_020, 1_480, vec![span(EXECUTE, 1_300, 1_310), span(EXECUTE, 1_500, 1_510)]),
            recorder(2, 1_050, 1_530, vec![span(SENT, 1_400, 1_560)]),
        ];
        let s = segments(1_000, 1_790, 1_800, &threads);
        assert_eq!(
            s,
            Segments {
                frame: 800,
                to_first: 20,
                recording: 490,
                tail: 280,
                present: 10,
                submissions: 2,
                last_submission: 1_510,
                lists_after: 20,
                sends_after: 50,
            }
        );
        assert_eq!(s.to_first + s.recording + s.tail + s.present, s.frame);
    }

    #[test]
    fn a_submission_of_another_frame_or_inside_the_present_does_not_count() {
        let threads = vec![recorder(
            1,
            900,
            1_400,
            vec![span(EXECUTE, 950, 990), span(EXECUTE, 1_100, 1_200), span(EXECUTE, 1_795, 1_799)],
        )];
        let s = segments(1_000, 1_790, 1_800, &threads);
        assert_eq!((s.submissions, s.last_submission, s.tail), (1, 1_200, 590));
        // A list call that began before the last present returned counts from there.
        assert_eq!(s.to_first, 0);
        assert_eq!(s.recording, 200);
    }

    #[test]
    fn a_frame_without_a_submission_has_no_tail() {
        let s = segments(1_000, 1_790, 1_800, &[recorder(1, 1_100, 1_300, Vec::new())]);
        assert_eq!((s.to_first, s.recording, s.tail, s.submissions), (100, 200, 0, 0));
        let s = segments(1_000, 1_790, 1_800, &[]);
        assert_eq!((s.frame, s.to_first, s.recording, s.tail, s.present), (800, 0, 0, 0, 10));
    }

    #[test]
    fn the_cards_time_is_split_by_the_stamps_and_the_sends() {
        let mut card = Card::default();
        // Ran 100..160. The next was sent at 180 and ran 200..260: the card
        // had nothing sent for 20 and took 20 more to begin.
        card.add(90, 100, 160);
        card.add(180, 200, 260);
        // Sent while the one before still ran: it begins where that ended.
        card.add(210, 260, 300);
        // Sent before the last ended, begun later: all of the gap is "sent, not begun".
        card.add(290, 330, 350);
        assert_eq!(card, Card { ran: 180, nothing_sent: 20, sent_not_begun: 50, submissions: 4, ended: 350 });
        assert_eq!(card.ran + card.nothing_sent + card.sent_not_begun, 350 - 100);
        let clock = CardClock { card: 1_000, pc: 50_000, rate: 2.0 };
        assert_eq!((clock.pc(1_000), clock.pc(1_500), clock.pc(900)), (50_000, 51_000, 49_800));
        assert_eq!(mean(&[1.0, 2.0, 6.0]), 3.0);
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn overlaps_and_percentiles() {
        assert_eq!(overlap(10, 20, 15, 30), 5);
        assert_eq!(overlap(10, 20, 0, 100), 10);
        assert_eq!(overlap(10, 20, 20, 30), 0);
        assert_eq!(overlap(10, 20, 30, 40), 0);
        let values: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&values, 50.0), 51.0);
        assert_eq!(percentile(&values, 95.0), 95.0);
        assert_eq!(percentile(&values, 100.0), 100.0);
        assert_eq!(percentile(&[], 50.0), 0.0);
        assert_eq!(percentile(&[7.0], 95.0), 7.0);
    }

    #[test]
    fn the_tail_is_split_into_the_presenting_threads_waits_and_processor_time() {
        let f = frequency();
        // A frame of 80 ms: the last submission returns at 50 ms, the present is called at 78 ms.
        let at = |ms: u64| 1_000_000 + ms * f / 1000;
        let presenter = recorder(
            7,
            at(2),
            at(49),
            vec![
                span(EXECUTE, at(48), at(50)),
                // A wait in the tail; a wait before it does not count there.
                wait("WaitForSingleObject", "the title's frame wait", at(60), at(72)),
                wait("Sleep", "", at(10), at(20)),
                span("a fence set", at(74), at(76)),
            ],
        );
        let mut other = recorder(8, at(3), at(40), vec![wait("WaitOnAddress", "", at(45), at(55))]);
        other.short = f / 1000;
        let mut ours = recorder(9, at(3), at(40), vec![wait("SleepConditionVariableSRW", "", at(50), at(78))]);
        ours.title = false;
        let threads = vec![presenter, other, ours];
        let s = segments(at(0), at(78), at(80), &threads);
        let cpu: HashMap<u32, f64> = [(7, 6.0), (8, 20.0)].into_iter().collect();
        let mut w = Window::default();
        let tail = w.add(&s, at(0), &threads, 7, Some(&cpu));
        assert!((ms(s.tail) - 28.0).abs() < 0.01, "{s:?}");
        assert!((tail.waited - 12.0).abs() < 0.01, "{tail:?}");
        assert!((tail.title_waited - 17.0).abs() < 0.01, "{tail:?}");
        assert_eq!(tail.cpu, Some(6.0));
        assert_eq!(tail.title_cpu, Some(26.0));
        assert!((w.tail_neither[0] - 10.0).abs() < 0.01, "{:?}", w.tail_neither);
        // Our own thread's wait is in the totals, not in the tail's.
        assert!(w.waits.contains_key(&(false, "SleepConditionVariableSRW", "")));
        assert!(!w.tail_waits.contains_key(&(false, "SleepConditionVariableSRW", "")));
        let in_tail = w.tail_waits[&(true, "WaitForSingleObject", "the title's frame wait")];
        assert!((ms(in_tail) - 12.0).abs() < 0.01);
        assert!((ms(w.tail_spans["a fence set"]) - 2.0).abs() < 0.01);
        assert!((ms(w.waits[&(true, "short waits", "")].0) - 1.0).abs() < 0.01);
        let text = top(w.tail_waits.iter().map(|(k, v)| (*k, *v)), 5, 1, wait_label);
        assert!(text.starts_with("title: WaitForSingleObject in 'the title's frame wait' 12.00; title: WaitOnAddress 5.00"), "{text}");
        let line = row(3, 80.0, &s, &tail, &threads, at(0));
        assert!(line.starts_with("3,80.0,80.000,2.000,48.000,28.000,2.000,1,"), "{line}");
        assert_eq!(line.split(',').count(), HEADER.split(',').count());
    }

    #[test]
    fn the_variable_asks_for_lines_or_a_file() {
        assert_eq!(parse(""), None);
        assert_eq!(parse("0"), None);
        assert_eq!(parse("1"), Some(None));
        assert_eq!(parse("csv"), Some(Some("victory-frames.csv".into())));
        assert_eq!(parse(" D:/runs/a.CSV "), Some(Some("D:/runs/a.CSV".into())));
    }

    /// Real threads: what they write is read back as the frame's.
    #[test]
    fn what_threads_write_is_read_as_the_frames() {
        make_room();
        let frame = FRAME.load(Relaxed);
        let worker = std::thread::spawn(|| {
            let t0 = ticks();
            let before = enter(EXECUTE);
            waited("Sleep", t0);
            called(0);
            left(EXECUTE, t0, ticks() + shortest(), before);
            thread()
        });
        let id = worker.join().unwrap();
        let read = gather(frame, &[id]);
        let thread = read.iter().find(|t| t.id == id).expect("the thread's half");
        assert!(thread.title);
        assert_eq!(thread.calls, 1);
        assert!(thread.items.iter().any(|i| i.span && i.name == EXECUTE));
        // The wait was short or kept: one of the two.
        let kept = thread.items.iter().any(|i| !i.span && i.name == "Sleep" && i.within == EXECUTE);
        assert!(kept || thread.short > 0 || thread.items.len() == 1, "{thread:?}");
        assert!(gather(frame + 1, &[id]).iter().all(|t| t.id != id));
    }
}
