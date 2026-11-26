//! The pass that uploads textures picked by index, on a thread of its own
//! (`VICTORY_HEAP_WORKER=1`; `alt`: twenty seconds with, twenty without, in
//! turn; `VICTORY_AB` can name it; off by default; needs flights,
//! `crate::host::late`). After the fork's worker (fxceless d4a528ba, measured
//! in f6a09d3b: 6.2-6.7 to 8.65-9.53 world frames a second at one place),
//! fitted to what this line already has.
//!
//! [`crate::draw::State::heap_textures`] runs on whichever thread of the
//! title records the frame's first shader that picks textures by index: it
//! asks what of each due texture is mapped, makes the host texture, makes a
//! staging buffer and records the copy, 5 to 12 ms of a frame in play
//! (session jon-fox-1126-0152: "the pass over the heap's views ... textures",
//! the part `VICTORY_HEAP_PASS_ASIDE` leaves on the recording thread) and up
//! to 250 ms while a backlog waits, and the threads waiting for that list
//! wait with it. With the switch the pass runs here, and the recording
//! thread only wakes it.
//!
//! What keeps the picture as it was:
//! - the uploads are recorded into host lists of ours, which the next
//!   submission of the title's runs ahead of its own lists ([`take_ready`]):
//!   texture states are settled by submission order, as between any two
//!   lists;
//! - a slot new to its texture shows it once the submission with its upload
//!   has finished, never before (`HostHeap::texture_slot_show_after`, what
//!   `VICTORY_SHOW_AFTER` does for the pass on a recording thread): until
//!   then it shows what it showed, or the neutral texture. (The fork gives
//!   the slot its view when that submission is sent; here lists sent before
//!   read slots by index while they fly, and a texture shown before its
//!   upload was what lost the device: `descriptor::host::show_new_later`.)
//! - the detiling done here is checked at submission as any texture's is:
//!   memory the title wrote in between is read again;
//! - a title texture the pass works on is held ([`crate::resource::hold`]),
//!   and so is the title's heap;
//! - the uploads before the title's decoration bake, which must be there
//!   for that draw, still run on the recording thread; the worker makes way
//!   for them ([`pass`]), and what it had recorded for slots the bake reads
//!   is taken over by the bake's own list (`State::flush_for_bake`).
// shortcut: what a submission shows for a texture is a frame or two later
// than when the recording thread did it all; with a backlog it is earlier,
// since the pass has the whole frame instead of a slice of it.

use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

use crate::descriptor::Heap;
use crate::host::HostList;
use crate::object::This;

/// Whether the pass runs on its own thread now.
pub(crate) fn on() -> bool {
    if !crate::host::late() {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        static BIT: OnceLock<Option<u32>> = OnceLock::new();
        if let Some(bit) = *BIT.get_or_init(|| crate::ab::bit("VICTORY_HEAP_WORKER")) {
            return crate::ab::on(bit);
        }
    }
    static MODE: OnceLock<u8> = OnceLock::new();
    match *MODE.get_or_init(|| crate::alt::mode(std::env::var("VICTORY_HEAP_WORKER").ok().as_deref(), 0)) {
        0 => false,
        1 => true,
        _ => {
            static LAST: AtomicU8 = AtomicU8::new(2);
            crate::alt::in_turn(&LAST, "VICTORY_HEAP_WORKER")
        }
    }
}

/// For the frame line, since the last one: nanoseconds the worker was at
/// work, of them detiling; its passes; lists given to submissions; textures
/// in those lists whose detiling was not done here; lists it recorded
/// nothing into; passes that made way for a recording thread.
pub(crate) static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// One upload list holds at most this much texture memory, or this many
/// milliseconds of work, so that its textures show soon: the pass then ends
/// and another begins with a list of its own.
const LIST_BYTES: u64 = 48 << 20;
const LIST_MS: u128 = 40;

/// Whether a list that holds `bytes` of uploads, `ms` after its pass began,
/// is handed over now.
fn list_full(bytes: u64, ms: u128) -> bool {
    bytes >= LIST_BYTES || (bytes > 0 && ms >= LIST_MS)
}

thread_local! {
    /// On the worker's thread, while it runs a pass: when the pass began,
    /// the bytes it has uploaded, and whether it was ended for its list
    /// being full.
    static SLICE: std::cell::Cell<Option<(std::time::Instant, u64, bool)>> = const { std::cell::Cell::new(None) };
}

/// Whether the calling thread is the worker, inside a pass.
pub(crate) fn here() -> bool {
    SLICE.get().is_some()
}

/// `bytes` more of texture memory in the worker's open list.
pub(crate) fn uploaded(bytes: u64) {
    if let Some((began, so_far, full)) = SLICE.get() {
        SLICE.set(Some((began, so_far + bytes, full)));
    }
}

/// Whether the worker's pass ends where it is: a recording thread wants the
/// pass ([`pass`]), or its list holds enough.
pub(crate) fn stop() -> bool {
    if URGENT.load(Relaxed) > 0 {
        return true;
    }
    let Some((began, bytes, _)) = SLICE.get() else {
        return false;
    };
    let full = list_full(bytes, began.elapsed().as_millis());
    if full {
        SLICE.set(Some((began, bytes, true)));
    }
    full
}

/// Holds a reference to a title heap.
struct HeldHeap(This);
// SAFETY: a reference counted title object, of which only the count and the
// parts behind locks are touched.
unsafe impl Send for HeldHeap {}

impl HeldHeap {
    fn new(heap: &Heap) -> Self {
        let this = std::ptr::from_ref(heap).cast();
        crate::object::add_ref(this);
        HeldHeap(this)
    }

    fn heap(&self) -> &Heap {
        crate::descriptor::heap(self.0)
    }
}

impl Drop for HeldHeap {
    fn drop(&mut self) {
        crate::object::release_boxed::<Heap>(self.0);
    }
}

/// A closed upload list.
struct Ready(HostList);
// SAFETY: D3D12 objects are free-threaded; the list is closed and touched
// by nobody until it is taken out.
unsafe impl Send for Ready {}

static READY: Mutex<Vec<Ready>> = Mutex::new(Vec::new());

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The upload lists ready now, for the submission about to be made, which
/// runs them ahead of its own lists. Called by the thread that submits.
pub(crate) fn take_ready() -> Vec<HostList> {
    // (Asked without the lock by every submission of a run without it.)
    if JOB.get().is_none() {
        return Vec::new();
    }
    let ready = std::mem::take(&mut *lock(&READY));
    STATS[3].fetch_add(ready.len() as u64, Relaxed);
    ready.into_iter().map(|r| r.0).collect()
}

/// `n` lists of ours put in front of a submission of `lists` lists that go
/// in `order`, another part beginning at each place of `parts`: they are
/// the lists from `lists` on, go first, and belong to the first part.
pub(crate) fn in_front(lists: usize, order: &mut Vec<usize>, parts: &[usize], n: usize) -> Vec<usize> {
    order.splice(0..0, lists..lists + n);
    parts.iter().map(|p| p + n).collect()
}

/// The heap the next pass is over, and what wakes the worker.
struct Job {
    heap: Mutex<Option<HeldHeap>>,
    wake: Condvar,
}

static JOB: OnceLock<Job> = OnceLock::new();
/// Held by whoever runs the pass, once there is a worker.
static PASS: Mutex<()> = Mutex::new(());
/// Recording threads that wait for [`PASS`].
static URGENT: AtomicU32 = AtomicU32::new(0);

/// A pass over `heap` is due (once a frame). False if there is no worker:
/// the caller runs the pass itself ([`pass`] first).
pub(crate) fn kick(heap: &Heap) -> bool {
    if !on() {
        return false;
    }
    static STARTED: OnceLock<bool> = OnceLock::new();
    let started = *STARTED.get_or_init(|| {
        JOB.get_or_init(|| Job { heap: Mutex::new(None), wake: Condvar::new() });
        match std::thread::Builder::new().name("victory-heap".into()).spawn(worker) {
            Ok(_) => {
                tracing::info!("textures picked by index are uploaded by a thread of their own (VICTORY_HEAP_WORKER)");
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no thread for textures picked by index");
                false
            }
        }
    });
    let Some(job) = JOB.get().filter(|_| started) else {
        return false;
    };
    *lock(&job.heap) = Some(HeldHeap::new(heap));
    job.wake.notify_one();
    true
}

/// For a recording thread that runs the pass itself (the uploads before a
/// bake draw, which cannot wait for a later submission; any pass while the
/// switch is off for a while): the worker stops where it is and the pass is
/// the caller's while it holds what comes back. `None` in a run that never
/// had a worker: nothing is waited for.
pub(crate) fn pass() -> Option<MutexGuard<'static, ()>> {
    JOB.get()?;
    if let Ok(pass) = PASS.try_lock() {
        return Some(pass);
    }
    URGENT.fetch_add(1, Relaxed);
    let pass = lock(&PASS);
    URGENT.fetch_sub(1, Relaxed);
    Some(pass)
}

fn worker() {
    let Some(job) = JOB.get() else {
        return;
    };
    // (The pass asks nothing of the state of a title's list.)
    let state = crate::draw::State::default();
    loop {
        let heap = {
            let mut next = lock(&job.heap);
            loop {
                match next.take() {
                    Some(heap) => break heap,
                    None => next = job.wake.wait(next).unwrap_or_else(|e| e.into_inner()),
                }
            }
        };
        loop {
            let more = {
                let _pass = lock(&PASS);
                let started = std::time::Instant::now();
                let more = one_pass(&state, heap.heap());
                STATS[0].fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
                more
            };
            // Whoever wants the pass now has it first.
            let mut made_way = false;
            while URGENT.load(Relaxed) > 0 {
                made_way = true;
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            STATS[6].fetch_add(u64::from(made_way), Relaxed);
            if !more || !on() {
                break;
            }
        }
    }
}

/// One pass over `heap` into a list of its own, which is handed to the next
/// submission. Whether the pass ended for its list being full: there is
/// more to do.
fn one_pass(state: &crate::draw::State, heap: &Heap) -> bool {
    let Some(host) = heap.host() else {
        return false;
    };
    let Some(list) = crate::list::fresh_list() else {
        return false;
    };
    STATS[2].fetch_add(1, Relaxed);
    SLICE.set(Some((std::time::Instant::now(), 0, false)));
    state.heap_textures_ahead(&list, heap, host);
    let full = SLICE.take().is_some_and(|(_, _, full)| full);
    let recorded = list.has_recorded_work();
    // Its textures are detiled here, ahead of the submission; not while a
    // recording thread waits for the pass.
    if recorded {
        let started = std::time::Instant::now();
        let left = list.refresh_textures_ahead(|| URGENT.load(Relaxed) > 0);
        STATS[1].fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
        STATS[4].fetch_add(left, Relaxed);
    }
    list.close();
    if recorded {
        lock(&READY).push(Ready(list));
    } else {
        // Nothing to run: reset (the pass may have asked it about
        // textures) and kept for another use.
        STATS[5].fetch_add(1, Relaxed);
        crate::list::recycle(vec![crate::list::Detached::ours(list)], &[0]);
    }
    full
}

/// The worker's line for the frame log, per frame; `None` in a run without
/// a worker.
#[cfg(feature = "diagnostics")]
pub(crate) fn line(frames: f64) -> Option<String> {
    JOB.get()?;
    let [busy, detiling, passes, lists, left, unused, made_way] =
        std::array::from_fn::<f64, 7, _>(|i| STATS[i].swap(0, Relaxed) as f64 / frames);
    Some(format!(
        "textures picked by index, their own thread (VICTORY_HEAP_WORKER is {}), per frame: {:.1} ms at work, of them {:.1} \
         detiling ahead; {passes:.1} passes, {lists:.1} lists given to the submissions, {unused:.1} with nothing to run, \
         {left:.1} textures left to the submission to detile, {made_way:.2} times made way for a recording thread",
        if on() { "on" } else { "off" },
        busy / 1e6,
        detiling / 1e6,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_list_is_handed_over_when_it_holds_enough_or_took_long() {
        assert!(!list_full(0, 0));
        // Nothing to show yet: however long the looking took.
        assert!(!list_full(0, 10_000));
        assert!(!list_full(1 << 20, 39));
        assert!(list_full(1 << 20, 40));
        assert!(list_full(48 << 20, 0));
    }

    #[test]
    fn lists_of_ours_go_first_and_the_parts_begin_where_they_did() {
        // Three lists of the title's, run as 2, 0, 1, another part at the
        // second; two of ours are lists 3 and 4.
        let mut order = vec![2, 0, 1];
        let parts = in_front(3, &mut order, &[1], 2);
        assert_eq!(order, [3, 4, 2, 0, 1]);
        assert_eq!(parts, [3]);
        let mut order = vec![0];
        assert!(in_front(1, &mut order, &[], 1).is_empty());
        assert_eq!(order, [1, 0]);
    }

    /// Off unless asked for: no thread, nothing waited for, nothing taken.
    #[test]
    fn without_the_switch_there_is_no_worker() {
        if std::env::var_os("VICTORY_HEAP_WORKER").is_some() || std::env::var_os("VICTORY_AB").is_some() {
            return;
        }
        assert!(!on());
        assert!(JOB.get().is_none());
        assert!(pass().is_none());
        assert!(take_ready().is_empty());
        assert!(!here());
        assert!(!stop());
        uploaded(1 << 30);
        assert!(!stop());
    }
}
