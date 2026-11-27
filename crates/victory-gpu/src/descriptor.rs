//! Descriptor heaps: zeroed memory, with handles that are plain addresses
//! (CPU and GPU alike, as on the console).
//!
//! Descriptor sizes: views 32 bytes and samplers 16, the size of RDNA 2's
//! image and sampler descriptors, so a title that copies descriptors itself
//! moves whole ones. Render target and depth views are ours alone; 128 bytes
//! leaves room for whatever stride a title assumes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, DESCRIPTOR_HEAP, E_INVALIDARG, HRESULT, Iid, Object, Table, This, guid};
use crate::resource::{Memory, PAGEABLE_IIDS};

#[cfg(windows)]
pub(crate) mod host;
mod ranges;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    kind: u32,
    count: u32,
    flags: u32,
    nodes: u32,
}

/// Newer titles (the title) pass the description followed by
/// memory of their own for the descriptors: an address and its size in
/// bytes, or a null address to have the heap allocate (inferred from the
/// title's call sites; older titles pass the plain description).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Placement {
    address: u64,
    size: u64,
}

#[repr(C)]
pub(crate) struct Heap {
    object: Object,
    desc: HeapDesc,
    /// The first descriptor: in `memory`, or in the title's own memory.
    pub(crate) start: u64,
    memory: Option<Memory>,
    /// Shader-visible view and sampler heaps get a host heap with the same
    /// descriptor order when a draw first uses them (decision 0007, step
    /// 3); slots written before wait in `pending`.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<host::HostHeap>>,
    #[cfg(windows)]
    pending: std::sync::Mutex<Vec<u32>>,
    /// Changed pages of a placed shader-visible heap. Some titles copy
    /// descriptor records into it as bytes, without a descriptor API call.
    #[cfg(windows)]
    written: Option<crate::watch::Tracked>,
}

pub(crate) fn heap<'a>(this: This) -> &'a Heap {
    // SAFETY: called from descriptor heap methods only.
    unsafe { &*this.cast::<Heap>() }
}

impl Heap {
    /// Whether this is a sampler heap.
    pub(crate) fn samplers(&self) -> bool {
        self.desc.kind == SAMPLERS
    }

    /// The host heap, made on first use from what the title wrote so far.
    /// Made late because a host view heap the size of the title's
    /// costs over 100 MB of commit, which its boot needs for its page pool.
    #[cfg(windows)]
    pub(crate) fn host(&self) -> Option<&host::HostHeap> {
        if let Some(h) = self.host.get() {
            return h.as_ref();
        }
        if self.desc.flags & SHADER_VISIBLE == 0 || !matches!(self.desc.kind, VIEWS_KIND | SAMPLERS)
        {
            return None;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // A recorder can take the fast path as soon as the OnceLock is
        // initialized. Publish only after replaying the queued writes.
        self.host
            .get_or_init(|| {
                let step = u64::from(increment(self.desc.kind)?);
                let h = host::HostHeap::new(self.samplers(), self.desc.count)?;
                for i in pending.drain(..) {
                    h.mirror(i, self.start + u64::from(i) * step, false);
                }
                Some(h)
            })
            .as_ref()
    }

    /// The handle of descriptor `index` of this heap.
    #[cfg(windows)]
    pub(crate) fn handle(&self, index: u32) -> Option<u64> {
        let step = u64::from(increment(self.desc.kind)?);
        (index < self.desc.count).then(|| self.start + u64::from(index) * step)
    }

    /// The descriptor index of `handle` in this heap, if it lies in it.
    pub(crate) fn index(&self, handle: u64) -> Option<u32> {
        let step = u64::from(increment(self.desc.kind)?);
        let at = handle.checked_sub(self.start)?;
        let index = at / step;
        (index < u64::from(self.desc.count)).then_some(index as u32)
    }
}

const VIEWS_KIND: u32 = 0;
const SAMPLERS: u32 = 1;
const SHADER_VISIBLE: u32 = 1;

/// Shader-visible heaps by address range, for mirroring descriptor writes:
/// start, end, heap.
#[cfg(windows)]
///
/// Held shared while a heap found here is used (by every thread that writes
/// descriptors, and by [`sync_heaps`]), alone only where heaps come and go.
static SHADER_HEAPS: std::sync::RwLock<Vec<(u64, u64, usize)>> = std::sync::RwLock::new(Vec::new());

/// Makes the host descriptor for what the title wrote at `handle`, if it
/// is in a shader-visible heap. Keeps `SHADER_HEAPS` locked for the duration
/// so `Heap::drop` cannot free the heap concurrently.
#[cfg(windows)]
fn mirror(handle: u64) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    let Some(&(_, _, at)) = heaps
        .iter()
        .find(|&&(start, end, _)| (start..end).contains(&handle))
    else {
        return;
    };
    // SAFETY: registered heaps stay alive until `Heap::drop`, which takes
    // the `SHADER_HEAPS` lock we hold here.
    let heap = unsafe { &*(at as *const Heap) };
    let Some(index) = heap.index(handle) else {
        return;
    };
    // Renderer port #2: its record is looked at again with the title's own.
    crate::port_descriptors::mark(handle);
    let mut pending = heap.pending.lock().unwrap_or_else(|e| e.into_inner());
    match heap.host.get() {
        None => pending.push(index),
        Some(host) => {
            drop(pending);
            if let Some(host) = host {
                host.mirror(index, handle, false);
            }
        }
    }
}

/// Whether [`sync_heaps`] is done (`VICTORY_HEAP_SYNC=0`: not).
#[cfg(windows)]
fn sync_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC").is_none_or(|v| v != "0"))
}

/// Whether a submission's [`sync_heaps`] is done by a thread of its own
/// beside the rest of the preparing (`VICTORY_HEAP_SYNC_ASIDE=0`: in front of
/// it, as before).
///
/// It took 13 ms a frame of the submitting thread (session play-1125-0203:
/// 230 changed heap pages and 470,000 slots in 30 submissions). What it makes
/// are the host heap's descriptors, which the card reads: it has to be done
/// before the lists go to the queue, and nothing of the preparing looks at
/// them (what a draw reads through a table it made itself, when recorded).
#[cfg(windows)]
fn sync_aside() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| sync_on() && std::env::var_os("VICTORY_HEAP_SYNC_ASIDE").is_none_or(|v| v != "0"))
}

/// How long that thread waits for a submission before it makes a round of its
/// own (`VICTORY_HEAP_SYNC_EARLY=<ms>`, 4 if not given; `0`: none). The title
/// copies its records while it records, and the first submission of a frame
/// found all of them: 8 ms waited for (run cpu-2). A round made meanwhile
/// leaves it those written since.
#[cfg(windows)]
fn sync_early() -> Option<std::time::Duration> {
    static EVERY: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let ms = std::env::var("VICTORY_HEAP_SYNC_EARLY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4u64);
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    })
}

/// The thread's work: rounds asked for, rounds done.
#[cfg(windows)]
static SYNC_ROUNDS: (std::sync::Mutex<(u64, u64)>, std::sync::Condvar) = (std::sync::Mutex::new((0, 0)), std::sync::Condvar::new());

/// [`sync_heaps`], begun; done when it is dropped (with [`sync_aside`]), or
/// when this returns.
#[cfg(windows)]
pub(crate) struct SyncedHeaps(Option<u64>);

#[cfg(windows)]
pub(crate) fn begin_sync_heaps() -> SyncedHeaps {
    if !sync_aside() {
        sync_heaps();
        return SyncedHeaps(None);
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    static THREAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    STARTED.call_once(|| {
        let made = std::thread::Builder::new().name("victory-heap-sync".into()).spawn(|| {
            crate::cores::keep_on_fast_cores("victory-heap-sync");
            let rounds = || SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
            /// A round is over, however it ends.
            struct Over;
            impl Drop for Over {
                fn drop(&mut self) {
                    SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner()).1 += 1;
                    SYNC_ROUNDS.1.notify_all();
                }
            }
            // Between the rounds asked for, while submissions come.
            let mut last = std::time::Instant::now();
            loop {
                let mut r = rounds();
                let mut asked = r.0 != r.1;
                while !asked {
                    let Some(every) = sync_early().filter(|_| last.elapsed().as_millis() < 500) else {
                        r = SYNC_ROUNDS.1.wait(r).unwrap_or_else(|e| e.into_inner());
                        asked = r.0 != r.1;
                        continue;
                    };
                    let (back, over) = SYNC_ROUNDS.1.wait_timeout(r, every).unwrap_or_else(|e| e.into_inner());
                    r = back;
                    asked = r.0 != r.1;
                    if over.timed_out() {
                        break;
                    }
                }
                drop(r);
                let _over = asked.then(|| {
                    last = std::time::Instant::now();
                    Over
                });
                let _round =
                    crate::timeline::span(if asked { "the heaps' records, for a submission" } else { "the heaps' records, early" });
                let _ = std::panic::catch_unwind(sync_heaps);
            }
        });
        THREAD.store(made.is_ok(), std::sync::atomic::Ordering::Release);
    });
    if !THREAD.load(std::sync::atomic::Ordering::Acquire) {
        sync_heaps();
        return SyncedHeaps(None);
    }
    let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    rounds.0 += 1;
    let mine = rounds.0;
    drop(rounds);
    SYNC_ROUNDS.1.notify_all();
    SyncedHeaps(Some(mine))
}

/// Renderer port #7 (`crate::port_journal`, from renderer patch #4): [`begin_sync_heaps`] for a
/// submission that needs round `mark` (one that began after what it waits
/// for was written, `crate::port_journal::round_after`): asked for if it has not
/// been. 0: as [`begin_sync_heaps`].
#[cfg(windows)]
pub(crate) fn begin_sync_heaps_after(mark: u64) -> SyncedHeaps {
    if mark == 0 || !sync_aside() {
        return begin_sync_heaps();
    }
    // (The thread is started by the first `begin_sync_heaps`.)
    let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    if rounds.1 == 0 && rounds.0 == 0 {
        drop(rounds);
        return begin_sync_heaps();
    }
    let asked = rounds.0 < mark;
    rounds.0 = rounds.0.max(mark);
    drop(rounds);
    if asked {
        SYNC_ROUNDS.1.notify_all();
        crate::port_journal::STATS[crate::port_journal::LATE_ROUNDS].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        crate::port_journal::STATS[crate::port_journal::EARLY_ROUNDS].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    SyncedHeaps(Some(mark))
}

/// Renderer port #7: the number of a round of the heaps' records that
/// begins after this call, asked for if none is waiting to begin (at the
/// title's submit: the round runs while the submission waits in the line).
/// 0 if the records are not synced by a thread of their own (or it has not
/// started yet).
#[cfg(windows)]
pub(crate) fn round_after_now() -> u64 {
    if !sync_aside() {
        return 0;
    }
    let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    if rounds.1 == 0 && rounds.0 == 0 {
        return 0;
    }
    let (round, ask) = crate::port_journal::round_after(rounds.0, rounds.1);
    if ask {
        rounds.0 = round;
        drop(rounds);
        SYNC_ROUNDS.1.notify_all();
        crate::port_journal::STATS[crate::port_journal::ROUNDS_ASKED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    round
}

/// Renderer port #7: the number of the first round that begins after this
/// call, not asked for ([`begin_sync_heaps_after`] asks for it if nobody did).
#[cfg(windows)]
pub(crate) fn round_mark() -> u64 {
    if !sync_aside() {
        return 0;
    }
    let rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    if rounds.1 == 0 && rounds.0 == 0 {
        return 0;
    }
    rounds.0 + 1
}

/// Whether `[at, at + len)` touches the title memory of a placed
/// shader-visible heap of descriptors.
#[cfg(windows)]
pub(crate) fn in_shader_heap(at: usize, len: usize) -> bool {
    let (from, to) = (at as u64, at as u64 + len as u64);
    SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner()).iter().any(|&(start, end, _)| from < end && start < to)
}

#[cfg(windows)]
impl Drop for SyncedHeaps {
    fn drop(&mut self) {
        let Some(mine) = self.0 else {
            return;
        };
        let began = std::time::Instant::now();
        let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
        while rounds.1 < mine {
            rounds = SYNC_ROUNDS.1.wait(rounds).unwrap_or_else(|e| e.into_inner());
        }
        drop(rounds);
        SYNCED[6].fetch_add(began.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Mirrors records copied directly into placed shader-visible heaps. Calls
/// to `Create*View` are already mirrored by [`mirror`]; raw CPU copies leave
/// only write-watch pages to reveal which slots changed.
///
/// Upstream VIctory 86a5ed6 does the same and counts 1,300 to 14,400 such
/// slots a frame in the world of the title this fork works on, against a
/// few hundred views made by calls; without it those host slots stayed
/// null, or went on showing what they showed before.
/// `VICTORY_HEAP_SYNC=0`: not done, as before. [`SYNCED`] counts.
#[cfg(windows)]
pub(crate) fn sync_heaps() {
    use std::sync::atomic::Ordering::Relaxed;
    if !sync_on() {
        return;
    }
    let started = std::time::Instant::now();
    let (mut pages, mut looked, mut mirrored, mut tabled, mut later) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut views, mut bytes) = (0u64, 0u64);
    let mut spent = [0u64; 3];
    let since = |at: std::time::Instant| at.elapsed().as_nanos() as u64;
    let mut kinds = [0u64; 5];
    let mut job_ns = [0u64; 3];
    let mut short = 0u64;
    let mut wait: Vec<u32> = Vec::new();
    let frame = crate::frame::current();
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    for &(start, end, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        let (Some(Some(host)), Some(written), Some(step)) =
            (heap.host.get(), &heap.written, increment(heap.desc.kind))
        else {
            continue;
        };
        let step = u64::from(step);
        // Capped host heaps have no counterparts for the title slots reserved
        // for per-draw tables; raw writes there must not be mirrored.
        let mirrored_count = heap.desc.count.min(host.count());
        // The records of some frames ago that no draw read through a table
        // since: a shader may pick them by index.
        let before = host.copied_before(frame);
        let of_tables = std::sync::atomic::AtomicU64::new(0);
        let (pages_a_job, records_a_job) = sync_jobs();
        let phase = crate::timeline::span("the heaps' records: those that waited");
        crate::land::each_view(true, before.len().div_ceil(records_a_job), &|n| {
            for &index in before.chunks(records_a_job).nth(n).unwrap_or_default() {
                if host.copied_tabled(index, frame) {
                    of_tables.fetch_add(1, Relaxed);
                } else {
                    host.mirror(index, start + u64::from(index) * step, false);
                }
            }
        });
        phase.end();
        let of_tables = of_tables.into_inner();
        tabled += of_tables;
        mirrored += before.len() as u64 - of_tables;
        kinds[4] += before.len() as u64 - of_tables;
        // Those of the changed pages, made together at the end.
        let mut make: Vec<u32> = Vec::new();
        let mut taken_in_check: Vec<(u32, u32)> = Vec::new();
        // Ranges of the heap to look at, and whether a range is one the
        // fallback of renderer port #2 looks at whole.
        let mut parts: Vec<(usize, usize, bool)> = Vec::new();
        let port = crate::port_descriptors::adopted(start);
        if port {
            // Renderer port #2: the slots the title's patched stores and our
            // calls marked, in place of the changed pages; and a few of the
            // pages the watch says were written, looked at whole.
            let phase = crate::timeline::span("the heaps' records: the marks taken (renderer port 2)");
            let runs = crate::port_descriptors::take(start).unwrap_or_default();
            phase.end();
            if crate::port_descriptors::ask_due(frame) {
                let began = std::time::Instant::now();
                let phase = crate::timeline::span("the heaps' records: asking (renderer port 2's fallback)");
                let taken = written.take(start as usize, end as usize);
                phase.end();
                spent[0] += since(began);
                crate::port_descriptors::asked(&taken);
            }
            let pages = crate::port_descriptors::to_check(frame);
            (parts, taken_in_check) = crate::port_descriptors::plan_parts(start, end, step, mirrored_count, &runs, &pages, crate::watch::PAGE);
        } else {
            let began = std::time::Instant::now();
            let phase = crate::timeline::span("the heaps' records: asking");
            let taken = written.take(start as usize, end as usize);
            phase.end();
            spent[0] += since(began);
            for (range_from, range_end) in taken {
                let mut page = range_from & !(crate::watch::PAGE - 1);
                while page < range_end {
                    parts.push((page.max(range_from), (page + crate::watch::PAGE).min(range_end), false));
                    page += crate::watch::PAGE;
                }
            }
        }
        // The marks' runs are short: more of them a job.
        let pages_a_job = if port { pages_a_job.max(parts.len() / 48) } else { pages_a_job };
        // A few pages a thread at a time: read, compared with what was last
        // seen there and sorted out, beside one another (14 ms a frame of one
        // thread at 160 pages, run faults-6).
        let began = std::time::Instant::now();
        let copied = host.copied_begin(step as usize);
        let samplers = heap.samplers();
        let in_place = sync_in_place() && step == 32 && crate::safecopy::on();
        let found: Vec<std::sync::Mutex<SyncPart>> = parts.chunks(pages_a_job).map(|_| Default::default()).collect();
        let phase = crate::timeline::span("the heaps' records: pages read and compared");
        crate::land::each_view(true, found.len(), &|n| {
            let mut out = SyncPart::default();
            let mut page_bytes: Vec<u8> = Vec::new();
            let mut changed: Vec<u32> = Vec::new();
            let mut bits: Vec<u64> = Vec::new();
            for &(from, to, check) in parts.chunks(pages_a_job).nth(n).unwrap_or_default() {
                let checking = check.then(std::time::Instant::now);
                let slots = changed_slots(from, to, start as usize, step, mirrored_count);
                // A record that straddles the range's ends is left
                // to the range that holds its first byte.
                let first = ((from - start as usize) as u64).div_ceil(step).max(slots.start);
                let lead = (start + first * step) as usize;
                let records: &[u8] = if in_place {
                    // Read where they are, each record once, and compared
                    // as it is read (`host::Copied::changes_in_place`).
                    let at = std::time::Instant::now();
                    let count = (to.saturating_sub(lead) as u64 / step).min(slots.end.saturating_sub(first));
                    changed.clear();
                    // SAFETY: as below; title memory is read by a copy that
                    // fails where a page is not there.
                    let whole = unsafe { copied.changes_in_place(first as u32, lead as *const u8, count as usize, &mut bits, &mut changed) };
                    out.ns[1] += since(at);
                    // (A run of marked slots is no page; nor does it ask the guards.)
                    if !port || check {
                        out.pages += 1;
                        out.views += u64::from(crate::watch::guarded(from, to));
                    }
                    out.bytes += (to - from) as u64;
                    out.looked += count;
                    out.short += u64::from(!whole);
                    // What was read of the changed records is in the records
                    // last seen now.
                    // SAFETY: this part's records alone, which nothing else
                    // writes meanwhile.
                    unsafe { copied.seen(first as u32, count as usize) }
                } else {
                    // The page's records are read through the kernel, into a
                    // copy: a page the title unmaps meanwhile fails the read
                    // where 2,048 reads in place would fault.
                    let at = std::time::Instant::now();
                    page_bytes.resize(to - from, 0);
                    if !crate::host::read_into(from as *const u8, &mut page_bytes) {
                        continue;
                    }
                    out.ns[0] += since(at);
                    out.pages += 1;
                    out.views += u64::from(crate::watch::guarded(from, to));
                    out.bytes += (to - from) as u64;
                    let Some(records) = page_bytes.get(lead.wrapping_sub(from)..) else {
                        continue;
                    };
                    let count = (records.len() as u64 / step).min(slots.end.saturating_sub(first));
                    out.looked += count;
                    // Only the records that are not the ones last seen
                    // there (`host::Copied::changes`).
                    changed.clear();
                    let at = std::time::Instant::now();
                    // SAFETY: the parts are ranges of the heap apart from one
                    // another, and only records that lie whole in one are its.
                    unsafe { copied.changes(first as u32, &records[..(count * step) as usize], &mut changed) };
                    out.ns[1] += since(at);
                    records
                };
                let at = std::time::Instant::now();
                for &index in &changed {
                    let at = (u64::from(index) - first) as usize * step as usize;
                    let word = |n: usize| {
                        records
                            .get(at + 4 * n..at + 4 * n + 4)
                            .and_then(|b| b.try_into().ok())
                            .map_or(0, u32::from_le_bytes)
                    };
                    if check && !crate::port_descriptors::was_taken_in_check(&taken_in_check, index) {
                        // Renderer port #2's fallback: what no mark told of.
                        crate::port_descriptors::missed(index, start + u64::from(index) * step, !samplers && word(0) == VIEW_TAG, host.copied_tabled(index, frame));
                    }
                    if !samplers && word(0) == VIEW_TAG {
                        // A draw's own table: the draw makes what it
                        // reads of it (`draw.rs` `tables`).
                        if host.copied_tabled(index, frame) {
                            out.tabled += 1;
                            continue;
                        }
                        // Buffer views and placed textures get no host
                        // slot from this anyway, they wait for a draw
                        // or a pass: noted some frames on, unless a
                        // draw's table turns out to hold them.
                        if matches!(word(2) >> KIND_SHIFT, CONSTANT | PLACED) && host.copied_waits() {
                            out.wait.push(index);
                            continue;
                        }
                    }
                    // No view, in a slot that shows none: nothing to make or
                    // unmake (32,000 of 41,000 a frame on the water, run
                    // b2-water).
                    if word(0) != VIEW_TAG && host.shows_none(index) {
                        out.nothing += 1;
                        continue;
                    }
                    out.make.push(index);
                    out.kinds[if samplers {
                        3
                    } else if word(0) != VIEW_TAG {
                        0
                    } else if word(2) >> KIND_SHIFT == 0 {
                        1
                    } else {
                        2
                    }] += 1;
                }
                out.ns[2] += since(at);
                if let Some(began) = checking {
                    crate::port_descriptors::CHECKED[4].fetch_add(since(began), Relaxed);
                }
            }
            *found[n].lock().unwrap_or_else(|e| e.into_inner()) = out;
        });
        phase.end();
        drop(copied);
        for part in found {
            let mut part = part.into_inner().unwrap_or_else(|e| e.into_inner());
            pages += part.pages;
            views += part.views;
            bytes += part.bytes;
            looked += part.looked;
            short += part.short;
            tabled += part.tabled;
            kinds[0] += part.nothing;
            later += part.wait.len() as u64;
            mirrored += part.make.len() as u64;
            wait.append(&mut part.wait);
            make.append(&mut part.make);
            for (kind, n) in kinds.iter_mut().zip(part.kinds) {
                *kind += n;
            }
            for (ns, n) in job_ns.iter_mut().zip(part.ns) {
                *ns += n;
            }
        }
        spent[1] += since(began);
        let began = std::time::Instant::now();
        let phase = crate::timeline::span("the heaps' records: made");
        crate::land::each_view(true, make.len().div_ceil(records_a_job), &|n| {
            for &index in make.chunks(records_a_job).nth(n).unwrap_or_default() {
                host.mirror(index, start + u64::from(index) * step, false);
            }
        });
        phase.end();
        spent[2] += since(began);
        if !wait.is_empty() {
            host.copied_wait(frame, std::mem::take(&mut wait));
        }
    }
    drop(heaps);
    let micros = started.elapsed().as_micros() as u64;
    for (count, n) in SYNCED.iter().zip([pages, looked, mirrored, micros, tabled, later]) {
        count.fetch_add(n, Relaxed);
    }
    let [taking, reading, making] = spent;
    for (count, n) in SYNC_SPLIT.iter().zip([views, bytes, taking, reading, making]) {
        count.fetch_add(n, Relaxed);
    }
    for (count, n) in SYNCED_KINDS.iter().zip(kinds) {
        count.fetch_add(n, Relaxed);
    }
    for (count, n) in SYNC_JOBS_NS.iter().zip(job_ns) {
        count.fetch_add(n, Relaxed);
    }
    SYNC_SHORT.fetch_add(short, Relaxed);
}

/// Pages [`sync_heaps`] read in place of which a part could not be read
/// ([`sync_in_place`]), since the last frame line.
#[cfg(windows)]
static SYNC_SHORT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether [`sync_heaps`] reads a changed page's records where they are, each
/// once, and compares each with the one last seen as it reads it, 32 bytes at
/// a time (`VICTORY_SYNC_IN_PLACE=0`: into a copy of the page first, then
/// compared, as before).
///
/// The copy and the compare were each a pass over all of the page: 7.4 and
/// 5.3 ms a frame of the six threads for 35 MB (run heapsync-ab1, in the
/// street). A page the title unmaps meanwhile fails where the copy failed;
/// the records before the place are taken as they were read.
///
/// On by default since run heapsync-ab2 (A/B in the street: the pass 12.7 ms
/// a frame against 15.7-16.2, the copy and compare 7.4 ms of its threads
/// against 13.3, the submitting thread's wait for it 2.0 ms against 2.2-2.6;
/// the picture the same).
#[cfg(windows)]
fn sync_in_place() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_SYNC_IN_PLACE") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SYNC_IN_PLACE").is_none_or(|v| v != "0"))
}

/// Of [`sync_heaps`]'s page jobs since the last frame line, on all threads:
/// nanoseconds reading pages, comparing them with what was last seen, and
/// sorting out the records that changed.
#[cfg(windows)]
static SYNC_JOBS_NS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Pages and records [`sync_heaps`] gives a thread at a time
/// (`VICTORY_SYNC_FINE=1`: [`SYNC_PAGES_FINE`] and [`SYNC_AT_A_TIME_FINE`];
/// otherwise [`SYNC_PAGES`] and [`SYNC_AT_A_TIME`], as before).
///
/// A round for a submission finds a dozen pages or so: by eight a job that
/// was two jobs, on two of the six threads, while the submission waited
/// for the round (2 to 3.4 ms at a time, run city4).
#[cfg(windows)]
fn sync_jobs() -> (usize, usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let default_fine = || *ON.get_or_init(|| std::env::var_os("VICTORY_SYNC_FINE").is_some_and(|v| v != "0"));
    #[cfg(feature = "diagnostics")]
    let fine = crate::ab::pick("VICTORY_SYNC_FINE").unwrap_or_else(default_fine);
    #[cfg(not(feature = "diagnostics"))]
    let fine = default_fine();
    if fine { (SYNC_PAGES_FINE, SYNC_AT_A_TIME_FINE) } else { (SYNC_PAGES, SYNC_AT_A_TIME) }
}

#[cfg(windows)]
const SYNC_PAGES_FINE: usize = 2;

#[cfg(windows)]
const SYNC_AT_A_TIME_FINE: usize = 64;

/// Of those mirrored, since the last frame line: no view, a view of an object,
/// another, a sampler; waited and no draw came.
#[cfg(windows)]
static SYNCED_KINDS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// What a thread found on its pages of a heap, for [`sync_heaps`].
#[cfg(windows)]
#[derive(Default)]
struct SyncPart {
    pages: u64,
    views: u64,
    bytes: u64,
    looked: u64,
    tabled: u64,
    nothing: u64,
    make: Vec<u32>,
    wait: Vec<u32>,
    kinds: [u64; 5],
    /// Nanoseconds reading, comparing, sorting out ([`SYNC_JOBS_NS`]).
    ns: [u64; 3],
    /// Pages read in place not all there ([`SYNC_SHORT`]).
    short: u64,
}

/// Pages [`sync_heaps`] gives a thread at a time.
#[cfg(windows)]
const SYNC_PAGES: usize = 8;

/// Records [`sync_heaps`] gives a thread at a time.
#[cfg(windows)]
const SYNC_AT_A_TIME: usize = 256;

/// Since the last frame line, of [`sync_heaps`]: changed pages of placed
/// shader-visible heaps looked at, slots on them, slots whose record was
/// not the one last seen and was mirrored, the microseconds it took; then
/// new records left to the draws that read them through a table, and new
/// records that waited for one ([`host::HostHeap::copied_wait`]); and the
/// microseconds submissions waited for it ([`sync_aside`]).
#[cfg(windows)]
pub(crate) static SYNCED: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];

/// The same, of [`SYNCED`]'s pages: those in view memory that was guarded;
/// bytes read; nanoseconds asking what was written, reading it and comparing
/// it with what was last seen, making views.
#[cfg(windows)]
static SYNC_SPLIT: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// With the frame line: what [`sync_heaps`] did per frame over `frames`.
/// Nothing while no heap page changed.
#[cfg(windows)]
pub(crate) fn sync_line(frames: f64) {
    crate::port_descriptors::line(frames);
    crate::draw::in_place_indirect_line(frames);
    let [twins, moved, lost] = [0, 1, 2].map(|i| host::TWINS[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    if twins > 0 {
        tracing::info!(
            "raw twins of buffer views made per frame (VICTORY_RAW_VIEWS=0: none): {:.0}, of them {:.1} that do not begin on 16 bytes ({:.1} null)",
            twins as f64 / frames,
            moved as f64 / frames,
            lost as f64 / frames
        );
    }
    let [pages, looked, mirrored, micros, tabled, later, waited] = [0, 1, 2, 3, 4, 5, 6].map(|i| SYNCED[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    let split = [0, 1, 2, 3, 4].map(|i| SYNC_SPLIT[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64);
    let kinds = [0, 1, 2, 3, 4].map(|i| SYNCED_KINDS[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames);
    let jobs = [0, 1, 2].map(|i| SYNC_JOBS_NS[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6 / frames);
    let quick = [0, 1].map(|i| host::SPANS_QUICK[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames);
    let short = SYNC_SHORT.swap(0, std::sync::atomic::Ordering::Relaxed);
    if pages == 0 {
        return;
    }
    tracing::info!(
        "the heaps' records' jobs per frame (VICTORY_SYNC_FINE={}, VICTORY_SYNC_IN_PLACE={}): on all threads {:.2} ms reading pages, \
         {:.2} comparing (in place: reading too), {:.2} sorting out; writes of slots {:.0} leaving the spans alone (VICTORY_SPANS_QUICK), \
         {:.0} under their lock; {} pages read in place not all there",
        u8::from(sync_jobs().0 == SYNC_PAGES_FINE),
        u8::from(sync_in_place()),
        jobs[0],
        jobs[1],
        jobs[2],
        quick[0],
        quick[1],
        short
    );
    tracing::info!(
        "descriptor records the title copied into its heap as bytes (VICTORY_HEAP_SYNC=0: not looked for): per frame \
         {:.1} changed heap pages, {:.0} slots looked at, {:.1} whose record was new and was mirrored, {:.1} new ones of \
         draws' tables left to the draws, {:.1} that waited for a draw first, {:.2} ms ({:.2} ms waited for); of the mirrored \
         {:.0} no views (with those of slots that show none), {:.0} of objects, {:.0} others, {:.0} samplers, {:.0} that waited and no draw came; \
         {:.1} of the pages guarded views, {:.1} MB read; {:.2} ms asking, {:.2} reading and comparing, {:.2} making",
        pages as f64 / frames,
        looked as f64 / frames,
        mirrored as f64 / frames,
        tabled as f64 / frames,
        later as f64 / frames,
        micros as f64 / 1000.0 / frames,
        waited as f64 / 1000.0 / frames,
        kinds[0],
        kinds[1],
        kinds[2],
        kinds[3],
        kinds[4],
        split[0] / frames,
        split[1] / (1 << 20) as f64 / frames,
        split[2] / 1e6 / frames,
        split[3] / 1e6 / frames,
        split[4] / 1e6 / frames
    );
}

/// Title descriptor slots in a changed byte range that have host counterparts.
/// `range_end` is exclusive.
#[cfg(windows)]
fn changed_slots(
    range_from: usize,
    range_end: usize,
    heap_start: usize,
    step: u64,
    mirrored_count: u32,
) -> std::ops::Range<u64> {
    let from = (range_from - heap_start) as u64;
    let to = (range_end - heap_start) as u64;
    let first = from / step;
    let last = to.div_ceil(step).min(u64::from(mirrored_count));
    first.min(last)..last
}

/// The title's view heap that shaders of a list without one get their views
/// from (`draw.rs`, `bind`): the title's structure builder binds root views
/// only, and the host shows those, and memory by address, through heap slots.
/// The biggest there is. `VICTORY_LENT_HEAP=0`: none, such a dispatch is
/// dropped, as before.
#[cfg(windows)]
pub(crate) fn lent_heap() -> Option<usize> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_LENT_HEAP").is_none_or(|v| v != "0")) {
        return None;
    }
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        // SAFETY: registered heaps are live (removed when dropped).
        .map(|&(_, _, h)| (h, unsafe { &*(h as *const Heap) }))
        .filter(|(_, heap)| !heap.samplers() && matches!(heap.host.get(), Some(Some(_))))
        .max_by_key(|(_, heap)| heap.desc.count)
        .map(|(h, _)| h)
}

/// `f` of the title heap at `heap`, if that heap still exists; it stays
/// until `f` is done.
#[cfg(windows)]
pub(crate) fn with_heap<T>(heap: usize, f: impl FnOnce(&Heap) -> T) -> Option<T> {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    // SAFETY: registered heaps are live (removed when dropped).
    heaps.iter().any(|&(_, _, h)| h == heap).then(|| f(unsafe { &*(heap as *const Heap) }))
}

/// Gives back dynamic slots of the host heap behind the title heap at
/// `heap`, if that heap is still there, and `id` says it is the one that
/// took them ([crate::descriptor::host::HostHeap::id]): the title can release
/// the heap a list's slots came from, and a later heap can be made at the
/// same address before the list is reset.
#[cfg(windows)]
pub(crate) fn release_dynamic(heap: usize, id: u64, first: u32, n: u32) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.release(id, first, n);
    }
}

/// Runs `f` with the host heap of the title's largest view heap, if there is
/// one (nothing is waited for).
#[cfg(windows)]
pub(crate) fn with_view_heap<T>(f: impl FnOnce(&host::HostHeap) -> T) -> Option<T> {
    let heaps = SHADER_HEAPS.try_read().ok()?;
    let host = heaps
        .iter()
        // SAFETY: registered heaps are live (removed when dropped).
        .map(|&(_, _, at)| unsafe { &*(at as *const Heap) })
        .filter(|heap| !heap.samplers())
        .filter_map(|heap| heap.host.get()?.as_ref())
        .max_by_key(|host| host.count())?;
    Some(f(host))
}

/// Makes dynamic slot `index` of the host heap behind the title heap at
/// `heap`, if that heap still exists, the constant buffer view of `size`
/// bytes of `buffer` at `gpu`.
#[cfg(windows)]
pub(crate) fn constant_slot(
    heap: usize,
    index: u32,
    buffer: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    gpu: u64,
    size: u32,
) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.constant_at(index, buffer, gpu, size);
    }
}

#[cfg(windows)]
impl Drop for Heap {
    fn drop(&mut self) {
        let at = std::ptr::from_ref::<Heap>(self) as usize;
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        heaps.retain(|&(_, _, h)| h != at);
        crate::port_descriptors::let_go(self.start);
        drop(heaps);
        // Its run of title-memory views goes with it: a heap made later at
        // the same address must not find this one's run (fork pull request
        // #589, AscilCH).
        if let Some(Some(host)) = self.host.get() {
            crate::memory::forget_heap(host);
        }
    }
}

/// Bytes per descriptor, by heap type (views, samplers, render targets,
/// depth).
fn increment(kind: u32) -> Option<u32> {
    [32, 16, 128, 128].get(kind as usize).copied()
}

/// Device method.
pub(crate) extern "system" fn increment_size(_: This, kind: u32) -> u32 {
    increment(kind).unwrap_or(0)
}

/// What our render target views hold: the resource and the view
/// description, turned into a host view when a command uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RtvRecord {
    tag: u32,
    described: u32,
    pub(crate) resource: This,
    desc: [u32; 6],
}
const RTV_TAG: u32 = u32::from_le_bytes(*b"vRTV");
const _: () = assert!(size_of::<RtvRecord>() <= 112);

/// Value snapshot for replaying a command list after a cut. Keep the
/// resource identity as an integer (like draw::State), never the address
/// of mutable descriptor storage. The caller still owns the resource
/// lifetime, as with the original recorded command.
#[cfg(windows)]
#[derive(Clone, Copy)]
pub(crate) struct RtvSnapshot {
    tag: u32,
    described: u32,
    resource: usize,
    desc: [u32; 6],
}

#[cfg(windows)]
impl RtvSnapshot {
    pub(crate) fn restore(self) -> RtvRecord {
        RtvRecord {
            tag: self.tag,
            described: self.described,
            resource: self.resource as This,
            desc: self.desc,
        }
    }
}

impl RtvRecord {
    /// All the view it records is made from: records with equal keys have
    /// equal views (of the same host texture).
    #[cfg(windows)]
    pub(crate) fn key(&self) -> (usize, [u32; 7]) {
        let mut words = [self.described; 7];
        words[1..].copy_from_slice(&self.desc);
        (self.resource as usize, words)
    }

    #[cfg(windows)]
    pub(crate) fn snapshot(self) -> RtvSnapshot {
        RtvSnapshot {
            tag: self.tag,
            described: self.described,
            resource: self.resource as usize,
            desc: self.desc,
        }
    }

    /// The view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC> {
        const TEXTURE_2D: u32 = 4;
        self.host_desc(TEXTURE_2D)
    }

    /// The depth view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn depth_desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_VIEW_DESC> {
        const TEXTURE_2D: u32 = 3;
        self.host_desc(TEXTURE_2D)
    }

    /// The description words as a host type, the format as the host has it.
    /// Without one, a typeless host copy gets a 2D view (`dimension`) in
    /// the title's format, which the host needs spelled out.
    #[cfg(windows)]
    fn host_desc<T>(&self, dimension: u32) -> Option<T> {
        assert!(size_of::<T>() <= size_of::<[u32; 6]>());
        let mut words = if self.described != 0 {
            self.desc
        } else {
            [
                crate::resource::view_format(self.resource)?,
                dimension,
                0,
                0,
                0,
                0,
            ]
        };
        // A target or depth view: of a target, in its stand-in format.
        words[0] = crate::resource::target_format(crate::resource::pc_format(words[0]));
        // SAFETY: both host view descriptions are 24 bytes in this layout.
        Some(unsafe { (&raw const words).cast::<T>().read_unaligned() })
    }
}

/// Device method: records the view at `handle` (descriptor memory).
pub(crate) extern "system" fn create_rtv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    if handle == 0 {
        return;
    }
    // SAFETY: the title passes a description or null.
    let described = unsafe { desc.as_ref() };
    let record = RtvRecord {
        tag: RTV_TAG,
        described: u32::from(described.is_some()),
        resource,
        desc: described.copied().unwrap_or_default(),
    };
    // SAFETY: the handle is a descriptor slot, at least 112 bytes.
    unsafe { (handle as *mut RtvRecord).write_unaligned(record) };
}

/// Device method: a depth view, recorded as render target views are.
pub(crate) extern "system" fn create_dsv(
    this: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    create_rtv(this, resource, desc, handle);
    if handle != 0 {
        // SAFETY: just written by `create_rtv`.
        unsafe { (handle as *mut u32).write_unaligned(DSV_TAG) };
    }
}
const DSV_TAG: u32 = u32::from_le_bytes(*b"vDSV");

/// A shader view or sampler as the title described it (descriptions in the
/// host's layout, kept as words), for the host to build its own from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum View {
    /// Constant buffer: GPU address and size.
    Constant(u64, u32),
    /// Shader resource: resource, description (none: the resource's own).
    Shader(usize, Option<[u32; 10]>),
    /// Unordered access: resource, counter resource, description.
    Unordered(usize, usize, Option<[u32; 10]>),
    Sampler([u32; 13]),
    /// The console's own sampler description: flags, filters per axis,
    /// addressing, LOD and anisotropy settings, border color, then
    /// console-only tuning words.
    ConsoleSampler([u32; 27]),
    /// A view of memory without a resource object: whether it is unordered
    /// access, the resource description, then the view's (the plain one
    /// followed by the memory address and console fields).
    Placed(bool, [u32; 14], [u32; 16]),
}

impl View {
    /// Whether the view reads buffer memory (constant buffers, buffer views
    /// of resources or of memory) or memory the title placed a texture in:
    /// such views get host slots only for draws.
    #[cfg(windows)]
    fn is_buffer(&self) -> bool {
        match self {
            View::Constant(..) | View::Placed(..) => true,
            View::Shader(r, _) | View::Unordered(r, _, _) => {
                crate::resource::buffer_bytes(*r as This).is_some()
            }
            View::Sampler(_) | View::ConsoleSampler(_) => false,
        }
    }
}

/// Every distinct view in use; a descriptor holds a tag, an index into it
/// and the entry's generation, which survive titles copying descriptors as
/// bytes. A view of a resource lives as long as the resource: when the
/// title releases it, its entries are freed for reuse under a new
/// generation, so descriptors (and host slots made from them) naming the
/// old one go stale instead of naming whatever is made at the same address
/// next.
///
/// Views without a resource object have nothing that releases them, and a
/// title copies descriptors as bytes, so nothing says when the last
/// descriptor naming an entry is gone: such entries were kept for the run.
/// The title writes some 14,000 constant buffer views a frame,
/// each at the address its constants have that frame: 2,450,139 entries
/// after four minutes of play, 1,047 MB with the index, and 11,000 more
/// every second (run 65). So the two kinds that carry an address are not
/// entries any more: their records name them whole ([`constant_words`],
/// [`placed_words`]), and the table holds only what is bounded: views of
/// resource objects (freed with the resource), samplers (as many as there
/// are sampler states), and the shapes of placed views (their descriptions
/// without the address: as many as the title has kinds of textures and
/// buffers, however often it moves them). A placed view whose shape does
/// not fit once that list is full is not an entry either ([`UNKEPT`]): the
/// slot reads as nothing, and the table stays bounded.
/// `VICTORY_VIEW_INLINE=0`: every view is an entry, as before.
struct Views {
    /// Each entry's view and generation.
    list: Vec<(View, u32)>,
    index: std::collections::BTreeMap<View, u32>,
    /// Entries naming each resource.
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
    /// The shapes of views of placed memory, and where each is.
    shapes: Vec<Shape>,
    shape_index: std::collections::BTreeMap<Shape, u32>,
}

/// Read far more often than written: every slot a draw reads through a
/// table is looked up here, on several recording threads at once in the
/// world. Readers share it; only a view not seen before (a new table entry
/// or shape) and a released resource take it alone.
static VIEWS: std::sync::RwLock<Views> = std::sync::RwLock::new(Views {
    list: Vec::new(),
    index: std::collections::BTreeMap::new(),
    by_resource: std::collections::BTreeMap::new(),
    free: Vec::new(),
    shapes: Vec::new(),
    shape_index: std::collections::BTreeMap::new(),
});
pub(crate) const VIEW_TAG: u32 = u32::from_le_bytes(*b"vVIE");

/// A record is the tag and two words, `i` and `generation` (the host keeps
/// them as one 64-bit key per slot). The top two bits of `generation` say
/// what the other 62 bits are: 0, entry `i` of the table at that
/// generation; [`CONSTANT`], a constant buffer view, whole; [`PLACED`], a
/// view of placed memory: its address and the number of its shape;
/// [`UNKEPT`], nothing.
const KIND_SHIFT: u32 = 30;
/// Table generations stay below the kind bits (and above 0).
const GENERATIONS: u32 = (1 << KIND_SHIFT) - 1;
const CONSTANT: u32 = 1;
const PLACED: u32 = 2;
/// A record that names no view ([`view_at`] returns nothing). A placed view
/// whose shape does not fit once [`SHAPES`] is full is this, not a table
/// entry: a view with no resource object is never freed, and session 0152
/// kept 18.8 million of them.
const UNKEPT: u32 = 3;
/// Bits of a constant buffer view's address in its record: any address a
/// program can have. The 15 bits above them are its size.
const CONSTANT_ADDRESS_BITS: u32 = 47;
/// Bits of a placed view's address in its record: placed memory is title
/// memory, which lies below 1 TiB. The 22 bits above them number its shape.
const PLACED_ADDRESS_BITS: u32 = 40;
const SHAPES: usize = 1 << (62 - PLACED_ADDRESS_BITS);

/// A view of placed memory without its address: whether it is unordered
/// access, the resource description, and the view's words but the two that
/// hold the address (10 and 11).
type Shape = (bool, [u32; 14], [u32; 14]);

/// Whether constant buffer views and views of placed memory are named whole
/// in their records ([`Views`]). On by default; `VICTORY_VIEW_INLINE=0`
/// makes every view a table entry, as before.
fn inline() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_INLINE").is_none_or(|v| v != "0"))
}

/// Views named whole in their records since the start: constant buffer
/// views, views of placed memory (for the memory line).
static NAMED_WHOLE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The two record words of the 62 bits `payload` of kind `kind`.
fn kind_words(kind: u32, payload: u64) -> [u32; 2] {
    [payload as u32, kind << KIND_SHIFT | (payload >> 32) as u32]
}

/// The 62 bits two record words carry beside their kind.
fn payload(i: u32, generation: u32) -> u64 {
    u64::from(generation & GENERATIONS) << 32 | u64::from(i)
}

/// The record words that name the constant buffer view of `size` bytes at
/// `address` whole; `None` if they cannot: an address past 47 bits, or a
/// size of 16 KiB or more that is no multiple of 256 or reaches 4 MiB (the
/// size has 15 bits: bytes below 16 KiB, else units of 256 bytes). Such a
/// view is a table entry.
fn constant_words(address: u64, size: u32) -> Option<[u32; 2]> {
    if address >> CONSTANT_ADDRESS_BITS != 0 {
        return None;
    }
    let code = if size < 1 << 14 {
        u64::from(size)
    } else if size.is_multiple_of(256) && size >> 8 < 1 << 14 {
        1 << 14 | u64::from(size >> 8)
    } else {
        return None;
    };
    Some(kind_words(CONSTANT, code << CONSTANT_ADDRESS_BITS | address))
}

/// The constant buffer view [`constant_words`] named.
fn constant_of(i: u32, generation: u32) -> View {
    let payload = payload(i, generation);
    let code = (payload >> CONSTANT_ADDRESS_BITS) as u32;
    let size = if code >> 14 == 0 { code } else { (code & 0x3fff) << 8 };
    View::Constant(payload & ((1 << CONSTANT_ADDRESS_BITS) - 1), size)
}

/// A placed view's shape and its address.
fn shape_of(unordered: bool, resource: &[u32; 14], desc: &[u32; 16]) -> (Shape, u64) {
    let mut rest = [0u32; 14];
    rest[..10].copy_from_slice(&desc[..10]);
    rest[10..].copy_from_slice(&desc[12..]);
    (
        (unordered, *resource, rest),
        u64::from(desc[10]) | u64::from(desc[11]) << 32,
    )
}

/// The record words that name the view of placed memory at `address` with
/// shape number `shape`.
fn placed_words(shape: u32, address: u64) -> [u32; 2] {
    kind_words(PLACED, u64::from(shape) << PLACED_ADDRESS_BITS | address)
}

/// The record words of a placed view that is not kept ([`UNKEPT`]).
fn unkept_words() -> [u32; 2] {
    kind_words(UNKEPT, 0)
}

/// How much of a placed view's two descriptions is blanked before the view
/// is kept (`VICTORY_SHAPE_BLANK`; unset: 1; 0: nothing, as before):
///
/// * 1: the words that are padding of the host's layouts: words 1 and 13 of
///   the resource description (which `resource::texture_at` blanks for its
///   own key; not with `VICTORY_PLACED_FILLER=1`) and word 3 of a shader
///   resource view's; and the last word of the view's union (word 9), which
///   only a 2D array shader view uses.
/// * 2: also the words the view's dimension leaves unused at the end of its
///   union.
///
/// The title leaves in such words what its stack held, so two views alike
/// in all that a view is had two shapes. Session 0152: 2,850 new shapes a
/// second in the world, the list of 4,194,304 full after 1,470 s, and from
/// then on every placed view a table entry kept for the run: 18.8 million
/// of them and 9.2 GB an hour in. Blanking is what keeps the list from
/// filling. A full list does not keep such a view any more ([`unkept_words`]).
/// Elsewhere (2026-11-26) the list was full after 27 minutes with only
/// level 1's first three words blanked: word 9 still took 2,003 values
/// among the newest 4,096 shapes.
fn shape_blank() -> u32 {
    #[cfg(test)]
    {
        // `u32::MAX`: the environment, as a run. A test sets 0, 1 or 2.
        let over = SHAPE_BLANK_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
        if over != u32::MAX {
            return over;
        }
    }
    static LEVEL: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("VICTORY_SHAPE_BLANK").as_deref() {
        Ok("0") => 0,
        Ok("2") => 2,
        _ => 1,
    })
}

/// See [`shape_blank`].
#[cfg(test)]
static SHAPE_BLANK_OVERRIDE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

/// `VICTORY_PLACED_FILLER=1` (`resource::texture_at`): the filler words of a
/// resource description are part of what a texture is, so they stay.
fn filler_kept() -> bool {
    static KEPT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KEPT.get_or_init(|| std::env::var_os("VICTORY_PLACED_FILLER").is_some_and(|v| v == "1"))
}

/// The words of a placed view's description (the plain one, words 0 to 9)
/// that its dimension leaves unused at the end of its union: the host's
/// layouts, where a shader resource view's union begins at word 4 and an
/// unordered access view's at word 2. Empty for a dimension not listed.
fn unused_words(unordered: bool, dimension: u32) -> std::ops::Range<usize> {
    // Where the union begins and how many of its words the dimension uses.
    let (begin, used) = if unordered {
        let used = match dimension {
            1 => 7, // buffer: first element (2), elements, stride, counter offset (2), flags
            2 => 1, // 1D: level
            3 => 3, // 1D array: level, first slice, slices
            4 => 2, // 2D: level, plane
            5 => 4, // 2D array: level, first slice, slices, plane
            8 => 3, // 3D: level, first slice, slices
            _ => return 0..0,
        };
        (2, used)
    } else {
        let used = match dimension {
            1 => 5,     // buffer: first element (2), elements, stride, flags
            2 => 3,     // 1D: first level, levels, least level of detail
            3 => 5,     // 1D array: and first slice, slices
            4 => 4,     // 2D: first level, levels, plane, least level of detail
            8 | 9 => 3, // 3D, cube: first level, levels, least level of detail
            10 => 5,    // cube array: and first face, cubes
            _ => return 0..0,
        };
        (4, used)
    };
    begin + used..10
}

/// Blanks what [`shape_blank`] `level` says of a placed view's two
/// descriptions. The address (words 10 and 11 of `desc`) and the console's
/// words after it are never touched.
fn blanked(level: u32, filler_kept: bool, unordered: bool, resource: &mut [u32; 14], desc: &mut [u32; 16]) {
    if level == 0 {
        return;
    }
    if !filler_kept {
        (resource[1], resource[13]) = (0, 0);
    }
    // The last word of the union: no unordered view's dimension reaches it
    // (a buffer's ends at word 8), and of the shader views only a 2D
    // array's does (its least level of detail).
    const TEXTURE_2D_ARRAY: u32 = 5;
    if unordered {
        desc[9] = 0;
    } else {
        desc[3] = 0;
        if desc[1] != TEXTURE_2D_ARRAY {
            desc[9] = 0;
        }
    }
    if level >= 2 {
        let unused = unused_words(unordered, desc[1]);
        desc[unused].fill(0);
    }
}

/// Words of a [`Shape`]: the resource description's 14, then the view's 14.
const SHAPE_WORDS: usize = 28;
/// New shapes since the start; placed views that found the shapes full and
/// were not kept; and for each word of a shape, the new shapes that
/// differed there from the shape made before them: the words that make
/// shapes many, for the memory line.
static SHAPES_NEW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SHAPES_REFUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SHAPE_WORD_NEW: [std::sync::atomic::AtomicU64; SHAPE_WORDS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; SHAPE_WORDS];
/// Set once the shapes are full: they never become fewer, so a view of a
/// shape not there need not take the table alone to be told so again.
static SHAPES_FULL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The words in which two shapes differ, as bits ([`SHAPE_WORDS`]).
fn shape_differs(a: &Shape, b: &Shape) -> u32 {
    let words = |s: &Shape| s.1.into_iter().chain(s.2);
    words(a).zip(words(b)).enumerate().fold(0, |bits, (k, (x, y))| bits | u32::from(x != y) << k)
}

/// What a word of a shape is called in the log: `r` and the word of the
/// resource description, or `v` and the word of the view's (whose words 10
/// and 11, the address, are not in a shape).
#[cfg_attr(not(any(test, all(windows, feature = "diagnostics"))), allow(dead_code))]
fn shape_word_name(k: usize) -> String {
    match k {
        0..14 => format!("r{k}"),
        14..24 => format!("v{}", k - 14),
        _ => format!("v{}", k - 12),
    }
}

/// For the memory line: how the shapes of placed views grow and which words
/// make them differ (the eight that differ most often).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn shapes_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let mut words: Vec<(u64, usize)> =
        SHAPE_WORD_NEW.iter().enumerate().map(|(k, n)| (n.load(Relaxed), k)).filter(|w| w.0 > 0).collect();
    words.sort_unstable_by(|a, b| b.cmp(a));
    let words: Vec<String> = words.iter().take(8).map(|&(n, k)| format!("{} {n}", shape_word_name(k))).collect();
    format!(
        "{} made since the start (at most {SHAPES}), {} placed views found them full and were not kept; \
         new shapes that differed from the one before them, by word (r: resource description, v: view): {}; \
         VICTORY_SHAPE_BLANK={} (1, the default, blanks r1, r13, a shader view's v3 and v9 but a 2D array's          v9; 0 nothing; 2 also a view's unused words)",
        SHAPES_NEW.load(Relaxed),
        SHAPES_REFUSED.load(Relaxed),
        if words.is_empty() { "none".into() } else { words.join(", ") },
        shape_blank(),
    )
}

impl Views {
    /// The number of `shape`, a new one if it is not there yet; `None` when
    /// the shapes are full (the view is then not kept: [`unkept_words`]).
    fn shape(&mut self, shape: Shape) -> Option<u32> {
        use std::sync::atomic::Ordering::Relaxed;
        if let Some(&at) = self.shape_index.get(&shape) {
            return Some(at);
        }
        if self.shapes.len() >= SHAPES {
            if !SHAPES_FULL.swap(true, Relaxed) {
                tracing::warn!(
                    shapes = SHAPES,
                    "the shapes of placed views are full: a placed view of a shape not here is not kept, \
                     so the table does not grow (VICTORY_SHAPE_BLANK=2 makes shapes fewer)"
                );
            }
            return None;
        }
        SHAPES_NEW.fetch_add(1, Relaxed);
        if let Some(before) = self.shapes.last() {
            let mut bits = shape_differs(before, &shape);
            while bits != 0 {
                SHAPE_WORD_NEW[bits.trailing_zeros() as usize].fetch_add(1, Relaxed);
                bits &= bits - 1;
            }
        }
        let at = self.shapes.len() as u32;
        self.shapes.push(shape);
        self.shape_index.insert(shape, at);
        Some(at)
    }

    /// The view of placed memory [`placed_words`] named.
    fn placed_of(&self, i: u32, generation: u32) -> Option<View> {
        let payload = payload(i, generation);
        let address = payload & ((1 << PLACED_ADDRESS_BITS) - 1);
        let &(unordered, resource, rest) = self.shapes.get((payload >> PLACED_ADDRESS_BITS) as usize)?;
        let mut desc = [0u32; 16];
        desc[..10].copy_from_slice(&rest[..10]);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        desc[12..].copy_from_slice(&rest[10..]);
        Some(View::Placed(unordered, resource, desc))
    }

    /// The record words of `view` as a table entry, if it has one.
    fn entry_of(&self, view: &View) -> Option<[u32; 2]> {
        let &i = self.index.get(view)?;
        Some([i, self.list[i as usize].1])
    }

    /// The record words of `view` as a table entry: its entry (a new or a
    /// freed one if the view is not there yet) and that entry's generation.
    fn entry(&mut self, view: View) -> [u32; 2] {
        let i = match self.index.get(&view) {
            Some(&i) => i,
            None => {
                let i = match self.free.pop() {
                    Some(i) => {
                        self.list[i as usize].0 = view.clone();
                        i
                    }
                    None => {
                        self.list.push((view.clone(), 1));
                        self.list.len() as u32 - 1
                    }
                };
                for r in view.resources().into_iter().filter(|&r| r != 0) {
                    self.by_resource.entry(r).or_default().push(i);
                }
                self.index.insert(view, i);
                i
            }
        };
        [i, self.list[i as usize].1]
    }
}

/// The two words that name `view` in a record: the view itself where its
/// kind allows ([`Views`]), else its table entry.
fn words_for(view: View) -> [u32; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    if inline() {
        match &view {
            View::Constant(address, size) => {
                if let Some(words) = constant_words(*address, *size) {
                    NAMED_WHOLE[0].fetch_add(1, Relaxed);
                    return words;
                }
            }
            View::Placed(unordered, resource, desc) => {
                let (shape, address) = shape_of(*unordered, resource, desc);
                if address >> PLACED_ADDRESS_BITS == 0 {
                    // Nearly every placed view has a shape seen before.
                    let known = views_read().shape_index.get(&shape).copied();
                    // Once full, the list stays full. Asking again would
                    // take the write lock on every recording thread, and
                    // the entry made then is never freed.
                    let found = known.or_else(|| {
                        if SHAPES_FULL.load(Relaxed) {
                            None
                        } else {
                            views_write().shape(shape)
                        }
                    });
                    if let Some(found) = found {
                        NAMED_WHOLE[1].fetch_add(1, Relaxed);
                        return placed_words(found, address);
                    }
                    SHAPES_REFUSED.fetch_add(1, Relaxed);
                    return unkept_words();
                }
            }
            _ => {}
        }
    }
    // A view written before keeps its entry and generation.
    if let Some(words) = views_read().entry_of(&view) {
        return words;
    }
    views_write().entry(view)
}

fn views_read() -> std::sync::RwLockReadGuard<'static, Views> {
    VIEWS.read().unwrap_or_else(|e| e.into_inner())
}

fn views_write() -> std::sync::RwLockWriteGuard<'static, Views> {
    VIEWS.write().unwrap_or_else(|e| e.into_inner())
}

/// Which of six kinds [`kept`] counts a view under. Plain indices, so the
/// breakdown is an array: constant buffer, shader resource, unordered
/// access, sampler, console sampler, placed memory.
#[cfg(all(windows, any(test, feature = "diagnostics")))]
const KIND_CONSTANT: usize = 0;
#[cfg(all(windows, any(test, feature = "diagnostics")))]
const KIND_SHADER: usize = 1;
#[cfg(all(windows, any(test, feature = "diagnostics")))]
const KIND_UNORDERED: usize = 2;
#[cfg(all(windows, any(test, feature = "diagnostics")))]
const KIND_SAMPLER: usize = 3;
#[cfg(all(windows, any(test, feature = "diagnostics")))]
const KIND_CONSOLE_SAMPLER: usize = 4;
#[cfg(all(windows, any(test, feature = "diagnostics")))]
const KIND_PLACED: usize = 5;

#[cfg(all(windows, any(test, feature = "diagnostics")))]
fn kind_index(view: &View) -> usize {
    match view {
        View::Constant(..) => KIND_CONSTANT,
        View::Shader(..) => KIND_SHADER,
        View::Unordered(..) => KIND_UNORDERED,
        View::Sampler(..) => KIND_SAMPLER,
        View::ConsoleSampler(..) => KIND_CONSOLE_SAMPLER,
        View::Placed(..) => KIND_PLACED,
    }
}

/// For the memory line: the distinct views kept ([`Views`]), those of them
/// free for reuse, and the bytes the table and its index hold for them (an
/// estimate: an entry of the table, and one of the index with its share of
/// a tree node; the shapes likewise); then the entries that are constant
/// buffer views and views of placed memory (those that did not fit a
/// record, or all of them with `VICTORY_VIEW_INLINE=0`), the shapes of
/// placed views kept, and the views named whole in their records since the
/// start: constant buffer views, views of placed memory.
///
/// Beside that, the same table by kind ([`kind_index`]): occupied entries,
/// free entries (a freed slot keeps the kind it last held), and those
/// slots' flat bytes (one list entry each, not the index or the spare
/// capacity, which the summary's byte estimate still folds in undivided).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn views_kept() -> ([u64; 8], [[u64; 3]; 6]) {
    kept(&views_read())
}

/// [`views_kept`] of one snapshot of the table, so a test can count a table
/// it owns. One pass over `list` and one over `free`.
#[cfg(all(windows, any(test, feature = "diagnostics")))]
fn kept(views: &Views) -> ([u64; 8], [[u64; 3]; 6]) {
    use std::sync::atomic::Ordering::Relaxed;
    let each = size_of::<(View, u32)>() as u64;
    let shape = size_of::<Shape>() as u64;
    let mut by_kind = [[0u64; 3]; 6];
    for (view, _) in &views.list {
        let k = kind_index(view);
        by_kind[k][0] += 1;
        by_kind[k][2] += each;
    }
    for &i in &views.free {
        let Some((view, _)) = views.list.get(i as usize) else {
            continue;
        };
        let k = kind_index(view);
        by_kind[k][0] -= 1;
        by_kind[k][1] += 1;
    }
    let constants = by_kind[KIND_CONSTANT][0] + by_kind[KIND_CONSTANT][1];
    let placed = by_kind[KIND_PLACED][0] + by_kind[KIND_PLACED][1];
    let summary = [
        views.list.len() as u64,
        views.free.len() as u64,
        views.list.capacity() as u64 * each
            + views.index.len() as u64 * each * 3 / 2
            + views.shapes.capacity() as u64 * shape
            + views.shape_index.len() as u64 * shape * 3 / 2,
        constants,
        placed,
        views.shapes.len() as u64,
        NAMED_WHOLE[0].load(Relaxed),
        NAMED_WHOLE[1].load(Relaxed),
    ];
    (summary, by_kind)
}

impl View {
    /// The resource objects the view names.
    fn resources(&self) -> [usize; 2] {
        match *self {
            View::Shader(r, _) => [r, 0],
            View::Unordered(r, counter, _) => [r, counter],
            _ => [0, 0],
        }
    }

    /// The texture object the view is of, if that texture has no host copy
    /// yet (`crate::resource::host_waits`): a slot written with such a
    /// view is left null until the texture is first used.
    #[cfg(windows)]
    fn waiting_texture(&self) -> Option<usize> {
        match *self {
            View::Shader(r, _) | View::Unordered(r, _, _)
                if crate::resource::host_waits(r as This) =>
            {
                Some(r)
            }
            _ => None,
        }
    }
}

/// Title slots (their handles) written with a view of a texture object
/// that had no host copy then, by texture. A title texture gets its host
/// copy on first use (`resource.rs` `Hosted`), and the title
/// writes each texture's views into its view heap when it creates the
/// texture: making those host slots would make every host copy at once.
/// They are made when the texture is: translated shaders index the heap
/// directly, so a slot is not always made again by the draw that reads it.
#[cfg(windows)]
static WAITING: std::sync::Mutex<std::collections::BTreeMap<usize, Vec<u64>>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// Textures given their host copy since [`make_waiting`] last ran.
#[cfg(windows)]
static MADE_LATE: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Title texture `resource` was given its host copy (called under that
/// texture's own lock: nothing else is locked here).
#[cfg(windows)]
pub(crate) fn texture_made(resource: usize) {
    MADE_LATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(resource);
}

/// The slot at `handle` was written with a view of texture `resource`,
/// which has no host copy: it waits for one.
#[cfg(windows)]
fn wait_for(resource: usize, handle: u64) {
    /// A texture with more slots waiting than this is in use for all we
    /// know (its views are being copied about): its host copy is made.
    const MOST: usize = 64;
    let many = {
        let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
        let slots = waiting.entry(resource).or_default();
        if !slots.contains(&handle) {
            slots.push(handle);
        }
        slots.len() > MOST
    };
    if many {
        // Makes it, and notes it made.
        let _ = crate::resource::host_copy(resource as This);
    } else if !crate::resource::host_waits(resource as This) {
        // Made by another thread since the slot was looked at: that making
        // may have been taken by `make_waiting` before this slot was noted.
        texture_made(resource);
    }
}

/// Makes the host slots that waited for a texture's host copy, for the
/// textures made since the last call. Called before a submission runs, by
/// the thread that submits: a slot is read when a list runs.
#[cfg(windows)]
pub(crate) fn make_waiting() {
    let made = std::mem::take(&mut *MADE_LATE.lock().unwrap_or_else(|e| e.into_inner()));
    for resource in made {
        let handles = WAITING
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&resource);
        for handle in handles.into_iter().flatten() {
            // Only if the slot still holds a view of that texture: the
            // title may have written it since (and a view of another
            // texture without a host copy must go on waiting).
            // (Looked up first: the handle is read only while its heap is
            // there.)
            // (The heap stays while the lock is held: `mirror`.)
            let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
            let Some(&(_, _, at)) = heaps.iter().find(|&&(start, end, _)| (start..end).contains(&handle)) else {
                continue;
            };
            // SAFETY: a registered heap, live while the lock is held.
            let heap = unsafe { &*(at as *const Heap) };
            let Some(index) = heap.index(handle) else {
                continue;
            };
            let same = match view(handle) {
                Some(View::Shader(r, _) | View::Unordered(r, _, _)) => r == resource,
                _ => false,
            };
            if same && let Some(Some(host)) = heap.host.get() {
                host.mirror(index, handle, true);
            }
        }
    }
}

fn record(handle: u64, view: View) {
    if handle == 0 {
        return;
    }
    let [i, generation] = words_for(view);
    // SAFETY: the handle is a descriptor slot, at least 16 bytes.
    unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
    #[cfg(windows)]
    mirror(handle);
}

/// The title released `resource`: the views naming it go stale.
pub(crate) fn forget(resource: usize) {
    #[cfg(windows)]
    WAITING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&resource);
    let mut views = views_write();
    let Some(entries) = views.by_resource.remove(&resource) else {
        return;
    };
    for i in entries {
        let (view, generation) = views.list[i as usize].clone();
        // Entries freed through a view's other resource may hold another
        // view by now.
        if views.index.get(&view) != Some(&i) || !view.resources().contains(&resource) {
            continue;
        }
        views.index.remove(&view);
        // (Below the bits that say a record is no table entry.)
        views.list[i as usize].1 = ((generation + 1) & GENERATIONS).max(1);
        // The view's other resource (a counter) keeps a stale entry, which
        // `forget` of that resource skips.
        views.free.push(i);
    }
}

/// The view the record words `i` and `generation` name: the one they hold
/// whole, or the view of entry `i` if `generation` is still its own.
fn view_at(i: u32, generation: u32) -> Option<View> {
    match generation >> KIND_SHIFT {
        0 => {
            let views = views_read();
            let (view, current) = views.list.get(i as usize)?;
            (*current == generation).then(|| view.clone())
        }
        CONSTANT => Some(constant_of(i, generation)),
        PLACED => views_read().placed_of(i, generation),
        _ => None,
    }
}

/// The words of the record at `handle`, if they are a view's.
pub(crate) fn record_at(handle: u64) -> Option<[u32; 3]> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let words = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    (words[0] == VIEW_TAG).then_some(words)
}

/// The view recorded at `handle`, if one is (and its resource is alive).
pub(crate) fn view(handle: u64) -> Option<View> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    if tag != VIEW_TAG {
        return None;
    }
    view_at(i, generation)
}

/// A description of `N` words, if the title passes one.
fn words<const N: usize>(desc: *const [u32; N]) -> Option<[u32; N]> {
    // SAFETY: the title passes a description or null.
    (!desc.is_null()).then(|| unsafe { desc.read_unaligned() })
}

/// Device method.
pub(crate) extern "system" fn create_cbv(_: This, desc: *const [u32; 4], handle: u64) {
    if let Some([lo, hi, size, _]) = words(desc) {
        let address = u64::from(lo) | u64::from(hi) << 32;
        record(handle, View::Constant(address, size));
    }
}

/// Device method.
pub(crate) extern "system" fn create_srv(
    _: This,
    resource: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    record(handle, View::Shader(resource as usize, words(desc)));
}

/// Device method.
pub(crate) extern "system" fn create_uav(
    _: This,
    resource: This,
    counter: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    let view = View::Unordered(resource as usize, counter as usize, words(desc));
    record(handle, view);
}

/// Device methods 0x230 and 0x238: shader resource and unordered access
/// views of memory the title placed itself, `(resource desc, view desc,
/// handle)`.
pub(crate) extern "system" fn create_placed_srv(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(mut resource), Some(mut desc)) = (words(resource), words(desc)) {
        blanked(shape_blank(), filler_kept(), false, &mut resource, &mut desc);
        record(handle, View::Placed(false, resource, desc));
    }
}

pub(crate) extern "system" fn create_placed_uav(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(mut resource), Some(mut desc)) = (words(resource), words(desc)) {
        blanked(shape_blank(), filler_kept(), true, &mut resource, &mut desc);
        record(handle, View::Placed(true, resource, desc));
    }
}

/// Device method.
pub(crate) extern "system" fn create_sampler(_: This, desc: *const [u32; 13], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::Sampler(desc));
    }
}

/// Device method 0x1e8.
pub(crate) extern "system" fn create_sampler_x(_: This, desc: *const [u32; 27], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::ConsoleSampler(desc));
    }
}

/// Device method: `count` descriptors of `kind` from `src` to `dst`.
pub(crate) extern "system" fn copy_simple(_: This, count: u32, dst: u64, src: u64, kind: u32) {
    let Some(step) = increment(kind).map(u64::from) else {
        return;
    };
    let len = u64::from(count) * step;
    if dst != 0 && src != 0 {
        // SAFETY: both ranges are descriptor slots the title owns.
        #[cfg(windows)]
        unsafe {
            crate::titlewrite::put(dst as *mut u8, src as *const u8, len as usize)
        };
        // SAFETY: as above.
        #[cfg(not(windows))]
        unsafe {
            std::ptr::copy(src as *const u8, dst as *mut u8, len as usize)
        };
        #[cfg(windows)]
        for i in 0..u64::from(count) {
            mirror(dst + i * step);
        }
    }
}

/// Device method: ranges of descriptors (start and count each) gathered
/// from the sources and scattered to the destinations, in order.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn copy(
    _: This,
    dst_ranges: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    src_ranges: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    kind: u32,
) {
    let Some(step) = increment(kind) else {
        return;
    };
    // Each range as (start, count); a null size array means one descriptor
    // per range.
    let ranges = |n: u32, starts: *const u64, sizes: *const u32| -> Vec<(u64, u32)> {
        (0..n as usize)
            .map(|i| {
                // SAFETY: the title passes `n` starts and sizes (or null sizes).
                unsafe {
                    (
                        *starts.add(i),
                        if sizes.is_null() { 1 } else { *sizes.add(i) },
                    )
                }
            })
            .collect()
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let src = ranges(src_ranges, src_starts, src_sizes);
    let mut dst = ranges(dst_ranges, dst_starts, dst_sizes).into_iter();
    let (mut to, mut room) = dst.next().unwrap_or((0, 0));
    for (mut from, mut left) in src {
        while left > 0 {
            while room == 0 {
                let Some(next) = dst.next() else { return };
                (to, room) = next;
            }
            let n = left.min(room);
            copy_simple(std::ptr::null(), n, to, from, kind);
            let bytes = u64::from(n) * u64::from(step);
            (to, from, room, left) = (to + bytes, from + bytes, room - n, left - n);
        }
    }
}

/// The view recorded at `handle`, if one is.
pub(crate) fn rtv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, RTV_TAG)
}

/// The depth view recorded at `handle`, if one is.
pub(crate) fn dsv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, DSV_TAG)
}

fn tagged_record(handle: u64, tag: u32) -> Option<RtvRecord> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let record = unsafe { (handle as *const RtvRecord).read_unaligned() };
    (record.tag == tag).then_some(record)
}

/// Device method.
pub(crate) extern "system" fn create_heap(
    _: This,
    desc_ptr: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc_ptr.as_ref() }) else {
        return E_INVALIDARG;
    };
    let Some(size) = increment(desc.kind).map(|i| u64::from(i) * u64::from(desc.count.max(1)))
    else {
        return E_INVALIDARG;
    };
    // SAFETY: read only as a hint: a title that passes the plain
    // description has its own stack or data after it.
    let placement = unsafe { desc_ptr.add(1).cast::<Placement>().read_unaligned() };
    // shortcut: a placement is recognised by its size matching the count,
    // since the plain and extended descriptions share the slot and the
    // interface version; find the flag that tells them apart if a title
    // trips over this.
    let placed = placement.address != 0 && placement.size == size;
    tracing::debug!(?desc, placed, "descriptor heap");
    let (start, memory) = if placed {
        if placement.address.checked_add(size).is_none() {
            return E_INVALIDARG;
        }
        (placement.address, None)
    } else {
        match Memory::zeroed(size, 64) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    };
    let visible = desc.flags & SHADER_VISIBLE != 0 && matches!(desc.kind, VIEWS_KIND | SAMPLERS);
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        start,
        memory,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        pending: Default::default(),
        #[cfg(windows)]
        written: (placed && visible)
            .then(|| crate::watch::track(start as usize, size as usize, true))
            .flatten(),
    });
    // Renderer port #2: the title's own stores tell this heap's slots.
    #[cfg(windows)]
    if placed && visible && desc.kind == VIEWS_KIND {
        crate::port_descriptors::adopt(start, desc.count);
    }
    #[cfg(windows)]
    if visible {
        let at = std::ptr::from_ref::<Heap>(&heap) as usize;
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        heaps.push((start, start + size, at));
    }
    #[cfg(not(windows))]
    let _ = visible;
    object::hand_out(DESCRIPTOR_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &HEAP_IIDS).then_some(this);
    object::answer(DESCRIPTOR_HEAP, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(heap(this).desc) };
    }
}
/// The same address for the CPU and the GPU start.
extern "system" fn start(this: This) -> u64 {
    heap(this).start
}

/// IUnknown 3, object 4, device child 1, heap 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<DESCRIPTOR_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = desc as *const (); // 0x40
    f[9] = start as *const (); // 0x48 CPU
    f[10] = start as *const (); // 0x50 GPU
    Table(f)
};

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    #[test]
    fn render_target_snapshot_preserves_original_descriptor_words() {
        let mut record = super::RtvRecord {
            tag: super::RTV_TAG,
            described: 1,
            resource: 0x1000usize as crate::object::This,
            desc: [24, 4, 0, 0, 0, 0],
        };
        let snapshot = record.snapshot();
        record.desc[0] = 26;
        record.resource = 0x2000usize as crate::object::This;
        let restored = snapshot.restore();
        assert_eq!(restored.tag, record.tag);
        assert_eq!(restored.described, record.described);
        assert_eq!(restored.desc[0], 24);
        assert_eq!(restored.resource as usize, 0x1000);
        assert_eq!(record.desc[0], 26);
        assert_eq!(record.resource as usize, 0x2000);
    }

    use super::*;
    use crate::object::{S_OK, slot};

    /// What newer titles pass; tests use it throughout, so that reading the
    /// placement stays inside the value.
    #[repr(C)]
    struct Extended(HeapDesc, Placement);

    fn create(desc: HeapDesc, placement: Placement, out: &mut This) -> HRESULT {
        let extended = Extended(desc, placement);
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], out)
    }

    #[cfg(windows)]
    #[test]
    fn raw_heap_sync_stops_before_dynamic_host_slots() {
        let start = 0x1000usize;
        let step = 32;
        let capped = 5;
        assert_eq!(
            changed_slots(
                start + 3 * step + 4,
                start + 4 * step + 8,
                start,
                step as u64,
                capped
            ),
            3..5
        );
        assert_eq!(
            changed_slots(
                start + 6 * step,
                start + 9 * step,
                start,
                step as u64,
                capped
            ),
            5..5
        );
    }

    #[test]
    fn heaps_hold_count_descriptors_from_their_start() {
        let null = std::ptr::null();
        let desc = HeapDesc {
            kind: 0,
            count: 1000,
            flags: 1,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = null;
        assert_eq!(create(desc, none, &mut h), S_OK);
        let cpu = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);
        assert_eq!(cpu % 64, 0);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), cpu);
        assert_eq!(increment_size(null, 0), 32);
        // SAFETY: the heap holds 1000 descriptors of 32 bytes.
        unsafe { (cpu as *mut u8).add(999 * 32 + 31).write(1) };
        assert_eq!(increment_size(null, 9), 0);
        let bad = HeapDesc { kind: 9, ..desc };
        assert_eq!(create(bad, none, &mut h), E_INVALIDARG);
    }

    /// A descriptor copied into a placed heap as bytes is mirrored before
    /// the next submission, even though no descriptor method saw the copy.
    #[cfg(windows)]
    #[test]
    fn records_copied_as_bytes_are_made_at_the_next_sync() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let memory = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                crate::watch::PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        };
        assert!(!memory.is_null());
        let desc = HeapDesc {
            kind: SAMPLERS,
            count: 64,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let placement = Placement {
            address: memory as u64,
            size: 64 * 16,
        };
        let mut h: This = std::ptr::null();
        assert_eq!(create(desc, placement, &mut h), S_OK);
        if let Some(host) = heap(h).host() {
            let mut elsewhere = [0u64; 2];
            let source = elsewhere.as_mut_ptr() as u64;
            let sampler = [21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
            create_sampler(std::ptr::null(), &sampler, source);
            // SAFETY: the heap slot and source both hold 16 descriptor bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source as *const u8,
                    memory.cast::<u8>().add(5 * 16),
                    16,
                )
            };
            assert!(!host.made(5));
            sync_heaps();
            assert!(host.made(5));
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        // SAFETY: this allocation belongs to the test.
        unsafe { VirtualFree(memory, 0, MEM_RELEASE) };
    }

    #[test]
    fn views_survive_copies() {
        let null = std::ptr::null();
        let mut heap = vec![0u64; 4 * 8]; // four 32-byte slots
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let (s0, s1, s2, s3) = (at(0), at(1), at(2), at(3));
        create_cbv(null, &[0x1000, 1, 256, 0], s0);
        assert_eq!(view(s0), Some(View::Constant(0x1_0000_1000, 256)));
        create_srv(null, null, std::ptr::null(), s1);
        assert_eq!(view(s1), Some(View::Shader(0, None)));
        // The same view twice is one entry.
        create_srv(null, null, std::ptr::null(), s2);
        // SAFETY: both slots are in `heap`.
        assert_eq!(unsafe { *(s1 as *const [u32; 3]) }, unsafe {
            *(s2 as *const [u32; 3])
        });
        // Two ranges of one into one range of two.
        let (dst, dst_n, src, src_n) = ([s2], [2u32], [s0, s1], [1u32, 1]);
        copy(
            null,
            1,
            dst.as_ptr(),
            dst_n.as_ptr(),
            2,
            src.as_ptr(),
            src_n.as_ptr(),
            0,
        );
        assert_eq!(view(s2), view(s0));
        assert_eq!(view(s3), view(s1));
        assert_eq!(view(at(0) + 8), None);
        heap[0] = 0;
    }

    /// Constant buffer views are named whole in their records: any address
    /// a program can have, any size below 16 KiB, multiples of 256 bytes
    /// up to 4 MiB; what does not fit says so.
    #[test]
    fn constant_views_fit_their_records() {
        let top = (1u64 << CONSTANT_ADDRESS_BITS) - 1;
        for address in [0u64, 0x100, 0x2602_0000, 0xd_6b60_0000, 0x7ffd_a34f_0000, top] {
            for size in [0u32, 1, 255, 256, 4000, (1 << 14) - 1, 1 << 14, 65536, (1 << 22) - 256] {
                let [i, generation] = constant_words(address, size).expect("fits");
                assert_eq!(generation >> KIND_SHIFT, CONSTANT);
                assert_eq!(view_at(i, generation), Some(View::Constant(address, size)), "{address:#x} {size}");
            }
        }
        assert_eq!(constant_words(top + 1, 256), None, "an address past 47 bits");
        assert_eq!(constant_words(0x1000, (1 << 14) + 1), None, "16 KiB or more, no multiple of 256");
        assert_eq!(constant_words(0x1000, 1 << 22), None, "4 MiB");
        // Different views never share their words.
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1000, 512));
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1100, 256));
        assert_ne!(constant_words(0x1000, 1 << 14), constant_words(0x1000, 64));
    }

    /// The leak of run 65: a title that writes constant buffer views and
    /// views of placed memory at ever new addresses adds nothing to the
    /// table; one shape stands for all the placed views alike.
    #[test]
    fn views_at_new_addresses_add_no_entries() {
        if !inline() {
            return;
        }
        let null = std::ptr::null();
        let slots = [0u64; 4];
        let at = slots.as_ptr() as u64;
        // A resource description and a view no other test uses (their
        // padding is zero: `blanked`).
        let mut resource: [u32; 14] = std::array::from_fn(|k| 0x5ee0_0000 + k as u32);
        (resource[1], resource[13]) = (0, 0);
        let mut desc: [u32; 16] = std::array::from_fn(|k| 0x7a70_0000 + k as u32);
        (desc[3], desc[9]) = (0, 0);
        let mine = |v: &View| match v {
            View::Constant(address, _) => (0x6_0000_0000..0x6_1000_0000).contains(address),
            View::Placed(_, r, _) => *r == resource,
            _ => false,
        };
        let entries = || VIEWS.read().unwrap().list.iter().filter(|(v, _)| mine(v)).count();
        let shapes = || VIEWS.read().unwrap().shapes.iter().filter(|s| s.1 == resource).count();
        for n in 0..20_000u64 {
            let address = 0x6_0000_0000 + n * 0x100;
            create_cbv(null, &[address as u32, (address >> 32) as u32, 512, 0], at);
            assert_eq!(view(at), Some(View::Constant(address, 512)));
            let placed = 0xd_4940_0000 + n * 0x1_0000;
            (desc[10], desc[11]) = (placed as u32, (placed >> 32) as u32);
            create_placed_srv(null, &resource, &desc, at);
            assert_eq!(view(at), Some(View::Placed(false, resource, desc)));
        }
        assert_eq!(entries(), 0, "no table entries");
        assert_eq!(shapes(), 1, "one shape for 20,000 placed views");
        // The same shape, unordered: another shape.
        create_placed_uav(null, &resource, &desc, at);
        assert_eq!(view(at), Some(View::Placed(true, resource, desc)));
        assert_eq!(shapes(), 2);
        // Placed memory past 1 TiB (not title memory) is a table entry.
        (desc[10], desc[11]) = (0, 0x100);
        create_placed_srv(null, &resource, &desc, at);
        assert_eq!(view(at), Some(View::Placed(false, resource, desc)));
        assert_eq!(entries(), 1);
        // A record's words are the same for the same view (the host knows
        // a slot's view by them), and differ with the address.
        // SAFETY: both handles are slots of 32 bytes above.
        let words = |handle: u64| unsafe { *(handle as *const [u32; 3]) };
        let other = [0u64; 4];
        create_cbv(null, &[0x40, 6, 512, 0], at);
        create_cbv(null, &[0x40, 6, 512, 0], other.as_ptr() as u64);
        assert_eq!(words(at), words(other.as_ptr() as u64));
        create_cbv(null, &[0x140, 6, 512, 0], other.as_ptr() as u64);
        assert_ne!(words(at), words(other.as_ptr() as u64));
    }

    /// Session 0152's leak: placed views alike in all but what the title's
    /// stack left in the padding of their descriptions each had a shape of
    /// their own. Blanked (`VICTORY_SHAPE_BLANK`), they share one; what a
    /// view is, its address and the console's words stay.
    #[test]
    fn placed_views_that_differ_in_padding_share_a_shape_once_blanked() {
        // A 2D texture's description and a 2D shader view of it.
        let resource: [u32; 14] = [3, 0, 0x1_0000, 0, 512, 0, 512, 10 << 16 | 1, 71, 1, 0, 0, 0, 0];
        let mut desc = [0u32; 16];
        desc[..8].copy_from_slice(&[71, 4, 0x1688, 0, 2, 8, 0, 0]);
        (desc[10], desc[11]) = (0x4940_0000, 0xd);
        desc[12..].copy_from_slice(&[7, 8, 9, 10]);
        let shape = |level: u32, kept: bool, unordered: bool, stack: u32| {
            let (mut r, mut d) = (resource, desc);
            (r[1], r[13], d[3], d[8], d[9]) = (stack, stack ^ 0x55, stack + 1, stack + 2, stack + 3);
            blanked(level, kept, unordered, &mut r, &mut d);
            (shape_of(unordered, &r, &d), r, d)
        };
        // As before: every stack value another shape.
        assert_ne!(shape(0, false, false, 1).0, shape(0, false, false, 2).0);
        let (as_was, r, d) = shape(0, false, false, 9);
        assert_eq!((r[1], r[13], d[3], d[8], d[9]), (9, 9 ^ 0x55, 10, 11, 12));
        assert_eq!(as_was.1, 0xd_4940_0000);
        // Level 1: the padding of both layouts.
        let (one, r, d) = shape(1, false, false, 1);
        assert_eq!((r[1], r[13], d[3]), (0, 0, 0));
        assert_eq!((d[8], d[9]), (3, 0), "word 8 stays at level 1; word 9 is the union's last");
        assert_ne!(one.0, shape(1, false, false, 2).0.0);
        // Level 2: and the words a 2D view does not use.
        let (two, r, d) = shape(2, false, false, 1);
        assert_eq!(two, shape(2, false, false, 0xdead_beef).0);
        assert_eq!((r[1], r[13], d[3], d[8], d[9]), (0, 0, 0, 0, 0));
        assert_eq!(d[..3], [71, 4, 0x1688]);
        assert_eq!(d[4..8], [2, 8, 0, 0], "first level, levels, plane and least level of detail");
        assert_eq!((d[10], d[11]), (0x4940_0000, 0xd), "the address");
        assert_eq!(d[12..], [7, 8, 9, 10], "the console's words");
        assert_eq!(r[2..13], resource[2..13]);
        assert_eq!(two.1, 0xd_4940_0000);
        // `VICTORY_PLACED_FILLER=1`: the resource's filler is part of the texture.
        let (_, r, _) = shape(2, true, false, 1);
        assert_eq!((r[1], r[13]), (1, 1 ^ 0x55));
        // An unordered access view's union begins at word 2: word 3 is its
        // plane for a 2D view, never padding, and words 4 to 9 are unused.
        let (_, _, d) = shape(1, false, true, 5);
        assert_eq!(d[3], 6);
        let mut d = [0x11u32; 16];
        d[1] = 4;
        blanked(2, false, true, &mut [0; 14], &mut d);
        assert_eq!(d, [0x11, 4, 0x11, 0x11, 0, 0, 0, 0, 0, 0, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11]);
    }

    /// Each dimension's unused words end at word 10 and begin after what
    /// the host's layout uses; a dimension not known loses nothing.
    #[test]
    fn unused_words_of_a_view_by_dimension() {
        // Shader resource views: buffer, 1D, 1D array, 2D, 3D, cube, cube array.
        for (dimension, first) in [(1, 9), (2, 7), (3, 9), (4, 8), (8, 7), (9, 7), (10, 9)] {
            assert_eq!(unused_words(false, dimension), first..10, "shader view {dimension}");
        }
        // Unordered access views: buffer, 1D, 1D array, 2D, 2D array, 3D.
        for (dimension, first) in [(1, 9), (2, 3), (3, 5), (4, 4), (5, 6), (8, 5)] {
            assert_eq!(unused_words(true, dimension), first..10, "unordered view {dimension}");
        }
        // A 2D array shader view uses all six; multisampled and unknown ones are left.
        let left = [(false, 5), (false, 0), (false, 6), (false, 7), (false, 11), (true, 0), (true, 6), (true, 99)];
        for (unordered, dimension) in left {
            assert!(unused_words(unordered, dimension).is_empty(), "{unordered} {dimension}");
        }
        // Never the address or the console's words, whatever the dimension says.
        for unordered in [false, true] {
            for dimension in 0..16 {
                let mut d = [u32::MAX; 16];
                d[1] = dimension;
                blanked(2, false, unordered, &mut [0; 14], &mut d);
                assert_eq!(d[10..], [u32::MAX; 6]);
                assert_eq!(d[..2], [u32::MAX, dimension]);
            }
        }
    }

    /// The memory line names the words that make shapes many.
    #[test]
    fn shapes_say_in_which_words_they_differ() {
        let a: Shape = (false, [0; 14], [0; 14]);
        let mut b = a;
        assert_eq!(shape_differs(&a, &b), 0);
        (b.1[1], b.1[13], b.2[3], b.2[13]) = (1, 1, 1, 1);
        assert_eq!(shape_differs(&a, &b), 1 << 1 | 1 << 13 | 1 << 17 | 1 << 27);
        // Whether it is unordered access is no word.
        assert_eq!(shape_differs(&a, &(true, a.1, a.2)), 0);
        let names: Vec<String> = [1, 13, 14, 17, 23, 24, 27].into_iter().map(shape_word_name).collect();
        assert_eq!(names, ["r1", "r13", "v0", "v3", "v9", "v12", "v15"]);
    }

    /// Puts [`shape_blank`] and [`SHAPES_FULL`] back when the test ends.
    struct Hold {
        level: u32,
        full: bool,
    }

    impl Hold {
        fn take() -> Self {
            Self {
                level: SHAPE_BLANK_OVERRIDE.swap(0, std::sync::atomic::Ordering::Relaxed),
                full: SHAPES_FULL.swap(false, std::sync::atomic::Ordering::Relaxed),
            }
        }
    }

    impl Drop for Hold {
        fn drop(&mut self) {
            SHAPE_BLANK_OVERRIDE.store(self.level, std::sync::atomic::Ordering::Relaxed);
            SHAPES_FULL.store(self.full, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Session 0152, finished: with blanking off, filler still makes a shape
    /// each; with it on, those views share one. Once the shape list is full,
    /// a new shape is not a table entry either way.
    #[test]
    fn full_shape_list_adds_no_entry() {
        let [i, generation] = unkept_words();
        assert_eq!(generation >> KIND_SHIFT, UNKEPT);
        assert_eq!(view_at(i, generation), None);
        if !inline() {
            return;
        }
        assert!(!filler_kept(), "VICTORY_PLACED_FILLER=1 keeps filler in the shape");
        let _hold = Hold::take();
        let null = std::ptr::null();
        let slots = [0u64; 8];
        let a = slots.as_ptr() as u64;
        let b = a + 32;
        let mut desc = [0u32; 16];
        desc[0] = 71;
        desc[1] = 4;
        desc[2] = 0x1688;
        desc[4] = 2;
        desc[5] = 8;
        (desc[10], desc[11]) = (0x4940_0000, 0xd);
        desc[12] = 7;
        let of = |mark: u32| {
            let views = views_read();
            let shapes = views.shapes.iter().filter(|s| s.1[0] == mark).count();
            let entries = views
                .list
                .iter()
                .filter(|(v, _)| matches!(v, View::Placed(_, r, _) if r[0] == mark))
                .count();
            (shapes, entries)
        };
        let put = |mark: u32, stack: u32, mip: u32, handle: u64| {
            let mut resource: [u32; 14] = std::array::from_fn(|k| mark + k as u32);
            let mut d = desc;
            (resource[1], resource[13], d[3], d[4]) = (stack, stack ^ 0x55, stack + 1, mip);
            create_placed_srv(null, &resource, &d, handle);
            (resource, d)
        };

        // Off: each stack value is its own shape, and neither is an entry.
        let old = 0x5400_0000;
        put(old, 1, 2, a);
        put(old, 2, 2, a);
        assert_eq!(of(old), (2, 0), "filler is part of the shape");

        // On: the same two share one shape. What a view is stays.
        SHAPE_BLANK_OVERRIDE.store(1, std::sync::atomic::Ordering::Relaxed);
        let mark = 0x5410_0000;
        let (mut resource, mut d) = put(mark, 1, 2, a);
        put(mark, 2, 2, a);
        blanked(1, false, false, &mut resource, &mut d);
        assert_eq!(of(mark), (1, 0), "padding blanked, one shape");
        assert_eq!(view(a), Some(View::Placed(false, resource, d)));
        assert_eq!((resource[1], resource[13], d[3]), (0, 0, 0));
        assert_eq!(d[4], 2, "the mip stays");
        assert_eq!((d[10], d[11]), (0x4940_0000, 0xd), "the address stays");

        let shapes = views_read().shapes.len();
        let entries = views_read().index.len();
        SHAPES_FULL.store(true, std::sync::atomic::Ordering::Relaxed);
        put(mark, 1, 4, b);
        assert_eq!(views_read().shapes.len(), shapes, "no new shape");
        assert_eq!(views_read().index.len(), entries, "no table entry");
        assert_eq!(view(b), None, "a shape that does not fit is not kept");
        // SAFETY: `b` is a slot of `slots`.
        let wrote = unsafe { *(b as *const [u32; 3]) };
        assert_eq!(wrote, [VIEW_TAG, i, generation]);
        // A shape already kept still resolves, and still adds nothing.
        put(mark, 9, 2, a);
        assert_eq!(view(a), Some(View::Placed(false, resource, d)));
        assert_eq!(views_read().shapes.len(), shapes);
        assert_eq!(views_read().index.len(), entries);
    }

    /// The leak of 2026-11-26 (on by default): placed views whose
    /// descriptions differ only in their filler words (r1, r13, a shader
    /// view's v3 and the union's last word, v9) are one shape and one
    /// record; a word that is part of the view still makes another.
    #[test]
    fn filler_words_make_no_new_shapes() {
        if std::env::var_os("VICTORY_SHAPE_BLANK").is_none() {
            assert_eq!(shape_blank(), 1, "blanking is on by default");
        }
        if !inline() {
            return;
        }
        assert!(!filler_kept(), "VICTORY_PLACED_FILLER=1 keeps filler in the shape");
        let _hold = Hold::take();
        SHAPE_BLANK_OVERRIDE.store(1, std::sync::atomic::Ordering::Relaxed);
        let null = std::ptr::null();
        let (slots, other) = ([0u64; 4], [0u64; 4]);
        let (at, beside) = (slots.as_ptr() as u64, other.as_ptr() as u64);
        // SAFETY: both handles are slots of 32 bytes above.
        let words = |handle: u64| unsafe { *(handle as *const [u32; 3]) };
        // A resource description and a 2D shader view no other test uses.
        let mut resource: [u32; 14] = std::array::from_fn(|k| 0x6f11_0000 + k as u32);
        let mut desc: [u32; 16] = std::array::from_fn(|k| 0x6f22_0000 + k as u32);
        (desc[1], desc[10], desc[11]) = (4, 0x4940_0000, 0xd);
        let middle = resource[2..13].to_vec();
        let shapes = || views_read().shapes.iter().filter(|s| s.1[2..13] == middle[..]).count();
        create_placed_srv(null, &resource, &desc, at);
        for n in 0..1000u32 {
            (resource[1], resource[13]) = (n, !n);
            (desc[3], desc[9]) = (n.wrapping_mul(0x9e37_79b9), n ^ 0x55aa);
            create_placed_srv(null, &resource, &desc, beside);
            assert_eq!(words(at), words(beside), "the same record whatever the filler");
        }
        assert_eq!(shapes(), 1, "one shape for 1,001 fillers");
        let Some(View::Placed(false, kept, view_kept)) = view(at) else {
            panic!("a placed view");
        };
        assert_eq!((kept[1], kept[13], view_kept[3], view_kept[9]), (0, 0, 0, 0));
        assert_eq!(kept[2..13], resource[2..13]);
        assert_eq!((&view_kept[4..9], &view_kept[10..]), (&desc[4..9], &desc[10..]));
        // Word 3 of an unordered view's description is part of it: another
        // shape. Its last word is filler too.
        create_placed_uav(null, &resource, &desc, at);
        (desc[3], desc[9]) = (desc[3] ^ 1, desc[9] ^ 1);
        create_placed_uav(null, &resource, &desc, beside);
        assert_ne!(words(at), words(beside));
        let mut said = desc;
        said[9] = 0;
        assert_eq!(view(beside), Some(View::Placed(true, kept, said)));
        // A word that is part of a shader view (its first level) too.
        desc[4] ^= 1;
        create_placed_srv(null, &resource, &desc, at);
        desc[4] ^= 1;
        create_placed_srv(null, &resource, &desc, beside);
        assert_ne!(words(at), words(beside));
        // A 2D array's last word is its least level of detail: kept.
        desc[1] = 5;
        create_placed_srv(null, &resource, &desc, at);
        desc[9] ^= 1;
        create_placed_srv(null, &resource, &desc, beside);
        assert_ne!(words(at), words(beside));
        said = desc;
        said[3] = 0;
        assert_eq!(view(beside), Some(View::Placed(false, kept, said)));
    }

    /// A released resource's views go stale; the same view of a new
    /// resource at the same address is a new record.
    #[test]
    fn views_die_with_their_resource() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let (a, b) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 32);
        let resource = 0x7777_0000usize as This;
        create_srv(null, resource, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Shader(0x7777_0000, None)));
        forget(0x7777_0000);
        assert_eq!(view(a), None, "stale after release");
        create_srv(null, resource, std::ptr::null(), b);
        assert_eq!(view(b), Some(View::Shader(0x7777_0000, None)));
        // SAFETY: both slots are in `slots`.
        let (old, new) = unsafe { (*(a as *const [u32; 3]), *(b as *const [u32; 3])) };
        assert_ne!(old, new, "a new generation");
    }

    /// The title places its shader-visible heap in its own memory
    /// and checks that the heap starts there.
    #[test]
    fn placed_heaps_start_at_the_title_memory() {
        let mut memory = vec![0u8; 64 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        assert_eq!(
            create(
                desc,
                Placement {
                    address,
                    size: 64 * 32
                },
                &mut h
            ),
            S_OK
        );
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), address);
        // A size that does not match the count is not a placement.
        assert_eq!(create(desc, Placement { address, size: 7 }, &mut h), S_OK);
        assert_ne!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
    }

    /// A released view heap takes its run of title-memory views with it
    /// (`memory::forget_heap`): a heap made later at the same address starts
    /// its own run. (Fork pull request #589's test, on our helpers.)
    #[cfg(windows)]
    #[test]
    fn a_released_view_heap_forgets_its_title_memory_run() {
        let Some(device) = crate::host::host() else {
            return; // no host GPU
        };
        let list = device.new_list().expect("a host list");
        let desc = HeapDesc {
            kind: VIEWS_KIND,
            count: 64,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h: This = std::ptr::null();
        assert_eq!(create(desc, none, &mut h), S_OK);
        let host_heap = heap(h).host().expect("a host view heap");
        let key = std::ptr::from_ref(host_heap) as usize;
        assert!(crate::memory::ready(&list, host_heap, &[], &[], &[], crate::memory::Use::Reads, None).is_some());
        assert!(crate::memory::has_run(key), "the heap's run is kept while it lives");
        assert_eq!(object::release_boxed::<Heap>(h), 0);
        assert!(!crate::memory::has_run(key), "a released heap's run is forgotten");
    }

    /// Occupied and free counts by kind, against a table this test owns.
    #[cfg(windows)]
    #[test]
    fn kind_breakdown_is_exact_for_a_known_table() {
        let list = vec![
            (View::Shader(1, None), 1),
            (View::Shader(2, None), 1),
            (View::Shader(3, None), 1),
            (View::Sampler([0; 13]), 1),
            (View::Placed(false, [0; 14], [0; 16]), 1),
            (View::Placed(true, [1; 14], [1; 16]), 1),
            (View::Constant(0x1000, 256), 2),
        ];
        let free = vec![6, 2];
        let views = Views {
            list,
            index: Default::default(),
            by_resource: Default::default(),
            free,
            shapes: Vec::new(),
            shape_index: Default::default(),
        };
        let (summary, by_kind) = kept(&views);
        let each = size_of::<(View, u32)>() as u64;
        assert_eq!(by_kind[KIND_CONSTANT], [0, 1, each]);
        assert_eq!(by_kind[KIND_SHADER], [2, 1, 3 * each]);
        assert_eq!(by_kind[KIND_UNORDERED], [0, 0, 0]);
        assert_eq!(by_kind[KIND_SAMPLER], [1, 0, each]);
        assert_eq!(by_kind[KIND_CONSOLE_SAMPLER], [0, 0, 0]);
        assert_eq!(by_kind[KIND_PLACED], [2, 0, 2 * each]);
        let (occupied, freed): (u64, u64) = by_kind.iter().fold((0, 0), |(o, f), k| (o + k[0], f + k[1]));
        assert_eq!(occupied + freed, views.list.len() as u64);
        assert_eq!(freed, views.free.len() as u64);
        assert_eq!(summary[0], views.list.len() as u64);
        assert_eq!(summary[1], views.free.len() as u64);
        assert_eq!(summary[3], by_kind[KIND_CONSTANT][0] + by_kind[KIND_CONSTANT][1]);
        assert_eq!(summary[4], by_kind[KIND_PLACED][0] + by_kind[KIND_PLACED][1]);
    }

    /// Reusing a freed slot for another kind moves the count. The table's
    /// length does not change.
    #[cfg(windows)]
    #[test]
    fn kind_breakdown_covers_all_six_kinds_and_cross_kind_recycling() {
        let each = size_of::<(View, u32)>() as u64;
        let before = Views {
            list: vec![
                (View::Shader(0x9_1000, None), 1),
                (View::Unordered(0x9_2000, 0, None), 1),
                (View::Sampler([7; 13]), 1),
                (View::ConsoleSampler([8; 27]), 1),
                (View::Placed(false, [0; 14], [0; 16]), 1),
                (View::Constant(0x9_3000, 256), 1),
            ],
            index: Default::default(),
            by_resource: Default::default(),
            free: Vec::new(),
            shapes: Vec::new(),
            shape_index: Default::default(),
        };
        let (_, before_kind) = kept(&before);
        for k in [KIND_CONSTANT, KIND_SHADER, KIND_UNORDERED, KIND_SAMPLER, KIND_CONSOLE_SAMPLER, KIND_PLACED] {
            assert_eq!(before_kind[k], [1, 0, each]);
        }
        let before_len = before.list.len();
        let mut after = before;
        after.list[0] = (View::Unordered(0x9_4000, 0, None), 2);
        after.free = vec![4];
        let (_, after_kind) = kept(&after);
        assert_eq!(after_kind[KIND_SHADER], [0, 0, 0]);
        assert_eq!(after_kind[KIND_UNORDERED], [2, 0, 2 * each]);
        assert_eq!(after_kind[KIND_PLACED], [0, 1, each]);
        assert_eq!(after_kind[KIND_SAMPLER], before_kind[KIND_SAMPLER]);
        assert_eq!(after_kind[KIND_CONSOLE_SAMPLER], before_kind[KIND_CONSOLE_SAMPLER]);
        assert_eq!(after_kind[KIND_CONSTANT], before_kind[KIND_CONSTANT]);
        let shader_total = |k: &[[u64; 3]; 6]| k[KIND_SHADER][0] + k[KIND_SHADER][1];
        assert!(shader_total(&after_kind) < shader_total(&before_kind));
        assert_eq!(before_len, after.list.len());
    }
}
