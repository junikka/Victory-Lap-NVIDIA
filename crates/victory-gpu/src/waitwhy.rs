//! Why the textures picked by index wait (ticket 129,
//! `VICTORY_TEXTURE_WAIT_WHY=1`, off unless set; a line beside the
//! residency line every 5 s, and nothing else changes).
//!
//! The residency line's "index waiting" is every slot of the queue
//! (`descriptor/host.rs` `TextureQueue`), and the queue line's "waiting for
//! their memory" counts every slot looked at before that has no part
//! uploaded: with `VICTORY_RECHECK_LONG` set, that is mostly slots uploaded
//! whole and parked for their next look, not slots waiting for anything.
//! This line splits the queue by what each slot waits for:
//!
//! - rechecking: uploaded whole, parked for a look at its memory (due now,
//!   later; of them past the fifth look, `VICTORY_RECHECK_LONG`);
//! - no memory: the title has not mapped the texture's memory (by the bytes
//!   the look asked for, how long the slot has waited, and what is known of
//!   its use: read through a table, shown, unknown);
//! - in part: uploaded with what was mapped, waiting for more;
//! - held for room: refused while the card had no room;
//! - queued: written and not looked at yet (due now: waiting for the pass;
//!   later: a table's slot or a write due some frames on).
//!
//! And why passes over the queue ended (all served, the frame's bytes, the
//! pass's time, the worker asked to stop, the RAM guard, the card's room)
//! with the due slots each left behind; bytes uploaded again after a look
//! found a texture changed; and the title's streaming requests as the
//! DirectStorage workers finished them ([`note_disk_window`]).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether the line is on (`VICTORY_TEXTURE_WAIT_WHY=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_TEXTURE_WAIT_WHY").is_ok_and(|v| v.trim() == "1"))
}

/// What the slots of one heap's queue wait for ([`crate::descriptor::host::TextureQueue::why`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WaitWhy {
    pub(crate) slots: u64,
    /// Uploaded whole, parked for a look: due now, later.
    pub(crate) rechecking: [u64; 2],
    /// Of them, past the fifth look (`VICTORY_RECHECK_LONG`).
    pub(crate) rechecking_long: u64,
    /// Looked at, memory not mapped.
    pub(crate) unmapped: u64,
    /// Of them by the bytes asked for ([`size_bucket`]; unknown ones not counted).
    pub(crate) unmapped_size: [u64; 4],
    /// By how long they have waited ([`age_bucket`]).
    pub(crate) unmapped_age: [u64; 4],
    /// By use: read through a table, shown, unknown.
    pub(crate) unmapped_use: [u64; 3],
    pub(crate) in_part: u64,
    pub(crate) held: u64,
    /// Never looked at: due now, later; and the frames the oldest due one
    /// has waited since its write.
    pub(crate) queued: [u64; 2],
    pub(crate) queued_oldest: u64,
}

/// The size bucket of a texture's bytes: under 256 KiB, under 4 MiB, under
/// 16 MiB, 16 MiB and over.
pub(crate) fn size_bucket(bytes: u64) -> usize {
    match bytes {
        0..0x4_0000 => 0,
        0x4_0000..0x40_0000 => 1,
        0x40_0000..0x100_0000 => 2,
        _ => 3,
    }
}

/// The bucket of a wait in milliseconds: under 1 s, under 10 s, under 60 s,
/// 60 s and over.
pub(crate) fn age_bucket(ms: u64) -> usize {
    match ms {
        0..1000 => 0,
        1000..10_000 => 1,
        10_000..60_000 => 2,
        _ => 3,
    }
}

/// Why a pass over the queue ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum End {
    /// Every slot handed to it was looked at.
    Served,
    /// The frame's bytes were spent.
    Bytes,
    /// The pass's time ran out.
    Time,
    /// Its worker was asked to stop (a recording thread wants the heap).
    Stop,
    /// The RAM guard (`crate::residency`) refused or put off staging.
    Ram,
    /// The card had no room and too many slots were refused.
    Room,
}

/// Passes by how they ended, and the due slots they left behind, since the start.
static ENDS: [[AtomicU64; 2]; 6] = [const { [const { AtomicU64::new(0) }; 2] }; 6];

/// Bytes uploaded again after a look found a texture changed, since the start.
static CHANGED_BYTES: AtomicU64 = AtomicU64::new(0);

/// A pass ended so, with `left` of the slots handed to it not looked at.
pub(crate) fn pass_ended(end: End, left: u64) {
    if on() {
        ENDS[end as usize][0].fetch_add(1, Relaxed);
        ENDS[end as usize][1].fetch_add(left, Relaxed);
    }
}

/// `bytes` were uploaded again after a look found the texture changed.
pub(crate) fn changed_upload(bytes: u64) {
    if on() {
        CHANGED_BYTES.fetch_add(bytes, Relaxed);
    }
}

/// The last 5 s window of the title's streaming requests the DirectStorage
/// workers finished: requests, bytes, seconds, worker seconds busy, worker
/// seconds decompressing BCPack, requests waiting for a worker.
static DISK: Mutex<Option<[f64; 6]>> = Mutex::new(None);

/// Called by the DirectStorage workers at the end of each window of about
/// 5 s (`victory-hle` `dstorage.rs`): kept for the line, nothing else.
pub fn note_disk_window(requests: u64, bytes: u64, secs: f64, busy: f64, bcpack: f64, waiting: u64) {
    if on() {
        *DISK.lock().unwrap_or_else(|e| e.into_inner()) =
            Some([requests as f64, bytes as f64, secs, busy, bcpack, waiting as f64]);
    }
}

/// The queue as last walked ([`note_queue`]), and when (`crate::host::now_ms`).
static QUEUE: Mutex<Option<(u64, WaitWhy)>> = Mutex::new(None);

/// Whether a pass at `now` (ms) should walk its queue for the line: at most
/// once a second, or when its queue holds more slots than the one kept
/// (several heaps: the largest is the one the residency line counts).
pub(crate) fn walk_due(now: u64, slots: u64) -> bool {
    on() && QUEUE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_none_or(|(at, kept)| now.saturating_sub(at) >= 1000 || slots > kept.slots)
}

/// Keeps a queue's walk for the line.
pub(crate) fn note_queue(now: u64, why: WaitWhy) {
    *QUEUE.lock().unwrap_or_else(|e| e.into_inner()) = Some((now, why));
}

/// The line, if it is on.
pub(crate) fn line() -> Option<String> {
    if !on() {
        return None;
    }
    let queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner()).map(|(_, why)| why);
    let ends = ENDS.each_ref().map(|e| [e[0].load(Relaxed), e[1].load(Relaxed)]);
    let disk = *DISK.lock().unwrap_or_else(|e| e.into_inner());
    Some(format_line(queue, ends, CHANGED_BYTES.load(Relaxed), disk))
}

fn format_line(queue: Option<WaitWhy>, ends: [[u64; 2]; 6], changed: u64, disk: Option<[f64; 6]>) -> String {
    let queue = match queue {
        None => "queue not walked yet".to_owned(),
        Some(w) => format!(
            "{} slots: rechecking {} (due {}, later {}; past the fifth look {}), no memory {} (by size <256K {} <4M {} \
             <16M {} more {}; waited <1s {} <10s {} <60s {} more {}; read {} shown {} unknown {}), in part {}, \
             held for room {}, queued {} (due {}, later {}; the oldest due written {} frames ago)",
            w.slots,
            w.rechecking[0] + w.rechecking[1],
            w.rechecking[0],
            w.rechecking[1],
            w.rechecking_long,
            w.unmapped,
            w.unmapped_size[0],
            w.unmapped_size[1],
            w.unmapped_size[2],
            w.unmapped_size[3],
            w.unmapped_age[0],
            w.unmapped_age[1],
            w.unmapped_age[2],
            w.unmapped_age[3],
            w.unmapped_use[0],
            w.unmapped_use[1],
            w.unmapped_use[2],
            w.in_part,
            w.held,
            w.queued[0] + w.queued[1],
            w.queued[0],
            w.queued[1],
            w.queued_oldest,
        ),
    };
    let names = ["all served", "bytes", "time", "stop", "ram guard", "room"];
    let passes: Vec<String> =
        names.iter().zip(ends).map(|(name, [n, left])| format!("{name} {n} (left {left})")).collect();
    let disk = match disk {
        None => "none finished yet".to_owned(),
        Some([requests, bytes, secs, busy, bcpack, waiting]) if secs > 0.0 => format!(
            "{:.0} requests, {:.1} MB/s, workers busy {:.2} (bcpack {:.2}), waiting {:.0}",
            requests / secs,
            bytes / secs / 1e6,
            busy / secs,
            bcpack / secs,
            waiting
        ),
        Some(_) => "none finished yet".to_owned(),
    };
    format!(
        "textures picked by index, why they wait (VICTORY_TEXTURE_WAIT_WHY=1): {queue}; passes ended by {}; \
         uploaded again after a look found them changed {} MB; the title's streaming {disk} (counts since the start)",
        passes.join(", "),
        changed >> 20,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_split_at_their_edges() {
        assert_eq!([0, 0x3_ffff, 0x4_0000, 0x3f_ffff, 0x40_0000, 0xff_ffff, 0x100_0000].map(size_bucket), [0, 0, 1, 1, 2, 2, 3]);
        assert_eq!([0, 999, 1000, 9999, 10_000, 59_999, 60_000, u64::MAX].map(age_bucket), [0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn the_line_names_every_reason() {
        let why = WaitWhy {
            slots: 40,
            rechecking: [2, 20],
            rechecking_long: 15,
            unmapped: 10,
            unmapped_size: [4, 3, 2, 1],
            unmapped_age: [1, 2, 3, 4],
            unmapped_use: [1, 0, 9],
            in_part: 3,
            held: 1,
            queued: [3, 1],
            queued_oldest: 7,
        };
        let mut ends = [[0; 2]; 6];
        ends[End::Bytes as usize] = [2, 300];
        let line = format_line(Some(why), ends, 5 << 20, Some([50.0, 100e6, 5.0, 4.0, 1.0, 12.0]));
        assert_eq!(
            line,
            "textures picked by index, why they wait (VICTORY_TEXTURE_WAIT_WHY=1): 40 slots: rechecking 22 (due 2, \
             later 20; past the fifth look 15), no memory 10 (by size <256K 4 <4M 3 <16M 2 more 1; waited <1s 1 <10s 2 \
             <60s 3 more 4; read 1 shown 0 unknown 9), in part 3, held for room 1, queued 4 (due 3, later 1; the oldest \
             due written 7 frames ago); passes ended by all served 0 (left 0), bytes 2 (left 300), time 0 (left 0), \
             stop 0 (left 0), ram guard 0 (left 0), room 0 (left 0); uploaded again after a look found them changed 5 MB; \
             the title's streaming 10 requests, 20.0 MB/s, workers busy 0.80 (bcpack 0.20), waiting 12 (counts since \
             the start)"
        );
        assert!(format_line(None, [[0; 2]; 6], 0, None).contains("queue not walked yet"));
        assert!(format_line(None, [[0; 2]; 6], 0, Some([0.0; 6])).contains("streaming none finished yet"));
    }
}
