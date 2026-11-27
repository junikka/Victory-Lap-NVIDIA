//! The title's barrier groups, read (`VICTORY_GROUP_WAITS=count` or `=1`; off
//! by default: they are not looked at, and each call records one wait for
//! every earlier shader write, as before).
//!
//! The title's enhanced barriers come as groups of global, texture and buffer
//! barriers. The host needs none of their layouts (every host use of a title
//! texture moves its host copy to the state it needs first), only their
//! ordering of shader writes, so each call became one unordered-access
//! barrier that names no resource: the card lets every shader write it has
//! under way run out. The title calls some 650 times a frame, between its
//! draws, and NVIDIA lists each such wait as a point where the card's queue
//! runs empty ("Advanced API Performance: Barriers", "Async Compute and
//! Overlap").
//!
//! A call whose groups hold nothing but texture barriers that name no
//! unordered access on either side (a target that becomes a shader's input, a
//! depth buffer read after it was written) orders no shader write: what the
//! host's own state change of that texture does not order, the wait did not
//! either. With `=1` no wait is recorded for such a call. `=count` only reads
//! and counts, for the 5 s line. Anything not understood (a kind of group
//! that is not known, counts out of range, memory that cannot be read) keeps
//! its wait.
//!
//! The layout read is the host interface's own (`D3D12_BARRIER_GROUP`, 16
//! bytes: kind, count, pointer; `D3D12_TEXTURE_BARRIER`, 64 bytes, with its
//! accesses at 8 and 12). Whether the title's interface lays them out the same
//! is not known from here: the line's count of calls that could not be read
//! says so in the first run.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// What a call's groups hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Holds {
    /// Only texture barriers that name no unordered access: no shader write
    /// is ordered by them.
    Textures,
    /// A global or buffer barrier, or a texture barrier that names unordered
    /// access (or names no access at all): the wait stands.
    Orders,
    /// Not understood: the wait stands.
    Unread,
}

/// The barriers of a call, by kind: global, buffer, texture, and of the
/// texture barriers those that keep the wait.
pub(crate) type Seen = [u32; 4];

const GROUP: usize = 16;
const TEXTURE_BARRIER: usize = 64;
const UNORDERED_ACCESS: u32 = 0x10;
/// More groups in a call, or barriers in a group, than any title passes: not
/// a group array.
const MOST_GROUPS: u32 = 64;
const MOST_BARRIERS: u32 = 4096;

/// What the `count` groups at `groups` hold. `read` gives the bytes at an
/// address of the title's, if they can be read.
pub(crate) fn classify(count: u32, groups: u64, read: &dyn Fn(u64, usize) -> Option<Vec<u8>>) -> (Holds, Seen) {
    let mut seen = [0u32; 4];
    if count == 0 || count > MOST_GROUPS || groups == 0 {
        return (Holds::Unread, seen);
    }
    let Some(bytes) = read(groups, count as usize * GROUP) else {
        return (Holds::Unread, seen);
    };
    let dword = |b: &[u8], at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
    let mut orders = false;
    for group in bytes.chunks_exact(GROUP) {
        let (kind, n) = (dword(group, 0), dword(group, 4));
        let at = u64::from(dword(group, 8)) | u64::from(dword(group, 12)) << 32;
        if kind > 2 || n > MOST_BARRIERS || (n > 0 && at == 0) {
            return (Holds::Unread, [0; 4]);
        }
        if n == 0 {
            continue;
        }
        match kind {
            0 => {
                seen[0] += n;
                orders = true;
            }
            2 => {
                seen[1] += n;
                orders = true;
            }
            _ => {
                let Some(barriers) = read(at, n as usize * TEXTURE_BARRIER) else {
                    return (Holds::Unread, [0; 4]);
                };
                for barrier in barriers.chunks_exact(TEXTURE_BARRIER) {
                    let (before, after) = (dword(barrier, 8), dword(barrier, 12));
                    seen[2] += 1;
                    // (No access named, 0, is any access the layout allows.)
                    if before == 0 || after == 0 || (before | after) & UNORDERED_ACCESS != 0 {
                        seen[3] += 1;
                        orders = true;
                    }
                }
            }
        }
    }
    (if orders { Holds::Orders } else { Holds::Textures }, seen)
}

/// `VICTORY_GROUP_WAITS`: 0 unset (the groups are not read), 1 `count` (read
/// and counted), 2 `1` (and no wait recorded for calls that order no shader
/// write).
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| mode_of(std::env::var("VICTORY_GROUP_WAITS").ok().as_deref()))
}

fn mode_of(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        Some("count") => 1,
        Some("1") => 2,
        _ => 0,
    }
}

/// Since the last line: calls read; those that hold only texture barriers
/// without unordered access; those not understood; waits left out; and the
/// barriers seen: global, buffer, texture, texture ones that keep the wait.
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// A call was read as `holds`, with `seen`; whether its wait is left out.
fn note(holds: Holds, seen: Seen, mode: u8) -> bool {
    STATS[0].fetch_add(1, Relaxed);
    match holds {
        Holds::Textures => STATS[1].fetch_add(1, Relaxed),
        Holds::Unread => STATS[2].fetch_add(1, Relaxed),
        Holds::Orders => 0,
    };
    for (stat, n) in STATS[4..].iter().zip(seen) {
        stat.fetch_add(u64::from(n), Relaxed);
    }
    let left = holds == Holds::Textures && mode == 2;
    if left {
        STATS[3].fetch_add(1, Relaxed);
    }
    left
}

/// In a call's count: its wait is left out ([`kept`]).
pub(crate) const LEFT_OUT: u32 = 1 << 31;

/// The title's call with `count` groups at `groups`, where it is made (the
/// groups are the title's to change afterwards): the count, with
/// [`LEFT_OUT`] if the call's wait is left out. Unset, or with nothing to
/// read (the call played back from a list recorded by a thread of ours,
/// which read it already): `count`, untouched.
#[cfg(windows)]
pub(crate) fn kept(count: u32, groups: *const u8) -> u32 {
    let mode = mode();
    if mode == 0 || count == 0 || count & LEFT_OUT != 0 || groups.is_null() {
        return count;
    }
    // (The copies below fail instead of faulting once this was asked.)
    if !crate::safecopy::on() {
        return count;
    }
    let read = |at: u64, len: usize| {
        let mut bytes = vec![0u8; len];
        // SAFETY: our own buffer of `len` bytes; the title's side may fail.
        unsafe { crate::safecopy::copy(bytes.as_mut_ptr(), at as *const u8, len) }.then_some(bytes)
    };
    let (holds, seen) = classify(count, groups as u64, &read);
    if note(holds, seen, mode) { count | LEFT_OUT } else { count }
}

/// The line's text, of `stats` over `frames` frames.
fn text(stats: [u64; 8], frames: f64, mode: u8) -> String {
    let n = |i: usize| stats[i] as f64 / frames;
    format!(
        "{:.0} calls read, {:.0} of them only texture barriers that name no unordered access ({:.0} waits left out: \
         VICTORY_GROUP_WAITS=1 leaves those out, now {}), {:.0} not understood; barriers in them: {:.0} global, {:.0} buffer, \
         {:.0} texture ({:.0} of those name unordered access or no access)",
        n(0),
        n(1),
        n(3),
        if mode == 2 { "on" } else { "off" },
        n(2),
        n(4),
        n(5),
        n(6),
        n(7),
    )
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    if mode() == 0 {
        return;
    }
    let stats: [u64; 8] = std::array::from_fn(|i| STATS[i].swap(0, Relaxed));
    tracing::info!("the title's barrier groups per frame (VICTORY_GROUP_WAITS): {}", text(stats, frames, mode()));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Title memory for a test: what lies at which address.
    struct Memory(Vec<(u64, Vec<u8>)>);

    impl Memory {
        fn read(&self, at: u64, len: usize) -> Option<Vec<u8>> {
            let (start, bytes) = self.0.iter().find(|(start, bytes)| *start <= at && at + len as u64 <= start + bytes.len() as u64)?;
            let from = (at - start) as usize;
            Some(bytes[from..from + len].to_vec())
        }
    }

    fn group(kind: u32, n: u32, at: u64) -> Vec<u8> {
        [kind.to_le_bytes().to_vec(), n.to_le_bytes().to_vec(), at.to_le_bytes().to_vec()].concat()
    }

    /// A texture barrier from access `before` to `after`.
    fn texture(before: u32, after: u32) -> Vec<u8> {
        let mut b = vec![0u8; TEXTURE_BARRIER];
        // (Sync before and after: draw, non-pixel shading.)
        b[0..4].copy_from_slice(&4u32.to_le_bytes());
        b[4..8].copy_from_slice(&0x20u32.to_le_bytes());
        b[8..12].copy_from_slice(&before.to_le_bytes());
        b[12..16].copy_from_slice(&after.to_le_bytes());
        b[24..32].copy_from_slice(&0x1234_5678_9abc_u64.to_le_bytes());
        b
    }

    const RENDER_TARGET: u32 = 0x8;
    const SHADER_RESOURCE: u32 = 0x80;
    const DEPTH_WRITE: u32 = 0x20;
    const NO_ACCESS: u32 = 0x8000_0000;

    fn holds(groups: Vec<u8>, barriers: Vec<(u64, Vec<u8>)>) -> (Holds, Seen) {
        let count = (groups.len() / GROUP) as u32;
        let mut memory = Memory(barriers);
        memory.0.push((0x1000, groups));
        classify(count, 0x1000, &|at, len| memory.read(at, len))
    }

    #[test]
    fn texture_barriers_without_unordered_access_order_no_shader_write() {
        // A target becomes a shader's input; a depth buffer too; a texture is made.
        let barriers = [texture(RENDER_TARGET, SHADER_RESOURCE), texture(DEPTH_WRITE, SHADER_RESOURCE), texture(NO_ACCESS, RENDER_TARGET)].concat();
        let got = holds([group(1, 2, 0x2000), group(1, 1, 0x2000 + 128)].concat(), vec![(0x2000, barriers)]);
        assert_eq!(got, (Holds::Textures, [0, 0, 3, 0]));
        // A group of no barriers changes nothing.
        let got = holds([group(1, 1, 0x2000), group(2, 0, 0)].concat(), vec![(0x2000, texture(RENDER_TARGET, SHADER_RESOURCE))]);
        assert_eq!(got, (Holds::Textures, [0, 0, 1, 0]));
    }

    #[test]
    fn unordered_access_on_either_side_keeps_the_wait() {
        for (before, after) in [(UNORDERED_ACCESS, SHADER_RESOURCE), (SHADER_RESOURCE, UNORDERED_ACCESS), (UNORDERED_ACCESS, UNORDERED_ACCESS)] {
            let barriers = [texture(RENDER_TARGET, SHADER_RESOURCE), texture(before, after)].concat();
            let got = holds(group(1, 2, 0x2000), vec![(0x2000, barriers)]);
            assert_eq!(got, (Holds::Orders, [0, 0, 2, 1]), "{before:#x} to {after:#x}");
        }
        // No access named (any the layout allows): kept too.
        let got = holds(group(1, 1, 0x2000), vec![(0x2000, texture(0, SHADER_RESOURCE))]);
        assert_eq!(got, (Holds::Orders, [0, 0, 1, 1]));
        // Unordered access with other bits beside it.
        let got = holds(group(1, 1, 0x2000), vec![(0x2000, texture(UNORDERED_ACCESS | SHADER_RESOURCE, SHADER_RESOURCE))]);
        assert_eq!(got.0, Holds::Orders);
    }

    #[test]
    fn buffer_and_global_barriers_keep_the_wait() {
        // (Their barriers are not read: none lie at the addresses.)
        assert_eq!(holds(group(2, 3, 0x5000), vec![]), (Holds::Orders, [0, 3, 0, 0]));
        assert_eq!(holds(group(0, 1, 0x5000), vec![]), (Holds::Orders, [1, 0, 0, 0]));
        let got = holds([group(1, 1, 0x2000), group(2, 1, 0x5000)].concat(), vec![(0x2000, texture(RENDER_TARGET, SHADER_RESOURCE))]);
        assert_eq!(got, (Holds::Orders, [0, 1, 1, 0]));
    }

    #[test]
    fn what_is_not_understood_keeps_the_wait() {
        // A kind that is none of the three; a count out of range; barriers
        // without an address; barriers that cannot be read; no groups.
        assert_eq!(holds(group(3, 1, 0x2000), vec![]).0, Holds::Unread);
        assert_eq!(holds(group(1, MOST_BARRIERS + 1, 0x2000), vec![]).0, Holds::Unread);
        assert_eq!(holds(group(1, 1, 0), vec![]).0, Holds::Unread);
        assert_eq!(holds(group(1, 2, 0x2000), vec![(0x2000, texture(RENDER_TARGET, SHADER_RESOURCE))]), (Holds::Unread, [0; 4]));
        let none = |_: u64, _: usize| None;
        assert_eq!(classify(1, 0x1000, &none).0, Holds::Unread);
        assert_eq!(classify(0, 0x1000, &none).0, Holds::Unread);
        assert_eq!(classify(MOST_GROUPS + 1, 0x1000, &none).0, Holds::Unread);
        assert_eq!(classify(1, 0, &none).0, Holds::Unread);
    }

    #[test]
    fn only_one_asks_for_waits_to_be_left_out_and_only_of_calls_that_order_nothing() {
        assert_eq!([mode_of(None), mode_of(Some("0")), mode_of(Some("count")), mode_of(Some(" 1 ")), mode_of(Some("yes"))], [0, 0, 1, 2, 0]);
        // (The counts are the process's: only what this test adds is looked at,
        // through what `note` answers.)
        assert!(note(Holds::Textures, [0, 0, 2, 0], 2));
        assert!(!note(Holds::Textures, [0, 0, 2, 0], 1));
        assert!(!note(Holds::Orders, [0, 1, 2, 0], 2));
        assert!(!note(Holds::Unread, [0; 4], 2));
    }

    #[test]
    fn the_line_says_what_the_groups_hold() {
        let t = text([1300, 900, 20, 900, 0, 380, 2600, 150], 2.0, 2);
        assert_eq!(
            t,
            "650 calls read, 450 of them only texture barriers that name no unordered access (450 waits left out: \
             VICTORY_GROUP_WAITS=1 leaves those out, now on), 10 not understood; barriers in them: 0 global, 190 buffer, 1300 \
             texture (75 of those name unordered access or no access)"
        );
    }
}
