//! How much of what windows of title memory with no card side take holds
//! what they last copied for its 4 KiB page (`VICTORY_WINDOW_SKIP_SAME=sample`,
//! [`mode`]; off by default). Measured only: nothing is left out.
//!
//! A window outside buffers whose one buffer the CPU fills (in the PC's
//! memory, or in a GPU upload heap with `VICTORY_WINDOW_UPLOAD=1`) has
//! nothing beside it to compare with, so every page the write watch or the
//! guards name is copied into it whole. A page written need not hold other
//! bytes: the title rewrites constant data. Measured in AnyXboxX: 67 to 73
//! MB a frame of what such windows took were the bytes they had taken last
//! time.
//!
//! With `sample` each window keeps a 64-bit print of the bytes it last
//! copied for the 4 KiB pages it samples ([`page_print`], 8 bytes a page:
//! 8 KiB for a window of 4 MiB); after each copy one page in 16 is read
//! again and compared with its print, and the frame line says how many MB a
//! frame held what the window last had. Upstream's `1`, which leaves such
//! pages out, is not ported here (ticket 140: the measure first); `1` is off.
//!
//! What leaving pages out would rest on: the window's buffer holds what the
//! window last copied for each page. Nothing but its copies writes it: no shader writes a
//! window (run 91 and since), and a window that shaders are said to write
//! (`Shadow::writing`) compares nothing from then on. A window let go takes
//! its prints with it, so one made again for the same memory starts with
//! none; a page copied only in part has none. A page that cannot be read is
//! read as zeros (`crate::host::copy_mapped`), which is what the copy as
//! before stores in the buffer, so its print is that of the buffer's bytes.
//! shortcut: two pages that differ and give the same print (about once in
//! 2^64) leave the window with the older bytes until the title writes the
//! page again.
//!
//! Ported from kitty b93b839d5, itself from KomanderDG88/VIniVIdiVIci
//! a53b093d and AnyXboxX PR #340 (VICTORY_WINDOW_SKIP_SAME, by Skyfrye),
//! where it is for windows with a card side and no CPU side; here for the
//! windows with no card side, whose buffer is the one the CPU fills.

use std::sync::atomic::{AtomicU64, Ordering};

/// What a window with no card side does with a 4 KiB page it takes that
/// holds the bytes it last copied for the page ([`mode`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SkipSame {
    /// Unset or `0`, the default: copies it again and compares nothing, the
    /// code path as before the switch.
    Off,
    /// `sample`: copies it again; one page in 16, picked by its address
    /// ([`sampled_page`]), is read again and compared, for the frame line.
    Sample,
}

/// [`SkipSame`] for a setting: only `sample` (trimmed) samples.
fn mode_value(value: Option<&str>) -> SkipSame {
    match value.map(str::trim) {
        Some("sample") => SkipSame::Sample,
        _ => SkipSame::Off,
    }
}

/// `VICTORY_WINDOW_SKIP_SAME` ([`SkipSame`]; read once). (AnyXboxX samples
/// when unset; here unset is off, so a run without the switch does no work
/// of it at all.)
pub(crate) fn mode() -> SkipSame {
    static MODE: std::sync::OnceLock<SkipSame> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| mode_value(std::env::var("VICTORY_WINDOW_SKIP_SAME").ok().as_deref()))
}

/// The pages windows keep a print of.
const PRINTED: usize = 4 << 10;

/// Bytes of the 4 KiB pages windows took since the last frame line, by what
/// they were ([`Sent`]): the bytes the window last copied there, other
/// bytes, nothing to compare with (never copied, a page taken in part, or a
/// window shaders write). With [`SkipSame::Sample`] the pages compared, 16
/// times over: an estimate.
static SAME: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Bytes of title memory windows took since the last frame line: copied,
/// and left out (always 0 here: upstream's `1` is not ported).
static TAKEN: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Bytes of the windows' prints alive.
static PRINT_BYTES: AtomicU64 = AtomicU64::new(0);

/// With the frame line, with the switch set: what windows with no card side
/// took of title memory per frame over `frames`, copied and left out, and
/// against what they last copied ([`SAME`]). Taken and reset.
pub(crate) fn line(frames: f64) {
    let skip = match mode() {
        SkipSame::Off => return,
        SkipSame::Sample => "sample: copied again, one 4 KiB page in 16 compared",
    };
    let mb = |n: &AtomicU64| n.swap(0, Ordering::Relaxed) as f64 / 1e6 / frames;
    let [sent, left] = TAKEN.each_ref().map(mb);
    let [same, other, unknown] = SAME.each_ref().map(mb);
    tracing::info!(
        "title memory windows with no card side took per frame: {sent:.1} MB copied, {left:.1} MB left out; of what they \
         took, {same:.1} MB held the bytes the window last copied for their 4 KiB page, {other:.1} MB other bytes, \
         {unknown:.1} MB nothing to compare with; prints {:.1} MB (VICTORY_WINDOW_SKIP_SAME={skip})",
        PRINT_BYTES.load(Ordering::Relaxed) as f64 / 1e6
    );
}

/// A window made prints for `n` pages.
pub(crate) fn prints_made(n: usize) {
    PRINT_BYTES.fetch_add(n as u64 * 8, Ordering::Relaxed);
}

/// A window with prints for `n` pages is gone.
pub(crate) fn prints_gone(n: usize) {
    PRINT_BYTES.fetch_sub(n as u64 * 8, Ordering::Relaxed);
}

/// A window took `sent` bytes of title memory into its buffer and left out
/// `left`.
pub(crate) fn taken(sent: usize, left: usize) {
    for (count, bytes) in TAKEN.iter().zip([sent, left]) {
        if bytes != 0 {
            count.fetch_add(bytes as u64, Ordering::Relaxed);
        }
    }
}

/// A window shaders are said to write took `bytes` and compared nothing.
pub(crate) fn unknown(bytes: usize) {
    SAME[Sent::Unknown as usize].fetch_add(bytes as u64, Ordering::Relaxed);
}

/// [`SAME`] takes `tally`.
pub(crate) fn tally_add(tally: &[u64; 3]) {
    for (n, &bytes) in SAME.iter().zip(tally) {
        if bytes != 0 {
            n.fetch_add(bytes, Ordering::Relaxed);
        }
    }
}

/// What the bytes of a 4 KiB page a window takes are against its print of
/// what it last copied for the page: the index into [`SAME`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sent {
    Same = 0,
    Other = 1,
    Unknown = 2,
}

/// Keys of [`page_print`]: 24 for the words of the 64-byte stripes (stripe
/// `n` of a 1 KiB block takes keys `n` to `n + 7`), then 8 to start the
/// lanes with and to mix them by. SplitMix64's numbers from 0.
const PRINT_KEYS: [u64; 32] = {
    let mut keys = [0u64; 32];
    let (mut x, mut i) = (0u64, 0);
    while i < 32 {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        keys[i] = z ^ (z >> 31);
        i += 1;
    }
    keys
};

/// A 64-bit print of a page's bytes: equal bytes give equal prints, and two
/// different pages the same print about once in 2^64. Eight lanes take the
/// 64-bit words of each 64-byte stripe as XXH3 accumulates them: a word is
/// added to the next lane, and the product of its halves, keyed by its
/// place in the stripe and the stripe's in its 1 KiB block, to its own; the
/// lanes are mixed after each block and folded into one at the end, with
/// the bytes left over. Some 40 GB/s on one core (AnyXboxX's measure). Not
/// a hash anyone can attack: the title does not aim at it.
fn page_print(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut lanes = [0u64; 8];
    for (i, lane) in lanes.iter_mut().enumerate() {
        *lane = PRINT_KEYS[24 + i] ^ bytes.len() as u64;
    }
    let mut stripes = bytes.chunks_exact(64);
    for (n, stripe) in (&mut stripes).enumerate() {
        let key = n % 16;
        let mut words = [0u64; 8];
        for (word, b) in words.iter_mut().zip(stripe.chunks_exact(8)) {
            *word = u64::from_le_bytes(b.try_into().unwrap_or([0; 8]));
        }
        for i in 0..8 {
            let x = words[i] ^ PRINT_KEYS[key + i];
            lanes[i ^ 1] = lanes[i ^ 1].wrapping_add(words[i]);
            lanes[i] = lanes[i].wrapping_add((x & 0xffff_ffff).wrapping_mul(x >> 32));
        }
        if key == 15 {
            for (i, lane) in lanes.iter_mut().enumerate() {
                *lane = (*lane ^ (*lane >> 47) ^ PRINT_KEYS[24 + i]).wrapping_mul(0x9e37_79b1);
            }
        }
    }
    let mut rest = stripes.remainder().chunks_exact(8);
    let words = rest.by_ref().map(|w| u64::from_le_bytes(w.try_into().unwrap_or([0; 8])));
    let mut h = K;
    for word in lanes.into_iter().chain(words) {
        h = (h ^ word).wrapping_mul(K);
        h ^= h >> 29;
    }
    for &b in rest.remainder() {
        h = (h ^ u64::from(b)).wrapping_mul(K);
    }
    h ^ (h >> 32)
}

/// Of 4 KiB page `index` of a window, `bytes` are taken (`whole`: all of
/// the page that lies in the window): what they are against the page's
/// print, which becomes theirs. A part of a page leaves it none: the buffer
/// then holds bytes of two takings.
fn page_sent(prints: &mut [u64], index: usize, whole: bool, bytes: &[u8]) -> Sent {
    let Some(print) = prints.get_mut(index) else {
        return Sent::Unknown;
    };
    if !whole {
        *print = 0;
        return Sent::Unknown;
    }
    // 0 is no print.
    let now = page_print(bytes).max(1);
    match std::mem::replace(print, now) {
        was if was == now => Sent::Same,
        0 => Sent::Unknown,
        _ => Sent::Other,
    }
}

/// The prints of a window over `[start, end)`: one per 4 KiB page that lies
/// in it.
pub(crate) fn prints_for(start: usize, end: usize) -> usize {
    (end - 1) / PRINTED - start / PRINTED + 1
}

/// Whether 4 KiB page `page` of title memory (its number) is one of the one
/// in 16 that [`SkipSame::Sample`] compares: picked by a multiply of its
/// number, so the same pages each time, spread over the 64 KiB pages and the
/// windows.
fn sampled_page(page: usize) -> bool {
    (page as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 60 == 0
}

/// [`SkipSame::Sample`]: of title memory `[from, to)` a window over
/// `[start, end)` has just copied, the 4 KiB pages [`sampled_page`] picks
/// are read by `read` (address, bytes to fill; false: not all there) and
/// compared with the window's `prints`, which take them; what they were goes
/// to `tally`, 16 times their bytes. Nothing is left out.
pub(crate) fn sample(
    (start, end): (usize, usize),
    (from, to): (usize, usize),
    prints: &mut [u64],
    tally: &mut [u64; 3],
    mut read: impl FnMut(usize, &mut [u8]) -> bool,
) {
    let first = start / PRINTED;
    let mut page_bytes = [0u8; PRINTED];
    for page in (from / PRINTED..to.div_ceil(PRINTED)).filter(|&p| sampled_page(p)) {
        let (a, b) = ((page * PRINTED).max(from), ((page + 1) * PRINTED).min(to));
        let whole = a == (page * PRINTED).max(start) && b == ((page + 1) * PRINTED).min(end);
        let bytes = &mut page_bytes[..b - a];
        // A page not all there now is as one taken in part.
        let what = page_sent(prints, page - first, whole && read(a, bytes), bytes);
        tally[what as usize] += 16 * (b - a) as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::{PRINTED, SkipSame, Sent, mode_value, page_print, page_sent, prints_for, sample, sampled_page};

    /// Title memory for the tests: bytes that differ from page to page and
    /// within each.
    fn memory(len: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    /// Equal bytes give equal prints; a page with one bit flipped, two
    /// words or two stripes swapped, a word moved by a byte, or one byte
    /// longer gives another.
    #[test]
    fn a_print_changes_with_any_change_of_the_page() {
        let page = memory(PRINTED, 7);
        let print = page_print(&page);
        assert_eq!(page_print(&page.clone()), print);
        for bit in 0..PRINTED * 8 {
            let mut other = page.clone();
            other[bit / 8] ^= 1 << (bit % 8);
            assert_ne!(page_print(&other), print, "bit {bit}");
        }
        let swapped = |a: usize, b: usize, n: usize| {
            let mut other = page.clone();
            for i in 0..n {
                other.swap(a + i, b + i);
            }
            page_print(&other)
        };
        // Words in one stripe, in two stripes of a block, in two blocks;
        // whole stripes the same keys meet in other blocks; the halves of a
        // word.
        let swaps = [(0, 8, 8), (8, 72, 8), (64, 1088, 8), (0, 64, 64), (64, 1088, 64), (0, 1024, 1024), (0, 4, 4)];
        for (a, b, n) in swaps {
            assert_ne!(swapped(a, b, n), print, "{a} {b} {n}");
        }
        // Pages of zeros and of one repeated word, their sign bits flipped
        // two at a time.
        let zeros = vec![0u8; PRINTED];
        let mut two = zeros.clone();
        two[7] ^= 0x80;
        two[15] ^= 0x80;
        assert_ne!(page_print(&two), page_print(&zeros));
        let mut ones = vec![0u8; PRINTED];
        ones.iter_mut().step_by(8).for_each(|b| *b = 1);
        assert_ne!(page_print(&ones), page_print(&zeros));
        let mut shifted = page.clone();
        shifted.rotate_right(1);
        assert_ne!(page_print(&shifted), print);
        assert_ne!(page_print(&page[..PRINTED - 1]), page_print(&page[..PRINTED - 2]));
        assert_ne!(page_print(&page[..13]), page_print(&page[..12]));
    }

    /// A page is the same when its bytes give the print last taken, unknown
    /// the first time and when taken in part (which leaves it no print),
    /// other when its bytes changed.
    #[test]
    fn a_page_is_the_same_only_after_the_same_bytes_were_taken_whole() {
        let mut prints = vec![0u64; 3];
        let (a, b) = (memory(PRINTED, 1), memory(PRINTED, 2));
        assert_eq!(page_sent(&mut prints, 1, true, &a), Sent::Unknown);
        assert_eq!(page_sent(&mut prints, 1, true, &a), Sent::Same);
        assert_eq!(page_sent(&mut prints, 1, true, &b), Sent::Other);
        assert_eq!(page_sent(&mut prints, 1, true, &a), Sent::Other);
        assert_eq!(page_sent(&mut prints, 1, false, &a[..100]), Sent::Unknown);
        assert_eq!(prints[1], 0, "a page taken in part keeps no print");
        assert_eq!(page_sent(&mut prints, 1, true, &a), Sent::Unknown);
        assert_eq!(page_sent(&mut prints, 1, true, &a), Sent::Same);
        assert_eq!((prints[0], prints[2]), (0, 0), "the other pages are not touched");
        // Past the window: nothing to compare with.
        assert_eq!(page_sent(&mut prints, 3, true, &a), Sent::Unknown);
    }

    /// Only `sample` samples; unset, `0` and upstream's `1` are off.
    #[test]
    fn only_sample_turns_the_measure_on() {
        assert_eq!(mode_value(None), SkipSame::Off);
        assert_eq!(mode_value(Some("0")), SkipSame::Off);
        assert_eq!(mode_value(Some("1")), SkipSame::Off);
        assert_eq!(mode_value(Some(" sample ")), SkipSame::Sample);
        assert_eq!(prints_for(0x40_0000, 0x80_0000), 1024);
        assert_eq!(prints_for(0x40_0100, 0x40_0200), 1);
        assert_eq!(prints_for(0x40_0fff, 0x40_1001), 2);
    }

    /// The pages compared with `sample` are about one in 16 of all, the
    /// same each time; their bytes count 16 times, and nothing compared
    /// changes what is copied.
    #[test]
    fn one_page_in_sixteen_is_compared_and_counted_sixteen_times() {
        let picked = (0..1 << 16).filter(|&p| sampled_page(p)).count();
        assert!((3600..=4600).contains(&picked), "{picked} of 65536");
        assert_eq!((0..1 << 16).filter(|&p| sampled_page(p)).count(), picked);
        const START: usize = 0x80_0000;
        const LEN: usize = 4 << 20;
        let title = memory(LEN, 9);
        let mut prints = vec![0u64; prints_for(START, START + LEN)];
        let read = |at: usize, into: &mut [u8]| {
            into.copy_from_slice(&title[at - START..at - START + into.len()]);
            true
        };
        let mut tally = [0u64; 3];
        sample((START, START + LEN), (START, START + LEN), &mut prints, &mut tally, read);
        let picked = (START / PRINTED..(START + LEN) / PRINTED).filter(|&p| sampled_page(p)).count() as u64;
        assert_eq!(tally, [0, 0, 16 * PRINTED as u64 * picked]);
        let mut tally = [0u64; 3];
        sample((START, START + LEN), (START, START + LEN), &mut prints, &mut tally, read);
        assert_eq!(tally, [16 * PRINTED as u64 * picked, 0, 0]);
        // A page not all there is not compared, and keeps no print.
        let p = (START / PRINTED..).find(|&p| sampled_page(p)).unwrap_or(0);
        let mut tally = [0u64; 3];
        let (from, to) = (p * PRINTED, (p + 1) * PRINTED);
        sample((START, START + LEN), (from, to), &mut prints, &mut tally, |_, _| false);
        assert_eq!(tally, [0, 0, 16 * PRINTED as u64]);
        assert_eq!(prints[p - START / PRINTED], 0);
    }
}
