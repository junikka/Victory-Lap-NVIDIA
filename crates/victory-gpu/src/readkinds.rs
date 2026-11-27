//! What kind of title data the card reads, window by window, and which
//! windows go to the card's memory for it.
//!
//! Three switches, all off unless set:
//! - `VICTORY_KIND_COUNTS=1` (as before) also gives, every 5 s, the line
//!   `card reads by kind`: per kind the bytes the binds named a frame, the
//!   windows read and the binds, and how much of each already lies on the
//!   card ([`reads_text`]).
//! - `VICTORY_STORE_COUNTS=1` (or `VICTORY_KIND_COUNTS=1`): the line of what
//!   the title's shaders are declared to store into title memory a frame, by
//!   what declares it and by where the bytes lie ([`stores_text`]).
//! - `VICTORY_CARD_WINDOWS_KINDS=<list>`: only windows read as one of the
//!   kinds listed get a side on the card (`VICTORY_CARD_WINDOWS_MB`), the
//!   most read first ([`choose`]). Not set: windows go there in the order
//!   they are made, as before.
//!
//! The counters are atomics of the copy bound and a few of this file: no
//! lock and no allocation where the title's threads record, and with every
//! switch off one read of a flag.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// A kind of read: what title memory is bound for.
pub(crate) const VERTEX: usize = 0;
pub(crate) const INDEX: usize = 1;
/// A root SRV or UAV: a raw view the shader indexes, with no size.
pub(crate) const ROOT_SRV: usize = 2;
/// A root constant buffer read from its copy.
pub(crate) const ROOT_CBV: usize = 3;
/// A buffer view in a descriptor table (SRV or UAV).
pub(crate) const TABLE_VIEW: usize = 4;
/// Constants of a table read in place.
pub(crate) const TABLE_CBV: usize = 5;
/// Indirect arguments and counts.
pub(crate) const ARGS: usize = 6;
pub(crate) const OTHER: usize = 7;
pub(crate) const KINDS: usize = 8;

/// The kinds as the line names them.
pub(crate) const NAMES: [&str; KINDS] = [
    "vertex",
    "index",
    "srv",
    "cbv",
    "table views",
    "table cbv",
    "args",
    "other",
];

/// What a bind with no size is taken to read (a root SRV or UAV reaches to
/// its copy's end, whatever the shader loads of it): as much as a root
/// constant buffer is taken as. A rank by it is a rank by binds.
pub(crate) const UNSIZED: u64 = 1 << 16;

/// Whether reads are counted by kind: with `VICTORY_KIND_COUNTS=1`, and with
/// `VICTORY_CARD_WINDOWS_KINDS`, which ranks by them.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::host::kinds::on() || allowed().is_some())
}

/// Whether shader stores are counted (`VICTORY_STORE_COUNTS=1`, or with
/// `VICTORY_KIND_COUNTS=1`).
pub(crate) fn stores_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        crate::host::kinds::on()
            || std::env::var_os("VICTORY_STORE_COUNTS").is_some_and(|v| v == "1")
    })
}

/// `VICTORY_CARD_WINDOWS_KINDS`: the kinds whose windows go to the card, as
/// bits by kind; `None` when not set (or naming no kind): every window may,
/// in the order they are made, as before.
pub(crate) fn allowed() -> Option<u8> {
    static ALLOWED: OnceLock<Option<u8>> = OnceLock::new();
    *ALLOWED.get_or_init(|| {
        let value = std::env::var("VICTORY_CARD_WINDOWS_KINDS").ok()?;
        let (mask, unknown) = parse_kinds(&value);
        if !unknown.is_empty() {
            tracing::warn!(
                "VICTORY_CARD_WINDOWS_KINDS: not a kind: {} (kinds: vertex, index, rootview, srv, cbv, rootsrv, rootcbv, \
                 table, tablecbv, args, other, all)",
                unknown.join(", ")
            );
        }
        mask
    })
}

/// [`allowed`] of its value: a list of kinds, by commas (or `+`, or blanks),
/// as bits by kind, and the words that name none. `rootview` is both root
/// kinds, `srv` every buffer view a shader indexes (root and table), `cbv`
/// both kinds of constants. `None` if no kind is named. Pure.
pub(crate) fn parse_kinds(value: &str) -> (Option<u8>, Vec<String>) {
    let bit = |kind: usize| 1u8 << kind;
    let mut mask = 0u8;
    let mut unknown = Vec::new();
    for word in value
        .split([',', '+', ' ', ';'])
        .map(str::trim)
        .filter(|w| !w.is_empty())
    {
        mask |= match word.to_ascii_lowercase().as_str() {
            "vertex" | "vertices" => bit(VERTEX),
            "index" | "indices" => bit(INDEX),
            "rootview" | "rootviews" | "root" => bit(ROOT_SRV) | bit(ROOT_CBV),
            "rootsrv" => bit(ROOT_SRV),
            "rootcbv" => bit(ROOT_CBV),
            "srv" => bit(ROOT_SRV) | bit(TABLE_VIEW),
            "cbv" | "constants" => bit(ROOT_CBV) | bit(TABLE_CBV),
            "table" | "tableview" | "tableviews" => bit(TABLE_VIEW),
            "tablecbv" => bit(TABLE_CBV),
            "args" | "indirect" => bit(ARGS),
            "other" => bit(OTHER),
            "all" => u8::MAX,
            _ => {
                unknown.push(word.to_owned());
                0
            }
        };
    }
    ((mask != 0).then_some(mask), unknown)
}

thread_local! {
    /// The kind the ask under way on this thread is for ([`asking`]);
    /// `KINDS`: not said.
    static ASKING: Cell<u8> = const { Cell::new(KINDS as u8) };
}

/// Says what the next asks for a copy on this thread are for, until the
/// value returned goes: a window made by one of them knows its first kind
/// ([`asked_for`]). Nothing is said without `VICTORY_CARD_WINDOWS_KINDS`.
pub(crate) fn asking(kind: usize) -> Asking {
    if allowed().is_none() {
        return Asking(None);
    }
    Asking(Some(ASKING.replace(kind as u8)))
}

/// See [`asking`].
pub(crate) struct Asking(Option<u8>);

impl Drop for Asking {
    fn drop(&mut self) {
        if let Some(before) = self.0 {
            ASKING.set(before);
        }
    }
}

/// The kind the ask under way on this thread is for, if it was said.
pub(crate) fn asked_for() -> Option<usize> {
    Some(usize::from(ASKING.get())).filter(|&kind| kind < KINDS)
}

/// A copy's reads since the last line, by kind: binds and the bytes they
/// named.
#[derive(Default)]
pub(crate) struct Reads {
    binds: [AtomicU64; KINDS],
    bytes: [AtomicU64; KINDS],
}

impl Reads {
    /// A bind of `bytes` for `kind`.
    pub(crate) fn add(&self, kind: usize, bytes: u64) {
        self.binds[kind].fetch_add(1, Relaxed);
        self.bytes[kind].fetch_add(bytes, Relaxed);
    }

    /// The counts, taken: `[binds, bytes]` by kind.
    pub(crate) fn take(&self) -> [[u64; 2]; KINDS] {
        std::array::from_fn(|k| {
            // (Not written when nothing was counted: the lines are shared.)
            if self.binds[k].load(Relaxed) == 0 {
                [0, 0]
            } else {
                [
                    self.binds[k].swap(0, Relaxed),
                    self.bytes[k].swap(0, Relaxed),
                ]
            }
        })
    }
}

/// Binds and bytes since the last line by kind, of every copy (windows,
/// title buffers' copies, memory in place), by whether a shader reads the
/// bytes from the card's memory (1) or over the bus (0).
static ALL: [[[AtomicU64; 2]; 2]; KINDS] =
    [const { [const { [const { AtomicU64::new(0) }; 2] }; 2] }; KINDS];

/// A bind of `bytes` for `kind` of a copy on the card or not.
pub(crate) fn read(kind: usize, on_card: bool, bytes: u64) {
    let at = &ALL[kind][usize::from(on_card)];
    at[0].fetch_add(1, Relaxed);
    at[1].fetch_add(bytes, Relaxed);
}

/// [`ALL`], taken.
pub(crate) fn take_all() -> [[[u64; 2]; 2]; KINDS] {
    std::array::from_fn(|k| {
        std::array::from_fn(|c| std::array::from_fn(|i| ALL[k][c][i].swap(0, Relaxed)))
    })
}

/// A window and what was read of it since the last line.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Window {
    /// Its start and length in title memory.
    pub(crate) key: (u64, u64),
    /// Whether shaders read it from the card's memory.
    pub(crate) on_card: bool,
    /// `[binds, bytes]` by kind.
    pub(crate) reads: [[u64; 2]; KINDS],
}

/// The classifier: the kinds a window was read as (bits by kind; a window
/// bound as two kinds has both), and the kind most of its bytes were named
/// for (the first of equals), if it was read at all. Pure.
pub(crate) fn classify(reads: &[[u64; 2]; KINDS]) -> (u8, Option<usize>) {
    let mut mask = 0u8;
    let mut main: Option<usize> = None;
    for (kind, read) in reads.iter().enumerate() {
        if read[0] == 0 {
            continue;
        }
        mask |= 1 << kind;
        if main.is_none_or(|m| read[1] > reads[m][1]) {
            main = Some(kind);
        }
    }
    (mask, main)
}

/// By kind: bytes named, windows and binds; and of them those of windows on
/// the card: bytes and windows. A window read as two kinds is among the
/// windows of both.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Tally {
    pub(crate) bytes: u64,
    pub(crate) windows: u64,
    pub(crate) binds: u64,
    pub(crate) card_bytes: u64,
    pub(crate) card_windows: u64,
}

/// The windows' reads summed by kind. Pure.
pub(crate) fn tally(windows: &[Window]) -> [Tally; KINDS] {
    let mut sums = [Tally::default(); KINDS];
    for window in windows {
        let (mask, _) = classify(&window.reads);
        for (kind, sum) in sums.iter_mut().enumerate() {
            if mask & 1 << kind == 0 {
                continue;
            }
            let [binds, bytes] = window.reads[kind];
            sum.bytes += bytes;
            sum.binds += binds;
            sum.windows += 1;
            if window.on_card {
                sum.card_bytes += bytes;
                sum.card_windows += 1;
            }
        }
    }
    sums
}

/// The text of the line of reads by kind: `windows` as read over `frames`
/// frames (`alive` of them there are), and `all`, the binds of every copy
/// ([`take_all`]). Kinds nothing was bound for are left out, but the five
/// the line always names. Pure.
pub(crate) fn reads_text(
    windows: &[Window],
    alive: usize,
    all: &[[[u64; 2]; 2]; KINDS],
    frames: f64,
) -> String {
    let mb = |bytes: u64| bytes as f64 / frames / 1048576.0;
    let sums = tally(windows);
    let named = |kind: usize| kind <= ROOT_CBV || kind == ARGS || sums[kind].binds > 0;
    let read: Vec<String> = (0..KINDS)
        .filter(|&k| named(k))
        .map(|k| {
            let s = &sums[k];
            format!(
                "{} {:.1} MB / {} windows / {:.0} binds",
                NAMES[k],
                mb(s.bytes),
                s.windows,
                s.binds as f64 / frames
            )
        })
        .collect();
    let there: Vec<String> = (0..KINDS)
        .filter(|&k| named(k))
        .map(|k| {
            format!(
                "{} {:.1} MB / {} windows",
                NAMES[k],
                mb(sums[k].card_bytes),
                sums[k].card_windows
            )
        })
        .collect();
    let every: Vec<String> = (0..KINDS)
        .filter(|&k| all[k].iter().any(|c| c[0] > 0))
        .map(|k| {
            let [bus, card] = all[k];
            format!(
                "{} {:.1} MB in {:.0} binds over the bus, {:.1} MB in {:.0} binds on the card",
                NAMES[k],
                mb(bus[1]),
                bus[0] as f64 / frames,
                mb(card[1]),
                card[0] as f64 / frames
            )
        })
        .collect();
    let read_windows = windows.iter().filter(|w| classify(&w.reads).0 != 0).count();
    let two = windows
        .iter()
        .filter(|w| classify(&w.reads).0.count_ones() > 1)
        .count();
    format!(
        "card reads by kind: {}; of that already on the card: {}; {read_windows} of {alive} windows read, {two} of them as \
         more than one kind; every copy (windows, title buffers' copies, title memory in place): {}",
        read.join(", "),
        there.join(", "),
        if every.is_empty() {
            "nothing bound".to_owned()
        } else {
            every.join("; ")
        }
    )
}

/// A window as [`choose`] ranks it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Ranked {
    pub(crate) key: (u64, u64),
    /// The bytes of the card it takes (or would).
    pub(crate) len: u64,
    pub(crate) on_card: bool,
    /// Bytes named a frame by kind (a running mean over the lines).
    pub(crate) bytes: [u64; KINDS],
}

/// What a window counts for under `allowed`: the bytes a frame named for
/// the kinds allowed. Pure.
pub(crate) fn score(bytes: &[u64; KINDS], allowed: u8) -> u64 {
    bytes
        .iter()
        .enumerate()
        .filter(|(kind, _)| allowed & 1 << kind != 0)
        .fold(0u64, |sum, (_, &b)| sum.saturating_add(b))
}

/// The windows that are to have a side on the card: those read as a kind
/// of `allowed`, the most bytes read a frame first, whatever their size, as
/// many as `budget` bytes hold (one too big for what is left is passed
/// over, and the next looked at). A window on the card already counts a
/// quarter more, so that two windows read alike do not change places with
/// every line. In rank order. Pure.
pub(crate) fn choose(windows: &[Ranked], allowed: u8, budget: u64) -> Vec<(u64, u64)> {
    let mut ranked: Vec<(u64, &Ranked)> = windows
        .iter()
        .map(|w| {
            let read = score(&w.bytes, allowed);
            (
                if w.on_card {
                    read.saturating_add(read / 4)
                } else {
                    read
                },
                w,
            )
        })
        .filter(|(read, _)| *read > 0)
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.key.cmp(&b.1.key)));
    let mut left = budget;
    let mut chosen = Vec::new();
    for (_, window) in ranked {
        if window.len <= left {
            left -= window.len;
            chosen.push(window.key);
        }
    }
    chosen
}

/// What declares a shader store.
pub(crate) const STORE_ROOT: usize = 0;
pub(crate) const STORE_TABLE: usize = 1;
pub(crate) const STORE_SMALL: usize = 2;
const STORE_BY: [&str; 3] = ["root UAVs", "table UAVs", "small buffers that come back"];
/// Where the bytes stored lie, as the copy's `lives`.
const STORE_IN: [&str; 5] = [
    "card copies of title buffers",
    "windows on the card",
    "windows in the PC's memory",
    "title memory in place (the PC's)",
    "small copies in the PC's memory",
];

/// Since the last line, by what declares it and by where the bytes lie:
/// declarations, of them those inside what the same copy was told already
/// in the frame, and the bytes the others added ([`span_grows`]; root views
/// have no size: to the copy's end, or `VICTORY_ROOT_UAV_MB`).
static STORED: [[[AtomicU64; 3]; 5]; 3] =
    [const { [const { [const { AtomicU64::new(0) }; 3] }; 5] }; 3];

/// The bytes a declaration of `[lo, hi)` adds to what a copy was told in
/// the frame: all of it if nothing was (`before` is `None`), else what
/// reaches out of the one span `before` (the gap to a range apart is not
/// counted, and is taken as told from then on: the world declares views of
/// one heap from dozens of offsets, and a list of ranges per copy would
/// need a lock). Pure.
pub(crate) fn span_grows(before: Option<(u64, u64)>, lo: u64, hi: u64) -> u64 {
    let len = hi.saturating_sub(lo);
    match before {
        None => len,
        Some((low, high)) => (low
            .saturating_sub(lo)
            .saturating_add(hi.saturating_sub(high)))
        .min(len),
    }
}

/// A store of `bytes` declared by `by` into a copy that lives in `place`;
/// `again`: inside what the copy was told already in the frame.
pub(crate) fn stored(by: usize, place: usize, bytes: u64, again: bool) {
    let at = &STORED[by][place];
    at[0].fetch_add(1, Relaxed);
    if again {
        at[1].fetch_add(1, Relaxed);
    } else {
        at[2].fetch_add(bytes, Relaxed);
    }
}

/// The text of the stores line: `counts[by][place]` are declarations, of
/// them repeats, and bytes, over `frames` frames. Pure.
pub(crate) fn stores_text(counts: &[[[u64; 3]; 5]; 3], frames: f64) -> String {
    let mb = |bytes: u64| bytes as f64 / frames / 1048576.0;
    // Places 2 to 4 are the PC's memory: stored over the bus.
    let pc: u64 = counts.iter().flat_map(|by| &by[2..]).map(|c| c[2]).sum();
    let card: u64 = counts.iter().flat_map(|by| &by[..2]).map(|c| c[2]).sum();
    let mut parts = Vec::new();
    for (by, per) in STORE_BY.iter().zip(counts) {
        let places: Vec<String> = STORE_IN
            .iter()
            .zip(per)
            .filter(|(_, c)| c[0] > 0)
            .map(|(place, c)| {
                format!(
                    "{:.1} MB in {:.0} ranges in {place} ({:.0} more inside those)",
                    mb(c[2]),
                    (c[0] - c[1]) as f64 / frames,
                    c[1] as f64 / frames
                )
            })
            .collect();
        if !places.is_empty() {
            parts.push(format!("{by}: {}", places.join(", ")));
        }
    }
    format!(
        "{:.1} MB into the PC's memory, {:.1} MB into the card's; {}",
        mb(pc),
        mb(card),
        if parts.is_empty() {
            "none declared".to_owned()
        } else {
            parts.join("; ")
        }
    )
}

/// For the frame-rate line: what shaders were declared to store since the
/// last one.
pub(crate) fn stores_line(frames: f64) {
    if !stores_on() {
        return;
    }
    let counts: [[[u64; 3]; 5]; 3] = std::array::from_fn(|b| {
        std::array::from_fn(|p| std::array::from_fn(|i| STORED[b][p][i].swap(0, Relaxed)))
    });
    tracing::info!(
        "shader stores declared into title memory per frame (VICTORY_STORE_COUNTS=1; the most the shaders may store, not \
         what they did: a declared range is a view's whole reach, a root view's to its copy's end, and bytes told twice in a frame count once; what was found \
         changed in copies is in the line of shader writes looked for): {}",
        stores_text(&counts, frames)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(at: u64, on_card: bool, reads: &[(usize, u64, u64)]) -> Window {
        let mut all = [[0u64; 2]; KINDS];
        for &(kind, binds, bytes) in reads {
            all[kind] = [binds, bytes];
        }
        Window {
            key: (at << 22, 4 << 20),
            on_card,
            reads: all,
        }
    }

    #[test]
    fn a_window_is_of_each_kind_it_was_bound_as() {
        // One window per kind.
        for kind in 0..KINDS {
            let w = window(kind as u64, false, &[(kind, 3, 4096)]);
            assert_eq!(
                classify(&w.reads),
                (1 << kind, Some(kind)),
                "{}",
                NAMES[kind]
            );
        }
        // Never bound: of no kind.
        assert_eq!(classify(&[[0; 2]; KINDS]), (0, None));
        // Bound as two kinds: of both, and mainly of the one more bytes were
        // named for.
        let both = window(9, false, &[(VERTEX, 10, 1 << 20), (ROOT_SRV, 2, 2 << 20)]);
        assert_eq!(
            classify(&both.reads),
            (1 << VERTEX | 1 << ROOT_SRV, Some(ROOT_SRV))
        );
        // Equal bytes: the first kind.
        let even = window(9, false, &[(INDEX, 1, 64), (ARGS, 1, 64)]);
        assert_eq!(classify(&even.reads).1, Some(INDEX));
        // A bind that named no bytes still makes the window of its kind.
        let unsized_only = window(9, false, &[(ROOT_SRV, 1, 0)]);
        assert_eq!(
            classify(&unsized_only.reads),
            (1 << ROOT_SRV, Some(ROOT_SRV))
        );
    }

    #[test]
    fn reads_are_summed_by_kind_and_by_where_the_window_lies() {
        let mut windows: Vec<Window> = (0..KINDS)
            .map(|k| window(k as u64, k == INDEX, &[(k, 2, 1 << 20)]))
            .collect();
        // One window bound as two kinds, on the card.
        windows.push(window(
            20,
            true,
            &[(VERTEX, 4, 3 << 20), (INDEX, 6, 1 << 20)],
        ));
        let sums = tally(&windows);
        assert_eq!(
            sums[VERTEX],
            Tally {
                bytes: 4 << 20,
                windows: 2,
                binds: 6,
                card_bytes: 3 << 20,
                card_windows: 1
            }
        );
        assert_eq!(
            sums[INDEX],
            Tally {
                bytes: 2 << 20,
                windows: 2,
                binds: 8,
                card_bytes: 2 << 20,
                card_windows: 2
            }
        );
        for kind in ROOT_SRV..KINDS {
            assert_eq!(
                sums[kind],
                Tally {
                    bytes: 1 << 20,
                    windows: 1,
                    binds: 2,
                    card_bytes: 0,
                    card_windows: 0
                }
            );
        }
    }

    #[test]
    fn the_line_of_reads_names_each_kind_and_what_is_on_the_card() {
        let windows = [
            window(0, false, &[(VERTEX, 20, 8 << 20)]),
            window(1, true, &[(VERTEX, 4, 2 << 20), (INDEX, 4, 1 << 20)]),
            window(2, false, &[(ROOT_SRV, 200, 200 * UNSIZED)]),
            window(3, false, &[]),
        ];
        let mut all = [[[0u64; 2]; 2]; KINDS];
        all[VERTEX] = [[20, 8 << 20], [4, 2 << 20]];
        all[ROOT_CBV][0] = [100, 100 << 16];
        // Over two frames.
        assert_eq!(
            reads_text(&windows, 4, &all, 2.0),
            "card reads by kind: vertex 5.0 MB / 2 windows / 12 binds, index 0.5 MB / 1 windows / 2 binds, srv 6.2 MB / 1 \
             windows / 100 binds, cbv 0.0 MB / 0 windows / 0 binds, args 0.0 MB / 0 windows / 0 binds; of that already on \
             the card: vertex 1.0 MB / 1 windows, index 0.5 MB / 1 windows, srv 0.0 MB / 0 windows, cbv 0.0 MB / 0 windows, \
             args 0.0 MB / 0 windows; 3 of 4 windows read, 1 of them as more than one kind; every copy (windows, title \
             buffers' copies, title memory in place): vertex 4.0 MB in 10 binds over the bus, 1.0 MB in 2 binds on the \
             card; cbv 3.1 MB in 50 binds over the bus, 0.0 MB in 0 binds on the card"
        );
    }

    #[test]
    fn the_list_of_kinds_is_read() {
        let bit = |kind: usize| 1u8 << kind;
        assert_eq!(
            parse_kinds("rootview,index"),
            (Some(bit(ROOT_SRV) | bit(ROOT_CBV) | bit(INDEX)), vec![])
        );
        assert_eq!(
            parse_kinds("rootview"),
            (Some(bit(ROOT_SRV) | bit(ROOT_CBV)), vec![])
        );
        assert_eq!(parse_kinds(" Vertex "), (Some(bit(VERTEX)), vec![]));
        assert_eq!(parse_kinds("index"), (Some(bit(INDEX)), vec![]));
        assert_eq!(
            parse_kinds("srv+cbv"),
            (
                Some(bit(ROOT_SRV) | bit(TABLE_VIEW) | bit(ROOT_CBV) | bit(TABLE_CBV)),
                vec![]
            )
        );
        assert_eq!(
            parse_kinds("args,other,table,tablecbv"),
            (
                Some(bit(ARGS) | bit(OTHER) | bit(TABLE_VIEW) | bit(TABLE_CBV)),
                vec![]
            )
        );
        assert_eq!(parse_kinds("all"), (Some(u8::MAX), vec![]));
        // A word that names no kind is said and passed over.
        assert_eq!(
            parse_kinds("index,textures"),
            (Some(bit(INDEX)), vec!["textures".to_owned()])
        );
        // No kind named: as not set.
        assert_eq!(parse_kinds(""), (None, vec![]));
        assert_eq!(parse_kinds(" , "), (None, vec![]));
        assert_eq!(parse_kinds("none"), (None, vec!["none".to_owned()]));
    }

    fn ranked(at: u64, len: u64, on_card: bool, reads: &[(usize, u64)]) -> Ranked {
        let mut bytes = [0u64; KINDS];
        for &(kind, b) in reads {
            bytes[kind] = b;
        }
        Ranked {
            key: (at << 22, len),
            len,
            on_card,
            bytes,
        }
    }

    #[test]
    fn windows_are_chosen_by_bytes_read_not_by_size() {
        const MB: u64 = 1 << 20;
        let root = 1 << ROOT_SRV | 1 << ROOT_CBV;
        let windows = [
            // Big and hardly read.
            ranked(0, 64 * MB, false, &[(ROOT_SRV, MB)]),
            // Small and read most.
            ranked(1, 4 * MB, false, &[(ROOT_SRV, 30 * MB)]),
            ranked(2, 4 * MB, false, &[(ROOT_CBV, 10 * MB)]),
            // Read much, as a kind not asked for.
            ranked(3, 4 * MB, false, &[(VERTEX, 500 * MB)]),
            // Bound as two kinds: only the kind asked for counts.
            ranked(4, 4 * MB, false, &[(VERTEX, 500 * MB), (ROOT_SRV, 20 * MB)]),
            // Never read.
            ranked(5, 4 * MB, false, &[]),
        ];
        // Room for all: every window of the kinds, most read first.
        assert_eq!(
            choose(&windows, root, 1024 * MB),
            vec![
                (1 << 22, 4 * MB),
                (4 << 22, 4 * MB),
                (2 << 22, 4 * MB),
                (0, 64 * MB)
            ]
        );
        // Room for two: the two most read, not the biggest.
        assert_eq!(
            choose(&windows, root, 8 * MB),
            vec![(1 << 22, 4 * MB), (4 << 22, 4 * MB)]
        );
        // One too big for what is left is passed over, the next taken.
        let tight = [
            ranked(0, 64 * MB, false, &[(INDEX, 9 * MB)]),
            ranked(1, 4 * MB, false, &[(INDEX, MB)]),
        ];
        assert_eq!(choose(&tight, 1 << INDEX, 8 * MB), vec![(1 << 22, 4 * MB)]);
        // Another kind, another choice; every kind: by all bytes.
        assert_eq!(
            choose(&windows, 1 << VERTEX, 4 * MB),
            vec![(3 << 22, 4 * MB)]
        );
        assert_eq!(
            choose(&windows, u8::MAX, 8 * MB),
            vec![(4 << 22, 4 * MB), (3 << 22, 4 * MB)]
        );
        // No room: none.
        assert!(choose(&windows, root, 0).is_empty());
    }

    #[test]
    fn a_window_on_the_card_keeps_its_place_against_one_read_a_little_more() {
        const MB: u64 = 1 << 20;
        let windows = [
            ranked(0, 4 * MB, true, &[(INDEX, 10 * MB)]),
            ranked(1, 4 * MB, false, &[(INDEX, 11 * MB)]),
        ];
        assert_eq!(choose(&windows, 1 << INDEX, 4 * MB), vec![(0, 4 * MB)]);
        // Read much more, it takes the place.
        let windows = [
            ranked(0, 4 * MB, true, &[(INDEX, 10 * MB)]),
            ranked(1, 4 * MB, false, &[(INDEX, 13 * MB)]),
        ];
        assert_eq!(
            choose(&windows, 1 << INDEX, 4 * MB),
            vec![(1 << 22, 4 * MB)]
        );
    }

    #[test]
    fn the_stores_line_says_what_goes_over_the_bus() {
        let mut counts = [[[0u64; 3]; 5]; 3];
        // Over two frames: root UAVs declared 40 times into title memory in
        // place, 36 of them again; table UAVs into a card copy and a window
        // in the PC's memory.
        counts[STORE_ROOT][3] = [40, 36, 348 << 20];
        counts[STORE_TABLE][0] = [6, 0, 12 << 20];
        counts[STORE_TABLE][2] = [2, 0, 1 << 20];
        assert_eq!(
            stores_text(&counts, 2.0),
            "174.5 MB into the PC's memory, 6.0 MB into the card's; root UAVs: 174.0 MB in 2 ranges in title memory in \
             place (the PC's) (18 more inside those); table UAVs: 6.0 MB in 3 ranges in card copies of title buffers (0 \
             more inside those), 0.5 MB in 1 ranges in windows in the PC's memory (0 more inside those)"
        );
        assert_eq!(
            stores_text(&[[[0; 3]; 5]; 3], 1.0),
            "0.0 MB into the PC's memory, 0.0 MB into the card's; none declared"
        );
    }

    #[test]
    fn bytes_declared_twice_in_a_frame_count_once() {
        // The first declaration of the frame: all of it.
        assert_eq!(span_grows(None, 100, 300), 200);
        // The same again, and one inside: nothing.
        assert_eq!(span_grows(Some((100, 300)), 100, 300), 0);
        assert_eq!(span_grows(Some((100, 300)), 150, 200), 0);
        // Reaching out at either end, or both: what is new.
        assert_eq!(span_grows(Some((100, 300)), 250, 400), 100);
        assert_eq!(span_grows(Some((100, 300)), 50, 150), 50);
        assert_eq!(span_grows(Some((100, 300)), 0, 1000), 800);
        // Apart: its own bytes, not the gap.
        assert_eq!(span_grows(Some((100, 300)), 1000, 1100), 100);
        assert_eq!(span_grows(Some((1000, 1100)), 100, 300), 200);
        // Empty.
        assert_eq!(span_grows(None, 5, 5), 0);
    }

    #[test]
    fn a_thread_says_what_it_asks_for_only_with_the_switch() {
        // (The switch is not set in tests: nothing is said.)
        if allowed().is_none() {
            let _ask = asking(VERTEX);
            assert_eq!(asked_for(), None);
        }
    }
}
