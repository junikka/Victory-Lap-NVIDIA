//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    /// Title memory's pages of 4 KiB (from the one `start` lies in on), a bit
    /// each, that what shaders wrote was put into since a refresh last looked
    /// at them ([`Shadow::take_landed`]); empty until then.
    landings: Mutex<Vec<u64>>,
    /// Whether any is.
    landing: std::sync::atomic::AtomicBool,
    /// The bytes of `buffer`, once asked.
    width: std::sync::OnceLock<u64>,
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Two ranges of title memory taken whole, the last one and the longest,
    /// and the asking after which each was ([`Shadow::copy_stale`]). A page
    /// a range holds in part cannot say so in `copied`. Locked inside
    /// `copied`'s lock.
    ranged: Mutex<[(usize, usize, u64); 2]>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: how many submissions not yet copied back declared
    /// each range of `pending` ([`Flights`]). Locked inside `pending`'s
    /// lock.
    flights: Mutex<Flights>,
    /// The copy's bytes before the submission ran, for the ranges its
    /// shaders write (`pending`), each at its offset in the copy: what the
    /// GPU wrote differs. Room for the whole copy; only those ranges' bytes
    /// are set (and read).
    before: Mutex<Option<Vec<u8>>>,
    /// Held shared while pieces of the copy are landed (`crate::land`), each
    /// its own bytes of it on several threads, and alone while title bytes
    /// go in under shader writes ([`Shadow::merge_under_writes`]), which
    /// takes `before`'s lock too. Taken ahead of that one.
    merging: RwLock<()>,
    /// With snapshots kept ([`keeps`]): which 64 KiB pieces of the snapshot
    /// hold the copy's bytes as they are now, and whether the snapshot is
    /// counted in [`KEPT_BYTES`].
    kept: Mutex<(Vec<bool>, bool)>,
    /// When the copy was last asked for (`crate::host::moment`): a window
    /// outside buffers nothing asked for in a while, and nothing holds, is
    /// let go ([`let_go_idle`]).
    used: AtomicU64,
    /// The lowest and highest title address asked for in the copy (a
    /// window is 4 MiB whatever was asked: measured for the memory line).
    asked: [AtomicU64; 2],
    /// Asks for the copy that gave no length (root views: they read from
    /// their address to the copy's end, whatever lies there).
    unsized_asks: AtomicU64,
    /// Whether shaders were ever declared to write it.
    written: std::sync::atomic::AtomicBool,
    /// A number no other copy has had. Its address is no name for a copy:
    /// one let go and another made may share it.
    id: u64,
    /// With [`cards`], for a big title buffer: `buffer` is in the card's
    /// own memory, and this is the rest of the copy.
    card: Option<Card>,
    /// A window of title memory outside buffers, not a title buffer's copy.
    window: bool,
    /// Of a copy with two sides: only the dwords the title changed go up,
    /// as with [`send_changed`] ([`by_address`]).
    changes_only: std::sync::atomic::AtomicBool,
    /// A window in card memory the CPU maps ([`window_mapped_for`]).
    mapped: bool,
    /// A window with no card side, with `VICTORY_WINDOW_SKIP_SAME=sample`:
    /// per 4 KiB page of title memory from the window's first on, a print
    /// of the bytes last copied into its buffer (0: none) (`crate::windowskip`);
    /// empty until the window first samples.
    prints: Mutex<Vec<u64>>,
    /// With `crate::inplace`: not a copy at all. `buffer` is placed in a
    /// heap opened on the title memory itself, so shaders read and write
    /// the title's own bytes and nothing is copied either way. The heap
    /// (it begins on a 64 KiB line; a title buffer need not: `lead`).
    place: Option<windows::Win32::Graphics::Direct3D12::ID3D12Heap>,
    /// Bytes from the buffer's first byte to the copy's (`data` and `gpu`
    /// are the copy's own): in place, from the heap's line to `start`; a
    /// copy in a block of them ([`exact_in_blocks`]), its place in the
    /// block; otherwise 0.
    lead: u64,
    /// With `VICTORY_MEMORY_BY_USE=1` (`crate::byuse`): what is done with it.
    uses: Option<Arc<crate::byuse::Uses>>,
    /// With `crate::readkinds::on`: binds since the last line, by kind.
    reads: crate::readkinds::Reads,
    /// With `crate::readkinds::stores_on`: the frame shaders were last
    /// declared to store into the copy (and one, 0: never), and the span
    /// declared in that frame, to count bytes declared twice once.
    stored_in: [AtomicU64; 3],
}

/// What a copy in the card's memory ([`cards`]) has beside the buffer
/// shaders see.
struct Card {
    /// The CPU's side, in the PC's memory: what `data` maps. Title memory
    /// is copied into it as into any copy, and what changed in it goes to
    /// the card's side before the next submission.
    staging: ID3D12Resource,
    /// The card side's bytes as they were when the submission's shaders
    /// started, in the card's memory too: made when shader writes are
    /// first declared for the copy, and compared with the card's side, on
    /// the card, after each submission that declares some.
    before: Mutex<Option<ID3D12Resource>>,
    /// Offsets of the copy whose bytes changed on the CPU's side and have
    /// not gone to the card's side yet.
    dirty: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: the offsets that went from `dirty` to the card's
    /// side, each with the number of the send that took it
    /// ([`card_dirty`]), kept until a fetch sent after it has been merged.
    /// Locked inside `dirty`'s lock.
    sent: Mutex<SentRuns>,
    /// Whether something was put into `dirty` since it was last taken.
    touched: std::sync::atomic::AtomicBool,
    /// With [`dirty_list`], once the copy is among [`CARDED`]: the copy this
    /// is of, which is what goes into [`TOUCHED`]. Set and read under
    /// `dirty`'s lock. Never set without the switch.
    me: std::sync::OnceLock<std::sync::Weak<Shadow>>,
    /// The buffers' bytes (the title buffer's, rounded up to 64 KiB).
    size: u64,
}

/// Whether the host copies of big title buffers are in the card's own
/// memory (on by default since run 75 showed it holds: the BVH skinning
/// list went from 50.5 to 4.5 ms on the card; `VICTORY_CARD_BUFFERS=0`
/// keeps every copy in the PC's memory, as before. Read when a copy is
/// made, so it takes a fresh start).
///
/// Every host copy was a buffer in the PC's memory that the card reads
/// and writes over the bus, so that our code can fill it and compare it.
/// The title's shaders read almost all their buffer data from a few big
/// title buffers (in the gameplay frame of run 69, 190-330 MB of distinct
/// ranges of the 174 MB heap per list against 0-13 MB elsewhere), and a
/// skinning dispatch of ONE group of 32 threads took 1.8 ms on the card
/// (run 70): the time of some 170 loads one after the other, not of the
/// arithmetic.
///
/// With the switch, a title buffer of [`card_limits`]' size or more has two
/// sides. Shaders see `Shadow::buffer`, in the card's memory. The CPU's
/// side (`Card::staging`, what `Shadow::data` maps) takes title memory as
/// before, and the ranges that changed in it are copied to the card's side
/// before each submission (`Host::card_before`). What shaders wrote is
/// found on the card: after the submission's lists a compute pass of ours
/// compares each declared range, by 64 KiB piece, with the card side's
/// bytes from before (`Card::before`), and only the pieces that differ are
/// fetched and merged into title memory, byte by byte as
/// [`Shadow::copy_back`] does (`Host::card_back`, [`Shadow::card_merge`]).
/// No snapshot is taken and nothing is compared on the CPU for such a copy.
///
/// What it assumes is what [`keeps`] assumes: shaders write a copy only in
/// ranges declared as written. A write outside them is seen by later
/// shaders, as before, and goes back to the title the next time a declared
/// range holds its piece (before: never).
pub(crate) fn cards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BUFFERS").is_none_or(|v| v != "0"))
}

/// Default `(min_mb, max_mb)` for [`card_limits`] chosen from the adapter's
/// local VRAM budget (`DXGI_QUERY_VIDEO_MEMORY_INFO.Budget`), in bytes.
///
/// - Unknown (`None`, e.g. WARP) or `<= 12 GiB` (8–12 GB cards, whose WDDM
///   budget is below 12 GiB): `(16, 512)` MiB, matching the 12 GB baseline.
/// - `> 12 GiB` and `< 20 GiB` (16 GB cards, whose WDDM budget is typically
///   ~13.5–15.2 GiB): `(4, 2048)` MiB.
/// - `>= 20 GiB` (24–32+ GB cards): `(1, (budget_mb / 4).min(8192))` MiB,
///   capping at 25% of local budget (at most 8 GiB) so "before" twins (up to 2x)
///   still leave headroom for textures and render targets.
fn default_card_limits_mb(local_vram_budget: Option<u64>) -> (u64, u64) {
    let Some(budget_bytes) = local_vram_budget else {
        return (16, 512);
    };
    let budget_mb = budget_bytes >> 20;
    const TWELVE_GIB_MB: u64 = 12 * 1024;
    const TWENTY_GIB_MB: u64 = 20 * 1024;
    if budget_mb <= TWELVE_GIB_MB {
        (16, 512)
    } else if budget_mb < TWENTY_GIB_MB {
        (4, 2048)
    } else {
        (1, (budget_mb / 4).min(8192))
    }
}

/// Resolve `(min_bytes, max_bytes)` from the adapter's local VRAM budget and
/// optional explicit MiB overrides from `VICTORY_CARD_BUFFERS_MIN_MB` /
/// `VICTORY_CARD_BUFFERS_MB`.
fn resolve_card_limits(
    local_vram_budget: Option<u64>,
    min_mb_override: Option<u64>,
    max_mb_override: Option<u64>,
) -> (u64, u64) {
    const MIB: u64 = 1 << 20;
    let (default_min_mb, default_max_mb) = default_card_limits_mb(local_vram_budget);
    let min_bytes = min_mb_override.unwrap_or(default_min_mb).saturating_mul(MIB).max(MIB);
    let max_bytes = max_mb_override.unwrap_or(default_max_mb).saturating_mul(MIB);
    (min_bytes, max_bytes)
}

/// The smallest title buffer whose copy goes to the card's memory, and the
/// most bytes of such copies (their "before" twins not counted: a copy
/// shaders write takes its size twice), both in bytes:
/// `VICTORY_CARD_BUFFERS_MIN_MB` and `VICTORY_CARD_BUFFERS_MB`.
/// Defaults scale with the adapter's local VRAM budget (16 MiB / 512 MiB on
/// `<= 12 GiB` budgets, 4 MiB / 2048 MiB on 16 GB cards, 1 MiB / up to 8192 MiB
/// on `>= 20 GiB` budgets), and explicit env vars always override.
fn card_limits() -> (u64, u64) {
    static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let env_mb = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok());
        let vram_budget = crate::host::host().and_then(|h| h.video_memory()).map(|[budget, ..]| budget);
        let (mut min_bytes, max_bytes) = resolve_card_limits(
            vram_budget,
            env_mb("VICTORY_CARD_BUFFERS_MIN_MB"),
            explicit_card_buffers_mb(),
        );
        // `VICTORY_CARD_BUFFERS_MIN_KB` (after fxceless c36971b2) says the
        // least size in KiB, so that it can go under 1 MiB: 64 is every title
        // buffer. Copies under 1 MiB are nearly all there are and few bytes;
        // in the PC's memory every load, store and atomic a shader does on
        // them crosses the bus.
        if let Some(kib) = std::env::var("VICTORY_CARD_BUFFERS_MIN_KB").ok().and_then(|v| v.trim().parse::<u64>().ok())
            && kib > 0
        {
            min_bytes = kib << 10;
        }
        if card_all() {
            min_bytes = 0;
        }
        tracing::info!(
            vram_budget_mb = vram_budget.map(|b| b >> 20),
            card_buffers_min_kb = min_bytes >> 10,
            card_buffers_max_mb = max_bytes >> 20,
            "card buffer limits"
        );
        (min_bytes, max_bytes)
    })
}

/// `VICTORY_CARD_BUFFERS_MB` as read once, for both the limit and the
/// reservation ([`card_room_kept`]).
fn explicit_card_buffers_mb() -> Option<u64> {
    static MB: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *MB.get_or_init(|| std::env::var("VICTORY_CARD_BUFFERS_MB").ok().as_deref().and_then(parse_mib))
}

/// A number of MiB from a switch's value. Pure, for testing.
fn parse_mib(value: &str) -> Option<u64> {
    value.trim().parse().ok()
}

/// Bytes alive of copies' card sides, and of their "before" twins.
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether the texture queue leaves room on the card for the card copies
/// of title buffers that are not made yet
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=1`; off by default).
///
/// The copies are made in one burst when the world loads (run 189: 94 of
/// its 95 within two seconds, right before the first world frame), and
/// [`card_for`] makes one only while the card is under eight tenths of its
/// budget. That held as long as the textures picked by index filled the
/// card after the burst. Since the texture queue keeps packed neighbours
/// (52b8060) it fills the card while the world loads: 85% and more at the
/// burst in runs 190 to 192, so one copy of 95 was made, the CPU looked
/// for shader writes in 600 to 720 MB a frame where it had been 30, and
/// the frame went from about 240 to 370-470 ms.
///
/// With the switch, the texture queue's rule (`resource::room`) counts the
/// part of [`card_limits`]' second number that no copy has taken yet as
/// used, and a copy may be made up to nine tenths of the budget: the room
/// it takes was kept for it. After the burst what is kept is what the
/// burst left of the limit (87 MB of 1,024 in run 189).
///
/// On by default from the build after run 199
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=0`: off, as before). Run 193 had it on and still made 31 copies of 95: the copies
/// and the world's render targets (+1,603 MB) arrive in the same first
/// world frame and raced for the 448 MB left under the line, and a copy
/// turned down is not asked for again (`lookup` asks once), while its room
/// went on being kept (616 MB to the end of the run). So now: the texture
/// queue also keeps room for the targets (`resource::targets_room_kept`),
/// a copy is made up to 97 hundredths of the budget (its room was kept),
/// and the room of a copy turned down is given up ([`CARD_REFUSED`]).
fn card_room_kept_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ROOM_FOR_BUFFERS").is_none_or(|v| v != "0"))
}

/// `VICTORY_CARD_RESERVE_BUDGETS=1` (off unless set to 1): the texture queue
/// keeps room for an explicit `VICTORY_CARD_BUFFERS_MB` in full, not at most
/// 1 GiB, and for what the windows have not taken yet of
/// `VICTORY_CARD_WINDOWS_MB` and of `VICTORY_WINDOWS_BAR_MB`; windows then go
/// to the card up to 97 hundredths of its budget, as buffer copies do, not
/// 80 ([`window_usage_limit`]). Without it the textures fill the card to
/// their own line first, a window is turned down at 80 hundredths, and each
/// megabyte of window that does land pushes a megabyte of texture out while
/// a fifth of the card stays free (runs spin23-allon-cardwin2g and -winbar:
/// 6.95 of 9.0 GB in use, 64-68 of 100 textures let go made again).
///
/// Ported from TaynanGT/Vice-Tay PR #69 (266ba60a, author zzzz-26; made
/// opt-in by f3dfb5b9, author TaynanGT), as kitty-core's 294c53a0 and
/// Voltexs's e5b6ae24 ported it. Ours besides: the windows the CPU maps
/// (`VICTORY_WINDOWS_BAR_MB`) are kept room for and take the same line.
fn reserve_card_budgets() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_RESERVE_BUDGETS").is_some_and(|v| v == "1"))
}

/// Whether the budgets are kept ([`reserve_card_budgets`]): only with the
/// room rule that keeps them ([`card_room_kept`]).
fn budgets_reserved() -> bool {
    reserve_card_budgets() && card_room_kept_on() && cards()
}

/// Title buffers big enough for a copy in the card's memory that were
/// turned down for want of room, since the start, and their bytes: their
/// copies stay in the PC's memory for as long as the title keeps the
/// buffers, so no room is kept for them any more.
static CARD_REFUSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The bytes of the card's memory the texture queue leaves free for card
/// copies of title buffers not made yet ([`card_room_kept_on`]): 0 with
/// the switch off or without card copies.
pub(crate) fn card_room_kept() -> u64 {
    if !card_room_kept_on() || !cards() {
        return 0;
    }
    // Not more than such copies came to in any session, whatever they may
    // come to: on a card of 24 GB a quarter of its budget was kept for them,
    // 5 GB that never came (941 MB in 97 copies after fifteen minutes), and
    // no texture was made any more with 5 GB of the card free (session
    // play-1125-0127: 1,453 passes refused, the ground and people black).
    card_room_parts().iter().fold(0u64, |sum, &[_, left]| sum.saturating_add(left))
}

/// What [`card_room_kept`] keeps, by whom it is kept for: buffer copies,
/// windows with a card side (`VICTORY_CARD_WINDOWS_MB`) and windows the CPU
/// maps (`VICTORY_WINDOWS_BAR_MB`); of each the most kept and what is left of
/// it now, in bytes.
fn card_room_parts() -> [[u64; 2]; 3] {
    let reserve = reserve_card_budgets();
    let (buffers, windows) =
        reservation_limits(card_limits().1, explicit_card_buffers_mb().is_some(), window_card_most(), reserve);
    let (_, mapped) = reservation_limits(0, false, window_mapped_most(), reserve);
    [
        [
            buffers,
            remaining_card_room(
                buffers,
                CARD_BYTES[0].load(Ordering::Relaxed),
                CARD_REFUSED[1].load(Ordering::Relaxed),
                0,
                0,
            ),
        ],
        [windows, remaining_card_room(0, 0, 0, windows, WINDOW_CARD_BYTES.load(Ordering::Relaxed))],
        [mapped, remaining_card_room(0, 0, 0, mapped, WINDOW_MAPPED_BYTES.load(Ordering::Relaxed))],
    ]
}

/// The room kept for buffer copies and for windows ([`card_room_kept`]):
/// buffers at most 1 GiB and no windows, unless `reserve`
/// ([`reserve_card_budgets`]); then the windows' budget, and the buffers'
/// in full if it was asked for. Pure.
fn reservation_limits(buffers: u64, explicit_buffers: bool, windows: u64, reserve: bool) -> (u64, u64) {
    const SEEN: u64 = 1 << 30;
    if reserve {
        (if explicit_buffers { buffers } else { buffers.min(SEEN) }, windows)
    } else {
        (buffers.min(SEEN), 0)
    }
}

/// What is left of the two reservations. Pure.
fn remaining_card_room(buffers: u64, used: u64, refused: u64, windows: u64, window_used: u64) -> u64 {
    buffers.saturating_sub(used.saturating_add(refused)).saturating_add(windows.saturating_sub(window_used))
}

/// The reservation line, every 5 s with the budget line: what is kept and
/// for whom, the line windows go to the card under, the card's use, and the
/// textures the budget pass let go (`let_go`: since the start).
pub(crate) fn reserve_line(memory: Option<[u64; 4]>, let_go: u64) -> String {
    static BEFORE: AtomicU64 = AtomicU64::new(0);
    let since = let_go.saturating_sub(BEFORE.swap(let_go, Ordering::Relaxed));
    let on = cards() && card_room_kept_on();
    let parts = if on { card_room_parts() } else { [[0; 2]; 3] };
    reserve_line_of(
        reserve_card_budgets(),
        parts,
        memory.map(|[budget, used, ..]| [budget, used]),
        window_cap(),
        budgets_reserved(),
        [let_go, since],
    )
}

/// [`reserve_line`] of its numbers. Pure.
fn reserve_line_of(
    switch: bool,
    parts: [[u64; 2]; 3],
    memory: Option<[u64; 2]>,
    cap: Option<u64>,
    reserved: bool,
    let_go: [u64; 2],
) -> String {
    let mb = |n: u64| n >> 20;
    let kept = parts.iter().fold(0u64, |sum, &[_, left]| sum.saturating_add(left));
    let card = match memory {
        Some([budget, used]) => format!(
            "budget {} MB, card in use {} MB, windows go to the card under {} MB ({} hundredths)",
            mb(budget),
            mb(used),
            mb(window_usage_limit(budget, reserved, cap)),
            window_cap_percent(reserved, cap)
        ),
        None => "the card's memory is not known".to_owned(),
    };
    format!(
        "card reserve: {}; reserved {} MB (buffer copies {} of {} MB, card windows {} of {} MB, mapped windows {} of {} MB); \
         {card}; textures let go by the budget pass {} ({} since the last line)",
        if switch { "on" } else { "off" },
        mb(kept),
        mb(parts[0][1]),
        mb(parts[0][0]),
        mb(parts[1][1]),
        mb(parts[1][0]),
        mb(parts[2][1]),
        mb(parts[2][0]),
        let_go[0],
        let_go[1],
    )
}

/// The copies in the card's memory.
static CARDED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// Whether the copies in the card's memory whose CPU side changed are kept
/// in a list ([`TOUCHED`]) that [`card_dirty`] takes, so that it does not
/// look at every copy (`VICTORY_DIRTY_LIST=1`; off by default: every copy is
/// looked at, as before). Read when a copy is made, so it takes a fresh
/// start. Port of miamiwalk ebcf56f1 (its shadow.rs part).
///
/// `card_dirty` runs before every submission, about 40 times a frame, and
/// took a strong reference of every copy in the card's memory and gave it
/// back to find the few that had changed: 3,526 copies in run spin19-base
/// (with copies from 64 KiB up), so about 140,000 looks a frame on the
/// submitting thread for some 270 changed ranges. The line "what changed in
/// copies in the card's memory was asked" says what it costs either way.
///
/// What makes it the same ranges: a copy goes into the list when its
/// `touched` goes from unset to set, under its `dirty` lock, and `touched`
/// is cleared only by the taking, which then takes the ranges under that
/// lock. So a copy whose `touched` is set is in the list, or in a list
/// already taken whose turn has not come; nothing changed is left out
/// (`dirty_list_tests`). A copy can be in the list twice (it changed again
/// while a taken list held it): the second look finds `touched` unset and
/// nothing to take, as a look at an unchanged copy did before.
fn dirty_list() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_DIRTY_LIST");
        if on {
            tracing::info!("the copies in the card's memory that changed are kept in a list (VICTORY_DIRTY_LIST=1)");
        }
        on
    })
}

/// With [`dirty_list`]: the copies among [`CARDED`] whose CPU side changed
/// since [`card_dirty`] last took what changed (see there for what holds).
static TOUCHED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// Since the last frame line: how often [`card_dirty`] was asked, the
/// copies it looked at, and its nanoseconds.
static DIRTY_LOOKS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// `shadow` is in the card's memory: among [`CARDED`] from here on.
fn carded(shadow: &Arc<Shadow>) {
    if dirty_list() {
        listed(shadow);
    }
    CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(shadow));
}

/// With [`dirty_list`]: from here on `shadow` goes into [`TOUCHED`] when it
/// changes. What changed in it while it was being made is taken as well.
fn listed(shadow: &Arc<Shadow>) {
    let Some(card) = &shadow.card else {
        return;
    };
    // (Under the lock a change takes: it sees `me` or we see its `touched`.)
    let _dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
    if card.me.set(Arc::downgrade(shadow)).is_ok() && card.touched.load(Ordering::Acquire) {
        TOUCHED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(shadow));
    }
}

/// Since the last frame line: bytes copied to card sides and the ranges
/// they were; pieces compared on the card, pieces fetched and their
/// bytes; the waits for the card the fetches added and their microseconds.
pub(crate) static CARD_STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// Bytes of buffer copies in the card's memory and in windows (for the
/// memory pressure log, [`crate::residency`]).
pub(crate) fn live_span_bytes() -> u64 {
    CARD_BYTES[0].load(Ordering::Relaxed)
        + WINDOW_CARD_BYTES.load(Ordering::Relaxed)
        + WINDOW_MAPPED_BYTES.load(Ordering::Relaxed)
}

/// [`CARD_STATS`], taken; then the copies in the card's memory, their
/// bytes and their "before" twins' bytes.
pub(crate) fn card_stats() -> [u64; 13] {
    let mut out = [0u64; 13];
    for (i, stat) in CARD_STATS.iter().enumerate() {
        out[i] = stat.swap(0, Ordering::Relaxed);
    }
    out[7] = CARDED.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|w| w.strong_count() > 0).count() as u64;
    out[8] = CARD_BYTES[0].load(Ordering::Relaxed)
        + WINDOW_CARD_BYTES.load(Ordering::Relaxed)
        + WINDOW_MAPPED_BYTES.load(Ordering::Relaxed);
    out[9] = CARD_BYTES[1].load(Ordering::Relaxed);
    for (i, stat) in DIRTY_LOOKS.iter().enumerate() {
        out[10 + i] = stat.swap(0, Ordering::Relaxed);
    }
    out
}

/// The bytes of the hashes kept on the card for a copy of `size` bytes: 16
/// for each 64 KiB piece (a copy that does not start on a piece boundary
/// has one piece more).
pub(crate) fn card_hash_bytes(size: u64) -> u64 {
    (size / PIECE as u64 + 2) * 16
}

/// A buffer of `size` bytes in the card's memory for a title buffer's
/// copy, if the copy is to be there: [`cards`] is on, the size is within
/// [`card_limits`] and below 1 GiB (the compare counts in 32 bits), and the
/// host has what it takes (`Host::card_ready`).
fn card_for(size: u64) -> Option<ID3D12Resource> {
    let there = card_room_for(size);
    if there.is_some() || !card_all() || size >= 1 << 30 {
        return there;
    }
    // Two sides all the same, the card's in the PC's memory.
    let host = host()?;
    if !cards() || !host.card_ready() {
        return None;
    }
    let (buffer, _) = host.shared_buffer(size)?;
    CARD_BYTES[0].fetch_add(size, Ordering::Relaxed);
    Some(buffer)
}

/// Whether every title buffer's copy has two sides, however small
/// (`VICTORY_CARD_ALL=1`; off by default).
///
/// A copy with one side, in the PC's memory, is what the card reads while
/// our refreshes write it: a submission that is prepared changes it under
/// the ones in the air ahead of it. The title's CPU does not write what they
/// still read, but its own buffer copies are done by us at preparing
/// (`crate::host::send` has the order of those right for copies with two
/// sides), and so are the bytes put in beside shaders' writes.
fn card_all() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ALL").is_some_and(|v| v == "1"))
}

/// [`card_for`] in the card's own memory.
fn card_room_for(size: u64) -> Option<ID3D12Resource> {
    if !cards() {
        return None;
    }
    let (least, most) = card_limits();
    if size < least || size >= 1 << 30 {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    // Only while the card has room: with this copy on it, a fifth of the
    // budget the system gives us must still be free. Copies are made when
    // the world loads, before the textures fill the card; the texture
    // queue's own rule (`resource::room`: nine tenths of the budget) counts
    // them, so the card's use stays where it was and the textures' share
    // shrinks by the copies' bytes, at most `card_limits`' second number.
    // (With `card_room_kept_on` the texture queue kept this copy's room
    // free, and the copy may go up to the queue's own line.)
    let hundredths = if card_room_kept_on() { 97 } else { 80 };
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 100 * hundredths
    {
        // Said, with the numbers it was decided by: run 193 made 31 copies
        // of 95 and no line named the others. The first 64, then every
        // 32nd.
        let n = CARD_REFUSED[0].fetch_add(1, Ordering::Relaxed) + 1;
        let bytes = CARD_REFUSED[1].fetch_add(size, Ordering::Relaxed) + size;
        if n <= 64 || n.is_multiple_of(32) {
            tracing::info!(
                "no card copy for a title buffer of {} KiB: the card has {} MB in use of a budget of {} MB and a copy is made up to \
                 {} MB; the copies made hold {} MB of at most {} MB; {} buffers turned down so far ({} MB), frame {}",
                size >> 10,
                used >> 20,
                budget >> 20,
                (budget / 100 * hundredths) >> 20,
                CARD_BYTES[0].load(Ordering::Relaxed) >> 20,
                most >> 20,
                n,
                bytes >> 20,
                crate::frame::current()
            );
        }
        return None;
    }
    if CARD_BYTES[0].fetch_add(size, Ordering::Relaxed) + size > most {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// The most bytes of windows outside buffers whose copies go to the card's
/// memory (`VICTORY_CARD_WINDOWS_MB`; 0, the default, none).
///
/// Windows are two thirds of the PC memory the card shares (run 104: 861
/// windows, 3,552 MB of 5,400), and every draw that takes vertices, indices
/// or constants from the title's own allocator reads them over the bus.
/// Shaders write none of them (run 91: "0.0 MB compared in 0 ranges of
/// windows"), so a window on the card needs only what a card copy has
/// already: the CPU's side takes title memory as before, and what changed
/// in it goes up before each submission. No hashes, no compare, no fetch,
/// unless a shader does declare a write to it one day.
///
/// A window goes to the card while there is room under this number and
/// while a fifth of the card's budget is free, as title buffers do
/// (`card_for`; [`window_usage_limit`]: 97 hundredths are used with
/// [`reserve_card_budgets`], whose room was kept): textures keep their room,
/// and once they have filled the
/// card later windows stay where they are. The CPU's side stays in the PC's
/// memory, so the memory shared with the card does not shrink by this
/// alone; what moves is what shaders read.
fn window_card_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_WINDOWS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0) << 20
    })
}

/// Bytes alive of windows' card sides ([`window_card_most`]).
static WINDOW_CARD_BYTES: AtomicU64 = AtomicU64::new(0);

/// A buffer in the card's memory for a window of `size` bytes, if it is to
/// be there ([`window_card_most`]).
fn window_card_for(key: (u64, u64), size: u64) -> Option<ID3D12Resource> {
    // Only windows of the kinds asked for (`VICTORY_CARD_WINDOWS_KINDS`).
    let chosen = match crate::readkinds::allowed() {
        Some(allowed) => {
            let plan = KIND_PLAN.lock().unwrap_or_else(|e| e.into_inner());
            let asked = crate::readkinds::asked_for().is_some_and(|kind| allowed & 1 << kind != 0);
            let used = WINDOW_CARD_BYTES.load(Ordering::Relaxed);
            if !plan.admits(key, size, asked, used, window_card_most()) {
                return None;
            }
            plan.waiting.contains(&key)
        }
        None => false,
    };
    let there = window_room_for(size);
    if chosen {
        // Made there, or turned down for want of room (and then not chosen
        // again until the list of those is cleared).
        KIND_PLAN.lock().unwrap_or_else(|e| e.into_inner()).landed(key, size, there.is_some());
    }
    if there.is_some() || !windows_own() || !cards() {
        return there;
    }
    // Two sides all the same, the card's in the PC's memory.
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    let (buffer, _) = host.shared_buffer(size)?;
    WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed);
    Some(buffer)
}

/// The most bytes of windows outside buffers that lie in the card's memory
/// and are filled by the CPU through the bus (`VICTORY_WINDOWS_BAR_MB`; 0, the
/// default, none): one buffer (`D3D12_HEAP_TYPE_GPU_UPLOAD`), no CPU's side, no
/// copy to send and no list of changes, unlike [`window_card_most`]. The CPU
/// only writes them, as no shader is declared to write a window; if one were,
/// its snapshot and compare would read through the bus, right but slow. In a
/// captured frame the lists take 26 ms of the card instead of 29, the frame 32
/// instead of 36; filling them costs the CPU what it did.
///
/// Windows go there while a fifth of the card's budget is free, as with
/// [`window_room_for`], which this comes before.
fn window_mapped_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_WINDOWS_BAR_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0) << 20
    })
}

/// Bytes alive of windows in card memory the CPU maps.
static WINDOW_MAPPED_BYTES: AtomicU64 = AtomicU64::new(0);

/// A buffer for a window of `size` bytes in card memory the CPU maps, if it is
/// to be there ([`window_mapped_most`]).
fn window_mapped_for(size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let most = window_mapped_most();
    if most == 0 {
        return None;
    }
    let host = host()?;
    if let Some([budget, used, ..]) = host.video_memory()
        && used.saturating_add(size) > window_usage_limit(budget, budgets_reserved(), window_cap())
    {
        return None;
    }
    if WINDOW_MAPPED_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        WINDOW_MAPPED_BYTES.fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let made = host.mapped_card_buffer(size, true);
    if made.is_none() {
        WINDOW_MAPPED_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    made
}

/// Whether windows outside buffers have two sides too, so that submissions
/// in the air read the bytes they were prepared with (`VICTORY_WINDOWS_OWN=1`;
/// off by default: it takes the windows' bytes again, 3.3 GB in capture
/// replay2, and what changes in them under submissions in the air is what the
/// title's CPU writes, which it does not while they read it).
fn windows_own() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOWS_OWN").is_some_and(|v| v == "1"))
}

/// [`window_card_for`] in the card's own memory.
fn window_room_for(size: u64) -> Option<ID3D12Resource> {
    let most = window_card_most();
    if most == 0 || !cards() {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    if let Some([budget, used, ..]) = host.video_memory()
        && used.saturating_add(size) > window_usage_limit(budget, budgets_reserved(), window_cap())
    {
        return None;
    }
    if WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// `VICTORY_CARD_WINDOWS_CAP=<percent>` (50 to 97; not set by default): the
/// share of the card's budget in use under which a window goes to the card
/// ([`window_usage_limit`]), whatever [`reserve_card_budgets`] says.
fn window_cap() -> Option<u64> {
    static CAP: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| std::env::var("VICTORY_CARD_WINDOWS_CAP").ok().as_deref().and_then(parse_window_cap))
}

/// [`window_cap`] of its value: a whole number, held to 50..=97. Pure.
fn parse_window_cap(value: &str) -> Option<u64> {
    value.trim().parse::<u64>().ok().map(|percent| percent.clamp(50, 97))
}

/// The hundredths of the card's budget in use under which a window goes
/// there: `cap` if it was asked for, else 97 with the budgets `reserved`
/// (their room was kept), else 80. Pure.
fn window_cap_percent(reserved: bool, cap: Option<u64>) -> u64 {
    cap.unwrap_or(if reserved { 97 } else { 80 })
}

/// The most of the card's budget in use, in bytes, under which a window
/// goes there ([`window_cap_percent`]). Pure.
fn window_usage_limit(budget: u64, reserved: bool, cap: Option<u64>) -> u64 {
    match (cap, reserved) {
        // (As it always was, to the byte.)
        (None, false) => budget / 10 * 8,
        _ => budget / 100 * window_cap_percent(reserved, cap),
    }
}

/// What went up, by send: its number, and the offsets, in order and joined
/// (a refresh sends thousands of runs of a few words).
#[derive(Default)]
struct SentRuns(Vec<(u64, Vec<(usize, usize)>)>);

impl SentRuns {
    #[cfg(test)]
    fn note(&mut self, number: u64, ranges: Vec<(usize, usize)>) {
        self.note_joined(number, joined(ranges));
    }

    /// `ranges` are in order and joined.
    fn note_joined(&mut self, number: u64, ranges: Vec<(usize, usize)>) {
        if ranges.is_empty() {
            return;
        }
        match self.0.iter_mut().find(|s| s.0 == number) {
            Some((_, all)) => *all = joined([std::mem::take(all), ranges].concat()),
            None => self.0.push((number, ranges)),
        }
    }

    /// Sends up to `number` are of no use any more.
    fn forget(&mut self, number: u64) {
        self.0.retain(|s| s.0 > number);
    }

    /// Adds to `out` what sends after `number` took of `[from, to)`.
    fn near(&self, number: u64, from: usize, to: usize, out: &mut Vec<(usize, usize)>) {
        for (_, ranges) in self.0.iter().filter(|s| s.0 > number) {
            let first = ranges.partition_point(|r| r.1 <= from);
            out.extend(ranges[first..].iter().take_while(|r| r.0 < to).copied());
        }
    }

    #[cfg(test)]
    fn of(all: &[(usize, usize, u64)]) -> SentRuns {
        let mut sent = SentRuns::default();
        for &(a, b, number) in all {
            sent.note(number, vec![(a, b)]);
        }
        sent
    }
}

/// The copies in the card's memory whose CPU side changed since they were
/// last asked: each with the changed offsets, in order and joined where
/// they touch. The ranges are taken: the caller copies them to the card's
/// side (`Host::card_before`).
///
/// With [`relay`], `sent_as` is the number of the send that takes them
/// (`crate::host`'s order of the queue): each range is noted with it, in
/// the same lock that took it, so that a merge of pieces fetched before
/// that send knows the card did not hold these bytes yet
/// ([`Shadow::card_merge_relay`]). Notes of sends that every fetch still to
/// be merged came after are dropped.
pub(crate) fn card_dirty(sent_as: u64) -> Vec<(Arc<Shadow>, Vec<(usize, usize)>)> {
    let started = std::time::Instant::now();
    let (out, looked) = if dirty_list() { card_dirty_listed(sent_as) } else { card_dirty_walked(sent_as) };
    DIRTY_LOOKS[0].fetch_add(1, Ordering::Relaxed);
    DIRTY_LOOKS[1].fetch_add(looked as u64, Ordering::Relaxed);
    DIRTY_LOOKS[2].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    out
}

/// What [`card_dirty`] takes of one copy: its changed offsets, if its
/// `touched` was set.
fn card_dirty_of(shadow: &Shadow, sent_as: u64, fetched: u64) -> Vec<(usize, usize)> {
    shadow
        .card
        .as_ref()
        .filter(|c| c.touched.swap(false, Ordering::AcqRel))
        .map(|c| {
            let mut dirty = c.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let ranges = joined(std::mem::take(&mut *dirty));
            if relay() {
                let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
                sent.forget(fetched);
                sent.note_joined(sent_as, ranges.clone());
            }
            ranges
        })
        .unwrap_or_default()
}

/// [`card_dirty`] by a look at every copy in the card's memory, and how many
/// were looked at.
fn card_dirty_walked(sent_as: u64) -> (Vec<(Arc<Shadow>, Vec<(usize, usize)>)>, usize) {
    let mut out = Vec::new();
    let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    let looked = carded.len();
    carded.retain(|weak| {
        let Some(shadow) = weak.upgrade() else {
            return false;
        };
        let ranges = card_dirty_of(&shadow, sent_as, fetched);
        if !ranges.is_empty() {
            out.push((shadow, ranges));
        }
        true
    });
    (out, looked)
}

/// [`card_dirty`] with [`dirty_list`]: only the copies in [`TOUCHED`] are
/// looked at, and given in the order they were made, as the walk gives
/// them. (Port of miamiwalk ebcf56f1.)
fn card_dirty_listed(sent_as: u64) -> (Vec<(Arc<Shadow>, Vec<(usize, usize)>)>, usize) {
    // (The copies that are gone leave the list of all of them now and then:
    // the walk did that at every call.)
    static CALLS: AtomicU64 = AtomicU64::new(0);
    if CALLS.fetch_add(1, Ordering::Relaxed) % 4096 == 4095 {
        CARDED.lock().unwrap_or_else(|e| e.into_inner()).retain(|weak| weak.strong_count() > 0);
    }
    let mut touched = std::mem::take(&mut *TOUCHED.lock().unwrap_or_else(|e| e.into_inner()));
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    let looked = touched.len();
    let mut out = Vec::new();
    for shadow in touched.iter().filter_map(std::sync::Weak::upgrade) {
        let ranges = card_dirty_of(&shadow, sent_as, fetched);
        if !ranges.is_empty() {
            out.push((shadow, ranges));
        }
    }
    out.sort_by_key(|(shadow, _)| shadow.id);
    // (Its room is kept for the next changes.)
    touched.clear();
    let mut list = TOUCHED.lock().unwrap_or_else(|e| e.into_inner());
    if list.capacity() < touched.capacity() {
        touched.append(&mut list);
        *list = touched;
    }
    (out, looked)
}

/// With the line of the copies in the card's memory: what finding the
/// changed ones cost ([`card_dirty`]; `stats` is [`card_stats`]'s, over
/// `frames` frames).
pub(crate) fn dirty_looks_line(stats: &[u64; 13], frames: f64) {
    let [asked, looked, ns] = [stats[10], stats[11], stats[12]];
    if asked == 0 {
        return;
    }
    tracing::info!(
        "what changed in copies in the card's memory was asked {:.1} times a frame: {:.0} copies looked at each \
         time, {:.2} ms a frame on the submitting thread (VICTORY_DIRTY_LIST=1 looks only at those that changed: \
         now {})",
        asked as f64 / frames,
        looked as f64 / asked as f64,
        ns as f64 / 1e6 / frames,
        if dirty_list() { "on" } else { "off" }
    );
}

#[cfg(test)]
mod dirty_list_tests {
    use super::{Shadow, TOUCHED, card_dirty_listed, card_dirty_of, joined, listed};
    use std::sync::{Arc, Mutex};

    /// The tests here share [`TOUCHED`] (nothing else puts into it without
    /// the switch): one at a time.
    static ONE: Mutex<()> = Mutex::new(());

    const LEN: usize = 1 << 16;

    /// Copies with a card side and no buffers; nothing reads their bytes.
    fn copies(n: usize) -> Vec<Arc<Shadow>> {
        let data = Box::leak(vec![0u8; LEN].into_boxed_slice()).as_mut_ptr();
        (0..n).map(|i| Shadow::unbacked(0x4000_0000 + (i * LEN) as u64, data, LEN as u64, true)).collect()
    }

    fn random(seed: &mut u64) -> usize {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 16) as usize
    }

    /// What the list gives is what a look at every copy gave: each changed
    /// copy once, in the order the copies were made, with its changed
    /// offsets joined; and nothing of a copy that did not change since.
    #[test]
    fn the_list_gives_each_changed_copy_once_with_its_ranges() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let all = copies(64);
        all.iter().for_each(listed);
        let _ = card_dirty_listed(0);
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut model: Vec<Vec<(usize, usize)>> = vec![Vec::new(); all.len()];
        for step in 0..4000 {
            let i = random(&mut seed) % all.len();
            let from = random(&mut seed) % (LEN - 64);
            let to = from + 1 + random(&mut seed) % 64;
            all[i].resend(from, to);
            model[i].push((from, to));
            if step % 37 != 36 {
                continue;
            }
            let (taken, looked) = card_dirty_listed(0);
            let changed = model.iter().filter(|m| !m.is_empty()).count();
            assert_eq!((taken.len(), looked), (changed, changed), "each changed copy once, no other looked at");
            assert!(taken.windows(2).all(|w| w[0].0.id < w[1].0.id), "in the order they were made");
            for (shadow, ranges) in taken {
                let i = all.iter().position(|c| Arc::ptr_eq(c, &shadow)).expect("one of ours");
                assert_eq!(ranges, joined(std::mem::take(&mut model[i])));
            }
            assert!(model.iter().all(Vec::is_empty));
        }
        let _ = card_dirty_listed(0);
        assert_eq!(card_dirty_listed(0).1, 0, "nothing changed since: nothing looked at");
    }

    /// What changed in a copy before it was among the listed ones is taken;
    /// a copy that is not listed (the switch unset) never goes into the list
    /// and keeps its mark for the look at every copy.
    #[test]
    fn a_copy_changed_while_made_is_taken_and_an_unlisted_one_is_left_to_the_walk() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let _ = card_dirty_listed(0);
        let all = copies(2);
        all[0].resend(8, 24);
        all[1].resend(8, 24);
        listed(&all[0]);
        // (Listed twice: once in the list.)
        listed(&all[0]);
        all[0].resend(40, 48);
        let (taken, looked) = card_dirty_listed(0);
        assert_eq!(looked, 1);
        assert_eq!(taken.len(), 1);
        assert!(Arc::ptr_eq(&taken[0].0, &all[0]));
        assert_eq!(taken[0].1, [(8, 24), (40, 48)]);
        assert_eq!(card_dirty_of(&all[1], 0, 0), [(8, 24)], "as before the list");
        assert!(card_dirty_of(&all[1], 0, 0).is_empty());
    }

    /// A copy let go while the list held it is passed over, and one that is
    /// in the list twice is taken once.
    #[test]
    fn copies_that_are_gone_or_listed_twice_do_no_harm() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let _ = card_dirty_listed(0);
        let all = copies(1);
        listed(&all[0]);
        all[0].resend(0, 4);
        {
            let mut list = TOUCHED.lock().unwrap_or_else(|e| e.into_inner());
            list.push(std::sync::Weak::new());
            list.push(Arc::downgrade(&all[0]));
        }
        let (taken, looked) = card_dirty_listed(0);
        assert_eq!(looked, 3);
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].1, [(0, 4)]);
    }

    /// Threads that change copies while another takes what changed: every
    /// offset changed is taken, by the end, and none that was not.
    #[test]
    fn nothing_changed_beside_the_taking_is_lost() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let _ = card_dirty_listed(0);
        let all = copies(8);
        all.iter().for_each(listed);
        const THREADS: usize = 4;
        const EACH: usize = 20_000;
        let done = std::sync::atomic::AtomicUsize::new(0);
        let mut got = vec![vec![false; LEN]; all.len()];
        let take = |got: &mut Vec<Vec<bool>>| {
            for (shadow, ranges) in card_dirty_listed(0).0 {
                let i = all.iter().position(|c| Arc::ptr_eq(c, &shadow)).expect("one of ours");
                for (from, to) in ranges {
                    got[i][from..to].fill(true);
                }
            }
        };
        let wanted: Vec<Vec<(usize, usize, usize)>> = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..THREADS)
                .map(|t| {
                    let (all, done) = (&all, &done);
                    scope.spawn(move || {
                        let mut seed = 0xD1B5_4A32_D192_ED03u64 ^ ((t as u64 + 1) << 32);
                        let mut wanted = Vec::with_capacity(EACH);
                        for _ in 0..EACH {
                            let i = random(&mut seed) % all.len();
                            let from = random(&mut seed) % (LEN - 8);
                            let to = from + 1 + random(&mut seed) % 8;
                            all[i].resend(from, to);
                            wanted.push((i, from, to));
                        }
                        done.fetch_add(1, std::sync::atomic::Ordering::Release);
                        wanted
                    })
                })
                .collect();
            while done.load(std::sync::atomic::Ordering::Acquire) < THREADS {
                take(&mut got);
            }
            threads.into_iter().map(|t| t.join().unwrap()).collect()
        });
        take(&mut got);
        let mut want = vec![vec![false; LEN]; all.len()];
        for (i, from, to) in wanted.into_iter().flatten() {
            want[i][from..to].fill(true);
        }
        assert!(got == want, "the offsets taken are the offsets changed");
        assert_eq!(card_dirty_listed(0).0.len(), 0);
    }
}

/// Diagnostics (`crate::host::upcheck`, nothing in the air): copies whose
/// pages the write watch and the guards hold for unchanged since they were
/// taken, compared with title memory, `most` bytes a call in turn. `say` gets
/// the copy, the title address of a page that is not the same, the words that
/// differ and the first of them (its address, the copy's value, title
/// memory's). Returns the bytes compared.
pub(crate) fn stale_pages(most: usize, say: &mut dyn FnMut(&Shadow, usize, usize, (usize, u32, u32))) -> usize {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let mut all: Vec<Arc<Shadow>> = buffers_read().values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(LOOSE.write().unwrap_or_else(|e| e.into_inner()).values().cloned());
    all.retain(|s| s.place.is_none() && s.watched.load(Ordering::Relaxed) == 1);
    all.sort_by_key(|s| s.id);
    let next = NEXT.load(Ordering::Relaxed);
    let turn = all.partition_point(|s| s.id < next);
    let mut done = 0;
    let mut title = vec![0u8; PAGE];
    for shadow in all[turn..].iter().chain(&all[..turn]) {
        if done >= most {
            NEXT.store(shadow.id, Ordering::Relaxed);
            break;
        }
        let (start, end) = (shadow.start as usize, (shadow.start + shadow.len) as usize);
        // (What it finds written stays noted for the next refresh.)
        let Some((_, written)) = crate::watch::writes(start, end, true) else {
            continue;
        };
        let written: std::collections::HashMap<usize, u64> = written.into_iter().collect();
        let copied = shadow.copied.lock().unwrap_or_else(|e| e.into_inner()).clone();
        for (i, &at) in copied.iter().enumerate() {
            let page = start / PAGE + i;
            if at == 0 || written.get(&page).is_some_and(|&w| w > at) {
                continue;
            }
            let (lo, hi) = ((page * PAGE).max(start), ((page + 1) * PAGE).min(end));
            let title = &mut title[..hi - lo];
            if !crate::host::read_into(lo as *const u8, title) {
                continue;
            }
            done += hi - lo;
            // SAFETY: inside the mapping; nothing is in the air and the
            // submission lock is held.
            let held = unsafe { std::slice::from_raw_parts(shadow.data.add(lo - start).cast_const(), hi - lo) };
            if held == title {
                continue;
            }
            let word = |b: &[u8], k: usize| u32::from_le_bytes([b[k], b[k + 1], b[k + 2], b[k + 3]]);
            let differ: Vec<usize> = (0..(hi - lo) & !3).step_by(4).filter(|&k| word(held, k) != word(title, k)).collect();
            // Written meanwhile: not stale, and the next refresh sees it.
            let again = crate::watch::writes(lo, hi, true).is_none_or(|(_, w)| w.iter().any(|&(p, w)| p == page && w > at));
            if let Some(&k) = differ.first()
                && !again
            {
                say(shadow, lo, differ.len(), (lo + k, word(held, k), word(title, k)));
            }
        }
    }
    done
}

/// Copies of memory (from, to, bytes: apart), on several threads if they are
/// many.
pub(crate) fn copy_all(copies: Vec<(usize, usize, usize)>) {
    let bytes: usize = copies.iter().map(|c| c.2).sum();
    if bytes > 4 << 20 {
        wide(copy_jobs(copies));
        return;
    }
    for (src, dst, len) in copies {
        // SAFETY: the caller's ranges.
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
    }
}

/// A title buffer copy was done on the CPU into `[dst, dst + len)`: all of it
/// goes up to the copies in the card's memory that hold it, whether their
/// CPU side held the same before or not. Left out is what shaders in the air
/// may be writing (such a copy waits for them, or is done on the card).
pub(crate) fn send_copied(dst: usize, len: usize) {
    for shadow in carded_copies() {
        let (start, end) = (shadow.start as usize, (shadow.start + shadow.len) as usize);
        let (from, to) = (dst.max(start), (dst + len).min(end));
        if from >= to {
            continue;
        }
        let busy: Vec<(usize, usize)> =
            shadow.pending.lock().unwrap_or_else(|e| e.into_inner()).iter().copied().filter(|&(a, b)| a < to && b > from).collect();
        for (a, b) in outside(from, to, busy) {
            shadow.copy(a, b);
        }
    }
}

/// The copies in the card's memory (diagnostics: `crate::host::upcheck`).
pub(crate) fn carded_copies() -> Vec<Arc<Shadow>> {
    CARDED.lock().unwrap_or_else(|e| e.into_inner()).iter().filter_map(std::sync::Weak::upgrade).collect()
}

/// Ranges in order, those that overlap or touch as one.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    // (The sort that finds what is in order already: they come as a few
    // stretches of tens of thousands in order, put behind one another.)
    ranges.sort();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Runs of pieces to hash again (`Host::card_before`), those that overlap
/// or touch as one, in order: every piece of `runs` is in the result once
/// and no other piece is. Ranges sent up lie close together (514 a frame
/// for 87 MB in run 102), and each had a pass of its own over its pieces,
/// so a piece that held several ranges was hashed once for each.
pub(crate) fn joined_piece_runs(mut runs: Vec<CardSpan>) -> Vec<CardSpan> {
    runs.sort_unstable_by_key(|r| r.first);
    let mut out: Vec<CardSpan> = Vec::new();
    for run in runs {
        match out.last_mut() {
            Some(last) if run.first <= last.first + last.pieces => {
                last.pieces = last.pieces.max(run.first + run.pieces - last.first);
                last.to = last.to.max(run.to);
            }
            _ => out.push(run),
        }
    }
    out
}

/// A declared range of a copy in the card's memory, as the compare on the
/// card takes it: its offsets in the copy, its first 64 KiB piece (pieces
/// are cut at multiples of 64 KiB of title memory; piece 0 holds offset
/// 0), how many pieces hold it, and where the copy starts in piece 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardSpan {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) first: u32,
    pub(crate) pieces: u32,
    pub(crate) skew: u32,
}

/// [`CardSpan`] for offsets `[from, to)` (`from < to`) of a copy of title
/// memory from `start` on.
fn card_span_of(start: usize, from: usize, to: usize) -> CardSpan {
    let pieces = pieces_of(start, from, to);
    CardSpan {
        from: from as u32,
        to: to as u32,
        first: pieces.start as u32,
        pieces: pieces.len() as u32,
        skew: (start % PIECE) as u32,
    }
}

/// The offsets of piece `i` of `span` that lie in the span, for a copy of
/// `len` bytes of title memory from `start` on: what is fetched and merged
/// of a piece the compare found differing.
fn card_piece_of(start: usize, len: usize, span: &CardSpan, i: u32) -> (usize, usize) {
    let (lo, hi) = piece_at(start, len, (span.first + i) as usize);
    (lo.max(span.from as usize), hi.min(span.to as usize))
}

/// The offsets [`Shadow::card_hashed`] gives, for a card's buffer of `size`
/// bytes for title memory from `start` on.
fn card_hashed_of(start: usize, size: usize, piece: usize) -> (usize, usize) {
    let (lo, hi) = piece_at(start, size, piece);
    (lo & !15, hi.next_multiple_of(16))
}

#[cfg(test)]
mod card_tests {
    use super::{CardSpan, PIECE, card_hashed_of, card_piece_of, card_span_of, joined, joined_piece_runs};

    /// Joined runs hold every piece of the runs given, once, and no other.
    #[test]
    fn piece_runs_join_without_losing_or_adding_pieces() {
        let run = |first: u32, pieces: u32| CardSpan {
            from: first * 65536,
            to: (first + pieces) * 65536,
            first,
            pieces,
            skew: 7,
        };
        let runs = vec![run(40, 1), run(3, 2), run(4, 1), run(5, 3), run(40, 1), run(9, 1), run(41, 2), run(100, 1)];
        let mut wanted: Vec<u32> = runs.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let joined = joined_piece_runs(runs);
        let got: Vec<u32> = joined.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        assert_eq!(got, wanted);
        assert_eq!(joined.iter().map(|r| (r.first, r.pieces)).collect::<Vec<_>>(), [(3, 5), (9, 1), (40, 3), (100, 1)]);
        assert!(joined.iter().all(|r| r.skew == 7));
        assert!(joined_piece_runs(Vec::new()).is_empty());
    }

    /// The offsets the compare's shader takes for a piece (`CARD_COMPARE`
    /// in host.rs), before it widens them to 16 bytes.
    fn shader_piece(span: &CardSpan, i: u32) -> (u32, u32) {
        let piece = span.first + i;
        let lo = piece * 65536;
        let lo = if lo > span.skew { lo - span.skew } else { 0 };
        let hi = (piece + 1) * 65536 - span.skew;
        (lo.max(span.from), hi.min(span.to))
    }

    /// The pieces of a span cover it without gaps, and the shader's
    /// arithmetic names the same bytes as ours.
    #[test]
    fn the_shader_and_the_merge_cut_the_same_pieces() {
        for start in [0usize, 4096, PIECE - 16, PIECE, 3 * PIECE + 0x1f00, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                let mut at = from;
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    assert_eq!(lo, at, "{start:#x} {from} {to} piece {i}");
                    assert!(hi > lo && hi - lo <= PIECE);
                    assert_eq!(shader_piece(&span, i), (lo as u32, hi as u32), "{start:#x} {from} {to} piece {i}");
                    at = hi;
                }
                assert_eq!(at, to, "{start:#x} {from} {to}");
            }
        }
    }

    /// Changed ranges are joined where they overlap or touch, and only there.
    /// What is fetched of a piece to hash it is what the shader hashes.
    #[test]
    fn a_piece_is_hashed_here_as_on_the_card() {
        for start in [0usize, 0x10_0000, 0xd_5edb_c780, 0x12_3456_7008] {
            for size in [1usize << 16, 5 << 16, 0x20_0000] {
                let span = card_span_of(start, 0, size);
                for i in 0..span.pieces {
                    // As `CARD_COMPARE` has it.
                    let (piece, skew, size) = (span.first + i, span.skew, size as u32);
                    let lo = (piece * 65536).saturating_sub(skew);
                    let hi = ((piece + 1) * 65536 - skew).min(size);
                    let shader = (lo & !15, (hi + 15) & !15);
                    let here = card_hashed_of(start, size as usize, piece as usize);
                    assert_eq!((here.0 as u32, here.1 as u32), shader, "start {start:#x} size {size:#x} piece {piece}");
                }
            }
        }
    }

    #[test]
    fn ranges_join_where_they_touch() {
        assert_eq!(joined(vec![(8, 12), (0, 4), (4, 6), (10, 20), (30, 31)]), [(0, 6), (8, 20), (30, 31)]);
        assert!(joined(Vec::new()).is_empty());
    }
}

/// Whether the copy-back of what shaders wrote runs on a thread of its own,
/// beside the submitting thread and outside the submission lock
/// (`VICTORY_RELAY=1` or `2`; off by default; not with kept snapshots,
/// [`keeps`]). `crate::host::relay` says what it does; here it changes
/// what a copy keeps:
///
/// - the ranges shaders may be writing are counted per submission
///   ([`Flights`]): a range two flights declared stays closed to refreshes
///   until both have been copied back;
/// - the merge of what the card wrote never stores into memory the card
///   may be writing ([`merge_piece`]): what it learns goes into title
///   memory and into the copy's reference (the snapshot, or the CPU's side
///   of a copy in the card's memory), under that reference's lock;
/// - what goes up to a copy in the card's memory is numbered
///   ([`card_dirty`]), so that a merge can tell the title's bytes the card
///   had not received from bytes the card wrote.
pub(crate) fn relay() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default (stage 2, `crate::host::relay_exact`); `VICTORY_RELAY=0`
    // turns it off, `=1` asks for the first stage.
    // The crate's own tests run without it unless they ask: several of
    // them look at title memory right after a submission, which the
    // relay's landing thread may still be writing.
    *ON.get_or_init(|| std::env::var("VICTORY_RELAY").map_or(!cfg!(test), |v| v == "1" || v == "2") && !keeps())
}

/// The submission being prepared (its number among those started): what
/// [`Shadow::writing`] counts a declared range for ([`Flights`]).
static PREPARED: AtomicU64 = AtomicU64::new(0);

/// A submission's preparing begins (`crate::host`, under the submission
/// lock).
pub(crate) fn preparing(number: u64) {
    PREPARED.store(number, Ordering::Relaxed);
}

/// With [`relay`]: the number of the last send of changed ranges to the
/// card (`crate::host`'s order of the queue) that every fetch still to be
/// merged came after. Notes of sends up to it are of no use any more.
static RELAY_FETCHED: AtomicU64 = AtomicU64::new(0);

/// A fetch sent after send `number` is the oldest one that may still be
/// merged ([`RELAY_FETCHED`]).
pub(crate) fn relay_fetched(number: u64) {
    RELAY_FETCHED.fetch_max(number, Ordering::AcqRel);
}

/// With [`relay`]: how many submissions that are not copied back yet
/// declared each range shaders may be writing (range, count, the last
/// submission that declared it).
///
/// Without it a range declared by two flights was one entry, and the end
/// of the first flight's copy-back removed it while the second was still
/// writing: the next refresh would have put title bytes over what the
/// second flight's shaders were writing.
#[derive(Default)]
struct Flights(std::collections::HashMap<(usize, usize), (u32, u64)>);

impl Flights {
    /// Submission `number` declares `range`: counted once for it, however
    /// often it declares it.
    fn declare(&mut self, range: (usize, usize), number: u64) {
        let f = self.0.entry(range).or_insert((0, number));
        if f.0 == 0 || f.1 != number {
            *f = (f.0 + 1, number);
        }
    }

    /// One submission that declared `ranges` (each once) is copied back:
    /// the ranges no other submission still holds, in order.
    fn done(&mut self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut free = Vec::new();
        for range in ranges {
            match self.0.get_mut(range) {
                Some(f) if f.0 > 1 => f.0 -= 1,
                Some(_) => {
                    self.0.remove(range);
                    free.push(*range);
                }
                // Not counted (declared before anything counted): free.
                None => free.push(*range),
            }
        }
        free.sort_unstable();
        free
    }
}

/// The relay's merge of a stretch of a copy ([`relay`]). `now` is what the
/// card's memory holds, read once; `held` is the copy's reference for the
/// stretch (its snapshot, or the CPU's side of a copy in the card's
/// memory): what the card's memory held when it was last looked at;
/// `title` is title memory as just read. Every byte the card changed
/// (`now` differs from `held`) goes into `title` and into `held`; every
/// other byte of both stays as it is, so a byte the title wrote and the
/// card did not keeps the title's value. Whether `title` changed.
///
/// Nothing is stored into `now`'s source: the card may be writing it for a
/// later flight at this moment, and a store there, even of the value just
/// read, could undo that write. (The merge of a waited-for submission
/// stores every byte of a differing stretch back into the copy.)
fn merge_piece(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((n, h), t) in now.iter().zip(held.iter_mut()).zip(title.iter_mut()) {
        let wrote = if *n != *h { 0xffu8 } else { 0 };
        let merged = (*t & !wrote) | (*n & wrote);
        differs |= merged ^ *t;
        *t = merged;
        *h = (*h & !wrote) | (*n & wrote);
    }
    differs != 0
}

/// `[from, to)` without the ranges of `skip`, in order.
fn outside(from: usize, to: usize, mut skip: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    skip.sort_unstable();
    let mut out = Vec::new();
    let mut at = from;
    for (a, b) in skip {
        if b <= at {
            continue;
        }
        if a >= to {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(b);
    }
    if at < to {
        out.push((at, to));
    }
    out
}

#[cfg(test)]
mod relay_tests {
    use super::{Flights, changed_runs, merge_piece, outside};

    /// A range two flights declared is free only when both are done, in
    /// either order; a submission that declares it many times counts once.
    #[test]
    fn a_range_stays_closed_until_every_flight_that_declared_it_is_done() {
        let (a, b) = ((0x1000, 0x2000), (0x1800, 0x3000));
        let mut flights = Flights::default();
        for _ in 0..3 {
            flights.declare(a, 7);
        }
        flights.declare(b, 7);
        flights.declare(a, 8);
        // Flight 7 lands: `a` is still flight 8's, `b` was only flight 7's.
        assert_eq!(flights.done(&[a, b]), vec![b]);
        // Declared again meanwhile by a third.
        flights.declare(a, 9);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        assert!(flights.0.is_empty());
        // The later flight first: the same.
        flights.declare(a, 1);
        flights.declare(a, 2);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        // A range never counted is free at once.
        assert_eq!(flights.done(&[b]), vec![b]);
    }

    /// The merge moves exactly the bytes the card changed, into title
    /// memory and into the reference, and stores nothing else.
    #[test]
    fn the_merge_stores_only_what_the_card_changed() {
        let held0: Vec<u8> = (0..200u32).map(|i| (i * 5 + 1) as u8).collect();
        let mut now = held0.clone();
        for at in [0usize, 3, 4, 5, 64, 199] {
            now[at] ^= 0x33;
        }
        // The title wrote other bytes meanwhile, and one the card wrote too.
        let mut title = held0.clone();
        for at in [1usize, 5, 100] {
            title[at] = 0xee;
        }
        let title0 = title.clone();
        let mut held = held0.clone();
        assert!(merge_piece(&now, &mut held, &mut title));
        for at in 0..200 {
            if now[at] != held0[at] {
                assert_eq!((title[at], held[at]), (now[at], now[at]), "the card's byte at {at}");
            } else {
                assert_eq!((title[at], held[at]), (title0[at], held0[at]), "untouched at {at}");
            }
        }
        // Again with nothing new: nothing changes and nothing is reported.
        let (title1, held1) = (title.clone(), held.clone());
        assert!(!merge_piece(&now, &mut held, &mut title));
        assert_eq!((title, held), (title1, held1));
    }

    #[test]
    fn ranges_are_cut_out() {
        assert_eq!(outside(10, 50, vec![]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(0, 10), (50, 60)]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(30, 35), (0, 12), (34, 40), (48, 99)]), vec![(12, 30), (40, 48)]);
        assert_eq!(outside(10, 50, vec![(0, 99)]), vec![]);
    }

    /// A model of a copy in the PC's memory under the relay: the card
    /// writes `copy` directly; `snap` is the snapshot; `title` is title
    /// memory. Flight A and flight B declare overlapping ranges and both
    /// are in the air; the title's CPU writes in between and its bytes are
    /// put in beside the flights' (`merge_under_writes`: into the copy and
    /// the snapshot); A lands while B is still writing; then B lands.
    #[test]
    fn two_flights_overlapping_ranges_and_a_cpu_write_in_between() {
        const N: usize = 64;
        let base: Vec<u8> = (0..N as u32).map(|i| i as u8).collect();
        let (mut copy, mut title) = (base.clone(), base.clone());
        // A declares [0, 40), B declares [24, 64): each byte's snapshot is
        // taken when the first range that holds it is declared.
        let (a, b) = ((0usize, 40usize), (24usize, 64usize));
        let mut snap = base.clone();
        // The card runs A: writes 4..12 and 30..34.
        for at in (4..12).chain(30..34) {
            copy[at] = 0xa0;
        }
        // The title's CPU writes 8..10 (over A's bytes), 20 and 50, after
        // A's refresh: the next submission's preparing finds them against
        // the snapshot and puts them in, into the copy and the snapshot.
        for at in [8usize, 9, 20, 50] {
            title[at] = 0x77;
        }
        for (at, n) in changed_runs(&title, &snap) {
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // The card starts B: it has written 32..36 so far, over A's 32..34.
        for at in 32..36 {
            copy[at] = 0xb0;
        }
        // A lands (B still on the card): its range is merged. The copy is
        // read once and not written.
        let seen = copy.clone();
        {
            let mut t = title[a.0..a.1].to_vec();
            assert!(merge_piece(&seen[a.0..a.1], &mut snap[a.0..a.1], &mut t));
            title[a.0..a.1].copy_from_slice(&t);
        }
        assert_eq!(copy, seen, "the merge stores nothing into the copy");
        // A's bytes are in title memory, except where the title wrote
        // after A's shaders did (8, 9: the title's); B's first bytes came
        // along; nothing of B's range beyond A's was touched.
        for at in (4..8).chain(10..12).chain(30..32) {
            assert_eq!(title[at], 0xa0, "A's byte at {at}");
        }
        assert_eq!((title[8], title[9], title[20], title[50]), (0x77, 0x77, 0x77, 0x77));
        for at in 32..36 {
            assert_eq!(title[at], 0xb0, "B's byte so far at {at}");
        }
        // The card finishes B: 36..44.
        for at in 36..44 {
            copy[at] = 0xb1;
        }
        // The title's CPU writes 40 after B's shader did: found against
        // the snapshot at the next preparing, which is before B lands.
        title[40] = 0x55;
        for (at, n) in changed_runs(&title[b.0..b.1], &snap[b.0..b.1]) {
            let at = at + b.0;
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // B lands.
        let seen = copy.clone();
        {
            let mut t = title[b.0..b.1].to_vec();
            assert!(merge_piece(&seen[b.0..b.1], &mut snap[b.0..b.1], &mut t));
            title[b.0..b.1].copy_from_slice(&t);
        }
        // B's results are there, never replaced by A's older ones.
        for at in 32..36 {
            assert_eq!(title[at], 0xb0);
        }
        for at in (36..40).chain(41..44) {
            assert_eq!(title[at], 0xb1, "B's byte at {at}");
        }
        assert_eq!(title[40], 0x55, "the title wrote it last");
        // In the declared ranges the copy, the snapshot and title memory
        // now agree byte for byte: nothing is left to hand back, and
        // nothing was lost.
        assert_eq!(copy[a.0..b.1], title[a.0..b.1]);
        assert_eq!(snap[a.0..b.1], title[a.0..b.1]);
        // A third merge finds nothing.
        let mut t = title.clone();
        assert!(!merge_piece(&copy, &mut snap, &mut t));
    }

    /// A model of a copy in the card's memory under the relay: `card` is
    /// the card's side, `cpu` the CPU's side, `title` title memory. The
    /// title's changed bytes go into the CPU's side at once and up to the
    /// card with a later send; pieces fetched before that send must not
    /// take the card's older bytes there for something the card wrote.
    #[test]
    fn title_bytes_not_yet_on_the_card_are_not_taken_for_shader_writes() {
        const N: usize = 48;
        let base: Vec<u8> = (0..N as u32).map(|i| (i + 100) as u8).collect();
        let (mut card, mut cpu, mut title) = (base.clone(), base.clone(), base.clone());
        // Flight A's shaders write 10..14 on the card.
        for at in 10..14 {
            card[at] = 0xa0;
        }
        // The title's CPU writes 12 (over A's) and 30; the preparing of
        // the next submission puts both into the CPU's side and notes them
        // to go up (send 5). They are not on the card yet.
        title[12] = 0x77;
        title[30] = 0x78;
        let mut noted = Vec::new();
        for (at, n) in changed_runs(&title, &cpu) {
            cpu[at..at + n].copy_from_slice(&title[at..at + n]);
            noted.push((at, at + n, 5u64));
        }
        // A's pieces are fetched after send 4, before send 5: the fetch
        // holds A's bytes and the card's old bytes at 12 and 30.
        let fetched = card.clone();
        let fetched_after = 4u64;
        // Send 5 goes up behind it.
        for &(a, b, _) in &noted {
            card[a..b].copy_from_slice(&cpu[a..b]);
        }
        // The merge leaves out what went up after the fetch.
        let skip: Vec<(usize, usize)> = noted.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)).collect();
        let mut changed = false;
        for (a, b) in outside(0, N, skip) {
            let mut t = title[a..b].to_vec();
            changed |= merge_piece(&fetched[a..b], &mut cpu[a..b], &mut t);
            title[a..b].copy_from_slice(&t);
        }
        assert!(changed);
        // A's bytes arrived; the title's two writes stand, in title memory
        // and on both sides of the copy.
        for at in [10usize, 11, 13] {
            assert_eq!((title[at], cpu[at]), (0xa0, 0xa0));
        }
        assert_eq!((title[12], cpu[12], card[12]), (0x77, 0x77, 0x77));
        assert_eq!((title[30], cpu[30], card[30]), (0x78, 0x78, 0x78));
        assert_eq!(card, cpu);
        assert_eq!(card, title);
        // Without the leaving out, the card's old byte at 30 would have
        // been taken for a shader write and put over the title's.
        let mut wrong_title = title.clone();
        let mut wrong_cpu = cpu.clone();
        merge_piece(&fetched, &mut wrong_cpu, &mut wrong_title);
        assert_eq!(wrong_title[30], base[30]);
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<SnapshotPool> = Mutex::new(SnapshotPool { buffers: Vec::new(), bytes: 0 });

#[derive(Default)]
struct SnapshotPool {
    /// Descending capacity order, preserved on insertion and removal.
    buffers: Vec<Vec<u8>>,
    bytes: usize,
}

impl SnapshotPool {
    fn take(&mut self, n: usize) -> Option<Vec<u8>> {
        let most = n.saturating_mul(4).max(1 << 20);
        let i = self.buffers.partition_point(|v| v.capacity() >= n).checked_sub(1)?;
        if self.buffers[i].capacity() > most {
            return None;
        }
        let v = self.buffers.remove(i);
        self.bytes -= v.capacity();
        Some(v)
    }

    fn put(&mut self, v: Vec<u8>) {
        let capacity = v.capacity();
        let i = self.buffers.partition_point(|b| b.capacity() >= capacity);
        self.buffers.insert(i, v);
        self.bytes += capacity;
    }

    /// Normal returns keep the largest buffers, including one oversized buffer.
    fn keep_largest(&mut self, count: usize, most: usize) -> Vec<Vec<u8>> {
        if self.buffers.len() <= count && self.bytes <= most {
            return Vec::new();
        }
        let (mut keep, mut bytes) = (0, 0);
        for v in self.buffers.iter().take(count) {
            let next = bytes + v.capacity();
            if keep > 0 && next > most {
                break;
            }
            keep += 1;
            bytes = next;
        }
        self.bytes = bytes;
        self.buffers.split_off(keep)
    }

    /// Pressure sweeps keep the smallest buffers. An idle sweep takes the whole pool.
    fn keep_smallest(&mut self, most: usize) -> Vec<Vec<u8>> {
        if most == 0 {
            self.bytes = 0;
            return std::mem::take(&mut self.buffers);
        }
        if self.bytes <= most {
            return Vec::new();
        }
        let (mut keep, mut bytes) = (0, 0);
        for v in self.buffers.iter().rev() {
            let next = bytes + v.capacity();
            if next > most {
                break;
            }
            keep += 1;
            bytes = next;
        }
        self.bytes = bytes;
        let kept = self.buffers.split_off(self.buffers.len() - keep);
        std::mem::replace(&mut self.buffers, kept)
    }
}

/// When a snapshot buffer was last taken from or given back to [`POOL`]
/// ([`crate::host::moment`]), for [`let_go_idle_snapshots`]. Read and
/// updated while holding the pool lock so a sweep sees the latest use.
static POOL_USED: AtomicU64 = AtomicU64::new(0);

/// An empty buffer with room for `n` bytes, from the pool if it has one
/// of about that size: a small copy's snapshot took the 174 MB buffer kept
/// for the big heap's, which then got another, until eight of them were
/// reserved (1.4 GB in run 24).
fn pooled(n: usize) -> Vec<u8> {
    let reused = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        POOL_USED.store(crate::host::moment(), Ordering::Relaxed);
        pool.take(n)
    };
    let mut v = reused.unwrap_or_else(|| Vec::with_capacity(n));
    v.clear();
    v
}

/// How many snapshot buffers are kept (`VICTORY_SNAPSHOT_POOL`; `8`: as
/// before).
///
/// A frame gives some 330 ranges of copies in the PC's memory to shaders, and
/// most are of copies of their own; with two or three frames in the air eight
/// buffers were the eight largest, and every other snapshot was memory fresh
/// from the kernel and given back to it: 11 % of the submitting thread in the
/// opening (profile cpu-1: `NtAllocateVirtualMemory` under
/// [`Shadow::writing`]), and the page faults of filling it.
fn snapshots_kept() -> usize {
    static MOST: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| std::env::var("VICTORY_SNAPSHOT_POOL").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1024))
}

/// Returns a snapshot buffer to the pool: at most [`snapshots_kept`] kept,
/// the largest, and of those the largest one and the next ones up to `KEPT`
/// bytes in all.
fn give_back(v: Vec<u8>) {
    const KEPT: usize = 512 << 20;
    let count = snapshots_kept();
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    POOL_USED.store(crate::host::moment(), Ordering::Relaxed);
    pool.put(v);
    // (Given back to the kernel without the lock: milliseconds.)
    let gone = pool.keep_largest(count, KEPT);
    drop(pool);
    drop(gone);
}

/// What [`POOL`] keeps after a sweep while the host is short of memory
/// ([`crate::residency::pressured`]): the smallest buffers, up to this many
/// bytes in all. They are the ones asked for most (most of a frame's 330
/// ranges are small), and the large ones are what fills the 512 MiB.
const POOL_PRESSURED_KEPT: usize = 32 << 20;

/// Lets go of the snapshot buffers [`POOL`] keeps when nobody wants them.
/// Only [`give_back`] ever trimmed the pool, and only when another buffer
/// came back, so once snapshots stopped the last 512 MiB stayed for the
/// rest of the run, low memory or not (a load test: 524,123 KiB kept after
/// five idle seconds; 13 MiB private once emptied by hand). Now
/// - idle for the time the other host caches wait ([`crate::host::ram_idle_ms`])
///   and four frames, all of it goes;
/// - while the host is short of memory, all but [`POOL_PRESSURED_KEPT`]
///   goes, the largest first, whatever the size of the largest. Returns can
///   refill the pool between sweeps.
///
/// The next snapshot is then memory fresh from the kernel, as it was for
/// every snapshot past the pool before. Called by the once-a-second sweep.
fn let_go_idle_snapshots() {
    const IDLE_FRAMES: u64 = 4;
    let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    let pressured = crate::residency::pressured();
    let gone = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let idle = crate::host::long_ago(POOL_USED.load(Ordering::Relaxed), idle_ms, IDLE_FRAMES);
        let Some(most) = snapshots_to_keep(idle, pressured, crate::residency::pressure_keep()) else { return };
        pool.keep_smallest(most)
    };
    // (Given back to the kernel without the lock.)
    drop(gone);
}

/// How many bytes of snapshot buffers [`POOL`] may keep: none once idle,
/// [`POOL_PRESSURED_KEPT`] while the host is short of memory (none with
/// `kept`, `crate::residency::pressure_keep`: they go before any
/// texture), otherwise what [`give_back`] keeps (`None`: nothing to do).
fn snapshots_to_keep(idle: bool, pressured: bool, kept: bool) -> Option<usize> {
    if idle || (pressured && kept) {
        Some(0)
    } else if pressured {
        Some(POOL_PRESSURED_KEPT)
    } else {
        None
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    fn sizes(pool: &[Vec<u8>]) -> Vec<usize> {
        pool.iter().map(Vec::capacity).collect()
    }

    fn pool_with_sizes(mib: &[usize]) -> SnapshotPool {
        let mut pool = SnapshotPool::default();
        for &n in mib {
            pool.put(Vec::with_capacity(n << 20));
        }
        pool
    }

    #[test]
    fn reuse_takes_the_smallest_fit_and_tracks_capacity() {
        let mut pool = pool_with_sizes(&[8, 1, 4, 2, 4]);
        assert!(pool.take(9 << 20).is_none());
        assert_eq!(pool.take(3 << 20).unwrap().capacity(), 4 << 20);
        assert_eq!(pool.take(5 << 20).unwrap().capacity(), 8 << 20);
        assert_eq!(sizes(&pool.buffers), [4 << 20, 2 << 20, 1 << 20]);
        assert_eq!(pool.bytes, 7 << 20);
        assert_eq!(pool.take(1).unwrap().capacity(), 1 << 20);
        assert!(pool.take(1).is_none());
        assert_eq!(pool.bytes, 6 << 20);
    }

    #[test]
    fn normal_returns_obey_size_and_count_limits() {
        let mut pool = pool_with_sizes(&[8, 2, 4, 1]);
        assert!(pool.keep_largest(4, 16 << 20).is_empty());
        assert_eq!(pool.bytes, 15 << 20);
        assert_eq!(sizes(&pool.keep_largest(2, 16 << 20)), [2 << 20, 1 << 20]);
        assert_eq!(pool.bytes, 12 << 20);
        assert_eq!(sizes(&pool.keep_largest(2, 7 << 20)), [4 << 20]);
        assert_eq!(pool.bytes, 8 << 20);
        assert_eq!(sizes(&pool.keep_largest(0, 7 << 20)), [8 << 20]);
        assert!(pool.buffers.is_empty());
        assert_eq!(pool.bytes, 0);
    }

    #[test]
    fn trimming_keeps_the_smallest_up_to_the_limit() {
        let mut pool = pool_with_sizes(&[8, 1, 4, 2]);
        let gone = pool.keep_smallest(7 << 20);
        assert_eq!(sizes(&pool.buffers), [4 << 20, 2 << 20, 1 << 20]);
        assert_eq!(sizes(&gone), [8 << 20]);
        assert_eq!(pool.bytes, 7 << 20);
        assert!(pool.keep_smallest(7 << 20).is_empty());
    }

    #[test]
    fn trimming_to_nothing_empties_the_pool() {
        let mut pool = pool_with_sizes(&[3, 1]);
        let gone = pool.keep_smallest(0);
        assert!(pool.buffers.is_empty());
        assert_eq!(pool.bytes, 0);
        assert_eq!(sizes(&gone), [3 << 20, 1 << 20]);
    }

    #[test]
    fn a_buffer_past_the_limit_goes_even_alone() {
        // `give_back` always keeps its largest buffer; under pressure that
        // one goes too.
        let mut pool = pool_with_sizes(&[600]);
        let gone = pool.keep_smallest(POOL_PRESSURED_KEPT);
        assert!(pool.buffers.is_empty());
        assert_eq!(pool.bytes, 0);
        assert_eq!(gone.len(), 1);
    }

    #[test]
    fn an_idle_pool_goes_and_a_pressed_one_shrinks() {
        for kept in [false, true] {
            assert_eq!(snapshots_to_keep(true, false, kept), Some(0));
            assert_eq!(snapshots_to_keep(true, true, kept), Some(0));
            assert_eq!(snapshots_to_keep(false, false, kept), None);
        }
        assert_eq!(snapshots_to_keep(false, true, false), Some(POOL_PRESSURED_KEPT));
        assert_eq!(snapshots_to_keep(false, true, true), Some(0));
    }

    /// A full pool used by eight snapshot workers. Run alone in a fresh process;
    /// reports wall-clock time for 80,000 acquire/return pairs in this process.
    #[test]
    #[ignore]
    fn pool_contention_load() {
        const WORKERS: usize = 8;
        const PAIRS_PER_WORKER: usize = 10_000;
        for i in 0..384 {
            give_back(Vec::with_capacity((16 + i % 64) << 10));
        }
        let barrier = std::sync::Barrier::new(WORKERS + 1);
        let elapsed = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..WORKERS)
                .map(|worker| {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        for i in 0..PAIRS_PER_WORKER {
                            let size = (16 + (i * 31 + worker * 7) % 64) << 10;
                            let mut buffer = pooled(size);
                            buffer.push(0);
                            give_back(buffer);
                        }
                    })
                })
                .collect();
            let start = std::time::Instant::now();
            barrier.wait();
            for worker in workers {
                worker.join().unwrap();
            }
            start.elapsed()
        });
        eprintln!("snapshot pool: {} pairs, {:.3} ms", WORKERS * PAIRS_PER_WORKER, elapsed.as_secs_f64() * 1000.0);
    }

    /// The pool after snapshots stop, in a process's private bytes: before,
    /// 512 MiB stayed however long nothing asked. Run alone (the pool is
    /// global): `cargo test -p victory-gpu --lib
    /// shadow::pool_tests::pool_memory_load -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore]
    fn pool_memory_load() {
        fn private_kib() -> usize {
            use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            // SAFETY: plain data the call fills in, for this process.
            let mut c: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
            c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut c).cast(), c.cb) };
            c.PrivateUsage >> 10
        }
        let start = private_kib();
        for i in 0..256usize {
            let mut v = pooled((1 + i % 32) << 20);
            v.resize(v.capacity(), 1);
            give_back(v);
        }
        let kept = POOL.lock().unwrap().bytes;
        let full = private_kib();
        let gone = POOL.lock().unwrap().keep_smallest(POOL_PRESSURED_KEPT);
        drop(gone);
        let pressed = private_kib();
        let gone = POOL.lock().unwrap().keep_smallest(0);
        drop(gone);
        let idle = private_kib();
        eprintln!(
            "pool kept {} KiB; private KiB over the start: full {}, pressed {}, idle {}",
            kept >> 10,
            full.saturating_sub(start),
            pressed.saturating_sub(start),
            idle.saturating_sub(start)
        );
        assert!(pressed.saturating_sub(start) <= (POOL_PRESSURED_KEPT >> 10) + 8 * 1024);
        assert!(POOL.lock().unwrap().buffers.is_empty());
    }
}

const NOT_WATCHED: u8 = 2;
/// Watched, not by the kernel but by the guards, when last asked about.
const GUARDED: u8 = 3;

// Fixed-size equality compiles to two SIMD comparisons on Windows x64.
// Only the first or last unequal chunk needs a byte-wise search.
const DIFF_CHUNK: usize = 32;

/// Where `a` and `b` first differ within their common length.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    let (offset, a, b) = match a_chunks.iter().zip(b_chunks).position(|(x, y)| x != y) {
        Some(i) => (i * DIFF_CHUNK, &a_chunks[i][..], &b_chunks[i][..]),
        None => (len - a_tail.len(), a_tail, b_tail),
    };
    a.iter().zip(b).position(|(x, y)| x != y).map(|i| offset + i)
}

/// Where `a` and `b` last differ within their common length.
fn last_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    if let Some(i) = a_tail.iter().zip(b_tail).rposition(|(x, y)| x != y) {
        return Some(len - a_tail.len() + i);
    }
    let chunk = a_chunks.iter().zip(b_chunks).rposition(|(x, y)| x != y)?;
    a_chunks[chunk]
        .iter()
        .zip(&b_chunks[chunk])
        .rposition(|(x, y)| x != y)
        .map(|i| chunk * DIFF_CHUNK + i)
}

#[cfg(test)]
mod difference_tests {
    use super::{first_difference, last_difference};

    /// The chunk searches agree with a byte-wise scan across chunk boundaries.
    #[test]
    fn differences_are_found_as_byte_by_byte() {
        let base: Vec<u8> = (0..98u32).map(|i| (i * 7 + 3) as u8).collect();
        for len in [0, 1, 31, 32, 33, 63, 64, 65, 97] {
            let a = &base[1..1 + len];
            let same = a.to_vec();
            assert_eq!(first_difference(&same, a), None);
            assert_eq!(last_difference(&same, a), None);
            for at in 0..len {
                for other in at..len {
                    let mut b = a.to_vec();
                    b[at] ^= 0x80;
                    b[other] ^= 0x01;
                    let first = a.iter().zip(&b).position(|(x, y)| x != y);
                    let last = a.iter().zip(&b).rposition(|(x, y)| x != y);
                    assert_eq!(first_difference(a, &b), first, "{len} {at} {other}");
                    assert_eq!(last_difference(a, &b), last, "{len} {at} {other}");
                }
            }
        }
    }

    #[test]
    fn differences_ignore_bytes_past_the_shorter_slice() {
        let a = [0; 65];
        let mut b = [0; 67];
        b[65] = 1;
        assert_eq!(first_difference(&a, &b), None);
        assert_eq!(last_difference(&b, &a), None);
        b[32] = 1;
        b[64] = 1;
        assert_eq!(first_difference(&b, &a), Some(32));
        assert_eq!(last_difference(&a, &b), Some(64));
    }
}

/// The unit shader writes are looked for and put back in.
const PIECE: usize = 64 << 10;

/// Calls `f` with `[from, to)` cut at multiples of [`PIECE`].
fn for_pieces_of(from: usize, to: usize, mut f: impl FnMut(usize, usize)) {
    let mut at = from;
    while at < to {
        let next = ((at / PIECE + 1) * PIECE).min(to);
        f(at, next);
        at = next;
    }
}

/// Copies `len` bytes, on several threads if they are many: a snapshot of
/// the world's big heap is 100-170 MB per submission, 2-4 GB a frame, and
/// one thread copies some 8 GB/s.
///
/// # Safety
/// Both ranges are valid for `len` bytes and do not overlap.
unsafe fn copy_wide(src: *const u8, dst: *mut u8, len: usize) {
    const PER_THREAD: usize = 16 << 20;
    let threads = (len / PER_THREAD).clamp(1, 6);
    if threads == 1 {
        // SAFETY: the caller's.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
        return;
    }
    let (src, dst, each) = (src as usize, dst as usize, len.div_ceil(threads));
    std::thread::scope(|s| {
        for i in 0..threads {
            let at = (i * each).min(len);
            let n = each.min(len - at);
            s.spawn(move || {
                // SAFETY: a part of the caller's ranges.
                unsafe { std::ptr::copy_nonoverlapping((src + at) as *const u8, (dst + at) as *mut u8, n) };
            });
        }
    });
}

/// Whether the snapshots of a submission's shader-written ranges, and the
/// compare that finds what the shaders wrote, run on several threads
/// across ranges (`VICTORY_WIDE_BACK=1`; off by default).
///
/// Both ran range by range on the submitting thread, a range on several
/// threads only if it was 16 MiB or more. The ranges left on the CPU's side
/// since the big buffers went to the card are small and many: in run 86,
/// 7,024 declared ranges a frame, 375 MB of them compared after the card
/// ran in 563 pieces of work, none of 16 MiB: 51 ms a frame of snapshots
/// and 35 ms of comparing, one thread, bound by one core's memory speed.
///
/// With the switch a submission's snapshots are noted as it is prepared
/// and taken together when its refreshes are done, just before it goes to
/// the card ([`take_snapshots`]), and the compares of everything declared
/// run together after the card ([`find_all`]), both cut into pieces of
/// work of about 1 MiB for a few threads that are kept ([`wide`]). The
/// bytes taken and the bytes compared are the same ones: a range's bytes
/// cannot change between the moment it is declared and the end of the
/// preparing (a declared range is left alone by every refresh, the title's
/// own copies write title memory and not the copy, and the card runs the
/// lists only afterwards).
pub(crate) fn wide_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default; `VICTORY_WIDE_BACK=0` turns it off.
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_BACK").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

/// Threads kept for [`wide`]: what they are to do, and the word that there
/// is something.
struct Workers {
    jobs: Mutex<std::collections::VecDeque<Job>>,
    more: std::sync::Condvar,
    /// How many jobs there are, for [`spin`].
    there: std::sync::atomic::AtomicUsize,
}

/// Waits awake until `there` says so, `VICTORY_WIDE_SPIN` microseconds at
/// most (0, the default: not at all): asleep, a thread is woken in 50 us and
/// more, 95 times a frame. Whether `there` said so.
///
/// `VICTORY_WIDE_SPIN=1`: 200 microseconds; `alt`: 200 and none in turn,
/// twenty seconds each, so that one run measures both (and `VICTORY_AB` can
/// name it). The submitting thread's refreshes were 9.8 ms a frame in 40
/// flushes, 245 us a flush, with nine in ten of a flush's copies known to
/// copy nothing before a thread is woken (session jon-fox-1126-0152, ticket
/// 064): what a flush waits for at its end is counted for the log
/// ([`wide_waits`]).
fn spin(there: impl Fn() -> bool) -> bool {
    spin_for(WIDE_SPIN.now(), there)
}

static WIDE_SPIN: crate::switch::MicrosSwitch = crate::switch::MicrosSwitch::new("VICTORY_WIDE_SPIN", 200);

/// [`spin`], for `most` microseconds at the longest.
fn spin_for(most: u64, there: impl Fn() -> bool) -> bool {
    if most == 0 {
        return false;
    }
    let began = std::time::Instant::now();
    loop {
        if there() {
            return true;
        }
        if began.elapsed().as_micros() as u64 >= most {
            return false;
        }
        for _ in 0..16 {
            std::hint::spin_loop();
        }
    }
}

/// Since the last line, of the threads that spread work ([`wide`]): times
/// they did, nanoseconds they then waited for the other threads' pieces, and
/// how many of those waits went to sleep.
static WIDE_WAITS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// [`WIDE_WAITS`], taken, for the log: per frame.
#[cfg(feature = "diagnostics")]
pub(crate) fn wide_waits(frames: f64) -> String {
    let [times, ns, asleep] = [0, 1, 2].map(|i| WIDE_WAITS[i].swap(0, Ordering::Relaxed) as f64 / frames);
    format!(
        "{times:.0} times, {:.2} ms waiting at the end for the other threads' pieces, {asleep:.0} of those waits asleep \
         (VICTORY_WIDE_SPIN: {} us now, 0 is off)",
        ns / 1e6,
        WIDE_SPIN.now()
    )
}

/// How many threads are kept for [`wide`].
fn wide_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get() / 4).clamp(2, 6)
}

fn workers() -> &'static Workers {
    static WORKERS: std::sync::OnceLock<Workers> = std::sync::OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let workers = WORKERS.get_or_init(|| Workers {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: std::sync::Condvar::new(),
        there: std::sync::atomic::AtomicUsize::new(0),
    });
    STARTED.call_once(|| {
        for i in 0..wide_threads() {
            let _ = std::thread::Builder::new().name(format!("victory-wide-{i}")).spawn(move || {
                crate::cores::keep_helper_on_fast_cores("victory-wide");
                let workers = WORKERS.get().expect("made above");
                loop {
                    let job = {
                        // A little while awake first: the next work comes
                        // within microseconds more often than not.
                        spin(|| workers.there.load(Ordering::Acquire) > 0);
                        let mut jobs = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if let Some(job) = jobs.pop_front() {
                                workers.there.fetch_sub(1, Ordering::AcqRel);
                                break job;
                            }
                            jobs = workers.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    let _working = crate::timeline::span("a job for another thread");
                    job();
                }
            });
        }
    });
    workers
}

/// Whether [`wide`] wakes only as many kept threads as there are jobs for
/// them (`VICTORY_WAKE_FEW=1`; off by default: all of them, as before).
/// Ported from foxhound 8c1fa5d5 (readmet3xt-lab), where it is on with no
/// switch.
fn wake_few() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAKE_FEW").is_some_and(|v| v == "1"))
}

/// The kept threads [`wide`] wakes for `jobs` jobs of `threads`: with
/// `few` one for each job beyond the caller's first, else all.
fn woken(jobs: usize, threads: usize, few: bool) -> usize {
    if few { jobs.saturating_sub(1).min(threads) } else { threads }
}

/// Runs `jobs` on the kept threads and on this one, and returns when all
/// of them are done. The jobs copy and compare memory; they do not panic.
fn wide(jobs: Vec<Job>) {
    if jobs.len() < 2 {
        for job in jobs {
            job();
        }
        return;
    }
    let workers = workers();
    let _wide = crate::timeline::span("work spread over threads");
    // Helpers woken ([`wake_few`]): one for each job beyond the one this
    // thread takes first, at most all of them. All were woken for every call
    // (4 for 2 jobs), and those that found nothing went back to sleep; a
    // helper busy meanwhile looks at the queue before it sleeps again, and
    // this thread takes what nobody took.
    let wake = woken(jobs.len(), wide_threads(), wake_few());
    let left = Arc::new((Mutex::new(jobs.len()), std::sync::Condvar::new()));
    {
        let mut queue = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for job in jobs {
            let left = left.clone();
            workers.there.fetch_add(1, Ordering::AcqRel);
            queue.push_back(Box::new(move || {
                job();
                let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
                *n -= 1;
                if *n == 0 {
                    left.1.notify_all();
                }
            }));
        }
    }
    if wake >= wide_threads() {
        workers.more.notify_all();
    } else {
        (0..wake).for_each(|_| workers.more.notify_one());
    }
    loop {
        let job = workers.jobs.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
        match job {
            Some(job) => {
                workers.there.fetch_sub(1, Ordering::AcqRel);
                job()
            }
            None => break,
        }
    }
    let waiting = std::time::Instant::now();
    spin(|| left.0.try_lock().is_ok_and(|n| *n == 0));
    let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
    let mut asleep = false;
    while *n > 0 {
        asleep = true;
        n = left.1.wait(n).unwrap_or_else(|e| e.into_inner());
    }
    drop(n);
    WIDE_WAITS[0].fetch_add(1, Ordering::Relaxed);
    WIDE_WAITS[1].fetch_add(waiting.elapsed().as_nanos() as u64, Ordering::Relaxed);
    WIDE_WAITS[2].fetch_add(u64::from(asleep), Ordering::Relaxed);
}

/// [`wide`], for the host's own work on host copies (a submission's
/// refreshes, the merge of what shaders wrote to card copies).
pub(crate) fn wide_run(jobs: Vec<Box<dyn FnOnce() + Send>>) {
    wide(jobs);
}

/// Calls `work` with every number under `n`, on the kept threads and on
/// this one, the next number to whichever is free: for thousands of small
/// pieces of work, where a job each ([`wide`]) is mostly the jobs' queue.
pub(crate) fn wide_each(n: usize, work: impl Fn(usize) + Send + Sync + 'static) {
    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let work = Arc::new(work);
    let threads = (wide_threads() + 1).min(n);
    let jobs = (0..threads)
        .map(|_| {
            let (next, work) = (next.clone(), work.clone());
            Box::new(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    work(i);
                }
            }) as Job
        })
        .collect();
    wide(jobs);
}

/// About this many bytes to a piece of work for [`wide`].
const WIDE_JOB: usize = 1 << 20;

/// With [`wide_back`]: the snapshots noted while a submission is prepared
/// (from, to, bytes), taken by [`take_snapshots`].
static SNAPS: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());

/// With [`wide_back`], when a submission's refreshes are done and before
/// its lists go to the card: takes the snapshots [`Shadow::writing`] noted.
/// Nothing to do without the switch.
pub(crate) fn take_snapshots() {
    let snaps = std::mem::take(&mut *SNAPS.lock().unwrap_or_else(|e| e.into_inner()));
    if snaps.is_empty() {
        return;
    }
    wide(copy_jobs(snaps));
}

/// `copies` (from, to, bytes) as pieces of work of about [`WIDE_JOB`]
/// bytes each: long ones cut, short ones together.
fn copy_jobs(copies: Vec<(usize, usize, usize)>) -> Vec<Job> {
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<(usize, usize, usize)> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<(usize, usize, usize)>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if !group.is_empty() {
            jobs.push(Box::new(move || {
                for (src, dst, len) in group {
                    // SAFETY: the caller's ranges: a host copy's mapping and
                    // its snapshot, which do not overlap.
                    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
                }
            }));
        }
    };
    for (src, dst, len) in copies {
        let mut at = 0;
        while at < len {
            let n = (WIDE_JOB - bytes).min(len - at);
            group.push((src + at, dst + at, n));
            at += n;
            bytes += n;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    jobs
}

/// With [`wide_back`]: for each of `work` (a copy not in the card's memory
/// and a range `[address, end)` its shaders may have written, declared and
/// not yet done), the 64 KiB pieces that differ from the snapshot, as
/// [`Shadow::copy_back_with`] takes them; found on several threads.
pub(crate) fn find_all(work: &[(Arc<Shadow>, usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let scans = work
        .iter()
        .map(|(shadow, address, end)| {
            let (from, to) = shadow.inside(*address, end.saturating_sub(*address));
            // The snapshot's memory stays until the range is done with
            // (`copy_back_done`), which comes after this.
            match shadow.before.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                Some(before) => (shadow.data as usize, before.as_ptr() as usize, shadow.start as usize, from, to),
                None => (0, 0, 0, 0, 0),
            }
        })
        .collect();
    find_pieces(scans)
}

/// For each of `scans` (where a copy's bytes are, where its snapshot's
/// are, the title address the copy starts at, offsets from and to): the
/// 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
/// [`Shadow::differing`] cuts them) of `[from, to)` whose bytes differ, as
/// (offset, bytes) in order. On several threads ([`wide`]), in pieces of
/// work of about [`WIDE_JOB`] bytes that hold whole pieces.
fn find_pieces(scans: Vec<(usize, usize, usize, usize, usize)>) -> Vec<Vec<(usize, usize)>> {
    // What a job finds: (which scan, its place among that scan's jobs, the
    // pieces).
    type Found = Vec<(usize, usize, Vec<(usize, usize)>)>;
    // A job's part of a scan: the scan, then which and place.
    type Part = (usize, usize, usize, usize, usize, usize, usize);
    let found: Arc<Mutex<Found>> = Arc::default();
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<Part> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<Part>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if group.is_empty() {
            return;
        }
        let found = found.clone();
        jobs.push(Box::new(move || {
            let mut mine: Found = Vec::with_capacity(group.len());
            for (data, before, start, from, to, which, place) in group {
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the copy's mapping and its snapshot
                    // (which holds the copy's bytes at the copy's offsets);
                    // nothing writes either while a submission's results
                    // are compared.
                    let (now, old) = unsafe {
                        (
                            std::slice::from_raw_parts((data + at) as *const u8, n),
                            std::slice::from_raw_parts((before + at) as *const u8, n),
                        )
                    };
                    if now != old {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                mine.push((which, place, pieces));
            }
            found.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
        }));
    };
    let count = scans.len();
    for (which, (data, before, start, from, to)) in scans.into_iter().enumerate() {
        let (mut at, mut place) = (from, 0);
        while at < to {
            // Cut where pieces end, so that every job sees whole pieces.
            let room = (WIDE_JOB - bytes).max(PIECE);
            let mut next = (at + room).min(to);
            if next < to {
                next -= (start + next) % PIECE;
                if next <= at {
                    next = (at + PIECE - (start + at) % PIECE).min(to);
                }
            }
            group.push((data, before, start, at, next, which, place));
            bytes += next - at;
            place += 1;
            at = next;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    wide(jobs);
    let mut found = std::mem::take(&mut *found.lock().unwrap_or_else(|e| e.into_inner()));
    found.sort_unstable_by_key(|f| (f.0, f.1));
    let mut out = vec![Vec::new(); count];
    for (which, _, pieces) in found {
        out[which].extend(pieces);
    }
    out
}

/// Since the last frame-rate line, of the shader-written ranges looked for
/// on the CPU (copies not in the card's memory): bytes snapshotted in
/// windows outside buffers and in title buffers' copies; bytes compared in
/// windows, in buffers' copies, and of those in copies of 1 MiB or more;
/// ranges compared in windows and in buffers' copies.
static BACK_CLASS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// For the frame-rate line: where the CPU's search for shader writes goes
/// (per frame). A copy of a title buffer can go to the card's memory,
/// where the card does the search (`VICTORY_CARD_BUFFERS_MIN_MB`); a
/// window outside buffers cannot as the code stands.
pub(crate) fn back_line(frames: f64) {
    let c: Vec<f64> = BACK_CLASS.iter().map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if c.iter().all(|&v| v == 0.0) {
        return;
    }
    let mb = |v: f64| v / (1 << 20) as f64;
    tracing::info!(
        "shader writes looked for on the CPU per frame: {:.1} MB compared in {:.0} ranges of windows outside buffers ({:.1} MB \
         snapshotted), {:.1} MB in {:.0} ranges of title buffers' copies ({:.1} MB snapshotted), of them {:.1} MB in copies \
         of 1 MiB or more (VICTORY_CARD_BUFFERS_MIN_MB=1 would have the card look there)",
        mb(c[2]),
        c[5],
        mb(c[0]),
        mb(c[3]),
        c[6],
        mb(c[1]),
        mb(c[4]),
    );
}

#[cfg(test)]
mod wide_tests {
    use super::{Job, PIECE, WIDE_JOB, copy_jobs, find_pieces, wide, woken};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// `VICTORY_WAKE_FEW`: one kept thread per job beyond the caller's
    /// first, at most all; off, all of them whatever the jobs.
    #[test]
    fn few_jobs_wake_few_threads_only_with_the_switch() {
        assert_eq!(woken(2, 4, true), 1);
        assert_eq!(woken(4, 4, true), 3);
        assert_eq!(woken(9, 4, true), 4);
        assert_eq!(woken(0, 4, true), 0);
        assert_eq!(woken(2, 4, false), 4);
        assert_eq!(woken(9, 4, false), 4);
    }

    /// Every job runs once, however many there are, and `wide` returns
    /// only when all have.
    #[test]
    fn all_jobs_run_before_wide_returns() {
        for n in [0usize, 1, 2, 7, 300] {
            let ran = Arc::new(AtomicUsize::new(0));
            let jobs: Vec<Job> = (0..n)
                .map(|i| {
                    let ran = ran.clone();
                    Box::new(move || {
                        ran.fetch_add(i + 1, Ordering::Relaxed);
                    }) as Job
                })
                .collect();
            wide(jobs);
            assert_eq!(ran.load(Ordering::Relaxed), n * (n + 1) / 2, "{n} jobs");
        }
    }

    /// Waiting awake ends when what is waited for is there, or when the
    /// time is over; with no time given nothing is looked at.
    #[test]
    fn waiting_awake_ends_when_met_or_when_time_is_over() {
        use super::{WIDE_SPIN, WIDE_WAITS, spin_for};
        let looks = AtomicUsize::new(0);
        assert!(!spin_for(0, || looks.fetch_add(1, Ordering::Relaxed) > 0));
        assert_eq!(looks.load(Ordering::Relaxed), 0);
        // Met at the third look, long before the minute is over.
        assert!(spin_for(60_000_000, || looks.fetch_add(1, Ordering::Relaxed) >= 2));
        assert_eq!(looks.load(Ordering::Relaxed), 3);
        let began = std::time::Instant::now();
        assert!(!spin_for(2_000, || false));
        assert!(began.elapsed() >= std::time::Duration::from_millis(2));
        // Off unless asked for; the waits at the end of spread work are counted.
        if std::env::var_os("VICTORY_WIDE_SPIN").is_none() && std::env::var_os("VICTORY_AB").is_none() {
            assert_eq!(WIDE_SPIN.now(), 0);
        }
        let before = WIDE_WAITS[0].load(Ordering::Relaxed);
        wide((0..4).map(|_| Box::new(|| {}) as Job).collect());
        assert!(WIDE_WAITS[0].load(Ordering::Relaxed) > before);
    }

    /// Copies cut into jobs copy every byte once: long ones, short ones
    /// and empty ones.
    #[test]
    fn copies_cut_into_jobs_copy_the_same_bytes() {
        let len = 3 * WIDE_JOB + 12345;
        let src: Vec<u8> = (0..len).map(|i| (i * 31 + i / 977) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr() as usize, dst.as_mut_ptr() as usize);
        // Ranges of the buffer, each to the same offsets of the other.
        let cuts = [0, 1, 1, 4096, 70_000, WIDE_JOB + 3, 2 * WIDE_JOB + 3, 2 * WIDE_JOB + 9, len];
        let copies: Vec<(usize, usize, usize)> = cuts.windows(2).map(|w| (s + w[0], d + w[0], w[1] - w[0])).collect();
        let jobs = copy_jobs(copies);
        assert!(jobs.len() >= 3);
        wide(jobs);
        assert!(src == dst);
    }

    /// The pieces found on several threads are the ones a walk piece by
    /// piece finds, for copies that start anywhere in a 64 KiB piece of
    /// title memory and ranges that start and end anywhere.
    #[test]
    fn pieces_found_wide_are_those_found_one_by_one() {
        let len = 5 * WIDE_JOB / 2 + 777;
        let old: Vec<u8> = (0..len).map(|i| (i * 7 + i / 513) as u8).collect();
        let mut now = old.clone();
        // Changes: single bytes at piece edges, a run across two pieces,
        // the last byte, and nothing in between.
        for at in [0usize, 1, PIECE - 1, PIECE, 3 * PIECE + 5, WIDE_JOB - 1, WIDE_JOB, WIDE_JOB + PIECE / 2, 2 * WIDE_JOB + 9, len - 1] {
            now[at] ^= 0x5a;
        }
        for byte in &mut now[9 * PIECE - 40..9 * PIECE + 40] {
            *byte ^= 0x01;
        }
        let (data, before) = (now.as_ptr() as usize, old.as_ptr() as usize);
        let mut scans = Vec::new();
        let mut wanted = Vec::new();
        for start in [0usize, 4096, PIECE - 16, 0xd_b790_df00] {
            for (from, to) in [(0, len), (0, 1), (1, 2), (PIECE - 1, PIECE + 1), (3 * PIECE, 3 * PIECE), (12345, len - 1), (WIDE_JOB - 3, 2 * WIDE_JOB + 10)] {
                scans.push((data, before, start, from, to));
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    if now[at..at + n] != old[at..at + n] {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                wanted.push(pieces);
            }
        }
        // A scan of nothing (a range whose snapshot is gone).
        scans.push((0, 0, 0, 0, 0));
        wanted.push(Vec::new());
        assert_eq!(find_pieces(scans), wanted);
    }
}

/// What [`Shadow::copy_back`] does with a piece that differs from the
/// snapshot: `host` is the copy's bytes there, `old` the snapshot's. From
/// the first byte that differs to the last, title memory is read into
/// `title` (scratch, sized to that stretch) by `read` (the stretch's offset
/// in the piece, where to), the GPU's bytes are put in and `host` takes the
/// result. The stretch's offset in the piece and its length if that changed
/// title memory; none if nothing differs or title memory is not there.
/// `runs` takes the runs of the stretch (offsets into `title`, lengths)
/// whose bytes are the GPU's and differ from the title's as read: only
/// those are to be written, so a title write that lands between the read
/// and the write elsewhere in the stretch stays (upstream VIctory #2009).
/// `mask` is scratch.
fn back_piece(
    host: &mut [u8],
    old: &[u8],
    title: &mut Vec<u8>,
    mask: &mut Vec<u8>,
    runs: &mut Vec<(usize, usize)>,
    read: impl FnOnce(usize, &mut [u8]) -> bool,
) -> Option<(usize, usize)> {
    runs.clear();
    // From the first byte that differs to the last: title memory is read
    // and written there only (a shader that sets a few words of a piece
    // cost a read, a walk and a write of all 64 KiB of it).
    let first = first_difference(host, old)?;
    let last = last_difference(host, old).unwrap_or(first);
    let n = last + 1 - first;
    let (host, old) = (&mut host[first..=last], &old[first..=last]);
    title.clear();
    title.resize(n, 0);
    if !read(first, title) {
        return None;
    }
    // Title memory with the GPU's bytes put in; the copy takes the title's
    // bytes in turn.
    // Without a branch, so the loop runs many bytes at a time: the GPU's
    // byte where the copy's byte is not what it was before the submission
    // and the title left its byte as it was, else the title's. (With the
    // two searches above this was 44 ms a frame for 970 pieces.)
    // Where the shader and the title both changed a byte while the GPU ran,
    // the title's stays: the shader computed its byte from the copy's older
    // data (one cleared a bit of a record the title had already replaced,
    // and copying it back broke the title's object: an access violation
    // through 0xffffffff00000000; PalmVIDev e0b70476, VIctory f39c6c2).
    mask.clear();
    mask.resize(n, 0);
    let mut differs = 0u8;
    for (((t, h), o), m) in title.iter_mut().zip(host.iter_mut()).zip(old).zip(mask.iter_mut()) {
        let gpu = if (*h != *o) & (*t == *o) { 0xffu8 } else { 0 };
        let merged = (*t & !gpu) | (*h & gpu);
        *m = merged ^ *t;
        differs |= *m;
        *t = merged;
        *h = merged;
    }
    if differs == 0 {
        return None;
    }
    let mut i = 0;
    while i < n {
        while i < n && mask[i] == 0 {
            i += 1;
        }
        let start = i;
        while i < n && mask[i] != 0 {
            i += 1;
        }
        if i > start {
            runs.push((start, i - start));
        }
    }
    Some((first, n))
}

/// Since the last frame-rate line, of [`Shadow::merge_under_writes`]: pages
/// compared, changed runs put in, and their bytes.
static MERGED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Why [`Shadow::merge_under_writes`] last left the caller to wait.
static STALE_WHY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn note_stale(why: usize) {
    STALE_WHY.store(why, Ordering::Relaxed);
}

/// [`STALE_WHY`] in words (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn stale_why() -> &'static str {
    const WHY: [&str; 4] = [
        "the memory is not watched",
        "the journal does not say what was written",
        "more changed runs than a submission sends up",
        "no snapshot",
    ];
    WHY[STALE_WHY.load(Ordering::Relaxed).min(3)]
}

/// Since the last frame-rate line, where [`Shadow::merge_under_writes`]'s
/// time went, in ns: 0 asking which pages were written, 1 finding the
/// stale ones, 2 waiting for the locks, 3 the pages compared (the wall
/// time), 4 noting the runs; summed over threads: 5 reading title memory,
/// 6 comparing and putting in. Then 7 (unused), 8 stale pages, 9 calls.
pub(crate) static UNDER_NS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

fn under_ns(i: usize, since: std::time::Instant) -> std::time::Instant {
    let now = std::time::Instant::now();
    UNDER_NS[i].fetch_add((now - since).as_nanos() as u64, Ordering::Relaxed);
    now
}

/// [`UNDER_NS`], taken.
pub(crate) fn under_stats() -> [u64; 10] {
    std::array::from_fn(|i| UNDER_NS[i].swap(0, Ordering::Relaxed))
}

/// [`MERGED`], taken.
pub(crate) fn merged_stats() -> [u64; 3] {
    [0, 1, 2].map(|i| MERGED[i].swap(0, Ordering::Relaxed))
}

/// Adds the time since it was made to [`UNDER_NS`]`[6]` when dropped.
struct Compared(std::time::Instant);

impl Drop for Compared {
    fn drop(&mut self) {
        under_ns(6, self.0);
    }
}

/// The runs of bytes where `now` differs from `was` (of one length), as
/// (offset, bytes) in order: every differing byte is in a run and no equal
/// byte is.
fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        // Equal bytes, many at a time while they last (a page looked at
        // under a flight's writes hardly ever differs at all).
        while i + 512 <= n && now[i..i + 512] == was[i..i + 512] {
            i += 512;
        }
        while i + 8 <= n && now[i..i + 8] == was[i..i + 8] {
            i += 8;
        }
        while i < n && now[i] == was[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let first = i;
        while i < n && now[i] != was[i] {
            i += 1;
        }
        runs.push((first, i - first));
    }
    runs
}

/// The kernel's page.
const KERNEL_PAGE: usize = 4096;

/// Whether a refresh looks if what counts as written is only what a landing
/// put there, and then sends nothing of it up (`VICTORY_LAND_ECHO=0`: it copies
/// and sends it, as before).
///
/// What shaders wrote comes down, goes into title memory and into the CPU's
/// side of the copy alike. In memory the kernel watches that is a write like
/// any other (it cannot be told as the guards' blocks can, `crate::land`), and
/// the next refresh of it copied the pages and sent them up again: in the dorm
/// 14.4 MB a frame of one copy of 90 MB whose 263 pieces were all fetched and
/// all sent, 28 of 72 MB over the bus in all (run submit-2). Only pages a
/// landing was noted in are compared, 4 KiB at a time: a page that is not the
/// same goes up whole, as before, and so does every other.
fn land_echo() -> bool {
    crate::toggled!("VICTORY_LAND_ECHO")
}

/// Since the last frame-rate line: bytes found to be a landing's alone, and
/// bytes a landing was noted in that were the title's too.
pub(crate) static ECHO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether a refresh of a copy in the card's memory takes, and sends up,
/// the dwords that title memory changed and no others
/// (`VICTORY_SEND_CHANGED=1`; off: the 64 KiB pages written since it last
/// took them, whole). Off since run tile1-0: the room lost its indirect light
/// and geometry exploded; run bis1-0, the same but for this, was right.
///
/// A page that counts as written is mostly not the title's doing, or not all
/// of it: a word the title wrote, and around it what shaders wrote and the
/// title never touched; or only what the relay put into title memory, which
/// the card and the CPU's side hold already ([`Shadow::card_merge_relay`]).
/// Sent up whole, such a page goes over whatever the card holds there that
/// did not come back yet, or never will (a write through a view nobody
/// declared): what stood on the card, newer, was gone.
///
/// What it rests on: the CPU's side is what the card's side was last known
/// to hold, for every range no shader may be writing (no refresh touches
/// the others). So a dword goes up only if title memory holds another value
/// of it than the CPU's side, which then holds it too; title memory is read
/// once, what is compared is what is stored and sent. Dwords, not bytes:
/// no shader writes less, and the title's bytes beside a changed one in its
/// dword are those the CPU's side holds.
///
/// A page never taken goes up whole as long as no shader's write was ever
/// declared for the copy. After that it is compared as any other: the CPU's
/// side and the card's began as zeros, both.
///
/// What it cannot know: that the title wrote a value over the same value.
/// Where a shader wrote in between, through a view nobody declared, the
/// shader's stays.
fn send_changed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        #[cfg(feature = "diagnostics")]
        if crate::copystat::mode() != 0 {
            return false;
        }
        std::env::var_os("VICTORY_SEND_CHANGED").is_some_and(|v| v != "0")
    })
}

/// Since the last frame-rate line, of pages taken into copies in the
/// card's memory ([`send_changed`]): bytes compared, those of pages that
/// held the same, bytes to go up, and the runs they are.
static SEND: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// [`SEND`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn send_stats() -> [u64; 4] {
    [0, 1, 2, 3].map(|i| SEND[i].swap(0, Ordering::Relaxed))
}

/// Whether what shaders wrote goes into title memory run by run
/// ([`put_changed`]; on by default, `VICTORY_EXACT_BACK=0` writes the whole
/// stretch from the first changed byte to the last, as before).
pub(crate) fn exact_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_EXACT_BACK").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line: pieces put back by [`put_changed`], the
/// writes they took, the bytes written, and the pieces whose runs were
/// joined over bytes that had not changed.
pub(crate) static PUT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The dwords (of title memory from `at` on) where `merged` differs from
/// `title`, as (offset, bytes) in order, joined where at most `gap` bytes
/// lie between.
fn dword_runs(at: usize, title: &[u8], merged: &[u8], gap: usize) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (from, n) in changed_runs(merged, title) {
        // Shaders write dwords: the whole of each.
        let lo = ((at + from) & !3).max(at) - at;
        let hi = ((at + from + n + 3) & !3).min(at + title.len()) - at;
        match runs.last_mut() {
            Some((a, len)) if lo <= *a + *len + gap => *len = (*len).max(hi - *a),
            _ => runs.push((lo, hi - lo)),
        }
    }
    runs
}

/// [`dword_runs`] with no gap, the fast way where it can be: a frame compares
/// 12 MB this way and finds 80,000 runs.
fn word_runs(at: usize, title: &[u8], held: &[u8]) -> Vec<(usize, usize)> {
    #[cfg(target_arch = "x86_64")]
    if at.is_multiple_of(4) && title.len().is_multiple_of(4) && title.len() == held.len() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2, and the lengths are the same.
        return unsafe { word_runs_avx2(title, held) };
    }
    dword_runs(at, title, held, 0)
}

/// # Safety
/// AVX2, and two slices of one length, whole words.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn word_runs_avx2(title: &[u8], held: &[u8]) -> Vec<(usize, usize)> {
    use std::arch::x86_64::*;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut push = |from: usize, len: usize| match runs.last_mut() {
        Some((a, n)) if *a + *n == from => *n += len,
        _ => runs.push((from, len)),
    };
    let n = title.len();
    let mut i = 0;
    while i + 256 <= n {
        let mut differ = 0u64;
        for k in 0..8 {
            // SAFETY: inside both slices (`i + 256 <= n`).
            let same = unsafe {
                _mm256_cmpeq_epi32(
                    _mm256_loadu_si256(title.as_ptr().add(i + 32 * k).cast()),
                    _mm256_loadu_si256(held.as_ptr().add(i + 32 * k).cast()),
                )
            };
            differ |= u64::from(!_mm256_movemask_ps(_mm256_castsi256_ps(same)) as u8) << (8 * k);
        }
        while differ != 0 {
            let first = differ.trailing_zeros();
            let count = (differ >> first).trailing_ones();
            push(i + 4 * first as usize, 4 * count as usize);
            if first + count >= 64 {
                break;
            }
            differ &= u64::MAX << (first + count);
        }
        i += 256;
    }
    while i + 4 <= n {
        if title[i..i + 4] != held[i..i + 4] {
            push(i, 4);
        }
        i += 4;
    }
    runs
}

/// Whether [`Shadow::merge_under_writes`] puts the title's bytes into the
/// CPU's side of a copy in the card's memory as it compares, a block of 32
/// bytes where one of its words differs (`VICTORY_UNDER_FUSED=0`: the runs
/// are found first and copied one by one afterwards, as before). The same
/// bytes end up there: the words of a block that do not differ are equal.
/// In session play-1125-1724 a frame had 190,000 runs in 400 pages, each
/// a copy of a few bytes of its own.
fn under_fused() -> bool {
    crate::toggled!("VICTORY_UNDER_FUSED")
}

/// [`word_runs`], and every 32-byte block of `held` in which a word differs
/// takes `title`'s bytes (the words of the tail that differ, one by one):
/// `held` is `title` afterwards.
fn word_runs_put(at: usize, title: &[u8], held: &mut [u8]) -> Vec<(usize, usize)> {
    #[cfg(target_arch = "x86_64")]
    if at.is_multiple_of(4) && title.len().is_multiple_of(4) && title.len() == held.len() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2, and the lengths are the same.
        return unsafe { word_runs_put_avx2(title, held) };
    }
    let runs = dword_runs(at, title, held, 0);
    for &(r, m) in &runs {
        held[r..r + m].copy_from_slice(&title[r..r + m]);
    }
    runs
}

/// # Safety
/// AVX2, and two slices of one length, whole words.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn word_runs_put_avx2(title: &[u8], held: &mut [u8]) -> Vec<(usize, usize)> {
    use std::arch::x86_64::*;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut push = |from: usize, len: usize| match runs.last_mut() {
        Some((a, n)) if *a + *n == from => *n += len,
        _ => runs.push((from, len)),
    };
    let n = title.len();
    let mut i = 0;
    while i + 256 <= n {
        let mut differ = 0u64;
        for k in 0..8 {
            let at = i + 32 * k;
            // SAFETY: inside both slices (`i + 256 <= n`).
            unsafe {
                let now = _mm256_loadu_si256(title.as_ptr().add(at).cast());
                let old = _mm256_loadu_si256(held.as_ptr().add(at).cast());
                let block = !_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(now, old))) as u8;
                if block != 0 {
                    _mm256_storeu_si256(held.as_mut_ptr().add(at).cast(), now);
                }
                differ |= u64::from(block) << (8 * k);
            }
        }
        while differ != 0 {
            let first = differ.trailing_zeros();
            let count = (differ >> first).trailing_ones();
            push(i + 4 * first as usize, 4 * count as usize);
            if first + count >= 64 {
                break;
            }
            differ &= u64::MAX << (first + count);
        }
        i += 256;
    }
    while i + 4 <= n {
        if title[i..i + 4] != held[i..i + 4] {
            held[i..i + 4].copy_from_slice(&title[i..i + 4]);
            push(i, 4);
        }
        i += 4;
    }
    runs
}

/// Whether the pages of [`Shadow::merge_under_writes`] are compared on several
/// threads (`VICTORY_UNDER_WIDE=0`: on the submitting one).
fn under_wide() -> bool {
    crate::toggled!("VICTORY_UNDER_WIDE")
}

/// What shaders wrote goes into title memory from `at` on: `title` is what
/// was read there, `merged` the same with their bytes put in. Only the
/// dwords that differ are written. The whole stretch from the first changed
/// byte to the last was: between the read and the write the title's own
/// threads, or data streamed in, may have written the bytes between (a heap
/// holds what shaders write beside what the CPU does), and those writes were
/// undone. Beside running title threads (the relay) that is a piece of
/// memory going back in time now and then: runs pool1, blame1 and seriess2
/// ended in the title decoding garbage.
///
/// shortcut: a piece with very many runs has them joined over small gaps,
/// where the old race is left.
fn put_changed(at: usize, title: &[u8], merged: &[u8]) -> usize {
    crate::watch::landing(at, title.len());
    if !exact_back() {
        crate::host::write_mapped(at as *mut u8, merged.as_ptr(), merged.len());
        return merged.len();
    }
    if crate::safecopy::on() {
        PUT[0].fetch_add(1, Ordering::Relaxed);
        PUT[2].fetch_add(merged.len() as u64, Ordering::Relaxed);
        let since = crate::watch::changes();
        // SAFETY: title memory; a page of it that is not there fails the
        // put, and the rest is put page by page.
        if unsafe { crate::safecopy::put_changed(at as *mut u8, title, merged) } {
            return merged.len();
        }
        let mut from = 0;
        while from < merged.len() {
            let n = (4096 - (at + from) % 4096).min(merged.len() - from);
            let (was, now) = (&title[from..from + n], &merged[from..from + n]);
            // SAFETY: as above.
            let put = || unsafe { crate::safecopy::put_changed((at + from) as *mut u8, was, now) };
            // A page that is away only while the runtime changes a mapping
            // beside it gets its bytes when that is done.
            if !put() && !crate::watch::again_after_change(since, put) {
                crate::watch::lost("what shaders wrote", at + from, n);
            }
            from += n;
        }
        return merged.len();
    }
    let mut runs = dword_runs(at, title, merged, 0);
    if runs.len() > 64 {
        runs = dword_runs(at, title, merged, 32);
        PUT[3].fetch_add(1, Ordering::Relaxed);
    }
    let mut written = 0;
    for &(from, n) in &runs {
        crate::host::write_mapped((at + from) as *mut u8, merged[from..].as_ptr(), n);
        written += n;
    }
    PUT[0].fetch_add(1, Ordering::Relaxed);
    PUT[1].fetch_add(runs.len() as u64, Ordering::Relaxed);
    PUT[2].fetch_add(written as u64, Ordering::Relaxed);
    written
}

/// For `crate::land`'s measurement: a piece as [`Shadow::card_merge_relay`]
/// lands it without `VICTORY_LAND`: `n` bytes at `now`, `held` what the CPU's
/// side held, both the caller's alone.
#[cfg(test)]
pub(crate) fn land_as_before(start: usize, now: usize, held: usize, n: usize, title: &mut Vec<u8>) {
    // SAFETY: the caller's.
    let (gpu, held) = unsafe { (std::slice::from_raw_parts(now as *const u8, n), std::slice::from_raw_parts_mut(held as *mut u8, n)) };
    let Some(first) = first_difference(gpu, held) else {
        return;
    };
    let last = last_difference(gpu, held).unwrap_or(first);
    let n = last + 1 - first;
    title.clear();
    title.resize(n, 0);
    assert!(crate::host::read_into((start + first) as *const u8, title));
    let was = title.clone();
    if merge_piece(&gpu[first..=last], &mut held[first..=last], title) {
        put_changed(start + first, &was, title);
    }
}

#[cfg(test)]
mod merge_tests {
    use super::{PAGE, PIECE, back_piece, changed_runs, dword_runs, put_changed, word_runs, word_runs_put};

    /// A piece put back takes the GPU's bytes where the copy changed and
    /// the title's everywhere else, reads and writes title memory only from
    /// the first change to the last, and its scratch holds that stretch and
    /// no more: not the 64 KiB per range it once was.
    #[test]
    fn a_piece_put_back_reads_only_its_stretch() {
        let old: Vec<u8> = (0..PIECE).map(|i| (i * 7 + i / 251) as u8).collect();
        // The title wrote 7 and 40,000 since the snapshot; the GPU wrote 5,
        // 9 and 40,000 (the GPU's byte wins where both wrote).
        let mut title_memory = old.clone();
        title_memory[7] ^= 0x11;
        title_memory[40_000] ^= 0x22;
        let mut host = old.clone();
        for at in [5, 9] {
            host[at] ^= 0x33;
        }
        let gpu = host.clone();
        let mut scratch = Vec::new();
        let (mut mask, mut runs) = (Vec::new(), Vec::new());
        let mut asked = None;
        let read = |first: usize, into: &mut [u8]| {
            asked = Some((first, into.len()));
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((5, 5)));
        assert_eq!(asked, Some((5, 5)));
        let mut wanted = title_memory[5..10].to_vec();
        wanted[0] = gpu[5];
        wanted[4] = gpu[9];
        assert_eq!(scratch, wanted);
        assert_eq!(host[5..10], wanted);
        // Only the GPU's two bytes are written back (upstream #2009).
        assert_eq!(runs, [(0, 1), (4, 1)]);
        // The copy outside the stretch is left as it was.
        assert!(host[..5] == gpu[..5] && host[10..] == gpu[10..]);
        assert!(scratch.capacity() < PIECE / 2, "scratch of {} bytes for a stretch of 5", scratch.capacity());

        // Nothing differs: title memory is not read.
        let mut same = old.clone();
        assert_eq!(back_piece(&mut same, &old, &mut scratch, &mut mask, &mut runs, |_, _| unreachable!()), None);
        // Title memory not there: nothing to write, the copy unchanged.
        let mut host = gpu.clone();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, _| false), None);
        assert!(host == gpu);
        // The GPU wrote what the title holds already: nothing to write.
        let mut title_memory = old.clone();
        title_memory[5] = gpu[5];
        title_memory[9] = gpu[9];
        let mut host = gpu.clone();
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), None);
        // A whole piece changed: the scratch grows to it.
        let mut host: Vec<u8> = old.iter().map(|b| !b).collect();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, into| into.len() == PIECE), Some((0, PIECE)));
        assert_eq!(scratch.len(), PIECE);
    }

    /// A byte the shader and the title both changed while the GPU ran keeps
    /// the title's (the shader computed it from the copy's older data); a
    /// byte only the shader changed goes back, and only such bytes are in
    /// the runs to write (PalmVIDev e0b70476, upstream #2009).
    #[test]
    fn a_byte_both_changed_keeps_the_titles_and_only_gpu_bytes_are_written() {
        let old = vec![1u8, 2, 3, 4, 5, 6];
        let mut host = old.clone();
        host[1] = 20; // the GPU alone
        host[3] = 40; // both
        host[4] = 50; // the GPU alone
        let mut title_memory = old.clone();
        title_memory[3] = 44;
        title_memory[5] = 66; // the title alone
        let (mut scratch, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((1, 4)));
        assert_eq!(scratch, [20, 3, 44, 50], "the title's byte where both changed");
        assert_eq!(host[1..5], [20, 3, 44, 50], "the copy takes the result");
        assert_eq!(runs, [(0, 1), (3, 1)], "the GPU's bytes only");
    }


    /// What shaders wrote reaches a page that is away only while the runtime
    /// changes a mapping; a page that stays away loses it, and nothing else.
    #[test]
    fn a_put_waits_for_a_change_of_mapping() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        // SAFETY: a reservation of ours, nothing of it committed.
        let at = unsafe { VirtualAlloc(std::ptr::null(), 1 << 16, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert_ne!(at, 0);
        put_changed(at, &[0; 64], &[3; 64]);
        let (began, wait) = std::sync::mpsc::channel();
        let change = std::thread::spawn(move || {
            let _changing = crate::watch::title_memory_remapping();
            began.send(()).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            // SAFETY: the reservation's first page.
            assert!(!unsafe { VirtualAlloc(at as _, 4096, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        });
        wait.recv().unwrap();
        put_changed(at, &[0; 64], &[1; 64]);
        change.join().unwrap();
        // SAFETY: committed by now.
        assert_eq!(unsafe { std::slice::from_raw_parts(at as *const u8, 64) }, [1; 64]);
        // SAFETY: our reservation.
        unsafe { VirtualFree(at as _, 0, MEM_RELEASE) };
    }

    /// Whole dwords around every changed byte and no other, whatever the
    /// alignment; small gaps joined only when asked.
    #[test]
    fn dword_runs_hold_the_dwords_that_changed() {
        let title: Vec<u8> = (0..200u32).map(|i| (i * 7 + 3) as u8).collect();
        for at in [0x1000usize, 0x1001, 0x1002, 0x1003] {
            for changes in [vec![0], vec![199], vec![5, 6], vec![8, 12, 40, 41, 43, 120]] {
                let mut merged = title.clone();
                for &c in &changes {
                    merged[c] ^= 0x55;
                }
                let runs = dword_runs(at, &title, &merged, 0);
                let held: Vec<usize> = runs.iter().flat_map(|&(a, n)| a..a + n).collect();
                let mut expect: Vec<usize> = (0..200).filter(|i| changes.iter().any(|c| (at + c) / 4 == (at + i) / 4)).collect();
                expect.dedup();
                assert_eq!(held, expect, "{at:#x} {changes:?}");
                let joined = dword_runs(at, &title, &merged, 32);
                assert!(joined.len() <= runs.len());
                assert!(runs.iter().all(|r| joined.iter().any(|j| j.0 <= r.0 && r.0 + r.1 <= j.0 + j.1)));
            }
        }
        assert!(dword_runs(0, &title, &title, 0).is_empty());
    }

    /// The runs hold exactly the bytes that differ: putting them into the
    /// old bytes gives the new ones, and no run holds an equal byte.
    /// What looking at a page that did not change costs, against the look
    /// eight bytes at a time that it was (a frame has some 250 of them, on
    /// the submitting thread). Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn runs_of_equal_pages() {
        let (a, b) = (vec![7u8; 64 << 10], vec![7u8; 64 << 10]);
        let eights = || {
            let (now, was) = (std::hint::black_box(&a), std::hint::black_box(&b));
            let mut i = 0;
            while i + 8 <= now.len() && now[i..i + 8] == was[i..i + 8] {
                i += 8;
            }
            std::hint::black_box(i);
        };
        let best = |f: &dyn Fn()| {
            (0..500)
                .map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e6
                })
                .fold(f64::MAX, f64::min)
        };
        let runs = || drop(std::hint::black_box(changed_runs(std::hint::black_box(&a), std::hint::black_box(&b))));
        println!("64 KiB that are the same: {:.1} us by eights, {:.1} us now", best(&eights), best(&runs));
    }

    #[test]
    fn word_runs_put_finds_the_word_runs_and_puts_the_title_in() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..300 {
            let n = [PAGE, 4096, 256, 260, 4, 0, 1000][round % 7];
            let held: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let mut title = held.clone();
            // Sparse words, dense stretches, single bytes of words.
            let changes = (next() % 600) as usize;
            for _ in 0..changes.min(n) {
                let at = (next() as usize) % n.max(1);
                let len = [1, 3, 4, 40, 300][(next() % 5) as usize].min(n - at);
                for b in &mut title[at..at + len] {
                    *b = b.wrapping_add(1 + (next() % 255) as u8);
                }
            }
            let want = word_runs(0, &title, &held);
            let mut merged = held.clone();
            let runs = word_runs_put(0, &title, &mut merged);
            assert_eq!(runs, want, "round {round}");
            assert_eq!(merged, title, "round {round}");
            // At most this many runs of more than 256 bytes (the page's
            // copies of their own, `merge_page_under_writes`).
            assert!(runs.iter().filter(|r| r.1 > 256).count() <= n / 260 + 1, "round {round}");
        }
    }

    #[test]
    fn word_runs_are_the_dword_runs() {
        let mut seed = 7u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        for round in 0..200 {
            let n = 4 * (random() as usize % 400);
            let was: Vec<u8> = (0..n).map(|_| random() as u8).collect();
            let mut now = was.clone();
            for _ in 0..random() % 200 {
                if n == 0 {
                    break;
                }
                let at = random() as usize % n;
                let len = (random() as usize % 24).min(n - at);
                for b in &mut now[at..at + len] {
                    *b = random() as u8;
                }
            }
            assert_eq!(word_runs(0x1000, &now, &was), dword_runs(0x1000, &now, &was, 0), "round {round}");
        }
    }

    #[test]
    fn changed_runs_are_exactly_the_bytes_that_differ() {
        let was: Vec<u8> = (0..300u32).map(|i| (i * 13 + 5) as u8).collect();
        for changes in [vec![], vec![0], vec![299], vec![7, 8, 9], vec![7, 9], vec![15, 16, 17, 23, 24, 64, 65, 66, 67, 68, 69, 70, 71, 72, 298, 299]] {
            let mut now = was.clone();
            for &at in &changes {
                now[at] ^= 0x41;
            }
            let runs = changed_runs(&now, &was);
            let mut put = was.clone();
            let mut held = Vec::new();
            for &(at, n) in &runs {
                assert!(n > 0);
                put[at..at + n].copy_from_slice(&now[at..at + n]);
                held.extend(at..at + n);
            }
            assert_eq!(put, now, "{changes:?}");
            assert_eq!(held, changes, "{changes:?}");
            // Runs do not touch: two that did would be one.
            assert!(runs.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0), "{changes:?}");
        }
        assert!(changed_runs(&[], &[]).is_empty());
    }
}

/// Host copies alive (also those only a list, a descriptor pass or a
/// snapshot still holds) and the bytes of title memory they copy.
static LIVE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Shadow {
    fn drop(&mut self) {
        crate::census::copy_gone(self.id, self.start, self.len);
        if self.uses.is_some() {
            crate::byuse::gone(self.id);
        }
        if self.place.is_some() {
            // Never counted with the copies (`Shadow::placed`).
            crate::inplace::gone(self.window, self.len);
            return;
        }
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(self.len, Ordering::Relaxed);
        let prints = self.prints.get_mut().unwrap_or_else(|e| e.into_inner()).len();
        if prints != 0 {
            crate::windowskip::prints_gone(prints);
        }
        if self.kept.get_mut().unwrap_or_else(|e| e.into_inner()).1 {
            KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
        let window = self.window;
        if self.mapped {
            WINDOW_MAPPED_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
        if let Some(card) = &mut self.card {
            if window {
                WINDOW_CARD_BYTES.fetch_sub(card.size, Ordering::Relaxed);
            } else {
                CARD_BYTES[0].fetch_sub(card.size, Ordering::Relaxed);
            }
            if card.before.get_mut().unwrap_or_else(|e| e.into_inner()).is_some() {
                CARD_BYTES[1].fetch_sub(card_hash_bytes(card.size), Ordering::Relaxed);
            }
        }
    }
}

/// Whether a copy's snapshot is kept from one submission to the next and
/// only the pieces that changed in between are taken again
/// (`VICTORY_KEEP_SNAPSHOTS=1`; off by default).
///
/// A snapshot is the copy's bytes before a submission, to find what its
/// shaders wrote. Taken anew for every submission it is 1,050 MB of copying
/// a frame (72-86 ms in gameplay, runs 61 and 62) for buffers of which
/// 100 MB changed. Kept, a piece of it is taken again only after something
/// changed the copy there: a refresh from title memory ([`Shadow::copy`]),
/// shader writes found after a submission, or the bytes put back then
/// ([`Shadow::copy_back`]).
///
/// What it assumes, and why it is not on by default: that nothing else
/// writes a copy. A shader that writes a buffer through a view we did not
/// declare as written (we declare the unordered-access views of shaders
/// that can store) changes the copy unseen; with a fresh snapshot that
/// write is simply never put back, with a kept one it would be taken for a
/// write of a later submission and put back then, over whatever the title
/// wrote there since.
fn keeps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_SNAPSHOTS").is_some_and(|v| v == "1"))
}

/// Bytes of the snapshots kept, and the most that may be.
static KEPT_BYTES: AtomicU64 = AtomicU64::new(0);
/// `VICTORY_KEEP_SNAPSHOTS_MB`, 256 unless set (it was 768: run 80, with
/// kept snapshots beside copies in the card's memory, ran the PC out of
/// memory in 27 s; kept snapshots are the PC's memory, all of it resident).
fn kept_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_KEEP_SNAPSHOTS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(256) << 20
    })
}

/// The 64 KiB pieces, cut at multiples of 64 KiB of title memory, that
/// hold offsets `[from, to)` of a copy of memory from `start` on
/// (`from < to`); piece 0 holds offset 0.
fn pieces_of(start: usize, from: usize, to: usize) -> std::ops::Range<usize> {
    let first = start / PIECE;
    (start + from) / PIECE - first..(start + to - 1) / PIECE - first + 1
}

/// The offsets that piece `piece` holds of a copy of `len` bytes of memory
/// from `start` on.
fn piece_at(start: usize, len: usize, piece: usize) -> (usize, usize) {
    let first = start / PIECE;
    let lo = ((first + piece) * PIECE).saturating_sub(start);
    let hi = ((first + piece + 1) * PIECE - start).min(len);
    (lo, hi)
}

#[cfg(test)]
mod piece_tests {
    use super::{PIECE, piece_at, pieces_of};

    /// The pieces cover a copy without gaps, each cut where the compare
    /// cuts (at multiples of 64 KiB of title memory).
    #[test]
    fn pieces_cover_the_copy_as_the_compare_cuts_it() {
        for start in [0usize, 1, 4096, PIECE - 1, PIECE, PIECE + 77, 5 * PIECE + 12345] {
            for len in [1usize, 100, PIECE - 1, PIECE, PIECE + 1, 3 * PIECE + 500] {
                let count = pieces_of(start, 0, len).end;
                let mut at = 0;
                for piece in 0..count {
                    let (lo, hi) = piece_at(start, len, piece);
                    assert_eq!(lo, at, "{start} {len} {piece}");
                    // As `Shadow::differing` cuts: to the next multiple.
                    let n = (PIECE - (start + at) % PIECE).min(len - at);
                    assert_eq!(hi, at + n, "{start} {len} {piece}");
                    assert_eq!(pieces_of(start, lo, hi), piece..piece + 1);
                    assert_eq!(pieces_of(start, hi - 1, hi), piece..piece + 1);
                    at = hi;
                }
                assert_eq!(at, len, "{start} {len}");
                assert_eq!(pieces_of(start, 0, len), 0..count);
            }
        }
    }
}

/// The pages of `[a, e)` (title addresses, `a < e`) that a copy whose
/// first page is `base` must take from title memory before it holds the
/// title's bytes: those it never took (`copied` 0), whether or not the
/// title wrote them since tracking began, and those `written` (as
/// `crate::watch::writes` gives it, by page in order) shows written after
/// the copy took them. As [`Shadow::copy_stale`] decides it.
///
/// [`Shadow::stale_under_writes`] and [`Shadow::merge_under_writes`] looked
/// at `written` alone, so a page of a range shaders may be writing that the
/// copy never took, and the title had not written lately, was neither
/// merged nor a reason to settle; the refresh leaves such ranges alone, and
/// shaders read zeros there (fxceless/victory-fork#206: missing geometry
/// with `VICTORY_LATE_WAIT`/`VICTORY_RELAY`).
fn stale_pages_in(base: usize, copied: &[u64], a: usize, e: usize, written: &[(usize, u64)]) -> Vec<usize> {
    (a / PAGE..e.div_ceil(PAGE))
        .filter(|&page| page >= base)
        .filter(|&page| match copied.get(page - base) {
            None => false,
            Some(0) => true,
            Some(&at) => written
                .binary_search_by_key(&page, |&(p, _)| p)
                .is_ok_and(|i| written[i].1 > at),
        })
        .collect()
}

/// `runs` (from, to, whether the copy held it before), cut in those of pages
/// held before (`pages`: page number and asking, in order) to the parts of
/// each page that were written since (`parts`: a bit for each 4 KiB).
fn by_parts(runs: Vec<(usize, usize, bool)>, pages: &[(usize, u64)], parts: &[u16]) -> Vec<(usize, usize, bool)> {
    const PART: usize = PAGE / 16;
    if parts.iter().all(|&p| p == u16::MAX) {
        return runs;
    }
    let mut out: Vec<(usize, usize, bool)> = Vec::with_capacity(runs.len() + pages.len());
    let mut pages = pages.iter().zip(parts).peekable();
    for (from, to, known) in runs {
        if !known {
            out.push((from, to, known));
            continue;
        }
        while let Some((&(page, _), &parts)) = pages.next_if(|((page, _), _)| page * PAGE < to) {
            for k in (0..16).filter(|k| parts >> k & 1 != 0) {
                let (a, b) = ((page * PAGE + k * PART).max(from), (page * PAGE + (k + 1) * PART).min(to));
                if a >= b {
                    continue;
                }
                match out.last_mut() {
                    Some((_, end, true)) if *end == a => *end = b,
                    _ => out.push((a, b, true)),
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod landing_tests {
    use super::{KERNEL_PAGE, Shadow};

    /// A page that only a landing wrote is not sent up again; one the title
    /// wrote too is, and so is every page no landing was noted in.
    #[test]
    fn what_a_landing_wrote_is_not_sent_up_again() {
        const PAGES: usize = 6;
        let room = vec![7u8; (PAGES + 1) * KERNEL_PAGE];
        let title = room.as_ptr() as usize;
        let title = title.next_multiple_of(KERNEL_PAGE);
        let mut side = vec![7u8; PAGES * KERNEL_PAGE];
        let copy = Shadow::unbacked(title as u64, side.as_mut_ptr(), side.len() as u64, true);
        let write = |page: usize| {
            // SAFETY: in `room`.
            unsafe { ((title + page * KERNEL_PAGE + 9) as *mut u8).write_volatile(8) }
        };
        let page = |from: usize, to: usize| (title + from * KERNEL_PAGE, title + to * KERNEL_PAGE);
        copy.note_landing(KERNEL_PAGE, 4 * KERNEL_PAGE);
        // Page 2: the title's too.
        write(2);
        let (from, to) = page(0, 6);
        copy.take_landed(from, to);
        let mut dirty = copy.take_dirty();
        dirty.sort_unstable();
        let part = |a: usize, b: usize| (a * KERNEL_PAGE, b * KERNEL_PAGE);
        assert_eq!(super::joined(dirty), [part(0, 1), part(2, 3), part(4, 6)]);
        assert_eq!(side[2 * KERNEL_PAGE + 9], 8);
        // Looked at: from here on as any other.
        let (from, to) = page(1, 2);
        copy.take_landed(from, to);
        assert_eq!(copy.take_dirty(), [part(1, 2)]);
        // In part: looked at again when the rest is.
        copy.note_landing(3 * KERNEL_PAGE, 3 * KERNEL_PAGE + 1);
        let (from, to) = page(3, 4);
        copy.take_landed(from, from + 100);
        copy.take_landed(from, to);
        assert_eq!(copy.take_dirty(), []);
        drop(room);
    }
}

#[cfg(test)]
mod parts_tests {
    use super::{PAGE, by_parts};

    #[test]
    fn runs_are_cut_to_the_parts_written() {
        const P: usize = PAGE / 16;
        let at = |page: usize, part: usize| page * PAGE + part * P;
        // Pages 10 and 11 again (from the middle of 10's part 1 on), 12
        // never taken, 14 again.
        let runs = vec![
            (at(10, 1) + 100, at(12, 0), true),
            (at(12, 0), at(13, 0), false),
            (at(14, 0), at(14, 9) + 7, true),
        ];
        let pages = [(10, 5), (11, 5), (14, 6)];
        assert_eq!(by_parts(runs.clone(), &pages, &[u16::MAX; 3]), runs);
        assert_eq!(
            by_parts(runs, &pages, &[0b1000_0000_0000_0011, 0b11, 0b10_0000_0000]),
            [
                (at(10, 1) + 100, at(10, 2), true),
                (at(10, 15), at(11, 2), true),
                (at(12, 0), at(13, 0), false),
                (at(14, 9), at(14, 9) + 7, true),
            ]
        );
    }
}

#[cfg(test)]
mod stale_tests {
    use super::*;

    #[test]
    fn pages_never_taken_are_stale_written_or_not() {
        let base = 0x100;
        let copied = [5, 0, 5, 5];
        let at = |page: usize| page * PAGE;
        // Page 0x101 was never taken; page 0x102 was written after it was
        // taken, page 0x103 before.
        let written = [(0x102, 6), (0x103, 4)];
        assert_eq!(stale_pages_in(base, &copied, at(0x100), at(0x104), &written), [0x101, 0x102]);
        // Part of a page counts as the page; pages outside the copy do not.
        assert_eq!(stale_pages_in(base, &copied, at(0x101) + 8, at(0x101) + 16, &[]), [0x101]);
        assert!(stale_pages_in(base, &copied, at(0x104), at(0x106), &[(0x104, 9)]).is_empty());
    }
}

/// Copies of windows outside buffers let go since the start, and their
/// bytes.
static LET_GO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// For the memory line: the windows let go since the start and their
/// bytes; of the bytes of the windows alive, how many lie between the
/// lowest and the highest address ever asked for in each; the windows made
/// again within a minute of a let-go at the same key since the start and
/// their bytes; the window buffers taken from the spare list since the
/// start, and those kept there now and their bytes ([`window_keep`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn windows() -> [u64; 8] {
    let asked: u64 = {
        let loose = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        loose
            .values()
            .map(|s| {
                let low = s.asked[0].load(Ordering::Relaxed);
                let high = s.asked[1].load(Ordering::Relaxed);
                high.saturating_sub(low)
            })
            .sum()
    };
    let (spares, spare_bytes) = {
        let spare = SPARE_WINDOWS.lock().unwrap_or_else(|e| e.into_inner());
        (spare.kept.len() as u64, spare.bytes)
    };
    [
        LET_GO[0].load(Ordering::Relaxed),
        LET_GO[1].load(Ordering::Relaxed),
        asked,
        REMADE[0].load(Ordering::Relaxed),
        REMADE[1].load(Ordering::Relaxed),
        SPARE_TAKEN.load(Ordering::Relaxed),
        spares,
        spare_bytes,
    ]
}

/// For the memory line, of the windows alive: the bytes of their pages
/// ever copied from title memory (the rest of each window is memory the
/// host zeroed and nothing read: what a window made no bigger than what is
/// read would not hold); the windows a descriptor slot shows and their
/// bytes (they stay until the title writes the slot again); the windows a
/// list or a pass over the heap holds and their bytes; the windows asked
/// for without a length (root views, which keep the 4 MiB grid) and their
/// bytes; and the window grid in bytes ([`window`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn window_use() -> [u64; 8] {
    let mut out = [0u64; 8];
    let loose = LOOSE.write().unwrap_or_else(|e| e.into_inner());
    for s in loose.values() {
        let copied = s.copied.lock().unwrap_or_else(|e| e.into_inner());
        let pages = copied.iter().filter(|&&at| at != 0).count() as u64;
        out[0] += (pages * PAGE as u64).min(s.len);
        if crate::host::references(&s.buffer) > 1 {
            out[1] += 1;
            out[2] += s.len;
        }
        if Arc::strong_count(s) > 1 {
            out[3] += 1;
            out[4] += s.len;
        }
        if s.unsized_asks.load(Ordering::Relaxed) > 0 {
            out[5] += 1;
            out[6] += s.len;
        }
    }
    out[7] = window();
    out
}

/// Optional byte cap for loose title-window copies, in MiB. Unset or `0`
/// keeps today's sweep. A positive value caps the copies only in the Watch
/// and High RAM bands; Calm is unchanged.
fn window_budget() -> Option<u64> {
    static MB: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *MB.get_or_init(|| {
        std::env::var("VICTORY_WINDOW_BUDGET_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&v| v > 0)
            .map(|v| v.saturating_mul(1 << 20))
    })
}

/// Oldest eligible copies to remove until their unique bytes fit `keep`.
/// Entries are `(last-used moment, bytes, unique copy identity)`, already
/// sorted oldest first; one copy standing under several keys counts once.
fn window_victims(entries: &[(u64, u64, usize)], keep: u64) -> Vec<usize> {
    let mut seen = std::collections::HashSet::new();
    let eligible = entries
        .iter()
        .filter(|&&(_, _, id)| seen.insert(id))
        .fold(0u64, |sum, &(_, bytes, _)| sum.saturating_add(bytes));
    window_victims_over(entries, eligible, keep)
}

/// Oldest eligible copies to remove until `alive` bytes, less theirs, fit
/// `keep`. Entries as in [`window_victims`].
fn window_victims_over(entries: &[(u64, u64, usize)], alive: u64, keep: u64) -> Vec<usize> {
    let mut left = alive;
    let mut taken = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (i, &(_, bytes, id)) in entries.iter().enumerate() {
        if left <= keep {
            break;
        }
        if taken.insert(id) {
            left = left.saturating_sub(bytes);
            out.push(i);
        }
    }
    out
}

/// Whether windows outside buffers go by bytes instead of by the clock, and
/// a window let go hands its buffer on to the next one made
/// (`VICTORY_WINDOW_KEEP=1`; off by default). On a PC of 32 GB or less
/// ([`crate::residency::limited_host_ram`]) the clock alone lets a window
/// go after 10 s idle, 5 s under 4 GiB free: session notwin let 2,369 go
/// (10.9 GB) and made 36,393 shared buffers (17.1 GB) in 520 s of world,
/// while the windows alive stayed between 2.9 and 4.2 GB. With this on
/// ([`window_rule`]):
/// - a window goes only while the windows alive exceed a line
///   (`VICTORY_WINDOW_BUDGET_MB` if set, else the most bytes of windows
///   named in any one frame of the last [`NAMED_FRAMES`], plus a quarter),
///   longest idle first, and never under 30 s idle; the High RAM band keeps
///   the rule without it (2 s);
/// - a 4 MiB window's buffer goes to a spare list ([`SPARE_WINDOWS`]) and
///   the next window of that size takes it instead of a new shared buffer.
fn window_keep() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_KEEP").is_some_and(|v| v == "1"))
}

/// How the sweep lets windows go: after how long idle, down to what line,
/// and whether the line is over all windows alive or (the rule before
/// [`window_keep`]) over those that may go.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WindowRule {
    idle_ms: u64,
    /// `u64::MAX`: every idle window goes.
    line: u64,
    over_alive: bool,
}

/// The windows' rule for a sweep: `band` the RAM band of this sweep (`None`
/// if not sampled), `budget` `VICTORY_WINDOW_BUDGET_MB` in bytes,
/// `ram_idle` [`crate::host::ram_idle_ms`] and `named` the most bytes of
/// windows named in one frame lately ([`Named::peak`]).
fn window_rule(
    keep: bool,
    band: Option<crate::host::RamBand>,
    budget: Option<u64>,
    ram_idle: u64,
    named: u64,
) -> WindowRule {
    use crate::host::RamBand;
    const KEPT_MS: u64 = 30_000;
    let mut rule = WindowRule { idle_ms: ram_idle, line: u64::MAX, over_alive: false };
    match (band, budget) {
        (Some(RamBand::High), _) if keep || budget.is_some() => {
            rule.idle_ms = ram_idle.min(2_000);
            rule.line = budget.map_or(u64::MAX, |b| b / 2);
        }
        _ if keep => {
            rule.idle_ms = ram_idle.max(KEPT_MS);
            rule.line = budget.unwrap_or(named.saturating_add(named / 4));
            rule.over_alive = true;
        }
        (Some(RamBand::Watch), Some(bytes)) => {
            rule.idle_ms = ram_idle.min(10_000);
            rule.line = bytes;
        }
        _ => {}
    }
    rule
}

/// The frames [`Named`] looks back over.
const NAMED_FRAMES: usize = 600;

/// The bytes of windows named in each of the last [`NAMED_FRAMES`] frames
/// (with [`window_keep`]): each window counts once a frame, when it is
/// first asked for in it or made.
struct Named {
    frame: u64,
    /// This frame's bytes so far.
    now: u64,
    /// The frames before it, the oldest first.
    before: std::collections::VecDeque<u64>,
}

impl Named {
    const fn new() -> Named {
        Named { frame: 0, now: 0, before: std::collections::VecDeque::new() }
    }

    /// A window of `bytes` named first in `frame`. (A thread a frame late
    /// counts it in the frame begun.)
    fn add(&mut self, frame: u64, bytes: u64) {
        if frame > self.frame {
            if self.before.len() + 1 >= NAMED_FRAMES {
                self.before.pop_front();
            }
            self.before.push_back(self.now);
            self.frame = frame;
            self.now = 0;
        }
        self.now = self.now.saturating_add(bytes);
    }

    /// The most bytes named in one frame of those looked back over.
    fn peak(&self) -> u64 {
        self.before.iter().copied().fold(self.now, u64::max)
    }
}

static NAMED: Mutex<Named> = Mutex::new(Named::new());

/// A window's buffer kept for the next window of its size
/// ([`window_keep`]), zeroed as a new one is.
struct SpareWindow {
    buffer: ID3D12Resource,
    data: *mut u8,
}

// SAFETY: a D3D12 buffer (free-threaded) and its persistent mapping, which
// only the holder of the spare writes.
unsafe impl Send for SpareWindow {}

/// Buffers kept for reuse, each with its bytes: [`Spares::MOST`] at most and
/// [`Spares::MOST_BYTES`].
struct Spares<T> {
    kept: Vec<(T, u64)>,
    bytes: u64,
}

impl<T> Spares<T> {
    const MOST: usize = 64;
    const MOST_BYTES: u64 = 256 << 20;

    const fn new() -> Spares<T> {
        Spares { kept: Vec::new(), bytes: 0 }
    }

    /// Whether one more of `bytes` fits.
    fn room(&self, bytes: u64) -> bool {
        self.kept.len() < Self::MOST && self.bytes.saturating_add(bytes) <= Self::MOST_BYTES
    }

    /// Keeps `item` if it fits; else hands it back.
    fn give(&mut self, item: T, bytes: u64) -> Option<T> {
        if !self.room(bytes) {
            return Some(item);
        }
        self.bytes += bytes;
        self.kept.push((item, bytes));
        None
    }

    /// One kept of exactly `bytes`.
    fn take(&mut self, bytes: u64) -> Option<T> {
        let at = self.kept.iter().rposition(|&(_, b)| b == bytes)?;
        self.bytes -= bytes;
        Some(self.kept.swap_remove(at).0)
    }

    /// All of them, the list left empty.
    fn drain(&mut self) -> Vec<T> {
        self.bytes = 0;
        std::mem::take(&mut self.kept).into_iter().map(|(item, _)| item).collect()
    }
}

/// Window buffers kept for reuse ([`window_keep`]).
static SPARE_WINDOWS: Mutex<Spares<SpareWindow>> = Mutex::new(Spares::new());

/// Window buffers taken from [`SPARE_WINDOWS`] since the start.
static SPARE_TAKEN: AtomicU64 = AtomicU64::new(0);

/// A buffer for a new window of `len` bytes from the spare list, if there
/// is one.
fn spare_window(len: u64) -> Option<(ID3D12Resource, *mut u8)> {
    if !window_keep() || len != WINDOW {
        return None;
    }
    let spare = SPARE_WINDOWS.lock().unwrap_or_else(|e| e.into_inner()).take(len)?;
    SPARE_TAKEN.fetch_add(1, Ordering::Relaxed);
    Some((spare.buffer, spare.data))
}

/// Lets go of the window buffers kept for reuse, when the system is short
/// of memory (from `crate::host`'s trim of its spare chunks).
pub(crate) fn trim_spare_windows() {
    let freed = SPARE_WINDOWS.lock().unwrap_or_else(|e| e.into_inner()).drain();
    // Freed outside the lock.
    drop(freed);
}

/// Retired after a window let go ([`let_go_idle`]): once the copy is
/// dropped and nothing else holds its buffer, the buffer goes to the spare
/// list.
struct ToSpare(Option<SpareWindow>);

impl Drop for ToSpare {
    fn drop(&mut self) {
        let Some(spare) = self.0.take() else { return };
        if crate::host::references(&spare.buffer) != 1
            || !SPARE_WINDOWS.lock().unwrap_or_else(|e| e.into_inner()).room(WINDOW)
        {
            return;
        }
        // SAFETY: the buffer's persistent mapping of `WINDOW` bytes, which
        // nothing else holds any more and no submission reads (retired).
        unsafe { std::ptr::write_bytes(spare.data, 0, WINDOW as usize) };
        let back = SPARE_WINDOWS.lock().unwrap_or_else(|e| e.into_inner()).give(spare, WINDOW);
        drop(back);
    }
}

/// When each window let go in the last [`LetGoBook::WITHIN_MS`] went, by
/// its key, to count those made again soon after.
struct LetGoBook {
    gone: std::collections::HashMap<(u64, u64), u64>,
}

impl LetGoBook {
    const WITHIN_MS: u64 = 60_000;

    fn new() -> LetGoBook {
        LetGoBook { gone: std::collections::HashMap::new() }
    }

    /// The window under `key` let go at `now` (ms); the old entries go.
    fn gone(&mut self, key: (u64, u64), now: u64) {
        self.gone.retain(|_, &mut at| now.saturating_sub(at) <= Self::WITHIN_MS);
        self.gone.insert(key, now);
    }

    /// A window made under `key` at `now`: whether one there went within
    /// the last minute.
    fn made(&mut self, key: (u64, u64), now: u64) -> bool {
        self.gone.remove(&key).is_some_and(|at| now.saturating_sub(at) <= Self::WITHIN_MS)
    }
}

static LET_GO_BOOK: std::sync::LazyLock<Mutex<LetGoBook>> =
    std::sync::LazyLock::new(|| Mutex::new(LetGoBook::new()));

/// Windows made again within a minute of a let-go at the same key since
/// the start, and their bytes.
static REMADE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Kept between sweeps so one sample near a watermark does not toggle the
/// policy on the next submission.
static WINDOW_BAND: std::sync::Mutex<crate::host::RamBand> =
    std::sync::Mutex::new(crate::host::RamBand::Calm);

/// Lets go of the copies of windows outside buffers that nothing names any
/// more. They were never freed, and the world's streaming keeps asking for
/// new ones: 648 windows (2.8 GB) when the cutscene began, 974 (4.3 GB)
/// eight minutes later (run 24). A window goes when
/// - only the table of windows holds its copy: no list not yet reset reads
///   or writes it, no pass over the heap keeps its range, no submission is
///   comparing it;
/// - only the copy holds its host buffer: no descriptor slot views it (a
///   slot keeps the buffer it views, `HostHeap::kept`), so no draw can read
///   it through a slot made earlier;
/// - nothing asked for it for `IDLE_MS` and `IDLE_FRAMES` frames.
///
/// The next read of that memory makes a new copy and copies the window
/// again. Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    const IDLE_FRAMES: u64 = 4;
    // Windows that are to change sides (`VICTORY_CARD_WINDOWS_KINDS`).
    kind_moves();
    // Before the windows: their table may be empty while the pool is not.
    let_go_idle_snapshots();
    let budget = window_budget();
    let kept = window_keep();
    let band = if budget.is_some() || kept {
        crate::host::ram_sample().map(|(working, private, free, commit, total, limit)| {
            let mut band = WINDOW_BAND.lock().unwrap_or_else(|e| e.into_inner());
            *band = crate::host::ram_band(*band, working, private, free, commit, total, limit);
            *band
        })
    } else {
        None
    };
    let named = if kept { NAMED.lock().unwrap_or_else(|e| e.into_inner()).peak() } else { 0 };
    let ram_idle = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    let rule = window_rule(kept, band, budget, ram_idle, named);
    let gone: Vec<Arc<Shadow>> = {
        // Held from the look to the removal: `loose` hands a copy out
        // under the same lock, so none is handed out in between.
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        if map.is_empty() {
            return;
        }
        let mut eligible: Vec<(u64, (u64, u64), usize, u64)> = map
            .iter()
            .filter(|(_, s)| {
                Arc::strong_count(s) == 1
                    && crate::host::long_ago(s.used.load(Ordering::Relaxed), rule.idle_ms, IDLE_FRAMES)
                    && crate::host::references(&s.buffer) == 1
            })
            .map(|(&key, s)| {
                (s.used.load(Ordering::Relaxed), key, Arc::as_ptr(s) as usize, s.len)
            })
            .collect();
        eligible.sort_unstable_by_key(|&(used, _, _, _)| used);
        let victims = if rule.line == u64::MAX {
            (0..eligible.len()).collect()
        } else {
            let entries: Vec<(u64, u64, usize)> = eligible
                .iter()
                .map(|&(used, _, id, len)| (used, len, id))
                .collect();
            if rule.over_alive {
                // Each copy once: a window may stand under several keys.
                let mut seen = std::collections::HashSet::new();
                let alive = map
                    .values()
                    .filter(|s| seen.insert(Arc::as_ptr(s) as usize))
                    .fold(0u64, |sum, s| sum.saturating_add(s.len));
                window_victims_over(&entries, alive, rule.line)
            } else {
                window_victims(&entries, rule.line)
            }
        };
        victims.into_iter().filter_map(|i| map.remove(&eligible[i].1)).collect()
    };
    if !gone.is_empty() {
        let (went, bytes) = (gone.len(), gone.iter().map(|s| s.len).sum::<u64>());
        crate::host::trail(|| {
            format!("sweep: {went} host copies of windows of title memory let go ({} KiB), held by no list and no slot", bytes >> 10)
        });
    }
    let now = crate::host::now_ms();
    for shadow in gone {
        LET_GO[0].fetch_add(1, Ordering::Relaxed);
        LET_GO[1].fetch_add(shadow.len, Ordering::Relaxed);
        LET_GO_BOOK.lock().unwrap_or_else(|e| e.into_inner()).gone(shadow.span(), now);
        // Nothing runs now and nothing names it: freed at once unless a
        // submission is running after all. A plain 4 MiB window's buffer
        // then goes to the spare list ([`window_keep`]).
        let plain = shadow.card.is_none() && !shadow.mapped && shadow.place.is_none();
        if kept && plain && shadow.len == WINDOW {
            let spare = SpareWindow { buffer: shadow.buffer.clone(), data: shadow.data };
            crate::host::retire((shadow, ToSpare(Some(spare))));
        } else {
            crate::host::retire(shadow);
        }
    }
}

/// For the memory line: host copies alive and their bytes, of them the
/// windows of memory outside buffers and their bytes, and the snapshot
/// buffers kept for reuse and the bytes they reserve.
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 6] {
    let (windows, window_bytes) = {
        let loose = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        // Each copy once: a window may stand under several keys.
        let mut seen = std::collections::HashSet::new();
        let bytes: u64 = loose
            .values()
            .filter(|s| seen.insert(Arc::as_ptr(s) as usize))
            .map(|s| s.len)
            .sum();
        (seen.len() as u64, bytes)
    };
    let (kept, kept_bytes) = {
        let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        (pool.buffers.len() as u64, pool.bytes as u64)
    };
    [
        LIVE[0].load(Ordering::Relaxed),
        LIVE[1].load(Ordering::Relaxed),
        windows,
        window_bytes,
        kept,
        kept_bytes,
    ]
}

/// With `VICTORY_CARD_WINDOWS_KINDS`: which windows are to have a side on
/// the card, as the last line of reads ranked them
/// (`crate::readkinds::choose`). A window keeps the side it was made with,
/// so one that is to change sides is let go when nothing holds it and made
/// again by the next ask for it: on the card if it is `waiting`, else in the
/// PC's memory.
#[derive(Default)]
struct KindPlan {
    /// Chosen and not on the card yet.
    waiting: std::collections::BTreeSet<(u64, u64)>,
    /// The bytes of `waiting`: room new windows leave them.
    reserved: u64,
    /// On the card and not chosen any more: to be let go.
    leaving: std::collections::BTreeSet<(u64, u64)>,
    /// Chosen, and the card had no room when they were made again: not
    /// chosen again until the list is cleared (every twelfth line).
    refused: std::collections::BTreeSet<(u64, u64)>,
    /// Bytes named a frame by kind, by window: a running mean over the lines.
    read: std::collections::HashMap<(u64, u64), [u64; crate::readkinds::KINDS]>,
    /// Lines so far.
    lines: u64,
    /// Since the start: windows let go to be made on the card, and to be
    /// made in the PC's memory.
    moved: [u64; 2],
}

impl KindPlan {
    /// Whether a window being made under `key`, of `size` bytes, may have a
    /// side on the card: one chosen may; one not known yet may if its first
    /// ask is for a kind asked for (`asked`) and it fits beside the windows
    /// chosen that are not there yet (`used` of `most` bytes are taken).
    fn admits(&self, key: (u64, u64), size: u64, asked: bool, used: u64, most: u64) -> bool {
        if self.waiting.contains(&key) {
            return true;
        }
        if self.refused.contains(&key) || self.read.contains_key(&key) || !asked {
            return false;
        }
        used.saturating_add(self.reserved).saturating_add(size) <= most
    }

    /// A window that was waiting was made again: on the card, or not
    /// (`there`).
    fn landed(&mut self, key: (u64, u64), size: u64, there: bool) {
        if self.waiting.remove(&key) {
            self.reserved = self.reserved.saturating_sub(size);
            if !there {
                self.refused.insert(key);
            }
        }
    }

    /// A line's reads: `seen` are the windows alive, each with whether it is
    /// on the card, `[binds, bytes]` by kind over `frames` frames, and its
    /// bytes. Sets who waits and who leaves; gives the number chosen and
    /// their bytes.
    fn rank(&mut self, seen: &[(crate::readkinds::Window, u64)], allowed: u8, most: u64, frames: f64) -> (usize, u64) {
        use crate::readkinds::{KINDS, Ranked};
        self.lines += 1;
        if self.lines.is_multiple_of(12) {
            self.refused.clear();
        }
        // Windows that are gone are forgotten; the others' means move half
        // the way to what this line saw.
        let alive: std::collections::HashSet<(u64, u64)> = seen.iter().map(|(w, _)| w.key).collect();
        self.read.retain(|key, _| alive.contains(key));
        let mut ranked = Vec::with_capacity(seen.len());
        for (window, len) in seen {
            let now: [u64; KINDS] = std::array::from_fn(|k| (window.reads[k][1] as f64 / frames) as u64);
            let known = self.read.get(&window.key).copied();
            let mean = match known {
                Some(before) => std::array::from_fn(|k| (before[k] + now[k]) / 2),
                None => now,
            };
            if known.is_some() || mean.iter().any(|&b| b > 0) {
                self.read.insert(window.key, mean);
            }
            if !self.refused.contains(&window.key) {
                ranked.push(Ranked { key: window.key, len: *len, on_card: window.on_card, bytes: mean });
            }
        }
        let chosen = crate::readkinds::choose(&ranked, allowed, most);
        let by_key: std::collections::HashMap<(u64, u64), &Ranked> = ranked.iter().map(|r| (r.key, r)).collect();
        let bytes = chosen.iter().map(|key| by_key[key].len).sum();
        let chosen_set: std::collections::BTreeSet<(u64, u64)> = chosen.iter().copied().collect();
        self.waiting = chosen.iter().copied().filter(|key| !by_key[key].on_card).collect();
        self.reserved = self.waiting.iter().map(|key| by_key[key].len).sum();
        self.leaving = ranked.iter().filter(|r| r.on_card && !chosen_set.contains(&r.key)).map(|r| r.key).collect();
        (chosen.len(), bytes)
    }
}

static KIND_PLAN: std::sync::LazyLock<Mutex<KindPlan>> = std::sync::LazyLock::new(Mutex::default);

/// Whether [`KIND_PLAN`] has windows that wait or leave: read before every
/// submission, so that nothing is locked when there are none.
static KIND_MOVES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// With `VICTORY_CARD_WINDOWS_KINDS`: lets go of the windows that are to
/// change sides ([`KindPlan`]) and that nothing holds, as [`let_go_idle`]
/// lets go of idle ones (no list, no slot), and that no shader was ever
/// declared to write. The next ask makes each again, on its new side.
/// Called before a submission starts, by the thread that submits.
fn kind_moves() {
    if !KIND_MOVES.load(Ordering::Relaxed) {
        return;
    }
    let gone: Vec<Arc<Shadow>> = {
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        let mut plan = KIND_PLAN.lock().unwrap_or_else(|e| e.into_inner());
        let free = |s: &Arc<Shadow>| {
            Arc::strong_count(s) == 1 && !s.written.load(Ordering::Relaxed) && crate::host::references(&s.buffer) == 1
        };
        let mut gone = Vec::new();
        // To the card: only a window that is in the PC's memory (one made
        // on the card meanwhile has left `waiting`).
        let up: Vec<(u64, u64)> =
            plan.waiting.iter().copied().filter(|key| map.get(key).is_some_and(|s| s.lives() == 2 && free(s))).collect();
        for key in up {
            gone.extend(map.remove(&key));
            plan.moved[0] += 1;
        }
        let down: Vec<(u64, u64)> = plan.leaving.iter().copied().filter(|key| map.get(key).is_none_or(free)).collect();
        for key in down {
            plan.leaving.remove(&key);
            if let Some(shadow) = map.remove(&key) {
                gone.push(shadow);
                plan.moved[1] += 1;
            }
        }
        // (A window that waits and is not in the table is made on the card
        // by its next ask; one that is held stays for a later submission.)
        if plan.leaving.is_empty() && !plan.waiting.iter().any(|key| map.contains_key(key)) {
            KIND_MOVES.store(false, Ordering::Relaxed);
        }
        gone
    };
    for shadow in gone {
        crate::host::retire(shadow);
    }
}

/// For the frame-rate line, with `crate::readkinds::on`: what the card read
/// of each window since the last one, by kind; and with
/// `VICTORY_CARD_WINDOWS_KINDS`, which windows are to be on the card for it.
pub(crate) fn read_kinds_line(frames: f64) {
    if !crate::readkinds::on() {
        return;
    }
    // With each window: its bytes, and whether it may change sides (not one
    // in place or mapped on the card, which are made so again whatever is
    // chosen, and not one shaders were declared to write, which stays).
    let mut fixed = 0u64;
    let (seen, movable): (Vec<(crate::readkinds::Window, u64)>, Vec<bool>) = {
        let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        // Each copy once: a window may stand under several keys.
        let mut once = std::collections::HashSet::new();
        map.iter()
            .filter(|(_, s)| once.insert(Arc::as_ptr(s) as usize))
            .map(|(&key, s)| {
                let window = crate::readkinds::Window { key, on_card: s.lives() == 1, reads: s.reads.take() };
                let movable = s.place.is_none() && !s.mapped && !s.written.load(Ordering::Relaxed);
                // (A side on the card that stays takes of the same room.)
                if !movable && s.window && s.card.is_some() {
                    fixed += s.len;
                }
                ((window, s.len), movable)
            })
            .unzip()
    };
    let all = crate::readkinds::take_all();
    if crate::host::kinds::on() {
        let windows: Vec<crate::readkinds::Window> = seen.iter().map(|(w, _)| w.clone()).collect();
        tracing::info!(
            "{} (per frame, VICTORY_KIND_COUNTS=1; MB are the bytes the binds named, a root view taken as 64 KiB at \
             most; binds of vertex and index are draws)",
            crate::readkinds::reads_text(&windows, seen.len(), &all, frames)
        );
    }
    let Some(allowed) = crate::readkinds::allowed() else {
        return;
    };
    let most = window_card_most().saturating_sub(fixed);
    let mut plan = KIND_PLAN.lock().unwrap_or_else(|e| e.into_inner());
    let planned: Vec<(crate::readkinds::Window, u64)> =
        seen.iter().zip(&movable).filter(|(_, movable)| **movable).map(|(w, _)| w.clone()).collect();
    let (chosen, bytes) = plan.rank(&planned, allowed, most, frames);
    KIND_MOVES.store(!plan.waiting.is_empty() || !plan.leaving.is_empty(), Ordering::Relaxed);
    tracing::info!(
        "windows to the card by kind (VICTORY_CARD_WINDOWS_KINDS, kinds {allowed:#010b}): {chosen} chosen of {} that may change sides ({} alive), \
         {:.0} of {:.0} MB (VICTORY_CARD_WINDOWS_MB less what stays there), {} of them not there yet ({:.0} MB kept for them), {} on the card \
         to leave, {} turned down for want of room; {:.0} MB of windows on the card now; let go to change sides since \
         the start: {} to the card, {} from it",
        planned.len(),
        seen.len(),
        bytes as f64 / 1048576.0,
        most as f64 / 1048576.0,
        plan.waiting.len(),
        plan.reserved as f64 / 1048576.0,
        plan.leaving.len(),
        plan.refused.len(),
        WINDOW_CARD_BYTES.load(Ordering::Relaxed) as f64 / 1048576.0,
        plan.moved[0],
        plan.moved[1],
    );
}

#[cfg(test)]
mod kind_plan_tests {
    use super::KindPlan;
    use crate::readkinds::{INDEX, KINDS, ROOT_SRV, VERTEX, Window};

    const MB: u64 = 1 << 20;

    fn seen(at: u64, on_card: bool, reads: &[(usize, u64)]) -> (Window, u64) {
        let mut all = [[0u64; 2]; KINDS];
        for &(kind, bytes) in reads {
            all[kind] = [1, bytes];
        }
        (Window { key: (at << 22, 4 * MB), on_card, reads: all }, 4 * MB)
    }

    #[test]
    fn the_most_read_windows_of_the_kinds_wait_for_the_card_and_the_others_leave() {
        let mut plan = KindPlan::default();
        let windows = [
            seen(0, false, &[(ROOT_SRV, 40 * MB)]),
            seen(1, false, &[(ROOT_SRV, 10 * MB)]),
            seen(2, false, &[(VERTEX, 90 * MB)]),
            // On the card, of a kind not asked for.
            seen(3, true, &[(VERTEX, 90 * MB)]),
            // On the card and chosen: stays.
            seen(4, true, &[(ROOT_SRV, 30 * MB)]),
        ];
        // Room for two windows; over one frame.
        let (chosen, bytes) = plan.rank(&windows, 1 << ROOT_SRV, 8 * MB, 1.0);
        assert_eq!((chosen, bytes), (2, 8 * MB));
        assert_eq!(plan.waiting.iter().copied().collect::<Vec<_>>(), vec![(0, 4 * MB)]);
        assert_eq!(plan.reserved, 4 * MB);
        assert_eq!(plan.leaving.iter().copied().collect::<Vec<_>>(), vec![(3 << 22, 4 * MB)]);
        // Made again on the card: waits no more, and nothing is kept for it.
        plan.landed((0, 4 * MB), 4 * MB, true);
        assert!(plan.waiting.is_empty());
        assert_eq!(plan.reserved, 0);
        assert!(plan.refused.is_empty());
    }

    #[test]
    fn a_new_window_goes_to_the_card_only_for_a_kind_asked_for_and_beside_those_chosen() {
        let mut plan = KindPlan::default();
        let windows = [seen(0, false, &[(INDEX, 40 * MB)]), seen(1, false, &[(VERTEX, 40 * MB)])];
        plan.rank(&windows, 1 << INDEX, 16 * MB, 1.0);
        let new = (9 << 22, 4 * MB);
        // Chosen: yes, whatever the ask and the room.
        assert!(plan.admits((0, 4 * MB), 4 * MB, false, 16 * MB, 16 * MB));
        // Known and not chosen: no, even asked for as a kind allowed.
        assert!(!plan.admits((1 << 22, 4 * MB), 4 * MB, true, 0, 16 * MB));
        // Not known: by its first ask, while it fits beside the 4 MB kept.
        assert!(plan.admits(new, 4 * MB, true, 8 * MB, 16 * MB));
        assert!(!plan.admits(new, 4 * MB, true, 12 * MB, 16 * MB));
        assert!(!plan.admits(new, 4 * MB, false, 0, 16 * MB));
    }

    #[test]
    fn a_window_the_card_had_no_room_for_is_not_chosen_again_for_a_while() {
        let mut plan = KindPlan::default();
        let windows = [seen(0, false, &[(INDEX, 40 * MB)])];
        plan.rank(&windows, 1 << INDEX, 16 * MB, 1.0);
        plan.landed((0, 4 * MB), 4 * MB, false);
        assert!(plan.refused.contains(&(0, 4 * MB)));
        assert!(!plan.admits((0, 4 * MB), 4 * MB, true, 0, 16 * MB));
        assert_eq!(plan.rank(&windows, 1 << INDEX, 16 * MB, 1.0), (0, 0));
        assert!(plan.waiting.is_empty());
        // The twelfth line clears the list.
        for _ in 0..9 {
            assert_eq!(plan.rank(&windows, 1 << INDEX, 16 * MB, 1.0), (0, 0));
        }
        assert_eq!(plan.rank(&windows, 1 << INDEX, 16 * MB, 1.0), (1, 4 * MB));
    }

    #[test]
    fn what_was_read_is_a_running_mean_and_gone_windows_are_forgotten() {
        let mut plan = KindPlan::default();
        plan.rank(&[seen(0, false, &[(INDEX, 40 * MB)])], 1 << INDEX, 16 * MB, 2.0);
        assert_eq!(plan.read[&(0, 4 * MB)][INDEX], 20 * MB);
        // Not read in the next line: half, and still chosen.
        assert_eq!(plan.rank(&[seen(0, false, &[])], 1 << INDEX, 16 * MB, 2.0), (1, 4 * MB));
        assert_eq!(plan.read[&(0, 4 * MB)][INDEX], 10 * MB);
        plan.rank(&[], 1 << INDEX, 16 * MB, 2.0);
        assert!(plan.read.is_empty());
        assert!(plan.waiting.is_empty());
    }
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

/// What title memory is bound for ([`Shadow::bound`]).
pub(crate) const BOUND_VERTICES: usize = 0;
pub(crate) const BOUND_INDICES: usize = 1;
pub(crate) const BOUND_ROOT_VIEW: usize = 2;
pub(crate) const BOUND_TABLE_CONSTANTS: usize = 3;
const BOUND_FOR: [&str; 4] = ["vertex streams", "index buffers", "root views", "constants of tables read in place"];
/// Where the card finds a copy's bytes when a shader reads them
/// ([`Shadow::lives`]).
const LIVES: [&str; 5] = [
    "card copies of title buffers",
    "windows on the card",
    "windows in the PC's memory",
    "title memory in place (the PC's)",
    "small copies in the PC's memory",
];

/// Since the last line, with `VICTORY_KIND_COUNTS=1`: times title memory
/// was bound for the card to read, by what for and by where its bytes lie,
/// and the bytes the views named (root views name none: they reach to their
/// copy's end).
static BOUND: [[[AtomicU64; 2]; 5]; 4] = [const { [const { [const { AtomicU64::new(0) }; 2] }; 5] }; 4];

/// The text of the line of what was bound: `counts[what for][where]` are
/// times and bytes over `frames` frames.
fn bound_text(counts: &[[[u64; 2]; 5]; 4], frames: f64) -> String {
    let all: u64 = counts.iter().flatten().map(|c| c[0]).sum();
    let pc: u64 = counts.iter().flat_map(|w| &w[2..]).map(|c| c[0]).sum();
    let mut parts = Vec::new();
    for (what, per) in BOUND_FOR.iter().zip(counts) {
        let places: Vec<String> = LIVES
            .iter()
            .zip(per)
            .filter(|(_, c)| c[0] > 0)
            .map(|(place, c)| {
                if c[1] > 0 {
                    format!("{:.0} in {place} ({:.1} MB named)", c[0] as f64 / frames, c[1] as f64 / frames / 1048576.0)
                } else {
                    format!("{:.0} in {place}", c[0] as f64 / frames)
                }
            })
            .collect();
        if !places.is_empty() {
            parts.push(format!("{what}: {}", places.join(", ")));
        }
    }
    format!(
        "{:.0} binds, {:.0} of them of memory the card reads over the bus; {}",
        all as f64 / frames,
        pc as f64 / frames,
        parts.join("; ")
    )
}

/// For the frame-rate line, with `VICTORY_KIND_COUNTS=1`: what was bound
/// since the last one.
pub(crate) fn bound_line(frames: f64) {
    let counts: [[[u64; 2]; 5]; 4] =
        std::array::from_fn(|w| std::array::from_fn(|l| std::array::from_fn(|i| BOUND[w][l][i].swap(0, Ordering::Relaxed))));
    if counts.iter().flatten().all(|c| c[0] == 0) {
        return;
    }
    tracing::info!(
        "title memory bound for the card to read per frame (VICTORY_KIND_COUNTS=1; views in descriptor heaps are not \
         counted, nor constants filled into upload memory): {}",
        bound_text(&counts, frames)
    );
}

#[cfg(test)]
mod bound_tests {
    use super::*;

    #[test]
    fn the_line_of_what_was_bound_says_how_much_of_it_crosses_the_bus() {
        let mut counts = [[[0u64; 2]; 5]; 4];
        // Over two frames: 6 vertex streams in windows in the PC's memory
        // (3 MiB named), 2 in card copies; 10 root views in windows on the card.
        counts[BOUND_VERTICES][2] = [6, 3 << 20];
        counts[BOUND_VERTICES][0] = [2, 1 << 20];
        counts[BOUND_ROOT_VIEW][1] = [10, 0];
        counts[BOUND_TABLE_CONSTANTS][3] = [4, 3 << 18];
        let text = bound_text(&counts, 2.0);
        assert_eq!(
            text,
            "11 binds, 5 of them of memory the card reads over the bus; vertex streams: 1 in card copies of title buffers \
             (0.5 MB named), 3 in windows in the PC's memory (1.5 MB named); root views: 5 in windows on the card; constants of \
             tables read in place: 2 in title memory in place (the PC's) (0.4 MB named)"
        );
    }
}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Where the card finds the bytes a shader reads of this copy (an index
    /// of [`LIVES`]). (With `VICTORY_CARD_ALL=1` or `VICTORY_WINDOWS_OWN=1` a
    /// "card" side may be the PC's memory all the same; neither is on in
    /// play.)
    fn lives(&self) -> usize {
        if self.place.is_some() {
            3
        } else if self.window {
            if self.card.is_some() || self.mapped { 1 } else { 2 }
        } else if self.card.is_some() {
            0
        } else {
            4
        }
    }

    /// With `VICTORY_KIND_COUNTS=1`: the copy is bound for the card to read,
    /// for `what` (`BOUND_*`), by a view of `bytes` (0: not said).
    pub(crate) fn bound(&self, what: usize, bytes: u64) {
        if !crate::host::kinds::on() {
            return;
        }
        let at = &BOUND[what][self.lives()];
        at[0].fetch_add(1, Ordering::Relaxed);
        at[1].fetch_add(bytes, Ordering::Relaxed);
    }

    /// With `crate::readkinds::on`: the copy is bound for the card to read
    /// as `kind` (of `crate::readkinds`), by a view of `bytes`.
    pub(crate) fn read_as(&self, kind: usize, bytes: u64) {
        if !crate::readkinds::on() {
            return;
        }
        self.reads.add(kind, bytes);
        crate::readkinds::read(kind, self.lives() < 2, bytes);
    }

    /// With `crate::readkinds::stores_on`: shaders are declared to store
    /// into `[address, address + len)` of the copy, by `by`
    /// (`crate::readkinds::STORE_*`). Counted for copies in place too, which
    /// [`Shadow::gpu_writes`] has nothing to do for.
    pub(crate) fn stored(&self, by: usize, address: u64, len: u64) {
        if !crate::readkinds::stores_on() {
            return;
        }
        let (lo, hi) = (address, address + len.min(self.left(address)));
        let [frame, low, high] = &self.stored_in;
        let now = crate::frame::current() + 1;
        // (Threads that declare at once may count a few bytes twice.)
        let before = if frame.swap(now, Ordering::Relaxed) == now {
            Some((low.fetch_min(lo, Ordering::Relaxed), high.fetch_max(hi, Ordering::Relaxed)))
        } else {
            low.store(lo, Ordering::Relaxed);
            high.store(hi, Ordering::Relaxed);
            None
        };
        let new = crate::readkinds::span_grows(before, lo, hi);
        crate::readkinds::stored(by, self.lives(), new, new == 0);
    }

    /// The title memory the copy is of: its start and length.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the copy, if it lies in it.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Whether the memory behind the copy turned out not to be write-watched:
    /// every refresh then copies its whole range.
    pub(crate) fn unwatched(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == NOT_WATCHED
    }

    /// Whether a submission asks the kernel about the copy's ranges ahead:
    /// not of memory that is not watched, nor of view memory, which the
    /// kernel would refuse (and with it the ranges asked about together).
    pub(crate) fn asked_ahead(&self) -> bool {
        let watched = self.watched.load(Ordering::Relaxed);
        watched != NOT_WATCHED && (watched != GUARDED || !crate::watch::light())
    }

    /// The copy's own number (see the field).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// The copy as `crate::census` names it.
    pub(crate) fn census(&self) -> crate::census::Which {
        crate::census::Which { id: self.id, start: self.start, len: self.len, window: self.window }
    }

    /// Whether it is no copy but title memory in place (`crate::inplace`).
    pub(crate) fn in_place(&self) -> bool {
        self.place.is_some()
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.lead
    }

    /// Title memory `[start, start + len)` in place: `placed` is a buffer
    /// over it in a heap opened on it. Shaped as a copy so that everything
    /// that binds, views or keeps a copy works on it unchanged; what
    /// copies does nothing ([`Shadow::update`], [`Shadow::gpu_writes`]).
    fn placed(placed: crate::inplace::Placed, start: u64, len: u64, window: bool) -> Shadow {
        crate::host::name(&placed.buffer, || format!("title memory {start:#x}+{len:#x} in place"));
        let mut shadow = Shadow::new(placed.buffer, start as *mut u8, placed.gpu, start, len);
        // Not a copy: not counted with them, and nothing asks what the
        // title wrote there.
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(len, Ordering::Relaxed);
        *shadow.watched.get_mut() = NOT_WATCHED;
        shadow.window = window;
        shadow.place = Some(placed.heap);
        shadow.lead = placed.bias;
        shadow
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        if let Some(uses) = &self.uses {
            uses.read(address, address + len, self.window, self.by_use_place());
        }
        list.defer_refresh(self, address as usize, len as usize);
    }

    /// The bytes of the host buffer.
    pub(crate) fn width(&self) -> u64 {
        // SAFETY: a live buffer.
        *self.width.get_or_init(|| unsafe { self.buffer.GetDesc() }.Width)
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        LIVE[0].fetch_add(1, Ordering::Relaxed);
        LIVE[1].fetch_add(len, Ordering::Relaxed);
        crate::census::copy_made(start, len);
        let id = {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            NEXT.fetch_add(1, Ordering::Relaxed)
        };
        Shadow {
            landings: Mutex::default(),
            landing: Default::default(),
            width: Default::default(),
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            ranged: Mutex::default(),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            flights: Mutex::default(),
            before: Mutex::default(),
            merging: RwLock::default(),
            kept: Mutex::default(),
            used: AtomicU64::new(crate::host::moment()),
            asked: [AtomicU64::new(u64::MAX), AtomicU64::new(0)],
            unsized_asks: AtomicU64::new(0),
            written: Default::default(),
            id,
            card: None,
            window: false,
            changes_only: Default::default(),
            mapped: false,
            prints: Mutex::default(),
            place: None,
            lead: 0,
            uses: crate::byuse::made(id, start, len),
            reads: Default::default(),
            stored_in: Default::default(),
        }
    }

    /// Where the copy's bytes lie, for `crate::byuse`. A copy in a block of
    /// views' copies ([`exact_in_blocks`]) lies in the block's shared
    /// buffer in the PC's memory, as other copies without a card side.
    fn by_use_place(&self) -> crate::byuse::Place {
        use crate::byuse::Place;
        match (&self.place, self.mapped, &self.card) {
            (Some(_), ..) => Place::InPlace,
            (None, true, _) => Place::Card,
            (None, false, Some(_)) => Place::Both,
            (None, false, None) => Place::Pc,
        }
    }

    /// `crate::byuse`: fact `bit` of the copy was seen.
    fn by_use(&self, bit: usize) {
        if let Some(uses) = &self.uses {
            uses.note(bit, self.window, self.by_use_place());
        }
    }

    /// `crate::byuse`: a piece of the copy is held on the card to see
    /// whether the title reads it (`crate::lazy_back`).
    pub(crate) fn by_use_sampled(&self) {
        if let Some(uses) = &self.uses {
            uses.sampled();
        }
    }

    /// `crate::byuse`: a read of the title's landed a held piece of it.
    pub(crate) fn by_use_title_read(&self) {
        if let Some(uses) = &self.uses {
            crate::byuse::title_read(uses, self.window, self.by_use_place());
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        // In place the shaders write title memory itself: no snapshot, no
        // compare, nothing to copy back, and no wait on its account.
        // shortcut: another host copy of the same bytes (a title buffer
        // inside this one with a copy of its own, a texture in it) is not
        // told: the card's writes are not seen by write watching.
        if self.place.is_some() {
            return;
        }
        self.written.store(true, Ordering::Relaxed);
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, each byte taken once, when the
    /// first range holding it is declared (the world declares overlapping
    /// views of its 174 MB heap dozens of times, from different offsets: a
    /// snapshot grown to cover each was copied whole every time), in reused
    /// memory (fresh snapshots cost ~2.3 million page faults per world
    /// frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if self.place.is_some() {
            return;
        }
        let range = (address, address + len);
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // With the copy-back on its own thread: counted for this submission
        // (once), whether or not an earlier one still holds the range.
        if relay() {
            self.flights.lock().unwrap_or_else(|e| e.into_inner()).declare(range, PREPARED.load(Ordering::Relaxed));
        }
        // Dispatches declare the same ranges thousands of times.
        if pending.contains(&range) {
            return;
        }
        // In the card's memory: no snapshot. What shaders wrote is found on
        // the card (`Host::card_before` reads the declared ranges from the
        // lists).
        if self.card.is_some() {
            pending.push(range);
            return;
        }
        // The parts no declared range holds yet: their bytes are taken now,
        // just after the range's refresh (a part declared earlier has not
        // been refreshed since, and keeps the bytes taken then).
        let mut parts = vec![self.inside(address, len)];
        for &(a, e) in pending.iter() {
            let (a, e) = self.inside(a, e - a);
            let mut rest = Vec::with_capacity(parts.len() + 1);
            for (from, to) in parts {
                if e <= from || a >= to {
                    rest.push((from, to));
                    continue;
                }
                if from < a {
                    rest.push((from, a));
                }
                if e < to {
                    rest.push((e, to));
                }
            }
            parts = rest;
        }
        pending.push(range);
        let taken: usize = parts.iter().map(|&(from, to)| to.saturating_sub(from)).sum();
        BACK_CLASS[usize::from(!self.window)].fetch_add(taken as u64, Ordering::Relaxed);
        let total = self.len as usize;
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let snapshot = before.get_or_insert_with(|| pooled(total));
        if keeps() {
            // Piece by piece, and only the pieces not known current. A
            // piece the part holds whole is current from now on; of a
            // piece it holds partly only the part's bytes are taken, as
            // without the switch, and it is taken again next time (the
            // rest of it may belong to a range declared earlier, whose
            // bytes stay as they were taken).
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if kept.0.is_empty() {
                kept.0 = vec![false; self.pieces(0, total.max(1)).end];
            }
            for (from, to) in parts {
                if from >= to {
                    continue;
                }
                for piece in self.pieces(from, to) {
                    if kept.0[piece] {
                        continue;
                    }
                    let (lo, hi) = self.piece(piece);
                    let whole = from <= lo && hi <= to;
                    let (lo, hi) = (lo.max(from), hi.min(to));
                    // SAFETY: as below; `hi <= total`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(self.data.add(lo), snapshot.as_mut_ptr().add(lo), hi - lo);
                    }
                    kept.0[piece] = whole;
                }
            }
            return;
        }
        if wide_back() {
            // Noted, and taken with the submission's other snapshots when
            // its refreshes are done (`take_snapshots`): the range is
            // declared from here on, so nothing changes its bytes before
            // then. The snapshot's memory stays where it is until the
            // range is done with (`copy_back_done`).
            let mut snaps = SNAPS.lock().unwrap_or_else(|e| e.into_inner());
            let (data, snapshot) = (self.data as usize, snapshot.as_mut_ptr() as usize);
            snaps.extend(parts.into_iter().filter(|&(from, to)| from < to).map(|(from, to)| (data + from, snapshot + from, to - from)));
            return;
        }
        for (from, to) in parts {
            if from < to {
                // SAFETY: the copy's mapping and the snapshot's allocation
                // (`pooled`) both hold `total` bytes, and `to <= total`.
                unsafe {
                    copy_wide(self.data.add(from), snapshot.as_mut_ptr().add(from), to - from);
                }
            }
        }
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written. The range must lie in ranges declared with
    /// [`Shadow::writing`] and not yet done: `Host::execute` passes the
    /// union of a submission's ranges, each byte once, then calls
    /// [`Shadow::copy_back_done`].
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_with(address, len, None)
    }

    /// [`Shadow::copy_back`]; `found`, if given, is the pieces of the range
    /// that differ from the snapshot, found beforehand ([`find_all`]).
    pub(crate) fn copy_back_with(&self, address: usize, len: usize, found: Option<Vec<(usize, usize)>>) -> usize {
        if self.place.is_some() {
            return 0;
        }
        let snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        self.compared(from, to);
        if let Some(before) = &*snapshot
            && from < to
        {
            // SAFETY: `writing` set the snapshot's bytes of every declared
            // range, at the copy's offsets.
            let before = unsafe { std::slice::from_raw_parts(before.as_ptr().add(from), to - from) };
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission (on several threads: gigabytes a frame, bound by
            // memory speed), title memory read only where the GPU wrote.
            #[cfg(feature = "diagnostics")]
            let scanning = std::time::Instant::now();
            let differing = match found {
                Some(found) => found,
                None => self.differing(from, to, before),
            };
            #[cfg(feature = "diagnostics")]
            let syncing = {
                use crate::copystat::{C, add};
                add(C::BackScanUs, scanning.elapsed().as_micros() as u64);
                add(C::BackDiffer, differing.len() as u64);
                std::time::Instant::now()
            };
            if !differing.is_empty() {
                self.by_use(crate::byuse::SHADER_WROTE);
            }
            // Where the copy changes (the GPU wrote, and below the title's
            // bytes come in), a kept snapshot is no longer current.
            for &(at, n) in &differing {
                self.changed(at, at + n);
            }
            // Sized to each stretch read (`back_piece`): a zeroed 64 KiB per
            // range, most of whose ranges differ nowhere, was most of the
            // work for small ranges.
            let (mut title, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
            for (at, n) in differing {
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let read = |first: usize, title: &mut [u8]| crate::host::read_into((start + at + first) as *const u8, title);
                let Some((first, n)) = back_piece(host, old, &mut title, &mut mask, &mut runs, read) else {
                    continue;
                };
                // Title memory as read, beside it with the card's bytes put
                // in: only the dwords that differ are written.
                let was: Vec<u8> = title.iter().zip(&mask).map(|(now, changed)| now ^ changed).collect();
                if !runs.is_empty() {
                    written += put_changed(start + at + first, &was[..n], &title[..n]);
                }
            }
            // Our write shows as a title write at the copy's next refresh,
            // which copies those pages once more (some dozens of MB a
            // frame). Asking about the whole range here, to note them
            // current, cost more: 19-31 askings a frame over 110-174 MB of
            // view memory, each a walk of its guards.
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackSyncUs, syncing.elapsed().as_micros() as u64);
        }
        written
    }

    /// Offsets `[from, to)` of the copy are compared with the snapshot: for
    /// the frame line's counts.
    fn compared(&self, from: usize, to: usize) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackCompared, (to - from) as u64);
        let (bytes, ranges) = if self.window { (2, 5) } else { (3, 6) };
        BACK_CLASS[bytes].fetch_add((to - from) as u64, Ordering::Relaxed);
        BACK_CLASS[ranges].fetch_add(1, Ordering::Relaxed);
        if !self.window && self.len >= 1 << 20 {
            BACK_CLASS[4].fetch_add((to - from) as u64, Ordering::Relaxed);
        }
    }

    /// [`relay`]: the declared range `[address, address + len)` of a copy
    /// in the PC's memory is about to be looked through
    /// (`crate::shadow::find_all`) by the thread that settles: counted as
    /// [`Shadow::copy_back_with`] counts it.
    pub(crate) fn relay_compared(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        self.compared(from, to);
    }

    /// [`relay`], for a copy in the PC's memory: offsets `[at, at + n)`
    /// are a 64 KiB piece of a declared range that differed from the
    /// snapshot when it was looked through. What the card wrote there
    /// since the snapshot was last brought up goes into title memory, from
    /// the first to the last such byte, and into the snapshot, so that the
    /// next flight's merge finds only what the card writes from now on.
    /// The copy is read once and never written: a later flight's shaders
    /// may be writing it at this moment. Under the snapshot's lock, which
    /// [`Shadow::merge_under_writes`] holds while it reads title memory
    /// and puts the title's bytes into the copy and the snapshot: the two
    /// never see each other half done. `now` and `title` are scratch
    /// memory. Returns the bytes written to title memory.
    pub(crate) fn relay_merge(&self, at: usize, n: usize, now: &mut Vec<u8>, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Some(before) = snapshot.as_mut() else {
            return 0;
        };
        now.clear();
        now.reserve(n);
        // SAFETY: inside the copy's mapping (checked above), read into
        // room just reserved; the card may write the mapping meanwhile, and
        // each byte read is one it held at some moment.
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.add(at), now.as_mut_ptr(), n);
            now.set_len(n);
        }
        // SAFETY: the snapshot's allocation holds the copy's length
        // (`writing`), and these offsets are in a range declared and not
        // done with, whose bytes it holds; nothing else touches them
        // without the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(before.as_mut_ptr().add(at), n) };
        let Some(first) = first_difference(now, held) else {
            return 0;
        };
        self.by_use(crate::byuse::SHADER_WROTE);
        let last = last_difference(now, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (now, held) = (&now[first..=last], &mut held[first..=last]);
        title.clear();
        title.resize(n, 0);
        // Title memory not there: looked at again when the next flight
        // that declared the range lands (the snapshot stays as it is).
        if !crate::host::read_into((start + at) as *const u8, title) {
            return 0;
        }
        let was = title.clone();
        if !merge_piece(now, held, title) {
            return 0;
        }
        put_changed(start + at, &was, title)
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory) of
    /// `[from, to)` of the copy that differ from `before` (the bytes from
    /// `from` on), as (offset, bytes) in order; compared on several threads
    /// when the range is big.
    fn differing(&self, from: usize, to: usize, before: &[u8]) -> Vec<(usize, usize)> {
        let (start, data) = (self.start as usize, self.data as usize);
        let scan = |a: usize, b: usize| -> Vec<(usize, usize)> {
            let mut found = Vec::new();
            let mut at = a;
            while at < b {
                let n = (PIECE - (start + at) % PIECE).min(b - at);
                // SAFETY: inside the mapping, which nothing writes while a
                // submission's results are compared.
                let host = unsafe { std::slice::from_raw_parts((data + at) as *const u8, n) };
                if host != &before[at - from..at - from + n] {
                    found.push((at, n));
                }
                at += n;
            }
            found
        };
        const PER_THREAD: usize = 16 << 20;
        let threads = ((to - from) / PER_THREAD).clamp(1, 8);
        if threads == 1 {
            return scan(from, to);
        }
        // Cut on piece boundaries, so that every thread sees whole pieces.
        let cut = |i: usize| {
            if i == threads {
                return to;
            }
            let at = from + (to - from) / threads * i;
            // (saturating: near the copy's start, the piece `at` lies in may
            // begin before the copy does; it wrapped and the first slice was never compared)
            at.saturating_sub((start + at) % PIECE).max(from)
        };
        let scan = &scan;
        std::thread::scope(|s| {
            let parts: Vec<_> = (0..threads)
                .map(|i| {
                    let (a, b) = (cut(i), cut(i + 1));
                    s.spawn(move || scan(a, b))
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|part| match part.join() {
                    Ok(found) => found,
                    Err(panic) => std::panic::resume_unwind(panic),
                })
                .collect()
        })
    }

    /// After [`Shadow::copy_back`] of a submission's ranges: every submitted
    /// step that declared one of `ranges` (`(address, end)`, as declared) is
    /// done with it (one submission runs at a time), and the snapshot with
    /// the last range.
    ///
    /// With [`relay`] several submissions may hold a range: `ranges` are
    /// those of one of them (each once), and a range is done with when the
    /// last submission that declared it is ([`Flights`]).
    pub(crate) fn copy_back_done(&self, ranges: &[(usize, usize)]) {
        if self.place.is_some() {
            return;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if relay() {
            let free = self.flights.lock().unwrap_or_else(|e| e.into_inner()).done(ranges);
            pending.retain(|r| free.binary_search(r).is_err());
        } else {
            pending.retain(|r| !ranges.contains(r));
        }
        if !pending.is_empty() || self.card.is_some() {
            return;
        }
        if keeps() {
            // Kept for the next submission, if it is worth it (a megabyte
            // or more) and there is room.
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if !kept.1 && self.len >= 1 << 20 {
                if KEPT_BYTES.fetch_add(self.len, Ordering::Relaxed) + self.len <= kept_most() {
                    kept.1 = true;
                } else {
                    KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
                }
            }
            if kept.1 {
                return;
            }
            // Given back: nothing of it is current next time. (The lock
            // on the pieces ends here: `copy_back` takes it while it holds
            // the snapshot's.)
            kept.0.clear();
        }
        if let Some(bytes) = self.before.lock().unwrap_or_else(|e| e.into_inner()).take() {
            give_back(bytes);
        }
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
    /// [`Shadow::differing`] cuts them) that hold offsets `[from, to)` of
    /// the copy; `from < to`.
    fn pieces(&self, from: usize, to: usize) -> std::ops::Range<usize> {
        pieces_of(self.start as usize, from, to)
    }

    /// The offsets in the copy that piece `piece` holds.
    fn piece(&self, piece: usize) -> (usize, usize) {
        piece_at(self.start as usize, self.len as usize, piece)
    }

    /// The copy's bytes at offsets `[from, to)` change: a kept snapshot is
    /// not current there any more.
    fn changed(&self, from: usize, to: usize) {
        if !keeps() || from >= to {
            return;
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.0.is_empty() {
            return;
        }
        for piece in self.pieces(from, to) {
            kept.0[piece] = false;
        }
    }

    /// Whether shaders of a submission not yet copied back may be writing
    /// the copy.
    pub(crate) fn writes_pending(&self) -> bool {
        !self.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Whether one of those submissions may be writing in title memory
    /// `[from, to)`.
    pub(crate) fn writes_pending_in(&self, from: usize, to: usize) -> bool {
        self.pending.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|&(a, e)| a < to && e > from)
    }

    /// Whether [`Shadow::update`] of `[address, address + len)` would leave
    /// title bytes behind: a range shaders may be writing holds a page the
    /// title wrote since the copy last took it. The answer comes from the
    /// write-watch journal alone, as the submission's one asking left it
    /// (`crate::watch::asked`); where the journal cannot say (memory not
    /// write-watched, a range not asked about) it is yes. A page the copy
    /// never took whole counts as written if the title ever wrote it.
    ///
    /// For submissions sent to the card without a wait
    /// (`crate::host::late`): the ranges are those of flights in the air,
    /// which are then settled before this refresh.
    pub(crate) fn stale_under_writes(&self, address: usize, len: usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        busy.into_iter().any(|(a, e)| match crate::watch::asked(a, e) {
            Some((_, written)) => !stale_pages_in(base, &copied, a, e, &written).is_empty(),
            None => true,
        })
    }

    /// Where [`Shadow::stale_under_writes`] says yes: puts the bytes the
    /// title changed into the copy, in the ranges shaders may be writing,
    /// and only those bytes, so that what the shaders write beside them
    /// stays (`crate::host::late_exact`). For each page of such a range
    /// that the title wrote since the copy last took it, title memory as
    /// it is now is compared with what the copy held when the range was
    /// given to the shaders: the copy's CPU side for a copy in the card's
    /// memory (the changed runs go up before the next submission, each a
    /// copy of its own bytes, `runs_left` of them at most), the snapshot
    /// for the others (the changed bytes go into the copy and into the
    /// snapshot, so the compare after the card ran takes them for the
    /// title's and not the shaders').
    ///
    /// Whether the caller must still settle the flights: the journal
    /// cannot say what was written, a snapshot is missing, or `runs_left`
    /// ran out. Pages done before that stay done.
    pub(crate) fn merge_under_writes(&self, address: usize, len: usize, runs_left: &mut usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            note_stale(0);
            return true;
        }
        // The declared ranges overlap (views of the title's whole heap from
        // dozens of offsets): as one range each page is asked about and
        // compared once. Every byte of it is in a declared range, and of a
        // copy in the PC's memory in the snapshot.
        let busy = if crate::watch::light() { joined(busy) } else { busy };
        let base = start / PAGE;
        UNDER_NS[9].fetch_add(1, Ordering::Relaxed);
        for (a, e) in busy {
            let t = std::time::Instant::now();
            // As the refresh itself asks (`update_free`): the journal if
            // the submission's asking covered the range, else the kernel,
            // else the guards. The journal alone said nothing for copies
            // of view memory, which is guarded and not write-watched, and
            // those were all of run 99's cases: the flights were settled
            // 4-5 times a frame for "0 pages looked at".
            let Some((now, written)) = crate::watch::writes(a, e, true) else {
                note_stale(1);
                return true;
            };
            let t = under_ns(0, t);
            let stale: Vec<usize> = {
                let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                stale_pages_in(base, &copied, a, e, &written)
            };
            UNDER_NS[8].fetch_add(stale.len() as u64, Ordering::Relaxed);
            let t = under_ns(1, t);
            if stale.is_empty() {
                continue;
            }
            // The lock of the copy's reference (the snapshot, or the
            // CPU's side of a copy in the card's memory), from before
            // title memory is read until the title's bytes are in: with
            // the copy-back on its own thread ([`relay`]) that thread
            // puts what the card wrote into title memory and into the
            // reference under the same lock, and a page read before
            // such a merge and compared after it would take the card's
            // bytes for the title's.
            let merging = self.merging.write().unwrap_or_else(|e| e.into_inner());
            let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
            let held = if self.card.is_some() {
                self.data as usize
            } else {
                before.as_mut().map_or(0, |snapshot| snapshot.as_mut_ptr() as usize)
            };
            if held == 0 {
                note_stale(3);
                return true;
            }
            let t = under_ns(2, t);
            // A page's part in the range, compared and put in: the runs it
            // changed; none for a page that is not there (left as it is, and
            // looked at again next time) or that had too many.
            type Done = Option<Vec<(usize, usize)>>;
            let left = std::sync::atomic::AtomicI64::new(*runs_left as i64);
            let over = std::sync::atomic::AtomicBool::new(false);
            let done: Vec<Mutex<Done>> = stale.iter().map(|_| Mutex::new(None)).collect();
            let page_of = |i: usize| -> (usize, usize) { ((stale[i] * PAGE).max(a), ((stale[i] + 1) * PAGE).min(e)) };
            let one = |i: usize| {
                let (lo, hi) = page_of(i);
                if lo < hi && !over.load(Ordering::Relaxed) {
                    // SAFETY: the reference holds the copy's bytes at the
                    // copy's offsets, and these are in a declared range; the
                    // locks held above keep out what else writes them, and
                    // no two pages are the same.
                    let runs = unsafe { self.merge_page_under_writes(lo, hi - lo, held as *mut u8, &left) };
                    match runs {
                        Ok(runs) => *done[i].lock().unwrap_or_else(|e| e.into_inner()) = runs,
                        Err(()) => over.store(true, Ordering::Relaxed),
                    }
                }
            };
            // (Not on the threads of `wide`: the thread that waits there
            // takes whatever work comes, and a landing's takes the lock.)
            if stale.len() > 2 && under_wide() {
                crate::land::each_under(stale.len(), &one);
            } else {
                (0..stale.len()).for_each(one);
            }
            let t = under_ns(3, t);
            *runs_left = left.into_inner().max(0) as usize;
            let mut taken = Vec::new();
            // The runs of all the pages, noted at once (`card.dirty` is
            // sorted and joined when it is taken).
            let mut changed: Vec<(usize, usize)> = Vec::new();
            let (mut runs_in, mut bytes_in) = (0u64, 0u64);
            for (i, runs) in done.into_iter().enumerate() {
                let Some(runs) = runs.into_inner().unwrap_or_else(|e| e.into_inner()) else {
                    continue;
                };
                let (lo, hi) = ((stale[i] * PAGE).max(a), ((stale[i] + 1) * PAGE).min(e));
                runs_in += runs.len() as u64;
                bytes_in += runs.iter().map(|r| r.1 as u64).sum::<u64>();
                if self.card.is_some() {
                    changed.extend(runs.into_iter().map(|(r, m)| (lo - start + r, lo - start + r + m)));
                }
                // Taken, if the range holds all of the page that is in the
                // copy; else the rest of it is the refresh's, which notes
                // only whole pages, and the page is looked at again.
                if lo < hi && (stale[i] * PAGE).max(start) >= a && ((stale[i] + 1) * PAGE).min(end) <= e {
                    taken.push(stale[i]);
                }
            }
            MERGED[1].fetch_add(runs_in, Ordering::Relaxed);
            MERGED[2].fetch_add(bytes_in, Ordering::Relaxed);
            self.card_changed_runs(changed.into_iter());
            drop(before);
            drop(merging);
            {
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in taken {
                    if let Some(at) = copied.get_mut(page - base) {
                        *at = now;
                    }
                }
            }
            under_ns(4, t);
            if over.into_inner() {
                note_stale(2);
                return true;
            }
        }
        false
    }

    /// Of [`Shadow::merge_under_writes`]: title memory `[lo, lo + n)`, in one
    /// page, compared with the reference, which begins at `held`, and put in.
    /// The runs that changed, none if it cannot be read, and an error if
    /// there were more to copy one by one than `left`.
    ///
    /// # Safety
    /// The reference holds the copy's length, and nothing else writes these
    /// bytes of it during the call.
    unsafe fn merge_page_under_writes(
        &self,
        lo: usize,
        n: usize,
        held: *mut u8,
        left: &std::sync::atomic::AtomicI64,
    ) -> Result<Option<Vec<(usize, usize)>>, ()> {
        thread_local! {
            static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        TITLE.with_borrow_mut(|title| {
            title.resize(PAGE, 0);
            let title = &mut title[..n];
            let t = std::time::Instant::now();
            if !crate::host::read_into(lo as *const u8, title) {
                return Ok(None);
            }
            let t = under_ns(5, t);
            let _compared = Compared(t);
            MERGED[0].fetch_add(1, Ordering::Relaxed);
            let at = lo - self.start as usize;
            // SAFETY: the caller's.
            let held = unsafe { std::slice::from_raw_parts_mut(held.add(at), n) };
            if self.card.is_some() {
                // Whole words go up as pairs, all of them in one
                // dispatch: only what is a copy of its own counts.
                // Compared and put in, in one pass ([`under_fused`]), where
                // the most copies of their own the page can have are left
                // to take: they are taken first, and what is not used is
                // given back. (Whole words at word offsets of the copy: a
                // copy of its own is a run of more than 256 bytes.)
                let most = (n / 260 + 1) as i64;
                let fused = crate::host::words_under()
                    && under_fused()
                    && !crate::host::upcheck::on()
                    && at.is_multiple_of(4)
                    && lo.is_multiple_of(4)
                    && n.is_multiple_of(4)
                    && {
                        if left.fetch_sub(most, Ordering::Relaxed) >= most {
                            true
                        } else {
                            left.fetch_add(most, Ordering::Relaxed);
                            false
                        }
                    };
                let runs = if fused {
                    word_runs_put(lo, title, held)
                } else if crate::host::words_under() {
                    word_runs(lo, title, held)
                } else {
                    changed_runs(title, held)
                };
                let own = |&&(r, m): &&(usize, usize)| !crate::host::words_under() || !crate::host::goes_as_words(at + r, m);
                let copies = runs.iter().filter(own).count() as i64;
                if fused {
                    debug_assert!(copies <= most);
                    left.fetch_add(most - copies, Ordering::Relaxed);
                    return Ok(Some(runs));
                }
                if copies > 0 && left.fetch_sub(copies, Ordering::Relaxed) < copies {
                    return Err(());
                }
                let was = (!runs.is_empty() && crate::host::upcheck::on()).then(|| held.to_vec());
                if !fused {
                    for &(r, m) in &runs {
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                    }
                }
                if let Some(was) = was {
                    crate::host::upcheck::note(self.id, at, &was, held);
                }
                return Ok(Some(runs));
            }
            let runs = changed_runs(title, held);
            for &(r, m) in &runs {
                // SAFETY: inside the copy's mapping. The card may be
                // writing the copy beside these bytes; they are
                // written here and nothing around them.
                unsafe { std::ptr::copy_nonoverlapping(title.as_ptr().add(r), self.data.add(at + r), m) };
                held[r..r + m].copy_from_slice(&title[r..r + m]);
            }
            Ok(Some(runs))
        })
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        // `crate::census`: the copy takes what other copies wrote back here.
        if crate::census::tracking() {
            use crate::census::Taker;
            let kind = match (self.place.is_some(), self.window) {
                (true, _) => Taker::InPlace,
                (false, true) => Taker::Window,
                (false, false) => Taker::Copy,
            };
            let (start, end) = (self.start as usize, (self.start + self.len) as usize);
            crate::census::asked(self.id, kind, address.max(start), address.saturating_add(len).min(end));
        }
        // In place: the card reads title memory itself.
        if self.place.is_some() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        // And the pieces whose shader writes the card still has alone
        // (`crate::lazy_back`), asked after the ranges above: a piece is
        // held before the flight that wrote it is done with its range. The
        // CPU's side there is what the merge compares the card's with.
        if crate::lazy_back::active() {
            busy.extend(crate::lazy_back::held(self, from, to));
        }
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// Whether [`Shadow::update`] of `[address, address + len)` would copy
    /// nothing, and knows it without asking anything
    /// (`crate::watch::nothing_new`). `false` if it would have to ask.
    pub(crate) fn nothing_new(&self, address: usize, len: usize) -> bool {
        if self.place.is_some() {
            return true;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return true;
        }
        if !matches!(self.watched.load(Ordering::Relaxed), 1 | GUARDED) {
            return false;
        }
        // Around ranges shaders are writing the refresh asks gap by gap.
        if self.pending.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|&(a, e)| a < to && e > from) {
            return false;
        }
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let same = crate::watch::nothing_new(from, to, start / PAGE, &copied, self.ranged(from, to));
        #[cfg(feature = "diagnostics")]
        if same {
            use crate::copystat::{C, add};
            add(C::Checked, (to - from) as u64);
            add(C::Unchanged, (to - from) as u64);
        }
        same
    }

    /// The latest asking after which a range that holds `[from, to)` was
    /// taken whole (0: none is known). The caller holds `copied`'s lock.
    fn ranged(&self, from: usize, to: usize) -> u64 {
        let ranged = self.ranged.lock().unwrap_or_else(|e| e.into_inner());
        ranged.iter().filter(|r| r.0 <= from && to <= r.1).map(|r| r.2).max().unwrap_or(0)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                // (A range whose blocks rest after a write has no guard now.)
                let guarded = crate::watch::guarded(from, to) || self.watched.load(Ordering::Relaxed) == GUARDED;
                self.watched.store(if guarded { GUARDED } else { 1 }, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                self.refresh_check(from, to);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        if let Some(uses) = &self.uses {
            uses.taken_whole(self.window, self.by_use_place());
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WholeBytes, (to - from) as u64);
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::whole(from, to - from, started.elapsed().as_micros() as u64);
        to - from
    }

    /// Diagnostic (`VICTORY_REFRESH_CHECK=1`; off by default): after a
    /// refresh of `[from, to)` that trusted the write watch or the guards,
    /// the first 256 bytes of every 64 KiB block of it are read from title
    /// memory and compared with the copy. A block that differs on three
    /// refreshes in a row was changed with nothing to tell of it (a write
    /// through another view of its pages, as DirectStorage's page
    /// destinations make) and the card reads old bytes there: said with
    /// the copy and the address. One that differs once is a write of the
    /// title's that arrived between the refresh and the look.
    fn refresh_check(&self, from: usize, to: usize) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_CHECK").is_some_and(|v| v == "1")) {
            return;
        }
        const SAMPLE: usize = 256;
        // Blocks found differing: how many refreshes in a row.
        static DIFFERING: Mutex<Option<std::collections::HashMap<usize, u32>>> = Mutex::new(None);
        static SAID: AtomicU64 = AtomicU64::new(0);
        let start = self.start as usize;
        let mut title = [0u8; SAMPLE];
        let mut at = from;
        while at < to {
            let n = SAMPLE.min(to - at);
            if crate::host::read_into(at as *const u8, &mut title[..n]) {
                // SAFETY: `[from, to)` lies inside the buffer (`update`),
                // so these bytes are inside the copy's mapping.
                let copy = unsafe { std::slice::from_raw_parts(self.data.add(at - start).cast_const(), n) };
                let mut map = DIFFERING.lock().unwrap_or_else(|e| e.into_inner());
                let map = map.get_or_insert_with(Default::default);
                if copy == &title[..n] {
                    map.remove(&at);
                } else {
                    let times = map.entry(at).or_insert(0);
                    *times += 1;
                    if *times == 3 {
                        let said = SAID.fetch_add(1, Ordering::Relaxed) + 1;
                        if said <= 64 || said.is_power_of_two() {
                            tracing::warn!(
                                said,
                                copy = self.id,
                                copy_start = format_args!("{start:#x}"),
                                copy_bytes = self.len,
                                address = format_args!("{at:#x}"),
                                refreshed = format_args!("{from:#x}..{to:#x}"),
                                on_the_card = self.card.is_some(),
                                window = self.window,
                                "a host copy of title memory is stale after its refresh, three times in a row:                                  the memory changed with nothing to tell of it (VICTORY_REFRESH_CHECK=1)"
                            );
                        }
                    }
                }
            }
            // The first bytes of the next 64 KiB block.
            at = (at / PAGE + 1) * PAGE;
        }
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        // Pages the range holds in part are never noted in `copied`, and
        // were copied again at every refresh of the range, written or not:
        // of the heap's buffer views alone two ends for each of 4,400
        // copies, every frame. A range that was taken whole, and not written
        // since, is as it was.
        let ranged = if crate::watch::light() { self.ranged(from, to) } else { 0 };
        let (mut runs, mut run) = (Vec::new(), None::<(usize, bool)>);
        // Pages taken before and written since: in the order of their runs.
        let mut again_at: Vec<(usize, u64)> = Vec::new();
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            let taken = (*at).max(ranged);
            if taken == 0 || write > taken {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                // Cut where pages taken before meet pages never taken.
                let again = taken != 0;
                if again {
                    again_at.push((page, taken));
                }
                if let Some((first, known)) = run
                    && known != again
                {
                    runs.push(((first * PAGE).max(from), (page * PAGE).min(to), known));
                    run = None;
                }
                run.get_or_insert((page, again));
            } else if let Some((first, known)) = run.take() {
                runs.push(((first * PAGE).max(from), (page * PAGE).min(to), known));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some((first, known)) = run {
            runs.push(((first * PAGE).max(from), ((last + 1) * PAGE).min(to), known));
        }
        if !again_at.is_empty() {
            self.by_use(crate::byuse::CPU_WROTE);
        }
        let runs = by_parts(runs, &again_at, &crate::watch::parts_written(&again_at));
        let bytes = self.take_all(runs);
        if crate::watch::light() {
            // All of the range is in the copy as of this asking.
            let mut ranged = self.ranged.lock().unwrap_or_else(|e| e.into_inner());
            ranged[0] = (from, to, now);
            if to - from >= ranged[1].1 - ranged[1].0 {
                ranged[1] = (from, to, now);
            }
        }
        bytes
    }

    /// [`Shadow::take`] of `runs` (from, to, known), on several threads if
    /// they are long: the big heap's copy has some 4 MB of them in each of
    /// 14 refreshes a frame, on one thread while the others had nothing.
    fn take_all(&self, runs: Vec<(usize, usize, bool)>) -> usize {
        const LONG: usize = 1 << 20;
        const PART: usize = 256 << 10;
        let bytes = runs.iter().map(|r| r.1 - r.0).sum();
        if bytes < LONG || !crate::watch::light() {
            for (from, to, known) in runs {
                self.take(from, to, known);
            }
            return bytes;
        }
        let mut parts = Vec::with_capacity(bytes / PART + runs.len());
        for (from, to, known) in runs {
            let mut at = from;
            while at < to {
                // Whole pages to a part.
                let next = ((at / PAGE) * PAGE + PART).min(to);
                parts.push((at, next, known));
                at = next;
            }
        }
        let this = std::ptr::from_ref(self) as usize;
        wide_each(parts.len(), move |i| {
            // SAFETY: `wide_each` returns when every part is done, before
            // the borrow of `self` ends. The parts lie apart.
            let this = unsafe { &*(this as *const Shadow) };
            let (from, to, known) = parts[i];
            this.take(from, to, known);
        });
        bytes
    }

    /// What shaders wrote in `[from, to)` (offsets) was put into title memory.
    fn note_landing(&self, from: usize, to: usize) {
        let first = self.start as usize / KERNEL_PAGE;
        let pages = (self.start as usize + from) / KERNEL_PAGE - first..(self.start as usize + to).div_ceil(KERNEL_PAGE) - first;
        let mut bits = self.landings.lock().unwrap_or_else(|e| e.into_inner());
        if bits.is_empty() {
            let all = (self.start + self.len).div_ceil(KERNEL_PAGE as u64) as usize - first;
            bits.resize(all.div_ceil(64), 0);
        }
        for page in pages {
            bits[page / 64] |= 1 << (page % 64);
        }
        self.landing.store(true, Ordering::Release);
    }

    /// [`Shadow::take`] of `[from, to)`, which the copy held before and which
    /// is written since: where it may be only the landing that wrote it, the
    /// 4 KiB that title memory and the CPU's side have alike are left as they
    /// are, and nothing of them is sent ([`land_echo`]).
    fn take_landed(&self, from: usize, to: usize) -> usize {
        thread_local! {
            static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        // Over shaders' bytes left on the card (`VICTORY_SKIP_WC_BACK`) a
        // differing 4 KiB here would go up whole, title memory's old bytes
        // with it: [`Shadow::copy`] sends only the title's changes there.
        if !crate::census::stale_pages(self.id, from, to).is_empty() {
            return self.copy(from, to);
        }
        let start = self.start as usize;
        let first = start / KERNEL_PAGE;
        // Runs of pages noted, and not noted (any more).
        let mut runs: Vec<(usize, usize, bool)> = Vec::new();
        {
            let mut bits = self.landings.lock().unwrap_or_else(|e| e.into_inner());
            let mut at = from;
            while at < to {
                let next = ((at / KERNEL_PAGE + 1) * KERNEL_PAGE).min(to);
                let page = at / KERNEL_PAGE - first;
                let noted = bits.get(page / 64).is_some_and(|word| word >> (page % 64) & 1 != 0);
                // (A page looked at in part is looked at again.)
                if noted && next - at == KERNEL_PAGE {
                    bits[page / 64] &= !(1 << (page % 64));
                }
                match runs.last_mut() {
                    Some(last) if last.2 == noted && next - last.0 <= PAGE => last.1 = next,
                    _ => runs.push((at, next, noted)),
                }
                at = next;
            }
        }
        TITLE.with_borrow_mut(|title| {
            for (from, to, noted) in runs {
                if !noted {
                    self.copy(from, to);
                    continue;
                }
                title.resize(to - from, 0);
                // (Pages that are not there read as zeros.)
                crate::host::copy_mapped(title.as_mut_ptr(), from as *const u8, to - from, false);
                let mut at = from;
                while at < to {
                    let next = ((at / KERNEL_PAGE + 1) * KERNEL_PAGE).min(to);
                    let title = &title[at - from..next - from];
                    // SAFETY: inside the mapping of the copy's CPU side, which
                    // nothing else reads or writes there: no shader's range
                    // holds it, and the parts of a refresh lie apart.
                    let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at - start), next - at) };
                    if title == held {
                        ECHO[0].fetch_add((next - at) as u64, Ordering::Relaxed);
                    } else {
                        ECHO[1].fetch_add((next - at) as u64, Ordering::Relaxed);
                        held.copy_from_slice(title);
                        self.changed(at - start, next - start);
                        self.card_changed(at - start, next - start);
                    }
                    at = next;
                }
            }
        });
        to - from
    }

    /// Copies title memory `[from, to)` (inside the buffer) into the copy;
    /// `known`: the copy held it before ([`send_changed`]).
    fn take(&self, from: usize, to: usize, known: bool) -> usize {
        thread_local! {
            /// A page of title memory, as read.
            static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let Some(card) = self.card.as_ref().filter(|_| self.changes_only.load(Ordering::Relaxed) || send_changed()) else {
            if known && self.card.is_some() && self.landing.load(Ordering::Acquire) && land_echo() && !crate::host::upcheck::on() {
                return self.take_landed(from, to);
            }
            return self.copy(from, to);
        };
        let whole = !known && card.before.lock().unwrap_or_else(|e| e.into_inner()).is_none();
        let start = self.start as usize;
        TITLE.with_borrow_mut(|title| {
            title.resize(PAGE, 0);
            let mut at = from;
            while at < to {
                let n = (PAGE - at % PAGE).min(to - at);
                let title = &mut title[..n];
                // (Pages that are not there read as zeros.)
                crate::host::copy_mapped(title.as_mut_ptr(), at as *const u8, n, false);
                // SAFETY: inside the mapping of the copy's CPU side, which
                // nothing else reads or writes there (the card copies from
                // it): no shader's range holds it, and the parts of a
                // refresh lie apart.
                let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at - start), n) };
                let was = crate::host::upcheck::on().then(|| held.to_vec());
                let runs = if whole { vec![(0, n)] } else { dword_runs(at, held, title, 0) };
                for &(a, len) in &runs {
                    held[a..a + len].copy_from_slice(&title[a..a + len]);
                    self.changed(at - start + a, at - start + a + len);
                }
                self.card_changed_runs(runs.iter().map(|&(a, len)| (at - start + a, at - start + a + len)));
                if let Some(was) = was {
                    crate::host::upcheck::note(self.id, at - start, &was, held);
                }
                SEND[0].fetch_add(n as u64, Ordering::Relaxed);
                if runs.is_empty() {
                    SEND[1].fetch_add(n as u64, Ordering::Relaxed);
                }
                SEND[2].fetch_add(runs.iter().map(|r| r.1 as u64).sum(), Ordering::Relaxed);
                SEND[3].fetch_add(runs.len() as u64, Ordering::Relaxed);
                at += n;
            }
        });
        to - from
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    ///
    /// `VICTORY_SKIP_WC_BACK` (`crate::census::skip_wc_back`): on a page
    /// where shaders' bytes were left on the card, title memory and the
    /// CPU's side both lack them, and title memory differs from the CPU's
    /// side only where the title wrote since. Only those bytes go in and
    /// up; the card keeps the shaders' bytes beside them (taken whole, the
    /// page would put title memory's old bytes over them).
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if self.card.is_some() {
            let stale = crate::census::stale_pages(self.id, from, to);
            if !stale.is_empty() {
                let mut at = from;
                for page in stale {
                    let (lo, hi) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
                    if at < lo {
                        self.copy_whole(at, lo);
                    }
                    self.copy_title_changes(lo, hi);
                    at = hi;
                }
                if at < to {
                    self.copy_whole(at, to);
                }
                return to - from;
            }
        }
        self.copy_whole(from, to)
    }

    /// [`Shadow::copy`] of a stale page's part `[from, to)` (inside the
    /// buffer): the runs where title memory differs from the CPU's side.
    fn copy_title_changes(&self, from: usize, to: usize) {
        let start = self.start as usize;
        let mut title = vec![0u8; to - from];
        crate::host::copy_mapped(title.as_mut_ptr(), from as *const u8, to - from, false);
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(from - start), to - from) };
        let mut up = 0;
        for (r, m) in changed_runs(&title, held) {
            held[r..r + m].copy_from_slice(&title[r..r + m]);
            let (a, b) = (from - start + r, from - start + r + m);
            self.changed(a, b);
            self.card_changed(a, b);
            up += m;
        }
        if up > 0 {
            crate::census::title_over(up);
        }
    }

    /// [`Shadow::copy`] of `[from, to)` (inside the buffer), whole.
    fn copy_whole(&self, from: usize, to: usize) -> usize {
        let start = self.start as usize;
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        if self.card.is_some() && from < to && crate::host::upcheck::on() {
            for_pieces_of(from - start, to - start, |a, b| {
                let was = self.card_held(a, b - a);
                // SAFETY: inside the mapping, as above.
                crate::host::copy_mapped(unsafe { self.data.add(a) }, (start + a) as *const u8, b - a, false);
                crate::host::upcheck::note(self.id, a, &was, &self.card_held(a, b - a));
            });
        } else {
            crate::host::copy_mapped(at, from as *const u8, to - from, false);
        }
        // After the bytes are in: a snapshot taken meanwhile is not current.
        self.changed(from - start, to - start);
        self.card_changed(from - start, to - start);
        if self.card.is_some() {
            crate::census::recopied(self.id, from, to);
        }
        if from < to && crate::windowskip::mode() == crate::windowskip::SkipSame::Sample {
            self.window_sample(from, to);
        }
        to - from
    }

    /// `VICTORY_WINDOW_SKIP_SAME=sample` (after kitty b93b839d5): of title
    /// memory `[from, to)` a window with no card side has just copied whole,
    /// one 4 KiB page in 16 is read again and compared with the window's
    /// prints, for the frame line (`crate::windowskip::sample`). Nothing is
    /// left out. Only a window in the PC's memory or in mapped card memory,
    /// not in place; one shaders were declared to write counts as nothing to
    /// compare with (its buffer may hold their bytes).
    fn window_sample(&self, from: usize, to: usize) {
        if !self.window || self.card.is_some() || self.place.is_some() {
            return;
        }
        if self.written() {
            crate::windowskip::unknown(to - from);
            crate::windowskip::taken(to - from, 0);
            return;
        }
        let window = (self.start as usize, (self.start + self.len) as usize);
        let mut tally = [0u64; 3];
        {
            let mut prints = self.prints.lock().unwrap_or_else(|e| e.into_inner());
            if prints.is_empty() {
                let n = crate::windowskip::prints_for(window.0, window.1);
                prints.resize(n, 0);
                crate::windowskip::prints_made(n);
            }
            crate::windowskip::sample(window, (from, to), &mut prints, &mut tally, |at, into| {
                crate::host::read_into(at as *const u8, into)
            });
        }
        crate::windowskip::tally_add(&tally);
        crate::windowskip::taken(to - from, 0);
    }

    /// Whether shaders read and write this copy in the card's own memory
    /// ([`cards`]).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// Shaders store in it what is not declared: from here on only the dwords
    /// the title changes go up ([`by_address`]).
    pub(crate) fn keep_stores(&self) {
        self.changes_only.store(true, Ordering::Relaxed);
    }

    /// Offsets `[from, to)` go up again whole, as the CPU's side has them.
    pub(crate) fn resend(&self, from: usize, to: usize) {
        self.card_changed(from, to);
    }

    /// The CPU side's bytes at offsets `[from, to)` changed: they go to the
    /// card's side before the next submission.
    fn card_changed(&self, from: usize, to: usize) {
        self.card_changed_runs(std::iter::once((from, to)));
    }

    /// [`Shadow::card_changed`] of each of `runs`.
    fn card_changed_runs(&self, runs: impl Iterator<Item = (usize, usize)>) {
        let Some(card) = &self.card else {
            return;
        };
        let most = runs.size_hint().0;
        let mut runs = runs.filter(|r| r.0 < r.1).peekable();
        if runs.peek().is_none() {
            return;
        }
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        dirty.reserve(most);
        for (from, to) in runs {
            match dirty.last_mut() {
                Some(last) if from <= last.1 && to >= last.0 => *last = (last.0.min(from), last.1.max(to)),
                _ => dirty.push((from, to)),
            }
        }
        // (Under the lock: what takes the list sees it or finds it set.)
        match card.me.get() {
            // With `dirty_list`: into the list when it was not set.
            Some(me) => {
                if !card.touched.swap(true, Ordering::AcqRel) {
                    TOUCHED.lock().unwrap_or_else(|e| e.into_inner()).push(me.clone());
                }
            }
            None => card.touched.store(true, Ordering::Release),
        }
    }

    /// Whether shaders were ever declared to write it.
    pub(crate) fn written(&self) -> bool {
        self.written.load(Ordering::Relaxed)
    }

    /// `bytes` were put into title memory at `address`, and the submission
    /// being prepared copies the same to the copy at that copy's place in its
    /// list: the reference (the CPU's side of a copy in the card's memory, the
    /// snapshot) takes them and nothing is sent. What the card makes of them
    /// is then what differs from them when the submission has run. The range
    /// is declared written by it, and its sends are taken.
    pub(crate) fn placed_copy(&self, address: usize, bytes: &[u8]) {
        let (from, to) = self.inside(address, bytes.len());
        if from >= to || self.place.is_some() {
            return;
        }
        let bytes = &bytes[self.start as usize + from - address..][..to - from];
        let _merging = self.merging.write().unwrap_or_else(|e| e.into_inner());
        if self.card.is_some() {
            // SAFETY: inside the mapping of the CPU's side, in a range no
            // refresh writes; the lock keeps the landing out.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(from), bytes.len()) };
        } else if let Some(snapshot) = self.before.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            // SAFETY: the snapshot's allocation holds the copy's length.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), snapshot.as_mut_ptr().add(from), bytes.len()) };
        }
        self.changed(from, to);
    }

    /// Of a copy in the card's memory: where its CPU side is.
    pub(crate) fn card_data(&self) -> *const u8 {
        self.data
    }

    /// Diagnostics: of a copy in the card's memory, what its CPU side holds
    /// at offsets `[at, at + len)` (cut to the copy).
    pub(crate) fn card_held(&self, at: usize, len: usize) -> Vec<u8> {
        let to = (at + len).min(self.len as usize);
        if self.card.is_none() || at >= to {
            return Vec::new();
        }
        // SAFETY: inside the mapping; the caller holds the submission lock,
        // under which it is written but for what is being landed.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast_const(), to - at) }.to_vec()
    }

    /// Diagnostics: the offsets whose bytes have not gone to the card's side
    /// yet.
    pub(crate) fn card_dirty_now(&self) -> Vec<(usize, usize)> {
        self.card.as_ref().map_or_else(Vec::new, |c| c.dirty.lock().unwrap_or_else(|e| e.into_inner()).clone())
    }

    /// Of a copy in the card's memory: the CPU's side (PC memory), the
    /// card side's bytes from before if made yet, and their size.
    pub(crate) fn card_sides(&self) -> Option<(&ID3D12Resource, Option<ID3D12Resource>, u64)> {
        let card = self.card.as_ref()?;
        let before = card.before.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Some((&card.staging, before, card.size))
    }

    /// Shader writes are declared for a copy in the card's memory: makes
    /// its "before" twin with `make` (bytes to buffer) if it has none.
    /// Whether one was made now: the caller fills it, whole, from the
    /// card's side before the submission's lists run. Without one (no
    /// memory) every piece of a declared range is fetched.
    pub(crate) fn card_make_before(&self, make: impl FnOnce(u64) -> Option<ID3D12Resource>) -> bool {
        let Some(card) = &self.card else {
            return false;
        };
        let mut before = card.before.lock().unwrap_or_else(|e| e.into_inner());
        if before.is_some() {
            return false;
        }
        *before = make(card.size);
        if before.is_some() {
            CARD_BYTES[1].fetch_add(card_hash_bytes(card.size), Ordering::Relaxed);
        }
        before.is_some()
    }

    /// The declared range `[address, end)` as the compare on the card
    /// takes it; `None` if none of it lies in the copy.
    pub(crate) fn card_span(&self, address: usize, end: usize) -> Option<CardSpan> {
        let (from, to) = self.inside(address, end.saturating_sub(address));
        (from < to).then(|| card_span_of(self.start as usize, from, to))
    }

    /// The pieces that hold offsets `[from, to)` of the copy (cut to it),
    /// as the card's hashing takes them.
    pub(crate) fn card_run(&self, from: usize, to: usize) -> CardSpan {
        let to = to.min(self.len as usize);
        card_span_of(self.start as usize, from.min(to.saturating_sub(1)), to)
    }

    /// The offsets of piece `i` of `span` that lie in the span.
    pub(crate) fn card_piece(&self, span: &CardSpan, i: u32) -> (usize, usize) {
        card_piece_of(self.start as usize, self.len as usize, span, i)
    }

    /// The piece that holds offset `at`.
    pub(crate) fn card_piece_of(&self, at: usize) -> u32 {
        self.pieces(at, at + 1).start as u32
    }

    /// The offsets the compare on the card hashes of a piece: what lies of
    /// it in the card's buffer (which may be longer than the copy), widened
    /// to 16 bytes.
    pub(crate) fn card_hashed(&self, piece: u32) -> (usize, usize) {
        let size = self.card_hash_size() as usize;
        card_hashed_of(self.start as usize, size, piece as usize)
    }

    /// The bytes of the copy the card's hashes cover: the title buffer's own,
    /// rounded up to the hash's 16-byte step. The card's buffer is the title
    /// buffer rounded up to 64 KiB, of which a title buffer of 288 bytes can
    /// give nothing back but those (after fxceless c36971b2).
    /// `VICTORY_CARD_HASH_LEN=0`: the whole buffer.
    pub(crate) fn card_hash_size(&self) -> u64 {
        static OWN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let size = self.card.as_ref().map_or(self.len, |c| c.size);
        if *OWN.get_or_init(|| std::env::var_os("VICTORY_CARD_HASH_LEN").is_none_or(|v| v != "0")) {
            self.len.next_multiple_of(16).min(size)
        } else {
            size
        }
    }

    /// The walls trace (`crate::host::walls_trace`): a piece of `fetched`
    /// bytes of this copy came back from the card and `written` of them
    /// went into title memory. Summed per buffer and logged once a frame:
    /// which shader-written buffers the title's CPU could see fresh in
    /// that frame, and how much of each changed.
    pub(crate) fn walls_merged(&self, fetched: usize, written: usize) {
        if !crate::host::walls_trace() {
            return;
        }
        static SEEN: Mutex<(u64, BTreeMap<u64, (u64, u32, u64, u64)>)> = Mutex::new((0, BTreeMap::new()));
        let frame = crate::frame::current();
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.0 != frame {
            let buffers: Vec<String> = seen
                .1
                .iter()
                .map(|(at, v)| format!("{at:#x}+{:#x}: {} pieces, {} fetched, {} written", v.0, v.1, v.2, v.3))
                .take(32)
                .collect();
            if !buffers.is_empty() {
                tracing::info!(
                    frame = seen.0,
                    count = seen.1.len(),
                    buffers = ?buffers,
                    "walls trace: shader-written card buffers merged into title memory in this frame"
                );
            }
            seen.1.clear();
            seen.0 = frame;
        }
        let entry = seen.1.entry(self.start).or_insert((self.len, 0, 0, 0));
        entry.1 += 1;
        entry.2 += fetched as u64;
        entry.3 += written as u64;
    }

    /// After a submission, for a copy in the card's memory: `gpu` is what
    /// the card's side holds at offset `at` on, in a piece the compare
    /// found differing. The bytes shaders wrote (those that differ from the
    /// CPU's side, which is what the card's side held when they started)
    /// go into title memory, from the first to the last of them, as
    /// [`Shadow::copy_back`] puts them; the CPU's side takes the result and
    /// sends it up before the next submission. `title` is scratch memory.
    /// Returns the bytes written to title memory.
    pub(crate) fn card_merge(&self, at: usize, gpu: &[u8], title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_none() || at + gpu.len() > self.len as usize {
            return 0;
        }
        // SAFETY: inside the mapping (checked above); only the submitting
        // thread writes it.
        let old = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        self.by_use(crate::byuse::SHADER_WROTE);
        let Some(first) = first_difference(gpu, old) else {
            return 0;
        };
        let last = last_difference(gpu, old).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, old) = (&gpu[first..=last], &mut old[first..=last]);
        title.clear();
        title.resize(n, 0);
        let mut written = 0;
        if crate::host::read_into((start + at) as *const u8, title) {
            // The stretch as read: what the card did not change is the
            // title's, and may have been written again since.
            // As in `copy_back`: the card's byte where it is not what it
            // was before the submission, else the title's.
            let mut differs = 0u8;
            let was = title.clone();
            for ((t, g), o) in title.iter_mut().zip(gpu).zip(old.iter_mut()) {
                let wrote = if *g != *o { 0xffu8 } else { 0 };
                let merged = (*t & !wrote) | (*g & wrote);
                differs |= merged ^ *t;
                *t = merged;
                *o = merged;
            }
            if differs != 0 {
                written = put_changed(start + at, &was, title);
            }
        } else {
            // Title memory is not mapped there: the copy keeps what the
            // card wrote, as before.
            old.copy_from_slice(gpu);
        }
        // The card's side and its "before" take the result.
        self.card_changed(at, at + n);
        written
    }

    /// [`relay`], for a copy in the card's memory: `gpu` is what the
    /// card's side held at offset `at` on when a fetch sent after send
    /// `fetched_after` (`crate::host`'s order of the queue) ran, in a piece
    /// the compare found differing. The bytes that differ from the CPU's
    /// side are what shaders wrote; they go into title memory, from the
    /// first to the last of them, and into the CPU's side. Nothing is sent
    /// up afterwards (the card holds these bytes or newer ones: sending
    /// them up could undo a later flight's write), and the title's other
    /// bytes are left to the refresh.
    ///
    /// Left out: bytes of the CPU's side that hold title bytes the card
    /// had not received when the fetch ran, those still to go up and those
    /// that went up with a later send. There the card's side is the older
    /// one, and taking it for a shader's write would put it over what the
    /// title wrote. Under the CPU side's lock (the snapshot's, which such
    /// a copy has no other use for), as [`Shadow::merge_under_writes`].
    /// `title` is scratch memory. Returns the bytes written to title
    /// memory.
    pub(crate) fn card_merge_relay(&self, at: usize, gpu: &[u8], fetched_after: u64, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        let Some(card) = &self.card else {
            return 0;
        };
        if at + gpu.len() > self.len as usize {
            return 0;
        }
        let _side = self.before.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: inside the mapping (checked above). The offsets are in a
        // range declared and not done with, which no refresh writes; what
        // does write it (the title's bytes put in under shader writes)
        // holds the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        self.by_use(crate::byuse::SHADER_WROTE);
        let Some(first) = first_difference(gpu, held) else {
            return 0;
        };
        let last = last_difference(gpu, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, held) = (&gpu[first..=last], &mut held[first..=last]);
        let skip: Vec<(usize, usize)> = {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
            let mut skip: Vec<(usize, usize)> = dirty.iter().copied().filter(|&(a, b)| a < at + n && b > at).collect();
            sent.near(fetched_after, at, at + n, &mut skip);
            skip
        };
        title.clear();
        title.resize(n, 0);
        let mapped = crate::host::read_into((start + at) as *const u8, title);
        let was = title.clone();
        let mut changed = false;
        for (a, b) in outside(at, at + n, skip) {
            let (a, b) = (a - at, b - at);
            // Title memory that is not mapped: the CPU's side keeps what
            // the card wrote, as before; `title` is scratch then.
            changed |= merge_piece(&gpu[a..b], &mut held[a..b], &mut title[a..b]);
        }
        if !mapped || !changed {
            return 0;
        }
        put_changed(start + at, &was, title)
    }
}

/// For tests: copies with no host buffer behind them.
#[cfg(test)]
impl Shadow {
    /// A copy at `data` of title memory `[start, start + len)`, with a card
    /// side if `card`. It is never let go: there is no buffer to release.
    pub(crate) fn unbacked(start: u64, data: *mut u8, len: u64, card: bool) -> Arc<Shadow> {
        // SAFETY: an interface pointer nothing calls.
        let none = || unsafe { std::mem::transmute::<usize, ID3D12Resource>(8) };
        let mut shadow = Shadow::new(none(), data, 0, start, len);
        if card {
            shadow.card = Some(Card {
                staging: none(),
                before: Mutex::default(),
                dirty: Mutex::default(),
                sent: Mutex::default(),
                touched: Default::default(),
                me: Default::default(),
                size: len.next_multiple_of(PIECE as u64),
            });
        }
        let shadow = Arc::new(shadow);
        std::mem::forget(shadow.clone());
        shadow
    }

    /// What shaders wrote goes into title memory ([`put_changed`]).
    pub(crate) fn put(at: usize, title: &[u8], merged: &[u8]) {
        put_changed(at, title, merged);
    }

    /// What is to go up to the card, noted as sent with `number`, as
    /// [`card_dirty`] does; notes of sends up to `merged` go.
    pub(crate) fn send_up(&self, number: u64, merged: u64) -> Vec<(usize, usize)> {
        let card = self.card.as_ref().unwrap();
        let mut dirty = card.dirty.lock().unwrap();
        let ranges = std::mem::take(&mut *dirty);
        let mut sent = card.sent.lock().unwrap();
        sent.forget(merged);
        sent.note(number, ranges.clone());
        ranges
    }

    /// What is to go up to the card, taken.
    pub(crate) fn take_dirty(&self) -> Vec<(usize, usize)> {
        std::mem::take(&mut *self.card.as_ref().unwrap().dirty.lock().unwrap())
    }
}

impl Shadow {
    /// [`Shadow::card_merge_relay`] of several pieces, each within one
    /// 64 KiB block of title memory: from and to (offsets in the copy), and
    /// where the fetched bytes are. On any thread: several may land pieces
    /// of one copy, no two the same bytes. Returns the bytes put into title
    /// memory.
    ///
    /// What is left out is as there, asked once for all of them: the notes
    /// of bytes that went up (or are to go up) behind the fetch only move
    /// from `dirty` to `sent`, with a later number, and grow only by
    /// [`Shadow::merge_under_writes`], which the lock held here keeps out.
    ///
    /// `check`: this landing is the periodic one that brings back what was
    /// left on the card (`crate::census::check_landing`).
    ///
    /// # Safety
    /// The third of each piece is the address of its bytes, which stay as
    /// they are during the call.
    pub(crate) unsafe fn land_card(
        &self,
        pieces: &[(usize, usize, usize)],
        fetched_after: u64,
        notes: &mut crate::land::Notes,
        check: bool,
    ) -> u64 {
        let Some(card) = &self.card else {
            return 0;
        };
        let waiting = std::time::Instant::now();
        let _merging = self.merging.read().unwrap_or_else(|e| e.into_inner());
        crate::land::spent(crate::land::stat::LOCK, waiting);
        let (from, to) = pieces.iter().fold((usize::MAX, 0), |(a, b), p| (a.min(p.0), b.max(p.1)));
        let skip: Vec<(usize, usize)> = {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
            let mut skip: Vec<(usize, usize)> = dirty.iter().copied().filter(|&(a, b)| a < to && b > from).collect();
            sent.near(fetched_after, from, to, &mut skip);
            skip
        };
        let mut put = 0;
        for &(lo, hi, now) in pieces {
            if lo >= hi || hi > self.len as usize {
                continue;
            }
            let near: Vec<(usize, usize)> = skip.iter().copied().filter(|&(a, b)| a < hi && b > lo).collect();
            let parts: Vec<(usize, usize)> = outside(lo, hi, near).into_iter().map(|(a, b)| (a - lo, b - lo)).collect();
            // SAFETY: inside the mapping (checked above), in a range
            // declared and not done with, which no refresh writes; the
            // title's bytes are kept out by the lock; the caller's.
            // Where title memory is not mapped the CPU's side keeps what
            // the card wrote, as before.
            let merged =
                unsafe { crate::land::piece(self.start as usize + lo, now as *const u8, self.data.add(lo), &parts, true, notes, self) };
            self.walls_merged(hi - lo, merged as usize);
            crate::census::merged(self.census(), lo, hi, merged as usize, check);
            self.by_use(crate::byuse::SHADER_WROTE);
            if merged > 0 {
                self.note_landing(lo, hi);
            }
            put += merged;
        }
        put
    }

    /// [`Shadow::relay_merge`] of a piece `[at, at + n)` (within one 64 KiB
    /// block of title memory) of a copy in the PC's memory; on any thread,
    /// as [`Shadow::land_card`]. The copy is read once and never written.
    /// Returns the bytes put into title memory.
    pub(crate) fn land_back(&self, at: usize, n: usize, notes: &mut crate::land::Notes) -> u64 {
        if self.card.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let waiting = std::time::Instant::now();
        let _merging = self.merging.read().unwrap_or_else(|e| e.into_inner());
        crate::land::spent(crate::land::stat::LOCK, waiting);
        // The snapshot's memory stays where it is until the range is done
        // with (`copy_back_done`), which comes after this.
        let Some(held) = self.before.lock().unwrap_or_else(|e| e.into_inner()).as_mut().map(Vec::as_mut_ptr) else {
            return 0;
        };
        // SAFETY: inside the copy's mapping (checked above), which only the
        // card writes there, and inside the snapshot's allocation, which
        // holds the copy's length (`writing`): these offsets are in a range
        // declared and not done with, whose bytes it holds, and the lock
        // keeps out what else writes them. Title memory that is not there
        // is looked at again when the next flight that declared the range
        // lands.
        let put =
            unsafe { crate::land::piece(self.start as usize + at, self.data.add(at), held.add(at), &[(0, n)], false, notes, self) };
        if put > 0 {
            self.by_use(crate::byuse::SHADER_WROTE);
        }
        put
    }

    /// What shaders wrote in 64 KiB block `block` (of title memory) of this
    /// copy was put into title memory beside the guard, which was noted as a
    /// write with number `now`, `before` the number of the block's last write
    /// before. If the copy held title memory's bytes of the block ahead of
    /// that it holds them still, and is noted to ([`crate::land`] says why);
    /// whether it was.
    ///
    /// That it did: it took the whole block after that write, and no other
    /// came that is not noted yet. Looked at after the note, whatever comes
    /// later has a number above it: a write of the title's takes the block's
    /// bit, until an asking has noted it.
    pub(crate) fn landed(&self, block: usize, before: u64, now: u64) -> bool {
        let Some(page) = block.checked_sub(self.start as usize / PAGE) else {
            return false;
        };
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let Some(at) = copied.get_mut(page) else {
            return false;
        };
        let held = *at != 0 && *at >= before && crate::watch::block_published(block);
        if held {
            *at = (*at).max(now);
        }
        held
    }
}

#[cfg(test)]
mod land_tests {
    use super::{Card, PIECE, Shadow};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    /// A copy in the card's memory of title memory at `title` (its card side
    /// is never looked at here), all bytes 0 on the CPU's side.
    fn copy_of(title: usize, len: usize) -> Option<Shadow> {
        let (buffer, data) = crate::host::host()?.shared_buffer(len as u64)?;
        // SAFETY: the mapping, `len` bytes.
        unsafe { std::ptr::write_bytes(data, 0, len) };
        let mut shadow = Shadow::new(buffer.clone(), data, 0, title as u64, len as u64);
        shadow.card = Some(Card {
            staging: buffer,
            before: Mutex::default(),
            dirty: Mutex::default(),
            sent: Mutex::default(),
                touched: Default::default(),
                me: Default::default(),
            size: len as u64,
        });
        Some(shadow)
    }

    /// Memory that stays (`crate::watch` knows its addresses): private, or a
    /// view noted as the runtime notes the title's.
    fn memory(len: usize, view: bool) -> usize {
        use windows_sys::Win32::System::Memory as mem;
        // SAFETY: plain allocations of our own.
        unsafe {
            if !view {
                return mem::VirtualAlloc(std::ptr::null(), len, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE) as usize;
            }
            let section = mem::CreateFileMappingW(-1isize as _, std::ptr::null(), mem::PAGE_READWRITE, 0, len as u32, std::ptr::null());
            let at = mem::MapViewOfFile(section, mem::FILE_MAP_WRITE, 0, 0, len).Value as usize;
            crate::inplace::title_memory_section(section as usize, len as u64, true);
            crate::title_memory_remapped(at, len);
            crate::inplace::title_memory_view(at, len, section as usize, 0);
            at
        }
    }

    fn random(seed: &mut u64) -> usize {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 8) as usize
    }

    /// [`Shadow::land_card`] leaves title memory and the CPU's side as
    /// [`Shadow::card_merge_relay`] does, with title bytes that went up
    /// behind the fetch, or have not gone up, left out.
    #[test]
    fn pieces_land_as_the_merge_one_by_one_lands_them() {
        assert!(crate::safecopy::on());
        const LEN: usize = 3 * PIECE;
        let mut seed = 0x1234_5678_9abcu64;
        for view in [false, true] {
            let (a, b) = (memory(LEN, view), memory(LEN, view));
            let (Some(one), Some(many)) = (copy_of(a, LEN), copy_of(b, LEN)) else {
                return; // no host GPU
            };
            // SAFETY: the two memories and the two mappings.
            let bytes = |at: usize| unsafe { std::slice::from_raw_parts_mut(at as *mut u8, LEN) };
            for round in 0..40u32 {
                let base: Vec<u8> = (0..LEN).map(|_| random(&mut seed) as u8).collect();
                let (mut held, mut title, mut now) = (base.clone(), base.clone(), base);
                let (mut dirty, mut sent) = (Vec::new(), Vec::new());
                for k in 0..120 {
                    let at = random(&mut seed) % LEN / 4 * 4;
                    let n = (random(&mut seed) % 900 / 4 * 4).min(LEN - at);
                    match k % 4 {
                        0 | 1 => now[at..at + n].fill(random(&mut seed) as u8),
                        2 => title[at..at + n].fill(random(&mut seed) as u8),
                        // Title bytes put into the CPU's side meanwhile.
                        _ => {
                            let byte = random(&mut seed) as u8;
                            held[at..at + n].fill(byte);
                            title[at..at + n].fill(byte);
                            if k % 8 == 3 {
                                dirty.push((at, at + n));
                            } else {
                                sent.push((at, at + n, u64::from(round % 3) + (k as u64 / 16)));
                            }
                        }
                    }
                }
                let after = 4u64;
                for (shadow, at) in [(&one, a), (&many, b)] {
                    crate::watch::writing(at, LEN);
                    bytes(at).copy_from_slice(&title);
                    bytes(shadow.data as usize).copy_from_slice(&held);
                    let card = shadow.card.as_ref().expect("made with one");
                    *card.dirty.lock().unwrap() = dirty.clone();
                    *card.sent.lock().unwrap() = super::SentRuns::of(&sent);
                }
                let pieces: Vec<(usize, usize)> = (0..3).map(|p| (p * PIECE + 16, (p + 1) * PIECE - 4)).collect();
                let mut scratch = Vec::new();
                for &(lo, hi) in &pieces {
                    one.card_merge_relay(lo, &now[lo..hi], after, &mut scratch);
                }
                let addressed: Vec<(usize, usize, usize)> =
                    pieces.iter().map(|&(lo, hi)| (lo, hi, now.as_ptr() as usize + lo)).collect();
                let mut notes = crate::land::Notes::default();
                // SAFETY: the fetched bytes are `now`'s.
                unsafe { many.land_card(&addressed, after, &mut notes, false) };
                notes.flush();
                assert!(bytes(a) == bytes(b), "title memory, round {round}");
                assert!(bytes(one.data as usize) == bytes(many.data as usize), "the CPU's side, round {round}");
            }
        }
    }

    /// Landings of one copy on several threads, the title writing dwords of
    /// the same blocks, and the title's bytes put into the CPU's side under
    /// the shaders' writes ([`Shadow::merge_under_writes`]), all at once. The
    /// card has never received the title's bytes: what is fetched holds older
    /// ones there, which must not be taken for shaders' writes.
    #[test]
    fn landings_and_title_bytes_going_in_keep_each_others_bytes() {
        assert!(crate::safecopy::on());
        const BLOCKS: usize = 16;
        const LEN: usize = BLOCKS * PIECE;
        let title = memory(LEN, true);
        let Some(shadow) = copy_of(title, LEN) else {
            return; // no host GPU
        };
        shadow.pending.lock().unwrap().push((title, title + LEN));
        // SAFETY: committed, aligned; touched as atomics or by the landing's
        // stores of whole dwords.
        let dword = |i: usize| unsafe { &*((title + i * 4) as *const AtomicU32) };
        let stop = AtomicBool::new(false);
        std::thread::scope(|s| {
            let writer = s.spawn(|| {
                let mut value = 1u32;
                while !stop.load(Ordering::Relaxed) {
                    for i in (1..LEN / 4).step_by(4 * 331) {
                        assert_eq!(dword(i).swap(value, Ordering::Relaxed), value - 1, "dword {i}: undone");
                    }
                    value += 1;
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
                value - 1
            });
            s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    let mut runs = usize::MAX;
                    assert!(!shadow.merge_under_writes(title, LEN, &mut runs));
                    // Sent up, as the preparing of a submission would: noted
                    // with a number behind the fetches'.
                    let card = shadow.card.as_ref().expect("made with one");
                    let mut dirty = card.dirty.lock().unwrap();
                    let mut sent = card.sent.lock().unwrap();
                    sent.note(9, std::mem::take(&mut *dirty));
                }
            });
            let mut now = vec![0u8; LEN];
            for round in 1..200u32 {
                for block in (0..BLOCKS).filter(|b| !(*b as u32 * 5 + round).is_multiple_of(3)) {
                    for i in (block * PIECE / 4 + 3..(block + 1) * PIECE / 4).step_by(4) {
                        now[i * 4..i * 4 + 4].copy_from_slice(&(round << 20 | i as u32).to_le_bytes());
                    }
                }
                let at = now.as_ptr() as usize;
                crate::land::each(BLOCKS, &|b| {
                    let mut notes = crate::land::Notes::default();
                    let (lo, hi) = (b * PIECE, (b + 1) * PIECE);
                    // SAFETY: the fetched bytes are `now`'s; each block once.
                    unsafe { shadow.land_card(&[(lo, hi, at + lo)], 8, &mut notes, false) };
                    notes.flush();
                });
                for i in (3..LEN / 4).step_by(4 * 53) {
                    let expect = u32::from_le_bytes(now[i * 4..i * 4 + 4].try_into().expect("four"));
                    assert_eq!(dword(i).load(Ordering::Relaxed), expect, "round {round} dword {i}");
                }
            }
            stop.store(true, Ordering::Relaxed);
            let last = writer.join().expect("nothing undone");
            assert!((1..LEN / 4).step_by(4 * 331).all(|i| dword(i).load(Ordering::Relaxed) == last));
        });
    }

    /// `VICTORY_SKIP_WC_BACK` (`crate::census::skip_wc_back`): a refresh
    /// over a page whose shaders' bytes were left on the card takes only
    /// what the title changed there into the CPU's side and up (the card
    /// keeps its bytes beside them); the pages around go in and up whole.
    #[test]
    fn a_refresh_over_bytes_left_on_the_card_sends_only_the_titles_changes() {
        use crate::watch::PAGE;
        const LEN: usize = 3 * PAGE;
        let content: Vec<u8> = (0..LEN).map(|i| (i * 3 + i / 977) as u8).collect();
        let mut title = content.clone();
        let start = title.as_ptr() as usize;
        let Some(shadow) = copy_of(start, LEN) else {
            return; // no host GPU
        };
        // SAFETY: the mapping, `LEN` bytes.
        unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), shadow.data, LEN) };
        // A page wholly inside the copy, at offset `at`.
        let at = (start / PAGE + 1) * PAGE - start;
        let at = if at < 16 { at + PAGE } else { at };
        let (before, inside, after) = (at - 10, at + 1000, at + PAGE + 10);
        for i in [before, inside, after] {
            title[i] ^= 0x77;
        }
        crate::census::left_for_test(shadow.id, Some((start + at + 100, start + at + 200)));
        let copied = shadow.copy(start, start + LEN);
        let dirty = shadow.take_dirty();
        crate::census::left_for_test(shadow.id, None);
        assert_eq!(copied, LEN);
        // SAFETY: the mapping, `LEN` bytes.
        let side = unsafe { std::slice::from_raw_parts(shadow.data, LEN) };
        assert!(side == &title[..], "every byte the title changed is in the CPU's side");
        assert_eq!(dirty, vec![(0, at), (inside, inside + 1), (at + PAGE, LEN)]);
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
///
/// Read by every view a draw names, on all recording threads at once, and
/// written when a buffer comes, goes or gets its copy: readers share it.
static BUFFERS: std::sync::RwLock<BTreeMap<(u64, usize), Entry>> = std::sync::RwLock::new(BTreeMap::new());

fn buffers() -> std::sync::RwLockWriteGuard<'static, BTreeMap<(u64, usize), Entry>> {
    let map = BUFFERS.write().unwrap_or_else(|e| e.into_inner());
    // What threads remember of it ([`FOUND`]) is of before.
    BUFFERS_CHANGED.fetch_add(1, Ordering::SeqCst);
    map
}

/// How often [`BUFFERS`] was taken to be changed.
static BUFFERS_CHANGED: AtomicU64 = AtomicU64::new(1);

/// Answers a thread remembers.
const FOUND_MOST: usize = 1024;

/// An ask (address, bytes), the count of changes it was answered at, and the
/// answer.
type Found = (u64, u64, u64, std::sync::Weak<Shadow>);

thread_local! {
    /// The last answers of [`lookup`] out of [`BUFFERS`], by a hash of the
    /// ask: the same buffers are asked about draw after draw, from five
    /// threads, each time through the map's lock and a dozen of its nodes
    /// (7 % of the recording threads' time, profile waits-1).
    static FOUND: std::cell::RefCell<Vec<Found>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Whether [`FOUND`] is kept (`VICTORY_QUICK_FIND=0`: not).
fn quick_find() -> bool {
    crate::toggled!("VICTORY_QUICK_FIND")
}

fn found_at(address: u64, len: u64) -> usize {
    let h = (address ^ len.rotate_left(32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> 54) as usize % FOUND_MOST
}

/// What this thread remembers of an ask, if nothing changed since.
fn found(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let changed = BUFFERS_CHANGED.load(Ordering::SeqCst);
    let shadow = FOUND.with(|f| {
        let f = f.borrow();
        let (a, n, at, shadow) = f.get(found_at(address, len))?;
        ((*a, *n, *at) == (address, len, changed)).then(|| shadow.upgrade()).flatten()
    })?;
    // Whoever takes the map to let a copy go that nobody holds finds it held
    // now, or had counted before: then it is not handed out.
    std::sync::atomic::fence(Ordering::SeqCst);
    (BUFFERS_CHANGED.load(Ordering::SeqCst) == changed).then_some(shadow)
}

/// Remembers an answer given while `changed` was the count.
fn remember(address: u64, len: u64, changed: u64, shadow: &Arc<Shadow>) {
    FOUND.with(|f| {
        let mut f = f.borrow_mut();
        if f.is_empty() {
            f.resize_with(FOUND_MOST, || (0, 0, 0, std::sync::Weak::new()));
        }
        f[found_at(address, len)] = (address, len, changed, Arc::downgrade(shadow));
    });
}

fn buffers_read() -> std::sync::RwLockReadGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.read().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The title buffer holding `address`, as (start, length), without making
/// its copy (diagnostics: which buffer a dropped program's root view names).
#[cfg(feature = "diagnostics")]
pub(crate) fn extent(address: u64) -> Option<(u64, u64)> {
    buffers()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end > address)
        .map(|(k, e)| (k.0, e.end - k.0))
}

/// Title memory `[at, at + len)` is about to show something else or be
/// given up: what is in place there (`crate::inplace`) is taken out of the
/// tables, so that the next ask makes a copy or opens the memory anew, and
/// goes when no list and no slot holds it any more. Returns how many.
pub(crate) fn in_place_let_go(at: u64, len: u64) -> u64 {
    let end = at.saturating_add(len);
    let mut gone: Vec<Arc<Shadow>> = Vec::new();
    {
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = map
            .range((at.saturating_sub(WINDOW), 0)..(end, 0))
            .filter(|(key, shadow)| shadow.place.is_some() && key.0 + key.1 > at)
            .map(|(key, _)| *key)
            .collect();
        gone.extend(keys.into_iter().filter_map(|key| map.remove(&key)));
    }
    {
        // As `lookup` finds a buffer: among the 256 that start below.
        let mut map = buffers();
        for (_, entry) in map.range_mut(..(end, 0)).rev().take(256) {
            if entry.end > at && entry.shadow.as_ref().is_some_and(|s| s.place.is_some()) {
                gone.extend(entry.shadow.take());
            }
        }
    }
    let n = gone.len() as u64;
    for shadow in gone {
        crate::host::retire(shadow);
    }
    n
}

/// For the in-place census (`crate::inplace`): the title buffers (start,
/// bytes, false) and the windows outside buffers that have a copy (start,
/// bytes, true).
pub(crate) fn spans() -> Vec<(u64, u64, bool)> {
    let mut spans: Vec<(u64, u64, bool)> = buffers_read().iter().map(|(key, entry)| (key.0, entry.end - key.0, false)).collect();
    let loose = LOOSE.write().unwrap_or_else(|e| e.into_inner());
    spans.extend(loose.keys().map(|&(start, len)| (start, len, true)));
    spans
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use. A `len` of 0 or 1 is an ask without a length
/// (a root view: the caller reads on to the copy's end).
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    lookup(address, len, false)
}

/// As [`find`], for an ask that is done with the copy when its list has
/// run (a draw's vertex and index buffers, indirect arguments): outside
/// buffer objects it takes a window that is there already and holds the
/// range ([`holder`]), and makes one of its own only if none does.
///
/// [`find`] names a window by the ask alone, which a descriptor slot needs
/// (the slot shows the window its view found when it was made, and the
/// pass that brings it up to date must find the same one). A draw needs no
/// such thing: it binds the copy it found and brings that copy up to date.
/// Named by the ask alone, every draw's range was a window of its own
/// wherever the windows are cut finer than 4 MiB: draws that bind
/// different ranges of the same memory then each copied their range, and
/// windows nothing asked for again for ten seconds were let go and made
/// (and copied) anew. Run 71, on a 256 KiB grid: 350 MB copied per frame
/// against 120 MB, 7,087 write-watch askings against 2,600, and 5,263
/// windows (4.5 GB) let go in two minutes against 393 in a whole run.
pub(crate) fn find_any(address: u64, len: u64) -> Option<Arc<Shadow>> {
    // With whole windows (the default) there is one window to a cell and
    // nothing to choose: exactly `find`.
    if !fine() {
        return find(address, len);
    }
    lookup(address, len, true)
}

/// The copy [`find`] gives for a range inside a title buffer, if it was made.
pub(crate) fn peek(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    buffers_read().range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= end).and_then(|(_, e)| e.shadow.clone())
}

fn lookup(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
    let quick = quick_find();
    if quick && let Some(shadow) = found(address, len) {
        return Some(shadow);
    }
    let end = address.checked_add(len.max(1))?;
    {
        let map = buffers_read();
        match map.range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= end) {
            // The copy is made already (nearly always).
            Some((_, Entry { shadow: Some(shadow), .. })) => {
                if quick {
                    // (Nobody changes the map while this lock is held.)
                    remember(address, len, BUFFERS_CHANGED.load(Ordering::SeqCst), shadow);
                }
                return Some(shadow.clone());
            }
            Some(_) => {}
            None => {
                drop(map);
                return loose(address, end, len <= 1, any);
            }
        }
    }
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end, len <= 1, any);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let size = len.next_multiple_of(1 << 16);
        // In place, if asked for and the memory can be opened
        // (`crate::inplace::place_for`): the buffer is the title's own
        // memory. Not the big ones `cards` puts in the card's memory: those
        // are read too often to stay in the PC's.
        let big = cards() && (size >= card_limits().0 || card_all());
        // Renderer port 1 (`crate::port`, `VICTORY_PORT_CONSTANTS=1`): the
        // title's 32-bit graphics heap in place, big or not; with
        // `VICTORY_PORT_SECTIONS` also 64-bit section stretches (`noncard`:
        // not the big ones the card takes).
        if let Some(placed) = crate::port::place_for_kind(key.0, len, false, big) {
            entry.shadow = Some(Arc::new(Shadow::placed(placed, key.0, len, false)));
            return entry.shadow.clone();
        }
        if !big && let Some(placed) = crate::inplace::place_for(key.0, len, false) {
            entry.shadow = Some(Arc::new(Shadow::placed(placed, key.0, len, false)));
            return entry.shadow.clone();
        }
        let (buffer, data) = host()?.shared_buffer(size)?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // A big one, with `cards`: shaders get a buffer in the card's own
        // memory, and the one just made is the CPU's side.
        let (buffer, card) = match card_for(size) {
            Some(on_card) => {
                crate::host::name(&on_card, || format!("title buffer {:#x}+{len:#x} on the card", key.0));
                tracing::info!(
                    start = format_args!("{:#x}", key.0),
                    len,
                    "a buffer's host copy is in the card's memory"
                );
                let card = Card {
                    staging: buffer,
                    before: Mutex::default(),
                    dirty: Mutex::default(),
                    sent: Mutex::default(),
                touched: Default::default(),
                me: Default::default(),
                    size,
                };
                (on_card, Some(card))
            }
            None => (buffer, None),
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        let mut shadow = Shadow::new(buffer, data, gpu, key.0, len);
        shadow.card = card;
        let shadow = Arc::new(shadow);
        if shadow.card.is_some() {
            carded(&shadow);
        }
        entry.shadow = Some(shadow);
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (
/// the title's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// A window nothing names any more is let go ([`let_go_idle`]).
/// shortcut: a window is 4 MiB however little of it is read (the memory
/// line says how much was ever asked for, and how much was ever copied);
/// [`window`] cuts the windows of asks that give a length on a finer grid.
const WINDOW: u64 = 4 << 20;

/// The grid the windows of asks with a length are cut on, in bytes: 4 MiB
/// unless `VICTORY_WINDOW_KB` names a smaller power of two (64 at least).
/// A window is the PC's memory, all of it, however little of it is read:
/// 803 windows held 3.3 GB in run 60's gameplay, and all that was ever
/// asked for in them lay within 1.6 GB. On a grid of 1 MiB or 256 KiB a
/// window ends nearer to what is read. An ask without a length keeps the
/// 4 MiB grid whatever is set: a root view reads from its address to its
/// window's end, and mesh streams bound that way reach megabytes.
/// shortcut: the default stays 4 MiB until a run with the memory line's
/// window counts has shown what the finer grids give; the same memory
/// asked for with different ranges then lies in more than one window.
fn window() -> u64 {
    static GRID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *GRID.get_or_init(|| {
        let kb = std::env::var("VICTORY_WINDOW_KB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        grid(kb)
    })
}

/// The window grid in bytes for a setting in KiB: a power of two from
/// 64 KiB to 4 MiB, 4 MiB for anything else.
fn grid(kb: Option<u64>) -> u64 {
    match kb {
        Some(kb) if kb.is_power_of_two() && (64..=4096).contains(&kb) => kb << 10,
        _ => WINDOW,
    }
}

/// The window (start, bytes) of title memory for an ask of `[address,
/// end)` on a grid of `grid` bytes.
fn window_of(address: u64, end: u64, grid: u64) -> (u64, u64) {
    let start = address & !(grid - 1);
    (start, end.next_multiple_of(grid) - start)
}

/// Whether the window of an ask without a length begins at the ask instead
/// of at the 4 MiB line below it. Such an ask (a root view) reads from its
/// address to the end of its 4 MiB cell; what lies below the lowest
/// address ever asked for in the cell is memory the host zeroed and nothing
/// reads (513 of run 65's 782 windows were asked for that way, 2,052 MB;
/// of all windows' 3,236 MB, 1,988 MB were ever copied). With this on, the
/// window is the cell's tail from the ask's step on ([`TAIL_STEP`]); an ask
/// lower in the cell later makes a longer tail, which every ask from then
/// on takes, and the shorter one goes when nothing holds it any more.
/// shortcut: off unless `VICTORY_WINDOW_TAILS=1`. A longer tail is another
/// host buffer than the shorter one: what was bound to the shorter stays
/// right only because every list brings up to date the copy it bound
/// (`draw.rs` keeps a root view's copy with its slot). It pays only
/// together with `VICTORY_WINDOW_KB` below 4096: on the 4 MiB grid an ask
/// with a length in the same cell makes the whole cell's window, and the
/// tail is that.
fn tails() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_TAILS").is_some_and(|v| v == "1"))
}

/// Tails begin on multiples of this. A longer tail is a new host buffer,
/// filled from title memory from its first byte to its last: begun on the
/// 64 KiB page of each ask, a cell whose asks came from the top down made
/// up to 63 of them, one after the other (part of run 71's copies). On
/// this step it is 15 at most, and a tail holds up to 192 KiB it did not
/// need.
const TAIL_STEP: u64 = 256 << 10;

/// The window (start, bytes) a tail ask at `address` makes: from the ask's
/// step ([`TAIL_STEP`]) to the end of its 4 MiB cell.
fn tail_of(address: u64, end: u64) -> (u64, u64) {
    let (cell, len) = window_of(address, end, WINDOW);
    let start = address & !(TAIL_STEP - 1);
    (start, cell + len - start)
}

/// Of `windows` (start, bytes), the one an ask of `[address, end)` that
/// may take any window takes ([`find_any`]): the longest that holds the
/// whole range.
fn holder(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    windows
        .filter(|&(start, bytes)| start <= address && start + bytes >= end)
        .max_by_key(|&(start, bytes)| (bytes, std::cmp::Reverse(start)))
}

/// Whether windows may be smaller than their 4 MiB cell (a finer grid, or
/// tails): only then is there more than one window to choose from.
fn fine() -> bool {
    tails() || window() < WINDOW
}

/// For the memory line: the bytes of title memory the windows alive lie
/// over (each byte once), beside the windows' own bytes in
/// [`memory`]: the difference is memory that is in more than one window,
/// which is copied once for each.
#[cfg(feature = "diagnostics")]
pub(crate) fn window_cover() -> u64 {
    let loose = LOOSE.write().unwrap_or_else(|e| e.into_inner());
    cover(loose.keys().copied())
}

/// The bytes `windows` (start, bytes; in order of start) lie over, each
/// byte once.
#[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
fn cover(windows: impl Iterator<Item = (u64, u64)>) -> u64 {
    let (mut covered, mut reached) = (0, 0);
    for (start, bytes) in windows {
        let end = start + bytes;
        if end > reached {
            covered += end - start.max(reached);
            reached = end;
        }
    }
    covered
}

/// Of `windows` (start, bytes; in order), the longest tail a tail ask at
/// `address` can take: the first that begins in the ask's cell at or below
/// the ask's page and ends where the cell ends.
fn longest_tail(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    let (cell, len) = window_of(address, end, WINDOW);
    let step = address & !(TAIL_STEP - 1);
    windows
        .filter(|&(start, bytes)| start >= cell && start <= step && start + bytes == cell + len)
        .min()
}

/// Whether `[address, address + len)` lies in one title buffer.
pub(crate) fn in_buffer(address: u64, len: u64) -> Option<(u64, u64)> {
    let end = address.checked_add(len)?;
    let map = buffers_read();
    let found = map.range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= end);
    found.map(|(k, e)| (k.0, e.end - k.0))
}

/// Whether a title buffer of at most `most` bytes lies in `[address, address
/// + len)`, all of it or in part.
pub(crate) fn small_buffer_in(address: u64, len: u64, most: u64) -> bool {
    let map = buffers_read();
    map.range((address.saturating_sub(most), 0)..(address + len, 0))
        .any(|(key, entry)| entry.end > address && entry.end - key.0 <= most)
}

/// A copy of the memory `[start, start + len)` (an arena of
/// [`crate::memory`]'s, its directory, title memory around a small buffer)
/// for the programs that find memory by address, on the card: the structures
/// rays are traced through, read with every step of every ray.
///
/// What shaders write there by address is not brought back, so the copy is
/// the only place that has it. The CPU's side holds the bytes as they were
/// last taken, and only dwords that changed since go up.
pub(crate) fn by_address(start: u64, len: u64) -> Option<Arc<Shadow>> {
    let host = host()?;
    if !cards() || !host.card_ready() {
        return None;
    }
    // At most so much on the card's own memory (4 GiB unless
    // `VICTORY_MEMORY_CARD_MB` says otherwise), and while it has room; past
    // that the side shaders see is in the PC's memory.
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    static USED: AtomicU64 = AtomicU64::new(0);
    let most = *MOST.get_or_init(|| {
        let mb = std::env::var("VICTORY_MEMORY_CARD_MB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        mb.unwrap_or(4096) << 20
    });
    let room = host.video_memory().is_none_or(|[budget, used, ..]| used + len < budget / 100 * 97);
    let (staging, data) = host.shared_buffer(len)?;
    let on_card = if room && USED.fetch_add(len, Ordering::Relaxed) + len <= most {
        host.uav_buffer(len)
    } else {
        None
    };
    // Counted but not made (past `most`, or the card would not have it):
    // its bytes are given back.
    if room && on_card.is_none() {
        USED.fetch_sub(len, Ordering::Relaxed);
    }
    let buffer = match on_card {
        Some(buffer) => buffer,
        None => host.shared_buffer(len)?.0,
    };
    crate::host::name(&staging, || format!("title memory {start:#x}+{len:#x} by address, the CPU's side"));
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x} by address"));
    // (Counted down with the windows' on the card when it goes.)
    WINDOW_CARD_BYTES.fetch_add(len, Ordering::Relaxed);
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = true;
    shadow.changes_only.store(true, Ordering::Relaxed);
    shadow.card = Some(Card {
        staging,
        before: Mutex::default(),
        dirty: Mutex::default(),
        sent: Mutex::default(),
        touched: Default::default(),
        me: Default::default(),
        size: len,
    });
    let shadow = Arc::new(shadow);
    carded(&shadow);
    Some(shadow)
}

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: std::sync::RwLock<BTreeMap<(u64, u64), Arc<Shadow>>> = std::sync::RwLock::new(BTreeMap::new());

fn loose(address: u64, end: u64, unsized_ask: bool, any: bool) -> Option<Arc<Shadow>> {
    let (cell, _) = window_of(address, end, WINDOW);
    let named = window_of(address, end, if unsized_ask { WINDOW } else { window() });
    // Under the lock, with the handing out: see `let_go_idle`.
    let kept = window_keep();
    // `fresh`: the window was just made.
    let note = |shadow: &Shadow, fresh: bool| {
        // (Read first: the lines are shared by all recording threads.)
        let now = crate::host::moment();
        let mut first = fresh;
        if shadow.used.load(Ordering::Relaxed) != now {
            let was = shadow.used.swap(now, Ordering::Relaxed);
            first |= was >> 32 != now >> 32;
        }
        if kept && first {
            NAMED.lock().unwrap_or_else(|e| e.into_inner()).add(now >> 32, shadow.len);
        }
        if shadow.asked[0].load(Ordering::Relaxed) > address {
            shadow.asked[0].fetch_min(address, Ordering::Relaxed);
        }
        if shadow.asked[1].load(Ordering::Relaxed) < end {
            shadow.asked[1].fetch_max(end, Ordering::Relaxed);
        }
        if unsized_ask {
            shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
        }
    };
    // The window is there already (nearly always): readers share the lock.
    if !(unsized_ask && tails()) && !(any && fine() && !unsized_ask) {
        let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        if let Some(shadow) = map.get(&named) {
            note(shadow, false);
            return Some(shadow.clone());
        }
    }
    let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
    let (start, len) = if unsized_ask && tails() {
        let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
        longest_tail(there, address, end).unwrap_or_else(|| tail_of(address, end))
    } else if any && fine() && !unsized_ask {
        // The windows that begin in the ask's cell at or below the ask (one
        // that begins in a cell below and reaches this far is passed over:
        // the ask then makes its own).
        let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
        holder(there, address, end).unwrap_or(named)
    } else {
        named
    };
    if let Some(shadow) = map.get(&(start, len)) {
        note(shadow, false);
        return Some(shadow.clone());
    }
    if LET_GO_BOOK.lock().unwrap_or_else(|e| e.into_inner()).made((start, len), crate::host::now_ms()) {
        REMADE[0].fetch_add(1, Ordering::Relaxed);
        REMADE[1].fetch_add(len, Ordering::Relaxed);
    }
    // In place, if asked for and the window's memory can be opened: the
    // window is the title's own memory, with nothing to copy.
    if let Some(placed) = crate::port::place_for(start, len, true).or_else(|| crate::inplace::place_for(start, len, true)) {
        let shadow = Arc::new(Shadow::placed(placed, start, len, true));
        note(&shadow, true);
        map.insert((start, len), shadow.clone());
        return Some(shadow);
    }
    // One buffer in the card's memory, filled by the CPU through the bus.
    if let Some((buffer, data)) = window_mapped_for(len) {
        crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x} mapped on the card"));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let mut shadow = Shadow::new(buffer, data, gpu, start, len);
        shadow.window = true;
        shadow.mapped = true;
        let shadow = Arc::new(shadow);
        note(&shadow, true);
        map.insert((start, len), shadow.clone());
        return Some(shadow);
    }
    let (buffer, data) = match spare_window(len) {
        Some(spare) => spare,
        None => host()?.shared_buffer(len)?,
    };
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // With room for windows on the card (`window_card_most`): shaders get a
    // buffer there, and the one just made is the CPU's side.
    let (buffer, card) = match window_card_for((start, len), len) {
        Some(on_card) => {
            crate::host::name(&on_card, || format!("title memory {start:#x}+{len:#x} on the card"));
            let card = Card {
                staging: buffer,
                before: Mutex::default(),
                dirty: Mutex::default(),
                sent: Mutex::default(),
                touched: Default::default(),
                me: Default::default(),
                size: len,
            };
            (on_card, Some(card))
        }
        None => (buffer, None),
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = true;
    shadow.card = card;
    let shadow = Arc::new(shadow);
    if shadow.card.is_some() {
        carded(&shadow);
    }
    note(&shadow, true);
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// Copies of single buffer views' own bytes ([`exact`]), by (start,
/// bytes), each with the asking that last took it; and the askings so far.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(u64, BTreeMap<(u64, u64), (Arc<Shadow>, u64)>)> = Mutex::new((0, BTreeMap::new()));

/// Since the last frame line: askings for such copies, copies made, and
/// host buffers made for them (each copy of [`exact`] is one; in blocks,
/// [`exact_in_blocks`], one a block).
static EXACT_STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Whether a view of `len` bytes at `address` may have a copy of its own
/// bytes: the largest is 64 MiB, and no view the title made on purpose is
/// off its dwords.
fn exact_fits(address: u64, len: u64) -> bool {
    const MOST: u64 = 64 << 20;
    len != 0 && len <= MOST && address.is_multiple_of(4)
}

/// A copy of the title memory `[address, address + len)` that starts at
/// `address` itself: for a buffer view whose bytes do not start on an
/// element of the copy its buffer has (`descriptor/host.rs` `view_home`).
/// A host view counts its first element from its buffer's start; a console
/// view starts at any address, and the title places views at ring
/// positions inside its big GPU heap. Such a view was refused and its slot
/// read zeros ("buffer view not on an element of its host copy", in every
/// run; upstream VIctory 86a5ed6 found its meshlet culling constants and
/// colour-bake parameter blocks among them and gives each a copy of its
/// own bytes, as here).
///
/// The copy is brought up to date and written back like any other (a list
/// refreshes the copy its view's slot shows, `buffer_copy`). Kept by
/// (start, bytes) for as long as a list or a slot holds it; of the others,
/// those not asked for lately go once there are many (ring positions
/// change every frame).
pub(crate) fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    /// Copies kept before the idle ones are looked for, and the askings an
    /// idle one stays for.
    const MANY: usize = 256;
    const LATELY: u64 = 512;
    if !exact_fits(address, len) {
        return None;
    }
    EXACT_STATS[0].fetch_add(1, Ordering::Relaxed);
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (tick, map) = &mut *guard;
    *tick += 1;
    let now = *tick;
    if let Some((shadow, used)) = map.get_mut(&(address, len)) {
        *used = now;
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        return Some(shadow.clone());
    }
    if map.len() >= MANY {
        // Held by the map alone (no list), its buffer by the copy alone (no
        // slot shows it), and not asked for lately.
        map.retain(|_, (shadow, used)| {
            Arc::strong_count(shadow) > 1 || crate::host::references(&shadow.buffer) > 1 || *used + LATELY > now
        });
    }
    let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(256))?;
    crate::host::name(&buffer, || format!("title view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, address, len));
    if let Some(uses) = &shadow.uses {
        uses.mark_view();
    }
    map.insert((address, len), (shadow.clone(), now));
    EXACT_STATS[1].fetch_add(1, Ordering::Relaxed);
    EXACT_STATS[2].fetch_add(1, Ordering::Relaxed);
    OWN_MADE.fetch_add(1, Ordering::Relaxed);
    Some(shadow)
}

/// The bytes of a block that copies of single views' bytes are taken from
/// ([`exact_in_blocks`]). A copy that needs more than a quarter of one, with
/// its place, gets a block of its own.
pub(crate) const BLOCK: u64 = 4 << 20;

/// Blocks with nothing in them kept for the next copies; the others go.
const SPARE_BLOCKS: usize = 16;

/// A copy in blocks: its start, its bytes, and the multiple its place in
/// the block is of.
type Key = (u64, u64, u64);

/// A shared buffer that copies of single views' bytes lie in, one behind
/// the other.
struct Block {
    buffer: ID3D12Resource,
    /// Its mapping, and its address for shaders.
    data: usize,
    gpu: u64,
    size: u64,
    /// The first byte behind the last copy.
    top: u64,
    /// The copies in it.
    keys: Vec<Key>,
}

// SAFETY: a D3D12 object (free-threaded) and its mapping, used under the
// lock of the store that has it.
unsafe impl Send for Block {}

impl Block {
    /// Where a copy of `len` bytes at a multiple of `align` goes, if it fits.
    fn room(&self, len: u64, align: u64) -> Option<u64> {
        let at = self.top.checked_next_multiple_of(align)?;
        (at.checked_add(len)? <= self.size).then_some(at)
    }
}

/// Copies of single buffer views' bytes in blocks ([`exact_in_blocks`]).
struct Blocks {
    map: BTreeMap<Key, Arc<Shadow>>,
    /// `None`: a block let go; the next one made takes its place.
    blocks: Vec<Option<Block>>,
    /// The block new copies go into, and blocks with nothing in them.
    current: Option<usize>,
    spare: Vec<usize>,
}

/// The views' copies in blocks ([`exact_in_blocks`]).
static BLOCKS: Mutex<Blocks> = Mutex::new(Blocks::new());

impl Blocks {
    const fn new() -> Self {
        Blocks { map: BTreeMap::new(), blocks: Vec::new(), current: None, spare: Vec::new() }
    }

    /// A kept copy of the bytes `key` names at a multiple of its multiple
    /// (one placed for another view of the same bytes may do: an unordered
    /// view and another of the same bytes then share it, as with [`exact`]).
    fn kept(&self, (address, len, align): Key) -> Option<Arc<Shadow>> {
        let mut same = self.map.range((address, len, 0)..=(address, len, u64::MAX));
        let (_, shadow) = same.find(|(_, s)| s.lead.is_multiple_of(align))?;
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        Some(shadow.clone())
    }

    /// The copy `key` names: a kept one, or one placed now in the current
    /// block (or a spare one). `Err`: the bytes of the block to make first.
    fn take(&mut self, key: Key) -> Result<Arc<Shadow>, u64> {
        let (_, len, align) = key;
        if let Some(shadow) = self.kept(key) {
            return Ok(shadow);
        }
        if len.saturating_add(align) > BLOCK / 4 {
            return Err(len.next_multiple_of(256));
        }
        loop {
            let Some(i) = self.current.or_else(|| self.spare.pop()) else {
                return Err(BLOCK);
            };
            self.current = Some(i);
            if let Some(at) = self.blocks.get(i).and_then(Option::as_ref).and_then(|b| b.room(len, align))
                && let Some(shadow) = self.place(i, key, at)
            {
                return Ok(shadow);
            }
            // Full: the sweep empties it once nothing holds its copies.
            self.current = None;
        }
    }

    /// A copy for `key` at byte `at` of block `i`.
    fn place(&mut self, i: usize, key: Key, at: u64) -> Option<Arc<Shadow>> {
        let block = self.blocks.get_mut(i)?.as_mut()?;
        let (address, len, _) = key;
        let data = (block.data + at as usize) as *mut u8;
        let mut shadow = Shadow::new(block.buffer.clone(), data, block.gpu + at, address, len);
        shadow.lead = at;
        // `crate::byuse`: a view's own bytes, in the PC's memory as a block is.
        if let Some(uses) = &shadow.uses {
            uses.mark_view();
        }
        let shadow = Arc::new(shadow);
        block.top = at + len;
        block.keys.push(key);
        self.map.insert(key, shadow.clone());
        EXACT_STATS[1].fetch_add(1, Ordering::Relaxed);
        OWN_MADE.fetch_add(1, Ordering::Relaxed);
        Some(shadow)
    }

    /// Takes in `block`, made for `key` outside the lock, and gives the
    /// copy `key` names.
    fn add(&mut self, block: Block, key: Key) -> Result<Arc<Shadow>, u64> {
        EXACT_STATS[2].fetch_add(1, Ordering::Relaxed);
        let shared = block.size == BLOCK;
        let i = match self.blocks.iter().position(Option::is_none) {
            Some(i) => {
                self.blocks[i] = Some(block);
                i
            }
            None => {
                self.blocks.push(Some(block));
                self.blocks.len() - 1
            }
        };
        if shared {
            self.spare.push(i);
            return self.take(key);
        }
        // A block of its own; another thread may have made the copy first.
        if let Some(shadow) = self.kept(key) {
            self.blocks[i] = None;
            return Ok(shadow);
        }
        self.place(i, key, 0).ok_or(BLOCK)
    }

    /// Empties the blocks none of whose copies anything holds but the store:
    /// no list (the store holds the copy alone) and no slot (the block and
    /// its copies alone hold the buffer). Blocks of their own go, and spare
    /// ones past [`SPARE_BLOCKS`].
    fn sweep(&mut self) {
        let Blocks { map, blocks, current, spare, .. } = self;
        for (i, place) in blocks.iter_mut().enumerate() {
            let Some(block) = place else {
                continue;
            };
            if block.keys.is_empty()
                || crate::host::references(&block.buffer) as usize != 1 + block.keys.len()
                || block.keys.iter().any(|k| map.get(k).is_some_and(|s| Arc::strong_count(s) > 1))
            {
                continue;
            }
            for key in block.keys.drain(..) {
                map.remove(&key);
            }
            block.top = 0;
            if block.size != BLOCK {
                *place = None;
            } else if *current != Some(i) {
                spare.push(i);
            }
        }
        while spare.len() > SPARE_BLOCKS {
            if let Some(i) = spare.pop() {
                blocks[i] = None;
            }
        }
    }

    /// Copies kept, and blocks held.
    fn held(&self) -> (usize, usize) {
        (self.map.len(), self.blocks.iter().flatten().count())
    }
}

/// [`exact`] with `VICTORY_TWIN_BYTES=1`: the copy lies in a block of them
/// at the next multiple of `align` (of 16 bytes and of the view's
/// elements: the view is made there), and a host buffer is made only when
/// a block is full. The store's lock is held to look and to place; blocks
/// are made outside it, and emptied once a frame ([`exact_frame`]).
pub(crate) fn exact_in_blocks(address: u64, len: u64, align: u64) -> Option<Arc<Shadow>> {
    if !exact_fits(address, len) {
        return None;
    }
    EXACT_STATS[0].fetch_add(1, Ordering::Relaxed);
    in_blocks(&BLOCKS, (address, len, align), |size| host()?.shared_buffer(size))
}

/// [`exact_in_blocks`] in `store`; `make` makes a block's buffer.
fn in_blocks(
    store: &Mutex<Blocks>,
    key: Key,
    make: impl Fn(u64) -> Option<(ID3D12Resource, *mut u8)>,
) -> Option<Arc<Shadow>> {
    let lock = || store.lock().unwrap_or_else(|e| e.into_inner());
    let mut size = match lock().take(key) {
        Ok(shadow) => return Some(shadow),
        Err(size) => size,
    };
    // (Again if other threads filled the block made here first.)
    for _ in 0..4 {
        let (buffer, data) = make(size)?;
        crate::host::name(&buffer, || "copies of single buffer views' bytes".into());
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let block = Block { buffer, data: data as usize, gpu, size, top: 0, keys: Vec::new() };
        match lock().add(block, key) {
            Ok(shadow) => return Some(shadow),
            Err(next) => size = next,
        }
    }
    None
}

/// Once a frame: the blocks of [`exact_in_blocks`] whose copies nothing
/// holds are emptied for the next ones.
pub(crate) fn exact_frame() {
    BLOCKS.lock().unwrap_or_else(|e| e.into_inner()).sweep();
}

/// Copies of single views' bytes made since the start.
pub(crate) static OWN_MADE: AtomicU64 = AtomicU64::new(0);

/// With the frame line: the buffer views that got a copy of their own
/// bytes ([`exact`]), per frame over `frames`, and the copies kept now.
/// Nothing while there are none.
pub(crate) fn exact_line(frames: f64) {
    let [asked, made, buffers] = [0, 1, 2].map(|i| EXACT_STATS[i].swap(0, Ordering::Relaxed));
    if asked == 0 {
        return;
    }
    let (in_blocks, blocks) = BLOCKS.lock().unwrap_or_else(|e| e.into_inner()).held();
    let kept = EXACT.lock().unwrap_or_else(|e| e.into_inner()).1.len() + in_blocks;
    tracing::info!(
        "buffer views off their copy's elements (they were refused and read zeros; VICTORY_VIEW_BYTES=0 refuses them again): \
         {:.1} asked for per frame, {:.1} copies of their own bytes made per frame, {kept} kept; \
         {:.2} host buffers made for them per frame, {blocks} blocks held (VICTORY_TWIN_BYTES=1: copies in blocks of 4 MiB)",
        asked as f64 / frames,
        made as f64 / frames,
        buffers as f64 / frames
    );
}

#[cfg(test)]
mod block_tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    use windows::core::Interface;

    use super::{Blocks, in_blocks};

    /// Views off 16 bytes with `VICTORY_TWIN_BYTES=1`: the copies of their
    /// own bytes lie in one block, each at a multiple of 16 and of its
    /// view's elements, and a view made there reads the title's bytes. One
    /// host buffer for all of them; emptied for new copies once nothing
    /// holds them, not while a list holds a copy or a slot the buffer.
    #[test]
    fn views_off_16_bytes_share_a_block() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let store = Mutex::new(Blocks::new());
        let made = AtomicU64::new(0);
        let make = |size| {
            made.fetch_add(1, Ordering::Relaxed);
            host.shared_buffer(size)
        };
        let title: Vec<u8> = (0..1usize << 16).map(|i| (i * 7 + i / 251) as u8).collect();
        let base = (title.as_ptr() as u64).next_multiple_of(4);
        // What the copy's buffer holds where a view of it begins.
        let seen = |copy: &super::Shadow, address: u64, len: u64| {
            let mut mapped = std::ptr::null_mut();
            // SAFETY: a live mapped buffer, mapped again; the bytes lie in it.
            unsafe {
                copy.buffer.Map(0, None, Some(&mut mapped)).unwrap();
                let bytes = std::slice::from_raw_parts(mapped.cast::<u8>().add(copy.offset(address) as usize), len as usize).to_vec();
                copy.buffer.Unmap(0, None);
                bytes
            }
        };
        let mut copies = Vec::new();
        // Raw views of 128 bytes and structured ones of four 456-byte
        // elements (their multiple: 912), at 8 + 24 i.
        for i in 0..64u64 {
            for (len, align) in [(128u64, 16u64), (4 * 456, 912)] {
                let address = base + 8 + 24 * i;
                let copy = in_blocks(&store, (address, len, align), make).expect("a copy");
                assert!(copy.offset(address).is_multiple_of(align), "placed at {}", copy.offset(address));
                copy.update(address as usize, len as usize);
                let at = (address - title.as_ptr() as u64) as usize;
                assert_eq!(seen(&copy, address, len), title[at..at + len as usize], "view at {i}, {len} bytes");
                copies.push(copy);
            }
        }
        // Asked again: the same copy; and for a view of the same bytes on
        // 16 bytes, the structured view's.
        let again = in_blocks(&store, (base + 8, 128, 16), make).unwrap();
        assert!(std::sync::Arc::ptr_eq(&again, &copies[0]));
        let raw = in_blocks(&store, (base + 8, 4 * 456, 16), make).unwrap();
        assert!(std::sync::Arc::ptr_eq(&raw, &copies[1]));
        drop((again, raw));
        let one = copies[0].buffer.as_raw();
        assert!(copies.iter().all(|c| c.buffer.as_raw() == one), "one block");
        assert_eq!(made.load(Ordering::Relaxed), 1, "one host buffer for 128 views");
        // A list holds a copy, then a slot shows the block: kept.
        let listed = copies[5].clone();
        let shown = copies[0].buffer.clone();
        copies.clear();
        store.lock().unwrap().sweep();
        assert_eq!(store.lock().unwrap().held(), (128, 1));
        drop(listed);
        store.lock().unwrap().sweep();
        assert_eq!(store.lock().unwrap().held(), (128, 1));
        drop(shown);
        store.lock().unwrap().sweep();
        assert_eq!(store.lock().unwrap().held(), (0, 1));
        // Emptied: the next copy begins the block again, in the same buffer.
        let next = in_blocks(&store, (base + 4, 64, 16), make).unwrap();
        assert_eq!((next.offset(base + 4), next.buffer.as_raw(), made.load(Ordering::Relaxed)), (0, one, 1));
        // A view too big to share a block gets one of its own, from its
        // first byte, let go once nothing holds it.
        let big = in_blocks(&store, (base, 2 << 20, 16), make).unwrap();
        assert_eq!((big.offset(base), made.load(Ordering::Relaxed)), (0, 2));
        assert_ne!(big.buffer.as_raw(), one);
        drop((big, next));
        store.lock().unwrap().sweep();
        assert_eq!(store.lock().unwrap().held(), (0, 1));
    }
}

#[cfg(test)]
mod window_tests {
    use super::{
        WINDOW, cover, default_card_limits_mb, grid, holder, longest_tail, resolve_card_limits,
        LetGoBook, NAMED_FRAMES, Named, Spares, WindowRule, tail_of, window_of, window_rule,
        window_victims, window_victims_over,
    };
    use crate::host::RamBand;

    const M: u64 = 1 << 20;

    // Ticket 156 (from Vice-Tay f3dfb5b9's tests, with ours).
    #[test]
    fn only_the_explicit_switch_reserves_full_budgets() {
        use super::{parse_mib, reservation_limits};
        assert_eq!(reservation_limits(1536 * M, true, 1024 * M, false), (1024 * M, 0));
        assert_eq!(reservation_limits(1536 * M, true, 1024 * M, true), (1536 * M, 1024 * M));
        // A missing or unreadable budget keeps the 1 GiB cap.
        assert_eq!(parse_mib(" 1536 "), Some(1536));
        assert_eq!(parse_mib("not a number"), None);
        assert_eq!(reservation_limits(2048 * M, false, 1024 * M, true), (1024 * M, 1024 * M));
        assert_eq!(reservation_limits(512 * M, false, 0, false), (512 * M, 0));
        // Off: as before the switch, whatever the windows' budget.
        assert_eq!(reservation_limits(512 * M, false, 2048 * M, false), (512 * M, 0));
    }

    #[test]
    fn texture_admission_reserves_buffers_and_windows_independently() {
        use super::remaining_card_room;
        assert_eq!(remaining_card_room(1536 * M, 512 * M, 0, 1024 * M, 256 * M), 1792 * M);
        assert_eq!(remaining_card_room(512 * M, 256 * M, 1024 * M, 1024 * M, 256 * M), 768 * M);
        assert_eq!(remaining_card_room(0, u64::MAX, 1, 0, 0), 0);
        // Windows past their budget (copies by address count with them) keep nothing and take nothing back.
        assert_eq!(remaining_card_room(512 * M, 0, 0, 1024 * M, 3000 * M), 512 * M);
        // As the windows land what is kept falls by what the card's use rises: the room rule's count stays.
        let used = 5000 * M;
        let counted = |landed: u64| used + landed + remaining_card_room(0, 0, 0, 2048 * M, landed);
        assert_eq!(counted(0), counted(1024 * M));
        assert_eq!(counted(0), counted(2048 * M));
    }

    #[test]
    fn windows_go_to_the_card_under_80_90_or_97_hundredths() {
        use super::{parse_window_cap, window_cap_percent, window_usage_limit};
        // Not reserved, no cap: 80, to the byte as before.
        assert_eq!(window_usage_limit(1_099, false, None), 872);
        assert_eq!(window_usage_limit(9000 * M, false, None), 9000 * M / 10 * 8);
        assert_eq!(window_cap_percent(false, None), 80);
        // Reserved: 97.
        assert_eq!(window_usage_limit(1_099, true, None), 970);
        assert_eq!(window_usage_limit(9000 * M, true, None), 9000 * M / 100 * 97);
        assert_eq!(window_cap_percent(true, None), 97);
        // A cap asked for is the line either way.
        assert_eq!(window_usage_limit(9000 * M, false, Some(90)), 9000 * M / 100 * 90);
        assert_eq!(window_usage_limit(9000 * M, true, Some(90)), 9000 * M / 100 * 90);
        assert_eq!(window_usage_limit(9000 * M, true, Some(80)), 9000 * M / 100 * 80);
        assert_eq!(window_usage_limit(9000 * M, false, Some(97)), window_usage_limit(9000 * M, true, None));
        assert_eq!(window_usage_limit(0, true, None), 0);
        assert_eq!(window_usage_limit(u64::MAX, true, None), u64::MAX / 100 * 97);
        // The switch's value.
        assert_eq!(parse_window_cap("90"), Some(90));
        assert_eq!(parse_window_cap(" 80 "), Some(80));
        assert_eq!(parse_window_cap("100"), Some(97));
        assert_eq!(parse_window_cap("10"), Some(50));
        assert_eq!(parse_window_cap("ninety"), None);
        assert_eq!(parse_window_cap(""), None);
    }

    #[test]
    fn a_budget_that_changes_mid_run_moves_the_window_line_not_the_reservation() {
        use super::{remaining_card_room, reservation_limits, window_usage_limit};
        // A window of 4 MiB with 7,000 MiB of the card in use.
        let fits = |budget: u64, reserved: bool, cap| 7000 * M + 4 * M <= window_usage_limit(budget, reserved, cap);
        // 9,000 MiB: the line is 7,200 (80), 8,100 (90), 8,730 (97).
        assert!(fits(9000 * M, false, None));
        assert!(fits(9000 * M, false, Some(90)));
        assert!(fits(9000 * M, true, None));
        // The system cuts the budget to 8,000 MiB: 6,400, 7,200, 7,760.
        assert!(!fits(8000 * M, false, None));
        assert!(fits(8000 * M, false, Some(90)));
        assert!(fits(8000 * M, true, None));
        // To 7,000 MiB: no window any more under any line.
        assert!(!fits(7000 * M, false, None));
        assert!(!fits(7000 * M, false, Some(90)));
        assert!(!fits(7000 * M, true, None));
        // And back up.
        assert!(fits(9000 * M, false, None));
        // What is kept is in bytes of the switches' budgets: the same under every budget.
        let (buffers, windows) = reservation_limits(1024 * M, true, 2048 * M, true);
        let kept = remaining_card_room(buffers, 800 * M, 0, windows, 1500 * M);
        assert_eq!(kept, 224 * M + 548 * M);
        // The room rule's count (the card's use with what is kept) against 90 hundredths of each budget.
        let counted = 7000 * M + kept;
        assert!(counted < 9000 * M / 100 * 90);
        assert!(counted >= 8000 * M / 100 * 90);
    }

    #[test]
    fn the_reserve_line_says_what_is_kept_and_the_line() {
        use super::reserve_line_of;
        let parts = [[1024 * M, 224 * M], [0, 0], [2048 * M, 548 * M]];
        assert_eq!(
            reserve_line_of(true, parts, Some([9000 * M, 7000 * M]), None, true, [31_000, 120]),
            "card reserve: on; reserved 772 MB (buffer copies 224 of 1024 MB, card windows 0 of 0 MB, mapped windows 548 of 2048 MB); \
             budget 9000 MB, card in use 7000 MB, windows go to the card under 8730 MB (97 hundredths); \
             textures let go by the budget pass 31000 (120 since the last line)"
        );
        assert_eq!(
            reserve_line_of(false, [[512 * M, 100 * M], [0, 0], [0, 0]], Some([9000 * M, 7000 * M]), None, false, [5, 0]),
            "card reserve: off; reserved 100 MB (buffer copies 100 of 512 MB, card windows 0 of 0 MB, mapped windows 0 of 0 MB); \
             budget 9000 MB, card in use 7000 MB, windows go to the card under 7200 MB (80 hundredths); \
             textures let go by the budget pass 5 (0 since the last line)"
        );
        assert!(reserve_line_of(false, [[0; 2]; 3], None, Some(90), false, [0, 0]).contains("the card's memory is not known"));
    }

    fn rule(idle_ms: u64, line: u64, over_alive: bool) -> WindowRule {
        WindowRule { idle_ms, line, over_alive }
    }

    #[test]
    fn without_the_switch_the_rule_is_as_before() {
        // No budget: the clock alone, whatever the band.
        for band in [None, Some(RamBand::Calm), Some(RamBand::Watch), Some(RamBand::High)] {
            assert_eq!(window_rule(false, band, None, 5_000, 999 * M), rule(5_000, u64::MAX, false));
        }
        // A budget caps the Watch and High bands only, over the windows
        // that may go.
        let budget = Some(2048 * M);
        assert_eq!(window_rule(false, Some(RamBand::Calm), budget, 300_000, 0), rule(300_000, u64::MAX, false));
        assert_eq!(window_rule(false, None, budget, 300_000, 0), rule(300_000, u64::MAX, false));
        assert_eq!(window_rule(false, Some(RamBand::Watch), budget, 300_000, 0), rule(10_000, 2048 * M, false));
        assert_eq!(window_rule(false, Some(RamBand::High), budget, 5_000, 0), rule(2_000, 1024 * M, false));
    }

    #[test]
    fn with_the_switch_windows_go_by_bytes_and_never_under_30_s() {
        // The line: the most named in a frame plus a quarter, over all
        // windows alive; never under 30 s idle, the longer clock kept.
        for band in [None, Some(RamBand::Calm), Some(RamBand::Watch)] {
            assert_eq!(window_rule(true, band, None, 5_000, 400 * M), rule(30_000, 500 * M, true));
            assert_eq!(window_rule(true, band, None, 300_000, 400 * M), rule(300_000, 500 * M, true));
        }
        // A budget is the line.
        assert_eq!(window_rule(true, Some(RamBand::Watch), Some(2048 * M), 2_000, 400 * M), rule(30_000, 2048 * M, true));
        // The High band keeps today's 2 s.
        assert_eq!(window_rule(true, Some(RamBand::High), None, 5_000, 400 * M), rule(2_000, u64::MAX, false));
        assert_eq!(window_rule(true, Some(RamBand::High), Some(2048 * M), 5_000, 0), rule(2_000, 1024 * M, false));
    }

    #[test]
    fn the_line_counts_every_window_alive() {
        // 100 MiB alive, of which these three (60 MiB) may go, oldest first.
        let entries = [(1u64, 10 * M, 1usize), (2, 20 * M, 2), (3, 30 * M, 3)];
        // Under the line: none goes.
        assert!(window_victims_over(&entries, 100 * M, 100 * M).is_empty());
        // 20 MiB over: the two oldest (10 would leave 90).
        assert_eq!(window_victims_over(&entries, 100 * M, 80 * M), vec![0, 1]);
        // Far over: all that may go, and no more.
        assert_eq!(window_victims_over(&entries, 100 * M, 0), vec![0, 1, 2]);
        // One copy under two keys counts once.
        let twice = [(1u64, 30 * M, 7usize), (2, 30 * M, 7), (3, 20 * M, 8)];
        assert_eq!(window_victims_over(&twice, 100 * M, 50 * M), vec![0, 2]);
    }

    #[test]
    fn named_keeps_the_peak_of_the_last_600_frames() {
        let mut named = Named::new();
        assert_eq!(named.peak(), 0);
        named.add(1, 10 * M);
        named.add(1, 5 * M);
        assert_eq!(named.peak(), 15 * M);
        named.add(2, 3 * M);
        assert_eq!(named.peak(), 15 * M);
        // A thread a frame late counts in the frame begun.
        named.add(1, 20 * M);
        assert_eq!(named.peak(), 23 * M);
        // 600 frames on, frame 1's bytes are out of the look back.
        for frame in 3..3 + NAMED_FRAMES as u64 {
            named.add(frame, M);
        }
        assert_eq!(named.peak(), M);
        assert_eq!(named.before.len(), NAMED_FRAMES - 1);
    }

    #[test]
    fn the_spare_list_holds_64_and_256_mb_at_most() {
        let mut spares = Spares::new();
        for i in 0..Spares::<u32>::MOST as u32 {
            assert!(spares.give(i, 4 * M).is_none());
        }
        assert_eq!(spares.bytes, 256 * M);
        assert_eq!(spares.give(99, 4 * M), Some(99));
        assert!(!spares.room(1));
        // A buffer is taken only of its own size.
        assert_eq!(spares.take(8 * M), None);
        assert!(spares.take(4 * M).is_some());
        assert_eq!(spares.bytes, 252 * M);
        assert_eq!(spares.drain().len(), Spares::<u32>::MOST - 1);
        assert_eq!(spares.bytes, 0);
        assert_eq!(spares.take(4 * M), None);
        // The bytes bind before the count.
        let mut big = Spares::new();
        assert!(big.give(1u32, 200 * M).is_none());
        assert_eq!(big.give(2, 100 * M), Some(2));
        assert!(big.give(3, 56 * M).is_none());
    }

    #[test]
    fn a_window_made_within_a_minute_of_its_let_go_counts_once() {
        let mut book = LetGoBook::new();
        let key = (0x4000_0000, 4 * M);
        book.gone(key, 1_000);
        assert!(!book.made((0x4040_0000, 4 * M), 2_000));
        assert!(book.made(key, 61_000));
        // Counted once: the entry is gone.
        assert!(!book.made(key, 61_000));
        book.gone(key, 1_000);
        assert!(!book.made(key, 61_001));
        // Old entries go at the next let-go.
        book.gone(key, 0);
        book.gone((0, 4 * M), 70_000);
        assert_eq!(book.gone.len(), 1);
    }

    #[test]
    fn window_budget_evicts_oldest_only_until_it_fits() {
        const M: u64 = 1 << 20;
        // Oldest first: 10, 20 and 30 MiB. Keeping 50 MiB must free only the
        // oldest copy (60 MiB left before, 50 MiB after).
        let entries = [(1u64, 10 * M, 1usize), (2, 20 * M, 2), (3, 30 * M, 3)];
        assert_eq!(window_victims(&entries, 50 * M), vec![0]);
        // A tighter cap takes the newest as well.
        assert_eq!(window_victims(&entries, 20 * M), vec![0, 1, 2]);
    }

    #[test]
    fn window_budget_counts_one_copy_under_several_keys_once() {
        const M: u64 = 1 << 20;
        // The first two entries are one copy; 50 MiB is live, not 80.
        // Without dedupe both would be picked to reach a 40 MiB cap.
        let entries = [(1u64, 30 * M, 7usize), (2, 30 * M, 7), (3, 20 * M, 8)];
        assert_eq!(window_victims(&entries, 40 * M), vec![0]);
    }

    #[test]
    fn card_limits_scale_with_local_vram_budget() {
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;

        // Unknown (WARP / query failure) and <= 12 GiB keep the 12 GB baseline.
        assert_eq!(default_card_limits_mb(None), (16, 512));
        assert_eq!(default_card_limits_mb(Some(8 * GIB)), (16, 512));
        assert_eq!(default_card_limits_mb(Some(11 * GIB)), (16, 512));
        assert_eq!(default_card_limits_mb(Some(12 * GIB)), (16, 512));

        // 16 GB tier (> 12 GiB, < 20 GiB), including WDDM budgets around 13.5–15 GiB.
        assert_eq!(default_card_limits_mb(Some(13 * GIB)), (4, 2048));
        assert_eq!(default_card_limits_mb(Some(14 * GIB)), (4, 2048));
        assert_eq!(default_card_limits_mb(Some(15 * GIB)), (4, 2048));
        assert_eq!(default_card_limits_mb(Some(16 * GIB)), (4, 2048));

        // 24–32+ GB tier (>= 20 GiB): 1 MiB min, 25% of budget capped at 8192 MiB.
        assert_eq!(default_card_limits_mb(Some(20 * GIB)), (1, 5120));
        assert_eq!(default_card_limits_mb(Some(24 * GIB)), (1, 6144));
        assert_eq!(default_card_limits_mb(Some(32 * GIB)), (1, 8192));
        assert_eq!(default_card_limits_mb(Some(48 * GIB)), (1, 8192));

        // Explicit env overrides always win, clamp min to >= 1 MiB, and saturate on overflow.
        assert_eq!(resolve_card_limits(Some(32 * GIB), Some(8), Some(4096)), (8 * MIB, 4096 * MIB));
        assert_eq!(resolve_card_limits(Some(12 * GIB), Some(0), Some(1024)), (MIB, 1024 * MIB));
        assert_eq!(resolve_card_limits(None, Some(u64::MAX), Some(u64::MAX)), (u64::MAX, u64::MAX));
        assert_eq!(resolve_card_limits(None, None, None), (16 * MIB, 512 * MIB));
    }

    /// An ask that may take any window takes the longest that holds all of
    /// it, and none that holds only a part.
    #[test]
    fn an_ask_takes_the_longest_window_that_holds_it() {
        let cell = 0xd_1200_0000u64;
        let (ask, end) = (cell + 0x9_0000, cell + 0xb_0000);
        assert_eq!(holder([].into_iter(), ask, end), None);
        let there = [
            (cell, 0x4_0000),            // ends below the ask
            (cell + 0x8_0000, 0x2_0000), // holds its first half only
            (cell + 0x8_0000, 0x4_0000), // holds it
            (cell + 0x4_0000, 0x10_0000), // holds it, longer
            (cell + 0xa_0000, 0x40_0000), // begins above it
        ];
        assert_eq!(holder(there.into_iter(), ask, end), Some((cell + 0x4_0000, 0x10_0000)));
        // Of two as long, the one that begins lower.
        let two = [(cell + 0x8_0000, 0x8_0000), (cell + 0x4_0000, 0x8_0000)];
        assert_eq!(holder(two.into_iter(), ask, end), Some((cell + 0x4_0000, 0x8_0000)));
    }

    /// What windows lie over counts each byte once.
    #[test]
    fn overlapping_windows_cover_their_memory_once() {
        assert_eq!(cover([].into_iter()), 0);
        let apart = [(0x10_0000u64, 0x4_0000u64), (0x20_0000, 0x4_0000)];
        assert_eq!(cover(apart.into_iter()), 0x8_0000);
        // One inside another, one overlapping its end, one touching.
        let over = [
            (0x10_0000u64, 0x10_0000u64),
            (0x14_0000, 0x4_0000),
            (0x1c_0000, 0x8_0000),
            (0x24_0000, 0x4_0000),
        ];
        assert_eq!(cover(over.into_iter()), 0x18_0000);
    }

    /// A tail runs from the ask's page to the end of its 4 MiB cell, and an
    /// ask takes the longest tail there is that holds it.
    #[test]
    fn tails_run_to_the_cells_end_and_the_longest_is_taken() {
        let cell = 0xd_1200_0000u64;
        let ask = cell + 0x12_3456;
        assert_eq!(tail_of(ask, ask + 1), (cell + 0x10_0000, WINDOW - 0x10_0000));
        // The cell's first page: the whole cell, the window there always was.
        assert_eq!(tail_of(cell + 0xffff, cell + 0x1_0000), (cell, WINDOW));
        let end = cell + WINDOW;
        let tail = |from: u64| (cell + from, WINDOW - from);
        // Nothing there, or only tails that begin above the ask.
        assert_eq!(longest_tail([].into_iter(), ask, ask + 1), None);
        assert_eq!(longest_tail([tail(0x20_0000)].into_iter(), ask, ask + 1), None);
        // The longest of those that hold the ask; windows that do not reach
        // the cell's end (cut on a finer grid for asks with a length) and
        // windows of other cells are not tails.
        let there = [
            (cell - WINDOW, WINDOW),
            (cell, 0x4_0000),
            tail(0x8_0000),
            tail(0x10_0000),
            (cell + 0x10_0000, 0x4_0000),
            tail(0x20_0000),
        ];
        assert_eq!(longest_tail(there.into_iter(), ask, ask + 1), Some(tail(0x8_0000)));
        let (start, bytes) = tail(0x8_0000);
        assert!(start <= ask && start + bytes == end);
        // The whole cell's window is the longest tail of all.
        let with_whole = [(cell, WINDOW), tail(0x8_0000)];
        assert_eq!(longest_tail(with_whole.into_iter(), ask, ask + 1), Some((cell, WINDOW)));
    }

    /// The setting takes powers of two from 64 KiB to 4 MiB; anything else
    /// leaves the grid as it was.
    #[test]
    fn the_grid_is_a_power_of_two_or_the_default() {
        assert_eq!(grid(None), WINDOW);
        assert_eq!(grid(Some(4096)), WINDOW);
        assert_eq!(grid(Some(1024)), 1 << 20);
        assert_eq!(grid(Some(256)), 256 << 10);
        assert_eq!(grid(Some(64)), 64 << 10);
        for bad in [0, 32, 100, 1000, 8192] {
            assert_eq!(grid(Some(bad)), WINDOW, "{bad}");
        }
    }

    /// A window holds its ask and is cut on the grid; on the default grid
    /// it is the window there always was.
    #[test]
    fn windows_hold_their_ask_on_the_grid() {
        let (address, end) = (0xd_1234_5678u64, 0xd_1234_5678u64 + 0x9000);
        assert_eq!(window_of(address, end, WINDOW), (0xd_1200_0000, 4 << 20));
        assert_eq!(window_of(address, end, 1 << 20), (0xd_1230_0000, 1 << 20));
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 256 << 10));
        // An ask across a grid line takes both cells.
        let end = address + 0x4_0000;
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 512 << 10));
        for g in [64u64 << 10, 256 << 10, 1 << 20, WINDOW] {
            let (start, len) = window_of(address, end, g);
            assert!(start <= address && start + len >= end);
            assert!(start % g == 0 && len % g == 0);
        }
    }
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}
