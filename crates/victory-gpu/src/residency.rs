//! Host-memory admission for speculative texture work and reusable upload storage.
//!
//! From codec-d/cool_fox (641835a4, fabb872a, c64fc9b4, 08e4574d, e94d277a; the
//! values of the last). Only with `VICTORY_RESIDENCY=1` ([`on`]): the memory
//! of the system is then asked once a second; below 3 GiB of free RAM or 4 GiB
//! of commit (out again above 5 / 6) texture prefetch is limited to 64 MiB a
//! frame and idle scratch and staging blocks are let go; below 1.5 GiB of
//! commit to 16 MiB; the texture pass stops below 512 MiB of commit, new
//! staging blocks for textures picked by index below 1 GiB. With 32 GiB of RAM
//! or less (or `VICTORY_RESIDENCY_LIMITED=1`) the budgets of a smaller PC.
//! Off, nothing here changes anything, except that for 500 ms after the card
//! refused memory (E_OUTOFMEMORY) the texture pass stops ([`upload_refused`]).
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

const MIB: u64 = 1 << 20;

/// The memory budgets of this module (`VICTORY_RESIDENCY`). On unless `=0`
/// here, as in the fox base this tree plays from (off by default upstream,
/// where the comments' `VICTORY_RESIDENCY=1` comes from).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RESIDENCY").is_none_or(|v| v != "0"))
}
const GIB: u64 = 1 << 30;
static PRESSURED: AtomicBool = AtomicBool::new(false);
static HEADROOM: AtomicU64 = AtomicU64::new(u64::MAX);

fn memory_status() -> Option<MEMORYSTATUSEX> {
    // SAFETY: the API initializes the fields of a correctly sized structure.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
}

// The texture sweep may run for seconds inside a single submitted frame.
static NEXT_COMMIT_SAMPLE: AtomicU64 = AtomicU64::new(0);
static AVAILABLE_COMMIT: AtomicU64 = AtomicU64::new(u64::MAX);
static RETRY_UPLOAD_AT: AtomicU64 = AtomicU64::new(0);

pub(crate) fn allocation_error(error: &windows::core::Error) {
    if error.code() == windows::Win32::Foundation::E_OUTOFMEMORY {
        RETRY_UPLOAD_AT.fetch_max(crate::host::coarse_ms().saturating_add(500), Ordering::Relaxed);
    }
}

thread_local! {
    static STAGING_DEFERRED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn note_staging_deferred() {
    STAGING_DEFERRED.set(true);
}

pub(crate) fn take_staging_deferred() -> bool {
    STAGING_DEFERRED.replace(false)
}

pub(crate) fn upload_refused() -> bool {
    crate::host::coarse_ms() < RETRY_UPLOAD_AT.load(Ordering::Relaxed)
}

fn sample_commit(now: u64) -> u64 {
    let next = NEXT_COMMIT_SAMPLE.load(Ordering::Relaxed);
    if now >= next
        && NEXT_COMMIT_SAMPLE
            .compare_exchange(next, now.saturating_add(32), Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        && let Some(status) = memory_status()
    {
        AVAILABLE_COMMIT.store(status.ullAvailPageFile, Ordering::Relaxed);
    }
    AVAILABLE_COMMIT.load(Ordering::Relaxed)
}

fn has_upload_reserve(commit: u64, bake: bool) -> bool {
    // Keep a protective floor against Rust/game OOM aborts without starving
    // bindless world textures or one-shot ped decoration bakes when steady-state
    // commit headroom sits in the 1-4 GiB range on 32 GiB hosts.
    commit >= if bake { 256 * MIB } else { 512 * MIB }
}

fn has_growth_reserve(commit: u64, bake: bool) -> bool {
    // Growing a fresh 64 MiB upload block for speculative bindless prefetch
    // requires more headroom than reusing slices already committed inside the
    // staging pool.
    commit >= if bake { 256 * MIB } else { 1024 * MIB }
}

/// Leave commit for game allocations and submissions already in flight while
/// still letting bindless textures and one-shot material bakes upload.
pub(crate) fn admit_texture_pass(bake: bool) -> bool {
    let now = crate::host::coarse_ms();
    if now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) {
        note(Churn::RamRefused, 1);
        return false;
    }
    if !on() {
        return true;
    }
    let admitted = has_upload_reserve(sample_commit(now), bake);
    if !admitted {
        note(Churn::RamRefused, 1);
    }
    admitted
}

/// Whether the staging allocator may commit a brand-new OS upload buffer
/// (reusing existing free ranges in already-committed pool blocks stays open
/// whenever `admit_texture_pass` is true).
pub(crate) fn allow_staging_growth(bake: bool) -> bool {
    if !on() {
        return true;
    }
    let now = crate::host::coarse_ms();
    if now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) {
        return false;
    }
    has_growth_reserve(sample_commit(now), bake)
}

pub(crate) fn limited_host_ram() -> bool {
    static LIMITED: OnceLock<bool> = OnceLock::new();
    *LIMITED.get_or_init(|| {
        on() && (std::env::var_os("VICTORY_RESIDENCY_LIMITED").is_some_and(|v| v == "1")
            || memory_status().is_some_and(|s| s.ullTotalPhys <= 32 * GIB))
    })
}

fn next_pressure(was: bool, physical: u64, commit: u64) -> bool {
    if was {
        physical < 5 * GIB || commit < 6 * GIB
    } else {
        physical <= 3 * GIB || commit <= 4 * GIB
    }
}

/// Whether the idle texture sweep keeps what it lets go under PC memory
/// pressure in bounds, and pressure has a way out (`VICTORY_PRESSURE_KEEP=1`,
/// off: ticket 124). Unset, under pressure every idle host copy of a
/// texture the title fills goes after 5 s, all of them at one look, and
/// pressure clears above 5 GiB free and 6 GiB of commit: in session notwin
/// it tripped at 3,025 MiB free and never cleared, 190,767 textures went
/// and 31 % of them were made again, while private bytes came back to
/// where they were within 3.5 minutes. On,
/// - a texture goes after [`PRESSURE_KEEP_MS`] idle at least, its back-off
///   ([`backed_off`]) more, the longest idle first, [`PRESSURE_MOST`] at
///   most at a look (`crate::resource::let_go_idle`);
/// - only what gives memory back goes: the textures of pool blocks that
///   come back whole, and those with a staging buffer of their own; spare
///   staging blocks and snapshot buffers go before them, all of them;
/// - pressure ends at [`way_out`].
pub(crate) fn pressure_keep() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_PRESSURE_KEEP"))
}

/// Under pressure with [`pressure_keep`]: the least a texture is idle
/// before it goes, ms.
pub(crate) const PRESSURE_KEEP_MS: u64 = 30_000;
/// The most a texture is kept by its back-off ([`backed_off`]), ms.
const PRESSURE_KEEP_LONGEST_MS: u64 = 600_000;
/// The bytes of textures let go at one look, at most.
pub(crate) const PRESSURE_MOST: u64 = 128 << 20;
/// A texture made again within this many ms of its let-go was let go too
/// soon ([`made_again_soon`]).
const AGAIN_WITHIN_MS: u64 = 60_000;

/// How long a texture with back-off `keep_ms` (0: none yet) must be idle to
/// go under pressure: [`PRESSURE_KEEP_MS`] to [`PRESSURE_KEEP_LONGEST_MS`].
pub(crate) fn pressure_idle_ms(keep_ms: u32) -> u64 {
    u64::from(keep_ms).clamp(PRESSURE_KEEP_MS, PRESSURE_KEEP_LONGEST_MS)
}

/// Whether a texture made again `since_ms` after its let-go was let go too
/// soon.
pub(crate) fn made_again_soon(since_ms: u64) -> bool {
    since_ms < AGAIN_WITHIN_MS
}

/// A texture's back-off once it is made again `since_ms` after its let-go:
/// twice its keep ([`pressure_idle_ms`]) if that was too soon
/// ([`made_again_soon`]), up to [`PRESSURE_KEEP_LONGEST_MS`]; else as it was.
pub(crate) fn backed_off(keep_ms: u32, since_ms: u64) -> u32 {
    if !made_again_soon(since_ms) {
        return keep_ms;
    }
    // (At most ten minutes: fits.)
    (pressure_idle_ms(keep_ms) * 2).min(PRESSURE_KEEP_LONGEST_MS) as u32
}

/// Where pressure entered at free RAM and commit `entry` ends, with
/// [`pressure_keep`]: at 4 GiB free or 1 GiB above the free RAM at entry,
/// whichever is lower; the commit line likewise, at 5 GiB or 1 GiB above
/// the commit at entry. (Today's 5 and 6 GiB are not reached on a PC of
/// 32 GB whose other programs hold 7.4 of it.)
fn way_out(entry: [u64; 2]) -> [u64; 2] {
    let [physical, commit] = entry;
    [(4 * GIB).min(physical.saturating_add(GIB)), (5 * GIB).min(commit.saturating_add(GIB))]
}

/// [`next_pressure`] with [`pressure_keep`]: in as there, out at
/// [`way_out`] of `entry`.
fn next_pressure_kept(was: bool, physical: u64, commit: u64, entry: [u64; 2]) -> bool {
    if was {
        let [physical_out, commit_out] = way_out(entry);
        physical < physical_out || commit < commit_out
    } else {
        next_pressure(false, physical, commit)
    }
}

/// Free RAM and commit when pressure last began ([`next_pressure_kept`]).
static ENTRY: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// When pressure last began (`crate::host::coarse_ms`), and the ms under
/// pressure before it.
static PRESSURED_SINCE: AtomicU64 = AtomicU64::new(0);
static PRESSURED_MS: AtomicU64 = AtomicU64::new(0);

/// Seconds under pressure since the start, by the ms under it before and
/// since `since` (`None`: not under it now) until `now`.
fn pressured_secs(before_ms: u64, since: Option<u64>, now: u64) -> u64 {
    (before_ms + since.map_or(0, |since| now.saturating_sub(since))) / 1000
}

/// Called by the once-per-second submission sweep. A failed sample keeps the last good state.
pub(crate) fn refresh() {
    if !on() {
        return;
    }
    let Some(status) = memory_status() else { return };
    let (physical, commit) = (status.ullAvailPhys, status.ullAvailPageFile);
    HEADROOM.store(physical.min(commit), Ordering::Relaxed);
    let was = pressured();
    let entry = ENTRY.each_ref().map(|n| n.load(Ordering::Relaxed));
    let pressure = if pressure_keep() {
        next_pressure_kept(was, physical, commit, entry)
    } else {
        next_pressure(was, physical, commit)
    };
    if pressure != was {
        let now = crate::host::coarse_ms();
        if pressure {
            ENTRY[0].store(physical, Ordering::Relaxed);
            ENTRY[1].store(commit, Ordering::Relaxed);
            PRESSURED_SINCE.store(now, Ordering::Relaxed);
        } else {
            PRESSURED_MS.fetch_add(now.saturating_sub(PRESSURED_SINCE.load(Ordering::Relaxed)), Ordering::Relaxed);
        }
        PRESSURED.store(pressure, Ordering::Relaxed);
        if pressure {
            tracing::warn!(
                pid = std::process::id(),
                frame = crate::frame::current(),
                physical_free_mib = physical >> 20,
                available_commit_mib = commit >> 20,
                buffer_mirror_span_mib = crate::shadow::live_span_bytes() >> 20,
                texture_staging_mib = crate::texture::live_staging_bytes() >> 20,
                spare_scratch_mib = crate::host::ALIVE[4].load(Ordering::Relaxed) >> 20,
                game_filled_gpu_texture_mib = crate::resource::KINDS[1].load(Ordering::Relaxed) >> 20,
                "host memory pressure: limiting texture prefetch and idle upload storage"
            );
        } else {
            tracing::info!(physical_free_mib = physical >> 20, available_commit_mib = commit >> 20, "host memory pressure cleared");
        }
    }
}

pub(crate) fn headroom() -> Option<u64> {
    let bytes = HEADROOM.load(Ordering::Relaxed);
    (bytes != u64::MAX).then_some(bytes)
}

pub(crate) fn pressured() -> bool {
    PRESSURED.load(Ordering::Relaxed)
}

/// True when OS commit headroom is critically low (< 1.5 GiB) or an allocation
/// just failed with `E_OUTOFMEMORY`, requiring emergency throttling.
pub(crate) fn critical_commit() -> bool {
    if !on() {
        return false;
    }
    let now = crate::host::coarse_ms();
    now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) || sample_commit(now) < 1536 * MIB
}

/// A soft per-frame admission limit. A single texture can cross it. The
/// queue retains unserved slots so they can be retried on later frames.
pub(crate) fn prefetch_limit(bytes: u64, pressure: bool, critical: bool, limited: bool, staging: u64) -> u64 {
    if critical {
        bytes.min(16 << 20)
    } else if pressure || (limited && staging >= GIB) {
        bytes.min(64 << 20)
    } else {
        bytes
    }
}

/// What the residency line counts, since the start: textures' host copies
/// let go by the sweep while the card had room (idle by the clock alone) and
/// to come down under the card's budget; bytes uploaded into textures; and
/// texture passes the RAM guard refused. (Slots of textures picked by index
/// let go because their memory was no longer mapped are
/// `descriptor::host::STALE[4]`.)
#[derive(Clone, Copy, Debug)]
pub(crate) enum Churn {
    Idle,
    Budget,
    UploadBytes,
    RamRefused,
}

static CHURN: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Adds `n` to a residency count: one relaxed add.
pub(crate) fn note(what: Churn, n: u64) {
    CHURN[what as usize].fetch_add(n, Ordering::Relaxed);
}

/// The residency counts at one moment, all since the start.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Totals {
    pub(crate) idle: u64,
    pub(crate) budget: u64,
    pub(crate) unmap: u64,
    pub(crate) made_again: u64,
    pub(crate) upload_bytes: u64,
    /// Passes over textures picked by index that made none for want of
    /// the card's room.
    pub(crate) refused: u64,
    pub(crate) ram_refused: u64,
    /// Textures made again within a minute of their let-go
    /// ([`made_again_soon`]), and their bytes.
    pub(crate) again_soon: u64,
    pub(crate) again_soon_bytes: u64,
    /// The bytes of pool blocks given back (textures' and staging's).
    pub(crate) given_back_bytes: u64,
    pub(crate) pressured_secs: u64,
}

impl Totals {
    fn now() -> Self {
        let c = |w: Churn| CHURN[w as usize].load(Ordering::Relaxed);
        Totals {
            idle: c(Churn::Idle),
            budget: c(Churn::Budget),
            unmap: crate::descriptor::host::STALE[4].load(Ordering::Relaxed),
            made_again: crate::resource::KINDS[4].load(Ordering::Relaxed),
            upload_bytes: c(Churn::UploadBytes),
            refused: crate::draw::HEAP_TEXTURES_FULL.load(Ordering::Relaxed),
            ram_refused: c(Churn::RamRefused),
            again_soon: crate::resource::AGAIN_SOON[0].load(Ordering::Relaxed),
            again_soon_bytes: crate::resource::AGAIN_SOON[1].load(Ordering::Relaxed),
            given_back_bytes: crate::pool::given_back_bytes(),
            pressured_secs: pressured_secs(
                PRESSURED_MS.load(Ordering::Relaxed),
                pressured().then(|| PRESSURED_SINCE.load(Ordering::Relaxed)),
                crate::host::coarse_ms(),
            ),
        }
    }
}

/// The RAM guard's state, worst first: an allocation just failed for want
/// of memory and uploads wait; commit is critically low; under pressure;
/// calm.
fn guard_state(oom_wait: bool, critical: bool, pressure: bool) -> &'static str {
    if oom_wait {
        "oom-wait"
    } else if critical {
        "critical"
    } else if pressure {
        "pressured"
    } else {
        "calm"
    }
}

/// Upload MB/s between two looks `secs` apart (0 when no time passed or a
/// count went back).
fn upload_rate(now: &Totals, before: &Totals, secs: f64) -> f64 {
    if secs <= 0.0 {
        return 0.0;
    }
    now.upload_bytes.saturating_sub(before.upload_bytes) as f64 / 1e6 / secs
}

/// The residency line: counts since the start; the upload rate since the
/// look before; slots of the index queue waiting now; the card's use and
/// budget in MB (`None`: not known).
fn line(now: &Totals, before: &Totals, secs: f64, waiting: u64, guard: &str, card: Option<[u64; 2]>) -> String {
    let card = card.map_or("unknown".to_owned(), |[used, budget]| format!("{}/{} MB", used >> 20, budget >> 20));
    format!(
        "residency: let go idle {} budget {} unmap {}, made again {}, upload {:.1} MB/s ({} MB so far), index waiting {waiting}, \
         refused passes {} (ram guard {}), guard {guard}, card {card}, made again within 60 s {} ({} MB), \
         blocks given back {} MB, under pressure {} s (counts since the start)",
        now.idle,
        now.budget,
        now.unmap,
        now.made_again,
        upload_rate(now, before, secs),
        now.upload_bytes >> 20,
        now.refused,
        now.ram_refused,
        now.again_soon,
        now.again_soon_bytes >> 20,
        now.given_back_bytes >> 20,
        now.pressured_secs,
    )
}

/// Logs the residency line; called with the other 5 s lines.
pub(crate) fn log_line(card: Option<[u64; 4]>) {
    static LAST: std::sync::Mutex<Option<(std::time::Instant, Totals)>> = std::sync::Mutex::new(None);
    let now = Totals::now();
    let at = std::time::Instant::now();
    let before = LAST.lock().unwrap_or_else(|e| e.into_inner()).replace((at, now));
    let (secs, before) = before.map_or((0.0, Totals::default()), |(t, b)| ((at - t).as_secs_f64(), b));
    let guard = guard_state(upload_refused(), critical_commit(), pressured());
    let waiting = crate::draw::HEAP_TEXTURES_WAITING.load(Ordering::Relaxed);
    let used_budget = card.map(|[budget, used, ..]| [used, budget]);
    tracing::info!("{}{}", line(&now, &before, secs, waiting, guard, used_budget), crate::priority::tail(card));
    if let Some(why) = crate::waitwhy::line() {
        tracing::info!("{why}");
    }
    static PLACEMENT: std::sync::Once = std::sync::Once::new();
    PLACEMENT.call_once(|| tracing::info!("{}", crate::pool::placement_line(crate::pool::small_placement())));
    let fill = crate::pool::account();
    tracing::info!("{}", crate::pool::fill_line(&fill));
    tracing::info!("{}", crate::pool::pack_line(crate::pool::pack(), &fill, crate::pool::emptying()));
    tracing::info!("{}", crate::resource::budget_line(card));
    tracing::info!("{}", crate::shadow::reserve_line(card, now.budget));
    let [renders, twins] = crate::resource::card_textures();
    tracing::info!("{}", card_account_line(card.map(|memory| memory[1]), fill.block_bytes, renders, crate::shadow::live_span_bytes(), twins));
}

/// The card account beside the residency line. `used` is the card's own
/// memory in use as the system counts it (`None`: not known). Pool blocks
/// are the whole blocks, which already hold the textures cut from them.
/// Textures the host renders to and depth twins do not overlap. `other` is
/// in use minus those parts, so the printed megabytes add up to the use.
fn card_account_line(used: Option<u64>, pool: u64, renders: u64, buffers: u64, twins: u64) -> String {
    let mb = |n: u64| n >> 20;
    let parts = format!(
        "pool blocks {} MB, textures the host renders to or writes {} MB, buffer copies {} MB, depth twins {} MB",
        mb(pool),
        mb(renders),
        mb(buffers),
        mb(twins),
    );
    match used {
        Some(used) => {
            let other = (used >> 20) as i64 - (mb(pool) + mb(renders) + mb(buffers) + mb(twins)) as i64;
            format!("card account: {} MB in use as the system counts it; {parts}, other {other} MB", mb(used))
        }
        None => format!("card account: in use unknown; {parts}, other unknown"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_names_the_worst_state() {
        assert_eq!(guard_state(true, true, true), "oom-wait");
        assert_eq!(guard_state(false, true, true), "critical");
        assert_eq!(guard_state(false, false, true), "pressured");
        assert_eq!(guard_state(false, false, false), "calm");
    }

    #[test]
    fn upload_rate_is_over_the_interval_and_never_negative() {
        let before = Totals { upload_bytes: 10_000_000, ..Totals::default() };
        let now = Totals { upload_bytes: 60_000_000, ..Totals::default() };
        assert!((upload_rate(&now, &before, 5.0) - 10.0).abs() < 1e-9);
        assert_eq!(upload_rate(&now, &before, 0.0), 0.0);
        assert_eq!(upload_rate(&before, &now, 5.0), 0.0);
    }

    #[test]
    fn line_holds_every_count_in_the_parsed_order() {
        let now = Totals {
            idle: 7,
            budget: 3,
            unmap: 2,
            made_again: 5,
            upload_bytes: 50 << 20,
            refused: 4,
            ram_refused: 1,
            again_soon: 6,
            again_soon_bytes: 12 << 20,
            given_back_bytes: 128 << 20,
            pressured_secs: 42,
        };
        let before = Totals { upload_bytes: 0, ..now };
        let l = line(&now, &before, 5.0, 9, "calm", Some([6038 << 20, 9285 << 20]));
        assert_eq!(
            l,
            "residency: let go idle 7 budget 3 unmap 2, made again 5, upload 10.5 MB/s (50 MB so far), index waiting 9, \
             refused passes 4 (ram guard 1), guard calm, card 6038/9285 MB, made again within 60 s 6 (12 MB), \
             blocks given back 128 MB, under pressure 42 s (counts since the start)"
        );
        assert!(line(&now, &before, 0.0, 0, "calm", None).contains("upload 0.0 MB/s (50 MB so far)"));
        assert!(line(&now, &before, 0.0, 0, "calm", None).contains("card unknown"));
    }

    #[test]
    fn card_account_names_each_part_and_what_is_left() {
        assert_eq!(
            card_account_line(Some(10_000 << 20), 5_376 << 20, 1_900 << 20, 970 << 20, 140 << 20),
            "card account: 10000 MB in use as the system counts it; pool blocks 5376 MB, textures the host renders to or writes 1900 MB, buffer copies 970 MB, depth twins 140 MB, other 1614 MB"
        );
        assert_eq!(
            card_account_line(Some(100 << 20), 80 << 20, 50 << 20, 0, 0),
            "card account: 100 MB in use as the system counts it; pool blocks 80 MB, textures the host renders to or writes 50 MB, buffer copies 0 MB, depth twins 0 MB, other -30 MB"
        );
        assert_eq!(
            card_account_line(None, 64 << 20, 0, 0, 0),
            "card account: in use unknown; pool blocks 64 MB, textures the host renders to or writes 0 MB, buffer copies 0 MB, depth twins 0 MB, other unknown"
        );
    }

    #[test]
    fn bake_and_lookahead_leave_room_for_live_game_allocations() {
        for (commit, bake, lookahead, growth) in [
            (128 * MIB, false, false, false),
            (256 * MIB, true, false, false),
            (512 * MIB, true, true, false),
            (768 * MIB, true, true, false),
            (1024 * MIB, true, true, true),
        ] {
            assert_eq!(has_upload_reserve(commit, true), bake);
            assert_eq!(has_upload_reserve(commit, false), lookahead);
            assert_eq!(has_growth_reserve(commit, false), growth);
        }
    }

    #[test]
    fn either_constraint_enters_pressure_and_both_must_recover() {
        assert!(next_pressure(false, 12 * GIB, 4 * GIB));
        assert!(next_pressure(false, 3 * GIB, 16 * GIB));
        assert!(next_pressure(true, 4 * GIB, 7 * GIB));
        assert!(next_pressure(true, 6 * GIB, 5 * GIB));
        assert!(!next_pressure(true, 5 * GIB, 6 * GIB));
        assert!(!next_pressure(false, 4 * GIB, 5 * GIB));
    }

    #[test]
    fn pressure_kept_ends_at_4_gib_or_1_gib_above_entry() {
        // Session notwin: in at 3,025 MiB free, never out at 5 GiB; now out
        // at 4,049 MiB.
        let notwin = [3025 * MIB, 20 * GIB];
        assert_eq!(way_out(notwin), [4049 * MIB, 5 * GIB]);
        assert!(next_pressure_kept(true, 4049 * MIB - 1, 20 * GIB, notwin));
        assert!(!next_pressure_kept(true, 4049 * MIB, 20 * GIB, notwin));
        // Entered at 3.5 GiB (by commit): out at 4 GiB, not 4.5.
        let entry = [3584 * MIB, 20 * GIB];
        assert_eq!(way_out(entry), [4 * GIB, 5 * GIB]);
        assert!(next_pressure_kept(true, 4 * GIB - 1, 20 * GIB, entry));
        assert!(!next_pressure_kept(true, 4 * GIB, 20 * GIB, entry));
        assert!(next_pressure(true, 4 * GIB, 20 * GIB));
        // Entered lower: 1 GiB above it is enough.
        let low = [2 * GIB, 3 * GIB];
        assert_eq!(way_out(low), [3 * GIB, 4 * GIB]);
        assert!(!next_pressure_kept(true, 3 * GIB, 4 * GIB, low));
        assert!(next_pressure_kept(true, 3 * GIB, 4 * GIB - 1, low));
        assert!(next_pressure_kept(true, 3 * GIB - 1, 4 * GIB, low));
        // In as before.
        for (physical, commit) in [(3 * GIB, 16 * GIB), (12 * GIB, 4 * GIB), (4 * GIB, 5 * GIB)] {
            assert_eq!(next_pressure_kept(false, physical, commit, entry), next_pressure(false, physical, commit));
        }
    }

    #[test]
    fn pressure_keep_is_30_s_at_least_and_backs_off_to_10_minutes() {
        assert_eq!(pressure_idle_ms(0), 30_000);
        assert_eq!(pressure_idle_ms(120_000), 120_000);
        assert_eq!(pressure_idle_ms(u32::MAX), 600_000);
        // Made again within a minute: twice as long each time, up to ten minutes.
        let mut keep = 0;
        let mut seen = Vec::new();
        for _ in 0..7 {
            keep = backed_off(keep, 59_999);
            seen.push(keep);
        }
        assert_eq!(seen, [60_000, 120_000, 240_000, 480_000, 600_000, 600_000, 600_000]);
        // Made again later: as it was.
        assert_eq!(backed_off(0, 60_000), 0);
        assert_eq!(backed_off(240_000, 3_600_000), 240_000);
        assert!(made_again_soon(0) && !made_again_soon(60_000));
    }

    #[test]
    fn seconds_under_pressure_count_the_spell_now() {
        assert_eq!(pressured_secs(0, None, 10_000), 0);
        assert_eq!(pressured_secs(5_500, None, 10_000), 5);
        assert_eq!(pressured_secs(5_500, Some(7_000), 10_000), 8);
        assert_eq!(pressured_secs(0, Some(12_000), 10_000), 0);
    }

    #[test]
    fn prefetch_respects_existing_budgets_and_resumes_after_recovery() {
        assert_eq!(prefetch_limit(512 << 20, false, false, true, GIB), 64 << 20);
        assert_eq!(prefetch_limit(512 << 20, false, false, true, GIB - 1), 512 << 20);
        assert_eq!(prefetch_limit(512 << 20, false, false, false, 2 * GIB), 512 << 20);
        assert_eq!(prefetch_limit(0, true, true, false, 0), 0);
        assert_eq!(prefetch_limit(8 << 20, true, true, false, 0), 8 << 20);
        assert_eq!(prefetch_limit(512 << 20, true, false, false, 0), 64 << 20);
        assert_eq!(prefetch_limit(512 << 20, true, true, false, 0), 16 << 20);
        assert_eq!(prefetch_limit(192 << 20, false, false, false, 0), 192 << 20);
    }
}
