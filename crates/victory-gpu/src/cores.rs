//! Which processors threads run on, where a processor has two kinds of cores.
//! `cpu_sets` and `keep_on_fast_cores` are fxceless/victory-fork's
//! (`framelog.rs`, commit 0b0074e0, GPL-3.0-or-later, as this). Our own
//! threads ask from there. The title's threads, disk workers and the
//! process's power throttle are separate switches, each off unless asked:
//! `VICTORY_TITLE_CORES=1`, `VICTORY_DISK_CORES=slow`, `VICTORY_NO_THROTTLE=1`,
//! `VICTORY_CORE_LOG=1`. `alt` on any of them is twenty seconds off and twenty
//! on; a diagnostics build's `VICTORY_AB` names the same switches.

/// One logical processor of group 0: its cpu-set id, efficiency class (higher
/// is faster), the core it shares, and its processor number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Proc {
    id: u32,
    class: u8,
    core: u8,
    logical: u8,
}

/// The system's processors of group 0.
fn cpu_sets() -> Vec<Proc> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut sets = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 {
            sets.push(Proc { id: set.Id, class: set.EfficiencyClass, core: set.CoreIndex, logical: set.LogicalProcessorIndex });
        }
        at += info.Size as usize;
    }
    sets
}

/// The bytes of the last-level cache each processor set of group 0 shares,
/// by set id (none known: empty).
fn l3_of_sets() -> Vec<(u32, u32)> {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, GetSystemCpuSetInformation, RelationCache, SYSTEM_CPU_SET_INFORMATION,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // The largest cache of each logical processor of group 0, by its index.
    let mut by_processor = [0u32; 64];
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetLogicalProcessorInformationEx(RelationCache, std::ptr::null_mut(), &mut bytes) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetLogicalProcessorInformationEx(RelationCache, buffer.as_mut_ptr().cast(), &mut bytes) };
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>()) };
        if info.Size == 0 {
            break;
        }
        if info.Relationship == RelationCache {
            // SAFETY: a cache record (asked for), with at least its first group mask.
            let cache = unsafe { &info.Anonymous.Cache };
            let mask = unsafe { cache.Anonymous.GroupMask };
            if mask.Group == 0 {
                for (p, size) in by_processor.iter_mut().enumerate() {
                    if mask.Mask >> p & 1 != 0 {
                        *size = (*size).max(cache.CacheSize);
                    }
                }
            }
        }
        at += info.Size as usize;
    }
    // Processor indices to set ids.
    let mut bytes = 0u32;
    // SAFETY: as for `cpu_sets`.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut sizes = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 && let Some(&size) = by_processor.get(set.LogicalProcessorIndex as usize) {
            sizes.push((set.Id, size));
        }
        at += info.Size as usize;
    }
    sizes
}

/// Whether the cores of the largest last-level cache count as the fast
/// ones where all cores are of one class (`VICTORY_CACHE_CORES=0`: not).
/// On a processor with two core complexes of which one has the stacked cache
/// (Ryzen 7900X3D, 7950X3D, 9900X3D, 9950X3D) every core reports the same
/// efficiency class, so nothing was asked for: our threads ran on either
/// complex. Their work (compares, hashes, copies of tens of megabytes a
/// frame) is the kind the larger cache serves. Nothing changes on a processor
/// whose caches are all alike, or with two kinds of cores.
fn cache_cores() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_CACHE_CORES") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CACHE_CORES").is_none_or(|v| v != "0"))
}

/// The sets on the largest last-level cache (`l3`: bytes by set id), if
/// another one is smaller and the largest serves at least eight sets.
fn fast_by_cache(sets: &[Proc], l3: &[(u32, u32)]) -> Vec<u32> {
    let size = |id: u32| l3.iter().find(|s| s.0 == id).map_or(0, |s| s.1);
    let sizes: Vec<u32> = sets.iter().map(|s| size(s.id)).collect();
    let (Some(&most), Some(&least)) = (sizes.iter().max(), sizes.iter().min()) else {
        return Vec::new();
    };
    if least == 0 || least == most {
        return Vec::new();
    }
    let fast: Vec<u32> = sets.iter().filter(|s| size(s.id) == most).map(|s| s.id).collect();
    if fast.len() < 8 { Vec::new() } else { fast }
}

/// The fast sets: those of the fastest class if there are slower ones, else
/// those of the largest cache ([`cache_cores`]). `now`: as things are at this
/// moment (else whether there could be any, for threads to ask at all).
/// For our own threads. The title's threads (`VICTORY_TITLE_CORES`) go by
/// class alone ([`fast_ids`]): where all cores are of one class they stay
/// where they are, whatever the caches.
fn fast_sets(sets: &[Proc], now: bool) -> Vec<u32> {
    let by_class = fast_ids(sets);
    if !by_class.is_empty() || (now && !cache_cores()) {
        return by_class;
    }
    static L3: std::sync::OnceLock<Vec<(u32, u32)>> = std::sync::OnceLock::new();
    fast_by_cache(sets, L3.get_or_init(l3_of_sets))
}

fn fast_of(sets: &[Proc]) -> Vec<u32> {
    fast_sets(sets, true)
}

/// [`cpu_sets`], once.
fn procs() -> &'static [Proc] {
    static PROCS: std::sync::OnceLock<Vec<Proc>> = std::sync::OnceLock::new();
    PROCS.get_or_init(cpu_sets)
}

/// The sets of the fastest class, if there are slower ones too. Empty when
/// every processor is the same class (nothing to ask for).
fn fast_ids(sets: &[Proc]) -> Vec<u32> {
    let Some(best) = sets.iter().map(|c| c.class).max() else { return Vec::new() };
    let fast: Vec<u32> = sets.iter().filter(|c| c.class == best).map(|c| c.id).collect();
    if fast.len() == sets.len() { Vec::new() } else { fast }
}

/// The sets of every slower class. Empty when there is no slower class.
fn slow_ids(sets: &[Proc]) -> Vec<u32> {
    let Some(best) = sets.iter().map(|c| c.class).max() else { return Vec::new() };
    sets.iter().filter(|c| c.class < best).map(|c| c.id).collect()
}

/// The first thread of each fast core, in core order: where a slow core's
/// request lands.
fn fast_targets(fast: &[Proc]) -> Vec<u8> {
    let mut cores: Vec<(u8, u8)> = Vec::new();
    for p in fast {
        if let Some((_, logical)) = cores.iter_mut().find(|(c, _)| *c == p.core) {
            *logical = (*logical).min(p.logical);
        } else {
            cores.push((p.core, p.logical));
        }
    }
    cores.sort_unstable_by_key(|(c, _)| *c);
    cores.into_iter().map(|(_, logical)| logical).collect()
}

fn split(sets: &[Proc]) -> Option<(Vec<Proc>, Vec<Proc>)> {
    let best = sets.iter().map(|c| c.class).max()?;
    let fast: Vec<Proc> = sets.iter().copied().filter(|c| c.class == best).collect();
    let slow: Vec<Proc> = sets.iter().copied().filter(|c| c.class < best).collect();
    (!fast.is_empty() && !slow.is_empty()).then_some((fast, slow))
}

/// An ideal processor the title asked for, moved onto a fast core when it
/// named a slow one. A fast processor, or one this layout does not know, stays.
fn map_ideal(logical: u32, sets: &[Proc]) -> u32 {
    let Some((fast, slow)) = split(sets) else { return logical };
    let Some(logical_u) = u8::try_from(logical).ok() else { return logical };
    if fast.iter().any(|p| p.logical == logical_u) {
        return logical;
    }
    let Some(index) = slow.iter().position(|p| p.logical == logical_u) else { return logical };
    let targets = fast_targets(&fast);
    u32::from(targets[index % targets.len()])
}

/// An affinity mask the title asked for. Every known processor, or the whole
/// mask, is not a request for specific cores and stays. A specific mask keeps
/// its fast processors and moves each slow one onto a fast core.
fn map_affinity(mask: u64, sets: &[Proc]) -> u64 {
    let Some((fast, slow)) = split(sets) else { return mask };
    if mask == 0 || mask == u64::MAX {
        return mask;
    }
    let known = sets.iter().filter(|p| p.logical < 64).fold(0u64, |m, p| m | (1 << p.logical));
    if mask & known == known {
        return mask;
    }
    let mut out = mask & !known;
    for p in &fast {
        if p.logical < 64 && mask & (1 << p.logical) != 0 {
            out |= 1 << p.logical;
        }
    }
    let targets = fast_targets(&fast);
    for (index, p) in slow.iter().enumerate() {
        if p.logical < 64 && mask & (1 << p.logical) != 0 {
            let to = targets[index % targets.len()];
            if to < 64 {
                out |= 1 << to;
            }
        }
    }
    out
}

/// Cpu-set ids the title selected, written as fast-core ids. An empty
/// selection is the whole fast set (they did not name a core).
fn map_set_ids(ids: &[u32], sets: &[Proc]) -> Vec<u32> {
    let Some((fast, slow)) = split(sets) else { return ids.to_vec() };
    if ids.is_empty() {
        return fast.iter().map(|p| p.id).collect();
    }
    let targets = fast_targets(&fast);
    let mut out = Vec::new();
    for &id in ids {
        let mapped = if fast.iter().any(|p| p.id == id) {
            Some(id)
        } else if let Some(index) = slow.iter().position(|p| p.id == id) {
            let logical = targets[index % targets.len()];
            fast.iter().find(|p| p.logical == logical).map(|p| p.id)
        } else {
            None
        };
        if let Some(id) = mapped && !out.contains(&id) {
            out.push(id);
        }
    }
    out
}

/// The sets of one fast core of the last, and all the others, for
/// [`lane_core`]; `None` with fewer than six fast cores.
fn one_core(sets: &[Proc]) -> Option<(Vec<u32>, Vec<u32>)> {
    let best = sets.iter().map(|c| c.class).max()?;
    let mut cores: Vec<u8> = sets.iter().filter(|c| c.class == best).map(|c| c.core).collect();
    cores.sort_unstable();
    cores.dedup();
    let core = *cores.last().filter(|_| cores.len() >= 6)?;
    let of = |own: bool| sets.iter().filter(|c| (c.core == core) == own).map(|c| c.id).collect();
    Some((of(true), of(false)))
}

/// Whether the submitting thread has a fast core to itself, as far as this
/// process goes (`VICTORY_LANE_CORE=1`): nothing else of it runs on the
/// core's two threads. It was the one a frame waited for (94 % of it in a
/// street), and shared its core's units with whatever ran beside it.
fn lane_core() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_LANE_CORE") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LANE_CORE").is_some_and(|v| v == "1"))
}

/// Threads that asked: a handle of each, and whether it is the submitting one.
static ASKED: std::sync::Mutex<Vec<(usize, bool)>> = std::sync::Mutex::new(Vec::new());

/// What the threads of the process, and those that asked, are told now
/// ([`lane_core`], which may have changed).
pub(crate) fn settle() {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetProcessDefaultCpuSets, SetThreadSelectedCpuSets};
    if stage() == 0 {
        return;
    }
    let sets = procs();
    let fast = fast_of(sets);
    let one = one_core(sets).filter(|_| lane_core());
    let others: &[u32] = one.as_ref().map_or(&[], |o| &o.1);
    let null = std::ptr::null::<u32>();
    // SAFETY: our own process and the system's own set ids (none: all).
    unsafe {
        let at = if others.is_empty() { null } else { others.as_ptr() };
        SetProcessDefaultCpuSets(GetCurrentProcess(), at, others.len() as u32);
    }
    let helpers: Vec<u32> = match &one {
        Some((own, _)) => fast.iter().copied().filter(|id| !own.contains(id)).collect(),
        None => fast.clone(),
    };
    for &(thread, lane) in ASKED.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        let sets = match &one {
            Some((own, _)) if lane => own,
            _ => &helpers,
        };
        // SAFETY: a handle kept open, and the system's own set ids.
        unsafe { SetThreadSelectedCpuSets(thread as _, sets.as_ptr(), sets.len() as u32) };
    }
    tracing::info!(on = one.is_some(), core = ?one.as_ref().map(|o| &o.0), "the submitting thread has a core of its own (VICTORY_LANE_CORE)");
    tracing::info!(
        fast = fast.len(),
        of = sets.len(),
        by_cache = fast_ids(sets).is_empty() && !fast.is_empty(),
        threads = ASKED.lock().unwrap_or_else(|e| e.into_inner()).len(),
        "our threads' processors (VICTORY_CACHE_CORES)"
    );
}

/// `VICTORY_FAST_CORES`: 0: no thread asks; 1: the threads a frame waits for
/// one by one; 2 (the default): the threads they share their work with too;
/// 3: those that work beside the rest as well (lists translated, descriptors
/// made): more threads by then than the fast cores run at a time, while the
/// slow ones stood idle.
fn stage() -> u32 {
    static STAGE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *STAGE.get_or_init(|| std::env::var("VICTORY_FAST_CORES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2))
}

/// Asks the system to run the calling thread, which a frame waits for, on
/// the fast cores only, where the processor has two kinds. That fork's submit
/// thread was found on a slow one in 1,529 looks of 24,021. (A selection of
/// processors is a preference the scheduler keeps unless nothing of it is
/// free.) The title's threads are asked for separately (`VICTORY_TITLE_CORES`).
pub(crate) fn keep_on_fast_cores(what: &str) {
    ask(what, 1);
}

/// [`keep_on_fast_cores`] for one of the threads a submission's preparing or
/// a landing shares its work with.
pub(crate) fn keep_helper_on_fast_cores(what: &str) {
    ask(what, 2);
}

/// [`keep_on_fast_cores`] for one of many threads whose work is ahead of what
/// a frame waits for.
pub(crate) fn keep_many_on_fast_cores(what: &str) {
    ask(what, 3);
}

fn ask(what: &str, from: u32) {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThreadId, OpenThread, THREAD_SET_LIMITED_INFORMATION,
    };
    if stage() < from {
        return;
    }
    static FAST: std::sync::OnceLock<(Vec<u32>, usize)> = std::sync::OnceLock::new();
    let (fast, of) = FAST.get_or_init(|| {
        let sets = procs();
        // (whether there can be fast sets at all: `VICTORY_CACHE_CORES` may
        // change while the title runs, and `settle` follows it)
        (fast_sets(sets, false), sets.len())
    });
    // All alike (or nothing known): nothing to ask for.
    if fast.is_empty() {
        return;
    }
    // SAFETY: our own thread; the handle stays open.
    let thread = unsafe { OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, GetCurrentThreadId()) };
    if thread.is_null() {
        tracing::warn!(what, error = %std::io::Error::last_os_error(), "the thread could not be kept on the fast cores");
        return;
    }
    ASKED.lock().unwrap_or_else(|e| e.into_inner()).push((thread as usize, what == "victory-submit"));
    settle();
    tracing::info!(what, fast = fast.len(), of, "the thread is kept on the fast cores (VICTORY_FAST_CORES=0: not)");
}

/// Asks the system to run the calling thread, whose work no frame waits for
/// (a frame shot's writing out, a disk worker), on the slow cores only, where
/// the processor has two kinds and at least eight slow ones. Whether it was
/// asked. (fxceless/victory-fork's `framelog::keep_on_slow_cores`.)
pub(crate) fn keep_on_slow_cores(what: &str) -> bool {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadSelectedCpuSets};
    let sets = procs();
    let slow = slow_ids(sets);
    if slow.len() < 8 {
        return false;
    }
    // SAFETY: our own thread and a list of the system's own set ids.
    let ok = unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), slow.as_ptr(), slow.len() as u32) } != 0;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(what, slow = slow.len(), of = sets.len(), ok, "the thread is kept on the slow cores: the first one");
    });
    ok
}

/// What a switch's value says. `on_value` is `1` or `slow`; `alt` takes turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Asked {
    Off,
    On,
    Alt,
}

fn asked(value: Option<&str>, on_value: &str) -> Asked {
    match value.map(str::trim) {
        Some(v) if v == on_value => Asked::On,
        Some("alt") => Asked::Alt,
        _ => Asked::Off,
    }
}

/// How long each turn of a switch set to `alt` lasts.
const ALT_SECONDS: u64 = 20;

/// Whether an `alt` switch is on this long after it was first read: twenty
/// seconds off, then twenty on.
fn alt_turn(since: std::time::Duration) -> bool {
    since.as_secs() / ALT_SECONDS % 2 == 1
}

fn alt_on() -> bool {
    static BEGAN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    alt_turn(BEGAN.get_or_init(std::time::Instant::now).elapsed())
}

/// One switch, read once. Diagnostics builds let `VICTORY_AB` turn it on and
/// off while the title runs.
struct Switch {
    name: &'static str,
    on_value: &'static str,
    asked: std::sync::OnceLock<Result<Asked, u32>>,
}

impl Switch {
    const fn new(name: &'static str, on_value: &'static str) -> Switch {
        Switch { name, on_value, asked: std::sync::OnceLock::new() }
    }

    fn stored(&self) -> Result<Asked, u32> {
        *self.asked.get_or_init(|| {
            #[cfg(feature = "diagnostics")]
            if let Some(bit) = crate::ab::bit(self.name) {
                return Err(bit);
            }
            Ok(asked(std::env::var(self.name).ok().as_deref(), self.on_value))
        })
    }

    fn on(&self) -> bool {
        match self.stored() {
            Ok(Asked::On) => true,
            Ok(Asked::Alt) => alt_on(),
            Ok(Asked::Off) => false,
            #[cfg(feature = "diagnostics")]
            Err(bit) => crate::ab::on(bit),
            #[cfg(not(feature = "diagnostics"))]
            Err(_) => false,
        }
    }

    /// Whether it can be on at some point in the run (`1`/`slow`, `alt`, or
    /// named by `VICTORY_AB`).
    fn may(&self) -> bool {
        !matches!(self.stored(), Ok(Asked::Off))
    }
}

static TITLE: Switch = Switch::new("VICTORY_TITLE_CORES", "1");
static DISK: Switch = Switch::new("VICTORY_DISK_CORES", "slow");
static THROTTLE: Switch = Switch::new("VICTORY_NO_THROTTLE", "1");
static LOG: Switch = Switch::new("VICTORY_CORE_LOG", "1");

/// Whether title-thread placement can come on (`VICTORY_TITLE_CORES`).
pub fn title_cores_may() -> bool {
    TITLE.may()
}

/// Whether the minute's cpu-time line can come on (`VICTORY_CORE_LOG`).
pub fn core_log_may() -> bool {
    LOG.may()
}

fn fast_cached() -> &'static [u32] {
    static FAST: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    FAST.get_or_init(|| fast_ids(procs()))
}

/// The mask the title asked for, with specific cores moved onto the fast ones
/// when [`TITLE`] is on.
pub fn affinity_mask_for_title(mask: usize) -> usize {
    if !TITLE.on() { return mask }
    map_affinity(mask as u64, procs()) as usize
}

/// See [`affinity_mask_for_title`] for a group. Other groups are left alone.
pub fn group_mask_for_title(group: u16, mask: usize) -> usize {
    if group != 0 { return mask }
    affinity_mask_for_title(mask)
}

/// The ideal processor the title asked for, moved onto a fast core when
/// [`TITLE`] is on.
pub fn ideal_processor_for_title(processor: u32) -> u32 {
    if !TITLE.on() { return processor }
    map_ideal(processor, procs())
}

/// See [`ideal_processor_for_title`] for a processor number in a group.
pub fn ideal_number_for_title(group: u16, number: u8) -> u8 {
    if group != 0 { return number }
    u8::try_from(ideal_processor_for_title(u32::from(number))).unwrap_or(number)
}

/// Cpu-set ids for a title thread's own selection, when [`TITLE`] is on.
/// `None`: pass the title's pointer through (the switch is off, or the ids
/// were not ones this layout knows).
pub fn cpu_sets_for_title(ids: &[u32]) -> Option<Vec<u32>> {
    if !TITLE.on() { return None }
    let mapped = map_set_ids(ids, procs());
    if mapped.is_empty() && !ids.is_empty() { return None }
    Some(mapped)
}

/// Whose cpu time a minute's line adds up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Title,
    Ours,
    Disk,
    Audio,
}

fn slot(kind: Kind, on_fast: bool) -> usize {
    let base = match kind {
        Kind::Title => 0,
        Kind::Ours => 2,
        Kind::Disk => 4,
        Kind::Audio => 6,
    };
    base + usize::from(!on_fast)
}

/// Adds `delta` 100 ns units into the fast or slow column of `kind`.
fn add_sample(into: &mut [u64; 8], kind: Kind, on_fast: bool, delta: u64) {
    into[slot(kind, on_fast)] += delta;
}

/// Milliseconds from a 100 ns count.
fn ms(ticks: u64) -> u64 {
    ticks / 10_000
}

/// A thread whose name says what it is, before its module is looked at.
fn kind_from_name(name: &str) -> Option<Kind> {
    let name = name.to_ascii_lowercase();
    if name.starts_with("dstorage") { return Some(Kind::Disk) }
    if name.contains("xma") || name.contains("audio") { return Some(Kind::Audio) }
    if name == "title-main" { return Some(Kind::Title) }
    None
}

/// A thread whose start lies in `path`. System threads are left out, apart
/// from the host's audio engine.
fn kind_from_path(path: &str) -> Option<Kind> {
    if path.is_empty() { return None }
    let path = path.to_ascii_lowercase();
    let file = path.rsplit(['\\', '/']).next().unwrap_or(&path);
    if file == "audioses.dll" || file.starts_with("xaudio") { return Some(Kind::Audio) }
    if path.contains("\\windows\\system32\\") || path.contains("\\windows\\syswow64\\") { return None }
    Some(Kind::Title)
}

static NOTED: std::sync::Mutex<Vec<(u32, Kind)>> = std::sync::Mutex::new(Vec::new());

fn notes() -> std::sync::MutexGuard<'static, Vec<(u32, Kind)>> {
    NOTED.lock().unwrap_or_else(|e| e.into_inner())
}

fn note(id: u32, kind: Kind) {
    if id == 0 { return }
    // `try_lock`: a title thread can arrive here before the Rust runtime has
    // set it up, and waiting would park that thread.
    let Ok(mut notes) = NOTED.try_lock() else { return };
    if let Some(slot) = notes.iter_mut().find(|n| n.0 == id) {
        // Disk and audio outrank a title note made as the thread started.
        if slot.1 == Kind::Title || kind != Kind::Title {
            slot.1 = kind;
        }
    } else {
        notes.push((id, kind));
    }
}

fn noted(id: u32) -> Option<Kind> {
    notes().iter().find(|n| n.0 == id).map(|n| n.1)
}

fn forget(id: u32) {
    notes().retain(|n| n.0 != id);
}

fn select(thread: *mut core::ffi::c_void, ids: &[u32]) {
    use windows_sys::Win32::System::Threading::SetThreadSelectedCpuSets;
    if ids.is_empty() { return }
    // SAFETY: a live thread handle and the system's own set ids.
    unsafe { SetThreadSelectedCpuSets(thread, ids.as_ptr(), ids.len() as u32) };
}

fn clear_sets(thread: *mut core::ffi::c_void) {
    use windows_sys::Win32::System::Threading::SetThreadSelectedCpuSets;
    // SAFETY: a live thread handle; no ids clears the selection.
    unsafe { SetThreadSelectedCpuSets(thread, std::ptr::null(), 0) };
}

/// The calling thread is one the title started. When the switch is on it
/// prefers the fast cores. Safe to call before the Rust runtime has set the
/// thread up: it does not log and does not touch a thread local.
pub fn place_title_current() {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId};
    // SAFETY: the calling thread.
    let id = unsafe { GetCurrentThreadId() };
    // Diagnostics: the frame timeline and the samples know the title's threads.
    #[cfg(feature = "diagnostics")]
    crate::timeline::title_thread_started(id);
    if !TITLE.may() { return }
    note(id, Kind::Title);
    if TITLE.on() {
        // SAFETY: the calling thread's pseudo-handle.
        select(unsafe { GetCurrentThread() }, fast_cached());
    }
}

/// `thread` was just created for the title. See [`place_title_current`].
///
/// # Safety
/// `thread` is null or a live thread handle of this process: the handle is
/// handed to the system to read and to set the thread's cores.
pub unsafe fn place_title_handle(thread: *mut core::ffi::c_void) {
    if thread.is_null() { return }
    use windows_sys::Win32::System::Threading::GetThreadId;
    // SAFETY: the handle `CreateThread` just returned; 0 if it is not a thread.
    let id = unsafe { GetThreadId(thread) };
    #[cfg(feature = "diagnostics")]
    crate::timeline::title_thread_started(id);
    if !TITLE.may() { return }
    note(id, Kind::Title);
    if TITLE.on() { select(thread, fast_cached()) }
}

/// The calling thread reads and decompresses. `VICTORY_DISK_CORES=slow` keeps
/// it on the slow cores; `alt` moves it back and forth.
pub fn keep_disk_on_slow_cores() {
    if !DISK.may() && !LOG.may() { return }
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId};
    std::thread_local! {
        static APPLIED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
        static NOTED_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    if !NOTED_HERE.get() {
        NOTED_HERE.set(true);
        // SAFETY: the calling thread.
        note(unsafe { GetCurrentThreadId() }, Kind::Disk);
    }
    let want = DISK.on();
    let was = APPLIED.get();
    if was == Some(want) { return }
    APPLIED.set(Some(want));
    if want {
        keep_on_slow_cores("dstorage worker");
    } else if was == Some(true) {
        // SAFETY: the calling thread.
        clear_sets(unsafe { GetCurrentThread() });
    }
}

/// The calling thread decodes or mixes audio. It is counted as audio by
/// [`VICTORY_CORE_LOG`], and it prefers the fast cores with the title's
/// threads when [`VICTORY_TITLE_CORES`] is on.
pub fn note_audio_thread() {
    if !LOG.may() && !TITLE.may() { return }
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId};
    // SAFETY: the calling thread.
    note(unsafe { GetCurrentThreadId() }, Kind::Audio);
    if TITLE.on() {
        // SAFETY: the calling thread's pseudo-handle.
        select(unsafe { GetCurrentThread() }, fast_cached());
    }
}

fn reapply_title(on: bool) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenThread, THREAD_SET_LIMITED_INFORMATION};
    let ids: Vec<u32> = notes().iter().filter(|n| n.1 == Kind::Title || n.1 == Kind::Audio).map(|n| n.0).collect();
    for id in ids {
        // SAFETY: a thread id of this process; a dead one fails the open.
        let thread = unsafe { OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, id) };
        if thread.is_null() {
            forget(id);
            continue;
        }
        if on { select(thread, fast_cached()) } else { clear_sets(thread) }
        // SAFETY: opened just above.
        unsafe { CloseHandle(thread) };
    }
}

/// `0xff`: not yet looked at. The first look while off does not apply: nothing
/// was set, so the process stays as it was.
fn changed(slot: &std::sync::atomic::AtomicU8, on: bool) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let now = u8::from(on);
    let prev = slot.swap(now, Relaxed);
    prev != now && (prev != 0xff || on)
}

fn sync_title() {
    use std::sync::atomic::AtomicU8;
    static WAS: AtomicU8 = AtomicU8::new(0xff);
    if !TITLE.may() { return }
    let on = TITLE.on();
    if !changed(&WAS, on) { return }
    reapply_title(on);
    tracing::info!(on, fast = fast_cached().len(), of = procs().len(), "title threads prefer the fast cores (VICTORY_TITLE_CORES=1)");
}

fn apply_throttle(disable: bool) {
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        PROCESS_POWER_THROTTLING_STATE, ProcessPowerThrottling, SetProcessInformation,
    };
    let state = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        StateMask: if disable { 0 } else { PROCESS_POWER_THROTTLING_EXECUTION_SPEED },
    };
    // SAFETY: our process; only the execution-speed bit is named, so the
    // timer-resolution opt-out is left as it was. `state` outlives the call.
    unsafe {
        SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, std::ptr::from_ref(&state).cast(), size_of_val(&state) as u32);
    }
}

fn sync_throttle() {
    use std::sync::atomic::AtomicU8;
    static WAS: AtomicU8 = AtomicU8::new(0xff);
    if !THROTTLE.may() { return }
    let off = THROTTLE.on();
    if !changed(&WAS, off) { return }
    apply_throttle(off);
    tracing::info!(off, "execution-speed throttling of the process (VICTORY_NO_THROTTLE=1: off)");
}

/// Re-reads the switches that can change while the title runs (`VICTORY_AB`).
#[cfg(feature = "diagnostics")]
pub(crate) fn refresh() {
    sync_title();
    sync_throttle();
}

/// Processor number in a thread's environment block. The kernel header names
/// `TeCurrentIdealProcessor` at 0x1744 (`ksamd64.inc`); its `Number` byte is
/// two bytes on, and it follows the processor the thread is running on.
const TEB_PROCESSOR: usize = 0x1746;

/// `THREAD_BASIC_INFORMATION` (ntddk), only the fields up through the TEB.
#[repr(C)]
struct BasicTeb {
    exit_status: i32,
    teb: *const u8,
    _rest: [usize; 4],
}

#[link(name = "ntdll")]
unsafe extern "system" {
    fn NtQueryInformationThread(thread: *mut core::ffi::c_void, class: i32, info: *mut core::ffi::c_void, len: u32, ret: *mut u32) -> i32;
}

fn current_processor_byte() -> u8 {
    let teb: *const u8;
    // SAFETY: gs:0x30 is this thread's environment block; the byte is the
    // processor number (see [`TEB_PROCESSOR`]).
    unsafe {
        std::arch::asm!("mov {}, gs:[0x30]", out(reg) teb, options(nostack, preserves_flags));
        *teb.add(TEB_PROCESSOR)
    }
}

/// The processor `thread` is on. `current`: this thread, which must not be
/// suspended. Other threads are suspended for the read so the block cannot
/// be freed under it.
fn processor_of(thread: *mut core::ffi::c_void, current: bool) -> Option<u8> {
    use windows_sys::Win32::System::Threading::{ResumeThread, SuspendThread};
    if current { return Some(current_processor_byte()) }
    // SAFETY: a thread of this process that is not this one. Resumed before
    // any further call that could panic or take a lock.
    if unsafe { SuspendThread(thread) } == u32::MAX { return None }
    let mut info = BasicTeb { exit_status: 0, teb: std::ptr::null(), _rest: [0; 4] };
    // SAFETY: writes `info`. Class 0 is ThreadBasicInformation.
    let status = unsafe { NtQueryInformationThread(thread, 0, (&raw mut info).cast(), size_of::<BasicTeb>() as u32, std::ptr::null_mut()) };
    let cpu = (status == 0 && !info.teb.is_null()).then(|| {
        // SAFETY: the thread is suspended, so this block stays. One byte.
        unsafe { *info.teb.add(TEB_PROCESSOR) }
    });
    // SAFETY: suspended just above.
    unsafe { ResumeThread(thread) };
    cpu
}

fn on_fast(logical: u8, sets: &[Proc]) -> Option<bool> {
    let (fast, slow) = split(sets)?;
    if fast.iter().any(|p| p.logical == logical) {
        Some(true)
    } else if slow.iter().any(|p| p.logical == logical) {
        Some(false)
    } else {
        None
    }
}

/// 100 ns of kernel and user time, if the call answers.
fn thread_cpu(thread: *mut core::ffi::c_void) -> Option<u64> {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::GetThreadTimes;
    let mut t = [FILETIME::default(); 4];
    let [created, exited, kernel, user] = &mut t;
    // SAFETY: a live thread handle and four structures the call fills.
    if unsafe { GetThreadTimes(thread, created, exited, kernel, user) } == 0 { return None }
    let ticks = |f: &FILETIME| u64::from(f.dwHighDateTime) << 32 | u64::from(f.dwLowDateTime);
    Some(ticks(&t[2]) + ticks(&t[3]))
}

struct Seen {
    id: u32,
    kind: Kind,
    time: u64,
    on_fast: Option<bool>,
}

fn thread_name(thread: *mut core::ffi::c_void) -> String {
    use windows_sys::Win32::Foundation::LocalFree;
    let mut name = std::ptr::null_mut();
    // SAFETY: the system allocates the description; we free it.
    unsafe {
        if windows_sys::Win32::System::Threading::GetThreadDescription(thread, &mut name) < 0 || name.is_null() {
            return String::new();
        }
        let len = (0..).take(64).take_while(|&i| *name.add(i) != 0).count();
        let text = String::from_utf16_lossy(std::slice::from_raw_parts(name, len));
        LocalFree(name.cast());
        text
    }
}

fn start_path(thread: *mut core::ffi::c_void) -> (bool, String) {
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW,
        GetModuleHandleExW, GetModuleHandleW,
    };
    let mut start = 0usize;
    // SAFETY: writes the Win32 start address. Class 9 is ThreadQuerySetWin32StartAddress.
    let status = unsafe { NtQueryInformationThread(thread, 9, (&raw mut start).cast(), size_of::<usize>() as u32, std::ptr::null_mut()) };
    if status != 0 || start == 0 { return (false, String::new()) }
    let mut module = std::ptr::null_mut();
    // SAFETY: the module that contains `start`, without a reference.
    let found = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            start as *const u16,
            &mut module,
        )
    };
    if found == 0 || module.is_null() { return (false, String::new()) }
    // SAFETY: our own image.
    let ours = module == unsafe { GetModuleHandleW(std::ptr::null()) };
    let mut path = [0u16; 260];
    // SAFETY: `module` is mapped; the buffer is 260 wchars.
    let n = unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) } as usize;
    (ours, String::from_utf16_lossy(&path[..n.min(path.len())]))
}

fn classify(id: u32, name: &str, in_our_exe: bool, path: &str) -> Option<Kind> {
    if let Some(kind) = noted(id) { return Some(kind) }
    if let Some(kind) = kind_from_name(name) { return Some(kind) }
    if in_our_exe { return Some(Kind::Ours) }
    kind_from_path(path)
}

/// One look at every thread of this process.
fn sample() -> Vec<Seen> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
    };
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcessId, GetCurrentThreadId, OpenThread, THREAD_QUERY_INFORMATION, THREAD_QUERY_LIMITED_INFORMATION,
        THREAD_SUSPEND_RESUME,
    };
    let mut out = Vec::new();
    // SAFETY: a snapshot of this process's threads, closed below.
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snapshot == INVALID_HANDLE_VALUE || snapshot.is_null() { return out }
        let (pid, me) = (GetCurrentProcessId(), GetCurrentThreadId());
        let mut entry = THREADENTRY32 { dwSize: size_of::<THREADENTRY32>() as u32, ..Default::default() };
        let mut more = Thread32First(snapshot, &mut entry) != 0;
        while more {
            if entry.th32OwnerProcessID == pid {
                let access = THREAD_QUERY_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION | THREAD_SUSPEND_RESUME;
                let thread = OpenThread(access, 0, entry.th32ThreadID);
                if !thread.is_null() {
                    let current = entry.th32ThreadID == me;
                    let time = thread_cpu(thread);
                    let logical = processor_of(thread, current);
                    let name = thread_name(thread);
                    let (ours, path) = start_path(thread);
                    if let (Some(time), Some(kind)) = (time, classify(entry.th32ThreadID, &name, ours, &path)) {
                        out.push(Seen { id: entry.th32ThreadID, kind, time, on_fast: logical.and_then(|n| on_fast(n, procs())) });
                    }
                    CloseHandle(thread);
                }
            }
            more = Thread32Next(snapshot, &mut entry) != 0;
        }
        CloseHandle(snapshot);
    }
    out
}

fn log_minute(prev: &std::collections::HashMap<u32, u64>, now: &[Seen]) {
    let mut acc = [0u64; 8];
    for seen in now {
        let Some(on_fast) = seen.on_fast else { continue };
        let delta = match prev.get(&seen.id) {
            Some(before) => seen.time.saturating_sub(*before),
            None => 0,
        };
        add_sample(&mut acc, seen.kind, on_fast, delta);
    }
    tracing::info!(
        title_p_ms = ms(acc[0]), title_e_ms = ms(acc[1]), ours_p_ms = ms(acc[2]), ours_e_ms = ms(acc[3]),
        disk_p_ms = ms(acc[4]), disk_e_ms = ms(acc[5]), audio_p_ms = ms(acc[6]), audio_e_ms = ms(acc[7]),
        "cpu time on the fast cores and the slow cores over the last minute (VICTORY_CORE_LOG=1)"
    );
}

fn watch() {
    let mut prev: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    if LOG.may() {
        prev = sample().into_iter().map(|s| (s.id, s.time)).collect();
    }
    let mut last = std::time::Instant::now();
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
        sync_title();
        sync_throttle();
        if LOG.may() && last.elapsed() >= std::time::Duration::from_secs(60) {
            let now = sample();
            if LOG.on() { log_minute(&prev, &now) }
            prev = now.into_iter().map(|s| (s.id, s.time)).collect();
            last = std::time::Instant::now();
        }
    }
}

/// Reads the switches and, when any of them can come on, applies what is on
/// now and keeps a thread that follows `alt` and the minute's log. Called
/// before the title starts.
pub fn prepare_core_placement() {
    if !TITLE.may() && !DISK.may() && !THROTTLE.may() && !LOG.may() { return }
    let _ = (TITLE.on(), DISK.on(), THROTTLE.on(), LOG.on());
    let _ = procs();
    if fast_cached().is_empty() && (TITLE.may() || DISK.may()) {
        tracing::info!("one kind of core: placement leaves threads where they are");
    }
    sync_throttle();
    sync_title();
    if std::thread::Builder::new().name("victory-cores".into()).spawn(watch).is_err() {
        tracing::warn!("no thread to follow core placement");
    }
}

/// A task name passed to `AvSetMmThreadCharacteristics`: audio, for the log.
fn audio_task(name: &str) -> bool {
    let name = name.trim();
    name.eq_ignore_ascii_case("Audio") || name.eq_ignore_ascii_case("Pro Audio")
}

fn wide_task(text: *const u16) -> Option<String> {
    if text.is_null() { return None }
    let mut buf = [0u16; 64];
    for (i, slot) in buf.iter_mut().enumerate() {
        // SAFETY: at most 64 wchars, stopped at the first NUL. No NUL in that many is not a task name.
        let c = unsafe { *text.add(i) };
        if c == 0 { return Some(String::from_utf16_lossy(&buf[..i])) }
        *slot = c;
    }
    None
}

fn narrow_task(text: *const u8) -> Option<String> {
    if text.is_null() { return None }
    let mut buf = [0u8; 64];
    for (i, slot) in buf.iter_mut().enumerate() {
        // SAFETY: as [`wide_task`], for bytes.
        let c = unsafe { *text.add(i) };
        if c == 0 { return Some(String::from_utf8_lossy(&buf[..i]).into_owned()) }
        *slot = c;
    }
    None
}

type NtSet = unsafe extern "system" fn(*mut core::ffi::c_void, u32, *mut core::ffi::c_void, u32) -> i32;
type AvA = unsafe extern "system" fn(*const u8, *mut u32) -> *mut core::ffi::c_void;
type AvW = unsafe extern "system" fn(*const u16, *mut u32) -> *mut core::ffi::c_void;

static REAL_NT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static REAL_AV_A: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static REAL_AV_W: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn call_nt(thread: *mut core::ffi::c_void, class: u32, info: *mut core::ffi::c_void, len: u32) -> i32 {
    let real = REAL_NT.load(std::sync::atomic::Ordering::Relaxed);
    if real == 0 { return 0xC000_0001_u32 as i32 }
    // SAFETY: the host function stored by [`thread_import`] before the title ran.
    let real: NtSet = unsafe { std::mem::transmute(real) };
    unsafe { real(thread, class, info, len) }
}

/// `NtSetInformationThread` classes that carry a core the title chose.
const NT_AFFINITY: u32 = 4;
const NT_IDEAL: u32 = 13;
const NT_GROUP: u32 = 30;
const NT_IDEAL_EX: u32 = 33;

#[repr(C)]
struct GroupAffinity {
    mask: usize,
    group: u16,
    reserved: [u16; 3],
}

#[repr(C)]
struct ProcessorNumber {
    group: u16,
    number: u8,
    reserved: u8,
}

unsafe extern "system" fn nt_set_information(thread: *mut core::ffi::c_void, class: u32, info: *mut core::ffi::c_void, len: u32) -> i32 {
    if info.is_null() || !TITLE.on() { return call_nt(thread, class, info, len) }
    match class {
        NT_AFFINITY if len == size_of::<usize>() as u32 => {
            // SAFETY: the title passes a mask of `len` bytes, as the call requires.
            let mask = unsafe { info.cast::<usize>().read() };
            let mask = affinity_mask_for_title(mask);
            call_nt(thread, class, (&raw const mask).cast_mut().cast(), len)
        }
        NT_IDEAL if len == 4 => {
            // SAFETY: as above, a processor number.
            let processor = unsafe { info.cast::<u32>().read() };
            let processor = ideal_processor_for_title(processor);
            call_nt(thread, class, (&raw const processor).cast_mut().cast(), 4)
        }
        NT_GROUP if len == size_of::<GroupAffinity>() as u32 => {
            // SAFETY: as above, a group affinity.
            let mut group = unsafe { info.cast::<GroupAffinity>().read() };
            group.mask = group_mask_for_title(group.group, group.mask);
            call_nt(thread, class, (&raw const group).cast_mut().cast(), len)
        }
        NT_IDEAL_EX if len == size_of::<ProcessorNumber>() as u32 => {
            // SAFETY: as above, a processor number.
            let mut number = unsafe { info.cast::<ProcessorNumber>().read() };
            number.number = ideal_number_for_title(number.group, number.number);
            call_nt(thread, class, (&raw const number).cast_mut().cast(), len)
        }
        _ => call_nt(thread, class, info, len),
    }
}

unsafe extern "system" fn av_set_a(task: *const u8, index: *mut u32) -> *mut core::ffi::c_void {
    if narrow_task(task).is_some_and(|name| audio_task(&name)) { note_audio_thread() }
    let real = REAL_AV_A.load(std::sync::atomic::Ordering::Relaxed);
    if real == 0 { return std::ptr::null_mut() }
    // SAFETY: the host function stored by [`thread_import`].
    let real: AvA = unsafe { std::mem::transmute(real) };
    unsafe { real(task, index) }
}

unsafe extern "system" fn av_set_w(task: *const u16, index: *mut u32) -> *mut core::ffi::c_void {
    if wide_task(task).is_some_and(|name| audio_task(&name)) { note_audio_thread() }
    let real = REAL_AV_W.load(std::sync::atomic::Ordering::Relaxed);
    if real == 0 { return std::ptr::null_mut() }
    // SAFETY: the host function stored by [`thread_import`].
    let real: AvW = unsafe { std::mem::transmute(real) };
    unsafe { real(task, index) }
}

/// What the title's import of `name`, which the host has at `address`, is
/// bound to. Thread calls the kernel module exports itself; this is the two
/// that live in other DLLs (`ntdll`, `avrt`).
pub fn thread_import(name: &str, address: usize) -> usize {
    use std::sync::atomic::Ordering::Relaxed;
    match name {
        "NtSetInformationThread" if title_cores_may() => {
            REAL_NT.store(address, Relaxed);
            nt_set_information as *const () as usize
        }
        "AvSetMmThreadCharacteristicsA" if core_log_may() => {
            REAL_AV_A.store(address, Relaxed);
            av_set_a as *const () as usize
        }
        "AvSetMmThreadCharacteristicsW" if core_log_may() => {
            REAL_AV_W.store(address, Relaxed);
            av_set_w as *const () as usize
        }
        _ => address,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: u32, class: u8, core: u8) -> Proc {
        Proc { id, class, core, logical: id as u8 }
    }

    /// 8 fast cores, two threads each, then 8 slow cores. Set ids are 0x100
    /// plus the processor number.
    fn hybrid_8p8e() -> Vec<Proc> {
        (0..16)
            .map(|n| Proc { id: 0x100 + n, class: 1, core: n as u8 / 2 * 2, logical: n as u8 })
            .chain((16..24).map(|n| Proc { id: 0x100 + n, class: 0, core: n as u8, logical: n as u8 }))
            .collect()
    }

    #[test]
    fn the_fast_cores_are_those_of_the_best_class_if_there_is_a_worse_one() {
        assert_eq!(fast_ids(&[row(256, 1, 0), row(257, 1, 0), row(258, 0, 2)]), [256, 257]);
        assert!(fast_ids(&[row(256, 0, 0), row(257, 0, 1)]).is_empty());
        assert!(one_core(&[row(256, 1, 0), row(257, 1, 0), row(258, 0, 2)]).is_none());
        let eight: Vec<Proc> = (0..16).map(|n| row(n, 1, n as u8 / 2 * 2)).chain((16..24).map(|n| row(n, 0, n as u8))).collect();
        let (own, others) = one_core(&eight).unwrap();
        assert_eq!((own, others.len()), (vec![14, 15], 22));
        assert!(fast_ids(&[]).is_empty());
        assert!(!cpu_sets().is_empty());
        std::thread::spawn(|| keep_on_fast_cores("test")).join().unwrap();
    }

    const MB: u32 = 1 << 20;

    /// `n` sets of one class, two to a core, from id 256.
    fn alike(n: u32) -> Vec<Proc> {
        (0..n).map(|i| row(256 + i, 0, (i / 2) as u8)).collect()
    }

    #[test]
    fn the_largest_cache_is_fast_only_where_the_caches_differ() {
        // Two complexes, one with the stacked cache (9950X3D: 16 sets each).
        let sets = alike(32);
        let l3: Vec<(u32, u32)> = (0..32).map(|i| (256 + i, if i < 16 { 96 * MB } else { 32 * MB })).collect();
        assert_eq!(fast_by_cache(&sets, &l3), (256..272).collect::<Vec<_>>());
        // The stacked cache on the second complex: that one.
        let flipped: Vec<(u32, u32)> = (0..32).map(|i| (256 + i, if i < 16 { 32 * MB } else { 96 * MB })).collect();
        assert_eq!(fast_by_cache(&sets, &flipped), (272..288).collect::<Vec<_>>());
        // Two complexes alike (9950X): nothing.
        let even: Vec<(u32, u32)> = (0..32).map(|i| (256 + i, 32 * MB)).collect();
        assert!(fast_by_cache(&sets, &even).is_empty());
        // One complex (7800X3D, 9800X3D): nothing.
        let one: Vec<(u32, u32)> = (0..16).map(|i| (256 + i, 96 * MB)).collect();
        assert!(fast_by_cache(&alike(16), &one).is_empty());
        // Caches not known: nothing.
        assert!(fast_by_cache(&sets, &[]).is_empty());
        assert!(fast_by_cache(&sets, &l3[..16]).is_empty());
        // A larger cache of under eight sets: nothing.
        let few: Vec<(u32, u32)> = (0..32).map(|i| (256 + i, if i < 4 { 64 * MB } else { 32 * MB })).collect();
        assert!(fast_by_cache(&sets, &few).is_empty());
    }

    #[test]
    fn two_kinds_of_cores_still_go_by_their_class() {
        // Eight fast cores (two sets each) and eight slow ones, the caches
        // told to differ too: the class decides, whatever the caches say.
        let sets: Vec<Proc> =
            (0..16).map(|n| row(n, 1, n as u8 / 2 * 2)).chain((16..24).map(|n| row(n, 0, n as u8))).collect();
        assert_eq!(fast_sets(&sets, true), (0..16).collect::<Vec<_>>());
        assert_eq!(fast_sets(&sets, false), (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn an_eight_p_and_eight_e_layout_selects_the_fast_sets_and_maps_a_specific_core() {
        let layout = hybrid_8p8e();
        assert_eq!(fast_ids(&layout), (0x100..0x110).collect::<Vec<_>>());
        assert_eq!(slow_ids(&layout), (0x110..0x118).collect::<Vec<_>>());
        // Already fast: stay. The eight slow processors land on the first
        // thread of fast cores 0, 2, 4, ... 14.
        assert_eq!(map_ideal(0, &layout), 0);
        assert_eq!(map_ideal(15, &layout), 15);
        assert_eq!(map_ideal(16, &layout), 0);
        assert_eq!(map_ideal(17, &layout), 2);
        assert_eq!(map_ideal(23, &layout), 14);
        assert_eq!(map_ideal(24, &layout), 24);
        assert_eq!(map_affinity(1 << 16, &layout), 1);
        assert_eq!(map_affinity((1 << 16) | (1 << 17), &layout), 1 | (1 << 2));
        assert_eq!(map_affinity(0xFF << 16, &layout), 0x5555);
        // Every processor, or the whole mask: not a request for specific cores.
        assert_eq!(map_affinity((1 << 24) - 1, &layout), (1 << 24) - 1);
        assert_eq!(map_affinity(u64::MAX, &layout), u64::MAX);
        assert_eq!(map_affinity(0, &layout), 0);
        assert_eq!(map_set_ids(&[], &layout), (0x100..0x110).collect::<Vec<_>>());
        assert_eq!(map_set_ids(&[0x110, 0x111], &layout), vec![0x100, 0x102]);
        assert_eq!(map_set_ids(&[0x105], &layout), vec![0x105]);
        // One class: nothing to move.
        let same = [row(1, 0, 0), row(2, 0, 1)];
        assert!(fast_ids(&same).is_empty());
        assert_eq!(map_ideal(1, &same), 1);
        assert_eq!(map_affinity(0b10, &same), 0b10);
        assert_eq!(on_fast(0, &layout), Some(true));
        assert_eq!(on_fast(16, &layout), Some(false));
        assert_eq!(on_fast(30, &layout), None);
    }

    #[test]
    fn switches_are_off_until_one_slow_or_alt_and_a_minute_splits_by_kind() {
        assert_eq!(asked(None, "1"), Asked::Off);
        assert_eq!(asked(Some("0"), "1"), Asked::Off);
        assert_eq!(asked(Some("1"), "1"), Asked::On);
        assert_eq!(asked(Some("slow"), "1"), Asked::Off);
        assert_eq!(asked(Some("slow"), "slow"), Asked::On);
        assert_eq!(asked(Some(" alt "), "slow"), Asked::Alt);
        let at = |s: u64| alt_turn(std::time::Duration::from_secs(s));
        assert!(!at(0) && !at(19) && at(20) && at(39) && !at(40) && at(60));
        assert_eq!(kind_from_name("dstorage 3"), Some(Kind::Disk));
        assert_eq!(kind_from_name("victory XMA decoder"), Some(Kind::Audio));
        assert_eq!(kind_from_name("title-main"), Some(Kind::Title));
        assert_eq!(kind_from_name("victory-submit"), None);
        assert_eq!(kind_from_path(r"C:\Windows\System32\ntdll.dll"), None);
        assert_eq!(kind_from_path(r"C:\Windows\System32\AudioSes.dll"), Some(Kind::Audio));
        assert_eq!(kind_from_path(r"D:\title\launch.dll"), Some(Kind::Title));
        assert_eq!(kind_from_path(""), None);
        assert!(audio_task("Pro Audio") && audio_task("audio") && !audio_task("Playback"));
        let mut acc = [0u64; 8];
        add_sample(&mut acc, Kind::Title, true, 10_000_000);
        add_sample(&mut acc, Kind::Title, false, 2_000_000);
        add_sample(&mut acc, Kind::Disk, false, 5_000_000);
        add_sample(&mut acc, Kind::Audio, true, 10_000);
        assert_eq!([ms(acc[0]), ms(acc[1]), ms(acc[2]), ms(acc[3]), ms(acc[4]), ms(acc[5]), ms(acc[6]), ms(acc[7])], [1000, 200, 0, 0, 0, 500, 1, 0]);
    }

    #[test]
    fn the_environment_block_names_the_processor_the_thread_is_on() {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
        struct Restore(usize);
        impl Drop for Restore {
            fn drop(&mut self) {
                // SAFETY: this thread, putting its previous mask back.
                unsafe { SetThreadAffinityMask(GetCurrentThread(), self.0) };
            }
        }
        // SAFETY: this thread. The returned mask is the one it had.
        let old = unsafe { SetThreadAffinityMask(GetCurrentThread(), 1 << 3) };
        assert_ne!(old, 0);
        let _restore = Restore(old);
        let start = std::time::Instant::now();
        let mut cpu = 0u32;
        while start.elapsed() < std::time::Duration::from_millis(500) {
            // SAFETY: the calling thread.
            cpu = unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessorNumber() };
            if cpu == 3 { break }
            std::hint::spin_loop();
        }
        assert_eq!(cpu, 3, "the thread did not move onto processor 3");
        assert_eq!(current_processor_byte(), 3);
    }
}
