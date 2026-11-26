//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and
//! the title's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The default budget is half a second per title frame. A pass the title
//! runs only once and whose pipeline takes longer is lost. Passes over the
//! screen (no vertex input, no depth) have workers and a longer budget of
//! their own ([`screen_budget`]): the title fills its lighting tables with
//! such draws once, at the world's first frame, and behind a world's worth of
//! pipelines they were dropped; that run had no lamps and no diffuse light to
//! its end (run textures-d2, the first after a new translation; Belobug
//! 18e57db8 found the same).

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline, TargetFormats};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    /// Taken by a worker from [`Queue::slow`].
    was_slow: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    used: AtomicBool,
    /// The formats its state declares for what a draw draws into (zeros
    /// without a graphics state).
    pub(crate) formats: TargetFormats,
    /// The same shaders and state for other formats, bound at a draw
    /// ([`for_formats`]): a chain from the title's pipeline, one link per
    /// set of formats met, never shortened.
    variant: OnceLock<Arc<Pending>>,
    /// A pass over the screen ([`GraphicsState::screen_pass`]) or a confirmed
    /// opening-scene / bake producer that must never sit behind cold RT shaders.
    screen: bool,
    /// Confirmed first-scene / bake producer prepared early even on a cold cache.
    scene_input: bool,
    /// The ray-tracing shader this pipeline is left out on purpose for (e.g.
    /// RT resolve without BVH); such a pipeline is never made.
    left_out: Option<&'static str>,
    /// A shader name taken from a container the first time a drop is
    /// described with `VICTORY_DROP_REASONS`. Empty until then, and when
    /// the container has no name.
    #[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
    shader_name: OnceLock<String>,
}

fn named(shaders: &[Arc<Shader>], name: &[u8]) -> bool {
    shaders
        .iter()
        .any(|s| s.stage == "compute" && s.container.windows(name.len()).any(|w| w == name))
}

fn left_out(shaders: &[Arc<Shader>]) -> Option<&'static str> {
    static SKIP_RT_RESOLVE: OnceLock<bool> = OnceLock::new();
    let skip_rt = *SKIP_RT_RESOLVE.get_or_init(|| {
        let builder = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1");
        if builder {
            return false;
        }
        // Without BVH, ~1,000 CS_RaytracingResolve shaders shade no hits while
        // clogging the cold compiler queue; skip unless VICTORY_RT_RESOLVE=1.
        std::env::var_os("VICTORY_RT_RESOLVE").is_none_or(|v| v == "0")
    });
    if skip_rt && named(shaders, b"CS_RaytracingResolve") {
        // Shades ray-tracing hits without BVH.
        return Some("CS_RaytracingResolve");
    }
    static SKIP_ACTIVATE_RAYS: OnceLock<bool> = OnceLock::new();
    let skip_activate = *SKIP_ACTIVATE_RAYS.get_or_init(|| {
        let builder = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1");
        if builder {
            return std::env::var_os("VICTORY_ACTIVATE_RAYS").is_some_and(|v| v == "0");
        }
        std::env::var_os("VICTORY_ACTIVATE_RAYS").is_none_or(|v| v == "0")
    });
    if skip_activate && named(shaders, b"CS_ScreenProbes_ActivateRays") {
        // Activates rays without BVH.
        return Some("CS_ScreenProbes_ActivateRays");
    }
    static SKIP_RT_TRACE: OnceLock<bool> = OnceLock::new();
    let skip_trace = *SKIP_RT_TRACE.get_or_init(|| {
        let builder = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1");
        let trace = std::env::var_os("VICTORY_RT_TRACE").is_some_and(|v| v == "1");
        victory_shader::bvh::rt_miss() && !builder && !trace
    });
    if skip_trace {
        return shaders.iter().filter(|s| s.stage == "compute").find_map(|s| traces_structures(&s.container));
    }
    None
}

/// The tracing against the ray structures, and the hair's geometry made
/// for them, by entry-point name (after edsuuu/VI-BR-PORT 24f8d2ed): the
/// `bvh_skinning_smh` program `CSGenerateHairVB`, the screen probes'
/// `CS_ScreenProbes_TraceRays*_BVH*` and the reflections'
/// `CS_RaytraceVBuffer*`. 8 % of the card's time in the opening, walking
/// structures that are not built. Their `Fallback` and `Voxels` forms march
/// through the title's voxels and light the picture: they are not named.
/// Left out only with every ray a miss (`VICTORY_RT_MISS=1`) and the
/// structure builder off; `VICTORY_RT_TRACE=1` runs them.
fn traces_structures(container: &[u8]) -> Option<&'static str> {
    let has = |name: &[u8]| container.windows(name.len()).any(|w| w == name);
    if has(b"CSGenerateHairVB") {
        Some("CSGenerateHairVB")
    } else if has(b"CS_ScreenProbes_TraceRays") && has(b"_BVH") {
        Some("CS_ScreenProbes_TraceRays_BVH")
    } else if has(b"CS_RaytraceVBuffer") && !has(b"Voxels") {
        Some("CS_RaytraceVBuffer")
    } else {
        None
    }
}

/// Confirmed first-scene producers. Prepare these at creation even on a cold
/// launch so cutscene lighting, tiling, and bakes never miss their first frame.
fn scene_input(shaders: &[Arc<Shader>]) -> bool {
    shaders.iter().any(|s| {
        matches!(
            s.hash.as_str(),
            "970c4fbb3b017c0e"
                | "068c90badc86f1fd"
                | "e4a7e088c7780ce8"
                | "ebe4e53f031f5249"
                | "2cc4c197ee29e291"
                | "369cbe906ea72764"
        ) || crate::draw::BAKE_PIXEL_SHADERS.contains(&s.hash.as_str())
    })
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        let scene_input = scene_input(&shaders);
        let screen = (graphics.as_ref().is_some_and(|g| g.screen_pass())
            && shaders.iter().all(|s| s.stage != "mesh"))
            || scene_input;
        let left_out = left_out(&shaders);
        Pending {
            screen,
            scene_input,
            left_out,
            shader_name: OnceLock::new(),
            shaders,
            formats: graphics.as_ref().map_or([0; 9], |g| g.formats()),
            variant: OnceLock::new(),
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            was_slow: AtomicBool::new(false),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
        }
    }

    pub(super) fn key(&self) -> &str {
        &self.key
    }

    fn make(&self) {
        self.make_with(|| {
            let _building = crate::timeline::span("pipeline: build a host pipeline");
            let began = Instant::now();
            let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
            super::library::note_time(&self.key, began.elapsed().as_millis() as u64);
            made
        });
    }

    fn make_with(&self, build: impl FnOnce() -> Option<HostPipeline>) {
        self.made.get_or_init(|| {
            // A failed translation must complete this queue entry and wake its waiters.
            match catch_unwind(AssertUnwindSafe(build)) {
                Ok(pipeline) => pipeline,
                Err(_) => {
                    tracing::warn!(shaders = ?self.shaders, "host pipeline build panicked");
                    None
                }
            }
        });
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics). With
/// `VICTORY_DROP_REASONS=1`: left out, queued (in the compile queue) or
/// never started (no request made), each with the pipeline hash and a
/// shader name. Otherwise left out on purpose ([`crate::failed::LEFT_OUT`]
/// and the shader's name; never made, so not "being made") or "being made".
/// Shaders that do not translate, else "refused". `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    if p.left_out.is_some() {
        return Some(unmade_why(p, drop_reasons()));
    }
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some(unmade_why(p, drop_reasons())),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// The hashes of the shaders `p` is made of.
pub(crate) fn hashes(p: &Pending) -> String {
    let hashes: Vec<&str> = p.shaders.iter().map(|s| s.hash.as_str()).collect();
    hashes.join("+")
}

/// `VICTORY_DROP_REASONS=1`. Off when unset or set to anything else.
#[cfg(any(test, feature = "diagnostics"))]
fn drop_reasons_value(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| v == "1")
}

/// Whether drops of pipelines that are not made yet name queued, never
/// started and left out. Off unless `VICTORY_DROP_REASONS=1`.
#[cfg(feature = "diagnostics")]
pub(super) fn drop_reasons() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = drop_reasons_value(std::env::var_os("VICTORY_DROP_REASONS").as_deref());
        if on {
            tracing::info!(
                "drops of pipelines that are not made yet say queued, never started or left out (VICTORY_DROP_REASONS=1)"
            );
        }
        on
    })
}

#[cfg(any(test, feature = "diagnostics"))]
fn ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

#[cfg(any(test, feature = "diagnostics"))]
fn ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// `CS_` / `VS_` / `PS_` and the other stages, with at least three
/// characters of name after the underscore.
#[cfg(any(test, feature = "diagnostics"))]
fn stage_entry(s: &[u8]) -> bool {
    const PREFIXES: [&[u8]; 8] = [b"CS_", b"VS_", b"PS_", b"GS_", b"HS_", b"DS_", b"MS_", b"AS_"];
    PREFIXES.iter().any(|p| s.starts_with(p) && s.len() >= p.len() + 3)
}

/// A shader name written in a container. `module:CS_Entry` when that form
/// is there (the longest such), otherwise the longest stage name.
#[cfg(any(test, feature = "diagnostics"))]
fn embedded_name(bytes: &[u8]) -> Option<String> {
    let mut colon: Option<&[u8]> = None;
    let mut bare: Option<&[u8]> = None;
    let mut i = 0;
    while i < bytes.len() {
        if (i > 0 && ident(bytes[i - 1])) || !ident_start(bytes[i]) {
            i += 1;
            continue;
        }
        let start = i;
        i += 1;
        while i < bytes.len() && ident(bytes[i]) {
            i += 1;
        }
        let head = i;
        if head < bytes.len() && bytes[head] == b':' && head + 1 < bytes.len() && ident_start(bytes[head + 1]) {
            let entry = head + 1;
            let mut end = entry + 1;
            while end < bytes.len() && ident(bytes[end]) {
                end += 1;
            }
            if stage_entry(&bytes[entry..end]) {
                let whole = &bytes[start..end];
                if colon.is_none_or(|had| whole.len() > had.len()) {
                    colon = Some(whole);
                }
                i = end;
                continue;
            }
        }
        i = head;
        let token = &bytes[start..head];
        if stage_entry(token) && bare.is_none_or(|had| token.len() > had.len()) {
            bare = Some(token);
        }
    }
    let found = colon.or(bare)?;
    Some(found.iter().map(|&b| b as char).collect())
}

#[cfg(any(test, feature = "diagnostics"))]
fn name_in(shaders: &[Arc<Shader>]) -> String {
    let mut best = String::new();
    for s in shaders {
        if let Some(name) = embedded_name(&s.container)
            && name.len() > best.len()
        {
            best = name;
        }
    }
    best
}

#[cfg(any(test, feature = "diagnostics"))]
fn shader_label(p: &Pending) -> String {
    if let Some(name) = p.left_out {
        return name.to_owned();
    }
    let found = p.shader_name.get_or_init(|| name_in(&p.shaders));
    if !found.is_empty() {
        return found.clone();
    }
    if p.shaders.is_empty() {
        return "none".into();
    }
    p.shaders.iter().map(|s| format!("{}:{}", s.stage, s.hash)).collect::<Vec<_>>().join("+")
}

/// Why a pipeline that is not made yet was dropped. `split` is
/// [`drop_reasons`]: off, the older "being made" / left-out line; on, the
/// three reasons with the pipeline hash and a shader name.
#[cfg(any(test, feature = "diagnostics"))]
fn unmade_why(p: &Pending, split: bool) -> String {
    if !split {
        return if let Some(name) = p.left_out {
            format!("{}{name}", crate::failed::LEFT_OUT)
        } else {
            "being made".into()
        };
    }
    let queued = p.queue_state.load(Ordering::Relaxed) != NOT_QUEUED;
    crate::failed::making_drop(queued, p.left_out.is_some(), p.key(), &shader_label(p))
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(500);
        tracing::info!(millis, "host pipeline wait budget per frame");
        Duration::from_millis(millis)
    })
}

/// As [`wait_budget`], for passes over the screen (`VICTORY_SCREEN_WAIT_MS`,
/// 10 s; 0: like the others). Below the title's 30 s watchdog.
fn screen_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_SCREEN_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(10_000);
        Duration::from_millis(millis)
    })
}

/// Whether passes over the screen go before the others.
fn screens_first() -> bool {
    !screen_budget().is_zero()
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.left_out.is_some() {
        return None;
    }
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    if p.made.get().is_none() {
        start(p, REQUESTED);
        wait(p, crate::frame::current());
    }
    p.made.get()?.as_ref()
}

/// Pipelines made again for the formats bound at a draw, since the start.
static VARIANTS: AtomicU64 = AtomicU64::new(0);

/// Whether a draw into views of other formats than its pipeline declares
/// gets a host pipeline for those formats ([`for_formats`]). Off unless
/// `VICTORY_TARGET_VARIANTS=1`; otherwise the draw goes out with the
/// title's pipeline, as before.
fn variants_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TARGET_VARIANTS").is_some_and(|v| v == "1"))
}

/// The host pipeline for a draw with `p` into the bound formats `bound`.
/// When they disagree with what `p` declares ([`super::host::formats_differ`])
/// and [`variants_on`], that is a pipeline of the same shaders and state
/// made for the bound formats. It is queued at its first draw, made by a
/// worker, and waited for within the frame's budget; the draw is dropped
/// while it is not there ([`ready`]). A draw whose formats agree pays the
/// compare and uses `p`.
pub(crate) fn for_draw<'a>(p: &'a Arc<Pending>, bound: &TargetFormats) -> Option<&'a HostPipeline> {
    if p.formats != *bound && super::host::formats_differ(&p.formats, bound) {
        return for_formats(p, bound);
    }
    ready(p)
}

/// The host pipeline of `p`'s shaders and state for the formats `bound`
/// does not declare. Off: `p` itself.
#[cold]
fn for_formats<'a>(p: &'a Arc<Pending>, bound: &TargetFormats) -> Option<&'a HostPipeline> {
    match p.graphics.as_deref() {
        Some(g) if variants_on() => ready(variant(p, g, &super::host::formats_for(&p.formats, bound))),
        _ => ready(p),
    }
}

/// The pipeline of `p`'s shaders and its state `g` for the formats
/// `bound`: found in the chain that starts at `p`, or added to it (logged
/// then: once per pipeline and set of formats).
fn variant<'a>(p: &'a Arc<Pending>, g: &GraphicsState, bound: &TargetFormats) -> &'a Arc<Pending> {
    let mut at = p;
    // `bound` is what `host::formats_for` gives, and a state made for it
    // declares exactly it, so the walk ends at the link made for it. The
    // count is a bound all the same: past it the title's own pipeline.
    for _ in 0..64 {
        if let Some(next) = at.variant.get() {
            if next.formats == *bound {
                return next;
            }
            at = next;
            continue;
        }
        // Built before the link is taken: the new pipeline reads `p` and
        // `g`, which are the same pipeline whose link is being filled.
        let shaders = p.shaders.clone();
        let declared = p.formats;
        let pipeline = hashes(p);
        let state = g.with_formats(bound);
        let next = at.variant.get_or_init(|| {
            let so_far = VARIANTS.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::info!(
                pipeline = %pipeline,
                declared = ?declared,
                bound = ?bound,
                so_far,
                "a pipeline is made again for the render target formats bound at its draw (eight render targets, then depth; VICTORY_TARGET_VARIANTS=1)"
            );
            Arc::new(Pending::new(shaders, Some(Box::new(state))))
        });
        if next.formats == *bound {
            return next;
        }
        at = next;
    }
    p
}

/// Makes `p` on this thread if it is not made yet (waiting for a worker
/// that is making it); whether it has a host pipeline.
pub(crate) fn now(p: &Arc<Pending>) -> bool {
    if p.left_out.is_some() {
        return false;
    }
    p.make();
    p.made.get().is_some_and(|made| made.is_some())
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it),
/// or if it is a confirmed opening-scene / bake producer.
pub(crate) fn early(p: &Arc<Pending>) {
    if p.left_out.is_some() {
        return;
    }
    if !p.shaders.is_empty() && (p.scene_input || super::library::wanted_early(&p.key)) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, if p.scene_input { REQUESTED } else { EARLY });
    }
}

/// Whether the frame's budget is one for all threads together: a thread
/// that waits takes what is left of it and gives back what it did not
/// use. Counted only when a wait had ended, the budget was one per thread:
/// each of the title's five recording threads began its first wait of a
/// frame with nothing counted yet and waited the whole half second, 2.5 s
/// of waiting in every frame in which pipelines were being made (runs
/// 34-49: "2504 ms waiting for pipelines" in each such frame).
/// On by default; `VICTORY_PIPELINE_WAIT_SHARED=0` counts as before.
fn shared_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_SHARED").is_none_or(|v| v != "0"))
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    let screen = p.screen && screens_first();
    let budget = if screen { screen_budget() } else { wait_budget() };
    if budget.is_zero() {
        return;
    }
    // The frame the budget belongs to, and what was waited of it (or is
    // taken by a thread waiting now); of the passes over the screen too.
    static SPENT: [Mutex<(u64, Duration)>; 2] = [const { Mutex::new((u64::MAX, Duration::ZERO)) }; 2];
    let spent_of = &SPENT[usize::from(screen)];
    let shared = shared_budget();
    let left = {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        if spent.0 != frame {
            *spent = (frame, Duration::ZERO);
        }
        let left = budget.saturating_sub(spent.1);
        if shared {
            spent.1 += left;
        }
        left
    };
    if left.is_zero() {
        return;
    }
    let waiting = crate::timeline::span("pipeline: first use waits for a host pipeline");
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    let waited = started.elapsed().min(left);
    waiting.end();
    if screen && waited > Duration::from_millis(500) {
        tracing::info!(
            ms = waited.as_millis() as u64,
            made = p.made.get().is_some(),
            shaders = ?p.shaders,
            "a pass over the screen was waited for (VICTORY_SCREEN_WAIT_MS=0: dropped past half a second a frame)"
        );
    }
    {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        // A frame that began meanwhile has its own budget.
        if spent.0 == frame {
            if shared {
                spent.1 = spent.1.saturating_sub(left - waited);
            } else {
                spent.1 += waited;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

/// How many workers may make pipelines at once that are known to take long
/// and that no draw has asked for: a quarter of them. On a fresh cache (a
/// change of the translator or the driver) the early list begins with
/// programs that take tens of seconds each: all 16 workers were in them
/// through the whole of the title's loading, 97 pipelines made in 20 s, and
/// the first minute and a half of the world ran at 0.5 to 1.6 s a frame,
/// dropping up to 2,900 draws a frame (run shaders-1).
/// `VICTORY_PIPELINE_SLOW=0`: in the order the title creates them, as before.
fn slow_limit() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        if std::env::var_os("VICTORY_PIPELINE_SLOW").is_some_and(|v| v == "0") {
            usize::MAX
        } else {
            (worker_count(None, std::thread::available_parallelism().map_or(2, |n| n.get())) / 4).max(1)
        }
    })
}

struct Queue {
    /// Requested passes over the screen: before all others, and the only ones
    /// [`SCREEN_WORKERS`] take.
    screen: VecDeque<Arc<Pending>>,
    early: VecDeque<Arc<Pending>>,
    /// Those of the early ones known to take long, and how many of them are
    /// being made ([`slow_limit`]).
    slow: VecDeque<Arc<Pending>>,
    slow_now: usize,
    requested: VecDeque<Arc<Pending>>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            screen: VecDeque::new(),
            early: VecDeque::new(),
            slow: VecDeque::new(),
            slow_now: 0,
            requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        self.push_as(p, priority, slow_limit() != usize::MAX && super::library::slow(&p.key))
    }

    fn push_as(&mut self, p: &Arc<Pending>, priority: u8, slow: bool) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED && p.screen && screens_first() {
            self.screen.push_back(p.clone());
        } else if priority == REQUESTED {
            self.requested.push_back(p.clone());
        } else if slow {
            self.slow.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    fn pop(&mut self, screens_only: bool, slow_limit: usize) -> Option<Arc<Pending>> {
        // An early entry can have been promoted and claimed already.
        let unclaimed = |list: &mut VecDeque<Arc<Pending>>| {
            std::iter::from_fn(|| list.pop_front()).find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        };
        let mut p = self.screen.pop_front();
        if p.is_none() && !screens_only {
            p = self.requested.pop_front().or_else(|| unclaimed(&mut self.early));
            if p.is_none() && self.slow_now < slow_limit {
                p = unclaimed(&mut self.slow);
                if let Some(p) = &p {
                    p.was_slow.store(true, Ordering::Relaxed);
                    self.slow_now += 1;
                }
            }
        }
        let p = p?;
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some(p)
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        // (One woken could be a worker that takes passes over the screen
        // alone: it would leave the pipeline to nobody.)
        AVAILABLE.notify_all();
    }
}

/// Workers that make passes over the screen alone, besides the others.
const SCREEN_WORKERS: usize = 2;

fn next(screens_only: bool) -> Arc<Pending> {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = queue.pop(screens_only, slow_limit()) {
            return p;
        }
        queue = AVAILABLE.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// How many workers make pipelines: `asked` (`VICTORY_PIPELINE_WORKERS`, 1 to
/// 64), else two thirds of the logical processors, 2 to 16. They run below the
/// title's own threads, and a cold cache queues hundreds at once.
fn worker_count(asked: Option<usize>, processors: usize) -> usize {
    worker_count_for(asked, processors, false)
}

fn worker_count_for(asked: Option<usize>, processors: usize, limited_ram: bool) -> usize {
    match asked {
        Some(n) => n.clamp(1, 64),
        None if limited_ram => (processors * 3 / 4).clamp(2, 8),
        None => (processors * 2 / 3).clamp(2, 16),
    }
}

/// Starts the workers ([`worker_count`]) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = worker_count_for(
            std::env::var("VICTORY_PIPELINE_WORKERS")
                .ok()
                .and_then(|v| v.trim().parse().ok()),
            std::thread::available_parallelism().map_or(2, |n| n.get()),
            crate::residency::limited_host_ram(),
        );
        tracing::info!(workers, "pipeline workers");
        let screens = if screens_first() { SCREEN_WORKERS } else { 0 };
        for i in 0..workers + screens {
            let screens_only = i >= workers;
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(move || {
                    // Below the title's own threads (loading, recording),
                    // and the streaming one, which is below normal itself
                    // (`VICTORY_PIPELINE_LOWEST=0`: level with it, as before).
                    let lowest = !screens_only && std::env::var_os("VICTORY_PIPELINE_LOWEST").is_none_or(|v| v != "0");
                    // SAFETY: a plain call on our own thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let priority = if lowest { THREAD_PRIORITY_LOWEST } else { THREAD_PRIORITY_BELOW_NORMAL };
                        let _ = SetThreadPriority(GetCurrentThread(), priority);
                    }
                    loop {
                        let p = next(screens_only);
                        p.make();
                        if p.was_slow.load(Ordering::Relaxed) {
                            let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
                            queue.slow_now -= 1;
                            AVAILABLE.notify_all();
                        }
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            super::library::save(false);
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            } else {
                running = true;
            }
        }
        running
    })
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more. The
/// cache is saved then only when the library's save policy says so (at
/// exit by default; during play, `VICTORY_PIPELINE_LIBRARY_SAVE=<minutes>`).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
fn shader_bytes(bytes: &[u8]) -> Arc<super::Shader> {
    Arc::new(super::Shader {
        stage: "compute",
        hash: "0011223344556677".into(),
        container: bytes.to_vec(),
        ambient: None,
        probe: None,
        translated: OnceLock::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structure_tracing_is_known_by_name() {
        assert_eq!(traces_structures(b"..\0CSGenerateHairVB_c88a8f19_Wrapped\0.."), Some("CSGenerateHairVB"));
        assert_eq!(
            traces_structures(b"..\0CS_ScreenProbes_TraceRays_RayBinning_BVH_Mixed\0.."),
            Some("CS_ScreenProbes_TraceRays_BVH")
        );
        assert_eq!(traces_structures(b"..\0CS_ScreenProbes_TraceRays_Voxels\0.."), None);
        assert_eq!(traces_structures(b"..\0CS_ScreenProbes_TraceRays_Fallback\0.."), None);
        assert_eq!(
            traces_structures(b"..\0CS_RaytraceVBuffer_ShaderBinning_WithFar\0.."),
            Some("CS_RaytraceVBuffer")
        );
        assert_eq!(traces_structures(b"..\0CS_RaytraceVBuffer_Voxels_RayBinning\0.."), None);
        assert_eq!(traces_structures(b"..\0CS_ScreenProbes_ActivateRays_0_Wrapped\0.."), None);
    }

    #[test]
    fn worker_count_follows_the_processors_and_the_switch() {
        assert_eq!(worker_count(None, 1), 2);
        assert_eq!(worker_count(None, 24), 16);
        assert_eq!(worker_count(None, 64), 16);
        assert_eq!(worker_count(Some(0), 24), 1);
        assert_eq!(worker_count(Some(20), 24), 20);
        assert_eq!(worker_count_for(None, 8, true), 6);
        assert_eq!(worker_count_for(None, 16, true), 8);
        assert_eq!(worker_count_for(None, 24, true), 8);
        assert_eq!(worker_count_for(Some(12), 8, true), 12);
    }

    #[test]
    fn a_panicking_pipeline_is_refused_and_the_worker_continues() {
        let bad = Arc::new(Pending::new(Vec::new(), None));
        let good = Arc::new(Pending::new(Vec::new(), None));
        let failed = bad.clone();
        let next = good.clone();
        let worker = std::thread::spawn(move || {
            failed.make_with(|| panic!("synthetic pipeline build panic"));
            next.make_with(|| None);
        });

        worker.join().expect("the pipeline worker should survive");
        assert!(matches!(bad.made.get(), Some(None)));
        assert!(matches!(good.made.get(), Some(None)));

        bad.make_with(|| panic!("the refused pipeline should stay complete"));
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// A draw into other formats than its pipeline declares finds the
    /// pipeline for them, which exists once per set of formats. It is not
    /// queued until a draw asks ([`for_formats`]).
    #[test]
    fn a_pipeline_for_bound_formats_is_made_once() {
        let p = Arc::new(Pending::new(Vec::new(), Some(Box::new(GraphicsState::declaring(24)))));
        assert_eq!(p.formats, [24, 0, 0, 0, 0, 0, 0, 0, 0]);
        let g = p.graphics.as_deref().unwrap();
        let (a, b) = ([26, 0, 0, 0, 0, 0, 0, 0, 0], [28, 0, 0, 0, 0, 0, 0, 0, 0]);
        let first = variant(&p, g, &a);
        assert_eq!(first.formats, a);
        assert_ne!(first.key, p.key);
        assert!(Arc::ptr_eq(first, variant(&p, g, &a)));
        let second = variant(&p, g, &b);
        assert_eq!(second.formats, b);
        assert!(!Arc::ptr_eq(first, second));
        assert_ne!(second.key, first.key);
        assert!(Arc::ptr_eq(first, variant(&p, g, &a)));
        assert!(Arc::ptr_eq(second, variant(&p, g, &b)));
        assert_eq!(first.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        assert_eq!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p).is_none());
        assert!(p.made.get().is_none());
        assert!(super::super::library::wanted_early(&p.key));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &earlier));
        assert!(queue.pop(false, 1).is_none());
    }

    #[test]
    fn pipelines_known_to_take_long_leave_workers_for_the_others() {
        let new = || Arc::new(Pending::new(Vec::new(), None));
        let (long, longer, asked, quick) = (new(), new(), new(), new());
        let mut queue = Queue::new();
        for p in [&long, &longer, &asked] {
            assert_eq!(queue.push_as(p, EARLY, true), Some(true));
        }
        assert_eq!(queue.push_as(&quick, EARLY, false), Some(true));
        // The quick one first, then one long one at a time; one a draw asks
        // for waits for nobody.
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &quick));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &long));
        assert!(queue.pop(false, 1).is_none());
        assert_eq!(queue.push_as(&asked, REQUESTED, true), Some(false));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &asked));
        assert!(!asked.was_slow.load(Ordering::Relaxed));
        assert!(queue.pop(false, 1).is_none());
        queue.slow_now -= 1;
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &longer));
        assert!(queue.pop(false, 1).is_none());
    }

    /// Passes over the screen go before all others, and workers of their own
    /// take nothing else.
    #[test]
    fn passes_over_the_screen_go_first() {
        let of = |screen| {
            let mut p = Pending::new(Vec::new(), None);
            p.screen = screen;
            Arc::new(p)
        };
        let (world, screen) = (of(false), of(true));
        let mut queue = Queue::new();
        assert_eq!(queue.push(&world, REQUESTED), Some(true));
        assert_eq!(queue.push(&screen, REQUESTED), Some(true));
        assert!(Arc::ptr_eq(&queue.pop(true, 1).unwrap(), &screen));
        assert!(queue.pop(true, 1).is_none());
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &world));
    }

    #[test]
    fn drop_reasons_are_off_unless_asked() {
        assert!(!drop_reasons_value(None));
        assert!(!drop_reasons_value(Some(std::ffi::OsStr::new("0"))));
        assert!(!drop_reasons_value(Some(std::ffi::OsStr::new(""))));
        assert!(drop_reasons_value(Some(std::ffi::OsStr::new("1"))));
    }

    #[test]
    fn without_the_switch_an_unmade_pipeline_is_still_being_made() {
        let p = Pending::new(vec![shader_bytes(b"mod:CS_KeepThisName")], None);
        assert_eq!(unmade_why(&p, false), "being made");
        let mut left = Pending::new(vec![shader_bytes(b"plain bytes")], None);
        left.left_out = Some("CS_RaytracingResolve");
        assert_eq!(
            unmade_why(&left, false),
            format!("{}CS_RaytracingResolve", crate::failed::LEFT_OUT)
        );
    }

    #[test]
    fn with_the_switch_each_reason_names_the_pipeline_and_the_shader() {
        let p = Pending::new(vec![shader_bytes(b"xx mod:CS_KeepThisName\0")], None);
        assert_eq!(
            unmade_why(&p, true),
            crate::failed::making_drop(false, false, p.key(), "mod:CS_KeepThisName")
        );
        for state in [EARLY, REQUESTED, MAKING] {
            p.queue_state.store(state, Ordering::Relaxed);
            assert_eq!(
                unmade_why(&p, true),
                crate::failed::making_drop(true, false, p.key(), "mod:CS_KeepThisName"),
                "{state}"
            );
        }
        let mut left = Pending::new(vec![shader_bytes(b"CS_OtherNameHere")], None);
        left.left_out = Some("CS_RaytracingResolve");
        left.queue_state.store(REQUESTED, Ordering::Relaxed);
        assert_eq!(
            unmade_why(&left, true),
            crate::failed::making_drop(true, true, left.key(), "CS_RaytracingResolve")
        );
        let plain = Pending::new(vec![shader_bytes(b"no name bytes")], None);
        assert_eq!(
            unmade_why(&plain, true),
            crate::failed::making_drop(false, false, plain.key(), "compute:0011223344556677")
        );
        let empty = Pending::new(Vec::new(), None);
        assert!(unmade_why(&empty, true).ends_with(" none"));
    }

    #[test]
    fn shader_names_are_taken_from_the_container() {
        assert_eq!(
            embedded_name(b"\0deferred_tiled:CS_WaterTransmittanceOpticalDepthProcessing\0CS_ShortName").as_deref(),
            Some("deferred_tiled:CS_WaterTransmittanceOpticalDepthProcessing")
        );
        // The module:entry form wins even when a bare name is longer.
        assert_eq!(
            embedded_name(b"VS_AVeryLongBareNameHere\0m:CS_ShortName").as_deref(),
            Some("m:CS_ShortName")
        );
        assert_eq!(
            embedded_name(b"a:CS_Short\0module:CS_LongerEntryName").as_deref(),
            Some("module:CS_LongerEntryName")
        );
        assert_eq!(embedded_name(b"CS_Short\0VS_MuchLongerName").as_deref(), Some("VS_MuchLongerName"));
        // Two characters after the stage, or a name glued to the byte before it, is not a name.
        assert_eq!(embedded_name(b"noise CS_No xCS_RaytracingResolve"), None);
        assert_eq!(embedded_name(b"no name"), None);
    }
}
