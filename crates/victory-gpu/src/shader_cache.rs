//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content, by the translator's
//! version ([`victory_shader::TRANSLATOR_VERSION`]) and by its settings.
//! A rebuild that leaves the version keeps every translation: the store
//! grows by the shaders a change made translate, not by a whole new set
//! each build (16 GB of sets of earlier builds had piled up when the key
//! was the translator's source). Entries go through `victory-pipeline`'s
//! content-addressed cache, which writes them atomically; an entry that
//! does not read back is translated again.
//!
//! Each entry also names the build of the translator that made it (the
//! BLAKE3 of its sources, `build.rs`). One of an earlier build is served
//! as it is, and counted; `victory shaders` ([`precompile`]) translates it
//! again, keeps the new one and says how many came out different (then
//! the version should have been bumped). `VICTORY_SHADER_FRESH=1` does the
//! same in a run; `VICTORY_SHADER_KEY=source` keys by the translator's
//! source as before (a separate set for each build).
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too, and [`FORMAT`] bumped.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The encoding of entries ([`encode`]), part of the key.
const FORMAT: u32 = 2;

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::resource::RootIndexedView;
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
/// Hits on entries an earlier build of the translator made (served as they
/// are, or when refreshing translated again); of those translated again,
/// how many came out different.
static STALE: AtomicU64 = AtomicU64::new(0);
static REFRESHED: AtomicU64 = AtomicU64::new(0);
static CHANGED: AtomicU64 = AtomicU64::new(0);

/// The build of the translator (the BLAKE3 of its sources, `build.rs`).
const SOURCE: &str = env!("VICTORY_TRANSLATOR");

/// How cached translations are keyed (`VICTORY_SHADER_KEY`).
fn keyed_by_source() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SHADER_KEY").is_ok_and(|v| v == "source"))
}

/// Whether a run translates entries of an earlier build again
/// (`VICTORY_SHADER_FRESH=1`; [`precompile`] always does).
fn refresh_in_runs() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SHADER_FRESH").is_some_and(|v| v == "1"))
}

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// Switches whose value is part of the cache key. A run with one of them
/// set must not be served the translations cached when it was unset.
fn watched_switches() -> &'static [&'static str] {
    &[
        "VICTORY_WAVE_WIDE",
        "VICTORY_WAVE_DEEP_SKIPS",
        "VICTORY_WAVE_OWN_BITS",
        "VICTORY_WAVE_LOCAL",
        "VICTORY_LDS_FLAGS",
        "VICTORY_F16_NEAREST",
        "VICTORY_HELPER_OWN",
        "VICTORY_FIRSTLANE_OWN",
        "VICTORY_FIRST_HALF",
        "VICTORY_STRICT_JOINS",
        "VICTORY_MESH_OUTPUT_STRICT",
        "VICTORY_BVH_FENCES",
        "VICTORY_BVH_BOUNDS",
        "VICTORY_RT_MISS",
        "VICTORY_RT_MISS_HIT",
    ]
}

/// Since the start: entries of an earlier build of the translator met,
/// those of them translated again, and those that came out different.
pub fn stale_counts() -> (u64, u64, u64) {
    (STALE.load(Ordering::Relaxed), REFRESHED.load(Ordering::Relaxed), CHANGED.load(Ordering::Relaxed))
}

/// What identifies a translation beside its container: the translator's
/// version (or with `VICTORY_SHADER_KEY=source` its sources), the form of
/// entries, the settings it runs with, and what it was told of the host.
/// Not our executable: a rebuild of anything else used to translate every
/// shader again.
fn build() -> [u8; 32] {
    // The view heap's null slot differs by host GPU: a translation cached
    // without the clamp must not be served (`host_settings` below).
    crate::descriptor::host::probe_view_limit();
    fingerprint()
}

/// [`build`] without the host probe, so a cache lookup from a log uses the
/// same fingerprint a translation stores under once the probe has already run.
fn fingerprint() -> [u8; 32] {
    static FIXED: OnceLock<Vec<u8>> = OnceLock::new();
    let fixed = FIXED.get_or_init(|| {
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        let mut bytes = if keyed_by_source() {
            SOURCE.as_bytes().to_vec()
        } else {
            format!("version {}", victory_shader::TRANSLATOR_VERSION).into_bytes()
        };
        bytes.extend_from_slice(&FORMAT.to_le_bytes());
        bytes.extend_from_slice(&limit.to_le_bytes());
        // Which group barriers wait is part of the translation too.
        bytes.push(victory_shader::emit::group_sync());
        // So are the switches that pick the form of a translated program
        // (a wave64 program, or a compute dispatch spread across z):
        // without them here a run with `VICTORY_WAVE_WIDE=0` was
        // served the translations an earlier run of the same
        // executable had cached with the switch unset.
        for name in watched_switches() {
            bytes.extend_from_slice(name.as_bytes());
            bytes.push(b'=');
            if let Some(value) = std::env::var_os(name) {
                bytes.extend_from_slice(value.as_encoded_bytes());
            }
            bytes.push(0);
        }
        // And which clean-up passes ran over the translated code.
        bytes.extend_from_slice(&victory_shader::emit::shader_opt().to_le_bytes());
        bytes.extend_from_slice(&opt_in_marks(
            victory_shader::emit::constants_raw(),
            victory_shader::emit::dispatch_spread_compiled(),
            victory_shader::emit::typed_offset_refused(),
        ));
        bytes
    });
    let mut hash = blake3::Hasher::new();
    hash.update(fixed);
    for value in victory_shader::emit::host_settings() {
        hash.update(&value.to_le_bytes());
    }
    // And the heap map bindless views are looked up in (`set_view_heap_map`).
    let map = victory_shader::emit::view_heap_map().map_or([u32::MAX; 2], |(w, s)| [w, s]);
    for word in map {
        hash.update(&word.to_le_bytes());
    }
    *hash.finalize().as_bytes()
}

/// What the opt-in switches that change translations add to [`build`]'s
/// bytes, each only when on, so the translations cached without them keep
/// their key:
/// - renderer port 5 (`VICTORY_PORT_CBV=1`): constant buffers read as raw
///   bytes;
/// - the spread prologue of compute programs (`VICTORY_SPREAD` 1 or alt,
///   which translate alike), when compiled in;
/// - `VICTORY_TYPED_OFFSET=refuse`: typed buffer accesses at an offset
///   refused.
///
/// In the order they came in, so every key made before the last was added
/// is still made.
fn opt_in_marks(constants_raw: bool, spread: bool, typed_offset_refused: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    if constants_raw {
        bytes.extend_from_slice(b"VICTORY_PORT_CBV=1");
    }
    if spread {
        bytes.extend_from_slice(b"VICTORY_SPREAD=1");
    }
    if typed_offset_refused {
        bytes.extend_from_slice(b"VICTORY_TYPED_OFFSET=refuse");
    }
    bytes
}

/// What tells one set of translations from another ([`build`]).
pub(crate) fn translation() -> [u8; 32] {
    build()
}

/// Puts the translation of `container` into the cache before any title asks
/// for it (`victory shaders`): whether it was there already, or why it does
/// not translate. The host is opened first, as a run does: what the card
/// can do decides the form of some programs.
pub fn precompile(container: &[u8]) -> Result<bool, String> {
    let _ = crate::host::host();
    let misses = MISSES.load(Ordering::Relaxed);
    // (by its hash and under its own key, as a run asks for it: the
    // per-program switches)
    let hash = program_hash(container);
    let translate = |c: &[u8]| translate_program(&hash, c);
    match CACHE.get() {
        Some(cache) => cached(cache, &key_of(&hash, container), container, translate, true)?,
        None => translate(container)?,
    };
    // (Counts other threads' misses too: the caller only adds them up.)
    Ok(MISSES.load(Ordering::Relaxed) == misses)
}

/// The translation of a console shader container, from the cache when
/// there is one.
pub(crate) fn host_shader(container: &[u8]) -> Result<HostShader, String> {
    let hash = program_hash(container);
    let translate = |c: &[u8]| translate_program(&hash, c);
    let shader = match CACHE.get() {
        Some(cache) => cached(cache, &key_of(&hash, container), container, translate, refresh_in_runs()),
        None => translate(container),
    }?;
    #[cfg(feature = "diagnostics")]
    let shader = paired_override(&hash, shader);
    Ok(shader)
}

/// Diagnostics (`VICTORY_PAIRED_SHADER_DIR=<dir>`, off when unset): a
/// compute shader's bytecode from `<dir>/<hash>.vsh`, an entry the paired
/// Wave32 translator made for a host of 32-lane waves and a view heap of
/// 999,999 slots, when its bindings are those of our own translation. Our
/// translation stays the source of the bindings and stays in the cache;
/// a missing or different entry leaves it as it is.
#[cfg(feature = "diagnostics")]
fn paired_override(hash: &str, mut shader: HostShader) -> HostShader {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    let Some(dir) = DIR.get_or_init(|| std::env::var_os("VICTORY_PAIRED_SHADER_DIR").map(PathBuf::from)) else {
        return shader;
    };
    if shader.stage != HwStage::Cs || victory_shader::emit::host_settings() != [32, 999_999, 0, 0] {
        return shader;
    }
    let bytes = match std::fs::read(dir.join(format!("{hash}.vsh"))) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return shader,
        Err(e) => {
            tracing::warn!(hash, error = %e, "could not read paired shader");
            return shader;
        }
    };
    if let Some(dxil) = paired_bytecode(&bytes, &shader) {
        shader.dxil = dxil;
        tracing::info!(hash, "using paired Wave32 bytecode with matching shader bindings");
        return shader;
    }
    tracing::warn!(hash, "paired shader entry did not match the translated shader bindings");
    shader
}

/// The bytecode of a paired entry (`VSH1`: this format without the build
/// that made it) whose bindings are `shader`'s.
#[cfg(any(test, feature = "diagnostics"))]
fn paired_bytecode(bytes: &[u8], shader: &HostShader) -> Option<Vec<u8>> {
    let body = bytes.strip_prefix(b"VSH1")?;
    let mut ours = MAGIC.to_vec();
    ours.extend(0u32.to_le_bytes());
    ours.extend(body);
    let (mut paired, _) = decode(&ours)?;
    let dxil = std::mem::replace(&mut paired.dxil, shader.dxil.clone());
    (paired == *shader && dxil.starts_with(b"DXBC")).then_some(dxil)
}

/// The hash the log names the shader of `container` by (`pipeline.rs`),
/// which the per-program switches take.
fn program_hash(container: &[u8]) -> String {
    blake3::hash(container).to_hex()[..16].to_owned()
}

/// The translation of `container`, the program whose hash is `hash`.
fn translate_program(hash: &str, container: &[u8]) -> Result<HostShader, String> {
    use victory_shader::emit;
    emit::for_program(hash, || emit::translate(container)).map_err(|e| e.to_string())
}

/// The key the entries of `container`, the program whose hash is `hash`,
/// are filed under ([`program_key`]).
fn key_of(hash: &str, container: &[u8]) -> [u8; 32] {
    key_from(&build(), hash, container)
}

/// [`key_of`] from the build's key given (a lookup's is made without the
/// host probe: [`fingerprint`]).
fn key_from(build: &[u8; 32], hash: &str, container: &[u8]) -> [u8; 32] {
    use victory_shader::emit;
    // (the container is parsed second: only a program a list names, or
    // with `VICTORY_WAVE_DEEP_SKIPS=0` or `VICTORY_WAVE_MASK_COPIES=all`
    // any, gets there)
    let off = !emit::deep_skips_allowed(Some(hash)) && emit::wave64_retried(container);
    let copies = emit::mask_copies(Some(hash)) && emit::wave64_program(container);
    let wide = emit::wave_wide_only(Some(hash)) && emit::wave64_retried(container);
    program_key(build, off, copies, wide)
}

/// The key of one program's entries: the build's, mixed with each
/// per-program decision that differs from the default:
/// - The program is kept out of the wave64 form "without skips, scalars
///   followed" (`VICTORY_WAVE_DEEP_SKIPS_NOT`, `_ONLY`:
///   `victory_shader::emit::deep_skips_allowed`). Only for a wave64 compute
///   program on a host of 32-lane waves, the only ones it can change.
/// - The program is translated with a wave64 mask restored from scalar
///   copies taken as the mask (`VICTORY_WAVE_MASK_COPIES`:
///   `victory_shader::emit::mask_copies`). Only for a console wave64
///   program, the only ones it can change.
/// - The program may take the wave-wide form (`VICTORY_WAVE_WIDE_ONLY`:
///   `victory_shader::emit::wave_wide_only`). Only for a wave64 compute
///   program on a host of 32-lane waves, the only ones that form is for.
///
/// So the lists are in no key: changing one translates again only the
/// programs whose decision it changes, and a program the lists leave alone
/// keeps the key it had without any list.
fn program_key(build: &[u8; 32], deep_skips_off: bool, mask_copies: bool, wave_wide: bool) -> [u8; 32] {
    if !deep_skips_off && !mask_copies && !wave_wide {
        return *build;
    }
    let mut h = blake3::Hasher::new();
    h.update(build);
    if deep_skips_off {
        h.update(b"VICTORY_WAVE_DEEP_SKIPS off for this program");
    }
    if mask_copies {
        h.update(b"VICTORY_WAVE_MASK_COPIES on for this program");
    }
    if wave_wide {
        h.update(b"VICTORY_WAVE_WIDE_ONLY on for this program");
    }
    *h.finalize().as_bytes()
}

/// The converter id and version stored translations were written with.
/// The build fingerprint in the input does the versioning.
const CONVERTER_ID: &str = "shader.dxil";
const CONVERTER_VERSION: u32 = 1;

/// The cache entry's converter: the build and the container in, the
/// encoded translation out. Keeps the result in memory for the caller on a
/// miss.
struct Translate<'a, F> {
    translate: F,
    /// The build of the translator, written into the entry.
    source: &'a str,
    made: Mutex<Option<HostShader>>,
}

/// Looks up a cache entry and does not convert.
struct Presence;

impl Converter for Presence {
    fn id(&self) -> &'static str {
        CONVERTER_ID
    }
    fn version(&self) -> u32 {
        CONVERTER_VERSION
    }
    fn convert(&self, _: &[u8]) -> anyhow::Result<Vec<u8>> {
        Err(anyhow::Error::msg("a cache probe does not convert"))
    }
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<'_, F> {
    fn id(&self) -> &'static str {
        CONVERTER_ID
    }
    fn version(&self) -> u32 {
        CONVERTER_VERSION
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        let shader = (self.translate)(&input[32..]).map_err(anyhow::Error::msg)?;
        let entry = encode(&shader, self.source);
        *self.made.lock().unwrap_or_else(|err| err.into_inner()) = Some(shader);
        Ok(entry)
    }
}

/// Whether `cache` already holds the translation `cached` would store for
/// `key` and `container`. Does not translate.
fn holds(cache: &Cache, key: &[u8; 32], container: &[u8]) -> bool {
    let input = [&key[..], container].concat();
    cache.contains(&Presence, &input)
}

/// Whether the shader cache already holds a translation of `container` for
/// this translator build. The key is the one [`host_shader`] stores under.
/// Does not translate and does not probe the host. False when no cache is
/// configured.
pub(crate) fn contains(container: &[u8]) -> bool {
    let Some(cache) = CACHE.get() else {
        return false;
    };
    let hash = program_hash(container);
    holds(cache, &key_from(&fingerprint(), &hash, container), container)
}

/// `refresh`: an entry an earlier build of the translator made is
/// translated again (and replaced) rather than served.
fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
    refresh: bool,
) -> Result<HostShader, String> {
    cached_by(cache, build, SOURCE, container, translate, refresh)
}

/// [`cached`] with the build of the translator given (`source`).
fn cached_by(
    cache: &Cache,
    build: &[u8; 32],
    source: &str,
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
    refresh: bool,
) -> Result<HostShader, String> {
    let input = [&build[..], container].concat();
    let mut converter = Translate {
        translate,
        source,
        made: Mutex::new(None),
    };
    let mut old: Option<HostShader> = None;
    for _ in 0..3 {
        let entry = cache.get_or_convert(&converter, &input);
        if let Some(shader) = converter.made.get_mut().unwrap_or_else(|err| err.into_inner()).take() {
            // Translated: handed over whether or not the entry reached the
            // disk (a full disk, a locked entry).
            match entry {
                Ok(path) => tracing::debug!(path = %path.display(), "shader translated and cached"),
                Err(error) => {
                    static WARNED: AtomicBool = AtomicBool::new(false);
                    if !WARNED.swap(true, Ordering::Relaxed) {
                        tracing::warn!(error = %format!("{error:#}"), "shader cache write failed; using the translation in memory");
                    }
                }
            }
            if let Some(old) = old.take() {
                REFRESHED.fetch_add(1, Ordering::Relaxed);
                if old != shader {
                    CHANGED.fetch_add(1, Ordering::Relaxed);
                }
            }
            MISSES.fetch_add(1, Ordering::Relaxed);
            return Ok(shader);
        }
        let path = entry.map_err(|e| format!("{e:#}"))?;
        if let Some((shader, made_by)) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            if made_by != source.as_bytes() {
                STALE.fetch_add(1, Ordering::Relaxed);
                if refresh && old.is_none() {
                    // Translated again in its place; what came out before
                    // is what the new one is compared with.
                    tracing::debug!(path = %path.display(), "shader cache entry of an earlier translator; translating again");
                    old = Some(shader);
                    let _ = std::fs::remove_file(&path);
                    continue;
                }
                static SAID: AtomicBool = AtomicBool::new(false);
                if !SAID.swap(true, Ordering::Relaxed) {
                    tracing::info!("shader translations of an earlier build of the translator (same version) are served; `victory shaders` translates them again");
                }
            }
            HITS.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader from the cache");
            return Ok(shader);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    (converter.translate)(container)
}

const MAGIC: &[u8; 4] = b"VSH2";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists, after the build of the
/// translator that made the entry (`source`, length-prefixed).
fn encode(s: &HostShader, source: &str) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    word(&mut out, source.len() as u32);
    out.extend(source.as_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, s.root_indexed_views.len() as u32);
    for view in &s.root_indexed_views {
        for value in [
            u32::from(view.table_slot),
            u32::from(view.constant_slot),
            view.constant_dword,
            view.byte_offset,
            u32::from(view.shift),
        ] {
            word(&mut out, value);
        }
    }
    word(&mut out, s.half_views.len() as u32);
    for &(slot, dword) in &s.half_views {
        word(&mut out, u32::from(slot));
        word(&mut out, dword);
    }
    out
}

/// The shader and the build of the translator that made it.
fn decode(bytes: &[u8]) -> Option<(HostShader, Vec<u8>)> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let made_by = r.list()?;
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let n = r.word()? as usize;
    let mut root_indexed_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        root_indexed_views.push(RootIndexedView {
            table_slot: u8::try_from(r.word()?).ok()?,
            constant_slot: u8::try_from(r.word()?).ok()?,
            constant_dword: r.word()?,
            byte_offset: r.word()?,
            shift: u8::try_from(r.word()?).ok().filter(|&s| s < 32)?,
        });
    }
    let n = r.word()? as usize;
    let mut half_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        half_views.push((u8::try_from(r.word()?).ok()?, r.word()?));
    }
    r.0.is_empty().then_some((
        HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_indexed_views,
        half_views,
        },
        made_by,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![1],
            root_indexed_views: vec![RootIndexedView {
                table_slot: 1,
                constant_slot: 2,
                constant_dword: 3,
                byte_offset: 32,
                shift: 5,
            }],
            half_views: vec![(0, 8), (4, 0)],
        }
    }

    #[test]
    fn a_paired_entry_lends_its_bytecode_only_with_the_same_bindings() {
        let paired = |s: &HostShader| [b"VSH1".as_slice(), &encode(s, "")[8..]].concat();
        let ours = shader();
        let mut theirs = shader();
        theirs.dxil = b"DXBC paired".to_vec();
        assert_eq!(paired_bytecode(&paired(&theirs), &ours), Some(b"DXBC paired".to_vec()));
        // Our own format is not a paired entry.
        assert_eq!(paired_bytecode(&encode(&theirs, ""), &ours), None);
        // Other bindings, or no DXBC container: ours stays.
        let mut bound = theirs.clone();
        bound.root_uavs = vec![2];
        assert_eq!(paired_bytecode(&paired(&bound), &ours), None);
        let mut other = theirs.clone();
        other.dxil = b"not a container".to_vec();
        assert_eq!(paired_bytecode(&paired(&other), &ours), None);
    }

    #[test]
    fn the_cache_key_names_the_ray_miss_switch() {
        assert!(super::watched_switches().contains(&"VICTORY_RT_MISS"));
        assert!(super::watched_switches().contains(&"VICTORY_RT_MISS_HIT"));
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate, false), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, translate, false), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], container, translate, false), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH2 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate, false), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate, false), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Failures are not stored.
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        assert!(cached(&cache, &build, b"other", fail, false).is_err());
    }

    #[test]
    fn a_failed_cache_write_keeps_the_translation() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"container whose entry cannot be written";
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate, false), Ok(shader()));
        let entries = walk(dir.path());
        assert_eq!(entries.len(), 1);
        std::fs::remove_file(&entries[0]).unwrap();
        std::fs::create_dir(&entries[0]).unwrap();

        assert_eq!(cached(&cache, &build, container, translate, false), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        assert!(entries[0].is_dir());
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        assert!(cached(&cache, &build, container, fail, false).unwrap_err().contains("unsupported"));
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn a_program_kept_out_of_the_deep_skips_form_has_a_key_of_its_own() {
        let build = [7u8; 32];
        assert_eq!(program_key(&build, false, false, false), build, "allowed: the build's key, as without lists");
        let off = program_key(&build, true, false, false);
        assert_ne!(off, build);
        assert_eq!(program_key(&build, true, false, false), off, "the same decision, the same key");
        assert_ne!(program_key(&[8; 32], true, false, false), off, "and still the build's");

        // So only the program whose decision changed is translated again.
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        let [a, b] = [b"program a".as_slice(), b"program b"];
        for c in [a, b] {
            cached(&cache, &program_key(&build, false, false, false), c, translate, false).unwrap();
        }
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        // a listed: a translated again, b from the cache
        cached(&cache, &program_key(&build, true, false, false), a, translate, false).unwrap();
        cached(&cache, &program_key(&build, false, false, false), b, translate, false).unwrap();
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        // the list emptied again: both from the cache
        for c in [a, b] {
            cached(&cache, &program_key(&build, false, false, false), c, translate, false).unwrap();
        }
        assert_eq!(runs.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn a_program_that_may_take_the_wave_wide_form_has_a_key_of_its_own() {
        let build = [7u8; 32];
        let wide = program_key(&build, false, false, true);
        assert_ne!(wide, build, "not served what was translated without the form");
        assert_ne!(wide, program_key(&build, false, true, false), "nor another decision's");
        assert_ne!(wide, program_key(&build, true, false, false));
        assert_ne!(program_key(&[8; 32], false, false, true), wide, "and still the build's");
    }

    #[test]
    fn a_program_that_takes_mask_copies_has_a_key_of_its_own() {
        let build = [7u8; 32];
        let copies = program_key(&build, false, true, false);
        assert_ne!(copies, build, "the build's key is for the programs that do not");
        assert_eq!(program_key(&build, false, true, false), copies, "the same decision, the same key");
        assert_ne!(program_key(&[8; 32], false, true, false), copies, "and still the build's");
        // Each combination a key of its own; the deep-skips one as before.
        let all: Vec<[u8; 32]> = (0..4u8).map(|n| program_key(&build, n & 1 != 0, n & 2 != 0, false)).collect();
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a, b);
            }
        }
        let mut h = blake3::Hasher::new();
        h.update(&build);
        h.update(b"VICTORY_WAVE_DEEP_SKIPS off for this program");
        assert_eq!(program_key(&build, true, false, false), *h.finalize().as_bytes());
    }

    #[test]
    fn opt_in_switches_mark_the_key_only_when_on() {
        assert!(opt_in_marks(false, false, false).is_empty(), "off: the key the cache already has");
        assert_eq!(opt_in_marks(true, false, false), b"VICTORY_PORT_CBV=1", "as before");
        assert_eq!(opt_in_marks(false, true, false), b"VICTORY_SPREAD=1", "as before");
        assert_eq!(opt_in_marks(true, true, false), b"VICTORY_PORT_CBV=1VICTORY_SPREAD=1", "as before, in that order");
        let refuse = opt_in_marks(false, false, true);
        assert!(!refuse.is_empty() && refuse != opt_in_marks(true, false, false));
        assert_ne!(opt_in_marks(true, false, true), refuse);
        assert_ne!(opt_in_marks(true, false, true), opt_in_marks(true, false, false));
        assert_ne!(opt_in_marks(false, true, true), opt_in_marks(false, true, false));
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader(), "made by");
        assert_eq!(decode(&bytes), Some((shader(), b"made by".to_vec())));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
        // Entries of the encoding before the translator's build was in them.
        let mut v1 = bytes.clone();
        v1[3] = b'1';
        assert_eq!(decode(&v1), None);
    }

    /// An entry an earlier build of the translator made, under the same
    /// version: served as it is; when refreshing translated again in its
    /// place, and a different result counted.
    #[test]
    fn entries_of_an_earlier_translator_are_served_or_refreshed() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [9u8; 32];
        let container = b"a container of an earlier build".as_slice();
        let runs = AtomicU64::new(0);
        let old = || {
            let mut s = shader();
            s.dxil = b"the old translation".to_vec();
            s
        };
        let first = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(old())
        };
        assert_eq!(cached_by(&cache, &build, "build a", container, first, false), Ok(old()));
        let now = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        let (stale, refreshed, changed) = stale_counts();
        // The next build, same version: the old translation, not translated.
        assert_eq!(cached_by(&cache, &build, "build b", container, now, false), Ok(old()));
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        // Refreshing: translated again and kept.
        assert_eq!(cached_by(&cache, &build, "build b", container, now, true), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        // Now the entry is this build's: served when refreshing too.
        assert_eq!(cached_by(&cache, &build, "build b", container, now, true), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        // (Other tests count too; these at least.)
        let (s, r, c) = stale_counts();
        assert!(s >= stale + 2 && r > refreshed && c > changed, "{s} {r} {c}");
    }

    #[test]
    fn a_stored_translation_is_found_without_converting_again() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        assert!(!holds(&cache, &build, container));
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert!(cached(&cache, &build, container, translate, false).is_ok());
        assert!(holds(&cache, &build, container));
        assert!(!holds(&cache, &build, b"other"));
        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }
}
