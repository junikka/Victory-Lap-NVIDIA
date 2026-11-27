//! A stand-in for the indirect light the title traces rays for.
//!
//! With the ray-tracing structure builder off (`crate::pipeline::builder_runs`)
//! the title's rays find nothing. Its program that resolves the diffuse light
//! of its screen probes (`CS_ScreenProbes_TemporalResolve_*`) then leaves the
//! sky's light outdoors, nothing indoors (walls are black where no lamp or
//! sun reaches) and, in blocks of the picture's tiles, what the probes'
//! memory held. Right after that program the light is raised to an estimate
//! and bounded by a multiple of it.
//!
//! That a stand-in can be put there, and in which of the program's images,
//! is beloooo's find (VIctory-Belobug-AMD, 06031237, 4f45ab88, cd9f78ce,
//! GPL-3.0-or-later), which puts a constant there, shaded in creases from the
//! scene's depth. The estimate here is worked out from the frame instead, so
//! it follows the hour, the room and its lamps by itself: the light the
//! visible surfaces send out (the title keeps it for its own screen traces:
//! `CS_CacheScreenSpaceRadiance`, the frame before) is averaged in 64x36
//! cells, and every point gets the mean of the cells weighted by the share of
//! its surroundings they cover, as a directional value (first-order harmonics
//! at 160x90 points, taken between them by depth).
//!
//! What it assumes: what is not on screen sends out what is (with a slow
//! mean of the last seconds, so that turning into a dark corner is not black
//! at once). No occlusion but the creases'.
//!
//! `VICTORY_AMBIENT`:
//! - unset or `0`/`off`: the title's own output.
//! - a number: that share of the estimate (1: all of it).
//! - `constant:<level>`: the constant, as beloooo's fork has it.
//!
//! `VICTORY_AMBIENT_SHADE=<strength>,<radius in metres>` sets the creases.
//!
//! `VICTORY_AMBIENT_REFLECT=<part>` holds the title's reflections (the
//! program that writes them last) to that part of the light around. Off
//! unless the part is set above 0, and it does not run the estimate above.
//!
//! `VICTORY_AMBIENT_SKY=on` (or `gain,least,power`) makes that light follow
//! the sky image the title reflects. Off unless asked. The estimate's own
//! shader is compiled without it.
//!
//! The air of a room is lit by the same probes: without rays they hold the
//! sky's light indoors too, a brown veil at sunset. `VICTORY_AMBIENT_AIR`
//! bounds what the air scatters by a multiple of the light around.
//!
//! `VICTORY_AMBIENT_DEPTH_MATCH=1` binds the scene depth only for the
//! resolve, and only when that descriptor is a shader view of the image's
//! size. The other roles leave the depth table unset. Unset, every role
//! binds the handle it found, as before.

use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
    DXGI_FORMAT_R32_UINT,
};
use windows::core::s;

use crate::host::{HostList, host};

/// Programs of the title the stand-in follows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Role {
    /// Resolves the diffuse light.
    Resolve,
    /// Keeps the light visible surfaces send out.
    Radiance,
    /// Adds the lamps' light to what the air scatters (the last program
    /// that writes that volume before it is summed up along the view).
    Air,
    /// Writes the reflections for the last time in a frame.
    Reflect,
    /// Draws the sky as the title reflects it.
    Sky,
}

/// What [`after`] recorded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Ran {
    /// Nothing was recorded.
    No,
    /// Recorded, still on the title's descriptor heap.
    Yes,
    /// Recorded on this pass's own descriptor heap. The list's heap is set
    /// again on the next bind.
    OwnHeap,
}

/// The program's part in this, by its name in the container.
pub(crate) fn role(container: &[u8]) -> Option<Role> {
    let has = |name: &[u8]| container.windows(name.len()).any(|w| w == name);
    if has(b"CS_ScreenProbes_TemporalResolve") {
        Some(Role::Resolve)
    } else if has(b"CS_CacheScreenSpaceRadiance") {
        Some(Role::Radiance)
    } else if has(b"CS_AccumulateLocalLightInscatteringVolume") {
        Some(Role::Air)
    } else if reflect_start() > 0.0 && has(REFLECTED) {
        Some(Role::Reflect)
    } else {
        None
    }
}

/// The program that writes the reflections last, and where it writes them
/// (R9G9B9E5 as 32-bit integers): root parameter and place in its table.
pub(crate) const REFLECTED: &[u8] = b"CS_UpdateHistory";
pub(crate) const REFLECTIONS: (u8, u32) = (1, 40);

/// The program that draws the sky as the title reflects it.
pub(crate) const SKY: &[u8] = b"ps_sky_paraboloid_reflection";

/// A shader's part in the stand-in. Pixel shaders are none unless the sky
/// switch is on, which is what a compute-only check did before that switch.
pub(crate) fn part(stage: &str, bytes: &[u8]) -> Option<Role> {
    if stage == "compute" {
        return role(bytes);
    }
    if stage == "pixel" && sky_on() && bytes.windows(SKY.len()).any(|w| w == SKY) {
        Some(Role::Sky)
    } else {
        None
    }
}

/// Whether the light follows the sky image (`VICTORY_AMBIENT_SKY`).
pub(crate) fn sky_on() -> bool {
    follow_sky().is_some()
}

/// The part a reflection may hold of the light around
/// (`VICTORY_AMBIENT_REFLECT=<part>`). 0 unless it is a finite number above 0.
fn reflect_part() -> f32 {
    if let Some(part) = live(&LIVE_REFLECT) {
        return part;
    }
    reflect_start()
}

/// The part the run started with: programs are marked to be held by it when
/// they are made ([`role`]), so a live change ([`set_live_reflect`]) acts
/// only on a run that started with a part.
pub(crate) fn reflect_start() -> f32 {
    #[cfg(test)]
    if let Some(part) = FORCE_REFLECT.with(|c| c.get()) {
        return part;
    }
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| parse_reflect(std::env::var("VICTORY_AMBIENT_REFLECT").ok().as_deref()))
}

fn parse_reflect(value: Option<&str>) -> f32 {
    value
        .and_then(|v| v.trim().parse().ok())
        .filter(|v: &f32| v.is_finite() && *v > 0.0)
        .unwrap_or(0.0)
}

/// Values the settings menu changed while the title runs
/// (`crate::settings_menu`, ticket 174): an f32's bits, `u32::MAX` while
/// none is set, which is always without the menu. Both are root constants
/// of every dispatch, so the next frame has them.
static LIVE_LEVEL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
static LIVE_REFLECT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

fn live(value: &std::sync::atomic::AtomicU32) -> Option<f32> {
    let bits = value.load(std::sync::atomic::Ordering::Relaxed);
    (bits != u32::MAX).then(|| f32::from_bits(bits))
}

/// The estimate's level from now on, if the run started with an estimate
/// ([`start_level`]); otherwise nothing changes.
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn set_live_level(level: f32) {
    if start_level().is_some() && level.is_finite() && level > 0.0 {
        LIVE_LEVEL.store(level.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }
}

/// The reflections' part from now on (0: not held), if the run started with
/// a part ([`reflect_start`]); otherwise nothing changes.
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn set_live_reflect(part: f32) {
    if reflect_start() > 0.0 && part.is_finite() {
        LIVE_REFLECT.store(part.max(0.0).to_bits(), std::sync::atomic::Ordering::Relaxed);
    }
}

/// The estimate's level the run started with; `None` when the stand-in is
/// off or a constant.
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn start_level() -> Option<f32> {
    match start_set() {
        Set::Estimate(level) => Some(level),
        _ => None,
    }
}

#[cfg(test)]
thread_local! {
    static FORCE_REFLECT: std::cell::Cell<Option<f32>> = const { std::cell::Cell::new(None) };
}

/// Where the resolve writes the diffuse light (R9G9B9E5 as 32-bit integers)
/// and reads the scene's depth, and where the other writes the light of
/// visible surfaces: root parameter and place in its table, in dwords.
pub(crate) const LIGHT: (u8, u32) = (1, 176);
pub(crate) const DEPTH: (u8, u32) = (0, 464);
pub(crate) const RADIANCE: (u8, u32) = (1, 16);
/// Where the light the air scatters is.
pub(crate) const AIR: (u8, u32) = (1, 32);

#[derive(Clone, Copy, PartialEq, Debug)]
enum Set {
    Off,
    Estimate(f32),
    Constant(f32),
}

fn parse(set: &str) -> Set {
    let set = set.trim();
    let number = |t: &str| t.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0);
    match set.strip_prefix("constant:") {
        Some(level) => number(level).map_or(Set::Off, Set::Constant),
        None => number(set).map_or(Set::Off, Set::Estimate),
    }
}

fn set() -> Set {
    if let Some(level) = live(&LIVE_LEVEL) {
        return Set::Estimate(level);
    }
    start_set()
}

/// What the run started with.
fn start_set() -> Set {
    static SET: OnceLock<Set> = OnceLock::new();
    *SET.get_or_init(|| parse(&std::env::var("VICTORY_AMBIENT").unwrap_or_default()))
}

/// Whether the stand-in is on: set, and the title's rays find nothing.
pub(crate) fn on() -> bool {
    set() != Set::Off && !crate::pipeline::builder_runs()
}

/// The most the air scatters, as a multiple of the light around
/// (`VICTORY_AMBIENT_AIR=<multiple>`; unset or 0: what the title has).
fn air() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| {
        let set = std::env::var("VICTORY_AMBIENT_AIR").unwrap_or_default();
        set.trim().parse().ok().filter(|v: &f32| v.is_finite() && *v > 0.0).unwrap_or(0.0)
    })
}

/// How the slow mean of the light around follows the frames
/// (`VICTORY_AMBIENT_SETTLE=<time constant in ms>[,<jump factor>[,<fresh
/// seconds>]]`; unset: 250,3; 0: 0.04 of the way a frame, as before, which is 25
/// frames: 0.8 s at the console's 30 frames a second, 2-4 s at 7-13). Set,
/// it is the same share of the way a second whatever the frame rate, and a
/// change by more than the jump factor (default 3; 0: none) is taken at
/// once. Fresh seconds (default 0: none): for that long after such a jump
/// the light the title's own resolve left is bounded by the estimate, not
/// four times it, rising back to four times by the end.
#[derive(Clone, Copy)]
struct Settle {
    ms: f32,
    jump: f32,
    fresh: f32,
}

fn settle() -> Option<Settle> {
    static SET: OnceLock<Option<Settle>> = OnceLock::new();
    *SET.get_or_init(|| {
        // Default on (250 ms, jump factor 3: reports/LIGHT-POPIN.md); `0`
        // gives the old 0.04 of the way a frame.
        let set = std::env::var("VICTORY_AMBIENT_SETTLE").unwrap_or_else(|_| "250,3".to_owned());
        let mut parts = set.split(',').map(|p| p.trim().parse::<f32>().ok());
        let ms = parts.next().flatten().filter(|v| v.is_finite() && *v > 0.0)?;
        let jump = parts.next().flatten().filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(3.0);
        let fresh = parts.next().flatten().filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(0.0);
        tracing::info!(
            ms,
            jump,
            fresh,
            "the stand-in's mean of the light around settles by the clock (VICTORY_AMBIENT_SETTLE)"
        );
        Some(Settle { ms, jump, fresh })
    })
}

/// The share of the way the slow mean goes at this frame's keeping
/// ([`Role::Radiance`]; the other passes do not read it).
fn follow(role: Role) -> f32 {
    match settle() {
        Some(settle) if role == Role::Radiance => 1.0 - (-frame_ms(role) / settle.ms).exp(),
        _ => 0.04,
    }
}

/// Milliseconds since the last keeping ([`Role::Radiance`], once a frame);
/// the first, and after a pause of over a second, as a frame of 30. Other
/// roles: the last answer.
fn frame_ms(role: Role) -> f32 {
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
    static LAST: AtomicU64 = AtomicU64::new(0);
    static DT: AtomicU32 = AtomicU32::new(0x4204_0000); // 33.0
    if role != Role::Radiance {
        return f32::from_bits(DT.load(Relaxed));
    }
    // (Asked twice in a keeping: the words and the share.)
    static ASKED: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms();
    if ASKED.swap(now, Relaxed) == now {
        return f32::from_bits(DT.load(Relaxed));
    }
    let last = LAST.swap(now, Relaxed);
    let dt = match now.saturating_sub(last) {
        0 => 1.0,
        dt if last == 0 || dt > 1000 => 33.0,
        dt => dt as f32,
    };
    DT.store(dt.to_bits(), Relaxed);
    dt
}

/// Strength and radius of the creases' shade.
fn shade() -> [f32; 2] {
    static SET: OnceLock<[f32; 2]> = OnceLock::new();
    *SET.get_or_init(|| {
        let mut shade = [1.0, 0.6];
        let set = std::env::var("VICTORY_AMBIENT_SHADE").unwrap_or_default();
        for (to, v) in shade.iter_mut().zip(set.split(',')) {
            if let Ok(v) = v.trim().parse() {
                *to = v;
            }
        }
        shade
    })
}

/// Gain, least level and power: the level is `gain * mean^power + least`.
/// `on` uses 0.36, 0.05 and 0.4. `off`, `0` and unset leave it off.
fn follow_sky() -> Option<[f32; 3]> {
    #[cfg(test)]
    if let Some(set) = FORCE_SKY.with(|c| c.get()) {
        return set;
    }
    static SET: OnceLock<Option<[f32; 3]>> = OnceLock::new();
    *SET.get_or_init(|| parse_sky(&std::env::var("VICTORY_AMBIENT_SKY").unwrap_or_default()))
}

fn parse_sky(set: &str) -> Option<[f32; 3]> {
    if matches!(set.trim(), "" | "off" | "0") {
        return None;
    }
    let mut follow = [0.36, 0.05, 0.4];
    for (to, v) in follow.iter_mut().zip(set.split(',')) {
        if let Ok(v) = v.trim().parse::<f32>()
            && v.is_finite()
        {
            *to = v;
        }
    }
    Some(follow)
}

#[cfg(test)]
thread_local! {
    static FORCE_SKY: std::cell::Cell<Option<Option<[f32; 3]>>> = const { std::cell::Cell::new(None) };
}

const SOURCE: &str = r#"
#ifdef FOLLOW_SKY
#define RS "RootConstants(num32BitConstants=20, b0), DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0)), DescriptorTable(SRV(t1)), UAV(u1), UAV(u2), UAV(u3)"
#else
#define RS "RootConstants(num32BitConstants=15, b0), DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0)), UAV(u1), UAV(u2), UAV(u3)"
#endif
#define CX 64
#define CY 36
#define CELLS (CX * CY)
#define PX 160
#define PY 90
cbuffer Set : register(b0) {
    float level; float constant; float strength; float radius;
    float near; float tan_y; uint width; uint height;
    float follow; float most; float air; uint probe;
    float jump; float fresh; float dt;
#ifdef FOLLOW_SKY
    // (Words 16 to 18: the one before fills the row the three above began.)
    float unused_sky; float gain; float least; float sky_power;
#endif
};
Texture2D<float> depth : register(t0);
// Cells: the light they send out (then, at CELLS, the slow mean), from
// CELLS + 1 on where they are.
RWStructuredBuffer<float4> cells : register(u1);
// Three a point: red, green and blue, each a mean and a direction.
RWStructuredBuffer<float4> points : register(u2);
#ifdef FOLLOW_SKY
Texture2DArray<float4> sky : register(t1);
#endif
// For the light probe (`crate::lightprobe`): sampled pixels, and the sums of
// what the title's resolve left, what was found and what was written.
RWByteAddressBuffer stats : register(u3);
static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);

// The point seen at a texel, in metres from the eye (x right, y up, z away).
float3 seen(int2 p, float2 size) {
    float d = depth.Load(int3(clamp(p, int2(0, 0), int2(size) - 1), 0));
    float z = min(near / max(d, 1e-7), 1000.0);
    float2 ndc = (float2(p) + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}

#ifdef KEEP
RWTexture2D<float3> radiance : register(u0);
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void keep(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    radiance.GetDimensions(w, h);
    float3 sum = 0;
    for (uint y = 0; y < 5; y++) for (uint x = 0; x < 5; x++) {
        uint2 p = uint2((float2(at.xy) + (float2(x, y) + 0.5) / 5.0) * float2(w, h) / float2(CX, CY));
        float3 c = max(radiance[p], 0.0);
        c = any(isnan(c)) || any(isinf(c)) ? 0.0 : c;
        // Lamps and glints count, but not for everything.
        sum += c / max(1.0, dot(c, LUMA) / 20.0);
    }
    cells[at.y * CX + at.x] = float4(sum / 25.0, 1.0);
}
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void mean() {
    float3 sum = 0;
    for (uint i = 0; i < CELLS; i++) sum += cells[i].rgb;
    float4 was = cells[CELLS];
    float3 now = sum / CELLS;
    // (`jump` > 0: a change of more than that factor either way is
    // followed at once, as a new room is.)
    float a = max(dot(now, LUMA), 1e-5), b = max(dot(was.rgb, LUMA), 1e-5);
    bool jumped = jump > 0.0 && max(a / b, b / a) > jump;
    // Its fourth part: seconds since the light around last jumped (from
    // 0.001; 0: nothing kept yet).
    float age = was.a > 0.0 && !jumped ? was.a + dt : 0.001;
    cells[CELLS] = float4(was.a > 0.0 ? lerp(was.rgb, now, jumped ? 1.0 : follow) : now, age);
}
#endif

#ifdef PLACE
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void place(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen(int2((float2(at.xy) + 0.5) * size / float2(CX, CY)), size);
    cells[CELLS + 1 + at.y * CX + at.x] = float4(p, p.z * p.z);
}
#endif

#ifdef GATHER
[RootSignature(RS)]
[numthreads(8, 2, 1)]
void gather(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen(int2((float2(at.xy) + 0.5) * size / float2(PX, PY)), size);
    float r2 = 0.25 + 0.0025 * p.z * p.z;
    float whole = 0;
    float3 toward = 0, light = 0, red = 0, green = 0, blue = 0;
    for (uint i = 0; i < CELLS; i++) {
        float4 c = cells[CELLS + 1 + i];
        float3 L = cells[i].rgb;
        float3 d = c.xyz - p;
        float dd = dot(d, d);
        float w = c.w / (dd + r2);
        float3 dir = d / sqrt(dd + 0.25 * r2);
        whole += w;
        toward += w * dir;
        light += w * L;
        red += w * L.r * dir;
        green += w * L.g * dir;
        blue += w * L.b * dir;
    }
    whole = max(whole, 1e-6);
    light /= whole;
    toward /= whole;
    uint at0 = (at.y * PX + at.x) * 3;
    points[at0] = float4(red / whole - light.r * toward, light.r);
    points[at0 + 1] = float4(green / whole - light.g * toward, light.g);
    points[at0 + 2] = float4(blue / whole - light.b * toward, light.b);
}
#endif

#ifdef AIR
RWTexture3D<float3> scattered : register(u0);
// No more than a multiple of the light around: of the room as the last
// seconds showed it, and of the surface seen in this direction (towards a
// window the air outside may be as bright as the sky is).
[RootSignature(RS)]
[numthreads(8, 8, 2)]
void clear_air(uint3 at : SV_DispatchThreadID) {
    uint w, h, d;
    scattered.GetDimensions(w, h, d);
    if (at.x >= w || at.y >= h || at.z >= d) return;
    uint2 q = min(uint2((float2(at.xy) + 0.5) / float2(w, h) * float2(PX, PY)), uint2(PX - 1, PY - 1));
    uint at0 = (q.y * PX + q.x) * 3;
    float3 seen_there = float3(points[at0].w, points[at0 + 1].w, points[at0 + 2].w);
    float far = float(at.z) / float(d);
    float cap = air * lerp(dot(cells[CELLS].rgb, LUMA), dot(seen_there, LUMA), far);
    float3 c = scattered[at];
    float has = dot(c, LUMA);
    if (has > cap) scattered[at] = c * (cap / has);
}
#endif

#ifdef RAISE
RWTexture2D<uint> light : register(u0);
float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2(float(v >> 27) - 24.0);
}
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float top = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(top, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (top * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
#ifdef FOLLOW_SKY
// The sky's colour, half washed toward white. A dark sky keeps a night blue.
float3 from_sky() {
    uint sw, sh, slices;
    sky.GetDimensions(sw, sh, slices);
    float3 sum = 0.0;
    const int N = 8;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            float3 there = sky.Load(int4((x * 2 + 1) * sw / (N * 2), (y * 2 + 1) * sh / (N * 2), 0, 0)).rgb;
            sum += clamp(there, 0.0, 8.0);
        }
    float3 mean = sum / float(N * N);
    float m = dot(mean, LUMA);
    const float3 NIGHT = float3(0.6, 0.75, 1.0);
    float3 hue = lerp(NIGHT, lerp(float3(1.0, 1.0, 1.0), mean / max(m, 1e-6), 0.5), saturate(m / 0.05));
    return hue * (gain * pow(max(m, 0.0), sky_power) + least);
}
#endif
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void raise(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    if (at.x >= width || at.y >= height) return;
    int2 p = int2(at.xy);
    float3 c = seen(p, size);
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx) + float3(0, 0, -1e-9));
    if (dot(n, c) > 0.0) n = -n;
    // How open it is (as in beloooo's cd9f78ce).
    float open = 1.0;
    float px = min(radius * size.y / (2.0 * tan_y * c.z), 128.0);
    float noise = frac(52.9829189 * frac(dot(float2(p), float2(0.06711056, 0.00583715))));
    if (strength > 0.0 && px >= 1.5 && c.z < 400.0) {
        float sum = 0.0;
        const int N = 12;
        for (int i = 0; i < N; i++) {
            float a = i * 2.399963 + noise * 6.2831853;
            float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
            float3 v = seen(p + int2(round(o)), size) - c;
            float vv = dot(v, v);
            float over = dot(v, n) - 0.004 * c.z - 0.005;
            sum += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
        }
        open = saturate(1.0 - strength * 2.0 * sum / N);
    }
#ifdef FOLLOW_SKY
    light[p] = pack(from_sky() * open);
    return;
#endif
    float3 found;
    if (constant > 0.0) {
        found = float3(1.0, 0.98, 0.94) * constant * open;
        light[p] = pack(found);
        return;
    }
    // Between the four points around, those at its own depth first; which
    // four is dithered a little.
    float2 g = (float2(p) + 0.5) / size * float2(PX, PY) - 0.5 + (noise - 0.5) * 0.5;
    int2 g0 = int2(floor(g));
    float2 f = g - float2(g0);
    float4 sr = 0, sg = 0, sb = 0;
    float sum = 0;
    for (int k = 0; k < 4; k++) {
        int2 q = clamp(g0 + int2(k & 1, k >> 1), int2(0, 0), int2(PX, PY) - 1);
        float z = seen(int2((float2(q) + 0.5) * size / float2(PX, PY)), size).z;
        float2 b = float2(k & 1 ? f.x : 1 - f.x, k >> 1 ? f.y : 1 - f.y);
        float w = (b.x * b.y + 1e-3) / (0.02 + abs(z - c.z) / c.z);
        uint at0 = (q.y * PX + q.x) * 3;
        sr += w * points[at0];
        sg += w * points[at0 + 1];
        sb += w * points[at0 + 2];
        sum += w;
    }
    float3 all = float3(sr.w, sg.w, sb.w) / sum;
    found = all + 1.2 * float3(dot(sr.xyz, n), dot(sg.xyz, n), dot(sb.xyz, n)) / sum;
    found = clamp(found, 0.2 * all, 2.5 * all);
    found = lerp(found, cells[CELLS].rgb, 0.2) * level * open;
    float3 was = unpack(light[p]);
    float left = dot(was, LUMA);
    // What the program left counts as far as it can be true: the sky's light
    // outdoors, not the blocks of other memory.
    // For `fresh` seconds after the light around jumped (a new room) the
    // title's leftover counts for no more than the estimate, then up to
    // `most` times it again: its history holds the place before.
    float bound = fresh > 0.0 ? lerp(1.0, most, saturate(cells[CELLS].a / fresh)) : most;
    float top = max(dot(found, LUMA) * bound, 1e-6);
    float has = dot(was, LUMA);
    was *= min(1.0, top / max(has, 1e-6));
    float3 written = max(was, found);
    light[p] = pack(written);
    if (probe != 0 && (at.x & 15) == 0 && (at.y & 15) == 0) {
        uint ignored;
        stats.InterlockedAdd(0, 1, ignored);
        stats.InterlockedAdd(4, uint(min(left, 1000.0) * 256.0), ignored);
        stats.InterlockedAdd(8, uint(min(dot(found, LUMA), 1000.0) * 256.0), ignored);
        stats.InterlockedAdd(12, uint(min(dot(written, LUMA), 1000.0) * 256.0), ignored);
    }
}
#endif
"#;

/// Bytes of the cells' buffer and of the points'.
const CELLS: u64 = 64 * 36;
const CELL_BYTES: u64 = (2 * CELLS + 1) * 16;
const POINT_BYTES: u64 = 160 * 90 * 48;
/// Where the slow mean is among the cells (float4s).
pub(crate) const CELLS_AT: u32 = CELLS as u32;

struct Pass {
    root: ID3D12RootSignature,
    keep: ID3D12PipelineState,
    mean: ID3D12PipelineState,
    place: ID3D12PipelineState,
    gather: ID3D12PipelineState,
    raise: ID3D12PipelineState,
    air: ID3D12PipelineState,
    cells: ID3D12Resource,
    points: ID3D12Resource,
    stats: ID3D12Resource,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn make(device: &ID3D12Device, buffer: impl Fn(u64) -> Option<ID3D12Resource>) -> windows::core::Result<Pass> {
    let mut root: Option<ID3D12RootSignature> = None;
    let mut pipeline = |part: &str, entry| -> windows::core::Result<ID3D12PipelineState> {
        let code = crate::host::compile(&format!("#define {part}\n{SOURCE}"), entry, s!("cs_5_1"))?;
        // SAFETY: the root signature is inside the shader; plain creations.
        unsafe {
            if root.is_none() {
                root = Some(device.CreateRootSignature(0, &code)?);
            }
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                ..Default::default()
            };
            device.CreateComputePipelineState(&desc)
        }
    };
    let keep = pipeline("KEEP", s!("keep"))?;
    let mean = pipeline("KEEP", s!("mean"))?;
    let place = pipeline("PLACE", s!("place"))?;
    let gather = pipeline("GATHER", s!("gather"))?;
    let raise = pipeline("RAISE", s!("raise"))?;
    let air = pipeline("AIR", s!("clear_air"))?;
    let made = |bytes| buffer(bytes).ok_or_else(windows::core::Error::empty);
    Ok(Pass {
        root: root.ok_or_else(windows::core::Error::empty)?,
        keep,
        mean,
        place,
        gather,
        raise,
        air,
        cells: made(CELL_BYTES)?,
        points: made(POINT_BYTES)?,
        stats: made(256)?,
    })
}

fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        make(&host.device, |bytes| host.uav_buffer(bytes))
            .inspect_err(|e| tracing::warn!(error = %e, "no stand-in for the indirect light"))
            .ok()
    })
    .as_ref()
}

/// The probe's sums and the cells (with the slow mean), once the passes are
/// made and the stand-in is on.
pub(crate) fn probe_buffers() -> Option<(ID3D12Resource, ID3D12Resource)> {
    if !on() {
        return None;
    }
    pass().map(|p| (p.stats.clone(), p.cells.clone()))
}

fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, ..Default::default() };
    // SAFETY: an open list; the barrier names no resource.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// Whether the light of a frame has been kept: the estimate has something
/// to begin with.
static KEPT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// What the stand-in does with the depth descriptor table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DepthBind {
    /// Set root parameter 1 to the depth view the caller found.
    Set,
    /// Leave root parameter 1 unset. This role does not sample depth.
    Leave,
    /// Do not run the stand-in.
    Skip,
}

/// Whether `value` turns on [`depth_match`]. Only `1` does.
pub(crate) fn depth_match_set(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| v == "1")
}

/// Bind the depth only when it is a matching shader view
/// (`VICTORY_AMBIENT_DEPTH_MATCH=1`). Unset: off, and the stand-in binds
/// whatever handle it found, as before.
pub(crate) fn depth_match() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| depth_match_set(std::env::var_os("VICTORY_AMBIENT_DEPTH_MATCH").as_deref()))
}

/// `match_depth` is [`depth_match`]. Off: bind when a handle was found; the
/// caller still checks that handle's size against the image. On: only
/// [`Role::Resolve`] binds a depth view, and only when one was found at the
/// image's size. The other roles leave the table unset.
pub(crate) fn depth_bind(role: Role, found: bool, same_size: bool, match_depth: bool) -> DepthBind {
    if !match_depth {
        return if found { DepthBind::Set } else { DepthBind::Skip };
    }
    match role {
        Role::Resolve if found && same_size => DepthBind::Set,
        Role::Resolve => DepthBind::Skip,
        Role::Radiance | Role::Air | Role::Reflect => DepthBind::Leave,
        Role::Sky => DepthBind::Skip,
    }
}

/// What the passes read and write, as views in the heap the list has set.
pub(crate) struct Views {
    /// The image the title's program just wrote.
    pub(crate) image: D3D12_GPU_DESCRIPTOR_HANDLE,
    /// The scene's depth, for [`Role::Resolve`]. None leaves root parameter
    /// 1 unset (`VICTORY_AMBIENT_DEPTH_MATCH` for a role that does not
    /// sample depth).
    pub(crate) depth: Option<D3D12_GPU_DESCRIPTOR_HANDLE>,
    /// The size of the image, and how many slices it has.
    pub(crate) size: [u32; 2],
    pub(crate) slices: u32,
    /// The title resources behind those views. 0 when the caller has none.
    /// A pass that makes its own views uses them; the estimate does not.
    pub(crate) image_resource: usize,
    pub(crate) depth_resource: usize,
}

/// After the title's program of that role, in `list` with the program's view
/// heap set. Replaces the list's compute root signature and pipeline.
/// [`Ran::No`] if it recorded nothing.
pub(crate) fn after(list: &HostList, role: Role, views: &Views) -> Ran {
    use std::sync::atomic::Ordering;
    if role == Role::Sky {
        return Ran::No;
    }
    if role == Role::Resolve
        && let Some(ran) = raise_with_sky(list, views)
    {
        return ran;
    }
    if role == Role::Reflect {
        if let Some(ran) = hold_with_sky(list, views) {
            return ran;
        }
        return hold_reflections(list, views);
    }
    let Some(pass) = pass() else {
        return Ran::No;
    };
    let (level, constant) = match set() {
        Set::Off => return Ran::No,
        Set::Estimate(level) => (level, 0.0f32),
        Set::Constant(level) => (0.0, level),
    };
    if role != Role::Radiance && constant == 0.0 && !KEPT.load(Ordering::Relaxed) {
        return Ran::No;
    }
    if role == Role::Air && (constant > 0.0 || air() == 0.0) {
        return Ran::No;
    }
    static SAID: Mutex<u32> = Mutex::new(0);
    {
        let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
        if *said < 4 {
            *said += 1;
            tracing::info!(?role, size = ?views.size, level, constant, "a stand-in for the indirect light");
        }
    }
    list.keep(&pass.cells);
    list.keep(&pass.points);
    list.keep(&pass.stats);
    let [strength, radius] = shade();
    let floats = [level, constant, strength, radius, 0.1, 0.45];
    let mut words = [0u32; 15];
    for (to, v) in words.iter_mut().zip(floats) {
        *to = v.to_bits();
    }
    words[6] = views.size[0];
    words[7] = views.size[1];
    words[8] = follow(role).to_bits();
    words[9] = 4.0f32.to_bits();
    words[10] = air().to_bits();
    words[11] = u32::from(crate::lightprobe::on());
    words[12] = settle().map_or(0.0, |s| s.jump).to_bits();
    words[13] = settle().map_or(0.0, |s| s.fresh).to_bits();
    words[14] = (frame_ms(role) / 1000.0).to_bits();
    let l = &list.list;
    // SAFETY: an open list with the heap of the views set; the title's
    // program has just used the views this pass binds, so their textures
    // are in the states these views need; buffers the list keeps.
    unsafe {
        uav_barrier(l);
        l.SetComputeRootSignature(&pass.root);
        l.SetComputeRoot32BitConstants(0, 15, words.as_ptr().cast(), 0);
        // Unset when this role does not sample depth
        // (`VICTORY_AMBIENT_DEPTH_MATCH`). Otherwise set, as before.
        if let Some(depth) = views.depth {
            l.SetComputeRootDescriptorTable(1, depth);
        }
        l.SetComputeRootDescriptorTable(2, views.image);
        l.SetComputeRootUnorderedAccessView(3, pass.cells.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(4, pass.points.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(5, pass.stats.GetGPUVirtualAddress());
        let run = |pso: &ID3D12PipelineState, x, y, z| {
            l.SetPipelineState(pso);
            l.Dispatch(x, y, z);
            uav_barrier(l);
        };
        match role {
            Role::Radiance => {
                run(&pass.keep, 8, 9, 1);
                run(&pass.mean, 1, 1, 1);
                KEPT.store(true, Ordering::Relaxed);
            }
            Role::Air => run(&pass.air, views.size[0].div_ceil(8), views.size[1].div_ceil(8), views.slices.div_ceil(2)),
            Role::Resolve => {
                if constant == 0.0 {
                    run(&pass.place, 8, 9, 1);
                    run(&pass.gather, 20, 45, 1);
                }
                run(&pass.raise, views.size[0].div_ceil(8), views.size[1].div_ceil(8), 1);
            }
            // Returned above. Present so this match stays exhaustive.
            Role::Reflect | Role::Sky => {}
        }
    }
    Ran::Yes
}

/// Holds a reflection to a part of the light around. A separate shader from
/// [`SOURCE`]: the estimate's pipelines stay what they were when this is off.
const HOLD: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), DescriptorTable(UAV(u0)), UAV(u1)"
#define CELLS (64 * 36)
cbuffer Set : register(b0) {
    float held; float have_mean; float level; float unused;
    float cap_r; float cap_g; float cap_b; float unused2;
};
RWTexture2D<uint> image : register(u0);
RWStructuredBuffer<float4> cells : register(u1);
static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);

float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2(float(v >> 27) - 24.0);
}
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float top = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(top, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (top * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void hold(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    image.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    // The slow mean once a frame has kept it; otherwise the fallback colour.
    float3 cap = float3(cap_r, cap_g, cap_b);
    if (have_mean > 0.0) cap = max(cells[CELLS].rgb, 0.0) * level;
    float3 seen = unpack(image[at.xy]);
    float most = dot(cap, LUMA) * held;
    float has = dot(seen, LUMA);
    if (has > most) image[at.xy] = pack(seen * (most / max(has, 1e-6)));
}
"#;

struct HoldPass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for HoldPass {}
// SAFETY: as above.
unsafe impl Sync for HoldPass {}

fn make_hold(device: &ID3D12Device) -> windows::core::Result<HoldPass> {
    let code = crate::host::compile(HOLD, s!("hold"), s!("cs_5_1"))?;
    // SAFETY: the root signature is inside the shader; plain creations.
    unsafe {
        let root = device.CreateRootSignature(0, &code)?;
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
            ..Default::default()
        };
        Ok(HoldPass { root, pso: device.CreateComputePipelineState(&desc)? })
    }
}

fn hold_pass() -> Option<&'static HoldPass> {
    static PASS: OnceLock<Option<HoldPass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        make_hold(&host.device)
            .inspect_err(|e| tracing::warn!(error = %e, "no hold for the reflections"))
            .ok()
    })
    .as_ref()
}

/// Level, whether the slow mean is live, and the colour used until it is.
/// With the estimate off, the level is 0.6, which is what a constant floor
/// used where this was written, so a reflection part still has a light to
/// hold to.
fn cap_for(set: Set, kept: bool) -> (f32, bool, [f32; 3]) {
    let warm = [1.0f32, 0.98, 0.94];
    let paint = |level: f32| warm.map(|c| c * level);
    match set {
        Set::Constant(level) => (level, false, paint(level)),
        Set::Estimate(level) => (level, kept, paint(level)),
        Set::Off => (0.6, false, paint(0.6)),
    }
}

fn reflection_cap() -> (f32, bool, [f32; 3]) {
    cap_for(set(), KEPT.load(std::sync::atomic::Ordering::Relaxed))
}

fn hold_reflections(list: &HostList, views: &Views) -> Ran {
    let part = reflect_part();
    if part <= 0.0 {
        return Ran::No;
    }
    let (Some(pass), Some(hold)) = (pass(), hold_pass()) else {
        return Ran::No;
    };
    let (level, have_mean, cap) = reflection_cap();
    let mut words = [0u32; 8];
    words[0] = part.to_bits();
    words[1] = (if have_mean { 1.0f32 } else { 0.0 }).to_bits();
    words[2] = level.to_bits();
    words[4] = cap[0].to_bits();
    words[5] = cap[1].to_bits();
    words[6] = cap[2].to_bits();
    list.keep(&pass.cells);
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        tracing::info!(part, level, have_mean, "reflections held to the light around")
    });
    let l = &list.list;
    // SAFETY: an open list; the title's heap is still set, and the reflection
    // view is the one its program just wrote. The cells buffer is kept by the list.
    unsafe {
        uav_barrier(l);
        l.SetComputeRootSignature(&hold.root);
        l.SetComputeRoot32BitConstants(0, 8, words.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(1, views.image);
        l.SetComputeRootUnorderedAccessView(2, pass.cells.GetGPUVirtualAddress());
        l.SetPipelineState(&hold.pso);
        l.Dispatch(views.size[0].div_ceil(8), views.size[1].div_ceil(8), 1);
        uav_barrier(l);
    }
    Ran::Yes
}

/// The sky image last drawn, if the switch is on and it is the dome
/// (R16G16B16A16 float, not the other sky image).
static SKY_IMAGE: Mutex<Option<std::sync::Arc<crate::resource::HostTexture>>> = Mutex::new(None);

pub(crate) fn sky_drawn(image: crate::object::This) {
    if !sky_on() {
        return;
    }
    let Some(copy) = crate::resource::host_copy(image) else {
        return;
    };
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    if d.Format != DXGI_FORMAT_R16G16B16A16_FLOAT || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
        return;
    }
    let mut held = SKY_IMAGE.lock().unwrap_or_else(|e| e.into_inner());
    if !held.as_ref().is_some_and(|h| std::sync::Arc::ptr_eq(h, &copy)) {
        *held = Some(copy);
    }
}

fn sky_image() -> Option<std::sync::Arc<crate::resource::HostTexture>> {
    SKY_IMAGE.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

const SKY_SLOTS: u32 = 1024;

/// Holds reflections to the sky colour. Own heap: a list has one descriptor
/// heap, and the sky image is not in the title's.
const HOLD_SKY: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), DescriptorTable(UAV(u0)), DescriptorTable(SRV(t0))"
cbuffer Set : register(b0) { float held; float gain; float least; float sky_power; };
RWTexture2D<uint> image : register(u0);
Texture2DArray<float4> sky : register(t0);
static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);
float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2(float(v >> 27) - 24.0);
}
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float top = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(top, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (top * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
float3 from_sky() {
    uint sw, sh, slices;
    sky.GetDimensions(sw, sh, slices);
    float3 sum = 0.0;
    const int N = 8;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            float3 there = sky.Load(int4((x * 2 + 1) * sw / (N * 2), (y * 2 + 1) * sh / (N * 2), 0, 0)).rgb;
            sum += clamp(there, 0.0, 8.0);
        }
    float3 mean = sum / float(N * N);
    float m = dot(mean, LUMA);
    const float3 NIGHT = float3(0.6, 0.75, 1.0);
    float3 hue = lerp(NIGHT, lerp(float3(1.0, 1.0, 1.0), mean / max(m, 1e-6), 0.5), saturate(m / 0.05));
    return hue * (gain * pow(max(m, 0.0), sky_power) + least);
}
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void hold(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    image.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    float3 cap = from_sky();
    float3 seen = unpack(image[at.xy]);
    float most = dot(cap, LUMA) * held;
    float has = dot(seen, LUMA);
    if (has > most) image[at.xy] = pack(seen * (most / max(has, 1e-6)));
}
"#;

struct SkyPass {
    raise_root: ID3D12RootSignature,
    raise: ID3D12PipelineState,
    hold_root: ID3D12RootSignature,
    hold: ID3D12PipelineState,
    views: ID3D12DescriptorHeap,
    step: u32,
    next: std::sync::atomic::AtomicU32,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for SkyPass {}
// SAFETY: as above.
unsafe impl Sync for SkyPass {}

fn make_sky(device: &ID3D12Device) -> windows::core::Result<SkyPass> {
    let compile = |source: &str, entry| crate::host::compile(source, entry, s!("cs_5_1"));
    let raise_code = compile(&format!("#define RAISE\n#define FOLLOW_SKY\n{SOURCE}"), s!("raise"))?;
    let hold_code = compile(HOLD_SKY, s!("hold"))?;
    let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    // SAFETY: the root signatures are inside the shaders; plain creations.
    unsafe {
        let raise_root = device.CreateRootSignature(0, &raise_code)?;
        let hold_root = device.CreateRootSignature(0, &hold_code)?;
        let pso = |root: &ID3D12RootSignature, code: &[u8]| {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            };
            device.CreateComputePipelineState(&desc)
        };
        let views = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: kind,
            NumDescriptors: SKY_SLOTS * 3,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })?;
        Ok(SkyPass {
            raise: pso(&raise_root, &raise_code)?,
            raise_root,
            hold: pso(&hold_root, &hold_code)?,
            hold_root,
            step: device.GetDescriptorHandleIncrementSize(kind),
            views,
            next: std::sync::atomic::AtomicU32::new(0),
        })
    }
}

fn sky_pass() -> Option<&'static SkyPass> {
    static PASS: OnceLock<Option<SkyPass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        make_sky(&host.device)
            .inspect_err(|e| tracing::warn!(error = %e, "no sky for the stand-in light"))
            .ok()
    })
    .as_ref()
}

/// Brings the sky image to a shader-read state. The first use in a list is
/// the list's prelude; a later one records the barrier here.
fn read_sky(list: &HostList, sky: &std::sync::Arc<crate::resource::HostTexture>) {
    const READ: u32 = 0x40 | 0x80;
    if let Some(before) = list.transition_copy(sky, READ) {
        let barrier = crate::host::transition(
            &sky.texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(READ as i32),
        );
        // SAFETY: an open list and a live texture the list keeps.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
    list.keep(&sky.texture);
}

fn sky_handles(pass: &SkyPass) -> (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE) {
    let slot = pass.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % SKY_SLOTS * 3;
    // SAFETY: the heap lives as long as the pass, which is process-long.
    let (cpu, gpu) = unsafe {
        (
            pass.views.GetCPUDescriptorHandleForHeapStart(),
            pass.views.GetGPUDescriptorHandleForHeapStart(),
        )
    };
    let step = pass.step;
    (
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + (slot * step) as usize },
        D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu.ptr + u64::from(slot) * u64::from(step) },
    )
}

fn shift_handle(
    cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    step: u32,
    n: u32,
) -> (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE) {
    (
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + (n * step) as usize },
        D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu.ptr + u64::from(n) * u64::from(step) },
    )
}

/// The resolve, with the sky's light in place of the estimate. None if the
/// switch is off, no sky image is held, or the depth cannot be read that
/// way: the caller then runs the estimate, which does nothing while that is off.
fn raise_with_sky(list: &HostList, views: &Views) -> Option<Ran> {
    let [gain, least, power] = follow_sky()?;
    let sky = sky_image()?;
    let host = host()?;
    let light = crate::resource::host_resource(views.image_resource as crate::object::This)?;
    let depth = crate::resource::host_resource(views.depth_resource as crate::object::This)?;
    // SAFETY: live resources.
    let (of_light, of_depth) = unsafe { (light.GetDesc(), depth.GetDesc()) };
    if of_light.Format != DXGI_FORMAT_R32_UINT {
        return None;
    }
    let same = (of_depth.Width, of_depth.Height) == (of_light.Width, of_light.Height);
    if of_depth.Format != DXGI_FORMAT_R32G8X24_TYPELESS || !same {
        return None;
    }
    let (pass, ambient) = (sky_pass()?, pass()?);
    let sky_format = unsafe { sky.texture.GetDesc() }.Format;
    read_sky(list, &sky);
    list.keep(light);
    list.keep(depth);
    list.keep(&ambient.cells);
    list.keep(&ambient.points);
    list.keep(&ambient.stats);
    list.keep(&pass.views);
    let (cpu, gpu) = sky_handles(pass);
    let step = pass.step;
    let (cpu_light, gpu_light) = shift_handle(cpu, gpu, step, 1);
    let (cpu_sky, gpu_sky) = shift_handle(cpu, gpu, step, 2);
    let depth_view = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
        },
    };
    let light_view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    let sky_view = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: sky_format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, ArraySize: 1, ..Default::default() },
        },
    };
    let [strength, radius] = shade();
    let mut words = [0u32; 20];
    for (to, v) in words.iter_mut().zip([0.0, 0.0, strength, radius, 0.1, 0.45]) {
        *to = v.to_bits();
    }
    words[6] = of_light.Width as u32;
    words[7] = of_light.Height;
    words[8] = 0.04f32.to_bits();
    words[9] = 4.0f32.to_bits();
    words[10] = air().to_bits();
    // (11 to 14, the probe and the settling, stay 0: this shader writes the
    // sky's light and returns before it reads them.)
    words[16] = gain.to_bits();
    words[17] = least.to_bits();
    words[18] = power.to_bits();
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| tracing::info!(gain, least, power, "the light follows the sky image"));
    let l = &list.list;
    // SAFETY: an open list; slots of our heap no list in flight reads; live
    // textures the list keeps. Our heap replaces the title's for this dispatch.
    unsafe {
        host.device.CreateShaderResourceView(depth, Some(&depth_view), cpu);
        host.device.CreateUnorderedAccessView(light, None::<&ID3D12Resource>, Some(&light_view), cpu_light);
        host.device.CreateShaderResourceView(&sky.texture, Some(&sky_view), cpu_sky);
        uav_barrier(l);
        l.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        l.SetComputeRootSignature(&pass.raise_root);
        l.SetPipelineState(&pass.raise);
        l.SetComputeRoot32BitConstants(0, 20, words.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(1, gpu);
        l.SetComputeRootDescriptorTable(2, gpu_light);
        l.SetComputeRootDescriptorTable(3, gpu_sky);
        l.SetComputeRootUnorderedAccessView(4, ambient.cells.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(5, ambient.points.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(6, ambient.stats.GetGPUVirtualAddress());
        l.Dispatch((of_light.Width as u32).div_ceil(8), of_light.Height.div_ceil(8), 1);
        uav_barrier(l);
    }
    Some(Ran::OwnHeap)
}

/// Reflections held to the sky colour. None if the switch is off or no sky
/// image is held yet: the caller then uses the reflection hold it already had.
fn hold_with_sky(list: &HostList, views: &Views) -> Option<Ran> {
    let part = reflect_part();
    let [gain, least, power] = follow_sky()?;
    if part <= 0.0 {
        return None;
    }
    let sky = sky_image()?;
    let host = host()?;
    let image = crate::resource::host_resource(views.image_resource as crate::object::This)?;
    let of = unsafe { image.GetDesc() };
    if of.Format != DXGI_FORMAT_R32_UINT {
        return None;
    }
    let pass = sky_pass()?;
    let sky_format = unsafe { sky.texture.GetDesc() }.Format;
    read_sky(list, &sky);
    list.keep(image);
    list.keep(&pass.views);
    let (cpu, gpu) = sky_handles(pass);
    let (cpu_sky, gpu_sky) = shift_handle(cpu, gpu, pass.step, 1);
    let light_view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    let sky_view = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: sky_format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, ArraySize: 1, ..Default::default() },
        },
    };
    let mut words = [0u32; 4];
    words[0] = part.to_bits();
    words[1] = gain.to_bits();
    words[2] = least.to_bits();
    words[3] = power.to_bits();
    let l = &list.list;
    // SAFETY: as in [`raise_with_sky`]. The title's heap is replaced here.
    unsafe {
        host.device.CreateUnorderedAccessView(image, None::<&ID3D12Resource>, Some(&light_view), cpu);
        host.device.CreateShaderResourceView(&sky.texture, Some(&sky_view), cpu_sky);
        uav_barrier(l);
        l.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        l.SetComputeRootSignature(&pass.hold_root);
        l.SetPipelineState(&pass.hold);
        l.SetComputeRoot32BitConstants(0, 4, words.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(1, gpu);
        l.SetComputeRootDescriptorTable(2, gpu_sky);
        l.Dispatch((of.Width as u32).div_ceil(8), of.Height.div_ceil(8), 1);
        uav_barrier(l);
    }
    Some(Ran::OwnHeap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings() {
        assert_eq!(parse(""), Set::Off);
        assert_eq!(parse("off"), Set::Off);
        assert_eq!(parse("0"), Set::Off);
        assert_eq!(parse("0.8"), Set::Estimate(0.8));
        assert_eq!(parse("constant:0.3"), Set::Constant(0.3));
        assert_eq!(parse("constant:nan"), Set::Off);
    }

    #[test]
    fn roles_by_name() {
        assert_eq!(role(b"..CS_ScreenProbes_TemporalResolve_a911b49c_Wrapped.."), Some(Role::Resolve));
        assert_eq!(role(b"..CS_CacheScreenSpaceRadiance_eed2fc35.."), Some(Role::Radiance));
        assert_eq!(role(b"CS_AccumulateLocalLightInscatteringVolume_7e2f5eef"), Some(Role::Air));
        assert_eq!(role(b"CS_TemporalResolve"), None);
    }

    /// Unset, a found depth handle is bound and the caller still checks its
    /// size. Set, only the resolve binds one, and only at the image's size.
    #[test]
    fn depth_is_bound_only_for_a_matching_view() {
        assert!(!depth_match_set(None));
        assert!(!depth_match_set(Some(std::ffi::OsStr::new("0"))));
        assert!(!depth_match_set(Some(std::ffi::OsStr::new(""))));
        assert!(depth_match_set(Some(std::ffi::OsStr::new("1"))));
        assert_eq!(depth_bind(Role::Resolve, true, false, false), DepthBind::Set);
        assert_eq!(depth_bind(Role::Radiance, true, true, false), DepthBind::Set);
        assert_eq!(depth_bind(Role::Air, true, true, false), DepthBind::Set);
        assert_eq!(depth_bind(Role::Reflect, true, true, false), DepthBind::Set);
        assert_eq!(depth_bind(Role::Resolve, false, true, false), DepthBind::Skip);
        assert_eq!(depth_bind(Role::Resolve, true, true, true), DepthBind::Set);
        assert_eq!(depth_bind(Role::Resolve, true, false, true), DepthBind::Skip);
        assert_eq!(depth_bind(Role::Resolve, false, true, true), DepthBind::Skip);
        assert_eq!(depth_bind(Role::Radiance, false, false, true), DepthBind::Leave);
        assert_eq!(depth_bind(Role::Air, true, false, true), DepthBind::Leave);
        assert_eq!(depth_bind(Role::Reflect, false, false, true), DepthBind::Leave);
        assert_eq!(depth_bind(Role::Sky, true, true, true), DepthBind::Skip);
    }

    /// Forces [`reflect_part`] on this thread and clears it afterwards.
    struct ReflectForced;
    impl ReflectForced {
        fn new(part: f32) -> Self {
            FORCE_REFLECT.with(|c| c.set(Some(part)));
            Self
        }
    }
    impl Drop for ReflectForced {
        fn drop(&mut self) {
            FORCE_REFLECT.with(|c| c.set(None));
        }
    }

    #[test]
    fn reflections_follow_the_switch() {
        assert_eq!(parse_reflect(None), 0.0);
        assert_eq!(parse_reflect(Some("")), 0.0);
        assert_eq!(parse_reflect(Some("off")), 0.0);
        assert_eq!(parse_reflect(Some("0")), 0.0);
        assert_eq!(parse_reflect(Some("-1")), 0.0);
        assert_eq!(parse_reflect(Some("nan")), 0.0);
        assert_eq!(parse_reflect(Some(" 0.5 ")), 0.5);
        let _off = ReflectForced::new(0.0);
        assert_eq!(role(b"CS_UpdateHistory_abc"), None);
        assert_eq!(role(b"..CS_ScreenProbes_TemporalResolve_a.."), Some(Role::Resolve));
        drop(_off);
        let _on = ReflectForced::new(0.5);
        assert_eq!(role(b"CS_UpdateHistory_abc"), Some(Role::Reflect));
        assert_eq!(role(b"..CS_ScreenProbes_TemporalResolve_a.."), Some(Role::Resolve));
        assert_eq!(role(b"..CS_CacheScreenSpaceRadiance_x.."), Some(Role::Radiance));
        assert_eq!(role(b"CS_AccumulateLocalLightInscatteringVolume_x"), Some(Role::Air));
        let (level, mean, cap) = cap_for(Set::Off, true);
        assert_eq!(level, 0.6);
        assert!(!mean);
        assert_eq!(cap, [0.6, 0.6 * 0.98, 0.6 * 0.94]);
        let (level, mean, _) = cap_for(Set::Estimate(0.4), true);
        assert_eq!(level, 0.4);
        assert!(mean);
        let (_, mean, cap) = cap_for(Set::Constant(0.3), true);
        assert!(!mean);
        assert_eq!(cap, [0.3, 0.3 * 0.98, 0.3 * 0.94]);
    }

    #[test]
    fn the_hold_compiles() {
        let Some(host) = host() else {
            return;
        };
        make_hold(&host.device).expect("the reflection hold");
    }

    /// The passes compile with the system's compiler and the host takes them.
    #[test]
    fn the_passes_build() {
        let Some(host) = host() else {
            return;
        };
        make(&host.device, |bytes| host.uav_buffer(bytes)).expect("the passes");
    }

    /// Forces [`follow_sky`] on this thread and clears it afterwards.
    struct SkyForced;
    impl SkyForced {
        fn new(set: Option<[f32; 3]>) -> Self {
            FORCE_SKY.with(|c| c.set(Some(set)));
            Self
        }
    }
    impl Drop for SkyForced {
        fn drop(&mut self) {
            FORCE_SKY.with(|c| c.set(None));
        }
    }

    #[test]
    fn the_sky_follows_the_switch() {
        assert_eq!(parse_sky(""), None);
        assert_eq!(parse_sky("off"), None);
        assert_eq!(parse_sky("0"), None);
        assert_eq!(parse_sky("on"), Some([0.36, 0.05, 0.4]));
        assert_eq!(parse_sky("1,2,3"), Some([1.0, 2.0, 3.0]));
        let _off = SkyForced::new(None);
        assert!(!sky_on());
        assert_eq!(part("pixel", b"ps_sky_paraboloid_reflection"), None);
        assert_eq!(part("compute", b"CS_ScreenProbes_TemporalResolve_a"), Some(Role::Resolve));
        drop(_off);
        let _on = SkyForced::new(Some([0.36, 0.05, 0.4]));
        assert!(sky_on());
        assert_eq!(part("pixel", b"ps_sky_paraboloid_reflection_x"), Some(Role::Sky));
        assert_eq!(part("pixel", b"ps_other"), None);
        assert_eq!(part("compute", b"CS_ScreenProbes_TemporalResolve_a"), Some(Role::Resolve));
        assert_eq!(part("compute", b"CS_CacheScreenSpaceRadiance_x"), Some(Role::Radiance));
        let rs =
            "num32BitConstants=15, b0), DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0)), UAV(u1), UAV(u2), UAV(u3)";
        assert!(SOURCE.contains(rs));
        assert!(SOURCE.contains("#ifdef FOLLOW_SKY"));
    }

    #[test]
    fn the_sky_pass_compiles() {
        let Some(host) = host() else {
            return;
        };
        make_sky(&host.device).expect("the sky pass");
    }
}
