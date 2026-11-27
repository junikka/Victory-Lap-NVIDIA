//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::resource::RootIndexedView;
use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// Shader interface metadata shared by every pipeline variant and draw.
pub(crate) struct TablePlan {
    groups: Vec<TableGroup>,
    // Direct entries can interleave slots. Preserve their declaration order
    // when recording resource reads, writes, and transitions.
    direct: Vec<(usize, usize)>,
}

struct TableGroup {
    slot: u8,
    has_cbv: bool,
    has_uav: bool,
    max_dword: u32,
    entries: Vec<UsedResource>,
    indexed: Vec<RootIndexedView>,
    // Descriptor number -> first declaration, matching the bind-time search.
    by_descriptor: Vec<(u32, usize)>,
}

impl TablePlan {
    pub(crate) fn new(used: &[UsedResource], indexed: &[RootIndexedView]) -> Self {
        let mut slots: Vec<_> = used
            .iter()
            .filter(|u| u.root == RootType::Table)
            .map(|u| u.slot)
            .collect();
        slots.sort_unstable();
        slots.dedup();
        let mut groups: Vec<_> = slots
            .iter()
            .map(|&slot| TableGroup {
                slot,
                has_cbv: false,
                has_uav: false,
                max_dword: 0,
                entries: Vec::new(),
                by_descriptor: Vec::new(),
                indexed: indexed.iter().filter(|v| v.table_slot == slot).copied().collect(),
            })
            .collect();
        let mut direct = Vec::new();
        for &u in used.iter().filter(|u| u.root == RootType::Table) {
            let i = slots
                .binary_search(&u.slot)
                .expect("table slot collected from these declarations");
            let group = &mut groups[i];
            group.has_cbv |= u.kind == ResourceType::Cbv;
            group.has_uav |= u.kind == ResourceType::Uav;
            group.max_dword = group.max_dword.max(u.table_offset_dwords);
            direct.push((i, group.entries.len()));
            group
                .by_descriptor
                .push((u.table_offset_dwords / 8, group.entries.len()));
            group.entries.push(u);
        }
        for group in &mut groups {
            group.by_descriptor.sort_unstable();
            group.by_descriptor.dedup_by_key(|item| item.0);
        }
        Self { groups, direct }
    }
}

impl TableGroup {
    fn entry(&self, descriptor: u32) -> Option<&UsedResource> {
        let i = self
            .by_descriptor
            .binary_search_by_key(&descriptor, |item| item.0)
            .ok()?;
        Some(&self.entries[self.by_descriptor[i].1])
    }

    fn needs_snapshot(&self, heap: &crate::descriptor::Heap, host_count: u32, table: u64) -> bool {
        let index_at = |dword: u32| {
            table
                .checked_add(u64::from(dword) * 4)
                .and_then(|end| heap.index(end))
        };
        match index_at(self.max_dword) {
            Some(i) => i >= host_count,
            None => self
                .entries
                .iter()
                .any(|u| index_at(u.table_offset_dwords).is_some_and(|i| i >= host_count)),
        }
    }
}

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// Host states of a depth target, and of a texture shaders read.
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;

thread_local! {
    /// While a draw is prepared (`State::bind`, then its targets' states):
    /// the depth target the draw only tests (0: none, or one it writes),
    /// and whether a view its shaders read is of that same texture.
    static TESTED_DEPTH: std::cell::Cell<(usize, bool)> =
        const { std::cell::Cell::new((0, false)) };
    /// While a stage's tables are walked (`State::tables`): whether that
    /// stage's shader can store to buffers (`Translated::stores`).
    static STORES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    /// While `State::early_textures` runs the texture pass: the
    /// milliseconds and the bytes that run may take, in place of the
    /// pass's own per frame.
    static PASS_BUDGET: std::cell::Cell<Option<(u64, u64, bool)>> = const { std::cell::Cell::new(None) };
}

/// The milliseconds of every frame the pass that uploads textures picked
/// by index takes before a shader that picks by index has asked for it
/// (`VICTORY_EARLY_TEXTURES_MS`, 12; 0: never, as before), and the
/// milliseconds it may take just before a draw of the title's decoration
/// bake (`VICTORY_BAKE_TEXTURES_MS`, 1500; 0: nothing special).
///
/// The pass ran only from the first world frame on, where it met some
/// 24,000 slots at once and needed 60-80 frames for them. The title bakes
/// its characters' skin and decorations into textures of its own while it
/// loads, 159-1,165 frames before that (runs 156-174), with a pixel shader
/// that picks its sources by index: whatever was not on the card then was
/// baked as nothing, once, for the run. On the console a texture is there
/// when its bytes are.
fn early_budgets() -> (u64, u64) {
    static MS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        let ms = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default)
        };
        (ms("VICTORY_EARLY_TEXTURES_MS", 12), ms("VICTORY_BAKE_TEXTURES_MS", 1500))
    })
}

/// Whether a draw of the title's bake finds the textures it picks by index
/// as title memory has them at that moment (`VICTORY_BAKE_EXACT=0`: as the
/// pass leaves them, as before). The pass looks at a slot whose memory was
/// written or mapped a frame or two later, by when the rest of it has
/// landed, and at one uploaded whole a few frames later again, for a title
/// that fills memory unnoticed: a bake in between took the bytes from before
/// (another texture's), and what it bakes stays: coloured specks all over
/// the waitress's face for the whole diner scene (run n01), none in other
/// runs. So the pass ahead of a bake takes those slots too, and its uploads
/// wait for title memory that is being remapped (`crate::texture` `BAKING`).
fn bake_exact() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BAKE_EXACT").is_none_or(|v| v != "0"))
}

/// Whether a texture found changed with nothing to tell of it is looked at
/// again in the next frame (`VICTORY_RECHECK_SOON=0`: after four, as
/// before).
fn recheck_soon() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RECHECK_SOON").is_none_or(|v| v != "0"))
}

/// Frames ahead the pass before a bake takes slots from that are due only
/// then ([`bake_exact`]): those of notices (1 and 2), and the first looks
/// again ([`RECHECK_FRAMES`]), these at the frame's first bake alone (a
/// look reads 48 KiB, and a frame bakes up to sixteen times).
const BAKE_AHEAD: [u64; 2] = [12, 2];

/// What [`crate::descriptor::host::HostHeap::texture_news`] said when the
/// pass before a bake last served all there was, plus one (0: it did not):
/// until something is new, the next bakes of the frame need no pass.
static BAKE_SERVED: AtomicU64 = AtomicU64::new(0);

/// Free-threaded wrapper around a reusable upload-only [`HostList`] protected
/// by [`BAKE_UPLOAD`]. D3D12 command allocators, lists, and retained COM
/// resources are free-threaded when externally synchronized.
struct BakeList(HostList);
// SAFETY: accessed exclusively under `BAKE_UPLOAD`'s mutex.
unsafe impl Send for BakeList {}

/// Serializes synchronous one-shot ped/decoration bake uploads across
/// concurrent `VICTORY_THIN=1` recording threads and owns the reusable
/// upload-only host command list.
static BAKE_UPLOAD: std::sync::Mutex<Option<BakeList>> = std::sync::Mutex::new(None);

/// Pending descriptor publications for a synchronous bake upload pass: slots
/// stay on `neutral()` (or their previous compatible view) until the dedicated
/// upload list has finished executing on the host GPU.
struct BakeSync {
    publish: Vec<(u32, u64, u64)>,
    keep: Vec<std::sync::Arc<crate::resource::HostTexture>>,
}

/// The pixel shaders of the title's decoration bake (the title:
/// `PS_BakeDecorations_*`), by the first 16 hex digits of their hash as the
/// frame listings print it. They draw into a 256x256 corner of 2048x2048
/// targets and name no texture in a table.
pub(crate) const BAKE_PIXEL_SHADERS: [&str; 7] = [
    "c0a7fe77b1862e10",
    "0d1575a396dfe513",
    "b7cfd8dda91ebb6b",
    "280967ad158cee0b",
    "fa455c8a3c32577b",
    "d126bcae5d5d8d26",
    "fcee612943c008e0",
];

/// Geometry stages shared by every ped texture compositor / decoration bake
/// permutation (`vertex 089d6271be6650ef` fullscreen-triangle ped baker and
/// `mesh d4111a6e6ea49d91` decoration projection mesh), catching pixel-shader
/// permutations beyond `BAKE_PIXEL_SHADERS`.
const BAKE_GEOMETRY_SHADERS: [&str; 2] = ["089d6271be6650ef", "d4111a6e6ea49d91"];

/// Since the start: early runs of the texture pass, and those of them
/// made for a bake draw.
pub(crate) static EARLY_TEXTURE_PASSES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Whether a buffer range a shader names through an unordered access view
/// is given to shaders as written (snapshot before the submission, compare
/// and write back after) only if that shader's code can store to buffers.
/// The title binds buffers its shaders only read through such views, and
/// every range declared costs a snapshot and a compare of its bytes: 3.97
/// GB a frame in 31 submissions, 0.49 s of snapshots and 0.28 s of
/// comparing (run 28). On by default; `VICTORY_WRITE_SKIP=0` declares
/// every such range, as before.
fn write_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WRITE_SKIP").is_none_or(|v| v != "0"))
}

/// A run of indirect records clamped in one pass ([`windows`]): the host
/// copy they came from (its number), their kind and stride, the offset of
/// the first in that copy, how many, and the list's scratch buffer that
/// holds them clamped, record `i` at `i * step`.
#[derive(Clone)]
struct Window {
    copy: u64,
    kind: u32,
    stride: u32,
    /// How far apart the records are in the copy, and in the scratch
    /// buffer: what the title walks them by ([`windows`]).
    step: u32,
    base: u64,
    records: u32,
    scratch: ID3D12Resource,
}

/// The most records clamped in one pass ([`windows`]).
const WINDOW_MOST: u64 = 64;

/// Whether indirect draws that read record after record of one argument
/// buffer are clamped a window at a time (`VICTORY_INDIRECT_WINDOW=1`; off
/// by default).
///
/// The title issues its mesh draws one record per call, each from the next
/// slot of the same argument buffer: 3,460 a frame in gameplay, and before
/// each our clamp ran as a compute dispatch of its own between two waits,
/// so the card went from drawing to compute and back for every record. In
/// the timing run of the lit scene (run 79) a shadow draw of next to
/// nothing cost 14-16 microseconds, 4,126 records a frame.
///
/// With the switch, when a draw asks for a record a short step after the
/// one drawn before, the clamp takes that record and the ones that step
/// apart after it (16, then twice as many each time, [`WINDOW_MOST`] at
/// most, never past the buffer) into
/// the list's scratch buffer in one pass, and the draws that follow read
/// their record there with no pass of ours and no barrier (the scratch
/// buffer stays readable as arguments until the next clamp), until one of
/// these: a compute
/// dispatch, a direct draw, an indirect call with a count buffer or more
/// than one record, a draw whose shaders can store to buffers, a record
/// outside the window, another host list.
///
/// The step is the one the title is seen to take, not the stride its
/// command signature names: it draws one record a call, for which the
/// stride means nothing, and the title names 12 bytes (the three
/// words of a mesh dispatch) for records it keeps 32 bytes apart (2,173 of
/// a frame's 3,349 mesh records in run 87's listed frame; 283 are 16
/// apart, which ends a window). Walked by the named stride the window
/// never opened: run 87, with the switch, still clamped 5,068 records a
/// frame one at a time.
///
/// What it costs the guard against garbage records: nothing against a
/// hang (every record is bounded as before and charged to the list's
/// budget as before), but the budget is charged when a window is clamped,
/// so also for records of the window's tail that are never drawn. Those
/// hold garbage in this title (slots of all ones: run 93 spent lists'
/// budgets on them and emptied the draws after), so in a window a record
/// that asks for more than can run is left empty and charges nothing
/// (`crate::indirect::bound_ahead`); a tail record that looks real is
/// charged what it asks, a few thousand of 2^27. And arguments the title's CPU side changes are seen as
/// before, at the next submission.
fn windows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_WINDOW").is_some_and(|v| v == "1"))
}

/// Whether indirect executions whose arguments or count lie in title memory
/// the card reads in place (renderer port 1, `VICTORY_PORT_CONSTANTS=1`, or
/// `VICTORY_PORT_SECTIONS`) are clamped alone, right before they run
/// (`VICTORY_PORT_CONSTANTS_ORDER=1`; off by default here: in batches and
/// windows as any other). After avatarmanz 631400f5, where it is on unless 0.
///
/// A batch or a window clamps records ahead: in the pass recorded where it
/// was opened, before the commands between it and the execution that reads
/// the record. For a host copy that is fine, its bytes are the title's of
/// the submission. Where the card reads the title's memory in place, the
/// records the title's culling passes write between the two are read before
/// they are written: the execution draws the records of an earlier frame,
/// or none. Upstream: with port 1 on, palms and trees in a cutscene were
/// missing for runs of frames while the camera moved, and with ports 1 and
/// 5 off they were not.
fn in_place_order() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PORT_CONSTANTS_ORDER").is_some_and(|v| v == "1"))
}

/// Since the last line: indirect executions with arguments in place, those
/// clamped alone for it, and the 1 MiB ranges they came from (diagnostics).
static IN_PLACE_INDIRECT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static IN_PLACE_RANGES: std::sync::Mutex<std::collections::BTreeMap<u64, [u64; 3]>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// The slot of [`IN_PLACE_RANGES`] an indirect execution of `kind` counts in:
/// draws (any kind not named), dispatches, mesh draws.
fn in_place_slot(kind: u32) -> usize {
    // (`execute_indirect`'s kinds: 2 a dispatch, 10 a mesh draw.)
    match kind {
        2 => 1,
        10 => 2,
        _ => 0,
    }
}

/// An indirect execution of `kind` with its arguments at `arguments` in place.
fn in_place_indirect(arguments: u64, kind: u32) {
    IN_PLACE_INDIRECT[0].fetch_add(1, Ordering::Relaxed);
    IN_PLACE_INDIRECT[1].fetch_add(u64::from(in_place_order()), Ordering::Relaxed);
    let mut ranges = IN_PLACE_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    ranges.entry(arguments >> 20).or_default()[in_place_slot(kind)] += 1;
}

/// With the frame line: indirect executions with arguments in place per
/// frame over `frames`, and the ranges most of them came from. Nothing while
/// there are none (nothing is in place unless port 1 or its sections are on).
pub(crate) fn in_place_indirect_line(frames: f64) {
    let [all, alone] = [0, 1].map(|i| IN_PLACE_INDIRECT[i].swap(0, Ordering::Relaxed));
    if all == 0 {
        return;
    }
    let ranges = std::mem::take(&mut *IN_PLACE_RANGES.lock().unwrap_or_else(|e| e.into_inner()));
    let mut by: Vec<(u64, [u64; 3])> = ranges.into_iter().collect();
    by.sort_by_key(|(_, n)| std::cmp::Reverse(n.iter().sum::<u64>()));
    let top: Vec<String> = by
        .iter()
        .take(6)
        .map(|(mb, n)| {
            format!(
                "{:#x}+1MiB: {:.0} draws {:.0} dispatches {:.0} mesh",
                mb << 20,
                n[0] as f64 / frames,
                n[1] as f64 / frames,
                n[2] as f64 / frames
            )
        })
        .collect();
    tracing::info!(
        "renderer port 1: indirect executions with arguments or count in place per frame: {:.0}, of them {:.0} clamped alone right before they ran (VICTORY_PORT_CONSTANTS_ORDER=1); from {} ranges of 1 MiB, the busiest: {}",
        all as f64 / frames,
        alone as f64 / frames,
        by.len(),
        top.join("; ")
    );
}

/// Whether indirect executions of one record are clamped in batches
/// (`crate::indirect::batch`; `VICTORY_INDIRECT_BATCH=0`: each on its own,
/// or in windows ([`windows`])).
///
/// A frame of gameplay has 4,809 indirect executions (a capture: 3,523 mesh
/// draws, 948 dispatches, 338 draws), and with the windows still 2,355
/// clamps: each a dispatch between two full waits, with the scratch buffer
/// moved twice. The replayed frame's 84.6 ms are 75.3 ms with the clamps in
/// 133 passes, and 77.7 ms in the 980 of this.
///
/// A batch is opened by the first execution after a point where arguments
/// may have changed, and all executions up to the next such point read
/// theirs from it, each exactly the record it asks for (windows guess which
/// are next, and clamp what they do not know). Arguments in buffers change
/// by dispatches and by draws whose shaders store to buffers; those end the
/// batch, so the record an execution reads is clamped after all that was
/// recorded before it, and the pass waits for all of it, exactly as the
/// clamp of its own did. Between two executions of a batch is no wait of
/// ours: there is nothing between them to wait for but draws that store to
/// no buffer.
///
/// `VICTORY_INDIRECT_BATCH=2`: a dispatch does not end the batch before the
/// title waits for it (a barrier, or a pipeline sync) or draws: what a
/// dispatch writes is not there for a later command without a wait, and
/// dispatches the title runs side by side (its tile classes, up to 92 at a
/// time) are not held up by waits of ours. Off: whether the title has ways
/// to wait that do not pass through its lists' methods is not known.
fn batches() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_INDIRECT_BATCH").as_deref() {
        Ok("0") => 0,
        Ok("2") => 2,
        _ => 1,
    })
}

/// The records clamped by the pass opened last ([`batches`]).
struct Batch {
    table: crate::indirect::batch::Table,
    scratch: ID3D12Resource,
    /// Dispatches since, which may have written arguments
    /// (`VICTORY_INDIRECT_BATCH=2`).
    wrote: bool,
}

/// The host copies arguments were read from last: what a batch opens with,
/// as the executions after the first may read theirs from another one.
static ARGUMENT_COPIES: std::sync::Mutex<Vec<std::sync::Weak<crate::shadow::Shadow>>> = std::sync::Mutex::new(Vec::new());

/// [`ARGUMENT_COPIES`] that live, and `first` put in front of them.
fn argument_copies(first: &std::sync::Arc<crate::shadow::Shadow>) -> Vec<(u64, ID3D12Resource)> {
    let mut copies = ARGUMENT_COPIES.lock().unwrap_or_else(|e| e.into_inner());
    let live: Vec<_> = copies.iter().filter_map(std::sync::Weak::upgrade).filter(|s| s.id() != first.id()).collect();
    *copies = std::iter::once(first)
        .chain(&live)
        .take(crate::indirect::batch::SOURCES)
        .map(std::sync::Arc::downgrade)
        .collect();
    live.iter().map(|s| (s.id(), s.buffer.clone())).collect()
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The host state the bound depth view needs ([`depth_state`]).
    depth_state: u32,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the title set that goes straight onto the host list, kept for
    /// the host list that continues this one after a cut ([`State::replay`]):
    /// whether a topology was set, the render target and depth handles, the
    /// viewports, the scissors, the blend factor and the stencil reference.
    topology_set: bool,
    target_handles: Vec<u64>,
    depth_handle: u64,
    /// With [`targets_kept`]: what the handles' descriptors held when the
    /// title set the targets, for [`State::replay`]. Empty otherwise.
    target_views: Vec<Option<crate::descriptor::RtvSnapshot>>,
    depth_view: Option<crate::descriptor::RtvSnapshot>,
    targets_set: bool,
    /// The formats of the views [`State::set_targets`] bound on the host
    /// list: a draw's pipeline is one for them when
    /// `VICTORY_TARGET_VARIANTS=1` (`crate::pipeline::host_pipeline_for`).
    target_formats: crate::pipeline::host::TargetFormats,
    /// The render target views and the depth view the host list has, once
    /// it has any ([`State::set_targets`]): per view the host texture (0:
    /// a null view) and what its descriptor says.
    host_targets: Option<(Vec<TargetKey>, Option<TargetKey>)>,
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<RECT>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    depth_bounds: Option<[f32; 2]>,
    /// The indirect records clamped last, if later draws may still read
    /// them ([`windows`]).
    window: Option<Window>,
    /// The records clamped in a batch, if later executions may still read
    /// them ([`batches`]).
    batch: Option<Batch>,
    /// The user data bound last, of the last stage that has some (for the
    /// report of a removed device, `crate::host::indirects`).
    #[cfg(feature = "diagnostics")]
    user: Vec<u32>,
    /// The tables the last bind made afresh for a program the stand-in for
    /// the indirect light follows (`crate::ambient`): root parameter, bytes
    /// into the view heap.
    ambient_tables: Vec<(u8, u32)>,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    host_heaps_set: bool,
    /// Tables made afresh in this list: (table, slot count, contents) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u32, u64), FreshTable, std::hash::BuildHasherDefault<crate::host::AddressHash>>,
    /// The dynamic slots the list holds: title heap, that heap's identity
    /// (`HostHeap::id`), first, count; given back when the list has run, is
    /// reset or released.
    dynamic: Vec<(usize, u64, u32, u32)>,
    /// Of `dynamic`: slots counted in the in-flight total, after the list
    /// was submitted and before it has run.
    flight_slots: u32,
    /// [`flight_slots`] was added to that total.
    in_flight: bool,
    /// Root SRVs bound as one-view tables in this list: (title heap, title
    /// address; 0 for none) → dynamic slot, and the host copy the slot
    /// shows: later draws of the list bring that one up to date, whatever
    /// copy the address would find by then (`shadow.rs` `tails`).
    root_srv_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// Root UAVs bound as one-view tables in this list, likewise.
    root_uav_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// The list ran and its dynamic slots were given back.
    tables_gone: bool,
    /// This recording, for [`crate::descriptor::host::HostHeap::make_listed`]
    /// (0: it has none yet).
    recording: u64,
}

impl Drop for State {
    fn drop(&mut self) {
        self.leave_flight();
        for &(heap, id, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, id, first, n);
        }
    }
}

impl State {
    /// The list's submission has run (submissions are waited for): the
    /// dynamic slots its tables were made in are given back, not kept
    /// until the title resets the list. A frame of the opening cutscene
    /// asks for 12,500 tables; held until the resets they ran the 131,072
    /// slots out, and up to 4,900 draws a frame went without their tables.
    /// shortcut: a list the title runs a second time without a reset then
    /// reads slots others may have taken (logged once).
    pub(crate) fn tables_done(&mut self) {
        self.leave_flight();
        if self.tables_gone && self.dynamic.is_empty() {
            return;
        }
        if self.tables_gone {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("a list ran again after its descriptor tables were given back"));
        }
        for (heap, id, first, n) in self.dynamic.drain(..) {
            crate::descriptor::release_dynamic(heap, id, first, n);
        }
        self.fresh_tables.clear();
        self.recording = 0;
        self.root_srv_slots.clear();
        self.root_uav_slots.clear();
        self.tables_gone = true;
    }

    /// The list has been submitted and has not run: its dynamic slots are
    /// held until [`State::tables_done`]. Counted once.
    pub(crate) fn hold_in_flight(&mut self) {
        if self.in_flight {
            return;
        }
        self.in_flight = true;
        self.flight_slots = self.dynamic.iter().fold(0u32, |sum, (_, _, _, n)| sum.saturating_add(*n));
        crate::descriptor::host::add_flight_slots(self.flight_slots);
    }

    fn leave_flight(&mut self) {
        if !self.in_flight {
            return;
        }
        self.in_flight = false;
        crate::descriptor::host::sub_flight_slots(self.flight_slots);
        self.flight_slots = 0;
    }
}

/// Whether a list continued after a cut ([`State::replay`]) sets the render
/// targets from what their descriptors held when the title set them
/// (`VICTORY_TARGETS_KEPT=1`; fork grosterr 8f7d05d5a). Off by default: the
/// views are then made again from the handles' descriptors as they are at
/// the replay, and a descriptor the title rewrote in between names another
/// view, so the part after the cut draws into the wrong target.
fn targets_kept() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| targets_kept_from(std::env::var("VICTORY_TARGETS_KEPT").ok().as_deref()))
}

/// [`targets_kept`]: on for `1` only.
fn targets_kept_from(value: Option<&str>) -> bool {
    value == Some("1")
}

/// An experiment on the title's scissor rectangles (`VICTORY_SCISSOR_TEST=full`):
/// they are all widened to the whole target. The title's room at
/// the start of the prologue lost its walls in five runs of seven, and in
/// those runs the player's body and the world outside were cut by one
/// rectangle around the window (run 85, frame 2046): one run with this on
/// says whether the walls are drawn and cut away by such a rectangle, or
/// not drawn at all. Off by default: a rectangle the title sets is part of
/// its picture.
fn scissor_test() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SCISSOR_TEST").as_deref() == Ok("full"))
}

/// An experiment on mesh draws (`VICTORY_MESH_TEST`): 1 for `skip` (none
/// is issued), 2 for `scissor` (each runs its shaders but rasterises one
/// pixel), 0 otherwise. It tells a card hung by a mesh shader from one hung
/// by what the shader's triangles cover.
fn mesh_test() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_MESH_TEST").as_deref() {
        Ok("skip") => 1,
        Ok("scissor") => 2,
        _ => 0,
    })
}

/// `VICTORY_MESH_TEST=keep:<hex digits>[:<hex digits>]`: only the mesh
/// draws whose mesh shader's hash ends in one of the first digits (and
/// whose digit before it is one of the second, if given) are issued: a
/// search for the shaders that stop the card.
fn mesh_kept(pipeline: usize) -> bool {
    static KEEP: std::sync::OnceLock<Option<(String, String)>> = std::sync::OnceLock::new();
    let Some((last, before)) = KEEP.get_or_init(|| {
        let value = std::env::var("VICTORY_MESH_TEST").ok()?;
        let mut parts = value.strip_prefix("keep:")?.split(':');
        Some((parts.next()?.to_owned(), parts.next().unwrap_or("").to_owned()))
    }) else {
        return true;
    };
    let Some(p) = crate::pipeline::host_pipeline(pipeline as This) else {
        return true;
    };
    let Some(at) = p.name.find("mesh ") else {
        return true;
    };
    let hash = p.name[at + 5..].as_bytes();
    if hash.len() < 16 {
        return true;
    }
    last.as_bytes().contains(&hash[15]) && (before.is_empty() || before.as_bytes().contains(&hash[14]))
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// The draws dropped since [`take_dropped`].
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn dropped_now() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
/// A root SRV's dynamic slot in a list, and the host copy it shows.
type RootSlot = (u32, Option<std::sync::Arc<crate::shadow::Shadow>>);

/// Buffer views of the heap whose slot showed another host buffer than
/// the copy the pass brought up to date for them, since the start
/// (diagnostics; for the memory line). It stays 0 while the pass and the
/// slots agree; with `VICTORY_VIEW_EXTENT=0` it counts the views that were
/// read stale.
#[cfg(feature = "diagnostics")]
pub(crate) static HEAP_VIEWS_ELSEWHERE: AtomicU64 = AtomicU64::new(0);
/// Passes over the view heap's buffer views for shaders that pick them by
/// index ([`State::heap_buffers`]), and the microseconds they took.
pub(crate) static HEAP_BUFFER_PASSES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_BUFFER_MICROS: AtomicU64 = AtomicU64::new(0);
/// Of the passes, nanoseconds since the last frame line: looking at slots,
/// the spans and refreshes of their copies, the texture side.
#[cfg(feature = "diagnostics")]
pub(crate) static HEAP_SPLIT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Passes after which every buffer slot of the heap was looked at again.
const HEAP_ROUND: u64 = 16;

/// Whether the slots looked at again for no write are a part of them at
/// every pass, each slot at every sixteenth (`VICTORY_HEAP_SPREAD=0`: all of
/// them at every sixteenth pass, as before).
///
/// All at once was some 115 ms on a thread that records the frame, every
/// sixteenth frame (72,000 slots; run record-1: 11 ms a pass in the lines
/// without such a pass, 22 in those of ten frames with one). A slot is
/// looked at as often as before.
fn heap_spread() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SPREAD").is_none_or(|v| v != "0"))
}

/// The host copy and range the buffer view of each of a heap's slots reads
/// ([`State::heap_buffers`]), by slot.
#[derive(Default)]
struct HeapRanges<T>(Vec<Option<(T, u64, u64)>>);

/// Whether the heap buffer pass runs once a frame for each view heap a list
/// binds, not once for all of them, and a list holds each heap's table of
/// strides by its own key (`VICTORY_SWEEP_PER_HEAP`, from codec-d/cool_fox
/// c64fc9b4). On unless `=0` here, as in the fox base this tree plays from
/// (off by default upstream).
fn sweep_per_heap() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SWEEP_PER_HEAP").is_none_or(|v| v != "0"))
}

/// What the heap buffer pass keeps of a view heap, on the heap
/// ([`crate::descriptor::host::HostHeap`]), so that a heap made where one was
/// let go starts afresh (codec-d/cool_fox c64fc9b4).
pub(crate) struct BufferSweep {
    passes: u64,
    ranges: HeapRanges<std::sync::Arc<crate::shadow::Shadow>>,
    // Last refresh frame and covered interval for throttled, unwatched copies.
    unwatched: std::collections::HashMap<u64, (u64, u64, u64)>,
}

impl Default for BufferSweep {
    fn default() -> Self {
        Self { passes: 0, ranges: HeapRanges(Vec::new()), unwatched: Default::default() }
    }
}

impl<T> HeapRanges<T> {
    fn set(&mut self, slot: u32, range: Option<(T, u64, u64)>) {
        let slot = slot as usize;
        if range.is_none() && slot >= self.0.len() {
            return;
        }
        if slot >= self.0.len() {
            self.0.resize_with(slot + 1, || None);
        }
        self.0[slot] = range;
    }

    /// Forgets the slots of one of `parts` parts
    /// (`HostHeap::buffer_slots_of`).
    fn forget_part(&mut self, part: u32, parts: u32) {
        self.0.iter_mut().skip(part as usize).step_by(parts as usize).for_each(|r| *r = None);
    }

    fn ranges(&self) -> impl Iterator<Item = &(T, u64, u64)> {
        self.0.iter().flatten()
    }
}

#[cfg(test)]
mod heap_range_tests {
    use super::{HEAP_ROUND, HeapRanges};

    /// Round by round every slot is looked at again once, whatever was
    /// written between, and what a slot holds is what was found last.
    #[test]
    fn every_slot_is_looked_at_again_in_a_round() {
        const SLOTS: u32 = 1000;
        let mut ranges = HeapRanges(Vec::new());
        ranges.set(7, None);
        assert_eq!(ranges.ranges().count(), 0);
        for slot in (0..SLOTS).step_by(3) {
            ranges.set(slot, Some(((), 0, 1)));
        }
        let parts = HEAP_ROUND as u32;
        for pass in 1..=HEAP_ROUND {
            let part = (pass % HEAP_ROUND) as u32;
            ranges.forget_part(part, parts);
            for slot in (0..SLOTS).step_by(3).filter(|slot| slot % parts == part) {
                assert!(ranges.0[slot as usize].is_none(), "slot {slot} not forgotten");
                ranges.set(slot, Some(((), pass, pass)));
            }
            assert_eq!(ranges.ranges().count(), SLOTS.div_ceil(3) as usize);
        }
        assert!(ranges.ranges().all(|r| r.1 != 0), "a slot not looked at in a round");
        ranges.set(3, None);
        assert_eq!(ranges.ranges().count(), SLOTS.div_ceil(3) as usize - 1);
    }

    /// The spans of 4,400 copies from the ranges of 72,000 slots, as a pass
    /// makes them, by slot and from the table they were in. Run with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn spans_of_a_pass() {
        use std::collections::HashMap;
        use std::sync::Arc;
        let copies: Vec<Arc<u64>> = (0..4400).map(Arc::new).collect();
        let (mut table, mut slots) = (HashMap::new(), HeapRanges(Vec::new()));
        for slot in 0..72_000u32 {
            let copy = &copies[slot as usize * 7919 % 4400];
            table.insert(slot, (copy.clone(), u64::from(slot), u64::from(slot) + 64));
            slots.set(slot, Some((copy.clone(), u64::from(slot), u64::from(slot) + 64)));
        }
        fn join<S: std::hash::BuildHasher + Default>(ranges: &mut dyn Iterator<Item = &(Arc<u64>, u64, u64)>) -> usize {
            let mut spans: HashMap<usize, (Arc<u64>, u64, u64), S> = HashMap::default();
            for (copy, from, to) in ranges {
                let span = spans.entry(Arc::as_ptr(copy) as usize).or_insert_with(|| (copy.clone(), *from, *to));
                span.1 = span.1.min(*from);
                span.2 = span.2.max(*to);
            }
            spans.len()
        }
        let best = |f: &mut dyn FnMut() -> usize| {
            (0..30)
                .map(|_| {
                    let t = std::time::Instant::now();
                    assert_eq!(f(), 4400);
                    t.elapsed().as_secs_f64() * 1e3
                })
                .fold(f64::MAX, f64::min)
        };
        println!(
            "{:.2} ms from the table, {:.2} ms by slot",
            best(&mut || join::<std::hash::RandomState>(&mut table.values())),
            best(&mut || join::<std::hash::BuildHasherDefault<crate::host::AddressHash>>(&mut slots.ranges())),
        );
    }
}

/// Textures brought up to date for shaders that pick them by index
/// ([`State::heap_textures`]), the bytes of their memory, and the slots
/// still waiting.
pub(crate) static HEAP_TEXTURES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURE_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURES_WAITING: AtomicU64 = AtomicU64::new(0);
/// Passes that made no texture because the card had no room
/// (`crate::resource::room`), for the memory line.
pub(crate) static HEAP_TEXTURES_FULL: AtomicU64 = AtomicU64::new(0);

/// Frames after a whole upload of a texture picked by index at which its
/// memory is looked at again, each counted from the look before.
const RECHECK_FRAMES: [u64; 5] = [4, 12, 40, 120, 400];

/// Looks at uploaded textures picked by index that found the memory as it
/// was, and that found it changed (uploaded again), since the start.
pub(crate) static HEAP_TEXTURES_RECHECKED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// The same of the looks after the last of [`RECHECK_FRAMES`]
/// ([`recheck_long`]).
pub(crate) static HEAP_TEXTURES_RECHECKED_LATE: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Frames between the looks at a texture picked by index after the last of
/// [`RECHECK_FRAMES`], for as long as its slot shows it
/// (`VICTORY_RECHECK_LONG=<frames>`, at least 60; `1`: 600; unset or 0:
/// none, as before: the fifth look, 576 frames after the upload, is the
/// last).
///
/// Session 0152: of 386 textures found changed with nothing to tell of it,
/// 27 were found 600 frames or more after their upload and 30 more after
/// 400, and a guard's wrist showed pink and green squares four minutes into
/// play: a texture the title filled after the last look keeps what its
/// memory held before, another texture's bytes, for the rest of the run. At
/// 5 to 20 frames a second the title's own streaming takes more frames than
/// at the console's 30. A look reads 48 KiB.
///
/// `VICTORY_HEAP_RECHECK_EVERY=<frames>` is upstream's name for the same looks
/// (e4245f3c, after RyanCase94's 920423a9), read when ours is unset or 0: the
/// frames as given.
pub(crate) fn recheck_long() -> u64 {
    static FRAMES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *FRAMES.get_or_init(|| match recheck_long_of(std::env::var("VICTORY_RECHECK_LONG").ok().as_deref()) {
        0 => std::env::var("VICTORY_HEAP_RECHECK_EVERY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0),
        frames => frames,
    })
}

/// [`recheck_long`] of the variable's value.
fn recheck_long_of(value: Option<&str>) -> u64 {
    match value.and_then(|v| v.trim().parse::<u64>().ok()) {
        None | Some(0) => 0,
        Some(1) => 600,
        Some(frames) => frames.max(60),
    }
}

/// Frames between the looks, after the last of [`RECHECK_FRAMES`], at a
/// texture picked by index whose pages the title also shows at other
/// addresses (`VICTORY_ALIASED_RECHECK=<frames>`; kitty's
/// `K6_ALIASED_RECHECK_EVERY` when ours is unset; unset or 0: as any other
/// texture, [`recheck_long`]; kitty's own default is 30). The title paints
/// such textures (faces, hair) through the other view with nothing to tell
/// of it here, long after their upload: kitty's logs found them changed 1,
/// 15 and 41 frames after upload, the same slots over and over, and what
/// changed after the last look stayed stale on the card (blocky tiles).
/// After kitty ae3cce821. Only asked about when [`recheck_long`] is 0.
fn aliased_every() -> u64 {
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let frames = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok());
        let every = frames("VICTORY_ALIASED_RECHECK").or_else(|| frames("K6_ALIASED_RECHECK_EVERY")).unwrap_or(0);
        if every > 0 {
            tracing::info!(
                "textures picked by index shown at other addresses too are looked at again every {every} frames \
                 after the schedule (VICTORY_ALIASED_RECHECK)"
            );
        }
        every
    })
}

/// The `long` of [`next_look`] for one texture: [`recheck_long`]'s when
/// set, else `aliased` frames for a texture shown at other addresses too
/// ([`aliased_every`]), else 0. `shown_elsewhere` is asked only then.
/// Pure, for testing.
fn long_for(long: u64, aliased: u64, shown_elsewhere: impl FnOnce() -> bool) -> u64 {
    if long == 0 && aliased > 0 && shown_elsewhere() { aliased } else { long }
}

/// What follows a look that found the memory as it was: the frames until
/// the next look and its number, or `None` when that was the last. `looks`
/// is the number of the look just made (0 the first); `long` is
/// [`recheck_long`]. The looks after the last of [`RECHECK_FRAMES`] all
/// carry the same number, one past the scheduled ones.
fn next_look(looks: u8, long: u64) -> Option<(u64, u8)> {
    match RECHECK_FRAMES.get(usize::from(looks) + 1) {
        Some(&wait) => Some((wait, looks + 1)),
        None if long > 0 => Some((long, RECHECK_FRAMES.len() as u8)),
        None => None,
    }
}

/// Whether look number `looks` is one after the last of [`RECHECK_FRAMES`].
fn late_look(looks: u8) -> bool {
    usize::from(looks) >= RECHECK_FRAMES.len()
}

/// Frames a texture found changed twice with nothing to tell of it is looked
/// at in every frame for, counted from its last change.
const VOLATILE_FRAMES: u64 = 2000;
/// How many such textures are kept at most (a look reads 48 KiB).
const VOLATILE_MOST: usize = 512;

/// Whether a texture picked by index that was found changed twice with
/// nothing to tell of it is looked at in every frame for [`VOLATILE_FRAMES`]
/// after its last change (`VICTORY_RECHECK_VOLATILE=1`; off by default: after
/// each change the looks back off again, 1 or 4, 12, 40, 120, 400 frames, as
/// before). Port of miamiwalk ebcf56f1 (its draw.rs part; on unless `0`
/// there).
///
/// The title fills these through another view of their pages, which neither
/// a descriptor write nor the runtime tells of, and it is the same few again
/// and again: in runs spin19-base, spot-p16 and move-p16 the looks found 32,
/// 66 and 60 changes, and 6, 4 and 11 textures changed twice or more and
/// were 19, 51 and 48 of them. Each of those changes was shown as late as
/// the look that found it. `VICTORY_ALIASED_RECHECK` looks at every texture
/// shown at other addresses too, at one interval, after the schedule; this
/// one looks at the few that were caught, in every frame, from the second
/// catch on. The line "textures picked by index found changed with nothing
/// to tell of it" counts them and the frames the changes waited, either way.
fn volatile_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_RECHECK_VOLATILE").is_ok_and(|v| v.trim() == "1");
        if on {
            tracing::info!(
                "textures picked by index found changed twice with nothing to tell of it are looked at in every frame \
                 for {VOLATILE_FRAMES} frames after (VICTORY_RECHECK_VOLATILE=1)"
            );
        }
        on
    })
}

/// Textures (by the address of their memory) found changed after their
/// upload with nothing to tell of it: how often, and the frame of the last
/// time.
#[derive(Default)]
struct Volatile(std::collections::HashMap<u64, (u32, u64)>);

impl Volatile {
    /// Notes a change of the texture at `at` in `frame`. With
    /// [`VOLATILE_MOST`] kept, those quiet for [`VOLATILE_FRAMES`] go; if
    /// none is, a new one is not kept.
    fn note(&mut self, at: u64, frame: u64) {
        if self.0.len() >= VOLATILE_MOST && !self.0.contains_key(&at) {
            self.0.retain(|_, (_, last)| frame.saturating_sub(*last) < VOLATILE_FRAMES);
            if self.0.len() >= VOLATILE_MOST {
                return;
            }
        }
        let entry = self.0.entry(at).or_insert((0, frame));
        *entry = (entry.0.saturating_add(1), frame);
    }

    /// Whether the texture at `at` is one to look at in every frame now.
    fn every_frame(&self, at: u64, frame: u64) -> bool {
        self.0.get(&at).is_some_and(|&(changes, last)| changes >= 2 && frame.saturating_sub(last) < VOLATILE_FRAMES)
    }

    /// How many were found changed twice or more.
    fn twice(&self) -> usize {
        self.0.values().filter(|v| v.0 >= 2).count()
    }
}

/// [`Volatile`] of the run. (Noted without the switch too: the line says how
/// many textures it would look at.)
static VOLATILE: std::sync::Mutex<Option<Volatile>> = std::sync::Mutex::new(None);

/// Whether the texture at `at` is looked at in every frame now
/// ([`volatile_on`]).
fn volatile(at: u64, frame: u64) -> bool {
    volatile_on()
        && VOLATILE.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|kept| kept.every_frame(at, frame))
}

/// The frames a change waited for the look that found it: look number
/// `looks` (0: the first after an upload, `since_upload` frames after it),
/// `every_frame` if the texture was looked at in every frame, `long` the
/// frames between the looks after the last of [`RECHECK_FRAMES`].
fn waited_for(looks: u8, since_upload: u64, every_frame: bool, long: impl FnOnce() -> u64) -> u64 {
    match looks {
        0 => since_upload,
        _ if every_frame => 1,
        later => RECHECK_FRAMES.get(usize::from(later)).copied().unwrap_or_else(long),
    }
}

/// Since the start: looks that found a texture changed with nothing to tell
/// of it, the frames those had waited since the look before (what was shown
/// meanwhile was wrong for some of them), the longest such wait, and looks
/// made a frame after the one before ([`volatile`]).
static RECHECK_WAITED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// A line of the changes the looks at uploaded textures found, and of the
/// walks of the queue those looks are kept in.
pub(crate) fn recheck_line() {
    let [changed, waited, longest, every] = [0, 1, 2, 3].map(|i| RECHECK_WAITED[i].load(Ordering::Relaxed));
    let twice = VOLATILE.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, Volatile::twice);
    let [walks, slots, ns] = crate::descriptor::host::queue_walks();
    static FRAME: AtomicU64 = AtomicU64::new(0);
    let now = crate::frame::current();
    let frames = now.saturating_sub(FRAME.swap(now, Ordering::Relaxed)).max(1) as f64;
    tracing::info!(
        "textures picked by index found changed with nothing to tell of it, since the start: {changed} changes had \
         waited {:.1} frames for their look on average, {longest} at most; {twice} textures changed twice or more, \
         {every} looks made in every frame at those (VICTORY_RECHECK_VOLATILE=1: now {}); since the last line the \
         queue was walked {:.1} times a frame over {:.0} slots each, {:.2} ms a frame on all threads (the wait for \
         its lock included)",
        waited as f64 / changed.max(1) as f64,
        if volatile_on() { "on" } else { "off" },
        walks as f64 / frames,
        slots as f64 / walks.max(1) as f64,
        ns as f64 / 1e6 / frames
    );
}

#[cfg(test)]
mod volatile_tests {
    use super::{RECHECK_FRAMES, VOLATILE_FRAMES, VOLATILE_MOST, Volatile, waited_for};

    /// One change is not enough; from the second on the texture is looked at
    /// in every frame, until it has been quiet for [`VOLATILE_FRAMES`], and
    /// a change after that counts on from where it was.
    #[test]
    fn a_texture_caught_twice_is_looked_at_in_every_frame_for_a_while() {
        let mut kept = Volatile::default();
        assert!(!kept.every_frame(0x1000, 10));
        kept.note(0x1000, 10);
        assert!(!kept.every_frame(0x1000, 11), "once is not enough");
        assert_eq!(kept.twice(), 0);
        kept.note(0x1000, 50);
        assert!(kept.every_frame(0x1000, 51));
        assert!(!kept.every_frame(0x2000, 51), "another texture");
        assert!(kept.every_frame(0x1000, 50 + VOLATILE_FRAMES - 1));
        assert!(!kept.every_frame(0x1000, 50 + VOLATILE_FRAMES), "quiet for long enough");
        assert_eq!(kept.twice(), 1);
        kept.note(0x1000, 50 + VOLATILE_FRAMES + 7);
        assert!(kept.every_frame(0x1000, 50 + VOLATILE_FRAMES + 8));
    }

    /// No more than [`VOLATILE_MOST`] are kept: the quiet ones go for a new
    /// one, and with none quiet the new one is not kept.
    #[test]
    fn the_kept_textures_are_bounded() {
        let mut kept = Volatile::default();
        for i in 0..VOLATILE_MOST as u64 {
            kept.note(i << 16, 100);
            kept.note(i << 16, 101);
        }
        assert_eq!(kept.twice(), VOLATILE_MOST);
        // None quiet yet: a new one is not kept, one that is there counts on.
        kept.note(0xdead_0000, 200);
        kept.note(0xdead_0000, 201);
        assert!(!kept.every_frame(0xdead_0000, 202));
        kept.note(0, 300);
        assert!(kept.every_frame(0, 300 + VOLATILE_FRAMES - 1));
        // All but that one quiet for long enough: they go, the new one stays.
        let later = 101 + VOLATILE_FRAMES;
        kept.note(0xdead_0000, later);
        kept.note(0xdead_0000, later + 1);
        assert!(kept.every_frame(0xdead_0000, later + 2));
        assert_eq!(kept.twice(), 2);
    }

    /// What a change waited: the first look counts from the upload, a look
    /// in every frame one, a later one its place in the schedule, and one
    /// after the schedule the interval asked for.
    #[test]
    fn a_change_waited_what_the_look_before_it_was_set_to() {
        assert_eq!(waited_for(0, 4, false, || unreachable!()), 4);
        assert_eq!(waited_for(0, 1, true, || unreachable!()), 1);
        assert_eq!(waited_for(2, 56, true, || unreachable!()), 1);
        assert_eq!(waited_for(2, 56, false, || unreachable!()), RECHECK_FRAMES[2]);
        assert_eq!(waited_for(RECHECK_FRAMES.len() as u8, 9000, false, || 600), 600);
    }
}

/// Uploads of textures already on the card made while it had no room for
/// new ones, since the start.
pub(crate) static HEAP_TEXTURES_REUPLOADED_FULL: AtomicU64 = AtomicU64::new(0);

/// Whether a pass over the textures picked by index on a full card holds
/// back the slots that need a new host copy (`VICTORY_FULL_CARD_PASS=0`:
/// no, as before: they stay due, each pass looks at the same ones again
/// and stops at the 257th). Held back, such a slot is looked at again
/// after about four seconds or as soon as the card has room, and the pass
/// goes on to the slots behind it: the ones whose host copy exists and
/// only wants bringing up to date, and the ones whose memory went (letting
/// their host copies go is what makes room). What is uploaded while the
/// card has room is the same either way.
fn full_card_pass() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FULL_CARD_PASS").is_none_or(|v| v != "0"))
}

/// Whether a texture that has its host copy is still brought up to date
/// while the card has no room for new ones (`VICTORY_REUPLOAD_WHEN_FULL=0`:
/// no).
fn reupload_when_full() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REUPLOAD_WHEN_FULL").is_none_or(|v| v != "0"))
}

/// Whether textures picked by index are looked at again after their upload
/// (`VICTORY_HEAP_RECHECK=0`: no).
fn heap_recheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_RECHECK").is_none_or(|v| v != "0"))
}
/// Times a placed texture's slot was passed over because its memory is not
/// mapped.
pub(crate) static HEAP_TEXTURES_SAME: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURES_AWAY: AtomicU64 = AtomicU64::new(0);
/// Slots the pass found up to date already: their texture is uploaded by
/// the draws that read it through a table (it has a staging buffer).
pub(crate) static HEAP_TEXTURES_TABLED: AtomicU64 = AtomicU64::new(0);
/// The queue of textures picked by index at the last pass
/// (`TextureQueue::census`): slots in it, due, waiting for their memory,
/// uploaded in part, and the frames the oldest due one has waited.
pub(crate) static HEAP_TEXTURES_QUEUE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Since the start: looks at a slot uploaded in part that found no more of
/// its memory mapped; uploads of textures not all mapped; and of those,
/// the ones whose first page was not there (asked by the first page alone,
/// as before, they counted as not there at all).
pub(crate) static HEAP_TEXTURES_PARTLY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);
/// Direct and indirect dispatches spread across z (`VICTORY_SPREAD`), and of
/// the indirect ones those run lean (`VICTORY_SPREAD_LEAN`).
pub(crate) static SPREAD: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// What a render target or depth view of the host is made from: the host
/// texture, and [`crate::descriptor::RtvRecord::key`].
type TargetKey = (usize, (usize, [u32; 7]));

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB
/// (cut at their buffer's end), other views as all that is left of their
/// buffer. (They were taken as 1 MiB: 1,452 of the 2,021 mesh vertex
/// streams of a cutscene frame reach further, and what lay past the first
/// megabyte was never brought up to date - meshes without bodies, exploded
/// triangles. `VICTORY_ROOT_VIEW_MB` sets a limit again.)
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<u64> {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_ROOT_VIEW_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(u64::MAX, |mb| mb << 20)
    });
    let shadow = {
        let _ask = crate::readkinds::asking(if srv { crate::readkinds::ROOT_SRV } else { crate::readkinds::ROOT_CBV });
        crate::shadow::find(address, 1)?
    };
    let len = shadow
        .left(address)
        .min(if srv { limit } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    shadow.bound(crate::shadow::BOUND_ROOT_VIEW, 0);
    if srv || gpu % 256 == 0 {
        // (Constants not on 256 bytes go to upload memory: no read of the copy.)
        shadow.read_as(
            if srv { crate::readkinds::ROOT_SRV } else { crate::readkinds::ROOT_CBV },
            len.min(crate::readkinds::UNSIZED),
        );
        shadow.refresh(list, address, len);
        #[cfg(feature = "diagnostics")]
        crate::book::root(gpu, len);
        return Some(gpu);
    }
    list.fill_scratch(address as *const u8, len)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

/// The pass over the buffer views of `heap` that shaders pick by index
/// ([`State::heap_buffers`]): their host slots are made, and the host copies
/// they show are to be refreshed.
fn heap_buffer_pass(heap: &crate::descriptor::Heap, host: &crate::descriptor::host::HostHeap, frame: u64) -> Vec<crate::host::aside::Refresh> {
    let started = std::time::Instant::now();
    // Per host heap: the host copy and range each slot's buffer view
    // reads, kept from pass to pass. Only the slots written since the
    // last pass are looked at again (83,000 slots a pass otherwise, 0.1 s
    // at best); all of them every 16 passes, because a view of a
    // released resource goes stale without a write, and a slot written
    // before its memory had a host copy is only made then.
    let mut kept = host.buffer_sweep.lock().unwrap_or_else(|e| e.into_inner());
    let BufferSweep { passes, ranges, unwatched } = &mut *kept;
    let changed = host.buffer_changes();
    let all = *passes == 0 || (*passes % HEAP_ROUND == 0 && !heap_spread());
    let slots = if all {
        ranges.0.clear();
        host.buffer_slots()
    } else if heap_spread() {
        // The written ones, and this pass's part of the others.
        let (part, parts) = ((*passes % HEAP_ROUND) as u32, HEAP_ROUND as u32);
        ranges.forget_part(part, parts);
        let mut slots = host.buffer_slots_of::<{ HEAP_ROUND as u32 }>(part);
        slots.extend(changed);
        slots
    } else {
        changed
    };
    *passes += 1;
    // Whether the device was there before the pass: a view that removes
    // it is named below.
    static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let alive = !GONE.load(Ordering::Relaxed) && crate::descriptor::host::device_removed().is_none();
    let mut unordered = Vec::new();
    /// The title's heap, for the threads that look.
    struct Shared<'a>(&'a crate::descriptor::Heap);
    // SAFETY: a heap is free-threaded (its methods are called by any thread of
    // the title's), and it lives until all of them have looked.
    unsafe impl Sync for Shared<'_> {}
    impl Shared<'_> {
        fn handle(&self, index: u32) -> Option<u64> {
            self.0.handle(index)
        }
    }
    let heap = Shared(heap);
    // What a slot of the title's shows: its copy and range, and whether it is
    // an unordered view.
    let look = |index: u32| {
        // A draw's table's: the draw brings what it reads up to date.
        if host.copied_tabled(index, frame) {
            return None;
        }
        let view = host.mirror(index, heap.handle(index)?, true)?;
        // The copy the slot shows (`buffer_copy`), not the one the
        // view's range alone would find.
        let (shadow, address, len) = crate::descriptor::host::buffer_copy(&view)?;
        #[cfg(feature = "diagnostics")]
        if host.shows(index, &shadow.buffer) == Some(false) {
            HEAP_VIEWS_ELSEWHERE.fetch_add(1, Ordering::Relaxed);
        }
        let written = crate::picked::on() && view_kind(&view) == Some(ResourceType::Uav);
        Some(((shadow, address, address + len), written))
    };
    // Some slots at a time, on several threads.
    const AT_A_TIME: usize = 256;
    let parts: Vec<&[u32]> = slots.chunks(AT_A_TIME).collect();
    let found: Vec<std::sync::OnceLock<Vec<_>>> = parts.iter().map(|_| Default::default()).collect();
    crate::land::each_view(false, parts.len(), &|n| {
        let looked: Vec<_> = parts[n].iter().map(|&index| (index, look(index))).collect();
        // (After each piece while the device lives: a cheap query.)
        if alive
            && let Some(why) = crate::descriptor::host::device_removed()
            && !GONE.swap(true, Ordering::Relaxed)
        {
            let last: Vec<String> = parts[n]
                .iter()
                .filter_map(|&i| crate::descriptor::view(heap.handle(i)?).map(|v| format!("{i}: {v:?}")))
                .map(|s| s.chars().take(200).collect())
                .collect();
            tracing::error!(why, "the host device went while buffer views of the heap were made; perhaps one of: {last:#?}");
            crate::host::log_removal_context();
        }
        let _ = found[n].set(looked);
    });
    for (index, shows) in found.into_iter().filter_map(std::sync::OnceLock::into_inner).flatten() {
        let range = shows.map(|(range, written)| {
            if written {
                unordered.push(range.clone());
            }
            range
        });
        ranges.set(index, range);
    }
    if crate::picked::on() {
        crate::picked::shown(unordered, all);
    }
    #[cfg(feature = "diagnostics")]
    let looked = std::time::Instant::now();
    // Per host copy, the span its picked views cover.
    let mut spans: crate::host::ByCopy<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> = Default::default();
    let mut views = 0u32;
    for (shadow, from, to) in ranges.ranges() {
        views += 1;
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        let span = spans.entry(key).or_insert_with(|| (shadow.clone(), *from, *to));
        span.1 = span.1.min(*from);
        span.2 = span.2.max(*to);
    }
    let bytes: u64 = spans.values().map(|s| s.2 - s.1).sum();
    // Memory that is not write-watched is copied whole by every refresh.
    // It was once refreshed only every 120 frames (unmapped pages read
    // one by one cost 14 GB a frame in run 11, before they were
    // remembered): streamed geometry then drew from stale buffers, and
    // the garbage ended runs (the card stopped answering 14-90 frames
    // into the opening cutscene; with every frame it ran on). The pass
    // costs some 30 ms a frame this way.
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let every = *EVERY.get_or_init(|| {
        std::env::var("VICTORY_UNWATCHED_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
    });
    let mut found = Vec::with_capacity(spans.len());
    let mut next_unwatched = std::collections::HashMap::new();
    for (shadow, from, to) in spans.values() {
        if every > 1 && shadow.unwatched() {
            let id = shadow.id();
            if let Some(&(last, low, high)) = unwatched.get(&id)
                && frame >= last && frame - last < every && low <= *from && *to <= high
            {
                next_unwatched.insert(id, (last, low, high));
                continue;
            }
            next_unwatched.insert(id, (frame, *from, *to));
        }
        found.push((shadow.clone(), *from, (*to).min(*from + shadow.left(*from))));
    }
    *unwatched = next_unwatched;
    drop(kept);
    #[cfg(feature = "diagnostics")]
    let refreshed = std::time::Instant::now();
    crate::watch::look_at_holes();
    #[cfg(feature = "diagnostics")]
    for (sum, time) in HEAP_SPLIT.iter().zip([looked - started, refreshed - looked]) {
        sum.fetch_add(time.as_nanos() as u64, Ordering::Relaxed);
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    // The first pass and every growth of the set by half.
    let seen = LOGGED.load(Ordering::Relaxed);
    if u64::from(views) > seen + seen / 2 {
        LOGGED.store(u64::from(views), Ordering::Relaxed);
        tracing::info!(
            slots = slots.len(),
            views,
            copies = spans.len(),
            mb = bytes >> 20,
            ms = started.elapsed().as_millis(),
            "buffer views of the heap made ready for shaders that pick them by index"
        );
    }
    found
}

/// Whether the stages of one draw share a table they both need made afresh
/// (`VICTORY_TABLE_STAGES=1`; off by default: each stage makes its own, as
/// before). `alt`: twenty seconds with, twenty without, in turn, each
/// change said in the log; `alt40`: forty and forty, so a run can carry
/// this switch and another in turn and the two can be told apart.
///
/// A draw's stages read the same root tables, and each asked for slots up
/// to the last entry it reads itself. The count is in the key of
/// [`State::fresh_tables`], so a mesh stage and a pixel stage that read one
/// table to different depths made it twice. With this both ask for the
/// larger count ([`staged_slots`]), the second finds the first one's table,
/// and one table is made.
fn table_stages() -> bool {
    /// 0 off, 1 on, 2 in turn every 20 s, 3 in turn every 40 s.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| {
        table_stages_mode(std::env::var("VICTORY_TABLE_STAGES").ok().as_deref())
    }) {
        0 => false,
        1 => true,
        mode => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let period = if mode == 3 { 40 } else { 20 };
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / period) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "the stages of a draw share the tables they need made afresh from now (VICTORY_TABLE_STAGES=alt): {}",
                    if on { "ON" } else { "off, each stage makes its own as usual" }
                );
            }
            on
        }
    }
}

/// [`table_stages`]: `None` and anything but `1`, `alt` and `alt40` are off.
fn table_stages_mode(value: Option<&str>) -> u8 {
    match value {
        Some("1") => 1,
        Some("alt") => 2,
        Some("alt40") => 3,
        _ => 0,
    }
}

/// One stage's read of root table `slot` folded into `spans`:
/// (slot, slots read by a stage that also reads a constant there, slots any
/// stage reads).
fn fold_stage_span(spans: &mut Vec<(u8, u32, u32)>, slot: u8, has_cbv: bool, max_dword: u32) {
    let n = max_dword / 8 + 1;
    let with = if has_cbv { n } else { 0 };
    match spans.iter_mut().find(|s| s.0 == slot) {
        Some(s) => {
            s.1 = s.1.max(with);
            s.2 = s.2.max(n);
        }
        None => spans.push((slot, with, n)),
    }
}

/// Per root table the stages of `p` read: [`fold_stage_span`].
fn stage_spans(p: &HostPipeline) -> Vec<(u8, u32, u32)> {
    let mut spans = Vec::new();
    for stage in &p.stages {
        let plan = &stage.translated().tables;
        for group in &plan.groups {
            fold_stage_span(&mut spans, group.slot, group.has_cbv, group.max_dword);
        }
    }
    spans
}

/// With [`table_stages`]: `n` slots, or more, so the stages of one draw that
/// make this table afresh ask for the same count and the second finds the
/// first one's table. `span` is (slots read by a stage that also reads a
/// constant there, slots any stage reads); `None` when the switch is off.
/// Past the host's own slots every stage shares the whole read, because
/// each of them makes the table afresh.
fn staged_slots(n: u32, span: Option<(u32, u32)>, title_at: Option<u32>, host_count: u32) -> u32 {
    let Some((with_constants, all)) = span else {
        return n;
    };
    let past = title_at.is_some_and(|at| u64::from(at) + u64::from(all) > u64::from(host_count));
    n.max(if past { all } else { with_constants })
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        self.topology_set = true;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    /// A host compute library bound its own pipeline, root signature and heaps.
    pub(crate) fn external_compute(&mut self) {
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_heaps = [usize::MAX; 2];
        self.host_heaps_set = false;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once, unless it has
    /// the same ones already: the title sets the same targets again and
    /// again, each time a host view per target and a call of the driver
    /// (25,662 times in a frame of another fork). `VICTORY_BIND_CACHE=0`
    /// sets them every time.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let records: Vec<_> = handles.iter().map(|&h| crate::descriptor::rtv_record(h)).collect();
        let depth_record = crate::descriptor::dsv_record(depth);
        if targets_kept() {
            self.target_views.clear();
            self.target_views.extend(records.iter().map(|r| r.map(|r| r.snapshot())));
            self.depth_view = depth_record.map(|r| r.snapshot());
        }
        self.record_targets(list, handles, &records, depth, depth_record);
    }

    /// [`State::set_targets`] from what the handles' descriptors hold
    /// (`records`, `depth_record`): as they are now, or, at a replay with
    /// [`targets_kept`], as they were when the title set the targets.
    fn record_targets(
        &mut self,
        list: &HostList,
        handles: &[u64],
        records: &[Option<crate::descriptor::RtvRecord>],
        depth: u64,
        depth_record: Option<crate::descriptor::RtvRecord>,
    ) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.target_handles.clear();
        self.target_handles.extend_from_slice(handles);
        self.depth_handle = depth;
        self.targets_set = true;
        self.targets.clear();
        let mut formats = [0u32; 9];
        let mut views = Vec::with_capacity(handles.len());
        let mut keys = Vec::with_capacity(handles.len());
        for (&h, record) in handles.iter().zip(records) {
            let Some(record) = *record else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                // A typed null view (or a target without a host copy) still
                // takes its slot: left out, the next live target would move
                // into it and the pixel shader's outputs reach the wrong
                // targets.
                // shortcut: a view without a description or a format has no
                // null view to stand for it and is left out, as before.
                let null = record
                    .desc()
                    .filter(|d| d.Format.0 != 0 && d.ViewDimension.0 != 0);
                if let Some(desc) = null {
                    if views.len() < 8 {
                        formats[views.len()] = desc.Format.0 as u32;
                    }
                    views.push((None, Some(desc)));
                    keys.push((0, record.key()));
                }
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            let desc = record.desc();
            let raw = texture.as_raw() as usize;
            if views.len() < 8 {
                formats[views.len()] = bound_format(texture, desc.as_ref().map(|d| d.Format));
            }
            views.push((Some(texture), desc));
            keys.push((raw, record.key()));
        }
        self.depth = 0;
        self.depth_state = DEPTH_WRITE;
        let dsv = depth_record.and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            let desc = record.depth_desc();
            self.depth_state = depth_state(texture, desc.as_ref());
            formats[8] = bound_format(texture, desc.as_ref().map(|d| d.Format));
            if crate::resource::depth_and_unordered(record.resource) {
                crate::resource::DEPTH_UNORDERED[1].fetch_add(1, Ordering::Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some((texture, record))
        });
        self.target_formats = formats;
        let now = (keys, dsv.as_ref().map(|(texture, record)| (texture.as_raw() as usize, record.key())));
        static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *CACHE.get_or_init(|| std::env::var("VICTORY_BIND_CACHE").map_or(true, |v| v != "0"))
            && self.host_targets.as_ref() == Some(&now)
        {
            return;
        }
        self.host_targets = Some(now);
        let rtvs: Vec<_> = views.iter().map(|(texture, desc)| host.rtv(*texture, desc.as_ref())).collect();
        let dsv = dsv.map(|(texture, record)| host.dsv(texture, record.depth_desc().as_ref()));
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// Viewports, scissors, the blend factor and the stencil reference: set
    /// on the host list at once, and kept for [`State::replay`].
    pub(crate) fn set_viewports(&mut self, list: &HostList, viewports: &[D3D12_VIEWPORT]) {
        #[cfg(feature = "diagnostics")]
        if let Some(v) = viewports.first() {
            crate::host::spans::viewport(v.Width as u32, v.Height as u32);
        }
        self.viewports.clear();
        self.viewports.extend_from_slice(viewports);
        crate::draw::viewports(list, viewports);
    }

    pub(crate) fn set_scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.scissors.clear();
        self.scissors.extend_from_slice(rects);
        if scissor_test() {
            // The experiment: every rectangle the title sets is widened to
            // the largest target there is, so nothing is cut by one. What
            // the listing shows (`op_text`) stays the title's own.
            let full = RECT {
                left: 0,
                top: 0,
                right: 16384,
                bottom: 16384,
            };
            crate::draw::scissors(list, &vec![full; rects.len()]);
            return;
        }
        crate::draw::scissors(list, rects);
    }

    pub(crate) fn set_blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        self.blend = factor.copied();
        crate::draw::blend_factor(list, factor);
    }

    pub(crate) fn set_stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.stencil = Some(reference);
        crate::draw::stencil_ref(list, reference);
    }

    pub(crate) fn set_depth_bounds(&mut self, list: &HostList, min: f32, max: f32) {
        self.depth_bounds = Some([min, max]);
        crate::draw::depth_bounds(list, min, max);
    }

    /// `list` is a new host list that continues the title's list (it was
    /// cut at a wait for a memory value, `crate::list::cut`): a new host
    /// list has nothing set, so what the title set so far is set again.
    /// The pipeline, the root signatures and the heaps are set by the next
    /// draw or dispatch (`bind`, which is told here that the host list has
    /// none); root arguments, vertex and index buffers are set by every
    /// draw anyway. The dynamic descriptor slots and the tables made in
    /// them stay: the parts before the cut have not run yet, and they are
    /// given back when the title's list is reset.
    /// shortcut: the render target views are made again from the handles'
    /// descriptors as they are now, not as they were when the title set
    /// the targets (a descriptor rewritten in between names another view).
    pub(crate) fn replay(&mut self, list: &HostList) {
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_heaps = [0; 2];
        self.host_heaps_set = false;
        if self.topology_set {
            self.set_topology(list, self.topology);
        }
        self.host_targets = None;
        if self.targets_set {
            let handles = std::mem::take(&mut self.target_handles);
            if targets_kept() && self.target_views.len() == handles.len() {
                // (Fork 8f7d05d5a.) The views as the title set them, not
                // what the handles' descriptors hold by now.
                let records: Vec<_> = self.target_views.iter().map(|v| v.map(|v| v.restore())).collect();
                let depth_record = self.depth_view.map(|v| v.restore());
                self.record_targets(list, &handles, &records, self.depth_handle, depth_record);
            } else {
                self.set_targets(list, &handles, self.depth_handle);
            }
        }
        if !self.viewports.is_empty() {
            crate::draw::viewports(list, &self.viewports);
        }
        if !self.scissors.is_empty() {
            let rects = std::mem::take(&mut self.scissors);
            self.set_scissors(list, &rects);
        }
        if let Some(factor) = self.blend {
            crate::draw::blend_factor(list, Some(&factor));
        }
        if let Some(reference) = self.stencil {
            crate::draw::stencil_ref(list, reference);
        }
        if let Some([min, max]) = self.depth_bounds {
            crate::draw::depth_bounds(list, min, max);
        }
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            // The host's own ([`Self::host_word`]): no root argument.
            (_, RootType::Other(_)) => 0,
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// A user-data dword the host computes for a translated shader
    /// (`victory_shader::emit::USER_MEMORY_BASE`, `USER_VIEW_ADDRESS`).
    fn host_word(
        &self,
        compute: bool,
        kind: u32,
        slot: u8,
        dword: u32,
        memory: Option<&crate::memory::Ready>,
    ) -> u32 {
        use victory_shader::emit::{
            USER_DISPATCH_SPREAD, USER_DISPATCH_THREADS_X, USER_MEMORY_BASE, USER_ROOT_ADDRESS, USER_VIEW_ADDRESS,
            USER_VIEW_BIAS,
        };
        if kind == USER_MEMORY_BASE {
            return memory.map_or(0, |m| m.base);
        }
        // All of every group, unless the dispatch says otherwise. The spread
        // words stay 0 (as dispatched) until a spread dispatch overwrites them.
        if kind == USER_DISPATCH_THREADS_X || kind == USER_DISPATCH_SPREAD {
            return 0;
        }
        if kind == USER_ROOT_ADDRESS || kind == USER_ROOT_ADDRESS + 1 {
            return match self.roots_ref(compute).get(usize::from(slot)) {
                Some(&Root::View(address)) if kind == USER_ROOT_ADDRESS => address as u32,
                Some(&Root::View(address)) => (address >> 32) as u32,
                _ => 0,
            };
        }
        if kind == USER_VIEW_BIAS {
            // The view begins on 16 bytes of the copy the address finds
            // (`root_uav`, `root_srv_table`).
            // shortcut: found again here, not taken from the list's slot.
            return match self.roots_ref(compute).get(usize::from(slot)) {
                Some(&Root::View(address)) => crate::shadow::find(address, 1).map_or(0, |s| {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    ((s.gpu(address) - base) & 15) as u32
                }),
                _ => 0,
            };
        }
        if kind != USER_VIEW_ADDRESS && kind != USER_VIEW_ADDRESS + 1 {
            once!(kind, "a shader reads a host value this build does not know");
            return 0;
        }
        let address = self.view_address(compute, slot, dword);
        if address == 0 {
            // The first few say what lies there instead.
            static SAID: AtomicU64 = AtomicU64::new(0);
            if SAID.fetch_add(1, Ordering::Relaxed) < 8 {
                let view = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::Table(table)) => crate::descriptor::view(table + u64::from(dword) * 4),
                    _ => None,
                };
                tracing::warn!(slot, dword, ?view, "a shader takes the address of a buffer view that is not there: it reads 0");
            }
        }
        if kind == USER_VIEW_ADDRESS {
            address as u32
        } else {
            (address >> 32) as u32
        }
    }

    /// The title address the view at dword `dword` of the table in root
    /// slot `slot` starts at, 0 without one.
    fn view_address(&self, compute: bool, slot: u8, dword: u32) -> u64 {
        match self.roots_ref(compute).get(usize::from(slot)) {
            Some(&Root::Table(table)) => crate::descriptor::view(table + u64::from(dword) * 4)
                .and_then(|view| crate::descriptor::host::view_address(&view))
                .unwrap_or(0),
            _ => 0,
        }
    }

    /// For a shader that reads title memory by address
    /// ([`crate::memory`]): its directory, fault notes and view slots,
    /// with the pages made ready of the root views the list has set and of
    /// the views whose addresses it takes (`t`: the ray-tracing structures
    /// it walks).
    fn memory_ready(
        &self,
        list: &HostList,
        t: &crate::pipeline::Translated,
        compute: bool,
        use_: crate::memory::Use,
    ) -> Option<crate::memory::Ready> {
        let heap = match self.heaps[0] {
            0 => None,
            h => crate::descriptor::heap(h as This).host(),
        }?;
        let (mut addresses, mut constants) = (Vec::new(), Vec::new());
        // (What it may write: its root views of unordered access.)
        let written: Vec<u64> = t
            .used
            .iter()
            .filter(|u| u.kind == ResourceType::Uav && u.root == RootType::Descriptor)
            .filter_map(|u| match self.roots_ref(compute).get(usize::from(u.slot)) {
                Some(&Root::View(address)) => Some(address),
                _ => None,
            })
            .collect();
        for root in self.roots_ref(compute) {
            match root {
                // (Of a program that only walks structures, the structures'
                // views below: its root views are constants and tables.)
                Root::View(address) if use_ != crate::memory::Use::Reads => addresses.push(*address),
                Root::Constants(values) => constants.extend_from_slice(values),
                _ => {}
            }
        }
        for u in &t.host.user_data {
            if u.root == RootType::Other(victory_shader::emit::USER_VIEW_ADDRESS) {
                addresses.push(self.view_address(compute, u.slot, u.dword));
            }
        }
        crate::memory::ready(list, heap, &addresses, &written, &constants, use_, t.threads)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline_for(self.pipeline as This, &self.target_formats) else {
            return false;
        };
        #[cfg(feature = "diagnostics")]
        crate::draw_rank::note(&p.name);
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            return false;
        }
        if p.rects && self.topology != RECTANGLES {
            // Its geometry stage would turn each triangle into a rectangle.
            once!("a rectangle pipeline drawn with another topology");
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let found = {
                let _ask = crate::readkinds::asking(crate::readkinds::VERTEX);
                crate::shadow::find_any(v.address, u64::from(v.size))
            };
            let Some(shadow) = found else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            shadow.bound(crate::shadow::BOUND_VERTICES, u64::from(v.size));
            shadow.read_as(crate::readkinds::VERTEX, u64::from(v.size).min(shadow.left(v.address)));
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size.min(u32::try_from(shadow.left(v.address)).unwrap_or(u32::MAX)),
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let found = {
                let _ask = crate::readkinds::asking(crate::readkinds::INDEX);
                crate::shadow::find_any(v.address, u64::from(v.size))
            };
            let Some(shadow) = found else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            shadow.bound(crate::shadow::BOUND_INDICES, u64::from(v.size));
            shadow.read_as(crate::readkinds::INDEX, u64::from(v.size).min(shadow.left(v.address)));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size.min(u32::try_from(shadow.left(v.address)).unwrap_or(u32::MAX)),
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(self.depth as This, false, true);
            // A view that writes neither plane keeps the texture readable:
            // the depth read state, together with the shader read states
            // if this draw's shaders read the same texture (the write state
            // put it out of the state those views need, and a move to it
            // and back for every such draw).
            let state = if self.depth_state == DEPTH_READ {
                let (tested, read) = TESTED_DEPTH.get();
                if read && tested == self.depth {
                    DEPTH_READ | SHADER_RESOURCE
                } else {
                    DEPTH_READ
                }
            } else {
                DEPTH_WRITE
            };
            crate::list::ensure_state(list, self.depth as This, state);
        }
        if p.stages.iter().any(|s| s.shader.ambient == Some(crate::ambient::Role::Sky))
            && let Some(&sky) = self.targets.first()
        {
            crate::ambient::sky_drawn(sky as This);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a root SRV it reads
    /// could not be given a view.
    pub(crate) fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        let l = &list.list;
        // Textures picked by index, before anything asks for them.
        self.early_textures(list, p, compute);
        if crate::host::walls_trace() {
            self.walls_census(p, compute);
        }
        // A list without a view heap: one of the title's for the views the
        // host itself puts into heap slots.
        if self.heaps[0] == 0
            && p.stages.iter().any(|s| !s.root_uavs.is_empty() || !s.root_srvs.is_empty())
            && let Some(heap) = crate::descriptor::lent_heap()
        {
            once!("a dispatch without a view heap that needs one gets the title's");
            self.heaps[0] = heap;
        }
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them, with both a view heap and a sampler heap bound.
        // (`VICTORY_FALLBACK_HEAPS=1`, codec-d/cool_fox 641835a4: a host
        // heap of one slot for one the title did not set; else only those it
        // set, as before.)
        let fallback = crate::host::fallback_heaps();
        if (fallback && !self.host_heaps_set) || self.host_heaps != self.heaps {
            let mut view_heap: Option<ID3D12DescriptorHeap> = None;
            let mut sampler_heap: Option<ID3D12DescriptorHeap> = None;
            for &h in &self.heaps {
                if h == 0 {
                    continue;
                }
                let title_heap = crate::descriptor::heap(h as This);
                if let Some(host_heap) = title_heap.host() {
                    if title_heap.samplers() {
                        sampler_heap = Some(host_heap.heap.clone());
                    } else {
                        view_heap = Some(host_heap.heap.clone());
                    }
                }
            }
            if fallback && view_heap.is_none() {
                view_heap = crate::host::host().and_then(|h| h.fallback_view_heap().cloned());
            }
            if fallback && sampler_heap.is_none() {
                sampler_heap = crate::host::host().and_then(|h| h.fallback_sampler_heap().cloned());
            }
            let mut heaps: Vec<Option<ID3D12DescriptorHeap>> = Vec::with_capacity(2);
            if let Some(h) = view_heap {
                list.keep(&h);
                heaps.push(Some(h));
            }
            if let Some(h) = sampler_heap {
                list.keep(&h);
                heaps.push(Some(h));
            }
            if !heaps.is_empty() {
                // SAFETY: an open list and shader-visible heaps.
                unsafe { l.SetDescriptorHeaps(&heaps) };
                crate::host::rec::HEAP_BINDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            self.host_heaps = self.heaps;
            self.host_heaps_set = true;
            self.host_roots = [0; 2];
        }
        // The depth target a draw only tests, for the views of it that the
        // draw's shaders read (`readable`).
        let tested = if !compute && self.depth_state == DEPTH_READ {
            self.depth
        } else {
            0
        };
        TESTED_DEPTH.set((tested, false));
        // With `table_stages`: what the stages of this draw read of each
        // table, and the tables its earlier stages made afresh (address,
        // slots). Empty when the switch is off, and each stage asks for
        // its own count as before.
        let spans = if table_stages() { stage_spans(p) } else { Vec::new() };
        let mut staged: Vec<(u64, u32)> = Vec::new();
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            // The title may release the pipeline before it submits the list.
            list.bind_pipeline(&p.name, &p.pso, &p.root);
        }
        for stage in &p.stages {
            let t = stage.translated();
            STORES.set(t.stores);
            if t.stores && crate::picked::on() && crate::picked::picks(&stage.shader, &t.picks) {
                crate::picked::runs_in(list);
            }
            let made = self.tables(list, &t.tables, &t.host.half_views, compute, &spans, &mut staged);
            STORES.set(true);
            if made.iter().any(|m| m.1 == u32::MAX) {
                once!("a draw whose descriptor table found no free slots is dropped");
                #[cfg(feature = "diagnostics")]
                crate::pipeline::note_why(format!("no free dynamic slots for a table of {}", p.name));
                return false;
            }
            if stage.shader.ambient.is_some() || stage.shader.probe.is_some() {
                self.ambient_tables.clear();
                self.ambient_tables.extend(made.iter().copied());
            }
            // What the shader stores through root views and raw addresses is
            // copied back into title memory afterwards.
            let writes = (t.stores || !write_skip())
                && !crate::pipeline::builder_quarantined(self.pipeline as This);
            // Title memory by address: what the shader finds it by.
            let memory = if t.host.root_srvs.contains(&victory_shader::emit::SLOT_MEMORY) {
                // Only through a store by address: what it stores through
                // views is declared with them.
                let use_ = if crate::pipeline::is_builder(self.pipeline as This) || writes && t.stores_by_address {
                    crate::memory::Use::Builds { back: writes }
                } else {
                    crate::memory::Use::Reads
                };
                let Some(ready) = self.memory_ready(list, t, compute, use_) else {
                    once!("a shader that reads title memory by address without a view heap to hold its views: the draw is dropped");
                    return false;
                };
                Some(ready)
            } else {
                None
            };
            if let Some((param, dwords)) = stage.user_data {
                // (A root signature holds 64 dwords.)
                let mut room = [0u32; 64];
                let mut more = Vec::new();
                let values: &mut [u32] = match room.get_mut(..dwords as usize) {
                    Some(values) => values,
                    None => {
                        more.resize(dwords as usize, 0);
                        &mut more
                    }
                };
                for u in &t.host.user_data {
                    if let RootType::Other(kind) = u.root {
                        values[u.index as usize] =
                            self.host_word(compute, kind, u.slot, u.dword, memory.as_ref());
                        continue;
                    }
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                #[cfg(feature = "diagnostics")]
                {
                    self.user.clear();
                    self.user.extend_from_slice(values);
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    if srv && slot == victory_shader::emit::SLOT_MEMORY {
                        // Ours: the directory of title memory by address.
                        if let Some(m) = &memory {
                            // SAFETY: an open list; the directory lives as
                            // long as the process and the list holds it.
                            unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, m.directory);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, m.directory);
                                }
                            }
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_STRIDES {
                        // Ours, not the title's: the shader picks buffers
                        // from the view heap by an index it computes.
                        match self.heap_buffers(list) {
                            // SAFETY: an open list; the table lives as long
                            // as the heap the list holds.
                            Some(gpu) => unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, gpu);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, gpu);
                                }
                            },
                            None => once!("a shader picks buffers from a view heap that is not set"),
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_TABLE_STRIDES {
                        // Ours: the shader inspects typed view kinds for its
                        // descriptor tables only (already mirrored by `tables`),
                        // so bind the strides buffer without sweeping the heap.
                        match self.heap_strides(list) {
                            // SAFETY: an open list; the table lives as long
                            // as the heap the list holds.
                            Some(gpu) => unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, gpu);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, gpu);
                                }
                            },
                            None => once!("a shader inspects table view kinds without a view heap set"),
                        }
                        continue;
                    }
                    let tabled = srv && crate::pipeline::host::bounded_root_srvs();
                    let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        // A table left unset is undefined: a null view.
                        if tabled && !self.root_srv_table(list, None, param, compute, false) {
                            return false;
                        }
                        continue;
                    };
                    // The shader adds the bytes a raw view begins early by.
                    let biased = srv
                        && t.host.user_data.iter().any(|u| {
                            u.slot == slot
                                && u.root == RootType::Other(victory_shader::emit::USER_VIEW_BIAS)
                        });
                    if tabled {
                        if !self.root_srv_table(list, Some(address), param, compute, biased) {
                            return false;
                        }
                        continue;
                    }
                    let Some(mut gpu) = root_view(list, address, srv) else {
                        continue;
                    };
                    if biased {
                        gpu -= u64::from(self.host_word(
                            compute,
                            victory_shader::emit::USER_VIEW_BIAS,
                            slot,
                            0,
                            None,
                        ));
                    }
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
            for &(slot, param) in &stage.root_uavs {
                if slot == victory_shader::emit::SLOT_FAULTS {
                    // Ours: where shaders note the pages they found empty.
                    if let Some(m) = &memory {
                        // SAFETY: an open list; the buffer lives as long as
                        // the process and the list holds it.
                        unsafe {
                            if compute {
                                l.SetComputeRootUnorderedAccessView(param, m.faults);
                            } else {
                                l.SetGraphicsRootUnorderedAccessView(param, m.faults);
                            }
                        }
                    }
                    continue;
                }
                let address = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::View(address)) => Some(address),
                    _ => {
                        once!(slot, "a shader writes through a root view the title did not set");
                        None
                    }
                };
                if !self.root_uav(list, address, param, compute, writes) {
                    return false;
                }
            }
        }
        // Depth textures it writes through unordered access: their colour
        // twins hold what they lack from here on.
        list.twins_written();
        true
    }

    /// Binds a root UAV: a raw unordered access view of the title buffer's
    /// host copy from `address` on, as a one-view table (a null view
    /// without an address, without a host copy shaders can write, or at an
    /// address not on 16 bytes), or as a root descriptor with
    /// `VICTORY_ROOT_SRV_DIRECT=1` (the dispatch is dropped then if there
    /// is nothing to bind). With `writes`, what the shader stored is copied
    /// back into title memory when the list has run. False if the draw
    /// must be dropped.
    /// shortcut: a root view has no size: the view, the refresh and the
    /// copy back cover all that is left of the title's buffer
    /// (`VICTORY_ROOT_UAV_MB` sets a limit for all three).
    fn root_uav(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        writes: bool,
    ) -> bool {
        const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
        static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let limit = *LIMIT.get_or_init(|| {
            std::env::var("VICTORY_ROOT_UAV_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map_or(u64::MAX, |mb| mb << 20)
        });
        let find = |a: u64| {
            let _ask = crate::readkinds::asking(crate::readkinds::ROOT_SRV);
            let s = crate::shadow::find(a, 1)?;
            // SAFETY: a live buffer.
            let flags = unsafe { s.buffer.GetDesc() }.Flags;
            if flags.0 & UNORDERED == 0 {
                once!("a root UAV of a buffer whose host copy shaders cannot write");
                return None;
            }
            Some(s)
        };
        let ready = |a: u64, s: &std::sync::Arc<crate::shadow::Shadow>| {
            let len = s.left(a).min(limit);
            s.refresh(list, a, len);
            if writes {
                #[cfg(feature = "diagnostics")]
                crate::host::wide_write(a, len, self.pipeline);
                s.stored(crate::readkinds::STORE_ROOT, a, len);
                s.gpu_writes(list, a, len);
            }
        };
        if !crate::pipeline::host::bounded_root_srvs() {
            let Some((a, s)) = address.and_then(|a| Some((a, find(a)?))) else {
                once!("a root UAV with nothing to bind: the draw is dropped");
                return false;
            };
            ready(a, &s);
            // From the 16 bytes below: the shader adds the rest.
            // SAFETY: a live buffer.
            let base = unsafe { s.buffer.GetGPUVirtualAddress() };
            let gpu = base + ((s.gpu(a) - base) & !15);
            s.bound(crate::shadow::BOUND_ROOT_VIEW, 0);
            s.read_as(crate::readkinds::ROOT_SRV, s.left(a).min(crate::readkinds::UNSIZED));
            // SAFETY: an open list; the address is in a live buffer.
            unsafe {
                if compute {
                    list.list.SetComputeRootUnorderedAccessView(param, gpu);
                } else {
                    list.list.SetGraphicsRootUnorderedAccessView(param, gpu);
                }
            }
            return true;
        }
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root UAV without a bound view heap: the draw is dropped");
            return false;
        };
        let key = (std::ptr::from_ref(heap) as usize, address.unwrap_or(0));
        let (index, shadow) = match self.root_uav_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| find(a).map(|s| (a, s)));
                let Some(index) = host.dynamic(1).or_else(|| {
                    crate::host::settle_for_fence();
                    host.dynamic(1)
                }).or_else(|| host.dynamic_waited(1)) else {
                    #[cfg(feature = "diagnostics")]
                    crate::pipeline::note_why("no free dynamic slot for the one-view table of a root view".to_owned());
                    return false;
                };
                self.dynamic.push((key.0, host.id(), index, 1));
                if let Some((a, s)) = &shadow {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    let offset = s.gpu(*a) - base;
                    s.bound(crate::shadow::BOUND_ROOT_VIEW, 0);
                    s.read_as(crate::readkinds::ROOT_SRV, s.left(*a).min(crate::readkinds::UNSIZED));
                    // Raw views start on 16 bytes: from the 16 bytes
                    // below, and the shader adds the rest to its offsets
                    // (`victory_shader::emit::USER_VIEW_BIAS`). (An address
                    // not on 16 bytes got a null view at first: the
                    // builder's counters are at such addresses.)
                    let bias = offset & 15;
                    host.raw_uav_at(index, &s.buffer, offset - bias, s.left(*a).min(limit) + bias);
                }
                if shadow.is_none() {
                    host.null_raw_uav_at(index);
                }
                self.root_uav_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                (index, shadow)
            }
        };
        if let Some((a, s)) = &shadow {
            ready(*a, s);
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Binds a root SRV as a one-view table
    /// ([`crate::pipeline::host::bounded_root_srvs`]): a raw view of the
    /// title buffer's host copy from `address` to the copy's end, or a null
    /// view (no address, no host copy, or an address not on 16 bytes). One
    /// dynamic slot per address per list, given back with the list's
    /// others. False if no slot could be had or no view heap is bound.
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        biased: bool,
    ) -> bool {
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root SRV without a bound view heap: the draw is dropped");
            return false;
        };
        // A view that begins early is another view than one that does not
        // (title addresses stay below bit 63).
        let key = (
            std::ptr::from_ref(heap) as usize,
            address.unwrap_or(0) | u64::from(biased) << 63,
        );
        let (index, shadow) = match self.root_srv_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = {
                    let _ask = crate::readkinds::asking(crate::readkinds::ROOT_SRV);
                    address.and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)))
                };
                let Some(index) = host.dynamic(1).or_else(|| {
                    crate::host::settle_for_fence();
                    host.dynamic(1)
                }).or_else(|| host.dynamic_waited(1)) else {
                    #[cfg(feature = "diagnostics")]
                    crate::pipeline::note_why("no free dynamic slot for the one-view table of a root view".to_owned());
                    return false;
                };
                self.dynamic.push((key.0, host.id(), index, 1));
                self.root_srv_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                match &shadow {
                    Some((a, s)) => {
                        // SAFETY: a live buffer.
                        let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                        let offset = s.gpu(*a) - base;
                        s.bound(crate::shadow::BOUND_ROOT_VIEW, 0);
                        s.read_as(crate::readkinds::ROOT_SRV, s.left(*a).min(crate::readkinds::UNSIZED));
                        if biased {
                            // From the 16 bytes below; the shader adds the
                            // rest (`victory_shader::emit::USER_VIEW_BIAS`).
                            let bias = offset & 15;
                            host.raw_at(index, &s.buffer, offset - bias, s.left(*a) + bias);
                        } else if offset.is_multiple_of(16) {
                            host.raw_at(index, &s.buffer, offset, s.left(*a));
                        } else {
                            // Raw views start on 16 bytes; one from the
                            // aligned start would shift every read.
                            once!(
                                address = format_args!("{a:#x}"),
                                "a root SRV not on 16 bytes reads zeros"
                            );
                            host.null_raw_at(index);
                        }
                    }
                    None => host.null_raw_at(index),
                }
                (index, shadow)
            }
        };
        // As for root descriptors (`root_view`): all that is left of the
        // buffer is brought up to date. The first 1 MiB only left what lay
        // past it stale (most mesh vertex streams reach further).
        if let Some((a, s)) = &shadow {
            s.refresh(list, *a, s.left(*a));
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// For a shader that picks buffers from the view heap by an index it
    /// computes (SM 6.6 dynamic resources): the GPU address of the heap's
    /// table of element strides, with every buffer view of the heap given
    /// its host slot and its memory brought up to date when `list` is
    /// submitted. Tables tell us which views a draw reads; nothing tells
    /// us which of the heap's a computed index names, so all of them are
    /// made ready, once per list.
    /// shortcut: the buffers are read, not written: what such a shader
    /// writes through a picked unordered access view stays in the host
    /// copy. A list refreshes them where it first runs such a shader;
    /// title writes recorded later in the same list are not seen.
    fn heap_buffers(&self, list: &HostList) -> Option<u64> {
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let gpu = host.strides_gpu()?;
        // One key per list, as textures have: no title resource is there.
        // (`VICTORY_SWEEP_PER_HEAP=1`, codec-d/cool_fox c64fc9b4: one per
        // heap's table of strides, whose GPU address is aligned; tagged so it
        // is not a title resource's.)
        const ONCE: usize = usize::MAX - 0x5e;
        let per_heap = sweep_per_heap();
        if !list.first_use(if per_heap { gpu as usize | 1 } else { ONCE }) {
            return Some(gpu);
        }
        // The list may outlive the title's heap object, which owns the table.
        if let Some(buffer) = host.strides_buffer() {
            list.keep(buffer);
        }
        // The pass itself once per frame: per list it ran up to 51 times a
        // frame over 81,000 views and 4,100 host copies, 18 s of every
        // frame (2026-11-24, run 11).
        // shortcut: what the title writes into such buffers between lists
        // of one frame is seen a frame late.
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        let frame = crate::frame::current();
        let last = if per_heap { &host.buffer_frame } else { &FRAME };
        if last.swap(frame, Ordering::Relaxed) == frame {
            return Some(gpu);
        }
        let started = std::time::Instant::now();
        let mut stage = crate::timeline::span("the heap pass: begun, or done here");
        if crate::host::aside::on() {
            let at = std::ptr::from_ref(heap) as usize;
            list.refreshes_coming(move || {
                crate::descriptor::with_heap(at, |heap| {
                    // The list retains the original stride buffer. A newly
                    // allocated heap at the same CPU address has another one.
                    heap.host()
                        .filter(|h| !per_heap || h.strides_gpu() == Some(gpu))
                        .map(|h| heap_buffer_pass(heap, h, frame))
                })
                    .flatten()
                    .unwrap_or_default()
            });
        } else {
            for (shadow, from, to) in heap_buffer_pass(heap, host, frame) {
                shadow.refresh(list, from, to - from);
            }
        }
        #[cfg(feature = "diagnostics")]
        let refreshed = std::time::Instant::now();
        stage.next("the heap pass: textures");
        // On a thread of its own (`crate::heap_worker`), or here.
        if !crate::heap_worker::kick(heap) {
            let _pass = crate::heap_worker::pass();
            let _ = self.heap_textures(list, heap, host, None);
        }
        stage.end();
        #[cfg(feature = "diagnostics")]
        HEAP_SPLIT[2].fetch_add(refreshed.elapsed().as_nanos() as u64, Ordering::Relaxed);
        HEAP_BUFFER_PASSES.fetch_add(1, Ordering::Relaxed);
        HEAP_BUFFER_MICROS.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        Some(gpu)
    }

    /// The GPU address of the bound view heap's stride/kind table without
    /// running the bindless heap buffer/texture sweep (for shaders that only
    /// inspect descriptors already mirrored via their descriptor tables).
    fn heap_strides(&self, list: &HostList) -> Option<u64> {
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let gpu = host.strides_gpu()?;
        // Tag bit 1 (distinct from `heap_buffers`'s `| 1` and aligned guest pointers).
        if list.first_use(gpu as usize | 2)
            && let Some(buffer) = host.strides_buffer()
        {
            list.keep(buffer);
        }
        Some(gpu)
    }

    /// Ensures textures claimed by background copy-queue uploads
    /// (`VICTORY_UPLOAD_AHEAD=1`) or deferred behind the `neutral()` gray
    /// stand-in (`VICTORY_SHOW_AFTER=1` / `shows_other`) have finished
    /// uploading and flipped to their real descriptors before a one-shot
    /// decoration bake draw samples them.
    fn flush_for_bake(
        &self,
        list: &HostList,
        host: &crate::descriptor::host::HostHeap,
        mut sync: Option<&mut BakeSync>,
    ) {
        static RESCUED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
        let drained = crate::texture::ahead::drain();
        let mut settled = false;
        let mut flipped = 0usize;
        for _ in 0..4 {
            let need_settle = host.needs_gpu_settle_for_bake();
            if need_settle {
                settled = true;
                crate::host::settle_for_fence();
            }
            flipped += host.texture_views_remade();
            flipped += host.texture_slots_flip();
            if !need_settle || !host.needs_gpu_settle_for_bake() {
                break;
            }
        }
        let pending = host.take_pending_list_flips();
        let mut promoted = 0usize;
        if !pending.is_empty() {
            let this_list = (std::ptr::from_ref(list) as usize, list.epoch());
            let sampled = UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 0,
                slot: 0,
            };
            for item in pending {
                let (origin, index, handle, key) = (item.origin, item.index(), item.handle(), item.key());
                if !host.slot_still_at(handle, key) {
                    continue;
                }
                let Some(view) = crate::descriptor::view(handle) else {
                    continue;
                };
                if let View::Placed(_, resource, desc) = &view {
                    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                    let len = crate::resource::placed_bytes(resource).unwrap_or(1);
                    if crate::texture::mapped_bytes(address as *const u8, len as usize) == 0 {
                        host.restore_pending_list_flip(item);
                        continue;
                    }
                }
                if sync.is_some() || origin != this_list {
                    let placed = crate::descriptor::host::placed_texture(&view);
                    if let Some(&View::Shader(resource, _)) = placed.as_ref().or(Some(&view)) {
                        let resource = resource as This;
                        let texture = crate::resource::host_resource(resource);
                        let title_fills = texture.is_some_and(crate::texture::title_written);
                        let first = match (&placed, &view) {
                            (Some(_), View::Placed(_, _, desc)) => crate::resource::first_mip(desc),
                            _ => 0,
                        };
                        if let (Some(sync), Some(copy)) = (sync.as_deref_mut(), crate::resource::host_copy(resource)) {
                            sync.keep.push(copy);
                        }
                        if title_fills
                            && !crate::texture::ahead::there(resource)
                            && (sync.is_some() || !crate::texture::unchanged(resource, first))
                        {
                            let was_staged = crate::texture::is_staged(resource);
                            let _ = crate::residency::take_staging_deferred();
                            readable(list, sampled.kind, &view);
                            let failed = crate::residency::take_staging_deferred() || crate::residency::upload_refused();
                            if !was_staged {
                                crate::texture::unstage(resource);
                            }
                            if failed {
                                host.restore_pending_list_flip(item);
                                continue;
                            }
                        }
                    }
                }
                if let Some(sync) = sync.as_deref_mut() {
                    sync.publish.push((index, handle, key));
                } else {
                    let _ = host.mirror(index, handle, true);
                }
                promoted += 1;
            }
        }
        if drained > 0 || settled || flipped > 0 || promoted > 0 {
            let n = RESCUED[0].fetch_add(1, Ordering::Relaxed) + 1;
            let all_drained = RESCUED[1].fetch_add(drained as u64, Ordering::Relaxed) + drained as u64;
            let all_flipped = RESCUED[2].fetch_add(flipped as u64, Ordering::Relaxed) + flipped as u64;
            let all_promoted = RESCUED[3].fetch_add(promoted as u64, Ordering::Relaxed) + promoted as u64;
            if n.is_power_of_two() {
                tracing::info!(
                    frame = crate::frame::current(),
                    flushes = n,
                    drained_ahead = all_drained,
                    flipped = all_flipped,
                    promoted_from_lists = all_promoted,
                    settled,
                    "flushed deferred texture stand-ins before decoration bake"
                );
            }
        }
    }

    /// Whether a graphics draw is a one-shot texture compositor / decoration
    /// bake whose bindless texture inputs must be flushed and up to date
    /// before recording the draw. Matches known bake pixel/geometry shaders
    /// and structurally detects depth-less 4+ MRT square atlas bakes
    /// (`>= 512x512`) so unseen NPC/head/clothing shader permutations never
    /// bake `neutral()` or `null` stand-ins into permanent ped textures.
    fn is_bake_draw(&self, p: &HostPipeline) -> bool {
        if p.stages.iter().any(|s| {
            let h = s.shader.hash.as_str();
            BAKE_PIXEL_SHADERS.contains(&h) || BAKE_GEOMETRY_SHADERS.contains(&h)
        }) {
            return true;
        }
        self.depth == 0
            && self.targets.len() >= 4
            && self
                .targets
                .first()
                .and_then(|&t| crate::resource::texture_kind(t as This))
                .is_some_and(|(w, h, _)| w >= 512 && w == u64::from(h))
    }

    /// Runs the pass that uploads textures picked by index
    /// ([`State::heap_textures`]) where nothing has asked for it yet
    /// ([`early_budgets`]): once a frame for a few milliseconds, at the
    /// frame's first draw with a view heap set, and with a larger budget
    /// of its own before every draw whose pixel shader is one of the
    /// title's bake programs. For decoration bakes, uploads run synchronously
    /// on a dedicated internal command list and descriptors are published only
    /// after the GPU copy finishes so concurrent `VICTORY_THIN=1` lists can
    /// never submit a bake draw ahead of its texture uploads.
    fn early_textures(&self, list: &HostList, p: &HostPipeline, compute: bool) {
        let (early_ms, bake_ms) = early_budgets();
        if compute || (early_ms == 0 && bake_ms == 0) {
            return;
        }
        let is_bake = bake_ms > 0 && self.is_bake_draw(p);
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        // The milliseconds this frame's bake draws have taken so far: a
        // frame has several, and together they get `bake_ms`.
        static BAKED_MS: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        let first = FRAME.swap(frame, Ordering::Relaxed) != frame;
        if first {
            BAKED_MS.store(0, Ordering::Relaxed);
        }
        if !is_bake && !(first && early_ms > 0) {
            return;
        }
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return;
        };
        let Some(host) = heap.host() else {
            return;
        };
        if is_bake {
            // These cannot wait for a later submission: the worker, if there
            // is one, stops where it is (`crate::heap_worker::pass`).
            let _pass = crate::heap_worker::pass();
            let began = std::time::Instant::now();
            let mut bake_slot = BAKE_UPLOAD.lock().unwrap_or_else(|e| e.into_inner());
            let gpu = crate::host::host();
            if bake_slot.is_none()
                && let Some(gpu) = gpu
            {
                match gpu.new_list() {
                    Ok(upload_list) => *bake_slot = Some(BakeList(upload_list)),
                    Err(e) => tracing::warn!(error = %e, "no host list for synchronous bake uploads"),
                }
            }
            let exact = bake_exact();
            if let (Some(gpu), Some(BakeList(bake_list))) = (gpu, bake_slot.as_ref()) {
                let mut sync = BakeSync { publish: Vec::new(), keep: Vec::new() };
                crate::texture::BAKING.set(exact);
                self.flush_for_bake(bake_list, host, Some(&mut sync));
                let left = bake_ms.saturating_sub(BAKED_MS.load(Ordering::Relaxed));
                let run_pass = left > 0
                    && (first
                        || !exact
                        || {
                            let (writes, noted) = host.texture_news();
                            noted || BAKE_SERVED.load(Ordering::Relaxed) != writes + 1
                        });
                let mut served = false;
                if run_pass {
                    let n = EARLY_TEXTURE_PASSES[1].fetch_add(1, Ordering::Relaxed) + 1;
                    if n.is_power_of_two() {
                        let [slots, due, ..] = HEAP_TEXTURES_QUEUE.each_ref().map(|c| c.load(Ordering::Relaxed));
                        tracing::info!(
                            frame,
                            bakes = n,
                            queued_at_the_last_pass = slots,
                            due_at_the_last_pass = due,
                            uploaded_so_far = HEAP_TEXTURES.load(Ordering::Relaxed),
                            "a draw of the title's decoration bake: textures picked by index are uploaded synchronously first (VICTORY_BAKE_TEXTURES_MS=0: not)"
                        );
                    }
                    BAKE_SERVED.store(0, Ordering::Relaxed);
                    PASS_BUDGET.set(Some((left, 2 << 30, true)));
                    served = self.heap_textures(bake_list, heap, host, Some(&mut sync));
                    PASS_BUDGET.set(None);
                }
                crate::texture::BAKING.set(false);
                if bake_list.has_recorded_work() {
                    bake_list.close();
                    gpu.execute_sync_internal(bake_list);
                    bake_list.reset();
                } else if !sync.publish.is_empty() {
                    crate::host::settle_for_fence();
                }
                for (index, handle, key) in sync.publish {
                    let _ = host.publish_uploaded_slot(index, handle, key);
                }
                host.texture_views_remade();
                host.texture_slots_flip();
                drop(sync.keep);
                if run_pass && served && exact {
                    BAKE_SERVED.store(host.texture_news().0 + 1, Ordering::Relaxed);
                }
                BAKED_MS.fetch_add(began.elapsed().as_millis() as u64, Ordering::Relaxed);
                return;
            }
            // Fallback when no host GPU list can be created (e.g. headless tests).
            crate::texture::BAKING.set(exact);
            self.flush_for_bake(list, host, None);
            crate::texture::BAKING.set(false);
            let left = bake_ms.saturating_sub(BAKED_MS.load(Ordering::Relaxed));
            if left == 0 {
                return;
            }
            if !first && exact {
                let (writes, noted) = host.texture_news();
                if !noted && BAKE_SERVED.load(Ordering::Relaxed) == writes + 1 {
                    return;
                }
            }
            EARLY_TEXTURE_PASSES[1].fetch_add(1, Ordering::Relaxed);
            PASS_BUDGET.set(Some((left, 2 << 30, true)));
            crate::texture::BAKING.set(exact);
            let _ = self.heap_textures(list, heap, host, None);
            crate::texture::BAKING.set(false);
            PASS_BUDGET.set(None);
            BAKED_MS.fetch_add(began.elapsed().as_millis() as u64, Ordering::Relaxed);
            return;
        }
        let budget = if first && early_ms > 0 {
            (early_ms, 64 << 20)
        } else {
            return;
        };
        // What can wait for a later list is the worker's.
        if crate::heap_worker::kick(heap) {
            return;
        }
        let _pass = crate::heap_worker::pass();
        EARLY_TEXTURE_PASSES[0].fetch_add(1, Ordering::Relaxed);
        PASS_BUDGET.set(Some((budget.0, budget.1, false)));
        let _ = self.heap_textures(list, heap, host, None);
        PASS_BUDGET.set(None);
    }

    /// The queue's side of the card accounting line
    /// (`crate::resource::card_accounting`, which says what the line
    /// holds): the slots that are due, by what their texture has on the
    /// card. Off unless `VICTORY_CARD_ACCOUNTING=1`; then at the first pass
    /// that finds the card without room, and every five seconds after it.
    /// Nothing is made, uploaded or counted as used here. For 18,000 due
    /// slots: 18,000 looks at a title slot and at the texture it views, a
    /// lock each, some milliseconds of the thread that records the frame.
    fn card_accounting(
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        frame: u64,
        room: bool,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ACCOUNTING").is_some_and(|v| v == "1")) {
            return;
        }
        // When the next line is due (0: none yet, the card has had room).
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms().max(1);
        let next = NEXT.load(Ordering::Relaxed);
        // (With levels on demand the card may never be without room: the
        // lines then start with the first texture made that way.)
        let by_level = crate::tiles::STATS[4].load(Ordering::Relaxed) > 0;
        let due_now = if next == 0 { !room || by_level } else { now >= next };
        if !due_now || NEXT.compare_exchange(next, now + 5000, Ordering::Relaxed, Ordering::Relaxed).is_err() {
            return;
        }
        // Due slots; of them uploaded before; of them viewing nothing the
        // title fills; and the textures (each once) whose host copy is
        // alive, with their bytes on the card.
        let mut due = [0u64; 5];
        // Textures (each once) that never had a host copy, and ones whose
        // host copy was let go: where their memory is, and its bytes.
        let (mut never, mut gone) = (Vec::new(), Vec::new());
        let mut seen = std::collections::BTreeSet::new();
        for (index, uploaded) in host.texture_slots_all_due(frame) {
            due[0] += 1;
            due[1] += u64::from(uploaded);
            let state = match heap.handle(index).and_then(crate::descriptor::view) {
                Some(View::Placed(false, resource, desc)) if resource[0] != 1 => {
                    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                    match crate::resource::placed_known(address, &resource) {
                        Some(texture) => crate::resource::copy_state(texture),
                        // No title texture was made for it: the pass never
                        // got as far as making one.
                        None => Some((0, crate::resource::placed_bytes(&resource).unwrap_or(0), address)),
                    }
                }
                Some(View::Shader(object, _)) if crate::resource::title_fills(object as This) => {
                    crate::resource::copy_state(object as This)
                }
                _ => None,
            };
            let Some((state, bytes, memory)) = state else {
                due[2] += 1;
                continue;
            };
            if !seen.insert(memory) {
                continue;
            }
            match state {
                2 => {
                    due[3] += 1;
                    due[4] += bytes;
                }
                1 => gone.push((memory, bytes)),
                _ => never.push((memory, bytes)),
            }
        }
        crate::resource::card_accounting(room, due, never, gone);
    }

    /// [`State::heap_textures`] for the thread of `crate::heap_worker`, into
    /// `list`, a list of its own.
    pub(crate) fn heap_textures_ahead(
        &self,
        list: &HostList,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
    ) {
        let _ = self.heap_textures(list, heap, host, None);
    }

    /// The texture side of [`State::heap_buffers`]: textures the title fills
    /// itself (streamed pages, layout-library copies) whose views sit in the
    /// view heap are created (placed textures have no host texture until
    /// something reads them) and uploaded, as a table read would do, for
    /// shaders that pick them by a computed index. Within a budget of bytes
    /// per frame and in all; the rest waits for later lists and frames.
    /// The most recently written slots first (`TextureQueue::due`); slots
    /// draws read through tables are not in the queue.
    /// shortcut: uploaded once, and again only when the title rewrites the
    /// slot. A placed texture not all mapped yet is uploaded with what is
    /// there and looked at again (1, 2, 4, ... frames later, at most about
    /// a second): uploaded again when more of it is mapped. Pages mapped
    /// in place of others, and contents changed with no descriptor write,
    /// are missed (hook the page-mapping events for those).
    /// `VICTORY_HEAP_TEXTURES=0` turns it off.
    /// On a thread of its own (`crate::heap_worker`) the pass records into
    /// a list of that thread's, a slot new to its texture shows it when
    /// that list has run, and the pass ends when a recording thread wants
    /// it or the list holds enough.
    fn heap_textures(
        &self,
        list: &HostList,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        mut sync: Option<&mut BakeSync>,
    ) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_TEXTURES").is_none_or(|v| v != "0")) {
            return true;
        }
        let memory = crate::host::host().and_then(|h| h.video_memory());
        // `VICTORY_TEXTURE_STAGE_MB` sets it: a backlog of tens of thousands of
        // textures (a scene's first minutes) then drains in fewer frames.
        static PER_FRAME_MB: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
        let asked = *PER_FRAME_MB.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_STAGE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok())
        });
        let per_frame = match asked {
            Some(mb) => mb << 20,
            None => {
                let base = staging_budget_per_frame(memory.map(|[budget, ..]| budget));
                // (`VICTORY_RESIDENCY=1` with 32 GiB of RAM or less: 128 MiB.)
                if crate::residency::limited_host_ram() { base.min(128 << 20) } else { base }
            }
        };
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        static SPENT: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) != frame {
            SPENT.store(0, Ordering::Relaxed);
        }
        // No more of them are made while the card has no room
        // (`resource::room`): by what is alive now, not by what was ever
        // staged (4 GB in all, after which later textures stayed empty for
        // the run). The ones nothing names any more are let go, which
        // makes room again: so the slots are still looked at (the pass
        // returned here, and a slot whose texture had been streamed out
        // kept its host copy for as long as the card was full).
        let mut room = crate::resource::room(memory);
        // Bytes uploaded by this pass since the card was last asked
        // whether it has room: asked again every 32 MiB (a pass may upload
        // 192 MB and more, and one asking a pass let the queue run some
        // 400 MB past its line while the world loaded: run 193).
        let mut since_asked = 0u64;
        if !room {
            HEAP_TEXTURES_FULL.fetch_add(1, Ordering::Relaxed);
            once!("textures picked by index: the card has no room for more; they wait");
        }
        let mut refused = 0;
        // Room again: the slots held back while the card was full are due
        // now (`full_card_pass`).
        if room {
            host.texture_slots_room(frame);
        }
        // Slots whose texture's memory the runtime mapped, unmapped or
        // wrote since the last pass are due again.
        host.texture_slots_remapped(frame);
        let baking = crate::texture::BAKING.get();
        if baking {
            host.texture_slots_bake_written(frame);
        }
        host.texture_slots_short(frame);
        // The slots of textures whose first level with memory (levels on
        // demand), or whose host copy for one with more levels, changed since
        // the last pass are made again from it; those whose new view's upload
        // has run are made from that.
        host.texture_views_remade();
        host.texture_slots_flip();
        // And with room, the textures whose upload left a held level
        // without the card's memory are uploaded again.
        if room {
            host.texture_slots_waiting(frame);
        }
        let (mut due, census) = host.texture_slots_due(frame, if baking { 1 << 16 } else { 4096 });
        // (Those that are due later after all that are due.)
        let on_time = due.len();
        if baking {
            let known: std::collections::HashSet<u32> = due.iter().map(|&(index, _)| index).collect();
            static AHEAD: AtomicU64 = AtomicU64::new(u64::MAX);
            let again = AHEAD.swap(frame, Ordering::Relaxed) == frame;
            let (ahead, _) = host.texture_slots_due(frame + BAKE_AHEAD[usize::from(again)], 1 << 16);
            due.extend(ahead.into_iter().filter(|(index, _)| !known.contains(index)));
        }
        HEAP_TEXTURES_WAITING.store(census[0], Ordering::Relaxed);
        // Why they wait (`VICTORY_TEXTURE_WAIT_WHY=1`): a walk of the queue
        // at most once a second.
        if crate::waitwhy::walk_due(crate::host::now_ms(), census[0]) {
            crate::waitwhy::note_queue(crate::host::now_ms(), host.texture_wait_why(frame));
        }
        for (count, now) in HEAP_TEXTURES_QUEUE.iter().zip(census) {
            count.store(now, Ordering::Relaxed);
        }
        // Where the card's memory is, and what the due slots would need
        // (`VICTORY_CARD_ACCOUNTING=1`; a line when the card first has no
        // room, and every five seconds after it).
        Self::card_accounting(heap, host, frame, room);
        // The frames in about four seconds, between 16 and 256: how long a
        // slot with nothing to upload waits at most for its next look. In
        // frames alone it was 120: four minutes at the opening cutscene's
        // half a frame per second, while the title streams by the clock.
        // (About a second before the runtime told of the memory it maps:
        // the looks are now only for what that does not tell.)
        static PASSED: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms();
        let took = now.saturating_sub(PASSED.swap(now, Ordering::Relaxed)).max(1);
        // On its own thread the pass may run several times in a frame: no
        // shorter than that.
        let own_thread = crate::heap_worker::here();
        let took = if own_thread { took.max(100) } else { took };
        // (`took` is since the last pass, of any heap and list: several a
        // frame, so this was often 256 frames, 13-18 s at 14-20 frames a
        // second; `VICTORY_TEXTURE_LOOK_MS` counts by the frame's time.)
        let most = match look_ms() {
            Some(ms) => (ms / frame_ms(frame, now)).clamp(1, 256),
            None => (4000 / took).clamp(16, 256),
        };
        // What a table entry of a texture a shader samples says.
        let sampled = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        // Also within a time per pass: the pass runs on a thread that
        // records the frame, and making and staging 192 MB of textures took
        // 1.6-2.2 s of each of the world's first twenty frames (run 20).
        // The rest waits for the next frames, as it does past the bytes.
        static SPAN_MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let span = std::time::Duration::from_millis(*SPAN_MS.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_STAGE_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(250)
        }));
        // A backlog (a scene's first minutes: 22,000 slots due, 2-3 minutes of black
        // surfaces at 192 MB a frame) gets three times the bytes and the time until it
        // drains; `VICTORY_TEXTURE_CATCHUP=0` keeps the plain budget.
        static CATCHUP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let catchup = *CATCHUP.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_CATCHUP").is_none_or(|v| v != "0"));
        let (per_frame, span) = if catchup && on_time >= 1024 { (per_frame * 3, span * 3) } else { (per_frame, span) };
        // A run made early (`State::early_textures`) has a budget of its
        // own: its bytes on top of what the frame has spent so far.
        // (Not `most`: that is the longest wait between looks, used below.)
        let (span, bytes_at_most) = match PASS_BUDGET.get() {
            Some((ms, bytes, _)) => {
                (std::time::Duration::from_millis(ms), SPENT.load(Ordering::Relaxed).saturating_add(bytes))
            }
            None => (span, per_frame),
        };
        // (`VICTORY_RESIDENCY=1`, codec-d/cool_fox: the frame's ceiling while
        // the system is short of memory, after the catch-up and early budgets;
        // not for a bake, which samples its textures at once.)
        let required = PASS_BUDGET.get().is_some_and(|(_, _, bake)| bake);
        let bytes_at_most = if required || asked.is_some() {
            bytes_at_most
        } else {
            crate::residency::prefetch_limit(
                bytes_at_most,
                crate::residency::pressured(),
                crate::residency::critical_commit(),
                crate::residency::limited_host_ram(),
                crate::texture::live_staging_bytes(),
            )
        };
        let began = std::time::Instant::now();
        if sync.is_none() {
            BAKE_SERVED.store(0, Ordering::Relaxed);
        }
        let mut served = true;
        // Why the pass ends, and how many of the slots handed to it it
        // looked at (`crate::waitwhy`, `VICTORY_TEXTURE_WAIT_WHY=1`).
        let (handed, mut looked, mut ended) = (due.len() as u64, 0u64, crate::waitwhy::End::Served);
        for (index, write) in due {
            looked += 1;
            if SPENT.load(Ordering::Relaxed) >= bytes_at_most || began.elapsed() > span {
                ended = if SPENT.load(Ordering::Relaxed) >= bytes_at_most {
                    crate::waitwhy::End::Bytes
                } else {
                    crate::waitwhy::End::Time
                };
                served = false;
                break;
            }
            if own_thread && crate::heap_worker::stop() {
                ended = crate::waitwhy::End::Stop;
                served = false;
                break;
            }
            // On a thread the title knows nothing of, the texture object a
            // slot names is held while the pass works on it. (Released
            // meanwhile, or not a texture: nothing to upload.)
            let _held = match heap.handle(index).and_then(crate::descriptor::view) {
                Some(View::Shader(object, _)) if own_thread => {
                    let held = crate::resource::hold(object as This);
                    if held.is_none() {
                        host.texture_slot_done(index, write, None);
                        continue;
                    }
                    held
                }
                _ => None,
            };
            // A placed texture whose memory is not mapped (not streamed in
            // yet, or streamed out again: the heap keeps tens of thousands
            // of such views) gets no host texture; looked at again soon (a
            // title writes a texture's view before its pages are mapped,
            // and the pass runs early in a frame). One of which some is
            // mapped (a texture streaming in: its smaller mips lie at the
            // start of its memory and arrive first) is uploaded with what
            // is there, and again when more is. (It was uploaded again
            // every 120 frames, changed or not; and asked by its first
            // page alone, one whose first page is the one missing counted
            // as not there.)
            // A slot uploaded whole is looked at again a few times
            // (`RECHECK_FRAMES`): the title writes a texture's view, and
            // may fill the texture's memory itself only afterwards, with
            // no descriptor write and nothing from the runtime to tell of
            // it. The host copy then kept what the memory held at the
            // first look for the rest of the run. The look reads 48 KiB;
            // only a changed texture is uploaded again.
            // `VICTORY_HEAP_RECHECK=0`: uploaded once, as before.
            // One found changed by such a look is looked at in the next frame
            // again: the textures left that change unnoticed are ones the
            // title paints all the time through another view of their pages
            // (all their blocks are shown at other addresses too), and each
            // was shown four frames late, over and over.
            let mut again_in = RECHECK_FRAMES[0];
            // Whether a look found the texture changed (`crate::waitwhy`).
            let mut changed_look = false;
            // What is known of the slot's use, before this pass makes it
            // show anything (`crate::serve`, counted with its upload).
            let used_as = crate::serve::by_use_set().then(|| host.texture_slot_use(index, frame));
            if let Some((at, len, print, looks, uploaded)) = host.texture_slot_checked(index) {
                if crate::texture::fingerprint_texture(at, len) == print {
                    HEAP_TEXTURES_RECHECKED[0].fetch_add(1, Ordering::Relaxed);
                    if late_look(looks) {
                        HEAP_TEXTURES_RECHECKED_LATE[0].fetch_add(1, Ordering::Relaxed);
                    }
                    if volatile(at, frame) {
                        // (Its looks stay at one: the waits begin anew when
                        // it has been quiet for long enough.)
                        RECHECK_WAITED[3].fetch_add(1, Ordering::Relaxed);
                        host.texture_slot_recheck(index, write, frame + 1, (at, len), print, 1, uploaded);
                        continue;
                    }
                    let long = long_for(recheck_long(), aliased_every(), || crate::texture::aliased(at, len));
                    match next_look(looks, long) {
                        Some((wait, next)) => {
                            host.texture_slot_recheck(index, write, frame + wait, (at, len), print, next, uploaded)
                        }
                        None => host.texture_slot_done(index, write, None),
                    }
                    continue;
                }
                changed_look = true;
                if recheck_soon() {
                    again_in = 1;
                }
                let n = HEAP_TEXTURES_RECHECKED[1].fetch_add(1, Ordering::Relaxed);
                // Since the look before: the first is counted from the upload.
                let waited = waited_for(looks, frame.saturating_sub(uploaded), volatile(at, frame), || {
                    long_for(recheck_long(), aliased_every(), || crate::texture::aliased(at, len))
                });
                VOLATILE.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Default::default).note(at, frame);
                RECHECK_WAITED[0].fetch_add(1, Ordering::Relaxed);
                RECHECK_WAITED[1].fetch_add(waited, Ordering::Relaxed);
                RECHECK_WAITED[2].fetch_max(waited, Ordering::Relaxed);
                // (Each of the first 200 found by a late look is said too:
                // they are the ones that kept wrong bytes before.)
                let late = late_look(looks) && HEAP_TEXTURES_RECHECKED_LATE[1].fetch_add(1, Ordering::Relaxed) < 200;
                if n < 200 || n.is_multiple_of(1000) || late {
                    // Whether its pages are the title's at other addresses
                    // too: what is written there is told of there alone.
                    let elsewhere = crate::inplace::shown_elsewhere(at as usize & !0xffff, (at + len) as usize);
                    let first = heap.handle(index).and_then(crate::descriptor::view).map_or(0, |view| match view {
                        View::Placed(false, _, desc) => crate::resource::first_mip(&desc),
                        _ => 0,
                    });
                    tracing::warn!(
                        first_level = first,
                        blocks_shown_elsewhere = elsewhere.len(),
                        slot = index,
                        address = format_args!("{at:#x}"),
                        bytes = len,
                        frames_after_upload = frame.saturating_sub(uploaded),
                        looks,
                        changed = n + 1,
                        "a texture picked by index changed after its upload with nothing to tell of it: uploaded again"
                    );
                }
            }
            let mut partly = None;
            let mut there = None;
            if let Some(View::Placed(false, resource, desc)) = heap.handle(index).and_then(crate::descriptor::view) {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                // (Under `VICTORY_TEXTURE_MAX` the levels over the cap are
                // not waited for: the host copy goes without them.)
                let first =
                    crate::resource::first_mip(&desc).max(crate::resource::placed_cap(&resource));
                let len = crate::resource::placed_bytes_from(&resource, first).unwrap_or(1);
                let mapped = crate::texture::mapped_bytes(address as *const u8, len as usize);
                if mapped == 0 {
                    HEAP_TEXTURES_AWAY.fetch_add(1, Ordering::Relaxed);
                    // Streamed out: what the slot showed is let go.
                    host.texture_slot_gone(index);
                    host.texture_slot_later(index, write, frame, most, len);
                    continue;
                }
                host.texture_slot_here(index);
                there = Some(mapped);
                if mapped < len {
                    if mapped <= host.texture_slot_mapped(index) {
                        HEAP_TEXTURES_PARTLY[0].fetch_add(1, Ordering::Relaxed);
                        host.texture_slot_later(index, write, frame, most, len);
                        continue;
                    }
                    if !crate::texture::committed(address as *const u8, 1) {
                        HEAP_TEXTURES_PARTLY[2].fetch_add(1, Ordering::Relaxed);
                    }
                    partly = Some(mapped);
                }
            }
            // A texture object the host renders to or writes holds the
            // host's own contents: nothing to bring up to date, and this
            // pass is no use of it (its host copy is made when something
            // uses it, and its slots then: `descriptor::make_waiting`).
            if let Some(View::Shader(object, _)) = heap.handle(index).and_then(crate::descriptor::view)
                && crate::resource::buffer_bytes(object as This).is_none()
                && !crate::resource::title_fills(object as This)
            {
                host.texture_slot_done(index, write, None);
                continue;
            }
            // From here on a host texture is made: not without room. The
            // slot stays due.
            // Not so a slot already made from this very view: its host
            // texture exists, and uploading it again takes none of the
            // card's memory. It was refused all the same, so a texture the
            // title filled after its first upload kept its empty host copy
            // from the moment the card reached the line, which is the first
            // world frame in eight launches of nine: the same slots were
            // found changed again at every pass and never uploaded (run
            // 179: 33,000 such findings), and the main character's face
            // had its colour only in the launches that entered the world
            // under the line. `VICTORY_REUPLOAD_WHEN_FULL=0`: refused, as
            // before.
            if !room {
                let has_its_texture = reupload_when_full()
                    && heap.handle(index).is_some_and(|handle| host.texture_slot_made_from(index, handle));
                // Nor a texture made level by level, unless the card is
                // over its line (`VICTORY_LEVELS_BY_ROOM`, `crate::serve`):
                // making it takes none of the card's memory, and its upload
                // gives tiles to its small levels alone, so it shows those
                // instead of nothing until there is room for the large ones.
                let by_levels = !has_its_texture
                    && crate::serve::by_room()
                    && !crate::serve::over_lately()
                    && heap
                        .handle(index)
                        .and_then(crate::descriptor::view)
                        .as_ref()
                        .and_then(crate::descriptor::host::placed_texture)
                        .is_some_and(|view| matches!(view, View::Shader(texture, _) if crate::resource::by_levels(texture as This)));
                if by_levels {
                    crate::serve::note(crate::serve::Count::PastFull, 1);
                } else if !has_its_texture {
                    refused += 1;
                    if full_card_pass() {
                        // Held back for the frames of about four seconds
                        // (`most`), or until the card has room; the pass
                        // goes on, within a count that bounds one pass.
                        host.texture_slot_held(index, write, frame + most);
                        if refused > 1024 {
                            ended = crate::waitwhy::End::Room;
                            break;
                        }
                    } else if refused > 256 {
                        ended = crate::waitwhy::End::Room;
                        break;
                    }
                    continue;
                } else {
                    let n = HEAP_TEXTURES_REUPLOADED_FULL.fetch_add(1, Ordering::Relaxed) + 1;
                    if n == 1 || n.is_multiple_of(500) {
                        tracing::info!(
                            so_far = n,
                            slot = index,
                            "a texture already on the card is brought up to date although the card has no room for new ones"
                        );
                    }
                }
            }
            // (`VICTORY_RESIDENCY=1`: no new host texture or staging while
            // commit is short; after the unmapped slots above were let go.)
            if !crate::residency::admit_texture_pass(required) {
                ended = crate::waitwhy::End::Ram;
                served = false;
                break;
            }
            // A slot that shows the texture by an earlier view of it (one
            // from another level) goes on doing so until the upload of what
            // the new one shows has run.
            let sync_bake = sync.is_some();
            let shows_other = !baking
                && !sync_bake
                && crate::resource::held_only()
                && heap.handle(index).is_some_and(|handle| host.texture_slot_shows_other(index, handle));
            // A slot that shows nothing of it yet, or something else, shows it
            // when the upload has run too. (Always so for the pass on its
            // own thread: its list runs with a later submission.)
            let late = !shows_other
                && !baking
                && !sync_bake
                && (own_thread || crate::descriptor::host::show_after())
                && heap.handle(index).is_some_and(|handle| !host.texture_slot_made_from(index, handle));
            // The view and the record it came from are read once: the upload
            // and whatever shows the slot later refer to that record, not to
            // one the title wrote to the slot meanwhile.
            let (view, upload_key) = if sync_bake || shows_other || late {
                match heap.handle(index).and_then(|handle| host.prepare_texture_slot(index, handle)) {
                    Some((view, key)) => (Some(view), Some(key)),
                    None => (None, None),
                }
            } else {
                let view = heap.handle(index).and_then(|handle| host.mirror(index, handle, true));
                (view, None)
            };
            let at_once = || {
                if (late || (sync_bake && upload_key.is_some())) && let Some(handle) = heap.handle(index) {
                    let _ = host.mirror(index, handle, true);
                }
            };
            let placed = view.as_ref().and_then(crate::descriptor::host::placed_texture);
            if crate::residency::upload_refused() {
                ended = crate::waitwhy::End::Ram;
                served = false;
                break;
            }
            let resource = match placed.as_ref().or(view.as_ref()) {
                Some(&View::Shader(resource, _)) => resource as This,
                _ => {
                    at_once();
                    host.texture_slot_done(index, write, None);
                    continue;
                }
            };
            // (The slot may name another object than a moment ago.)
            let _held_now = match (&placed, own_thread) {
                (None, true) => {
                    let held = crate::resource::hold(resource);
                    if held.is_none() {
                        host.texture_slot_done(index, write, None);
                        continue;
                    }
                    held
                }
                _ => None,
            };
            let texture = crate::resource::host_resource(resource);
            if crate::residency::upload_refused() {
                ended = crate::waitwhy::End::Ram;
                served = false;
                break;
            }
            let title_fills = texture.is_some_and(crate::texture::title_written);
            if !title_fills {
                at_once();
                // a render target, a depth or unordered access texture: the host's own
                host.texture_slot_done(index, write, None);
                continue;
            }
            // A texture that has a staging buffer now is one draws read
            // through a table: they upload it, and check it at every
            // submission that samples it.
            let tabled = crate::texture::is_staged(resource);
            // The host copy holds these very bytes: nothing to upload.
            let first = match (&placed, &view) {
                (Some(_), Some(View::Placed(_, _, desc))) => crate::resource::first_mip(desc),
                _ => 0,
            };
            let same = !tabled && crate::texture::unchanged(resource, first);
            if same {
                HEAP_TEXTURES_SAME.fetch_add(1, Ordering::Relaxed);
                if let (Some(sync), Some(key), Some(handle)) = (sync.as_deref_mut(), upload_key, heap.handle(index)) {
                    if let Some(copy) = crate::resource::host_copy(resource) {
                        sync.keep.push(copy);
                    }
                    sync.publish.push((index, handle, key));
                } else {
                    if shows_other && let Some(handle) = heap.handle(index) {
                        let _ = host.mirror(index, handle, true);
                    }
                    // Its bytes are in, or on the way there in a list before.
                    if late && texture.is_some_and(crate::pool::fresh) && !crate::texture::ahead::there(resource) {
                        if let (Some(handle), Some(key)) = (heap.handle(index), upload_key) {
                            host.texture_slot_show_after(list, index, handle, key);
                        }
                    } else {
                        at_once();
                    }
                }
                let again = partly.map(|mapped| (frame + 1, mapped));
                let memory = if again.is_none() && heap_recheck() { crate::texture::title_span(resource, first) } else { None };
                match memory {
                    Some((at, len)) => {
                        let print = crate::texture::fingerprint_texture(at, len);
                        host.texture_slot_recheck(index, write, frame + again_in, (at, len), print, 0, frame);
                    }
                    None => host.texture_slot_done(index, write, again),
                }
                continue;
            }
            // The first upload of a texture no list has used: on a queue of
            // its own, and the slot shows it as soon as that has run.
            let ahead = if late && !tabled && texture.is_some_and(crate::pool::fresh) {
                crate::texture::ahead::take(resource, first, there.filter(|_| partly.is_none()).unwrap_or(u64::MAX))
            } else {
                None
            };
            if let Some(ahead) = &ahead {
                use crate::texture::ahead::Taken;
                let (Taken::New(job) | Taken::Known(job)) = ahead;
                if let (Some(handle), Some(key)) = (heap.handle(index), upload_key) {
                    host.texture_slot_show_ahead(job, index, handle, key);
                }
                if let Taken::New(job) = ahead {
                    crate::texture::ahead::send(job);
                }
            }
            if ahead.is_none() && crate::residency::upload_refused() {
                ended = crate::waitwhy::End::Ram;
                served = false;
                break;
            }
            let late = late && ahead.is_none();
            let _ = crate::residency::take_staging_deferred();
            if let Some(view) = &view
                && ahead.is_none()
            {
                // (What was found mapped of it a moment ago is not asked
                // again, unless the host copy may need more.)
                crate::texture::PICKING.set(Some(there.filter(|_| partly.is_none()).unwrap_or(u64::MAX)));
                let _ = crate::residency::take_staging_deferred();
                readable(list, sampled.kind, view);
                crate::texture::PICKING.set(None);
            }
            // A staging buffer refused or put off (`crate::residency`) leaves
            // the slot due: marking it served would lose its only retry.
            if ahead.is_none()
                && (crate::residency::take_staging_deferred() || crate::residency::upload_refused())
            {
                ended = crate::waitwhy::End::Ram;
                served = false;
                break;
            }
            if let (Some(sync), Some(key), Some(handle)) = (sync.as_deref_mut(), upload_key, heap.handle(index)) {
                if let Some(copy) = crate::resource::host_copy(resource) {
                    sync.keep.push(copy);
                }
                sync.publish.push((index, handle, key));
            }
            if let Some((since, waited, away)) = host.texture_slot_since(index) {
                use crate::texlate::{Kind, note};
                note(if waited { Kind::Waited } else { Kind::Ready }, since);
                note(Kind::Noticed, away);
            }
            if shows_other && let (Some(handle), Some(key)) = (heap.handle(index), upload_key) {
                host.texture_slot_flip(list, index, handle, key);
            }
            if late && let (Some(handle), Some(key)) = (heap.handle(index), upload_key) {
                host.texture_slot_show_after(list, index, handle, key);
            }
            #[cfg(feature = "diagnostics")]
            if crate::book::on()
                && let Some(texture) = texture.filter(|t| crate::pool::fresh(t))
            {
                use windows::core::Interface;
                let raw = texture.as_raw() as usize;
                let asked = crate::host::submissions();
                list.defer_host(move || {
                    let (started, finished) = crate::host::submissions();
                    crate::book::note(raw, || {
                        format!(
                            "; upload for slot {index} recorded at submissions {asked:?}, sent as submission {started} (finished {finished}) at {:.3} s",
                            crate::host::now_ms() as f64 / 1e3
                        )
                    });
                });
            }
            // A texture view the host refuses removes the device: name it.
            static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !GONE.load(Ordering::Relaxed)
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                GONE.store(true, Ordering::Relaxed);
                let text: String = format!("{view:?}").chars().take(300).collect();
                tracing::error!(why, slot = index, "the host device went while textures picked by index were made ready; last: {text}");
                crate::host::log_removal_context();
                return false;
            }
            let (bytes, whole) = crate::texture::staged(resource, there.is_none()).unwrap_or((0, true));
            // Not all of it there: looked at again next frame, and from
            // then on as a slot with nothing new (a placed texture: until
            // more of it is mapped; a texture object at an address the
            // title mapped: again in 120 frames, as before).
            let again = match partly {
                Some(mapped) => Some((frame + 1, mapped)),
                None => (!whole).then_some((frame + 120, 0)),
            };
            if again.is_some() {
                HEAP_TEXTURES_PARTLY[1].fetch_add(1, Ordering::Relaxed);
            }
            if tabled {
                // Nothing was staged here, so nothing is counted or spent,
                // and the staging buffer stays: taking it away made the
                // next draw that reads the texture through a table stage,
                // detile and upload all of it once more.
                HEAP_TEXTURES_TABLED.fetch_add(1, Ordering::Relaxed);
                host.texture_slot_done(index, write, again);
                continue;
            }
            // Uploaded: the list holds the staging buffer until it has run;
            // a table read, or the next look at a texture not all mapped
            // yet, makes it anew. (Kept for those until now: every partly
            // streamed texture of the heap held a staging buffer of its full
            // size for good.)
            let memory = if again.is_none() && heap_recheck() {
                crate::texture::staged_memory(resource)
            } else {
                None
            };
            crate::texture::unstage(resource);
            crate::heap_worker::uploaded(bytes);
            if changed_look {
                crate::waitwhy::changed_upload(bytes);
            }
            SPENT.fetch_add(bytes, Ordering::Relaxed);
            since_asked += bytes;
            if room && since_asked >= 32 << 20 {
                since_asked = 0;
                room = crate::resource::room(crate::host::host().and_then(|h| h.video_memory()));
            }
            HEAP_TEXTURE_BYTES.fetch_add(bytes, Ordering::Relaxed);
            HEAP_TEXTURES.fetch_add(1, Ordering::Relaxed);
            if let Some(class) = used_as {
                crate::serve::served(class);
            }
            match memory {
                // Whole and uploaded: looked at again (see above).
                Some((at, len)) => {
                    let print = crate::texture::fingerprint_texture(at, len);
                    host.texture_slot_recheck(index, write, frame + again_in, (at, len), print, 0, frame);
                }
                None => host.texture_slot_done(index, write, again),
            }
        }
        // (A pass that stopped left the slot at hand due too.)
        crate::waitwhy::pass_ended(ended, handed.saturating_sub(looked) + u64::from(ended != crate::waitwhy::End::Served));
        if sync.is_none() && baking && served {
            BAKE_SERVED.store(host.texture_news().0 + 1, Ordering::Relaxed);
        }
        served
    }

    /// For the frame listing, after a dropped dispatch's line: its group
    /// counts and the compute root arguments it was recorded with (root
    /// constants as hex dwords, root views with the title buffer they
    /// point into). A program that does not translate has no bindings to
    /// list; these say which memory it would have read and written (the
    /// ray-tracing structure builder passes its buffers this way).
    #[cfg(feature = "diagnostics")]
    fn list_dropped_roots(&self, groups: [u32; 3]) {
        crate::capture::op(0, || {
            use std::fmt::Write;
            let [x, y, z] = groups;
            let mut text = format!("  its roots ({x}x{y}x{z}):");
            for (slot, root) in self.compute.iter().enumerate() {
                match root {
                    Root::None => {}
                    Root::Table(a) => {
                        let _ = write!(text, " {slot}=table {a:#x}");
                    }
                    Root::Constants(v) => {
                        let _ = write!(text, " {slot}=constants {:x?}", &v[..v.len().min(48)]);
                    }
                    Root::View(a) => {
                        let _ = write!(text, " {slot}=view {a:#x}");
                        match crate::shadow::extent(*a) {
                            Some((start, len)) => {
                                let _ = write!(text, " in buffer {start:#x}+{len:#x}");
                            }
                            None => text.push_str(" in no buffer"),
                        }
                    }
                }
            }
            text
        });
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32, flags: u32) {
        // It may write the arguments a window holds clamped.
        self.window = None;
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        // The ray-tracing structure builder's stages run all or none.
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_why("a stage of the ray-tracing structure builder, which is off".to_owned());
                self.list_dropped_roots([x, y, z]);
            }
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_drop(self.pipeline as This);
                self.list_dropped_roots([x, y, z]);
            }
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            // A stage of the structure builder that cannot be bound is a
            // dropped stage: the others stop too.
            if crate::pipeline::is_builder(self.pipeline as This) {
                crate::pipeline::builder_rests("one of its dispatches could not be bound");
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_builder(self.pipeline as This) {
            crate::capture::op(list as *const _ as usize, || format!("builder {}", p.name));
            self.list_dropped_roots([x, y, z]);
        }
        // Counted in threads: whole groups of them, and the program is told
        // how many of the last one run, where it asks (its `bind` set that to
        // all of them).
        let size = (flags & THREAD_DIMENSIONS != 0 && thread_dimensions())
            .then(|| p.stages.first().and_then(|s| s.translated().threads))
            .flatten();
        let threads = [x, y, z];
        let [mut x, mut y, mut z] = size.map_or(threads, |size| dispatch_groups(threads, size));
        if size.is_some() {
            if !crate::pipeline::is_builder(self.pipeline as This) {
                once!(pipeline = p.name, "a dispatch counted in threads that is not the structure builder's");
            }
            if let Some((param, word)) = compute_word(p, victory_shader::emit::USER_DISPATCH_THREADS_X) {
                // SAFETY: an open list; the root signature has the word.
                unsafe { list.list.SetComputeRoot32BitConstant(param, threads[0], word) };
            }
        }
        // More than 65,535 groups in x: spread across z for a program that
        // takes the spread words (`VICTORY_SPREAD`, off unless asked).
        if let Some((param, word)) = spreads(p) {
            let (row, asked, gx, gy, gz) = victory_shader::emit::spread_groups(x, y, z, true);
            // SAFETY: an open list; the root signature has the two adjacent words.
            unsafe {
                list.list.SetComputeRoot32BitConstant(param, row, word);
                list.list.SetComputeRoot32BitConstant(param, if row == 0 { 0 } else { asked }, word + 1);
            }
            if row != 0 {
                SPREAD[0].fetch_add(1, Ordering::Relaxed);
                (x, y, z) = (gx, gy, gz);
            }
        } else if spread_on() && (x > 65_535 || y > 65_535 || z > 65_535) {
            once!(x, y, z, "a dispatch asks for more than 65,535 groups in a dimension and is not spread");
        }
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        self.buffers_written();
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
        if let Some(role) = p.stages.first().and_then(|s| s.shader.ambient)
            && (crate::ambient::on()
                || role == crate::ambient::Role::Reflect
                || (role == crate::ambient::Role::Resolve && crate::ambient::sky_on()))
        {
            let ran = self.ambient(list, p, role);
            self.finish_ambient(ran);
        }
        if p.stages.first().is_some_and(|s| s.shader.probe.is_some()) {
            self.light_probe(list, p, true);
        }
    }

    /// Clears the cached compute bindings after a stand-in pass, and the
    /// cached heaps when that pass bound a heap of its own.
    fn finish_ambient(&mut self, ran: crate::ambient::Ran) {
        if ran == crate::ambient::Ran::No {
            return;
        }
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        if ran == crate::ambient::Ran::OwnHeap {
            self.host_heaps_set = false;
        }
    }

    /// The stand-in for the indirect light after a dispatch of `p`
    /// (`crate::ambient`). Whether it recorded anything.
    fn ambient(&self, list: &HostList, p: &HostPipeline, role: crate::ambient::Role) -> crate::ambient::Ran {
        if crate::ambient::depth_match() {
            return self.ambient_matched(list, p, role);
        }
        use crate::ambient::{AIR, DEPTH, LIGHT, RADIANCE, REFLECTIONS, Ran, Role, Views};
        let Some(t) = p.stages.first().map(|s| s.translated()) else {
            return Ran::No;
        };
        let entry = |kind: ResourceType, at: (u8, u32)| self.table_view(t, true, kind, at);
        let at = match role {
            Role::Resolve => LIGHT,
            Role::Radiance => RADIANCE,
            Role::Air => AIR,
            Role::Reflect => REFLECTIONS,
            Role::Sky => return Ran::No,
        };
        let image = entry(ResourceType::Uav, at);
        let depth = match role {
            Role::Resolve => entry(ResourceType::Srv, DEPTH),
            Role::Radiance | Role::Air | Role::Reflect => image,
            Role::Sky => return Ran::No,
        };
        let (Some((of, image, image_resource)), Some((seen, depth, depth_resource))) = (image, depth) else {
            once!(?role, "no stand-in for the indirect light: other tables");
            return Ran::No;
        };
        // (R32_UINT holding R9G9B9E5; R11G11B10_FLOAT.)
        let format = match role {
            Role::Resolve | Role::Reflect => 42,
            Role::Radiance | Role::Air => 26,
            Role::Sky => return Ran::No,
        };
        let volume = of.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        if of.Format.0 != format || (seen.Width, seen.Height) != (of.Width, of.Height) || volume != (role == Role::Air) {
            once!(?role, format = of.Format.0, "no stand-in for the indirect light: another image");
            return Ran::No;
        }
        crate::ambient::after(list, role, &Views {
            image,
            depth: Some(depth),
            size: [of.Width as u32, of.Height],
            slices: u32::from(of.DepthOrArraySize),
            image_resource,
            depth_resource,
        })
    }

    /// The stand-in when [`crate::ambient::depth_match`] is set. The resolve
    /// binds a depth shader view of the image's size. The other roles leave
    /// that table unset. A descriptor of the wrong kind is not a view here
    /// ([`view_resource`]): binding it would be an invalid handle.
    fn ambient_matched(
        &self,
        list: &HostList,
        p: &HostPipeline,
        role: crate::ambient::Role,
    ) -> crate::ambient::Ran {
        use crate::ambient::{AIR, DEPTH, DepthBind, LIGHT, RADIANCE, REFLECTIONS, Ran, Role, Views};
        let Some(t) = p.stages.first().map(|s| s.translated()) else {
            return Ran::No;
        };
        let entry = |kind: ResourceType, at: (u8, u32)| self.table_view(t, true, kind, at);
        let at = match role {
            Role::Resolve => LIGHT,
            Role::Radiance => RADIANCE,
            Role::Air => AIR,
            Role::Reflect => REFLECTIONS,
            Role::Sky => return Ran::No,
        };
        let Some((of, image, image_resource)) = entry(ResourceType::Uav, at) else {
            once!(?role, "no stand-in for the indirect light: missing image UAV");
            return Ran::No;
        };
        let looked = match role {
            Role::Resolve => entry(ResourceType::Srv, DEPTH),
            Role::Radiance | Role::Air | Role::Reflect => None,
            Role::Sky => return Ran::No,
        };
        let found = looked.is_some();
        let same_size = looked.as_ref().is_some_and(|(seen, _, _)| (seen.Width, seen.Height) == (of.Width, of.Height));
        let (depth, depth_resource) = match crate::ambient::depth_bind(role, found, same_size, true) {
            DepthBind::Skip => {
                if !found {
                    once!(?role, "no stand-in for the indirect light: missing depth SRV");
                } else {
                    once!(?role, "no stand-in for the indirect light: depth dimensions differ");
                }
                return Ran::No;
            }
            DepthBind::Leave => (None, 0),
            DepthBind::Set => {
                let Some((_, depth, depth_resource)) = looked else {
                    return Ran::No;
                };
                (Some(depth), depth_resource)
            }
        };
        // (R32_UINT holding R9G9B9E5; R11G11B10_FLOAT.)
        let format = match role {
            Role::Resolve | Role::Reflect => 42,
            Role::Radiance | Role::Air => 26,
            Role::Sky => return Ran::No,
        };
        let volume = of.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        if of.Format.0 != format || volume != (role == Role::Air) {
            once!(?role, format = of.Format.0, "no stand-in for the indirect light: another image");
            return Ran::No;
        }
        crate::ambient::after(list, role, &Views {
            image,
            depth,
            size: [of.Width as u32, of.Height],
            slices: u32::from(of.DepthOrArraySize),
            image_resource,
            depth_resource,
        })
    }

    /// The view a shader of the bound pipeline reads through a table, at
    /// (root parameter, dwords into the table): its texture's or buffer's
    /// description, its place in the bound view heap and the title's
    /// resource. Tables made afresh for the draw are found where they were
    /// made (`ambient_tables`).
    fn table_view(
        &self,
        t: &crate::pipeline::Translated,
        compute: bool,
        kind: ResourceType,
        (slot, offset): (u8, u32),
    ) -> Option<(D3D12_RESOURCE_DESC, D3D12_GPU_DESCRIPTOR_HANDLE, usize)> {
        let heap = (self.heaps[0] != 0).then(|| crate::descriptor::heap(self.heaps[0] as This).host()).flatten()?;
        t.used.iter().find(|u| {
            u.kind == kind && u.root == RootType::Table && (u.slot, u.table_offset_dwords) == (slot, offset)
        })?;
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
            return None;
        };
        let viewed = crate::descriptor::view(table + u64::from(offset) * 4)?;
        let resource = view_resource(kind, &viewed, crate::ambient::depth_match())?;
        let start = match self.ambient_tables.iter().find(|m| m.0 == slot) {
            Some(m) => m.1,
            None => self.user_value(compute, slot, RootType::Table, 0),
        };
        let host = crate::resource::host_resource(resource as This)?;
        // SAFETY: a live resource.
        let desc = unsafe { host.GetDesc() };
        Some((desc, heap.gpu_handle((start + offset * 4) / 32), resource))
    }

    /// The place in the bound view heap of the view a shader of the bound
    /// pipeline reads through a table at (root parameter, dwords into the
    /// table), whatever it views (buffers too).
    fn table_handle(
        &self,
        t: &crate::pipeline::Translated,
        compute: bool,
        kind: ResourceType,
        (slot, offset): (u8, u32),
    ) -> Option<D3D12_GPU_DESCRIPTOR_HANDLE> {
        let heap = (self.heaps[0] != 0).then(|| crate::descriptor::heap(self.heaps[0] as This).host()).flatten()?;
        t.used.iter().find(|u| {
            u.kind == kind && u.root == RootType::Table && (u.slot, u.table_offset_dwords) == (slot, offset)
        })?;
        let start = match self.ambient_tables.iter().find(|m| m.0 == slot) {
            Some(m) => m.1,
            None => self.user_value(compute, slot, RootType::Table, 0),
        };
        Some(heap.gpu_handle((start + offset * 4) / 32))
    }

    /// The light probe after a draw or dispatch (`crate::lightprobe`).
    fn light_probe(&mut self, list: &HostList, p: &HostPipeline, compute: bool) {
        use crate::lightprobe::{AVERAGE, BEFORE, LUX_IN, LUX_OUT, Role};
        let Some(stage) = p.stages.iter().find(|s| s.shader.probe.is_some()) else {
            return;
        };
        let t = stage.translated();
        let done = match (stage.shader.probe, compute) {
            (Some(Role::Exposure), false) => match (
                self.table_handle(t, false, ResourceType::Srv, AVERAGE),
                self.table_handle(t, false, ResourceType::Srv, BEFORE),
            ) {
                (Some(a), Some(b)) => crate::lightprobe::after_exposure(list, a, b),
                _ => {
                    once!("light probe: the exposure program's inputs are not where they were");
                    false
                }
            },
            (Some(Role::Lux), true) => match (
                self.table_handle(t, true, ResourceType::Srv, LUX_IN),
                self.table_handle(t, true, ResourceType::Uav, LUX_OUT),
            ) {
                (Some(a), Some(b)) => crate::lightprobe::after_lux(list, a, b),
                _ => {
                    once!("light probe: the lux program's views are not where they were");
                    false
                }
            },
            _ => false,
        };
        if done {
            self.host_pipeline = 0;
            self.host_roots[1] = 0;
        }
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        // What it draws into (a dispatch writes through its views below).
        if !compute {
            let described = |t: usize| {
                crate::resource::host_made(t as This).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!("{t:#x} {}x{} f{}", d.Width, d.Height, d.Format.0)
                })
            };
            let targets: Vec<String> = self.targets.iter().filter_map(|&t| described(t)).collect();
            let _ = write!(text, " targets {targets:?} depth {:?} state [{}]", described(self.depth), p.state);
            // The title's scissor and viewport at this draw (the first of
            // each): the title draws what it sees through a
            // portal (a window, from inside a room) under the portal's
            // rectangle, and the listing did not say which draws those are.
            if let Some(r) = self.scissors.first() {
                let _ = write!(text, " scissor [{} {} {} {}]", r.left, r.top, r.right, r.bottom);
            }
            if let Some(v) = self.viewports.first() {
                let _ = write!(
                    text,
                    " viewport [{} {} {} {} depth {}..{}]",
                    v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth
                );
            }
        }
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = match u.root {
                    RootType::Other(kind) => self.host_word(compute, kind, u.slot, u.dword, None),
                    _ => self.user_value(compute, u.slot, u.root, u.dword),
                };
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs).chain(&stage.root_uavs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        if !compute && !list_slots().is_empty() {
            text.push_str(&self.slots_text());
        }
        text
    }

    /// For the frame listing, after a draw whose shader `VICTORY_LIST_SLOTS`
    /// names: the heap slots it will pick textures from by index, and what
    /// the title and the host hold there (one `~slot` line each), with the
    /// words they were read from (`~raw` lines, to decode offline what is
    /// not decoded here).
    ///
    /// Where the slots are, for the title's character shaders
    /// (`PS_DeferredPedTexturedNC`, from their native code): in the small
    /// buffers and constants of the draw's own table (every word of those
    /// that names a texture view gets a line); in the material record, at
    /// the byte offset the draw's constants (table 1, dword 56) hold at
    /// 0x1c, of the buffer at table 0, dword 160; and in the record at the
    /// byte offset the second root constant holds of the buffer at table 0,
    /// dword 688. A word that is a small number by chance and names a
    /// texture gets a line too.
    #[cfg(feature = "diagnostics")]
    fn slots_text(&self) -> String {
        use std::fmt::Write;
        const MATERIALS: u32 = 160;
        const GLOBAL: u32 = 688;
        const DRAW_CONSTANTS: u32 = 56;
        let wanted = list_slots();
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return String::new();
        };
        let Some(stage) = p.stages.iter().find(|s| wanted.iter().any(|w| s.shader.hash.starts_with(w.as_str())))
        else {
            return String::new();
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return String::new();
        };
        let t = stage.translated();
        let mut text = String::new();
        let entry = |slot: u8, dwords: u32| -> Option<View> {
            let Some(Root::Table(table)) = self.roots_ref(false).get(usize::from(slot)) else {
                return None;
            };
            crate::descriptor::view(table + u64::from(dwords) * 4)
        };
        let mut material = None;
        for u in t.used.iter().filter(|u| u.root == RootType::Table && u.kind != ResourceType::Sampler) {
            let label = format!("s{}+{}", u.slot, u.table_offset_dwords);
            let Some(view) = entry(u.slot, u.table_offset_dwords) else {
                let _ = write!(text, "\n    ~table {label} no view");
                continue;
            };
            let range = match &view {
                View::Constant(address, size) => Some((*address, u64::from(*size))),
                other => crate::descriptor::host::buffer_range(other),
            };
            match range {
                // A small buffer or constants: its words.
                Some((address, len)) if len <= 2048 => {
                    Self::slot_words(&mut text, heap, &label, address, len as usize);
                    if u.slot == 1 && u.table_offset_dwords == DRAW_CONSTANTS && len >= 0x20 {
                        let mut word = [0u8; 4];
                        if crate::host::read_into((address + 0x1c) as *const u8, &mut word) {
                            material = Some(u32::from_le_bytes(word));
                        }
                    }
                }
                Some(_) => {}
                // A texture read through the table: the title slot it sits in.
                None => {
                    let Some(Root::Table(table)) = self.roots_ref(false).get(usize::from(u.slot)) else {
                        continue;
                    };
                    match heap.index(table + u64::from(u.table_offset_dwords) * 4) {
                        Some(index) => text.push_str(&Self::slot_line(heap, &format!("table:{label}"), 0, index)),
                        None => {
                            let _ = write!(text, "\n    ~table {label} title=[{}]", view_state(Some(&view)));
                        }
                    }
                }
            }
        }
        let uses = |slot: u8, dwords: u32| {
            t.used.iter().any(|u| u.root == RootType::Table && u.slot == slot && u.table_offset_dwords == dwords)
        };
        if let Some(at) = material
            && uses(0, MATERIALS)
            && let Some((base, len)) = entry(0, MATERIALS).as_ref().and_then(crate::descriptor::host::buffer_range)
        {
            let at = u64::from(at);
            let _ = write!(text, "\n    ~material record at {at:#x} of the buffer at {base:#x} bytes {len}");
            if at + 64 <= len {
                Self::slot_words(&mut text, heap, "M", base + at, 64);
            }
            if at + 1936 + 512 <= len {
                Self::slot_words(&mut text, heap, "M+1936", base + at + 1936, 512);
            }
        }
        if uses(0, GLOBAL)
            && let Some(slot) = t.used.iter().find(|u| u.root == RootType::Constants).map(|u| u.slot)
            && let Some(Root::Constants(words)) = self.roots_ref(false).get(usize::from(slot))
            && let Some(&at) = words.get(1)
            && let Some((base, len)) = entry(0, GLOBAL).as_ref().and_then(crate::descriptor::host::buffer_range)
        {
            let at = u64::from(at);
            let _ = write!(text, "\n    ~root constants {words:08x?}: record at {at:#x} of the buffer at {base:#x} bytes {len}");
            if at + 64 <= len {
                Self::slot_words(&mut text, heap, "G", base + at, 64);
            }
        }
        text
    }

    /// Up to 512 bytes of title memory at `address` as hex words (a `~raw`
    /// line), and a `~slot` line for every word of them that names a
    /// texture view in `heap`.
    #[cfg(feature = "diagnostics")]
    fn slot_words(text: &mut String, heap: &crate::descriptor::Heap, label: &str, address: u64, len: usize) {
        use std::fmt::Write;
        let mut bytes = vec![0u8; len.min(512) & !3];
        if bytes.is_empty() || !crate::host::read_into(address as *const u8, &mut bytes) {
            let _ = write!(text, "\n    ~raw {label} at {address:#x} bytes {len}: unreadable");
            return;
        }
        let words: Vec<u32> =
            bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        let _ = write!(text, "\n    ~raw {label} at {address:#x} bytes {len}:");
        for w in &words {
            let _ = write!(text, " {w:08x}");
        }
        for (i, &index) in words.iter().enumerate() {
            let names = index != 0
                && heap
                    .handle(index)
                    .and_then(crate::descriptor::view)
                    .as_ref()
                    .is_some_and(crate::descriptor::host::is_texture_view);
            if names {
                text.push_str(&Self::slot_line(heap, label, i * 4, index));
            }
        }
    }

    /// One `~slot` line: where the slot number was read (`from`, byte
    /// `at`), what the title's heap holds in slot `index` with the state
    /// of that texture ([`view_state`]), what the host slot was made from,
    /// and what the queue of textures picked by index knows of it.
    #[cfg(feature = "diagnostics")]
    fn slot_line(heap: &crate::descriptor::Heap, from: &str, at: usize, index: u32) -> String {
        let title = heap.handle(index).and_then(crate::descriptor::view);
        let host: String = heap
            .host()
            .and_then(|h| h.describe_slot(index))
            .unwrap_or_else(|| "none".into())
            .chars()
            .take(90)
            .collect();
        let queue = match heap.host() {
            Some(h) => h.texture_slot_state(index, crate::frame::current()),
            None => "no host heap".into(),
        };
        format!(
            "\n    ~slot from={from}+{at:#x} slot={index} title=[{}] host=[{host}] queue=[{queue}]",
            view_state(title.as_ref())
        )
    }

    /// A mesh draw under `VICTORY_MESH_TEST=scissor` rasterises one pixel:
    /// the scissors before it (`on`), the title's after it.
    fn mesh_scissor(&self, list: &HostList, on: bool) {
        if mesh_test() != 2 {
            return;
        }
        if on {
            let one = RECT { left: 0, top: 0, right: 1, bottom: 1 };
            crate::draw::scissors(list, &vec![one; self.scissors.len().max(1)]);
        } else if !self.scissors.is_empty() {
            crate::draw::scissors(list, &self.scissors);
        }
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = mesh_test() != 1
            && mesh_kept(self.pipeline)
            && self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    self.mesh_scissor(list, true);
                    #[cfg(feature = "diagnostics")]
                    if let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) {
                        list.ran_indirect(p.pso.as_raw() as usize, &p.name, crate::host::indirects::DIRECT, x, &self.user);
                    }
                    // (The host takes 65,535 groups in a dimension: more is
                    // undefined there.)
                    if x > 65_535 {
                        once!(x, "a mesh dispatch of more groups than the host takes: cut");
                    }
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x.min(65_535), 1, 1) };
                    self.drew(list, crate::pipeline::host_pipeline(self.pipeline as This));
                    self.mesh_scissor(list, false);
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// The walls trace (`crate::host::walls_trace`): one log line per
    /// frame with the draws of its main pass (four or more colour targets)
    /// by scissor rectangle, and how often each of the title's culling
    /// programs was dispatched. The title clips what it draws
    /// through a window or door to that opening's rectangle while it holds
    /// the camera to be in a room (run 182, frame 1529: 1,386 draws under
    /// `531 0 1789 1337`); in a launch without the room's walls the same
    /// shot has no such rectangle (run 187, frame 1414). The line shows
    /// the frame in which that changes.
    fn walls_census(&self, p: &HostPipeline, compute: bool) {
        const CULL: [(&str, &str); 9] = [
            ("55679a0547649d0b", "tree"),
            ("89d08e561897d4e5", "tree_imposter"),
            ("9209777e8face9dd", "grass"),
            ("17af15c4015cb4b6", "prop"),
            ("fdd56d97a806bcca", "prop_nocull"),
            ("42f717f1b4d18e94", "simple"),
            ("1c4b3c0547e3e6fa", "slod"),
            ("9146291b38ea9ce1", "meshlet_args"),
            ("42683eebec081520", "depth_pyramid_cpu_occlusion"),
        ];
        type Census = (u64, std::collections::BTreeMap<[i32; 4], u32>, [u32; 9]);
        static CENSUS: std::sync::Mutex<Census> =
            std::sync::Mutex::new((u64::MAX, std::collections::BTreeMap::new(), [0; 9]));
        let frame = crate::frame::current();
        let mut census = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
        if census.0 != frame {
            let draws: u32 = census.1.values().sum();
            if draws > 0 {
                let mut rects: Vec<([i32; 4], u32)> = census.1.iter().map(|(r, n)| (*r, *n)).collect();
                rects.sort_unstable_by_key(|r| std::cmp::Reverse(r.1));
                let scissors: Vec<String> = rects
                    .iter()
                    .take(5)
                    .map(|(r, n)| format!("{} {} {} {} x{n}", r[0], r[1], r[2], r[3]))
                    .collect();
                let counts = census.2;
                let culling: Vec<String> = CULL
                    .iter()
                    .zip(counts)
                    .filter(|(_, n)| *n > 0)
                    .map(|(c, n)| format!("{}={n}", c.1))
                    .collect();
                tracing::info!(
                    frame = census.0,
                    main_pass_draws = draws,
                    rectangles = rects.len(),
                    scissors = ?scissors,
                    culling = ?culling,
                    "walls trace: the frame's main pass by scissor rectangle, and its culling dispatches"
                );
            }
            census.0 = frame;
            census.1.clear();
            census.2 = [0; 9];
        }
        if compute {
            for (i, c) in CULL.iter().enumerate() {
                if p.stages.iter().any(|s| s.shader.hash.as_str() == c.0) {
                    census.2[i] += 1;
                }
            }
        } else if self.targets.iter().filter(|&&t| t != 0).count() >= 4 {
            let r = self.scissors.first().map_or([0; 4], |r| [r.left, r.top, r.right, r.bottom]);
            *census.1.entry(r).or_insert(0) += 1;
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u.kind, &view);
            // Up to date by this draw: not the texture pass's to make.
            if crate::descriptor::host::is_texture_view(&view) {
                host.texture_slot_tabled(index, true);
            } else {
                host.slot_tabled(index);
            }
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    ///
    /// `half`: the table descriptors the shader moves sixteen bits a
    /// channel as unsigned integers (`HostShader::half_views`). A table
    /// with a 16-bit float or signed buffer view at one of them is made
    /// afresh as well, with that view in the unsigned format of the same
    /// layout (`descriptor::host::half_view`): the title's own slot keeps
    /// the title's format for the shaders that read it as floats.
    ///
    /// `spans` and `staged` are empty unless [`table_stages`] is on: then
    /// the stages of one draw ask for one count per table, and a later
    /// stage that finds the earlier one's table still stamps the title
    /// slots it reads.
    pub(crate) fn tables(
        &mut self,
        list: &HostList,
        plan: &TablePlan,
        half: &[(u8, u32)],
        compute: bool,
        spans: &[(u8, u32, u32)],
        staged: &mut Vec<(u64, u32)>,
    ) -> Few<(u8, u32), 16> {
        let mut fresh = [0u64; 4];
        let selected_slot = |mask: &[u64; 4], slot: u8| mask[usize::from(slot / 64)] & (1u64 << (slot % 64)) != 0;
        for group in &plan.groups {
            let slot = group.slot;
            let needs_snapshot = group.has_cbv
                || match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::Table(table)) => self.heap_of(table).is_some_and(|heap| {
                        heap.host()
                            .is_some_and(|host| group.needs_snapshot(heap, host.count(), table))
                    }),
                    _ => false,
                };
            if needs_snapshot {
                fresh[usize::from(slot / 64)] |= 1u64 << (slot % 64);
            }
        }
        // A 16-bit float or signed buffer view the shader moves as unsigned
        // integers cannot stay in the title's slot: that slot keeps the
        // title's format for the shaders that read it as floats.
        for &(slot, at) in half {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let Some(view) = crate::descriptor::view(table + u64::from(at) * 4) else {
                continue;
            };
            if crate::descriptor::host::half_view(&view).is_some() {
                fresh[usize::from(slot / 64)] |= 1u64 << (slot % 64);
            } else if let Some(format) = crate::descriptor::host::half_inexact(&view) {
                once!(
                    format,
                    "a shader moves a buffer's 16-bit channels as integers through a view whose values the console converts: they are read and written unconverted"
                );
            }
        }
        for &(g, e) in &plan.direct {
            let group = &plan.groups[g];
            if !selected_slot(&fresh, group.slot) {
                self.table_entry(list, &group.entries[e], compute);
            }
        }
        let mut made = Few::default();
        for group in &plan.groups {
            let slot = group.slot;
            if !selected_slot(&fresh, slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries = &group.entries;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            if heap.samplers() || (table - heap.start) % 32 != 0 {
                continue;
            }
            let Some(host) = heap.host() else {
                continue;
            };
            let mut selected: Vec<u32> = group
                .indexed
                .iter()
                .map(|index| {
                    let constant = self.user_value(
                        compute,
                        index.constant_slot,
                        RootType::Constants,
                        index.constant_dword,
                    );
                    // The shader adds i32 byte offsets and shifts by 5.
                    index.byte_offset.wrapping_add(constant << index.shift) / 32
                })
                .collect();
            selected.sort_unstable();
            selected.dedup();
            // The descriptors the shader moves halves through are made too,
            // and the table reaches them: they may lie past the range starts
            // the entries give (a range of several views names only its first).
            let half_last = half
                .iter()
                .filter(|&&(s, _)| s == slot)
                .map(|&(_, at)| at / 8)
                .max();
            let n = (group.max_dword / 8)
                .max(selected.last().copied().unwrap_or(0))
                .max(half_last.unwrap_or(0))
                + 1;
            let span = spans.iter().find(|s| s.0 == slot).map(|s| (s.1, s.2));
            let n = staged_slots(n, span, heap.index(table), host.count());
            let last = table.checked_add(u64::from(n - 1) * 32);
            if n > crate::descriptor::host::DYNAMIC
                || last.and_then(|handle| heap.index(handle)).is_none()
            {
                once!(slot, n, "fresh descriptor table exceeds the available span");
                continue;
            }
            // The table and what its entries hold (record words).
            let key = {
                let mut h = RecordHash::default();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    h.add(unsafe { (at as *const [u32; 3]).read_unaligned() });
                    // Another table than the same slots made for a shader
                    // that reads them in the title's formats.
                    if half.contains(&(slot, k * 8)) {
                        h.add([0x6861_6c66, k, 1]);
                    }
                }
                (table, n, h.0)
            };
            // Who reads of it, and as what, as far as [`readable`] goes by it.
            let reader = (std::ptr::from_ref(plan) as usize, plan.direct.len(), STORES.get(), TESTED_DEPTH.get().0);
            if let Some(known) = self.fresh_tables.get_mut(&key) {
                let first = known.first;
                // This reader had it brought up to date, and nothing in the
                // list changed since: nothing to do again.
                let once = ready_once() && selected.is_empty() && !group.has_uav;
                if once
                    && known.epoch == list.epoch()
                    && let Some((_, depth)) = known.ready.iter().find(|(r, _)| *r == reader)
                {
                    if *depth {
                        TESTED_DEPTH.set((reader.3, true));
                    }
                    made.push((slot, first * 32));
                    continue;
                }
                let depth_before = TESTED_DEPTH.get();
                TESTED_DEPTH.set((depth_before.0, false));
                // Made by an earlier stage of this draw: the title's own
                // slots this stage reads are stamped, as making the table
                // itself would have done. Empty when the switch is off.
                let staged_title = staged
                    .iter()
                    .any(|&(at, slots)| at == table && slots == n)
                    .then(|| heap.index(table).filter(|at| at + n <= host.count()))
                    .flatten();
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u.kind, &view);
                        if let Some(title) = staged_title
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            let k = u.table_offset_dwords / 8;
                            host.texture_slot_tabled(title + k, true);
                            host.mirror_after(list, title + k, handle);
                        }
                    }
                }
                for k in selected {
                    if group.entry(k).is_some() {
                        continue;
                    }
                    let handle = table + u64::from(k) * 32;
                    if let Some(view) = crate::descriptor::view(handle)
                        && let Some(kind) = view_kind(&view)
                    {
                        readable(list, kind, &view);
                    }
                }
                let depth = TESTED_DEPTH.get().1;
                TESTED_DEPTH.set((depth_before.0, depth_before.1 || depth));
                if once && let Some(known) = self.fresh_tables.get_mut(&key) {
                    // (The epoch as it is now: what this changed itself it
                    // need not do again.)
                    let epoch = list.epoch();
                    if known.epoch != epoch {
                        known.epoch = epoch;
                        known.ready.clear();
                    }
                    if known.ready.len() < 16 {
                        known.ready.push((reader, depth));
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n).or_else(|| {
                crate::host::settle_for_fence();
                host.dynamic(n)
            }).or_else(|| host.dynamic_waited(n)) else {
                // No room: the draw must not run with the title's own
                // offsets (slots never made for it); `bind` drops it.
                made.push((slot, u32::MAX));
                continue;
            };
            self.fresh_tables.insert(key, FreshTable { first, epoch: 0, ready: Vec::new() });
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, host.id(), first, n));
            if !spans.is_empty() {
                staged.push((table, n));
            }
            // The title's own slots of this table: what the draw reads of
            // them it brings up to date here, in the dynamic slots.
            let title_first = heap.index(table).filter(|first| first + n <= host.count());
            // They are such a table's, whatever they hold. (Of a big one,
            // which may lie over slots shaders pick by index, only the ones
            // the shader names.)
            if let Some(title) = title_first {
                for k in (0..n).filter(|&k| n <= 64 || group.entry(k).is_some()) {
                    host.slot_tabled(title + k);
                }
            }
            let list_views = crate::descriptor::host::list_views();
            if self.recording == 0 {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                self.recording = NEXT.fetch_add(1, Ordering::Relaxed);
            }
            let recording = self.recording;
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let half_here = half.contains(&(slot, k * 8));
                // A view this list has made before: its descriptor copied,
                // and nothing more if nothing changed in the list since.
                // A half place is made below, in the unsigned format.
                if !half_here
                    && list_views
                    && let Some(words) = crate::descriptor::record_at(table + u64::from(k) * 32)
                    && let Some(mut listed) = host.make_listed(recording, words, index)
                {
                    let kind = group
                        .entry(k)
                        .map(|u| u.kind)
                        .or_else(|| selected.binary_search(&k).ok().and(listed.kind));
                    let was = (listed.ready, listed.tabled);
                    if let Some(kind) = kind {
                        let tested = TESTED_DEPTH.get();
                        if kind == ResourceType::Srv && listed.ready.0 == list.epoch() && listed.ready.1 == tested.0 {
                            if listed.ready.2 {
                                TESTED_DEPTH.set((tested.0, true));
                            }
                        } else if let Some(view) = crate::descriptor::host::listed_view(recording, words) {
                            let _ = crate::residency::take_staging_deferred();
                            TESTED_DEPTH.set((tested.0, false));
                            readable(list, kind, &view);
                            let depth = TESTED_DEPTH.get().1;
                            TESTED_DEPTH.set((tested.0, tested.1 || depth));
                            if kind == ResourceType::Srv && !crate::residency::take_staging_deferred() {
                                listed.ready = (list.epoch(), tested.0, depth);
                            }
                        }
                    }
                    if let Some(title) = title_first
                        && listed.texture
                        && listed.tabled != Some((title + k, kind.is_some()))
                    {
                        listed.tabled = Some((title + k, kind.is_some()));
                        host.texture_slot_tabled(title + k, kind.is_some());
                        if kind.is_some() {
                            host.mirror_after(list, title + k, table + u64::from(k) * 32);
                        }
                    }
                    if was != (listed.ready, listed.tabled) {
                        crate::descriptor::host::note_listed(recording, words, listed);
                    }
                    continue;
                }
                let used = group.entry(k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        // Renderer port 5: title memory in place is the
                        // card's to read where it is; nothing is copied.
                        if crate::port_cbv::in_place_on() {
                            match crate::shadow::find(address, u64::from(size)).filter(|s| s.in_place()) {
                                Some(shadow) => {
                                    shadow.bound(crate::shadow::BOUND_TABLE_CONSTANTS, u64::from(size));
                                    shadow.read_as(crate::readkinds::TABLE_CBV, u64::from(size));
                                    if host.constant_raw_at(index, &shadow.buffer, shadow.gpu(address), size) {
                                        crate::port_cbv::in_place(size);
                                        continue;
                                    }
                                    crate::port_cbv::copied(false);
                                }
                                None => crate::port_cbv::copied(true),
                            }
                        }
                        let Some((buffer, gpu, to)) = list.scratch_held(u64::from(size)) else {
                            host.make_null(index);
                            continue;
                        };
                        // (`VICTORY_PLACED_FILLS`: bytes a copy at its place
                        // earlier in the list wrote, the card copies here at
                        // once; else the fill at submission, as before.)
                        // SAFETY: a live buffer.
                        let offset = gpu - unsafe { buffer.GetGPUVirtualAddress() };
                        if !list.fill_from_placed(&buffer, offset, address as usize, u64::from(size)) {
                            let slot = (std::ptr::from_ref(heap) as usize, index);
                            list.defer_constants(to, address as *const u8, size as usize, slot);
                        }
                        host.constant_at(index, &buffer, gpu, size);
                    }
                    Some(view) => {
                        if half_here
                            && let Some(shown) = crate::descriptor::host::half_view(&view)
                        {
                            host.make_at(index, &shown);
                        } else {
                            host.make_at(index, &view);
                        }
                        let kind = used.map(|u| u.kind).or_else(|| {
                            selected.binary_search(&k).ok().and_then(|_| view_kind(&view))
                        });
                        if let Some(kind) = kind {
                            readable(list, kind, &view);
                        }
                        // The title's slot is a table's: out of the queue
                        // of textures picked by index if the draw read it
                        // (its texture is up to date now), and made in
                        // place as the texture pass would have made it,
                        // for a shader that picks it by index all the same.
                        if let Some(title) = title_first
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            host.texture_slot_tabled(title + k, kind.is_some());
                            if kind.is_some() {
                                host.mirror_after(list, title + k, table + u64::from(k) * 32);
                            }
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_made(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let ready = self.prepare(list, false);
        // A pipeline with the rectangle geometry stage draws the title's
        // vertices as they are; any other gets its rectangles expanded.
        let rects = self.topology == RECTANGLES && !self.completes_rects();
        let ran = ready && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            self.drew(list, crate::pipeline::host_pipeline(self.pipeline as This));
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Whether the bound pipeline completes rectangles itself, in the
    /// geometry stage rectangle pipelines get (`HostPipeline::rects`).
    fn completes_rects(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rects)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host_read(from, n * 3 * stride, move || {
                // The vertices are read through the kernel, at submission; in
                // place only if that fails, with no change of mapping under
                // the read (see `crate::watch::title_memory_reading`): while
                // the world streams a change is under way most of the time,
                // and this is the thread that submits.
                let mut read = vec![0u8; n * 3 * stride];
                let held = (!crate::host::read_into(from as *const u8, &mut read)).then(crate::watch::title_memory_reading);
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        if held.is_some() { std::slice::from_raw_parts(from as *const u8, n * 3 * stride) } else { &read[..] },
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        // Indexed rectangles are drawn by the rectangle geometry stage
        // alone (nothing expands them on the CPU).
        let lost = self.topology == RECTANGLES && !self.completes_rects();
        if lost {
            once!("indexed rectangles not drawn");
        }
        let ran = !lost && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            self.drew(list, crate::pipeline::host_pipeline(self.pipeline as This));
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Shaders that can write buffers were recorded: arguments may be others
    /// from here on.
    fn buffers_written(&mut self) {
        match &mut self.batch {
            Some(batch) if batches() == 2 => batch.wrote = true,
            batch => *batch = None,
        }
    }

    /// A draw with the pipeline `p` was recorded (as [`State::buffers_written`]
    /// if its shaders can store to buffers).
    fn drew(&mut self, list: &HostList, p: Option<&HostPipeline>) {
        if p.is_none_or(|p| p.stages.iter().any(|stage| stage.translated().stores)) {
            self.batch = None;
        }
        if let Some(p) = p
            && p.stages.iter().any(|s| s.shader.probe.is_some())
        {
            self.light_probe(list, p, false);
        }
    }

    /// A buffer was copied to: what was clamped of it ahead is clamped anew.
    pub(crate) fn copied(&mut self) {
        self.window = None;
        self.batch = None;
    }

    /// The title waits for earlier work (a barrier or a pipeline sync).
    pub(crate) fn synced(&mut self) {
        if self.batch.as_ref().is_some_and(|b| b.wrote) {
            self.batch = None;
        }
    }

    /// Where the clamped record of `kind` at `arguments` of the copy `args`
    /// is, from the open batch or a new one ([`batches`]). `None` if it
    /// cannot be clamped in one.
    fn batched(
        &mut self,
        list: &HostList,
        args: &std::sync::Arc<crate::shadow::Shadow>,
        arguments: u64,
        kind: u32,
    ) -> Option<(ID3D12Resource, u64)> {
        const BATCH_USE: usize = usize::MAX - 0x78;
        const DISPATCH: u32 = 2;
        // Another host list, or this one after its reset ([`windows`]).
        if list.first_use(BATCH_USE) {
            self.batch = None;
        }
        // A draw after dispatches waits for them.
        if kind != DISPATCH {
            self.synced();
        }
        let (id, offset) = (args.id(), args.offset(arguments));
        if let Some(b) = &mut self.batch
            && let Some(at) = b.table.place(id, offset, kind)
        {
            return Some((b.scratch.clone(), at));
        }
        self.batch = None;
        let (mut table, scratch) = crate::indirect::open_batch(list, (id, &args.buffer), &argument_copies(args))?;
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        let at = table.place(id, offset, kind)?;
        self.batch = Some(Batch { table, scratch: scratch.clone(), wrote: false });
        Some((scratch, at))
    }

    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        // Indirect rectangles (the title's tiled lighting, 256
        // indirect draws a frame, makes their corners from the vertex
        // number) are completed by the pipeline's geometry stage: the
        // arguments stay the title's, three vertices a rectangle. They
        // cannot be expanded on the CPU, which does not know their count.
        // shortcut: in a pipeline without that stage, a rectangle with no
        // per-vertex inputs is drawn as its first triangle (three vertices,
        // half of it), and one with vertex buffers is not drawn.
        let made_rects = !p.rects
            && kind == DRAW
            && self.topology == RECTANGLES
            && !p.inputs.iter().any(|e| !e.per_instance);
        if made_rects {
            once!("indirect rectangles without vertex inputs drawn as their first triangle");
        }
        let matches_kind = match kind {
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || made_rects || p.rects) && !p.compute && !p.mesh
            }
            MESH => !p.compute && p.mesh,
            DISPATCH => p.compute,
            _ => false,
        };
        if !matches_kind {
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let _ask = crate::readkinds::asking(crate::readkinds::ARGS);
        let Some(args) = crate::shadow::find_any(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        args.read_as(crate::readkinds::ARGS, len.min(args.left(arguments)));
        let counter = match count {
            Some(at) => match crate::shadow::find_any(at, 4) {
                Some(s) => {
                    s.read_as(crate::readkinds::ARGS, 4);
                    Some((s, at))
                }
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        // A dispatch of a program that takes the spread words: the clamp
        // writes [row, asked x, x, y, z] records, run through a signature
        // that sets the two words, then dispatches. If that signature is
        // refused, the ordinary one is used and the words stay 0.
        let spread_signature = (kind == DISPATCH)
            .then(|| spreads(p))
            .flatten()
            .and_then(|(param, index)| host_spread_signature(&p.root, param, index));
        let is_spread = spread_signature.is_some();
        // Spread lean (`VICTORY_SPREAD_LEAN`, off unless asked): the records
        // within 65,535 groups go through the plain dispatch signature. The
        // spread signature runs only under a predicate the clamp sets when a
        // record passes 65,535. Unset, every record stays on the spread
        // signature.
        let plain = (is_spread && max > 0 && max <= crate::indirect::SPREAD_LEAN_RECORDS && spread_lean_on())
            .then(|| host_signature(DISPATCH, 12))
            .flatten();
        let Some(signature) = spread_signature.or_else(|| {
            matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
                .then(|| host_signature(kind, stride))
                .flatten()
        }) else {
            why("no host signature");
            return false;
        };
        if kind == MESH && (mesh_test() == 1 || !mesh_kept(self.pipeline)) {
            why("mesh draws are switched off (VICTORY_MESH_TEST=skip)");
            return false;
        }
        // Renderer port 1: arguments (or their count) in the title's memory
        // read in place ([`in_place_order`]): clamped alone, right before
        // their execution, not ahead in a batch or window.
        let in_place = args.in_place() || counter.as_ref().is_some_and(|(s, _)| s.in_place());
        if in_place {
            in_place_indirect(arguments, kind);
        }
        let alone = in_place && in_place_order();
        if alone {
            self.batch = None;
            self.window = None;
        }
        // One record without a count: clamped in a batch (`batches`).
        let batch = if !alone && !is_spread && batches() > 0 && max == 1 && count.is_none() {
            args.refresh(list, arguments, len);
            self.batched(list, &args, arguments, kind)
        } else {
            None
        };
        if batch.is_some() {
            self.window = None;
        }
        // A window of records clamped in one pass (`windows`): for single
        // records without a count buffer, drawn by shaders that store to no
        // buffer. Anything else ends the window: it uses the scratch buffer
        // or may write the arguments.
        const WINDOW_USE: usize = usize::MAX - 0x77;
        let offset = args.offset(arguments);
        let windowed = windows()
            && !alone
            && kind != DISPATCH
            && max == 1
            && count.is_none()
            && !p.stages.iter().any(|stage| stage.translated().stores);
        // Another host list (a new one after a cut, or this one after its
        // reset) has another scratch buffer, or this one's with old bytes.
        if !windowed || list.first_use(WINDOW_USE) {
            self.window = None;
        }
        let same = |w: &Window| w.copy == args.id() && w.kind == kind && w.stride == stride;
        let hit = self
            .window
            .as_ref()
            .filter(|w| same(w) && offset >= w.base)
            .filter(|w| (offset - w.base) % u64::from(w.step) == 0 && (offset - w.base) / u64::from(w.step) < u64::from(w.records))
            .map(|w| (w.scratch.clone(), offset - w.base));
        // With a spread-lean clamp: the plain signature, the predicate
        // word's and the spread records' offsets in the scratch buffer
        // (plain records at 0).
        let mut spread_lean = None;
        let (bounded, at) = match batch.or(hit) {
            Some(hit) => hit,
            None => {
                // A record a short step after the last window's last one
                // (the same step as that window's, if it had one): the
                // title walks the buffer, so the ones that step apart
                // after it are clamped with it.
                let (step, records) = match &self.window {
                    Some(w) if windowed && same(w) => {
                        let last = w.base + u64::from(w.records - 1) * u64::from(w.step);
                        let step = offset.saturating_sub(last);
                        let walks = step >= u64::from(stride)
                            && step <= 256
                            && step % 4 == 0
                            && (w.records == 1 || step == u64::from(w.step));
                        if walks {
                            let fit = (args.left(arguments) / step).min(WINDOW_MOST) as u32;
                            (step as u32, (if w.records == 1 { 16 } else { w.records * 2 }).min(fit).max(1))
                        } else {
                            (stride, max)
                        }
                    }
                    _ => (stride, max),
                };
                args.refresh(list, arguments, len.max(u64::from(step) * u64::from(records)));
                if let Some((s, at)) = &counter {
                    s.refresh(list, *at, 4);
                }
                // Bounded first: the clamp binds a compute pipeline of its
                // own and copies clamped active records into a scratch
                // buffer at offset 0.
                let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                // More than the record asked for: a window clamped ahead,
                // whose unfilled slots must not spend the list's budget.
                let bounded = if windowed && records > 1 {
                    crate::indirect::bound_ahead(list, (&args.buffer, offset), (kind, step, records))
                } else if let Some((plain, (dst, pred, spread_at))) = plain.and_then(|plain| {
                    crate::indirect::bound_spread_lean(list, (&args.buffer, offset), count_arg, (kind, step, records))
                        .map(|bounded| (plain, bounded))
                }) {
                    spread_lean = Some((plain, pred, spread_at));
                    Some(dst)
                } else if is_spread {
                    // A spread dispatch is never windowed: `windowed` is
                    // false for dispatches.
                    crate::indirect::bound_spread(list, (&args.buffer, offset), count_arg, (kind, step, records))
                } else {
                    crate::indirect::bound(list, (&args.buffer, offset), count_arg, (kind, step, records))
                };
                let Some(bounded) = bounded else {
                    self.window = None;
                    why("not bounded (no clamp, or arguments not 4-byte aligned)");
                    return false;
                };
                self.host_pipeline = 0;
                self.host_roots[1] = 0;
                self.window = windowed.then(|| Window {
                    copy: args.id(),
                    kind,
                    stride,
                    step,
                    base: offset,
                    records,
                    scratch: bounded.clone(),
                });
                (bounded, 0)
            }
        };
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        if !ran {
            why("prepare failed");
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        #[cfg(feature = "diagnostics")]
        list.ran_indirect(p.pso.as_raw() as usize, &p.name, kind, max, &self.user);
        if kind == MESH {
            self.mesh_scissor(list, true);
        }
        let count_buffer = counter.as_ref().map(|(s, _)| &s.buffer);
        let count_offset = counter.as_ref().map_or(0, |(s, at)| s.offset(*at));
        let ran_spread_lean = spread_lean.is_some();
        if let Some((plain, pred, spread_at)) = spread_lean {
            // SAFETY: an open list; the scratch buffer and host copies are
            // kept alive by the list, and both signatures are cached.
            // `bound_spread_lean` left the scratch buffer in `INDIRECT_ARGUMENT`,
            // which is also the predication state. The bound program's
            // spread words are 0 (its bind sets them so), which is what the
            // plain records run with; the spread signature sets them for
            // the spread records. Predication is off again before anything
            // else is recorded.
            unsafe {
                list.list.ExecuteIndirect(&plain, max, &bounded, 0, count_buffer, count_offset);
                list.list.SetPredication(&bounded, pred, D3D12_PREDICATION_OP_EQUAL_ZERO);
                list.list.ExecuteIndirect(&signature, max, &bounded, spread_at, count_buffer, count_offset);
                list.list.SetPredication(None::<&ID3D12Resource>, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
            }
        } else {
            // SAFETY: an open list with everything set; the scratch buffer and
            // host copies are kept alive by the list, and the signature is cached.
            unsafe { list.list.ExecuteIndirect(&signature, max, &bounded, at, count_buffer, count_offset) };
        }
        if kind == MESH {
            self.mesh_scissor(list, false);
        }
        if kind == DISPATCH {
            self.buffers_written();
        } else {
            self.drew(list, Some(p));
        }
        // The scratch buffer stays readable as arguments: the next draw of
        // the window reads it with no barrier, and the next clamp moves it
        // back (`crate::indirect`).
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
            if is_spread {
                SPREAD[1].fetch_add(1, Ordering::Relaxed);
                if ran_spread_lean {
                    SPREAD[2].fetch_add(1, Ordering::Relaxed);
                }
            }
            // The reflections' program also runs indirect. Only that role:
            // the estimate stays on the direct dispatch, as before.
            if let Some(crate::ambient::Role::Reflect) = p.stages.first().and_then(|s| s.shader.ambient)
            {
                let ran = self.ambient(list, p, crate::ambient::Role::Reflect);
                self.finish_ambient(ran);
            }
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// `DispatchX`: the counts are threads (the console's dispatch initiator bit
/// "use thread dimensions"; DexNom/VIctory bc3cd4b0). Taken for groups, a
/// rehydration stage of the structure builder ran up to 64 times its threads
/// and stored over the structures after its own (runs light-1 to light-9).
const THREAD_DIMENSIONS: u32 = 0x20;

/// `VICTORY_THREAD_DIMENSIONS=0`: such counts are taken for groups, as before.
fn thread_dimensions() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_THREAD_DIMENSIONS").is_none_or(|v| v != "0"))
}

/// The groups of `size` threads that hold `threads`.
fn dispatch_groups(threads: [u32; 3], size: [u32; 3]) -> [u32; 3] {
    std::array::from_fn(|i| threads[i].div_ceil(size[i]))
}

/// Where a compute pipeline's user data has a word the host computes: the
/// root parameter and the dword in it.
fn compute_word(p: &HostPipeline, kind: u32) -> Option<(u32, u32)> {
    let stage = p.stages.first().filter(|_| p.compute)?;
    let (param, _) = stage.user_data?;
    let word = stage.translated().host.user_data.iter().find(|u| u.root == RootType::Other(kind))?;
    Some((param, word.index))
}

#[cfg(test)]
mod spread_tests {
    use super::{spread_mode_of, spread_phase};

    #[test]
    fn spread_stays_off_unless_asked() {
        assert_eq!(spread_mode_of(None), 0);
        assert_eq!(spread_mode_of(Some("0")), 0);
        assert_eq!(spread_mode_of(Some("")), 0);
        assert_eq!(spread_mode_of(Some("1")), 1);
        assert_eq!(spread_mode_of(Some("alt")), 2);
        assert!(!spread_phase(0, 0));
        assert!(spread_phase(1, 0));
        assert!(!spread_phase(2, 0));
        assert!(!spread_phase(2, 19));
        assert!(spread_phase(2, 20));
        assert!(spread_phase(2, 39));
        assert!(!spread_phase(2, 40));
    }
}

#[cfg(test)]
mod thread_tests {
    use super::*;

    #[test]
    fn threads_are_counted_in_the_programs_groups() {
        assert_eq!(dispatch_groups([8847, 1, 1], [64, 1, 1]), [139, 1, 1]);
        assert_eq!(dispatch_groups([74, 1, 1], [64, 1, 1]), [2, 1, 1]);
        assert_eq!(dispatch_groups([32, 1, 1], [32, 1, 1]), [1, 1, 1]);
        assert_eq!(dispatch_groups([0, 12, 9], [64, 4, 4]), [0, 3, 3]);
    }
}

/// `VICTORY_SPREAD`: `1` on, `alt` twenty seconds off then twenty on, anything
/// else (including unset) off. Off, dispatches stay as the title asked.
fn spread_mode_of(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        Some("1") => 1,
        Some("alt") => 2,
        _ => 0,
    }
}

/// Whether spreading is on at `secs` into an `alt` run (`mode` 2), always
/// (`1`), or never.
fn spread_phase(mode: u8, secs: u64) -> bool {
    match mode {
        1 => true,
        2 => secs / 20 % 2 == 1,
        _ => false,
    }
}

fn spread_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| spread_mode_of(std::env::var("VICTORY_SPREAD").ok().as_deref()))
}

/// `VICTORY_SPREAD_LEAN`, read as [`spread_mode_of`] reads `VICTORY_SPREAD`:
/// unset, a spread indirect dispatch runs every record through the spread
/// signature, as before.
fn spread_lean_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| spread_mode_of(std::env::var("VICTORY_SPREAD_LEAN").ok().as_deref()))
}

/// Whether a spread indirect dispatch of at most
/// [`crate::indirect::SPREAD_LEAN_RECORDS`] records runs spread lean
/// ([`spread_lean_mode`]).
fn spread_lean_on() -> bool {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
    switch_now(spread_lean_mode(), &EPOCH, &LAST, |on| {
        tracing::info!(
            "indirect dispatches spread across z run lean from now (VICTORY_SPREAD_LEAN=alt): {}",
            if on { "ON" } else { "off, every record through the spread signature" }
        );
    })
}

fn spread_on() -> bool {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
    switch_now(spread_mode(), &EPOCH, &LAST, |on| {
        tracing::info!(
            "dispatches past 65,535 groups are spread across z from now (VICTORY_SPREAD=alt): {}",
            if on { "ON" } else { "off" }
        );
    })
}

/// Whether a switch of `mode` ([`spread_mode_of`]) is on now: in an `alt`
/// run by the seconds since `epoch` was first asked, `say` told each change
/// (`last` the phase told last).
fn switch_now(
    mode: u8,
    epoch: &std::sync::OnceLock<std::time::Instant>,
    last: &std::sync::atomic::AtomicU8,
    say: impl FnOnce(bool),
) -> bool {
    if mode != 2 {
        return spread_phase(mode, 0);
    }
    let on = spread_phase(2, epoch.get_or_init(std::time::Instant::now).elapsed().as_secs());
    if last.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
        say(on);
    }
    on
}

/// The user-data root parameter of compute pipeline `p` whose program takes
/// the spread words, and the index of the first (the second follows: one
/// root-constant argument sets both). `None` when the switch is off or the
/// program was not translated with them.
fn spreads(p: &HostPipeline) -> Option<(u32, u32)> {
    if !spread_on() {
        return None;
    }
    let stage = p.stages.first().filter(|_| p.compute)?;
    let (param, _) = stage.user_data?;
    let kind = RootType::Other(victory_shader::emit::USER_DISPATCH_SPREAD);
    let words = &stage.translated().host.user_data;
    let first = words.iter().find(|u| u.root == kind && u.dword == 0)?.index;
    words
        .iter()
        .any(|u| u.root == kind && u.dword == 1 && u.index == first + 1)
        .then_some((param, first))
}

/// The command signature of a spread dispatch: two root constants (row,
/// asked x) at `index` of root parameter `param`, then the dispatch; 20
/// bytes a record. Made once per root signature, parameter and index.
/// (After upstream VIctory pull request #2312.)
fn host_spread_signature(root: &ID3D12RootSignature, param: u32, index: u32) -> Option<ID3D12CommandSignature> {
    type Key = (usize, u32, u32);
    static MADE: std::sync::Mutex<Vec<(Key, ID3D12CommandSignature)>> = std::sync::Mutex::new(Vec::new());
    let key = (root.as_raw() as usize, param, index);
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == key) {
        return Some(s.clone());
    }
    let args = [
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                    RootParameterIndex: param,
                    DestOffsetIn32BitValues: index,
                    Num32BitValuesToSet: 2,
                },
            },
        },
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        },
    ];
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: 20,
        NumArgumentDescs: args.len() as u32,
        pArgumentDescs: args.as_ptr(),
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: two valid argument descriptions and a live root signature.
    let result = unsafe { crate::host::host()?.device.CreateCommandSignature(&desc, Some(root), &mut signature) };
    if let Err(e) = result {
        once!(error = %e, param, index, "spread command signature refused");
        return None;
    }
    let signature = signature?;
    made.push((key, signature.clone()));
    Some(signature)
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// The title resource `view` names, when it may be bound as `kind`.
/// `match_kind` ([`crate::ambient::depth_match`]): an SRV has to be a shader
/// view and a UAV an unordered view. Otherwise either of those views is
/// accepted, whichever kind was asked for, as before. Anything else is not
/// a resource view.
fn view_resource(kind: ResourceType, view: &View, match_kind: bool) -> Option<usize> {
    let (resource, suits) = match view {
        View::Shader(r, _) => (*r, kind == ResourceType::Srv),
        View::Unordered(r, ..) => (*r, kind == ResourceType::Uav),
        _ => return None,
    };
    if match_kind && !suits { None } else { Some(resource) }
}

#[cfg(test)]
mod view_bind_tests {
    use super::view_resource;
    use crate::descriptor::View;
    use victory_shader::ResourceType;

    /// A UAV where an SRV was asked for is bound only while the depth match
    /// is off. That binding is an invalid handle.
    #[test]
    fn a_descriptor_has_to_match_its_kind_only_when_asked() {
        let srv = View::Shader(4, None);
        let uav = View::Unordered(8, 0, None);
        let sampler = View::Sampler([0; 13]);
        assert_eq!(view_resource(ResourceType::Srv, &srv, false), Some(4));
        assert_eq!(view_resource(ResourceType::Srv, &uav, false), Some(8));
        assert_eq!(view_resource(ResourceType::Uav, &srv, false), Some(4));
        assert_eq!(view_resource(ResourceType::Cbv, &srv, false), Some(4));
        assert_eq!(view_resource(ResourceType::Srv, &sampler, false), None);
        assert_eq!(view_resource(ResourceType::Srv, &srv, true), Some(4));
        assert_eq!(view_resource(ResourceType::Srv, &uav, true), None);
        assert_eq!(view_resource(ResourceType::Uav, &uav, true), Some(8));
        assert_eq!(view_resource(ResourceType::Uav, &srv, true), None);
        assert_eq!(view_resource(ResourceType::Cbv, &srv, true), None);
        assert_eq!(view_resource(ResourceType::Srv, &sampler, true), None);
    }
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
pub(crate) fn view_kind(view: &View) -> Option<ResourceType> {
    match view {
        View::Shader(..) | View::Placed(false, ..) => Some(ResourceType::Srv),
        View::Unordered(..) | View::Placed(true, ..) => Some(ResourceType::Uav),
        _ => None,
    }
}

/// The host state a depth view needs of its texture: the read state if the
/// view writes no plane the format has (the title's read-only flags), else
/// the write state.
/// shortcut: whole textures have one state, so a view that writes one of
/// two planes counts as writing.
/// The format of a host view of `texture` whose description names
/// `described` (none without a description): that one, or the texture's
/// own, which is what the host gives a view without a format.
fn bound_format(texture: &ID3D12Resource, described: Option<DXGI_FORMAT>) -> u32 {
    match described {
        Some(format) if format.0 != 0 => format.0 as u32,
        // SAFETY: a live texture.
        _ => unsafe { texture.GetDesc() }.Format.0 as u32,
    }
}

fn depth_state(texture: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> u32 {
    // D16_UNORM and D32_FLOAT: depth formats without a stencil plane.
    const ONE_PLANE: [i32; 2] = [55, 40];
    let Some(desc) = desc else {
        return DEPTH_WRITE;
    };
    let format = if desc.Format.0 == 0 {
        // SAFETY: a live texture; a view without a format has its own.
        unsafe { texture.GetDesc() }.Format.0
    } else {
        desc.Format.0
    };
    let mut needed = D3D12_DSV_FLAG_READ_ONLY_DEPTH.0;
    if !ONE_PLANE.contains(&format) {
        needed |= D3D12_DSV_FLAG_READ_ONLY_STENCIL.0;
    }
    if desc.Flags.0 & needed == needed {
        DEPTH_READ
    } else {
        DEPTH_WRITE
    }
}

/// A few things, up to `N` of them without memory of the heap: what a draw's
/// tables are gone through with, some six lists a stage.
pub(crate) struct Few<T, const N: usize> {
    few: [T; N],
    len: usize,
    /// All of them, if they are more.
    many: Vec<T>,
}

impl<T: Copy + Default, const N: usize> Default for Few<T, N> {
    fn default() -> Self {
        Few { few: [T::default(); N], len: 0, many: Vec::new() }
    }
}

impl<T: Copy, const N: usize> Few<T, N> {
    fn push(&mut self, what: T) {
        if self.many.is_empty() && self.len < N {
            self.few[self.len] = what;
            self.len += 1;
        } else {
            if self.many.is_empty() {
                self.many.extend_from_slice(&self.few);
            }
            self.many.push(what);
        }
    }
}

impl<T, const N: usize> std::ops::Deref for Few<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        if self.many.is_empty() { &self.few[..self.len] } else { &self.many }
    }
}

impl<T, const N: usize> std::ops::DerefMut for Few<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        if self.many.is_empty() { &mut self.few[..self.len] } else { &mut self.many }
    }
}

/// A table made afresh in dynamic slots, in a list ([`State::tables`]).
struct FreshTable {
    first: u32,
    /// The list's epoch ([`HostList::epoch`]) at which all its entries that
    /// these readers (a stage's list of what it reads, whether it stores, the
    /// tested depth target) read were brought up to date, and whether that read the
    /// tested depth target.
    epoch: u64,
    ready: Vec<((usize, usize, bool, usize), bool)>,
}

/// Whether a table a list reads again, with nothing changed in the list
/// since, is left as it is (`VICTORY_READY_ONCE=0`: every entry is gone
/// through again at every draw).
fn ready_once() -> bool {
    crate::toggled!("VICTORY_READY_ONCE")
}

/// A hash of records' words: 64 bits, two multiplications a record. (The
/// standard library's took 3 % of the recording threads' time.)
#[derive(Default)]
struct RecordHash(u64);

impl RecordHash {
    fn add(&mut self, [a, b, c]: [u32; 3]) {
        let fold = |x: u64, y: u64| {
            let p = u128::from(x) * u128::from(y);
            p as u64 ^ (p >> 64) as u64
        };
        let words = u64::from(a) << 32 | u64::from(b);
        let first = fold(self.0 ^ words ^ 0x2d35_8dcc_aa6c_78a5, 0x8bb8_4b93_962e_acc9 ^ u64::from(c));
        self.0 = fold(first ^ 0x4b33_a62e_d433_d4a3, 0x4d5a_2da5_1de1_aa47 ^ self.0.rotate_left(23)) ^ first;
    }
}

fn readable(list: &HostList, kind: ResourceType, view: &View) {
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(view) {
        shadow.refresh(list, address, len);
        shadow.read_as(crate::readkinds::TABLE_VIEW, len.min(shadow.left(address)));
        if kind == ResourceType::Uav {
            // A shader that cannot store to buffers leaves the range as
            // the title has it: nothing to find afterwards.
            if STORES.get() || !write_skip() {
                #[cfg(feature = "diagnostics")]
                crate::host::wide_write(address, len, 0);
                shadow.stored(crate::readkinds::STORE_TABLE, address, len);
                shadow.gpu_writes(list, address, len);
            } else {
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOut, 1);
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOutBytes, len);
                }
            }
        }
    }
    match (view, kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            // A view the title placed, in another format, in the memory of
            // a texture the host writes: that texture's texels, copied on
            // the host (title memory does not have them).
            let aliased = crate::resource::alias_of(*resource as This)
                .is_some_and(|source| crate::texture::alias(list, *resource as This, source.get()));
            if !aliased
                && let Some(t) = crate::resource::host_copy(*resource as This)
                && t.flags() & 0x7 == 0
            {
                // (A texture the title writes: no target, depth or unordered
                // access, `crate::texture::title_written`.)
                // Of a texture the title placed, the levels the view shows.
                let first = match (&placed, view) {
                    (Some(_), View::Shader(_, Some(desc))) => crate::resource::first_mip(desc),
                    _ => 0,
                };
                crate::texture::upload_from(list, *resource as This, first);
            }
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(*resource as This, false, false);
            // The depth target the draw only tests is read in both ways at
            // once: both read states, which its preparation asks for too.
            let (tested, _) = TESTED_DEPTH.get();
            let state = if tested != 0 && tested == *resource {
                TESTED_DEPTH.set((tested, true));
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            crate::list::ensure_state(list, *resource as This, state);
            #[cfg(feature = "diagnostics")]
            if tracing::enabled!(tracing::Level::DEBUG)
                && let Some(t) = crate::resource::host_resource(*resource as This)
            {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            // Some compute-written textures are also read by the title CPU.
            list.texture_written(*resource as This);
            // Unordered access to a depth texture is its colour twin's.
            if !list.twin_use(*resource as This, true, true) {
                crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            }
        }
        _ => {}
    }
}

/// The pixel shaders whose draws get their slots listed
/// ([`State::slots_text`]): `VICTORY_LIST_SLOTS`, hash prefixes separated
/// by commas. None by default.
#[cfg(feature = "diagnostics")]
fn list_slots() -> &'static [String] {
    static LIST: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        std::env::var("VICTORY_LIST_SLOTS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    })
}

/// A fingerprint of the 4 KiB of title memory at `at` (FNV-1a, 64 bits; of
/// 4 KiB of zeros it is always the same value, which `slots.py` knows), or
/// "unreadable".
#[cfg(feature = "diagnostics")]
fn print_4k(at: u64) -> String {
    let mut bytes = [0u8; 4096];
    if at == 0 || !crate::host::read_into(at as *const u8, &mut bytes) {
        return "unreadable".into();
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for b in bytes {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// The `bytes` of title memory at `at`: how many are mapped, and the
/// fingerprints of the first and the last 4 KiB (a texture's smallest
/// levels lie first, its largest last).
#[cfg(feature = "diagnostics")]
fn memory_state(at: u64, bytes: u64) -> String {
    let mapped = crate::texture::mapped_bytes(at as *const u8, bytes as usize);
    let last = at + bytes.saturating_sub(4096);
    format!("memory={at:#x} mapped={mapped}/{bytes} first4k={} last4k={}", print_4k(at), print_4k(last))
}

/// Title texture `this` for the frame listing: its size, whether it has a
/// host copy (not made here), its staging buffer's state, whether the host
/// copied into it, and its memory ([`memory_state`]).
#[cfg(feature = "diagnostics")]
fn texture_state(this: This) -> String {
    let Some((at, desc)) = crate::resource::texture_memory(this) else {
        return "not a texture".into();
    };
    // SAFETY: 56 bytes in the description's layout.
    let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
    let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
    let staged = match crate::texture::staged_state(this) {
        Some((uploaded, dirty)) => format!("staged=yes uploaded={uploaded} dirty={dirty}"),
        None => "staged=no".into(),
    };
    format!(
        "{}x{} f{} levels={} host_copy={} {staged} host_written={} title_fills={} {}",
        u64::from(words[4]) | u64::from(words[5]) << 32,
        words[6],
        words[8],
        words[7] >> 16,
        crate::resource::host_made(this).is_some(),
        crate::texture::host_wrote(this as usize),
        crate::resource::title_fills(this),
        memory_state(at as u64, bytes)
    )
}

/// A view of the title's heap for the frame listing: a texture with its
/// state ([`texture_state`]; a placed one that has no resource yet with
/// its memory alone), anything else in a few words.
#[cfg(feature = "diagnostics")]
fn view_state(view: Option<&View>) -> String {
    match view {
        None => "no view".into(),
        Some(View::Placed(false, r, d)) if r[0] != 1 => {
            let address = u64::from(d[10]) | u64::from(d[11]) << 32;
            let state = match crate::resource::texture_at_made(address, r) {
                Some(this) => texture_state(this),
                None => format!(
                    "{}x{} f{} levels={} resource=none {}",
                    u64::from(r[4]) | u64::from(r[5]) << 32,
                    r[6],
                    r[8],
                    r[7] >> 16,
                    memory_state(address, crate::resource::placed_bytes(r).unwrap_or(0))
                ),
            };
            // (Last, so that readers of the older lines still find
            // theirs: the level the view starts at, and how many it shows.)
            format!("placed texture view_format={} {state} view_levels={}+{}", d[0], d[4], d[5] as i32)
        }
        Some(View::Shader(r, d)) if crate::resource::texture_memory(*r as This).is_some() => {
            format!(
                "texture object {r:#x} view_format={} {} view_levels={}",
                d.as_ref().map_or(0, |d| d[0]),
                texture_state(*r as This),
                d.as_ref().map_or("all".to_owned(), |d| format!("{}+{}", d[4], d[5] as i32))
            )
        }
        Some(v) => format!("{v:?}").chars().take(80).collect(),
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_made(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{}x{} f{} flags {:#x} of {r:?}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        // (with the title's view description: a typed view's DXGI format first, then
        // dimension, first element, element count, stride, counter offset, flags)
        Some(v @ View::Unordered(r, _, desc)) => match texture(*r as This) {
            Some(t) => format!("uav texture {t} view {desc:?}"),
            None => format!(
                "uav resource {r:#x} buffer {:?} view {desc:?}",
                crate::descriptor::host::buffer_range(v)
            ),
        },
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || format!("STENCIL REF {reference:#x}"));
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

pub(crate) fn depth_bounds(list: &HostList, min: f32, max: f32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || {
        format!("DEPTH BOUNDS {min}..{max}")
    });
    if crate::host::host().is_some_and(|h| h.depth_bounds())
        && let Ok(l1) = list.list.cast::<ID3D12GraphicsCommandList1>()
    {
        // SAFETY: an open list on a device that supports depth bounds.
        unsafe { l1.OMSetDepthBounds(min, max) };
    }
}

/// The longest wait, in milliseconds, between two looks at a texture slot
/// whose memory is not there yet (`VICTORY_TEXTURE_LOOK_MS`; unset: about
/// four seconds counted in passes, as before). The looks back off 1, 2, 4,
/// ... frames up to it. Measured (run light-22, `crate::texlate`): from
/// the last look that found a slot's memory away to its upload, half under
/// 0.5 s but nine in ten under 32 s: the memory had come long before.
fn look_ms() -> Option<u64> {
    static MS: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        let ms = std::env::var("VICTORY_TEXTURE_LOOK_MS").ok()?.trim().parse::<u64>().ok().filter(|&ms| ms > 0)?;
        tracing::info!(ms, "texture slots waiting for their memory are looked at at least this often (VICTORY_TEXTURE_LOOK_MS)");
        Some(ms)
    })
}

/// The milliseconds of a frame lately (a mean over the last frames), from
/// the pass's frame number and clock.
fn frame_ms(frame: u64, now: u64) -> u64 {
    use std::sync::Mutex;
    static LAST: Mutex<(u64, u64, f64)> = Mutex::new((0, 0, 50.0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let (was_frame, was_ms, mean) = *last;
    if frame > was_frame {
        if was_frame != 0 {
            let each = now.saturating_sub(was_ms) as f64 / (frame - was_frame) as f64;
            last.2 = mean * 0.9 + each.clamp(1.0, 2000.0) * 0.1;
        }
        last.0 = frame;
        last.1 = now;
    }
    (last.2 as u64).max(1)
}

/// Per-frame texture staging byte budget scaled from the adapter's local VRAM
/// budget (`None` keeps the 192 MiB 12 GiB baseline).
fn staging_budget_per_frame(vram_budget: Option<u64>) -> u64 {
    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    match vram_budget {
        Some(budget) if budget <= 6 * GIB => 96 * MIB,
        Some(budget) if budget >= 20 * GIB => 384 * MIB,
        Some(budget) if budget > 12 * GIB => 256 * MIB,
        _ => 192 * MIB,
    }
}

#[cfg(test)]
mod tests {
    use super::staging_budget_per_frame;

    /// Indirect executions in place count by kind: dispatches, mesh draws,
    /// and every other kind as a draw; the order switch is off unset.
    #[test]
    fn in_place_indirect_executions_count_by_kind() {
        assert_eq!([0, 1, 2, 10, 7].map(super::in_place_slot), [0, 0, 1, 2, 0]);
        if std::env::var_os("VICTORY_PORT_CONSTANTS_ORDER").is_none() {
            assert!(!super::in_place_order());
        }
    }
    use super::{RECHECK_FRAMES, late_look, long_for, next_look, recheck_long_of};

    /// The five looks of before, 576 frames in all; with
    /// `VICTORY_RECHECK_LONG` they go on at its interval for as long as the
    /// slot is looked at, and every such look is a late one.
    #[test]
    fn looks_after_an_upload_end_or_go_on() {
        // As before: looks 0 to 4, then done.
        let mut frames = RECHECK_FRAMES[0];
        let mut looks = 0u8;
        while let Some((wait, next)) = next_look(looks, 0) {
            assert_eq!(next, looks + 1);
            assert!(!late_look(looks));
            frames += wait;
            looks = next;
        }
        assert_eq!((looks, frames), (4, 576));
        assert!(!late_look(4));
        // Going on: the same five, then one every 600 frames, without end
        // and without the count running over.
        let mut looks = 0u8;
        let mut waits = Vec::new();
        for _ in 0..1000 {
            let (wait, next) = next_look(looks, 600).expect("no last look");
            waits.push(wait);
            looks = next;
        }
        assert_eq!(waits[..6], [12, 40, 120, 400, 600, 600]);
        assert!(waits[4..].iter().all(|&w| w == 600));
        assert_eq!(looks, 5);
        assert!(late_look(5));
    }

    /// `VICTORY_ALIASED_RECHECK` (kitty ae3cce821): a texture shown at
    /// other addresses too keeps being looked at after the schedule, every
    /// so many frames; any other, or with the switch unset, ends as before;
    /// `VICTORY_RECHECK_LONG` when set goes for all.
    #[test]
    fn aliased_textures_are_looked_at_after_the_schedule() {
        let last = RECHECK_FRAMES.len() as u8 - 1;
        assert_eq!(next_look(last, long_for(0, 30, || true)), Some((30, RECHECK_FRAMES.len() as u8)));
        assert_eq!(next_look(last, long_for(0, 30, || false)), None, "not shown elsewhere");
        assert_eq!(next_look(last, long_for(0, 0, || panic!("asked while off"))), None, "unset");
        assert_eq!(long_for(600, 30, || panic!("asked with the long looks on")), 600);
    }

    /// `VICTORY_RECHECK_LONG`: unset, 0 or not a number is off; 1 is 600
    /// frames; a number is that many, at least 60.
    #[test]
    fn recheck_long_reads_its_value() {
        for off in [None, Some(""), Some("0"), Some("on"), Some("-5")] {
            assert_eq!(recheck_long_of(off), 0, "{off:?}");
        }
        assert_eq!(recheck_long_of(Some("1")), 600);
        assert_eq!(recheck_long_of(Some(" 1200 ")), 1200);
        assert_eq!(recheck_long_of(Some("2")), 60);
        assert_eq!(recheck_long_of(Some("60")), 60);
    }

    #[test]
    fn staging_budget_scales_with_vram_budget() {
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;
        assert_eq!(staging_budget_per_frame(None), 192 * MIB);
        assert_eq!(staging_budget_per_frame(Some(4 * GIB)), 96 * MIB);
        assert_eq!(staging_budget_per_frame(Some(6 * GIB)), 96 * MIB);
        assert_eq!(staging_budget_per_frame(Some(8 * GIB)), 192 * MIB);
        assert_eq!(staging_budget_per_frame(Some(12 * GIB)), 192 * MIB);
        assert_eq!(staging_budget_per_frame(Some(13 * GIB)), 256 * MIB);
        assert_eq!(staging_budget_per_frame(Some(14 * GIB)), 256 * MIB);
        assert_eq!(staging_budget_per_frame(Some(16 * GIB)), 256 * MIB);
        assert_eq!(staging_budget_per_frame(Some(20 * GIB)), 384 * MIB);
        assert_eq!(staging_budget_per_frame(Some(24 * GIB)), 384 * MIB);
        assert_eq!(staging_budget_per_frame(Some(32 * GIB)), 384 * MIB);
    }
}

#[cfg(test)]
mod record_hash_tests {
    use super::{Few, RecordHash};

    #[test]
    fn a_few_things_are_kept_in_order_however_many() {
        for n in 0..10u32 {
            let mut few: Few<u32, 4> = Few::default();
            (0..n).for_each(|i| few.push(n - i));
            assert_eq!(few.to_vec(), (0..n).map(|i| n - i).collect::<Vec<_>>());
            few.sort_unstable();
            assert_eq!(few.to_vec(), (1..=n).collect::<Vec<_>>());
        }
    }

    #[test]
    fn records_that_differ_hash_apart() {
        let of = |records: &[[u32; 3]]| {
            let mut h = RecordHash::default();
            records.iter().for_each(|r| h.add(*r));
            h.0
        };
        let mut seen = std::collections::HashSet::new();
        // One record, one word or two of it different; runs of zeros; order.
        for a in 0..64u32 {
            for b in 0..64u32 {
                for c in [0, 1, 0x8000_0000, u32::MAX] {
                    assert!(seen.insert(of(&[[a, b, c]])));
                    assert!(seen.insert(of(&[[1, 2, 3], [a, b, c]])));
                    assert!(seen.insert(of(&[[a, b, c], [1, 2, 3], [0, 0, 0]])));
                }
            }
        }
        assert!(seen.insert(of(&[])));
        // (One record of zeros is among the above.)
        let zeros: Vec<u64> = (2..100).map(|n| of(&vec![[0; 3]; n])).collect();
        assert!(zeros.iter().all(|h| seen.insert(*h)));
    }
}

#[cfg(test)]
mod table_stage_tests {
    use super::{fold_stage_span, staged_slots, table_stages_mode, State};

    /// Off unless the switch is `1`, `alt` or `alt40`.
    #[test]
    fn targets_kept_is_off_unless_asked() {
        assert!(!super::targets_kept_from(None));
        assert!(!super::targets_kept_from(Some("0")));
        assert!(!super::targets_kept_from(Some("")));
        assert!(super::targets_kept_from(Some("1")));
    }

    #[test]
    fn table_stages_is_off_unless_asked() {
        assert_eq!(table_stages_mode(None), 0);
        assert_eq!(table_stages_mode(Some("0")), 0);
        assert_eq!(table_stages_mode(Some("")), 0);
        assert_eq!(table_stages_mode(Some("yes")), 0);
        assert_eq!(table_stages_mode(Some("1")), 1);
        assert_eq!(table_stages_mode(Some("alt")), 2);
        assert_eq!(table_stages_mode(Some("alt40")), 3);
    }

    /// Two stages that read one table to different depths, both through a
    /// constant, ask for the longer of those reads. A stage with no constant
    /// does not lengthen that, unless the table runs past the host's slots,
    /// where every stage makes it afresh and they share the whole read.
    /// With the switch off the stage keeps its own count.
    #[test]
    fn stages_that_share_a_table_ask_for_the_longer_read() {
        let mut spans = Vec::new();
        // Dwords 24 and 72: 4 and 10 slots.
        fold_stage_span(&mut spans, 0, true, 24);
        fold_stage_span(&mut spans, 0, true, 72);
        assert_eq!(spans, vec![(0, 10, 10)]);
        // Dword 80: 11 slots, and no constant in that stage.
        fold_stage_span(&mut spans, 0, false, 80);
        assert_eq!(spans, vec![(0, 10, 11)]);
        assert_eq!(staged_slots(4, None, Some(0), 1000), 4);
        assert_eq!(staged_slots(4, Some((10, 11)), Some(0), 1000), 10);
        assert_eq!(staged_slots(4, Some((10, 11)), Some(995), 1000), 11);
        assert_eq!(staged_slots(4, Some((0, 4)), Some(0), 1000), 4);
    }

    /// Slots a list holds after it was submitted are counted until it has
    /// run, and a second count of the same list does not add them again.
    /// Dropping the list without that subtracts them too.
    #[test]
    fn slots_a_submitted_list_holds_are_counted_until_it_has_run() {
        let before = crate::descriptor::host::flight_slots();
        {
            let mut state = State::default();
            state.dynamic.push((1, 0, 4, 10));
            state.dynamic.push((1, 0, 20, 5));
            state.hold_in_flight();
            state.hold_in_flight();
            assert_eq!(crate::descriptor::host::flight_slots(), before + 15);
            state.tables_done();
            assert_eq!(crate::descriptor::host::flight_slots(), before);
        }
        assert_eq!(crate::descriptor::host::flight_slots(), before);
        {
            let mut state = State::default();
            state.dynamic.push((1, 0, 0, 7));
            state.hold_in_flight();
            assert_eq!(crate::descriptor::host::flight_slots(), before + 7);
        }
        assert_eq!(crate::descriptor::host::flight_slots(), before);
    }
}
