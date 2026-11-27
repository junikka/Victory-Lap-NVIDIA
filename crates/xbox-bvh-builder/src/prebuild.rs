//! Conservative sizes for `GetRaytracingAccelerationStructurePrebuildInfo`.
//!
//! Microsoft does not publish one closed formula. The driver fills
//! `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO` (three `UINT64`s:
//! result, scratch, update scratch). What is public, in `d3d12.h` and the
//! Direct3D 12 raytracing docs:
//!
//! - Buffers and the returned sizes are aligned to
//!   `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT` (256).
//! - A top-level structure has `NumDescs` instances. `InstanceDescs` is a GPU
//!   virtual address, so this code does not read it.
//! - A bottom-level triangle geometry uses `IndexCount / 3` primitives when
//!   `IndexCount` is non-zero, and `VertexCount / 3` otherwise
//!   (`D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC`).
//! - A bottom-level AABB geometry uses `AABBCount`.
//!
//! The numbers below are our budget, shaped as a fixed header plus a
//! per-primitive cost, then rounded up to 256. They are deliberately larger
//! than a packed BVH so a caller that rejects a zero size still gets room.

use std::mem::{offset_of, size_of};
use std::ptr;

/// Fixed header, in bytes. Sixteen alignment quanta.
const HEADER: u64 = 4096;
/// Bytes of result storage budgeted for one primitive (one alignment quantum).
const BYTES_PER_PRIMITIVE: u64 = 256;
/// Extra result bytes per vertex. Vertices are not primitives, but the build
/// has to account for them.
const BYTES_PER_VERTEX: u64 = 16;
/// Extra scratch bytes per primitive, on top of the result size.
const SCRATCH_PER_PRIMITIVE: u64 = 128;

const ALIGN: u64 = 256;

const TYPE_TOP_LEVEL: u32 = 0;
const TYPE_BOTTOM_LEVEL: u32 = 1;
const TYPE_OPACITY_MICROMAP_ARRAY: u32 = 2;

const GEOMETRY_TRIANGLES: u32 = 0;
const GEOMETRY_AABBS: u32 = 1;
const GEOMETRY_OMM_TRIANGLES: u32 = 2;

const LAYOUT_ARRAY: u32 = 0;
const LAYOUT_ARRAY_OF_POINTERS: u32 = 1;

const FLAG_ALLOW_UPDATE: u32 = 0x1;
const FLAG_PREFER_FAST_TRACE: u32 = 0x4;
const FLAG_PREFER_FAST_BUILD: u32 = 0x8;
const FLAG_MINIMIZE_MEMORY: u32 = 0x10;
const FLAG_PERFORM_UPDATE: u32 = 0x20;
const FLAG_ALLOW_OMM_UPDATE: u32 = 0x40;

const UPDATE_FLAGS: u32 = FLAG_ALLOW_UPDATE | FLAG_PERFORM_UPDATE | FLAG_ALLOW_OMM_UPDATE;

/// `D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC` — 48 bytes in the Windows SDK.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct TrianglesDesc {
    transform: u64,
    index_format: u32,
    vertex_format: u32,
    index_count: u32,
    vertex_count: u32,
    index_buffer: u64,
    vertex_buffer: u64,
    vertex_stride: u64,
}

/// `D3D12_RAYTRACING_GEOMETRY_AABBS_DESC` — 24 bytes. Overlaps the triangle desc.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct AabbsDesc {
    count: u64,
    start: u64,
    stride: u64,
}

/// `D3D12_RAYTRACING_GEOMETRY_OMM_TRIANGLES_DESC` — two CPU pointers.
#[repr(C)]
#[derive(Clone, Copy)]
struct OmmTrianglesDesc {
    triangles: u64,
    linkage: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
union GeometryBody {
    triangles: TrianglesDesc,
    aabbs: AabbsDesc,
    omm: OmmTrianglesDesc,
}

/// `D3D12_RAYTRACING_GEOMETRY_DESC` — 56 bytes, 8-byte aligned.
#[repr(C)]
#[derive(Clone, Copy)]
struct GeometryDesc {
    ty: u32,
    flags: u32,
    body: GeometryBody,
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct OpacityHistogramEntry {
    count: u32,
    subdivision_level: u32,
    format: u32,
}

/// `D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC` — 40 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct OpacityArrayDesc {
    num_entries: u32,
    _pad: u32,
    histogram: u64,
    input_buffer: u64,
    per_omm_start: u64,
    per_omm_stride: u64,
}

/// `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS` — 24 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct BuildInputs {
    ty: u32,
    flags: u32,
    num_descs: u32,
    layout: u32,
    /// GPU virtual address for a top-level build. CPU pointer for bottom-level
    /// geometry descriptors and for an opacity micromap array descriptor.
    descs: u64,
}

/// `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrebuildInfo {
    pub result: u64,
    pub scratch: u64,
    pub update: u64,
}

pub(crate) struct Budget {
    pub info: PrebuildInfo,
    pub primitives: u64,
    pub vertices: u64,
    pub flags: u32,
}

const _: () = assert!(size_of::<TrianglesDesc>() == 48);
const _: () = assert!(offset_of!(TrianglesDesc, index_count) == 16);
const _: () = assert!(offset_of!(TrianglesDesc, vertex_count) == 20);
const _: () = assert!(size_of::<AabbsDesc>() == 24);
const _: () = assert!(size_of::<GeometryDesc>() == 56);
const _: () = assert!(std::mem::align_of::<GeometryDesc>() == 8);
const _: () = assert!(offset_of!(GeometryDesc, body) == 8);
const _: () = assert!(size_of::<OpacityHistogramEntry>() == 12);
const _: () = assert!(size_of::<OpacityArrayDesc>() == 40);
const _: () = assert!(offset_of!(OpacityArrayDesc, histogram) == 8);
const _: () = assert!(size_of::<BuildInputs>() == 24);
const _: () = assert!(offset_of!(BuildInputs, num_descs) == 8);
const _: () = assert!(offset_of!(BuildInputs, descs) == 16);
const _: () = assert!(size_of::<PrebuildInfo>() == 24);

fn align_256(n: u64) -> u64 {
    let n = n.max(ALIGN);
    // `u64::MAX` saturates; masking still yields a non-zero multiple of 256.
    n.saturating_add(ALIGN - 1) & !(ALIGN - 1)
}

fn per_primitive_bytes(flags: u32) -> u64 {
    let minimize = flags & FLAG_MINIMIZE_MEMORY != 0;
    let fast_trace = flags & FLAG_PREFER_FAST_TRACE != 0;
    match (minimize, fast_trace) {
        (true, false) => BYTES_PER_PRIMITIVE / 2,
        (false, true) => BYTES_PER_PRIMITIVE * 2,
        // Fast trace wants more memory and minimize wants less. Together they
        // stay on the default quantum. Neither flag is set: default.
        (false, false) | (true, true) => BYTES_PER_PRIMITIVE,
    }
}

fn budget_from_counts(primitives: u64, vertices: u64, flags: u32) -> PrebuildInfo {
    let result = align_256(
        HEADER
            .saturating_add(primitives.saturating_mul(per_primitive_bytes(flags)))
            .saturating_add(vertices.saturating_mul(BYTES_PER_VERTEX)),
    );
    let scratch_per_prim = if flags & FLAG_PREFER_FAST_BUILD != 0 {
        SCRATCH_PER_PRIMITIVE * 2
    } else {
        SCRATCH_PER_PRIMITIVE
    };
    let scratch = align_256(
        result
            .saturating_add(primitives.saturating_mul(scratch_per_prim))
            .saturating_add(HEADER),
    );
    let update = if flags & UPDATE_FLAGS != 0 {
        scratch
    } else {
        align_256(result / 2)
    };
    PrebuildInfo {
        result,
        scratch,
        update,
    }
}

fn triangle_counts(tri: &TrianglesDesc) -> (u64, u64) {
    let primitives = if tri.index_count > 0 {
        u64::from(tri.index_count).div_ceil(3)
    } else {
        u64::from(tri.vertex_count).div_ceil(3)
    };
    (primitives, u64::from(tri.vertex_count))
}

/// # Safety
/// `ptr` is null or points at a `D3D12_RAYTRACING_GEOMETRY_DESC`. For opacity
/// micromap triangles, the nested triangle pointer is null or points at a
/// `D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC`.
unsafe fn geometry_counts(ptr: *const GeometryDesc) -> (u64, u64) {
    if ptr.is_null() {
        return (0, 0);
    }
    let geom = ptr::read_unaligned(ptr);
    match geom.ty {
        GEOMETRY_TRIANGLES => triangle_counts(&geom.body.triangles),
        GEOMETRY_AABBS => (geom.body.aabbs.count, 0),
        GEOMETRY_OMM_TRIANGLES => {
            let triangles = geom.body.omm.triangles;
            if triangles == 0 {
                (1, 0)
            } else {
                let tri = ptr::read_unaligned(triangles as *const TrianglesDesc);
                triangle_counts(&tri)
            }
        }
        _ => (1, 0),
    }
}

/// # Safety
/// `descs` is a CPU pointer to `num_descs` geometry descriptors, or to an
/// array of that many pointers, matching `layout`. A null `descs` falls back
/// to `num_descs` and reads nothing.
unsafe fn bottom_level_counts(num_descs: u32, layout: u32, descs: u64) -> (u64, u64) {
    if num_descs == 0 {
        return (0, 0);
    }
    if descs == 0 || (layout != LAYOUT_ARRAY && layout != LAYOUT_ARRAY_OF_POINTERS) {
        return (u64::from(num_descs), 0);
    }
    let mut primitives = 0u64;
    let mut vertices = 0u64;
    for index in 0..num_descs {
        let ptr = if layout == LAYOUT_ARRAY_OF_POINTERS {
            let slot = ptr::read_unaligned((descs as *const u64).add(index as usize));
            slot as *const GeometryDesc
        } else {
            let addr = descs as usize + (index as usize) * size_of::<GeometryDesc>();
            addr as *const GeometryDesc
        };
        let (prims, verts) = geometry_counts(ptr);
        primitives = primitives.saturating_add(prims);
        vertices = vertices.saturating_add(verts);
    }
    (primitives, vertices)
}

/// # Safety
/// `descs` is null or points at a `D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC`
/// whose histogram pointer is null or has `num_entries` entries.
unsafe fn opacity_counts(descs: u64) -> (u64, u64) {
    if descs == 0 {
        return (1, 0);
    }
    let desc = ptr::read_unaligned(descs as *const OpacityArrayDesc);
    if desc.histogram == 0 || desc.num_entries == 0 {
        return (1, 0);
    }
    // A wild count must not walk off into the process. Past this cap, count
    // the unread entries as one micromap each.
    const MAX_WALK: u32 = 65_536;
    let walk = desc.num_entries.min(MAX_WALK);
    let mut primitives = 0u64;
    let entries = desc.histogram as *const OpacityHistogramEntry;
    for index in 0..walk {
        let entry = ptr::read_unaligned(entries.add(index as usize));
        primitives = primitives.saturating_add(u64::from(entry.count));
    }
    if desc.num_entries > walk {
        primitives = primitives.saturating_add(u64::from(desc.num_entries - walk));
    }
    (primitives.max(1), 0)
}

/// # Safety
/// `inputs` is null or points at a
/// `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS`. Bottom-level
/// descriptor pointers in that struct follow the same rules as
/// [`bottom_level_counts`].
pub(crate) unsafe fn budget(inputs: *const u8) -> Budget {
    if inputs.is_null() {
        return Budget {
            info: budget_from_counts(0, 0, 0),
            primitives: 0,
            vertices: 0,
            flags: 0,
        };
    }
    let inputs = ptr::read_unaligned(inputs.cast::<BuildInputs>());
    let (primitives, vertices) = match inputs.ty {
        TYPE_TOP_LEVEL => (u64::from(inputs.num_descs), 0),
        TYPE_BOTTOM_LEVEL => bottom_level_counts(inputs.num_descs, inputs.layout, inputs.descs),
        TYPE_OPACITY_MICROMAP_ARRAY => opacity_counts(inputs.descs),
        _ => (u64::from(inputs.num_descs), 0),
    };
    Budget {
        info: budget_from_counts(primitives, vertices, inputs.flags),
        primitives,
        vertices,
        flags: inputs.flags,
    }
}
