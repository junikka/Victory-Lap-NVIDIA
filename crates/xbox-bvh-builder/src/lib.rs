//! Stand-in for the Xbox system DLL `XboxBvhBuilder.dll`.
//!
//! The title calls `GetRaytracingAccelerationStructurePrebuildInfo` during
//! Game Init. A stub that leaves the out-struct at zero trips an assert.
//! This DLL fills that struct with a conservative non-zero size and ignores
//! the other entry points, so ray tracing is not built.
//!
//! The caller owns the `XboxBvhBuilder` object and we do not know its size,
//! so none of these functions write through `this`. Per-object state lives in
//! a side table keyed by the `this` pointer.
//!
//! `D3D12XBOX_XDXR_STREAM_DESC` is not in the public Windows SDK (`d3d12.h`)
//! and there is no public GDK header on this machine, so the stream pointer
//! is never read.
//!
//! This crate does not open a socket or call any network API.
//!
//! The crate is named `XboxBvhBuilder` so the built file is `XboxBvhBuilder.dll`.
#![allow(non_snake_case)]

#[macro_use]
mod caller;
mod log;
mod prebuild;

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::AtomicBool;
use std::sync::{Mutex, OnceLock};

use prebuild::{budget, PrebuildInfo};

fn guard(body: impl FnOnce()) {
    let _ = catch_unwind(AssertUnwindSafe(body));
}

const CTOR: &str = "XboxBvhBuilder::ctor";
const DTOR: &str = "XboxBvhBuilder::dtor";
const PREBUILD: &str = "XboxBvhBuilder::GetRaytracingAccelerationStructurePrebuildInfo";
const BUILD: &str = "XboxBvhBuilder::BuildRaytracingAccelerationStructureBatched2X";
const COPY: &str = "XboxBvhBuilder::CopyRaytracingAccelerationStructure";
const REHYDRATE: &str = "XboxBvhBuilder::RehydrateRaytracingAccelerationStructureBatched2X";
const MORTON: &str = "XboxBvhBuilder::Set64BitMortonCodes";

static LOGGED_BUILD: AtomicBool = AtomicBool::new(false);
static LOGGED_COPY: AtomicBool = AtomicBool::new(false);
static LOGGED_REHYDRATE: AtomicBool = AtomicBool::new(false);
static LOGGED_MORTON: AtomicBool = AtomicBool::new(false);
/// `VICTORY_BVH_LOG_ONCE=1` (ticket 186): the prebuild call, ~110 a frame, logs once like the other entry points.
static LOGGED_PREBUILD: AtomicBool = AtomicBool::new(false);

struct Instance {
    device: usize,
    flag: bool,
}

fn objects() -> &'static Mutex<BTreeMap<usize, Instance>> {
    static OBJECTS: OnceLock<Mutex<BTreeMap<usize, Instance>>> = OnceLock::new();
    OBJECTS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

asm_entry!("??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z" => ctor_entry);

/// # Safety
/// `this` is null or an address the caller owns. This function does not read
/// or write that memory. `device` is not called.
#[inline(never)]
unsafe extern "system" fn ctor_entry(this: *mut u8, device: *mut c_void, flag: u8) -> *mut u8 {
    guard(|| constructor(this, device, flag));
    this
}

fn constructor(this: *mut u8, device: *mut c_void, flag: u8) {
    if !this.is_null() {
        let mut map = objects().lock().unwrap_or_else(|err| err.into_inner());
        map.insert(
            this as usize,
            Instance {
                device: device as usize,
                flag: flag != 0,
            },
        );
    }
    log::log_event(
        CTOR,
        None,
        &format!("this={this:p} device={device:p} flag={flag}"),
    );
}

asm_entry!("??1XboxBvhBuilder@@QEAA@XZ" => dtor_entry);

/// # Safety
/// `this` is null or an address previously passed to the constructor. This
/// function does not read or write that memory.
#[inline(never)]
unsafe extern "system" fn dtor_entry(this: *mut u8) {
    guard(|| destructor(this));
}

fn destructor(this: *mut u8) {
    let removed = if this.is_null() {
        None
    } else {
        let mut map = objects().lock().unwrap_or_else(|err| err.into_inner());
        map.remove(&(this as usize))
    };
    let detail = match removed {
        Some(instance) => {
            let device = instance.device;
            let flag = instance.flag;
            format!("this={this:p} device={device:#x} flag={flag}")
        }
        None => format!("this={this:p} no-instance"),
    };
    log::log_event(DTOR, None, &detail);
}

asm_entry!("?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z" => prebuild_entry);

/// # Safety
/// `inputs` is null or points at a
/// `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS`, including any
/// bottom-level geometry descriptors it names. `out` is null or points at a
/// `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO`. `stream` is not read.
#[inline(never)]
unsafe extern "system" fn prebuild_entry(
    this: *mut u8,
    inputs: *const u8,
    stream: *const u8,
    out: *mut u8,
) {
    guard(|| unsafe { prebuild_body(this, inputs, stream, out.cast()) });
}

/// # Safety
/// Same contract as [`get_prebuild_info`].
unsafe fn prebuild_body(
    this: *mut u8,
    inputs: *const u8,
    stream: *const u8,
    out: *mut PrebuildInfo,
) {
    if out.is_null() {
        log::log_event(
            PREBUILD,
            log::once_if_switched(&LOGGED_PREBUILD),
            &format!("this={this:p} stream={stream:p} null-out"),
        );
        return;
    }
    // `stream` is the Xbox XDXR descriptor. Public headers do not define it.
    let computed = budget(inputs);
    ptr::write_unaligned(out, computed.info);
    let flags = computed.flags;
    let prims = computed.primitives;
    let verts = computed.vertices;
    let result = computed.info.result;
    let scratch = computed.info.scratch;
    let update = computed.info.update;
    log::log_event(
        PREBUILD,
        log::once_if_switched(&LOGGED_PREBUILD),
        &format!(
            "this={this:p} stream={stream:p} flags={flags:#x} prims={prims} verts={verts} result={result} scratch={scratch} update={update}"
        ),
    );
}

asm_entry!("?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z" => build_entry);

/// # Safety
/// The pointers are not dereferenced. This is a no-op.
#[inline(never)]
unsafe extern "system" fn build_entry(
    this: *mut u8,
    command_list: *mut c_void,
    count: u32,
    descs: *const u8,
) {
    guard(|| {
        log::log_event(
            BUILD,
            Some(&LOGGED_BUILD),
            &format!("this={this:p} list={command_list:p} count={count} descs={descs:p}"),
        );
    });
}

asm_entry!("?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z" => copy_entry);

/// # Safety
/// The pointers are not dereferenced. This is a no-op.
#[inline(never)]
unsafe extern "system" fn copy_entry(
    this: *mut u8,
    command_list: *mut c_void,
    dest: u64,
    source: u64,
    mode: u32,
) {
    guard(|| {
        log::log_event(
            COPY,
            Some(&LOGGED_COPY),
            &format!(
                "this={this:p} list={command_list:p} dest={dest:#x} source={source:#x} mode={mode}"
            ),
        );
    });
}

asm_entry!("?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z" => rehydrate_entry);

/// # Safety
/// The pointers are not dereferenced. Returns `S_OK`.
#[inline(never)]
unsafe extern "system" fn rehydrate_entry(
    this: *mut u8,
    command_list: *mut c_void,
    count: u32,
    descs: *const u8,
    gpu_addresses: *const u64,
) -> i32 {
    guard(|| {
        log::log_event(
            REHYDRATE,
            Some(&LOGGED_REHYDRATE),
            &format!(
                "this={this:p} list={command_list:p} count={count} descs={descs:p} addresses={gpu_addresses:p}"
            ),
        );
    });
    0
}

asm_entry!("?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z" => morton_entry);

/// # Safety
/// `this` is not dereferenced. This is a no-op.
#[inline(never)]
unsafe extern "system" fn morton_entry(this: *mut u8, enabled: u8) {
    guard(|| {
        log::log_event(
            MORTON,
            Some(&LOGGED_MORTON),
            &format!("this={this:p} enabled={enabled}"),
        );
    });
}
