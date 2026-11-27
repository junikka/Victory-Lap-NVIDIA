//! Load the real `XboxBvhBuilder.dll`, check its export table, and call it.

use std::collections::BTreeSet;
use std::env;
use std::ffi::{c_void, CString};
use std::fs;
use std::mem::{self, size_of};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr;
use std::sync::OnceLock;

const EXPORTS: [&str; 7] = [
    "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z",
    "??1XboxBvhBuilder@@QEAA@XZ",
    "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z",
    "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z",
    "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z",
    "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z",
    "?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z",
];

/// Public `d3d12.h` build flags used by the calls below.
const ALLOW_UPDATE: u32 = 0x1;
const PREFER_FAST_TRACE: u32 = 0x4;
const PREFER_FAST_BUILD: u32 = 0x8;
const MINIMIZE_MEMORY: u32 = 0x10;

const TRIANGLES: u32 = 0;
const TOP_LEVEL: u32 = 0;
const BOTTOM_LEVEL: u32 = 1;
const LAYOUT_ARRAY: u32 = 0;
const LAYOUT_POINTERS: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
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

#[repr(C)]
#[derive(Clone, Copy)]
struct GeometryDesc {
    ty: u32,
    flags: u32,
    triangles: TrianglesDesc,
}

#[repr(C)]
struct Inputs {
    ty: u32,
    flags: u32,
    num_descs: u32,
    layout: u32,
    descs: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PrebuildInfo {
    result: u64,
    scratch: u64,
    update: u64,
}

const _: () = assert!(size_of::<TrianglesDesc>() == 48);
const _: () = assert!(size_of::<GeometryDesc>() == 56);
const _: () = assert!(size_of::<Inputs>() == 24);
const _: () = assert!(size_of::<PrebuildInfo>() == 24);

type Ctor = unsafe extern "system" fn(*mut u8, *mut c_void, u8) -> *mut u8;
type Dtor = unsafe extern "system" fn(*mut u8);
type Prebuild = unsafe extern "system" fn(*mut u8, *const Inputs, *const u8, *mut PrebuildInfo);
type Build = unsafe extern "system" fn(*mut u8, *mut c_void, u32, *const u8);
type CopyAs = unsafe extern "system" fn(*mut u8, *mut c_void, u64, u64, u32);
type Rehydrate = unsafe extern "system" fn(*mut u8, *mut c_void, u32, *const u8, *const u64) -> i32;
type Morton = unsafe extern "system" fn(*mut u8, u8);

#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
    fn GetLastError() -> u32;
    fn GetModuleHandleW(name: *const u16) -> *mut c_void;
}

struct Library(*mut c_void);

impl Drop for Library {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { FreeLibrary(self.0) };
        }
    }
}

#[test]
fn dll_exports_prebuild_sizes_and_log() {
    let dll = build_release_dll();
    assert_export_table(&dll);
    assert_no_network_imports(&dll);

    let wide = wide_null(dll.to_str().expect("dll path is utf-8"));
    let module = unsafe { LoadLibraryW(wide.as_ptr()) };
    assert!(
        !module.is_null(),
        "LoadLibraryW {} failed, GetLastError={}",
        dll.display(),
        unsafe { GetLastError() }
    );
    let _library = Library(module);

    let ctor: Ctor = symbol(module, EXPORTS[0]);
    let dtor: Dtor = symbol(module, EXPORTS[1]);
    let build: Build = symbol(module, EXPORTS[2]);
    let copy: CopyAs = symbol(module, EXPORTS[3]);
    let prebuild: Prebuild = symbol(module, EXPORTS[4]);
    let rehydrate: Rehydrate = symbol(module, EXPORTS[5]);
    let morton: Morton = symbol(module, EXPORTS[6]);

    let log_path = log_path();
    let log_at = fs::metadata(&log_path).map(|meta| meta.len()).unwrap_or(0);

    let mut object = vec![0xA5u8; 4096];
    let this = object.as_mut_ptr();
    // Not a real device. A call through it must fault the test.
    let device = 0x8 as *mut c_void;
    let returned = unsafe { ctor(this, device, 1) };
    assert_eq!(returned, this, "constructor returns this");
    assert!(
        object.iter().all(|&byte| byte == 0xA5),
        "constructor wrote into the caller-owned object"
    );

    let mut prebuild_calls = 0u32;
    let null_inputs = unsafe { call_prebuild(prebuild, this, ptr::null(), ptr::null_mut()) };
    prebuild_calls += 1;
    assert_budget("null inputs", &null_inputs);

    let one = tlas(1, 0);
    let thousand = tlas(1000, 0);
    let small_top = unsafe { call_prebuild(prebuild, this, &one, ptr::null()) };
    let top = unsafe { call_prebuild(prebuild, this, &thousand, ptr::null()) };
    prebuild_calls += 2;
    assert_budget("top level 1", &small_top);
    assert_budget("top level 1000", &top);
    assert_grows("top level instances", &small_top, &top);

    // Instance descriptors are a GPU virtual address. Do not read them.
    let thousand_gpu = tlas(1000, 0x1_0000_0000);
    let top_gpu = unsafe { call_prebuild(prebuild, this, &thousand_gpu, ptr::null()) };
    prebuild_calls += 1;
    assert_eq!(
        (top.result, top.scratch, top.update),
        (top_gpu.result, top_gpu.scratch, top_gpu.update),
        "top level must ignore the instance GPU address"
    );

    let stream = [0xFFu8; 64];
    let top_stream = unsafe { call_prebuild(prebuild, this, &thousand, stream.as_ptr()) };
    prebuild_calls += 1;
    assert_eq!(
        (top.result, top.scratch, top.update),
        (top_stream.result, top_stream.scratch, top_stream.update),
        "XDXR stream desc must be ignored"
    );

    // Keep the descriptors alive: `Inputs.descs` is a pointer into them.
    let few_index_geom = triangles(3, 30_000);
    let few_vert_geom = triangles(90_000, 3);
    let large_geom = triangles(90_000, 30_000);
    let few_indices = bottom_array(&few_index_geom, 0);
    let few_verts = bottom_array(&few_vert_geom, 0);
    let large = bottom_array(&large_geom, 0);
    let small_index_info = unsafe { call_prebuild(prebuild, this, &few_indices, ptr::null()) };
    let small_vert_info = unsafe { call_prebuild(prebuild, this, &few_verts, ptr::null()) };
    let large_info = unsafe { call_prebuild(prebuild, this, &large, ptr::null()) };
    prebuild_calls += 3;
    assert_budget("bottom level large", &large_info);
    assert_grows("index count", &small_index_info, &large_info);
    assert_grows("vertex count", &small_vert_info, &large_info);

    let non_indexed_small_geom = triangles(0, 3);
    let non_indexed_large_geom = triangles(0, 300);
    let non_indexed_small = bottom_array(&non_indexed_small_geom, 0);
    let non_indexed_large = bottom_array(&non_indexed_large_geom, 0);
    let non_indexed_small_info =
        unsafe { call_prebuild(prebuild, this, &non_indexed_small, ptr::null()) };
    let non_indexed_large_info =
        unsafe { call_prebuild(prebuild, this, &non_indexed_large, ptr::null()) };
    prebuild_calls += 2;
    assert_grows(
        "non-indexed vertex count",
        &non_indexed_small_info,
        &non_indexed_large_info,
    );

    let geom = triangles(90_000, 30_000);
    let pair = [geom, geom];
    let two = Inputs {
        ty: BOTTOM_LEVEL,
        flags: 0,
        num_descs: 2,
        layout: LAYOUT_ARRAY,
        descs: pair.as_ptr() as u64,
    };
    let ptrs = [
        &pair[0] as *const GeometryDesc as u64,
        &pair[1] as *const GeometryDesc as u64,
    ];
    let two_ptrs = Inputs {
        ty: BOTTOM_LEVEL,
        flags: 0,
        num_descs: 2,
        layout: LAYOUT_POINTERS,
        descs: ptrs.as_ptr() as u64,
    };
    let two_info = unsafe { call_prebuild(prebuild, this, &two, ptr::null()) };
    let two_ptr_info = unsafe { call_prebuild(prebuild, this, &two_ptrs, ptr::null()) };
    prebuild_calls += 2;
    assert_grows("geometry count", &large_info, &two_info);
    assert_eq!(
        (two_info.result, two_info.scratch, two_info.update),
        (
            two_ptr_info.result,
            two_ptr_info.scratch,
            two_ptr_info.update
        ),
        "array and array-of-pointers layouts must agree"
    );

    let fast_trace = bottom_array(&geom, PREFER_FAST_TRACE);
    let fast_build = bottom_array(&geom, PREFER_FAST_BUILD);
    let allow_update = bottom_array(&geom, ALLOW_UPDATE);
    let mut minimize_one = tlas(1, 0);
    minimize_one.flags = MINIMIZE_MEMORY;
    let mut minimize_thousand = tlas(1000, 0);
    minimize_thousand.flags = MINIMIZE_MEMORY;
    let fast_trace_info = unsafe { call_prebuild(prebuild, this, &fast_trace, ptr::null()) };
    let fast_build_info = unsafe { call_prebuild(prebuild, this, &fast_build, ptr::null()) };
    let allow_update_info = unsafe { call_prebuild(prebuild, this, &allow_update, ptr::null()) };
    let minimize_one_info = unsafe { call_prebuild(prebuild, this, &minimize_one, ptr::null()) };
    let minimize_thousand_info =
        unsafe { call_prebuild(prebuild, this, &minimize_thousand, ptr::null()) };
    prebuild_calls += 5;
    assert_budget("fast trace", &fast_trace_info);
    assert_budget("fast build", &fast_build_info);
    assert_budget("allow update", &allow_update_info);
    assert!(
        fast_trace_info.result > large_info.result,
        "PREFER_FAST_TRACE should raise the result size"
    );
    assert!(
        fast_build_info.scratch > large_info.scratch,
        "PREFER_FAST_BUILD should raise the scratch size"
    );
    assert!(
        allow_update_info.update >= large_info.update,
        "ALLOW_UPDATE should not shrink update scratch"
    );
    assert_grows(
        "minimize memory",
        &minimize_one_info,
        &minimize_thousand_info,
    );

    unsafe { call_prebuild_null_out(prebuild, this) };
    prebuild_calls += 1;

    let list = 0x10 as *mut c_void;
    unsafe {
        build(this, list, 1, ptr::null());
        build(this, list, 2, ptr::null());
        copy(this, list, 0x1000, 0x2000, 1);
        copy(this, list, 0x1000, 0x2000, 0);
        assert_eq!(rehydrate(this, list, 1, ptr::null(), ptr::null()), 0);
        assert_eq!(rehydrate(this, list, 4, ptr::null(), ptr::null()), 0);
        morton(this, 1);
        morton(this, 0);
        dtor(this);
        dtor(this);
    }
    assert!(
        object.iter().all(|&byte| byte == 0xA5),
        "a later export wrote into the caller-owned object"
    );

    let suffix = log_suffix(&log_path, log_at);
    assert_eq!(
        count_lines(&suffix, "XboxBvhBuilder::ctor"),
        1,
        "ctor log\n{suffix}"
    );
    assert_eq!(
        count_lines(&suffix, "XboxBvhBuilder::dtor"),
        2,
        "dtor log\n{suffix}"
    );
    assert_eq!(
        count_lines(
            &suffix,
            "XboxBvhBuilder::GetRaytracingAccelerationStructurePrebuildInfo"
        ),
        prebuild_calls,
        "prebuild log\n{suffix}"
    );
    assert_eq!(
        count_lines(
            &suffix,
            "XboxBvhBuilder::BuildRaytracingAccelerationStructureBatched2X"
        ),
        1,
        "build should log once\n{suffix}"
    );
    assert_eq!(
        count_lines(
            &suffix,
            "XboxBvhBuilder::CopyRaytracingAccelerationStructure"
        ),
        1,
        "copy should log once\n{suffix}"
    );
    assert_eq!(
        count_lines(
            &suffix,
            "XboxBvhBuilder::RehydrateRaytracingAccelerationStructureBatched2X"
        ),
        1,
        "rehydrate should log once\n{suffix}"
    );
    assert_eq!(
        count_lines(&suffix, "XboxBvhBuilder::Set64BitMortonCodes"),
        1,
        "morton should log once\n{suffix}"
    );

    let ctor_line = suffix
        .lines()
        .map(|line| line.trim_end_matches('\r'))
        .find(|line| line.contains("XboxBvhBuilder::ctor"))
        .expect("ctor line");
    let ra = parse_ra(ctor_line);
    assert_ne!(ra, 0, "missing caller in {ctor_line}");
    let test_exe = unsafe { GetModuleHandleW(ptr::null()) };
    let test_range = image_range(test_exe);
    let dll_range = image_range(module);
    assert!(
        ra >= test_range.0 && ra < test_range.1,
        "caller return address {ra:#x} is outside the test image {test_range:#x?}\n\
         dll range {dll_range:#x?}\n{ctor_line}"
    );
    assert!(
        ra < dll_range.0 || ra >= dll_range.1,
        "caller return address {ra:#x} points inside XboxBvhBuilder.dll {dll_range:#x?}"
    );

    println!("tested {}", dll.display());
}

fn build_release_dll() -> PathBuf {
    static DLL: OnceLock<PathBuf> = OnceLock::new();
    DLL.get_or_init(build_release_dll_once).clone()
}

fn build_release_dll_once() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf();
    // A nested `cargo build` into the default target directory deadlocks: the
    // outer `cargo test` holds that directory's lock until this test returns.
    let target = workspace.join("target").join("xbox-bvh-builder-dll");
    let output = Command::new(env!("CARGO"))
        .args([
            "build",
            "--release",
            "--offline",
            "-p",
            "xbox-bvh-builder",
            "--manifest-path",
        ])
        .arg(workspace.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(&target)
        // The stand-in may import only KERNEL32, ntdll and api-ms-win-core-*: link the C runtime
        // statically for this crate alone (the workspace's own build keeps the dynamic one).
        .env("CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS", "-C target-feature=+crt-static")
        .output()
        .expect("spawn cargo build");
    assert!(
        output.status.success(),
        "cargo build failed\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let dll = target.join("release").join("XboxBvhBuilder.dll");
    assert!(dll.is_file(), "missing {}", dll.display());
    dll
}

fn dumpbin() -> PathBuf {
    let vswhere =
        PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    let output = Command::new(&vswhere)
        .args([
            "-latest",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-find",
            r"VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe",
        ])
        .output()
        .expect("spawn vswhere");
    let text = String::from_utf8_lossy(&output.stdout);
    let path = text
        .lines()
        .map(str::trim)
        .find(|line| line.ends_with("dumpbin.exe"))
        .expect("dumpbin.exe from vswhere");
    PathBuf::from(path)
}

fn assert_export_table(dll: &Path) {
    let output = Command::new(dumpbin())
        .args(["/exports"])
        .arg(dll)
        .output()
        .expect("spawn dumpbin");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "dumpbin /exports failed\n{stdout}\n{stderr}"
    );
    let mut names = export_names(&stdout);
    names.sort();
    let mut expected: Vec<&str> = EXPORTS.to_vec();
    expected.sort();
    assert_eq!(names.len(), expected.len(), "export count\n{stdout}");
    assert_x64(dll);
    assert_eq!(
        names.iter().map(String::as_str).collect::<BTreeSet<_>>(),
        expected.into_iter().collect::<BTreeSet<_>>(),
        "dumpbin /exports names\n{stdout}"
    );
}

fn export_names(dump: &str) -> Vec<String> {
    dump.lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let ordinal = parts.next()?;
            if ordinal.parse::<u32>().is_err() {
                return None;
            }
            let _hint = parts.next()?;
            let _rva = parts.next()?;
            // dumpbin prints `mangled = mangled (demangled...)`, so the name is
            // the fourth token and more tokens follow on the same line.
            let name = parts.next()?;
            if name.starts_with('?') {
                Some(name.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn assert_x64(dll: &Path) {
    let output = Command::new(dumpbin())
        .args(["/headers"])
        .arg(dll)
        .output()
        .expect("spawn dumpbin /headers");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "dumpbin /headers failed\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("8664 machine (x64)"),
        "DLL is not x64\n{stdout}"
    );
}

fn assert_no_network_imports(dll: &Path) {
    let output = Command::new(dumpbin())
        .args(["/imports"])
        .arg(dll)
        .output()
        .expect("spawn dumpbin /imports");
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "dumpbin /imports failed\n{text}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lower = text.to_ascii_lowercase();
    for banned in ["ws2_32.dll", "winhttp.dll", "wininet.dll", "urlmon.dll"] {
        assert!(
            !lower.contains(banned),
            "XboxBvhBuilder.dll imports {banned}\n{text}"
        );
    }
}

fn symbol<T>(module: *mut c_void, name: &str) -> T {
    assert_eq!(size_of::<T>(), size_of::<*mut c_void>());
    let c_name = CString::new(name).expect("export name");
    let addr = unsafe { GetProcAddress(module, c_name.as_ptr()) };
    assert!(!addr.is_null(), "GetProcAddress missed {name}");
    unsafe { mem::transmute_copy(&addr) }
}

fn triangles(index_count: u32, vertex_count: u32) -> GeometryDesc {
    GeometryDesc {
        ty: TRIANGLES,
        flags: 0,
        triangles: TrianglesDesc {
            transform: 0,
            // DXGI_FORMAT_R32_UINT when indexed, DXGI_FORMAT_UNKNOWN otherwise.
            index_format: if index_count == 0 { 0 } else { 42 },
            // DXGI_FORMAT_R32G32B32_FLOAT
            vertex_format: 6,
            index_count,
            vertex_count,
            index_buffer: 0,
            vertex_buffer: 0,
            vertex_stride: 12,
        },
    }
}

fn tlas(instances: u32, instance_descs: u64) -> Inputs {
    Inputs {
        ty: TOP_LEVEL,
        flags: 0,
        num_descs: instances,
        layout: LAYOUT_ARRAY,
        descs: instance_descs,
    }
}

fn bottom_array(geom: &GeometryDesc, flags: u32) -> Inputs {
    Inputs {
        ty: BOTTOM_LEVEL,
        flags,
        num_descs: 1,
        layout: LAYOUT_ARRAY,
        descs: geom as *const GeometryDesc as u64,
    }
}

unsafe fn call_prebuild(
    prebuild: Prebuild,
    this: *mut u8,
    inputs: *const Inputs,
    stream: *const u8,
) -> PrebuildInfo {
    let mut info = PrebuildInfo {
        result: u64::MAX,
        scratch: u64::MAX,
        update: u64::MAX,
    };
    prebuild(this, inputs, stream, &mut info);
    info
}

unsafe fn call_prebuild_null_out(prebuild: Prebuild, this: *mut u8) {
    let inputs = tlas(1, 0);
    prebuild(this, &inputs, ptr::null(), ptr::null_mut());
}

fn assert_budget(label: &str, info: &PrebuildInfo) {
    for (name, value) in [
        ("result", info.result),
        ("scratch", info.scratch),
        ("update", info.update),
    ] {
        assert_ne!(value, 0, "{label} {name} is zero");
        assert_eq!(
            value % 256,
            0,
            "{label} {name} = {value} is not 256-byte aligned"
        );
    }
    assert!(
        info.scratch >= info.result,
        "{label} scratch {} is below result {}",
        info.scratch,
        info.result
    );
    assert!(
        info.update <= info.scratch,
        "{label} update {} is above scratch {}",
        info.update,
        info.scratch
    );
}

fn assert_grows(label: &str, small: &PrebuildInfo, large: &PrebuildInfo) {
    assert_budget(label, small);
    assert_budget(label, large);
    assert!(
        large.result > small.result,
        "{label} result did not grow: {} -> {}",
        small.result,
        large.result
    );
    assert!(
        large.scratch > small.scratch,
        "{label} scratch did not grow: {} -> {}",
        small.scratch,
        large.scratch
    );
    assert!(
        large.update > small.update,
        "{label} update did not grow: {} -> {}",
        small.update,
        large.update
    );
}

fn log_path() -> PathBuf {
    let base = env::var_os("LOCALAPPDATA").expect("LOCALAPPDATA");
    PathBuf::from(base)
        .join("victory-lap")
        .join("xbox-bvh-builder.log")
}

fn log_suffix(path: &Path, start: u64) -> String {
    let bytes = fs::read(path).expect("read log");
    let start = usize::try_from(start).expect("log offset");
    assert!(bytes.len() >= start, "log shrank");
    String::from_utf8_lossy(&bytes[start..]).into_owned()
}

fn count_lines(text: &str, needle: &str) -> u32 {
    text.lines()
        .filter(|line| line.contains(needle))
        .count()
        .try_into()
        .expect("line count")
}

fn parse_ra(line: &str) -> usize {
    let rest = line
        .split("ra=0x")
        .nth(1)
        .unwrap_or_else(|| panic!("no return address in {line}"));
    let hex = rest.split_whitespace().next().expect("ra hex");
    usize::from_str_radix(hex, 16).unwrap_or_else(|_| panic!("bad ra in {line}"))
}

fn image_range(module: *mut c_void) -> (usize, usize) {
    assert!(!module.is_null(), "null module");
    let base = module as usize;
    let lfanew = unsafe { ptr::read_unaligned((base + 0x3C) as *const i32) };
    assert!(lfanew > 0, "bad e_lfanew");
    let optional = base + lfanew as usize + 24;
    let magic = unsafe { ptr::read_unaligned(optional as *const u16) };
    assert_eq!(magic, 0x20B, "expected a PE32+ image");
    let size = unsafe { ptr::read_unaligned((optional + 56) as *const u32) } as usize;
    (base, base + size)
}

fn wide_null(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

include!("../pe_import_table.rs");

#[test]
fn release_dll_imports_only_kernel32_ntdll_or_api_sets() {
    let dll = build_release_dll();
    assert_console_imports(&dll);
}
