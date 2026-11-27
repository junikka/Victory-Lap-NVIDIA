//! Diagnostics: where one of the title's threads is, looked at about every
//! two milliseconds. `VICTORY_THREAD_SAMPLES=busiest` follows the thread of
//! the title's that used most processor time in the last five seconds;
//! `VICTORY_THREAD_SAMPLES=<n>` the one bound to logical processor `n`
//! alone (the busiest, if there is none).
//!
//! A look: the thread is suspended, its registers and the top of its stack
//! are copied, and it runs on; nothing is asked of the system while it
//! stands (it may hold any lock). The frames are then walked on the copy
//! with the unwind data of our own code and the system's. The title's code
//! is never read: a frame in it ends the walk, and all that is kept of it is
//! a count for the 4 KiB of the module it lies in.
//!
//! Every five seconds a line says how the samples divide: the title's code,
//! each of our functions (by name if our symbols lie beside the program,
//! else by place), the driver, the system, and how many were waits.

// The arithmetic and its tests are built with the tests too; what records
// and reports is used by a diagnostics build alone.
#![cfg_attr(not(feature = "diagnostics"), allow(dead_code, unused_imports))]

use std::collections::HashMap;
use std::fmt::Write as _;

use windows_sys::Win32::System::Diagnostics::Debug::CONTEXT;

#[cfg(feature = "diagnostics")]
use super::cycle::{frequency, ticks};

/// Whose thread to follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Target {
    Busiest,
    Processor(u8),
}

/// What `VICTORY_THREAD_SAMPLES` asks for.
pub(super) fn parse(value: &str) -> Option<Target> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("busiest") {
        return Some(Target::Busiest);
    }
    value.parse::<u8>().ok().filter(|n| *n < 64).map(Target::Processor)
}

#[cfg(feature = "diagnostics")]
fn asked() -> Option<Target> {
    static ASKED: std::sync::OnceLock<Option<Target>> = std::sync::OnceLock::new();
    *ASKED.get_or_init(|| parse(&std::env::var("VICTORY_THREAD_SAMPLES").ok()?))
}

/// Starts the sampling thread: once, if asked for.
#[cfg(feature = "diagnostics")]
pub(super) fn start() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let Some(target) = asked() else {
        return;
    };
    ONCE.call_once(|| {
        if std::thread::Builder::new().name("victory-samples".into()).spawn(move || work(target)).is_err() {
            tracing::warn!("no thread for the samples");
        } else {
            tracing::info!(?target, "samples of one of the title's threads are on (VICTORY_THREAD_SAMPLES)");
        }
    });
}

/// Whose code an address lies in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Class {
    /// A module that is neither ours nor the system's: the title's.
    Title,
    Ours,
    /// The card's driver.
    Driver,
    /// Direct3D and DXGI.
    Graphics,
    System,
    /// A library of ours beside the program.
    Beside,
    /// No module.
    Unknown,
}

impl Class {
    const ALL: [Class; 7] = [Class::Title, Class::Ours, Class::Driver, Class::Graphics, Class::System, Class::Beside, Class::Unknown];

    /// Whether frames in it may be walked with the system's unwind data.
    fn walked(self) -> bool {
        !matches!(self, Class::Title | Class::Unknown)
    }

    fn name(self) -> &'static str {
        match self {
            Class::Title => "title",
            Class::Ours => "ours",
            Class::Driver => "driver",
            Class::Graphics => "d3d",
            Class::System => "system",
            Class::Beside => "beside",
            Class::Unknown => "no module",
        }
    }
}

/// The class of the module at `path`; `ours`: it is the program, which lies
/// in `home`.
pub(super) fn class_of(path: &str, ours: bool, home: &str) -> Class {
    if ours {
        return Class::Ours;
    }
    let path = path.to_ascii_lowercase().replace('/', "\\");
    let file = path.rsplit('\\').next().unwrap_or(&path);
    if ["d3d12", "d3d11", "dxgi", "dxcore", "d3dscache"].iter().any(|p| file.starts_with(p)) {
        return Class::Graphics;
    }
    let system = path.contains("\\windows\\");
    if system && ["nv", "amd", "ati", "igd", "igc"].iter().any(|p| file.starts_with(p)) {
        return Class::Driver;
    }
    if system {
        return Class::System;
    }
    let home = home.to_ascii_lowercase().replace('/', "\\");
    if (!home.is_empty() && path.starts_with(&home)) || file.contains("ffx") || file.contains("fidelityfx") {
        return Class::Beside;
    }
    Class::Title
}

/// A frame of a sample: the instruction, the function it lies in (the
/// instruction itself where that is not known), and whose it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Frame {
    at: usize,
    function: usize,
    class: Class,
}

/// What a sample counts as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Bucket {
    /// In the title's code: the 4 KiB it lies in.
    Title(usize),
    /// The title called the system, the driver or Direct3D itself: the
    /// innermost function.
    TitleCalls(usize),
    /// In a function of ours (`.0`), or in what it called: the innermost
    /// function not ours (0: none, the sample is in ours).
    Ours(usize, usize),
    /// No frame of the title's or ours: the innermost function.
    Other(usize),
}

/// The bucket of a sample, innermost frame first.
pub(super) fn bucket(frames: &[Frame]) -> Option<Bucket> {
    let top = frames.first()?;
    let owner = frames.iter().position(|f| matches!(f.class, Class::Title | Class::Ours));
    Some(match owner.map(|at| (at, frames[at])) {
        Some((0, f)) if f.class == Class::Title => Bucket::Title(f.at >> 12),
        Some((_, f)) if f.class == Class::Title => Bucket::TitleCalls(top.function),
        Some((0, f)) => Bucket::Ours(f.function, 0),
        Some((_, f)) => Bucket::Ours(f.function, top.function),
        None => Bucket::Other(top.function),
    })
}

/// Whether the system function `name` is a wait, and whether it gives the
/// processor up without one.
pub(super) fn wait_or_yield(name: &str) -> (bool, bool) {
    let name = name.rsplit('!').next().unwrap_or(name);
    let waits = ["WaitFor", "DelayExecution", "RemoveIoCompletion", "SignalAndWait", "WaitOnAddress", "SleepEx", "WaitHighEventPair"];
    let yields = ["YieldExecution", "SwitchToThread", "YieldProcessor"];
    (waits.iter().any(|w| name.contains(w)), yields.iter().any(|y| name.contains(y)))
}

/// Five seconds of samples.
#[derive(Default)]
pub(super) struct Tally {
    samples: u32,
    missed: u32,
    top: HashMap<Class, u32>,
    buckets: HashMap<Bucket, u32>,
}

impl Tally {
    pub(super) fn add(&mut self, frames: &[Frame]) {
        let Some(bucket) = bucket(frames) else {
            self.missed += 1;
            return;
        };
        self.samples += 1;
        *self.top.entry(frames[0].class).or_default() += 1;
        *self.buckets.entry(bucket).or_default() += 1;
    }

    fn share(&self, count: u32) -> f64 {
        f64::from(count) * 100.0 / f64::from(self.samples.max(1))
    }

    /// The shares by the class of the innermost frame.
    pub(super) fn classes(&self) -> String {
        let mut text = String::new();
        for class in Class::ALL {
            let count = self.top.get(&class).copied().unwrap_or(0);
            if count > 0 {
                let _ = write!(text, "{}{} {:.1}%", if text.is_empty() { "" } else { ", " }, class.name(), self.share(count));
            }
        }
        text
    }
}

/// A module of the process.
struct Module {
    base: usize,
    end: usize,
    class: Class,
    file: String,
    /// Its named exports by place, if they were asked for.
    exports: Option<Vec<(u32, String)>>,
}

/// The modules met so far.
#[derive(Default)]
struct Modules {
    all: Vec<Module>,
    home: String,
}

impl Modules {
    fn at(&mut self, address: usize) -> Option<usize> {
        use windows_sys::Win32::System::LibraryLoader::{
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW,
            GetModuleHandleExW, GetModuleHandleW,
        };
        use windows_sys::Win32::System::ProcessStatus::{K32GetModuleInformation, MODULEINFO};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        if let Some(at) = self.all.iter().position(|m| m.base <= address && address < m.end) {
            return Some(at);
        }
        let mut module = std::ptr::null_mut();
        let mut path = [0u16; 520];
        // SAFETY: an address, no name; plain queries.
        unsafe {
            let flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
            if GetModuleHandleExW(flags, address as *const u16, &mut module) == 0 || module.is_null() {
                return None;
            }
            let base = module as usize;
            let len = GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) as usize;
            let path = String::from_utf16_lossy(&path[..len.min(path.len())]);
            let ours = module == GetModuleHandleW(std::ptr::null());
            if self.home.is_empty() {
                self.home = std::env::current_exe()
                    .ok()
                    .and_then(|p| Some(p.parent()?.to_string_lossy().into_owned()))
                    .unwrap_or_default();
            }
            let class = class_of(&path, ours, &self.home);
            // Its size, as the system has it (no module's memory is read).
            let mut info = MODULEINFO { lpBaseOfDll: std::ptr::null_mut(), SizeOfImage: 0, EntryPoint: std::ptr::null_mut() };
            K32GetModuleInformation(GetCurrentProcess(), module, &mut info, size_of::<MODULEINFO>() as u32);
            let size = info.SizeOfImage as usize;
            let file = path.rsplit('\\').next().unwrap_or("?").to_string();
            self.all.push(Module { base, end: base + size.max(0x1000), class, file, exports: None });
        }
        Some(self.all.len() - 1)
    }

    /// A name for the function at `function` in a module that is not the
    /// title's: the nearest named export before it, by the module's own
    /// table.
    fn export(&mut self, function: usize) -> String {
        let Some(at) = self.at(function) else {
            return format!("{function:#x}");
        };
        let module = &mut self.all[at];
        if !module.class.walked() {
            return module.class.name().to_string();
        }
        let base = module.base;
        // SAFETY: a module of the system's or ours, mapped.
        let exports = module.exports.get_or_insert_with(|| unsafe { exports(base) });
        let rva = (function - base) as u32;
        let stem = module.file.strip_suffix(".dll").unwrap_or(&module.file);
        match exports.partition_point(|e| e.0 <= rva).checked_sub(1).map(|i| &exports[i]) {
            Some((start, name)) if rva - start < 0x4000 => format!("{stem}!{name}"),
            _ => format!("{stem}+{rva:#x}"),
        }
    }
}

/// The named exports of the module at `base`, by their place.
///
/// # Safety
/// `base` is a module the system mapped that is not the title's.
unsafe fn exports(base: usize) -> Vec<(u32, String)> {
    let mut out = Vec::new();
    // SAFETY: the headers and the export directory of a mapped image.
    unsafe {
        let read = |at: usize| (at as *const u32).read_unaligned();
        let headers = base + read(base + 0x3c) as usize;
        let size = read(headers + 0x18 + 0x38) as usize;
        // The first data directory of the 64-bit optional header.
        let (directory, length) = (read(headers + 0x18 + 0x70) as usize, read(headers + 0x18 + 0x74) as usize);
        if directory == 0 || directory + 40 > size {
            return out;
        }
        let at = base + directory;
        let (count, functions, names, ordinals) = (read(at + 24) as usize, read(at + 28) as usize, read(at + 32) as usize, read(at + 36) as usize);
        for i in 0..count.min(20_000) {
            let name = read(base + names + i * 4) as usize;
            let ordinal = ((base + ordinals + i * 2) as *const u16).read_unaligned() as usize;
            let rva = read(base + functions + ordinal * 4);
            // A forwarder lies in the directory itself.
            if name == 0 || name >= size || (rva as usize >= directory && (rva as usize) < directory + length) {
                continue;
            }
            let text = std::ffi::CStr::from_ptr((base + name) as *const _).to_string_lossy();
            if text.len() <= 96 {
                out.push((rva, text.into_owned()));
            }
        }
    }
    out.sort();
    out
}

/// How much of the stack is copied, and the room behind it that a frame's
/// own reads may reach into.
const COPY: usize = 64 << 10;
const ROOM: usize = 2 << 20;
/// Frames of a sample.
const DEEP: usize = 32;

/// `GetThreadContext` wants a 16-byte aligned context.
#[repr(C, align(16))]
struct Aligned(CONTEXT);

/// The control and integer registers (`CONTEXT_AMD64 | CONTROL | INTEGER`).
const REGISTERS: u32 = 0x0010_0003;

/// Stops `thread`, copies its registers and the top of its stack to `copy`
/// and lets it run: its stack pointer and the bytes copied.
///
/// # Safety
/// `thread` is another thread of this process, open to be suspended and
/// read; `copy` has room for [`COPY`] bytes. Nothing in here takes a lock or
/// allocates while the thread stands.
unsafe fn take(thread: usize, context: &mut Aligned, copy: *mut u8) -> Option<(usize, usize)> {
    use windows_sys::Win32::System::Diagnostics::Debug::GetThreadContext;
    use windows_sys::Win32::System::Threading::{ResumeThread, SuspendThread};
    // SAFETY: see above; the stack between its pointer and its base is
    // committed, and the thread cannot leave while it is suspended.
    unsafe {
        context.0.ContextFlags = REGISTERS;
        if SuspendThread(thread as _) == u32::MAX {
            return None;
        }
        let mut taken = None;
        if GetThreadContext(thread as _, &mut context.0) != 0 {
            let sp = context.0.Rsp as usize & !7;
            // The thread's block: its stack's base and limit at 8 and 16.
            let mut info = [0u64; 6];
            let status = super::cycle::NtQueryInformationThread(thread as _, 0, info.as_mut_ptr().cast(), 48, std::ptr::null_mut());
            let (mut base, mut limit) = (0usize, 0usize);
            if status == 0 && info[1] != 0 {
                let block = info[1] as usize as *const usize;
                (base, limit) = (block.add(1).read_volatile(), block.add(2).read_volatile());
            }
            let len = if sp != 0 && limit <= sp && sp < base {
                (base - sp).min(COPY)
            } else if sp != 0 {
                // Not the stack the block names: to the end of its page.
                (sp | 0xfff) + 1 - sp
            } else {
                0
            };
            let len = len & !7;
            if len >= 8 {
                std::ptr::copy_nonoverlapping(sp as *const u8, copy, len);
                taken = Some((sp, len));
            }
        }
        ResumeThread(thread as _);
        taken
    }
}

/// The sixteen integer registers of `context`, `rax` first.
fn registers(context: &mut CONTEXT) -> &mut [u64; 16] {
    // SAFETY: `Rax` through `R15` are sixteen `u64` fields in a row.
    unsafe { &mut *(std::ptr::from_mut(context).cast::<u8>().add(std::mem::offset_of!(CONTEXT, Rax)).cast::<[u64; 16]>()) }
}

/// Registers that point into the stack that was copied point into the copy.
fn moved(context: &mut CONTEXT, sp: usize, len: usize, copy: usize) {
    for value in registers(context).iter_mut() {
        let v = *value as usize;
        if v >= sp && v < sp + len {
            *value = (v - sp + copy) as u64;
        }
    }
}

/// Walks the frames of a sample on the copy of its stack (`len` bytes at
/// `copy`, taken from `sp`), innermost first. Frames of the title's and
/// those outside any module end the walk.
///
/// # Safety
/// `context` and the copy are what [`take`] left; behind the copy lie
/// [`ROOM`] readable bytes.
unsafe fn walk(context: &mut Aligned, sp: usize, len: usize, copy: usize, modules: &mut Modules, out: &mut Vec<Frame>) {
    use windows_sys::Win32::System::Diagnostics::Debug::{RtlLookupFunctionEntry, RtlVirtualUnwind, UNW_FLAG_NHANDLER};
    out.clear();
    let context = &mut context.0;
    moved(context, sp, len, copy);
    let inside = |at: usize| at >= copy && at + 8 <= copy + len;
    while out.len() < DEEP {
        let at = context.Rip as usize;
        if at < 0x10000 {
            break;
        }
        let class = modules.at(at).map_or(Class::Unknown, |m| modules.all[m].class);
        if !class.walked() {
            out.push(Frame { at, function: at, class });
            break;
        }
        let mut image = 0;
        // SAFETY: the function table entry covering an instruction of a
        // module that is not the title's.
        let entry = unsafe { RtlLookupFunctionEntry(context.Rip, &mut image, std::ptr::null_mut()) };
        let rsp = context.Rsp as usize;
        if entry.is_null() {
            out.push(Frame { at, function: at, class });
            if !inside(rsp) {
                break;
            }
            // SAFETY: inside the copy. A function without unwind data has
            // its return address at the stack pointer.
            context.Rip = unsafe { (rsp as *const u64).read() };
            context.Rsp += 8;
        } else {
            // SAFETY: the entry (begin, end, unwind data: three `u32`) and
            // the unwind data's header, in the mapped image.
            let (begin, data) = unsafe { ((entry as *const u32).read_unaligned(), (entry as *const u32).add(2).read_unaligned()) };
            out.push(Frame { at, function: image as usize + begin as usize, class });
            if !inside(rsp) || data & 1 != 0 {
                break;
            }
            // A function with a frame register is unwound from it: it must
            // point into the copy.
            // SAFETY: as above; the fourth byte holds the register.
            let register = unsafe { ((image as usize + data as usize) as *const u8).add(3).read() } & 0xf;
            if register != 0 && !inside(registers(context)[register as usize] as usize) {
                break;
            }
            let (mut handler, mut frame) = (std::ptr::null_mut(), 0);
            // SAFETY: the entry belongs to the image; the context is ours,
            // its stack pointer and frame register lie in the copy, and what
            // the function saved lies above them, in the copy or the room
            // behind it.
            unsafe {
                RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, context.Rip, entry, context, &mut handler, &mut frame, std::ptr::null_mut());
            }
        }
        moved(context, sp, len, copy);
        if !inside(context.Rsp as usize) {
            break;
        }
    }
}

/// A thread of the title's as the sampler knows it.
#[cfg(feature = "diagnostics")]
struct Known {
    id: u32,
    handle: usize,
    cycles: u64,
    /// The cycles it used in the last window.
    used: u64,
    mask: u64,
    priority: i32,
}

#[cfg(feature = "diagnostics")]
fn cycles_of(handle: usize) -> u64 {
    let mut cycles = 0u64;
    // SAFETY: a handle we opened; a valid out pointer.
    unsafe { super::cycle::QueryThreadCycleTime(handle as _, &mut cycles) };
    cycles
}

/// Which of `threads` to follow: the one bound to the processor asked for
/// alone, else the one that used most cycles.
pub(super) fn pick(target: Target, threads: &[(u64, u64)]) -> Option<usize> {
    let busiest = |among: &mut dyn Iterator<Item = usize>| among.max_by_key(|&i| threads[i].1);
    if let Target::Processor(n) = target
        && let Some(at) = busiest(&mut (0..threads.len()).filter(|&i| threads[i].0 == 1u64 << n))
    {
        return Some(at);
    }
    busiest(&mut (0..threads.len()))
}

/// Opens the title's threads not known yet and reads what all of them used.
#[cfg(feature = "diagnostics")]
fn look(threads: &mut Vec<Known>) {
    use windows_sys::Win32::System::Threading::{
        OpenThread, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_QUERY_LIMITED_INFORMATION, THREAD_SUSPEND_RESUME,
    };
    // SAFETY: a plain query.
    let me = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
    for id in super::cycle::title_ids() {
        if id == me || threads.iter().any(|t| t.id == id) {
            continue;
        }
        let access = THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION;
        // SAFETY: a thread of this process, by its number; kept open.
        let handle = unsafe { OpenThread(access, 0, id) } as usize;
        if handle != 0 {
            threads.push(Known { id, handle, cycles: cycles_of(handle), used: 0, mask: 0, priority: 0 });
        }
    }
    for thread in threads.iter_mut() {
        let now = cycles_of(thread.handle);
        (thread.used, thread.cycles) = (now.saturating_sub(thread.cycles), now);
        (thread.mask, thread.priority) = super::cycle::mask_and_priority(thread.handle);
    }
}

/// The line of a window.
#[cfg(feature = "diagnostics")]
fn report(tally: &Tally, modules: &mut Modules, names: &mut Option<super::report::Names>, of: &Known, core: f64) {
    let names = names.get_or_insert_with(super::report::Names::new);
    let (mut waiting, mut yielding) = (0, 0);
    let (mut ours, mut calls, mut pages, mut other): (Vec<(String, u32)>, Vec<(String, u32)>, Vec<(String, u32)>, Vec<(String, u32)>) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut title = 0;
    /// The name of the innermost function of `count` samples, which are
    /// added to the waits and the yields if it is one.
    fn inner(modules: &mut Modules, function: usize, count: u32, waiting: &mut u32, yielding: &mut u32) -> String {
        let name = modules.export(function);
        let (waits, yields) = wait_or_yield(&name);
        *waiting += if waits { count } else { 0 };
        *yielding += if yields { count } else { 0 };
        name
    }
    for (bucket, &count) in &tally.buckets {
        match *bucket {
            Bucket::Title(page) => {
                title += count;
                let base = modules.at(page << 12).map_or(0, |m| modules.all[m].base);
                pages.push((format!("title+{:#x}", (page << 12) - base), count));
            }
            Bucket::TitleCalls(function) => {
                let called = inner(modules, function, count, &mut waiting, &mut yielding);
                calls.push((format!("title > {called}"), count));
            }
            Bucket::Ours(function, 0) => ours.push((names.of(function + 1), count)),
            Bucket::Ours(function, called) => {
                let called = inner(modules, called, count, &mut waiting, &mut yielding);
                ours.push((format!("{} > {called}", names.of(function + 1)), count));
            }
            Bucket::Other(function) => {
                let name = inner(modules, function, count, &mut waiting, &mut yielding);
                other.push((name, count));
            }
        }
    }
    let listed = |all: &mut Vec<(String, u32)>, n: usize| {
        all.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut text = String::new();
        for (name, count) in all.iter().take(n) {
            let _ = write!(text, "{}{} {:.1}%", if text.is_empty() { "" } else { "; " }, name.replace("victory_gpu::", "").replace("victory_hle::", "hle::"), tally.share(*count));
        }
        text
    };
    let in_ours: u32 = ours.iter().map(|o| o.1).sum();
    tracing::info!(
        thread = of.id,
        mask = %format!("{:#x}", of.mask),
        priority = of.priority,
        of_a_core = %format!("{:.0}%", core * 100.0),
        samples = tally.samples,
        missed = tally.missed,
        innermost = %tally.classes(),
        in_title_code = %format!("{:.1}%", tally.share(title)),
        in_or_under_ours = %format!("{:.1}%", tally.share(in_ours)),
        waiting = %format!("{:.1}%", tally.share(waiting)),
        yielding = %format!("{:.1}%", tally.share(yielding)),
        ours = %listed(&mut ours, 10),
        title_calls = %listed(&mut calls, 5),
        title_pages = %format!("{} of 4 KiB; {}", pages.len(), listed(&mut pages, 3)),
        other = %listed(&mut other, 3),
        "samples of the title's thread (VICTORY_THREAD_SAMPLES)"
    );
}

#[cfg(feature = "diagnostics")]
fn work(target: Target) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    // SAFETY: fresh memory of ours, kept for good; zero until touched.
    let copy = unsafe { VirtualAlloc(std::ptr::null(), COPY + ROOM, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) }.cast::<u8>();
    if copy.is_null() {
        tracing::warn!("no memory for the samples");
        return;
    }
    let mut modules = Modules::default();
    let mut names = None;
    let mut threads: Vec<Known> = Vec::new();
    let mut tally = Tally::default();
    let mut followed: Option<usize> = None;
    // SAFETY: zeroes are a valid context.
    let mut context = Aligned(unsafe { std::mem::zeroed() });
    let mut frames = Vec::with_capacity(DEEP);
    let mut began = ticks();
    // SAFETY: reads the time stamp counter.
    let mut clock = unsafe { std::arch::x86_64::_rdtsc() };
    let mut step = 0x9e37_79b9u32;
    loop {
        if followed.is_none() || ticks() - began >= 5 * frequency() {
            // SAFETY: as above.
            let now = unsafe { std::arch::x86_64::_rdtsc() };
            look(&mut threads);
            if let Some(at) = followed
                && tally.samples + tally.missed > 0
            {
                let core = threads[at].used as f64 / now.saturating_sub(clock).max(1) as f64;
                report(&tally, &mut modules, &mut names, &threads[at], core);
            }
            let known: Vec<(u64, u64)> = threads.iter().map(|t| (t.mask, t.used)).collect();
            followed = pick(target, &known);
            (tally, began, clock) = (Tally::default(), ticks(), now);
            if followed.is_none() {
                std::thread::sleep(std::time::Duration::from_millis(500));
                continue;
            }
        }
        if let Some(at) = followed {
            // SAFETY: a thread of the title's, not this one, opened for it;
            // the copy has `COPY` bytes and `ROOM` behind them.
            unsafe {
                match take(threads[at].handle, &mut context, copy) {
                    Some((sp, len)) => {
                        std::ptr::write_bytes(copy.add(len), 0, COPY - len);
                        walk(&mut context, sp, len, copy as usize, &mut modules, &mut frames);
                        tally.add(&frames);
                    }
                    None => tally.missed += 1,
                }
            }
        }
        // About two milliseconds, never in step with the frame.
        step = step.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        std::thread::sleep(std::time::Duration::from_micros(1500 + u64::from(step >> 22)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(class: Class, function: usize) -> Frame {
        Frame { at: function + 5, function, class }
    }

    #[test]
    fn the_variable_names_a_processor_or_the_busiest() {
        assert_eq!(parse("busiest"), Some(Target::Busiest));
        assert_eq!(parse(" 13 "), Some(Target::Processor(13)));
        assert_eq!(parse("64"), None);
        assert_eq!(parse(""), None);
        assert_eq!(parse("on"), None);
    }

    #[test]
    fn a_module_is_classed_by_where_it_lies() {
        let home = r"C:\play\release";
        assert_eq!(class_of(r"C:\play\release\victory.exe", true, home), Class::Ours);
        assert_eq!(class_of(r"C:\Windows\System32\ntdll.dll", false, home), Class::System);
        assert_eq!(class_of(r"C:\WINDOWS\System32\DriverStore\FileRepository\nv_dispi.inf_amd64_1\nvwgf2umx.dll", false, home), Class::Driver);
        assert_eq!(class_of(r"C:\Windows\System32\D3D12Core.dll", false, home), Class::Graphics);
        assert_eq!(class_of(r"C:\play\release\D3D12\D3D12Core.dll", false, home), Class::Graphics);
        assert_eq!(class_of(r"C:\play\release\dxcompiler.dll", false, home), Class::Beside);
        assert_eq!(class_of(r"C:\data\fidelityfx\amd_fidelityfx_upscaler_dx12.dll", false, home), Class::Beside);
        assert_eq!(class_of(r"D:\elsewhere\title.exe", false, home), Class::Title);
    }

    #[test]
    fn a_sample_counts_for_the_innermost_frame_of_the_titles_or_ours() {
        use Class::*;
        assert_eq!(bucket(&[]), None);
        let title = Frame { at: 0x1_4000_3456, function: 0x1_4000_3456, class: Title };
        assert_eq!(bucket(&[title]), Some(Bucket::Title(0x1_4000_3)));
        // The title waits itself.
        assert_eq!(bucket(&[frame(System, 0x700), frame(System, 0x800), title]), Some(Bucket::TitleCalls(0x700)));
        // In a function of ours, and in the system under one: the innermost of ours.
        assert_eq!(bucket(&[frame(Ours, 0x100), frame(Ours, 0x200), title]), Some(Bucket::Ours(0x100, 0)));
        assert_eq!(bucket(&[frame(System, 0x700), frame(Driver, 0x900), frame(Ours, 0x100), title]), Some(Bucket::Ours(0x100, 0x700)));
        assert_eq!(bucket(&[frame(System, 0x700), frame(Unknown, 0x10)]), Some(Bucket::Other(0x700)));
        let mut tally = Tally::default();
        tally.add(&[title]);
        tally.add(&[title]);
        tally.add(&[frame(System, 0x700), frame(Ours, 0x100)]);
        tally.add(&[frame(Ours, 0x100)]);
        tally.add(&[]);
        assert_eq!((tally.samples, tally.missed), (4, 1));
        assert_eq!(tally.buckets[&Bucket::Title(0x1_4000_3)], 2);
        assert_eq!(tally.classes(), "title 50.0%, ours 25.0%, system 25.0%");
    }

    #[test]
    fn waits_and_yields_are_told_by_the_systems_names() {
        assert_eq!(wait_or_yield("ntdll!NtWaitForSingleObject"), (true, false));
        assert_eq!(wait_or_yield("ntdll!ZwWaitForAlertByThreadId"), (true, false));
        assert_eq!(wait_or_yield("ntdll!NtDelayExecution"), (true, false));
        assert_eq!(wait_or_yield("ntdll!NtYieldExecution"), (false, true));
        assert_eq!(wait_or_yield("KERNELBASE!SwitchToThread"), (false, true));
        assert_eq!(wait_or_yield("ntdll!RtlAllocateHeap"), (false, false));
    }

    #[test]
    fn the_thread_bound_to_the_processor_is_followed_else_the_busiest() {
        let threads = [(0xff, 900), (0x2000, 500), (0x2000, 700), (0x3fff, 100)];
        assert_eq!(pick(Target::Busiest, &threads), Some(0));
        assert_eq!(pick(Target::Processor(13), &threads), Some(2));
        assert_eq!(pick(Target::Processor(5), &threads), Some(0));
        assert_eq!(pick(Target::Busiest, &[]), None);
    }

    #[inline(never)]
    fn a_synthetic_thread_spins(stop: &std::sync::atomic::AtomicBool) -> u64 {
        let mut n = 0u64;
        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
            n = std::hint::black_box(n.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1));
        }
        n
    }

    /// A real thread of the test's own: while it spins the samples lie in
    /// our code, and while it sleeps they lie in the system under ours.
    #[test]
    fn a_synthetic_thread_is_sampled_in_its_loop_and_in_its_wait() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        use windows_sys::Win32::System::Threading::{GetCurrentThreadId, OpenThread, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME};
        let (stop, id, sleeping) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU32::new(0)), Arc::new(AtomicBool::new(false)));
        let thread = {
            let (stop, id, sleeping) = (stop.clone(), id.clone(), sleeping.clone());
            std::thread::spawn(move || {
                // SAFETY: a plain query.
                id.store(unsafe { GetCurrentThreadId() }, Ordering::SeqCst);
                std::hint::black_box(a_synthetic_thread_spins(&stop));
                sleeping.store(true, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(2500));
            })
        };
        while id.load(Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }
        // SAFETY: the test's own thread; memory of the test's own.
        let (handle, copy) = unsafe {
            (
                OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, id.load(Ordering::SeqCst)) as usize,
                VirtualAlloc(std::ptr::null(), COPY + ROOM, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE).cast::<u8>(),
            )
        };
        assert!(handle != 0 && !copy.is_null());
        let mut modules = Modules::default();
        // SAFETY: zeroes are a valid context.
        let mut context = Aligned(unsafe { std::mem::zeroed() });
        let mut frames = Vec::new();
        let mut sample = |tally: &mut Tally, modules: &mut Modules| {
            // SAFETY: as in `work`.
            unsafe {
                let (sp, len) = take(handle, &mut context, copy).expect("a sample");
                std::ptr::write_bytes(copy.add(len), 0, COPY - len);
                walk(&mut context, sp, len, copy as usize, modules, &mut frames);
            }
            tally.add(&frames);
            frames.clone()
        };
        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut spinning = Tally::default();
        let mut deepest = 0;
        for _ in 0..40 {
            deepest = deepest.max(sample(&mut spinning, &mut modules).len());
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(spinning.samples, 40);
        let in_ours = spinning.top.get(&Class::Ours).copied().unwrap_or(0);
        assert!(in_ours >= 30, "{}", spinning.classes());
        // The walk leaves the loop: its caller, the thread's start, the system's.
        assert!(deepest >= 3, "{deepest} frames");
        stop.store(true, Ordering::SeqCst);
        while !sleeping.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
        let mut asleep = Tally::default();
        for _ in 0..40 {
            sample(&mut asleep, &mut modules);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let (mut under_ours, mut waits) = (0, 0);
        for (bucket, count) in &asleep.buckets {
            if let Bucket::Ours(_, called) = *bucket
                && called != 0
            {
                under_ours += count;
                waits += if wait_or_yield(&modules.export(called)).0 { *count } else { 0 };
            }
        }
        assert!(asleep.top.get(&Class::System).copied().unwrap_or(0) >= 30, "{}", asleep.classes());
        assert!(under_ours >= 30, "{:?}", asleep.buckets);
        assert!(waits >= 30, "{:?}", asleep.buckets.keys().map(|b| match *b { Bucket::Ours(_, c) | Bucket::Other(c) | Bucket::TitleCalls(c) => modules.export(c), Bucket::Title(_) => String::new() }).collect::<Vec<_>>());
        thread.join().unwrap();
    }
}
