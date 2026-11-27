//! The title's own waits (`monitorx` / `mwaitx` loops on a word in memory), counted, timed and
//! waited another way. All of it is off unless asked for: without the two variables below the
//! wait instructions are replaced where they stand, as before (`console_insn`).
//!
//! `VICTORY_WAIT_COUNT=1`: every five seconds a line says, for each thread and each place in
//! the title's code that waits so, how many waits it made, how many passes of the loop they
//! took, how long they lasted in all, the longest, and how they divide by length; and for
//! each word waited on, what memory it lies in.
//! `VICTORY_WAIT_COUNT=fault` counts the same from the exception each pass raises, with no
//! code replaced: slower passes, for when no place could be hooked.
//!
//! `VICTORY_MWAITX=pause` (the default: what the replaced instruction does), `yield` (the rest
//! of the time slice goes to another thread ready for this processor), `sleep0`, `umwait`
//! and `umwait1` (the processor's own light wait on the word, as `mwaitx` is on the console's:
//! C0.2 or C0.1, until the word's line is written or 100 microseconds passed; `pause` if the
//! processor lacks WAITPKG). Any value but `pause` counts too.
//!
//! How: a wait instruction is 3 bytes, a jump out of the title's code 5. A hooked place
//! becomes a short jump to 5 bytes of padding between two functions within its reach (a run of
//! `int3` up to a 16-byte boundary), which jump on to a stub beside the title's code; the stub
//! calls [`pass`] with every register and the flags kept, and jumps back behind the
//! instruction. A place with no such padding in reach is replaced as before and said so in
//! the line.
//!
//! On the title's threads a pass costs two time-stamp reads and a thread-local store; the
//! shared counters are written once a wait, each thread's on lines of its own. Nothing is
//! allocated or locked there (one lock when a place is first hooked).

use crate::console_insn::{jump, near, patch, readable, STUBS_LEN};
use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};

/// How an `mwaitx` waits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    Pause,
    Yield,
    Sleep0,
    /// `umwait` asking for C0.2 (`false`) or C0.1 (`true`).
    Umwait(bool),
}

impl Mode {
    fn parse(value: &str) -> Option<Self> {
        Some(match value.trim() {
            "" | "pause" => Self::Pause,
            "yield" => Self::Yield,
            "sleep0" => Self::Sleep0,
            "umwait" => Self::Umwait(false),
            "umwait1" => Self::Umwait(true),
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::Pause => "pause",
            Self::Yield => "yield",
            Self::Sleep0 => "sleep0",
            Self::Umwait(false) => "umwait",
            Self::Umwait(true) => "umwait1",
        }
    }
}

/// Whether the processor has `umonitor` / `umwait` (WAITPKG: leaf 7, ECX bit 5).
pub(crate) fn waitpkg() -> bool {
    static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HAS.get_or_init(|| {
        // SAFETY: every x86-64 processor answers these.
        #[allow(unused_unsafe)]
        unsafe {
            let leaves = std::arch::x86_64::__cpuid(0).eax;
            leaves >= 7 && std::arch::x86_64::__cpuid_count(7, 0).ecx & 1 << 5 != 0
        }
    })
}

/// `VICTORY_MWAITX`, with `umwait` on a processor without it as `pause`.
pub(crate) fn mode() -> Mode {
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let asked = std::env::var("VICTORY_MWAITX").unwrap_or_default();
        let mode = Mode::parse(&asked).unwrap_or_else(|| {
            tracing::warn!("VICTORY_MWAITX={asked}: not pause, yield, sleep0, umwait or umwait1; pause");
            Mode::Pause
        });
        if matches!(mode, Mode::Umwait(_)) && !waitpkg() {
            tracing::warn!("VICTORY_MWAITX={asked}: this processor has no umwait (WAITPKG); pause");
            return Mode::Pause;
        }
        mode
    })
}

/// `VICTORY_WAIT_COUNT`: 0 not set, 1 counted at hooked places, 2 (`fault`) counted from the
/// exceptions with nothing replaced.
fn count() -> u8 {
    static COUNT: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *COUNT.get_or_init(|| match std::env::var("VICTORY_WAIT_COUNT").as_deref() {
        Ok("1") => 1,
        Ok("fault") => 2,
        _ => 0,
    })
}

/// Whether waits are hooked or counted at all: else `console_insn` does what it always did.
pub(crate) fn wanted() -> bool {
    count() != 0 || mode() != Mode::Pause
}

/// Whether every pass is to stay an exception (`VICTORY_WAIT_COUNT=fault`).
pub(crate) fn by_fault() -> bool {
    count() == 2
}

const MONITOR: u32 = 0;
const WAIT: u32 = 1;

/// The places hooked or counted: the instruction's address, and whether it is an `mwaitx`.
static PLACES: [(AtomicUsize, AtomicU32); 32] = [const { (AtomicUsize::new(0), AtomicU32::new(0)) }; 32];
/// The address each `monitorx` place last named: the word its loop waits on.
static WORDS: [AtomicUsize; 32] = [const { AtomicUsize::new(0) }; 32];
/// Places hooked, and those left as they were for want of padding in reach.
static HOOKED: AtomicU32 = AtomicU32::new(0);
static UNHOOKED: AtomicU32 = AtomicU32::new(0);

/// The number of the place at `at`, a new one if it has none; `None` when all are taken.
fn place(at: usize, kind: u32) -> Option<u32> {
    for (i, p) in PLACES.iter().enumerate() {
        let there = p.0.compare_exchange(0, at, Relaxed, Relaxed).unwrap_or_else(|there| there);
        if there == 0 {
            p.1.store(kind, Relaxed);
        }
        if there == 0 || there == at {
            return Some(i as u32);
        }
    }
    None
}

/// What one thread did at one place since the last line. A line of its own, so that two
/// threads never write the same one.
#[repr(align(128))]
struct Row {
    /// Thread and place + 1; 0: free.
    key: AtomicU64,
    waits: AtomicU64,
    passes: AtomicU64,
    ticks: AtomicU64,
    longest: AtomicU64,
    /// The logical processor the last wait ended on.
    processor: AtomicU32,
    /// Waits by length: under 10 and 100 microseconds, 1, 5 and 20 milliseconds, and longer.
    lengths: [AtomicU64; 6],
}

static ROWS: [Row; 64] = [const {
    Row {
        key: AtomicU64::new(0),
        waits: AtomicU64::new(0),
        passes: AtomicU64::new(0),
        ticks: AtomicU64::new(0),
        longest: AtomicU64::new(0),
        processor: AtomicU32::new(0),
        lengths: [const { AtomicU64::new(0) }; 6],
    }
}; 64];

/// Time-stamp ticks a second, measured by the thread that writes the line (a guess until
/// its first line: 2.4 GHz).
static TICKS: AtomicU64 = AtomicU64::new(2_400_000_000);

fn row(thread: u32, place: u32) -> Option<&'static Row> {
    let key = u64::from(thread) << 32 | u64::from(place + 1);
    let mut i = (thread as usize ^ place as usize * 7) % ROWS.len();
    for _ in 0..ROWS.len() {
        let r = &ROWS[i];
        let there = r.key.compare_exchange(0, key, Relaxed, Relaxed).unwrap_or_else(|there| there);
        if there == 0 || there == key {
            return Some(r);
        }
        i = (i + 1) % ROWS.len();
    }
    None
}

/// A thread's wait under way.
#[derive(Clone, Copy)]
struct Open {
    /// Place + 1; 0: none.
    place: u32,
    passes: u32,
    thread: u32,
    start: u64,
    /// When its last pass went back to the title.
    left: u64,
}

thread_local! {
    static OPEN: Cell<Open> = const { Cell::new(Open { place: 0, passes: 0, thread: 0, start: 0, left: 0 }) };
}

/// The index in [`Row::lengths`] of a wait of `ticks` at `second` ticks a second.
fn length(ticks: u64, second: u64) -> usize {
    let us = ticks.saturating_mul(1_000_000) / second.max(1);
    [10, 100, 1_000, 5_000, 20_000].iter().position(|&under| us < under).unwrap_or(5)
}

/// Adds a wait that ended to its thread's row.
fn close(open: &Open) {
    if open.place == 0 || open.passes == 0 {
        return;
    }
    let Some(row) = row(open.thread, open.place - 1) else { return };
    let took = open.left.saturating_sub(open.start);
    row.waits.fetch_add(1, Relaxed);
    row.passes.fetch_add(u64::from(open.passes), Relaxed);
    row.ticks.fetch_add(took, Relaxed);
    row.longest.fetch_max(took, Relaxed);
    row.lengths[length(took, TICKS.load(Relaxed))].fetch_add(1, Relaxed);
    // SAFETY: a plain query.
    row.processor.store(unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessorNumber() }, Relaxed);
}

fn stamp() -> u64 {
    // SAFETY: every x86-64 processor has it.
    unsafe { std::arch::x86_64::_rdtsc() }
}

/// Two passes of one wait lie no further apart than this in the title's own code (ticks:
/// about 15 microseconds). A longer gap starts another wait, so a thread taken off its
/// processor inside the loop counts two.
const GAP: u64 = 36_000;
/// The longest a `umwait` stands before the title looks at its word again (ticks: about 100
/// microseconds), whatever the title asked for.
const LONGEST_UMWAIT: u64 = 240_000;

/// One `mwaitx` as `mode` has it. `timeout`: the title's, in ticks, 0 for none.
fn wait(mode: Mode, timeout: u64) {
    match mode {
        Mode::Pause => std::hint::spin_loop(),
        // SAFETY: plain calls.
        Mode::Yield => unsafe {
            if windows_sys::Win32::System::Threading::SwitchToThread() == 0 {
                std::hint::spin_loop();
            }
        },
        // SAFETY: a plain call.
        Mode::Sleep0 => unsafe { windows_sys::Win32::System::Threading::Sleep(0) },
        Mode::Umwait(light) => {
            let until = stamp() + if timeout == 0 { LONGEST_UMWAIT } else { timeout.min(LONGEST_UMWAIT) };
            // SAFETY: `umwait ecx` (asked for only where the processor has it): waits for a
            // write to the line last named to `umonitor`, the time in edx:eax, or anything
            // that interrupts; it writes the flags and nothing else.
            unsafe {
                std::arch::asm!(".byte 0xf2, 0x0f, 0xae, 0xf1",
                    in("ecx") u32::from(light), in("eax") until as u32, in("edx") (until >> 32) as u32,
                    options(nostack, nomem));
            }
        }
    }
}

/// One pass of a wait loop: `id` is the place's number and, from bit 8, [`MONITOR`] or
/// [`WAIT`]; `rax`, `rbx` and `rcx` are the title's at the instruction (`monitorx`: the
/// address in `rax`; `mwaitx`: a time limit in `ebx` if bit 1 of `ecx` is set).
///
/// Called from the stubs on the title's threads: no allocation, no lock, no panic.
pub(crate) extern "C" fn pass(id: u32, rax: u64, rbx: u64, rcx: u64) {
    let mode = mode();
    if id >> 8 == MONITOR {
        // (Read first: the line is written only when the word is another.)
        let word = &WORDS[(id & 0xff) as usize % WORDS.len()];
        if word.load(Relaxed) != rax as usize {
            word.store(rax as usize, Relaxed);
        }
        if matches!(mode, Mode::Umwait(_)) {
            // SAFETY: `umonitor rcx` (asked for only where the processor has it) arms the
            // wait on that address and touches nothing; a bad address makes the wait end
            // at once, as the title's own instruction would.
            unsafe { std::arch::asm!(".byte 0xf3, 0x0f, 0xae, 0xf1", in("rcx") rax, options(nostack, nomem, preserves_flags)) };
        }
        return;
    }
    let place = (id & 0xff) + 1;
    let now = stamp();
    let mut open = OPEN.with(Cell::get);
    if open.place != place || now.saturating_sub(open.left) > GAP {
        close(&open);
        if open.thread == 0 {
            // SAFETY: a plain query.
            open.thread = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
        }
        open = Open { place, passes: 0, start: now, ..open };
    }
    open.passes += 1;
    wait(mode, if rcx & 2 != 0 { rbx & 0xffff_ffff } else { 0 });
    open.left = stamp();
    OPEN.with(|o| o.set(open));
    started();
}

/// A pass that came as an exception (`VICTORY_WAIT_COUNT=fault`, or a place not hooked):
/// counted and waited like any.
pub(crate) fn pass_from_fault(at: usize, wait: bool, rax: u64, rbx: u64, rcx: u64) {
    let kind = if wait { WAIT } else { MONITOR };
    match place(at, kind) {
        Some(n) => pass(kind << 8 | n, rax, rbx, rcx),
        None => std::hint::spin_loop(),
    }
}

// The stubs' common part. In: the return address, the place's id and the title's rax and
// flags on the stack (the stub pushed them); every other register as the title has it. The
// integer registers a call may change are kept here, and the vector registers in full (the
// halves above 128 bits belong to no callee): all sixteen with AVX, else the six a call may
// change. The stack is the title's, whose code keeps nothing below its stack pointer.
std::arch::global_asm!(
    ".globl victory_wait_pass_avx",
    "victory_wait_pass_avx:",
    "push rcx", "push rdx", "push r8", "push r9", "push r10", "push r11", "push rbp",
    "mov rbp, rsp",
    "and rsp, -32",
    "sub rsp, 0x220",
    "vmovups [rsp + 0x20], ymm0", "vmovups [rsp + 0x40], ymm1", "vmovups [rsp + 0x60], ymm2", "vmovups [rsp + 0x80], ymm3",
    "vmovups [rsp + 0xa0], ymm4", "vmovups [rsp + 0xc0], ymm5", "vmovups [rsp + 0xe0], ymm6", "vmovups [rsp + 0x100], ymm7",
    "vmovups [rsp + 0x120], ymm8", "vmovups [rsp + 0x140], ymm9", "vmovups [rsp + 0x160], ymm10", "vmovups [rsp + 0x180], ymm11",
    "vmovups [rsp + 0x1a0], ymm12", "vmovups [rsp + 0x1c0], ymm13", "vmovups [rsp + 0x1e0], ymm14", "vmovups [rsp + 0x200], ymm15",
    "mov ecx, [rbp + 64]",
    "mov rdx, [rbp + 72]",
    "mov r8, rbx",
    "mov r9, [rbp + 48]",
    "call {pass}",
    "vmovups ymm0, [rsp + 0x20]", "vmovups ymm1, [rsp + 0x40]", "vmovups ymm2, [rsp + 0x60]", "vmovups ymm3, [rsp + 0x80]",
    "vmovups ymm4, [rsp + 0xa0]", "vmovups ymm5, [rsp + 0xc0]", "vmovups ymm6, [rsp + 0xe0]", "vmovups ymm7, [rsp + 0x100]",
    "vmovups ymm8, [rsp + 0x120]", "vmovups ymm9, [rsp + 0x140]", "vmovups ymm10, [rsp + 0x160]", "vmovups ymm11, [rsp + 0x180]",
    "vmovups ymm12, [rsp + 0x1a0]", "vmovups ymm13, [rsp + 0x1c0]", "vmovups ymm14, [rsp + 0x1e0]", "vmovups ymm15, [rsp + 0x200]",
    "mov rsp, rbp",
    "pop rbp", "pop r11", "pop r10", "pop r9", "pop r8", "pop rdx", "pop rcx",
    "ret",
    ".globl victory_wait_pass_sse",
    "victory_wait_pass_sse:",
    "push rcx", "push rdx", "push r8", "push r9", "push r10", "push r11", "push rbp",
    "mov rbp, rsp",
    "and rsp, -16",
    "sub rsp, 0x80",
    "movups [rsp + 0x20], xmm0", "movups [rsp + 0x30], xmm1", "movups [rsp + 0x40], xmm2",
    "movups [rsp + 0x50], xmm3", "movups [rsp + 0x60], xmm4", "movups [rsp + 0x70], xmm5",
    "mov ecx, [rbp + 64]",
    "mov rdx, [rbp + 72]",
    "mov r8, rbx",
    "mov r9, [rbp + 48]",
    "call {pass}",
    "movups xmm0, [rsp + 0x20]", "movups xmm1, [rsp + 0x30]", "movups xmm2, [rsp + 0x40]",
    "movups xmm3, [rsp + 0x50]", "movups xmm4, [rsp + 0x60]", "movups xmm5, [rsp + 0x70]",
    "mov rsp, rbp",
    "pop rbp", "pop r11", "pop r10", "pop r9", "pop r8", "pop rdx", "pop rcx",
    "ret",
    pass = sym pass,
);

unsafe extern "C" {
    fn victory_wait_pass_avx();
    fn victory_wait_pass_sse();
}

/// The stub of the place `id` whose instruction ends at `back`, as code at `stub`.
fn stub_code(stub: usize, id: u32, back: usize) -> Option<Vec<u8>> {
    let common =
        if std::is_x86_feature_detected!("avx") { victory_wait_pass_avx as usize } else { victory_wait_pass_sse as usize };
    let mut code = vec![0x9c, 0x50, 0x68]; // pushfq; push rax; push id
    code.extend(id.to_le_bytes());
    code.extend([0x48, 0xb8]); // mov rax, the common part
    code.extend((common as u64).to_le_bytes());
    code.extend([0xff, 0xd0]); // call rax
    code.extend([0x48, 0x8d, 0x64, 0x24, 0x08]); // lea rsp, [rsp+8]
    code.extend([0x58, 0x9d]); // pop rax; popfq
    code.extend(jump(stub + code.len(), back)?);
    Some(code)
}

/// What hooking keeps: the stubs' memory (where, bytes used) and the padding taken.
struct Room {
    stubs: Option<(usize, usize)>,
    taken: [usize; 64],
}

static ROOM: std::sync::Mutex<Room> = std::sync::Mutex::new(Room { stubs: None, taken: [0; 64] });

/// Five bytes of padding a short jump at `at` reaches: the end of a run of `int3` that goes
/// up to a 16-byte boundary (what stands between two functions), not in `taken`.
/// shortcut: five `int3` up to a boundary are taken for padding without the code around
/// them being read; data of that shape inside code would be overwritten.
///
/// # Safety
/// `at` is in mapped code.
unsafe fn padding(at: usize, taken: &[usize]) -> Option<usize> {
    // A short jump at `at` lands from `at - 126` to `at + 129`.
    let (low, high) = (at.checked_sub(126)?, at + 129);
    let first = (low + 5).next_multiple_of(16);
    let mut best: Option<usize> = None;
    for end in (first..=high + 5).step_by(16) {
        let slot = end - 5;
        if slot < low || slot > high || taken.contains(&slot) || !readable(slot, 5) {
            continue;
        }
        // SAFETY: readable, asked above.
        if unsafe { std::slice::from_raw_parts(slot as *const u8, 5) } != [0xcc; 5] {
            continue;
        }
        if best.is_none_or(|b| slot.abs_diff(at) < b.abs_diff(at)) {
            best = Some(slot);
        }
    }
    best
}

/// Hooks the wait instruction at `at`, whose 3 bytes are `was`: whether it is hooked now.
/// A place that cannot be (no padding in reach, no room) is counted as such, once.
///
/// # Safety
/// `at` is in the title's code, where the instruction just faulted.
pub(crate) unsafe fn hook(at: usize, was: [u8; 3], wait: bool) -> bool {
    let kind = if wait { WAIT } else { MONITOR };
    let mut room = ROOM.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: the caller's.
    let done = unsafe { hook_in(&mut room, at, was, kind) };
    drop(room);
    if done {
        HOOKED.fetch_add(1, Relaxed);
    } else {
        UNHOOKED.fetch_add(1, Relaxed);
    }
    started();
    done
}

unsafe fn hook_in(room: &mut Room, at: usize, was: [u8; 3], kind: u32) -> bool {
    let Some(free) = room.taken.iter().position(|&t| t == 0) else { return false };
    // SAFETY: the caller's.
    let Some(slot) = (unsafe { padding(at, &room.taken) }) else { return false };
    let Ok(short) = i8::try_from(slot as i64 - (at as i64 + 2)) else { return false };
    let Some(n) = place(at, kind) else { return false };
    if room.stubs.is_none() {
        room.stubs = near(at).map(|place| (place, 0));
    }
    let Some((block, used)) = room.stubs.as_mut() else { return false };
    let stub = *block + *used;
    let (Some(body), Some(far)) = (stub_code(stub, kind << 8 | n, at + 3), jump(slot, stub)) else { return false };
    if *used + body.len() > STUBS_LEN {
        return false;
    }
    // SAFETY: the stub goes into memory made for it, which nothing runs yet; then the
    // padding, which nothing runs; then the instruction, in one store, by a jump that does
    // what it did (a wait that may end at once) with every register and the flags kept.
    unsafe {
        std::ptr::copy_nonoverlapping(body.as_ptr(), stub as *mut u8, body.len());
        *used += body.len().next_multiple_of(16);
        if !patch(slot, [0xcc; 5], far) {
            return false;
        }
        room.taken[free] = slot;
        patch(at, was, [0xeb, short as u8, 0x90])
    }
}

/// Starts the thread that writes the line, once. (Not in tests, which read the rows.)
fn started() {
    if cfg!(test) {
        return;
    }
    static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if STARTED.load(Relaxed) || STARTED.swap(true, Relaxed) {
        return;
    }
    // (The one allocation a title thread may pay for here: once a run.)
    let _ = std::thread::Builder::new().name("victory-wait-count".into()).spawn(|| {
        let mut before = (std::time::Instant::now(), stamp());
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let now = (std::time::Instant::now(), stamp());
            let seconds = now.0.duration_since(before.0).as_secs_f64();
            let second = ((now.1.saturating_sub(before.1)) as f64 / seconds) as u64;
            TICKS.store(second.max(1), Relaxed);
            before = now;
            tracing::info!("{}", line(seconds, second.max(1)));
        }
    });
}

/// What Windows says of the memory at `at`: its kind, its protection and the allocation it
/// lies in. Asked by the thread that writes the line, never by the title's.
fn memory(at: usize) -> String {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_IMAGE, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, PAGE_WRITECOMBINE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 || m.State != MEM_COMMIT {
        return "not committed memory".to_owned();
    }
    let kind = match m.Type {
        MEM_PRIVATE => "private memory",
        MEM_MAPPED => "a mapped section",
        MEM_IMAGE => "an image",
        _ => "memory of another kind",
    };
    format!(
        "{kind}, protection {:#x}{}, in the allocation at {:#x}, {} KiB into it",
        m.Protect,
        if m.Protect & PAGE_WRITECOMBINE != 0 { " (write-combined)" } else { "" },
        m.AllocationBase as usize,
        (at - m.AllocationBase as usize) >> 10
    )
}

/// The five-second line, and the rows emptied for the next.
fn line(seconds: f64, second: u64) -> String {
    let ms = |ticks: u64| ticks as f64 * 1000.0 / second as f64;
    let mut rows: Vec<(u64, String)> = Vec::new();
    for r in &ROWS {
        let key = r.key.load(Relaxed);
        let waits = r.waits.swap(0, Relaxed);
        if key == 0 || waits == 0 {
            continue;
        }
        let (passes, ticks, longest) = (r.passes.swap(0, Relaxed), r.ticks.swap(0, Relaxed), r.longest.swap(0, Relaxed));
        let lengths: Vec<String> = r.lengths.iter().map(|n| n.swap(0, Relaxed).to_string()).collect();
        let place = PLACES[(key as u32 - 1) as usize % PLACES.len()].0.load(Relaxed);
        rows.push((
            ticks,
            format!(
                "thread {} (last on processor {}) at {place:#x}: {:.0} waits a second, {:.0} passes a second, {:.1} ms a second waiting, {:.3} ms a wait, longest {:.2} ms, by length (under 10 us/100 us/1 ms/5 ms/20 ms/over) {}",
                key >> 32,
                r.processor.load(Relaxed),
                waits as f64 / seconds,
                passes as f64 / seconds,
                ms(ticks) / seconds,
                ms(ticks) / waits as f64,
                ms(longest),
                lengths.join("/")
            ),
        ));
    }
    rows.sort_unstable_by_key(|r| std::cmp::Reverse(r.0));
    let all: u64 = rows.iter().map(|r| r.0).sum();
    let rows: Vec<String> = rows.into_iter().take(8).map(|r| r.1).collect();
    let words: Vec<String> = WORDS
        .iter()
        .map(|w| w.load(Relaxed))
        .filter(|&w| w != 0)
        .map(|w| format!("{w:#x} ({})", memory(w)))
        .collect();
    format!(
        "title waits (mwaitx as {}{}): places hooked {}, left as pause for want of padding in reach {}; {:.1} ms a second in all; {}; the words waited on last: {}",
        mode().name(),
        if by_fault() { ", counted by exception" } else { "" },
        HOOKED.load(Relaxed),
        UNHOOKED.load(Relaxed),
        ms(all) / seconds,
        if rows.is_empty() { "no wait ended in this time".to_owned() } else { rows.join("; ") },
        if words.is_empty() { "none seen".to_owned() } else { words.join(", ") }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect,
    };

    #[test]
    fn modes_are_read_by_name() {
        assert_eq!(Mode::parse(""), Some(Mode::Pause));
        assert_eq!(Mode::parse("pause"), Some(Mode::Pause));
        assert_eq!(Mode::parse(" yield "), Some(Mode::Yield));
        assert_eq!(Mode::parse("sleep0"), Some(Mode::Sleep0));
        assert_eq!(Mode::parse("umwait"), Some(Mode::Umwait(false)));
        assert_eq!(Mode::parse("umwait1"), Some(Mode::Umwait(true)));
        assert_eq!(Mode::parse("1"), None);
        for mode in [Mode::Pause, Mode::Yield, Mode::Sleep0, Mode::Umwait(false), Mode::Umwait(true)] {
            assert_eq!(Mode::parse(mode.name()), Some(mode));
        }
    }

    #[test]
    fn waits_divide_by_length() {
        let second = 2_000_000_000;
        assert_eq!(length(0, second), 0);
        assert_eq!(length(19_999, second), 0);
        assert_eq!(length(20_000, second), 1);
        assert_eq!(length(1_999_999, second), 2);
        assert_eq!(length(2_000_000, second), 3);
        assert_eq!(length(10_000_000, second), 4);
        assert_eq!(length(40_000_000, second), 5);
        assert_eq!(length(u64::MAX, second), 5);
        assert_eq!(length(5, 0), 5, "no rate: nothing divides by zero");
    }

    /// Every way of waiting comes back, the timed ones within their limit.
    #[test]
    fn every_mode_waits_and_returns() {
        let mut modes = vec![Mode::Pause, Mode::Yield, Mode::Sleep0];
        if waitpkg() {
            modes.extend([Mode::Umwait(false), Mode::Umwait(true)]);
        }
        for mode in modes {
            let t = std::time::Instant::now();
            for timeout in [0, 1, 1000, u64::from(u32::MAX)] {
                wait(mode, timeout);
            }
            assert!(t.elapsed().as_millis() < 500, "{mode:?} took {:?}", t.elapsed());
        }
        if waitpkg() {
            // Armed on a word nothing writes: the wait ends by its time.
            let word = 0u64;
            pass(MONITOR << 8 | 31, &raw const word as u64, 0, 0);
            let t = std::time::Instant::now();
            wait(Mode::Umwait(false), 0);
            assert!(t.elapsed().as_millis() < 100);
        }
    }

    /// A page of `int3` with `code` at 0x40, made runnable.
    unsafe fn page(code: &[u8]) -> *mut u8 {
        // SAFETY: memory of our own.
        unsafe {
            let page = VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE).cast::<u8>();
            assert!(!page.is_null());
            std::ptr::write_bytes(page, 0xcc, 4096);
            std::ptr::copy_nonoverlapping(code.as_ptr(), page.add(0x40), code.len());
            let mut old = 0;
            assert_ne!(VirtualProtect(page.cast(), 4096, PAGE_EXECUTE_READ, &mut old), 0);
            page
        }
    }

    #[test]
    fn padding_is_five_int3_up_to_a_boundary_within_a_short_jump() {
        // SAFETY: a page of our own.
        unsafe {
            // A function that fills its sixteen bytes, then one that leaves four, then seven.
            let mut code = vec![0x90u8; 16 + 12 + 4 + 9];
            code[28..32].fill(0xcc);
            code.extend([0xcc; 7]);
            code.extend([0x90; 16 * 20]);
            let page = page(&code);
            let at = page as usize + 0x40;
            // Before the code the page is all padding: the nearest boundary behind wins.
            assert_eq!(padding(at, &[]), Some(at - 5));
            assert_eq!(padding(at, &[at - 5]), Some(at - 21));
            // From further in, the seven bytes before the third boundary (four are too few).
            let from = at + 0xa0;
            assert_eq!(padding(from, &[]), Some(at + 48 - 5));
            assert_eq!(padding(from, &[at + 48 - 5]), None, "the page's own padding is out of reach");
            VirtualFree(page.cast(), 0, MEM_RELEASE);
        }
    }

    /// `monitorx; mwaitx; ret` hooked: both run through their stubs, every register and the
    /// flags come back as they went in, and the waits are counted for this thread.
    #[test]
    fn a_hooked_wait_keeps_the_registers_and_is_counted() {
        // SAFETY: a page of our own, run within it.
        unsafe {
            let page = page(&[0x0f, 0x01, 0xfa, 0x0f, 0x01, 0xfb, 0xc3]);
            let at = page as usize + 0x40;
            assert!(hook(at, [0x0f, 0x01, 0xfa], false));
            assert!(hook(at + 3, [0x0f, 0x01, 0xfb], true));
            let now = std::slice::from_raw_parts(page.add(0x40), 7);
            assert_eq!((now[0], now[2], now[3], now[5], now[6]), (0xeb, 0x90, 0xeb, 0x90, 0xc3));
            let word = 7u64;
            let regs_in: [u64; 7] = [&raw const word as u64, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666];
            let mut regs = regs_in;
            let (carry, low, high): (u8, u64, u64);
            // (The ymm instructions of this test are of AVX2; the stubs need AVX only.)
            let avx = std::is_x86_feature_detected!("avx2");
            std::arch::asm!(
                "test {avx:e}, {avx:e}",
                "jz 2f",
                "vpcmpeqd ymm15, ymm15, ymm15",
                "vpcmpeqd ymm0, ymm0, ymm0",
                "2:",
                "sub rsp, 32",
                "stc",
                "call {f}",
                "setc {carry}",
                "add rsp, 32",
                "movq {low}, xmm0",
                "xor {high:e}, {high:e}",
                "test {avx:e}, {avx:e}",
                "jz 3f",
                "vpmovmskb {high:e}, ymm15",
                "3:",
                f = in(reg) at,
                avx = in(reg) u64::from(avx),
                carry = out(reg_byte) carry,
                low = out(reg) low,
                high = out(reg) high,
                inout("rax") regs[0], inout("rcx") regs[1], inout("rdx") regs[2], inout("r8") regs[3],
                inout("r9") regs[4], inout("r10") regs[5], inout("r11") regs[6],
                out("xmm0") _, out("xmm15") _,
            );
            assert_eq!(regs, regs_in);
            assert_eq!(carry, 1, "the flags");
            if avx {
                assert_eq!(low, u64::MAX, "xmm0");
                assert_eq!(high, 0xffff_ffff, "all of ymm15");
            }
            // The wait is this thread's open one; the next at another place closes it.
            let open = OPEN.with(Cell::get);
            assert_eq!(open.passes, 1);
            assert!(open.left >= open.start);
            let place = open.place - 1;
            assert_eq!(PLACES[place as usize].0.load(Relaxed), at + 3);
            close(&open);
            let row = row(open.thread, place).unwrap();
            assert!(row.waits.load(Relaxed) >= 1 && row.passes.load(Relaxed) >= 1);
            let text = line(5.0, 2_400_000_000);
            assert!(text.starts_with("title waits (mwaitx as "), "{text}");
            assert!(text.contains(&format!("thread {} ", open.thread)), "{text}");
            // The word the loop named (this test's own, on its stack: private memory).
            assert!(text.contains(&format!("{:#x} (private memory, protection 0x4, ", &raw const word as usize)), "{text}");

            // What a pass costs, against the replaced instruction's own (printed, not judged).
            let plain = self::page(&[0x0f, 0x1f, 0x00, 0xf3, 0x90, 0x90, 0xc3]);
            let run = |f: *const u8| {
                let f: extern "system" fn() = std::mem::transmute(f);
                let t = std::time::Instant::now();
                for _ in 0..200_000 {
                    f();
                }
                t.elapsed().as_nanos() as f64 / 200_000.0
            };
            OPEN.with(|o| o.set(Open { place: 0, ..open }));
            let (hooked, replaced) = (run(page.add(0x40)), run(plain.add(0x40)));
            println!("a pass of the loop: hooked {hooked:.1} ns, replaced in place {replaced:.1} ns ({:?})", mode());
            assert!(OPEN.with(Cell::get).passes >= 1);
            VirtualFree(plain.cast(), 0, MEM_RELEASE);
            // (The hooked page stays: its padding is in the table of taken places.)
        }
    }

    /// A place with no padding in reach is not touched.
    #[test]
    fn a_place_without_padding_is_left() {
        // SAFETY: a page of our own.
        unsafe {
            let mut code = vec![0x90u8; 600];
            code[300..303].copy_from_slice(&[0x0f, 0x01, 0xfb]);
            let page = page(&code);
            let at = page as usize + 0x40 + 300;
            let mut room = Room { stubs: None, taken: [0; 64] };
            assert!(!hook_in(&mut room, at, [0x0f, 0x01, 0xfb], WAIT));
            assert_eq!(std::slice::from_raw_parts(at as *const u8, 3), [0x0f, 0x01, 0xfb]);
            assert!(room.taken.iter().all(|&t| t == 0));
            VirtualFree(page.cast(), 0, MEM_RELEASE);
        }
    }

    /// Passes from exceptions are counted like hooked ones.
    #[test]
    fn a_pass_from_a_fault_is_counted() {
        std::thread::spawn(|| {
            pass_from_fault(0x7000_1000, true, 0, 0, 0);
            pass_from_fault(0x7000_1000, true, 0, 0, 0);
            let open = OPEN.with(Cell::get);
            assert_eq!(open.passes, 2);
            assert_eq!(PLACES[open.place as usize - 1].0.load(Relaxed), 0x7000_1000);
        })
        .join()
        .unwrap();
    }
}
