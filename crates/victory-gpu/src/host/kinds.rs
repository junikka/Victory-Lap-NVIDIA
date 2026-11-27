//! Diagnostics: what the card is asked to change between, counted
//! (`VICTORY_KIND_COUNTS=1`; off by default).
//!
//! A queue that goes from draws to dispatches to copies and back, an
//! unordered-access barrier, a change of descriptor heaps: each can make the
//! card let everything it has under way run out before it goes on (NVIDIA,
//! "Advanced API Performance": command buffers, async compute, barriers,
//! descriptors). The card's time by kind of list (`VICTORY_PASS_TIMES`) says
//! how long, not how often. These counts say how often, for every list of
//! the process: the title's, translated, and ours.
//!
//! How: with the switch set, the entries of the host list's method table for
//! draws, dispatches, copies, barriers, heaps and indirect execution, and
//! the queue's for `ExecuteCommandLists`, are pointed at functions here that
//! count and call what was there. No call site knows of it, so none can be
//! forgotten. Unset: nothing is touched and nothing is counted.
//!
//! What a call costs with the switch on: a load, a compare and one or two
//! relaxed adds (a few nanoseconds; 10,000 calls a frame are some tens of
//! microseconds).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

type Raw = *mut core::ffi::c_void;

/// Whether the counts are asked for.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KIND_COUNTS").is_some_and(|v| v == "1"))
}

/// A kind of work. A change from one to another inside a list, or from a
/// list's last to the next list's first, is what is counted.
pub(crate) const DRAW: usize = 1;
pub(crate) const DISPATCH: usize = 2;
pub(crate) const COPY: usize = 3;
/// `ExecuteIndirect`: draws or dispatches, as its signature says (not seen
/// here), so a kind of its own.
pub(crate) const INDIRECT: usize = 4;

const KIND_BITS: usize = 7;
const SLOTS: usize = 4096;

/// What is counted.
#[derive(Clone, Copy)]
#[repr(usize)]
enum N {
    Draws,
    Dispatches,
    Copies,
    Indirects,
    Clears,
    ToDraw,
    ToDispatch,
    ToCopy,
    ToIndirect,
    BarrierCalls,
    Waits,
    UavBarriers,
    Transitions,
    Aliasings,
    HeapSets,
    HeapChanges,
    OneGroup,
    FewGroups,
    Calls,
    Lists,
    ListsWithWork,
    Borders,
    Count,
}

/// The counts, and what the lists open now did last. A list is known by its
/// address, in a table without locks: two lists open at once whose addresses
/// fall on one place forget each other's last kind, and a change of kind
/// goes uncounted (never one too many).
pub(crate) struct Tally {
    /// Of lists of ours, and of the title's (translated; with what we put
    /// between its work).
    counts: [[AtomicU64; N::Count as usize]; 2],
    /// The title's lists' addresses ([`Tally::title`]).
    title: [AtomicUsize; SLOTS],
    /// A list's address with the kind of its last work.
    last: [AtomicUsize; SLOTS],
    /// And of its first.
    first: [AtomicUsize; SLOTS],
    /// A list's address, and the heaps it had set last.
    heaps_of: [AtomicUsize; SLOTS],
    heaps: [AtomicUsize; SLOTS],
    /// A queue's address with the kind of the last work sent to it.
    queues: [AtomicUsize; 16],
    /// A list's address, and what was asked of it since it began: draws,
    /// dispatches, copies, indirect executions, waits for every earlier
    /// shader write, barrier calls ([`Tally::of_list`]).
    per_of: [AtomicUsize; SLOTS],
    per: [[AtomicU32; PER]; SLOTS],
}

/// What is counted for each list by itself.
pub(crate) const PER: usize = 6;

fn slot(this: usize, of: usize) -> usize {
    (this >> 4).wrapping_mul(0x9E37_79B9_7F4A_7C15_u64 as usize) >> 20 & (of - 1)
}

impl Tally {
    pub(crate) const fn new() -> Self {
        Self {
            counts: [const { [const { AtomicU64::new(0) }; N::Count as usize] }; 2],
            title: [const { AtomicUsize::new(0) }; SLOTS],
            last: [const { AtomicUsize::new(0) }; SLOTS],
            first: [const { AtomicUsize::new(0) }; SLOTS],
            heaps_of: [const { AtomicUsize::new(0) }; SLOTS],
            heaps: [const { AtomicUsize::new(0) }; SLOTS],
            queues: [const { AtomicUsize::new(0) }; 16],
            per_of: [const { AtomicUsize::new(0) }; SLOTS],
            per: [const { [const { AtomicU32::new(0) }; PER] }; SLOTS],
        }
    }

    /// The list at `this` is one of the title's, from here on.
    pub(crate) fn title(&self, this: usize) {
        self.title[slot(this, SLOTS)].store(this, Relaxed);
    }

    fn add(&self, this: usize, n: N, by: u64) {
        let whose = usize::from(self.title[slot(this, SLOTS)].load(Relaxed) == this);
        self.counts[whose][n as usize].fetch_add(by, Relaxed);
    }

    /// Counts `by` of the list's own count `which` (of [`PER`]).
    fn own(&self, this: usize, which: usize, by: u32) {
        let at = slot(this, SLOTS);
        if self.per_of[at].swap(this, Relaxed) != this {
            // Another list's, or none's: this one's begin here.
            for c in &self.per[at] {
                c.store(0, Relaxed);
            }
        }
        self.per[at][which].fetch_add(by, Relaxed);
    }

    /// What was asked of the list at `this` since it began, as [`PER`]
    /// (nothing, if it is not known here).
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    pub(crate) fn of_list(&self, this: usize) -> [u32; PER] {
        let at = slot(this, SLOTS);
        if self.per_of[at].load(Relaxed) != this {
            return [0; PER];
        }
        std::array::from_fn(|i| self.per[at][i].load(Relaxed))
    }

    /// The list at `this` is asked for work of `kind`.
    pub(crate) fn work(&self, this: usize, kind: usize) {
        self.add(this, [N::Draws, N::Dispatches, N::Copies, N::Indirects][kind - 1], 1);
        self.own(this, kind - 1, 1);
        let at = slot(this, SLOTS);
        let was = self.last[at].swap(this | kind, Relaxed);
        if was & !KIND_BITS != this {
            // The list's first work (or a list forgotten).
            self.first[at].store(this | kind, Relaxed);
        } else if was & KIND_BITS != kind {
            self.add(this, [N::ToDraw, N::ToDispatch, N::ToCopy, N::ToIndirect][kind - 1], 1);
        }
    }

    /// A dispatch of `groups` thread groups.
    pub(crate) fn dispatch(&self, this: usize, groups: u64) {
        self.work(this, DISPATCH);
        if groups == 1 {
            self.add(this, N::OneGroup, 1);
        }
        if groups < 64 {
            self.add(this, N::FewGroups, 1);
        }
    }

    /// The list begins again.
    pub(crate) fn reset(&self, this: usize) {
        let at = slot(this, SLOTS);
        for table in [&self.last, &self.first, &self.heaps_of, &self.per_of] {
            if table[at].load(Relaxed) & !KIND_BITS == this {
                table[at].store(0, Relaxed);
            }
        }
    }

    /// A barrier call: waits for every earlier shader write, waits for one
    /// resource's, state changes and aliasings in it.
    pub(crate) fn barriers(&self, this: usize, waits: u64, uavs: u64, transitions: u64, aliasings: u64) {
        self.add(this, N::BarrierCalls, 1);
        self.own(this, 5, 1);
        if waits > 0 {
            self.own(this, 4, waits as u32);
        }
        for (n, by) in [(N::Waits, waits), (N::UavBarriers, uavs), (N::Transitions, transitions), (N::Aliasings, aliasings)] {
            if by > 0 {
                self.add(this, n, by);
            }
        }
    }

    /// A clear of a target or of depth.
    pub(crate) fn clear(&self, this: usize) {
        self.add(this, N::Clears, 1);
    }

    /// Descriptor heaps are set on the list: `set` stands for them.
    pub(crate) fn heaps(&self, this: usize, set: usize) {
        self.add(this, N::HeapSets, 1);
        let at = slot(this, SLOTS);
        if self.heaps_of[at].swap(this, Relaxed) == this && self.heaps[at].load(Relaxed) != set {
            self.add(this, N::HeapChanges, 1);
        }
        self.heaps[at].store(set, Relaxed);
    }

    /// `lists` go to `queue` in one call.
    pub(crate) fn sent(&self, queue: usize, lists: impl Iterator<Item = usize>) {
        // (The calls are counted with ours; a list and its border with the
        // list's own.)
        self.add(0, N::Calls, 1);
        let q = &self.queues[slot(queue, 16)];
        let mut before = match q.load(Relaxed) {
            was if was & !KIND_BITS == queue => was & KIND_BITS,
            _ => 0,
        };
        for this in lists {
            self.add(this, N::Lists, 1);
            let at = slot(this, SLOTS);
            let (first, last) = (self.first[at].load(Relaxed), self.last[at].load(Relaxed));
            // (A list of nothing but barriers or a stamp changes nothing.)
            if first & !KIND_BITS != this || last & !KIND_BITS != this {
                continue;
            }
            self.add(this, N::ListsWithWork, 1);
            if before != 0 && before != first & KIND_BITS {
                self.add(this, N::Borders, 1);
            }
            before = last & KIND_BITS;
        }
        q.store(queue | before, Relaxed);
    }

    /// The counts since the last call, each divided by `frames`.
    fn take(&self, whose: usize, frames: f64) -> [f64; N::Count as usize] {
        std::array::from_fn(|i| self.counts[whose][i].swap(0, Relaxed) as f64 / frames)
    }

    /// The line's text, for the lists of ours (0) or the title's (1).
    fn text(&self, whose: usize, frames: f64) -> String {
        let c = self.take(whose, frames);
        let n = |n: N| c[n as usize];
        format!(
            "{:.0} draws, {:.0} dispatches ({:.0} of one thread group, {:.0} of fewer than 64 groups), {:.0} copies, \
             {:.0} ExecuteIndirect calls, {:.0} clears; changes of kind inside lists {:.0} (to draws {:.0}, to dispatches \
             {:.0}, to copies {:.0}, to indirect {:.0}), at the lists' borders {:.0}; {:.0} lists sent, {:.0} of them \
             with such work; barrier calls {:.0}: {:.0} waits for every earlier shader write, {:.0} for one resource's, \
             {:.0} state changes, {:.0} aliasings; descriptor heaps set {:.0} times, {:.0} of them a change inside a list",
            n(N::Draws),
            n(N::Dispatches),
            n(N::OneGroup),
            n(N::FewGroups),
            n(N::Copies),
            n(N::Indirects),
            n(N::Clears),
            n(N::ToDraw) + n(N::ToDispatch) + n(N::ToCopy) + n(N::ToIndirect),
            n(N::ToDraw),
            n(N::ToDispatch),
            n(N::ToCopy),
            n(N::ToIndirect),
            n(N::Borders),
            n(N::Lists),
            n(N::ListsWithWork),
            n(N::BarrierCalls),
            n(N::Waits),
            n(N::UavBarriers),
            n(N::Transitions),
            n(N::Aliasings),
            n(N::HeapSets),
            n(N::HeapChanges),
        )
    }
}

static TALLY: Tally = Tally::new();

/// The methods that were there.
struct Was {
    reset: unsafe extern "system" fn(Raw, Raw, Raw) -> windows::core::HRESULT,
    draw: unsafe extern "system" fn(Raw, u32, u32, u32, u32),
    draw_indexed: unsafe extern "system" fn(Raw, u32, u32, u32, i32, u32),
    dispatch: unsafe extern "system" fn(Raw, u32, u32, u32),
    copy_buffer: unsafe extern "system" fn(Raw, Raw, u64, Raw, u64, u64),
    copy_texture: unsafe extern "system" fn(
        Raw,
        *const D3D12_TEXTURE_COPY_LOCATION,
        u32,
        u32,
        u32,
        *const D3D12_TEXTURE_COPY_LOCATION,
        *const D3D12_BOX,
    ),
    copy_resource: unsafe extern "system" fn(Raw, Raw, Raw),
    resolve: unsafe extern "system" fn(Raw, Raw, u32, Raw, u32, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT),
    barrier: unsafe extern "system" fn(Raw, u32, *const D3D12_RESOURCE_BARRIER),
    heaps: unsafe extern "system" fn(Raw, u32, *const Raw),
    clear_depth: unsafe extern "system" fn(
        Raw,
        D3D12_CPU_DESCRIPTOR_HANDLE,
        D3D12_CLEAR_FLAGS,
        f32,
        u8,
        u32,
        *const windows::Win32::Foundation::RECT,
    ),
    clear_target: unsafe extern "system" fn(Raw, D3D12_CPU_DESCRIPTOR_HANDLE, *const f32, u32, *const windows::Win32::Foundation::RECT),
    indirect: unsafe extern "system" fn(Raw, Raw, u32, Raw, u64, Raw, u64),
    execute: unsafe extern "system" fn(Raw, u32, *const Raw),
}

static WAS: OnceLock<Was> = OnceLock::new();

fn was() -> &'static Was {
    // (Set before any entry points here.)
    WAS.get().expect("the methods that were there")
}

// SAFETY (all of these): called by the list's or the queue's own callers with
// that method's arguments, which go on unchanged to the method that was
// there.
unsafe extern "system" fn reset(this: Raw, allocator: Raw, state: Raw) -> windows::core::HRESULT {
    TALLY.reset(this as usize);
    unsafe { (was().reset)(this, allocator, state) }
}
unsafe extern "system" fn draw(this: Raw, a: u32, b: u32, c: u32, d: u32) {
    TALLY.work(this as usize, DRAW);
    unsafe { (was().draw)(this, a, b, c, d) }
}
unsafe extern "system" fn draw_indexed(this: Raw, a: u32, b: u32, c: u32, d: i32, e: u32) {
    TALLY.work(this as usize, DRAW);
    unsafe { (was().draw_indexed)(this, a, b, c, d, e) }
}
unsafe extern "system" fn dispatch(this: Raw, x: u32, y: u32, z: u32) {
    TALLY.dispatch(this as usize, u64::from(x) * u64::from(y) * u64::from(z));
    unsafe { (was().dispatch)(this, x, y, z) }
}
unsafe extern "system" fn copy_buffer(this: Raw, to: Raw, at: u64, from: Raw, from_at: u64, bytes: u64) {
    TALLY.work(this as usize, COPY);
    unsafe { (was().copy_buffer)(this, to, at, from, from_at, bytes) }
}
unsafe extern "system" fn copy_texture(
    this: Raw,
    to: *const D3D12_TEXTURE_COPY_LOCATION,
    x: u32,
    y: u32,
    z: u32,
    from: *const D3D12_TEXTURE_COPY_LOCATION,
    part: *const D3D12_BOX,
) {
    TALLY.work(this as usize, COPY);
    unsafe { (was().copy_texture)(this, to, x, y, z, from, part) }
}
unsafe extern "system" fn copy_resource(this: Raw, to: Raw, from: Raw) {
    TALLY.work(this as usize, COPY);
    unsafe { (was().copy_resource)(this, to, from) }
}
unsafe extern "system" fn resolve(
    this: Raw,
    to: Raw,
    to_sub: u32,
    from: Raw,
    from_sub: u32,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
) {
    TALLY.work(this as usize, COPY);
    unsafe { (was().resolve)(this, to, to_sub, from, from_sub, format) }
}
unsafe extern "system" fn barrier(this: Raw, count: u32, barriers: *const D3D12_RESOURCE_BARRIER) {
    let (mut waits, mut uavs, mut transitions, mut aliasings) = (0, 0, 0, 0);
    if !barriers.is_null() {
        // SAFETY: `count` barriers, as the method takes them.
        for b in unsafe { std::slice::from_raw_parts(barriers, count as usize) } {
            match b.Type {
                // SAFETY: the union's member is the one `Type` says; a pointer read.
                D3D12_RESOURCE_BARRIER_TYPE_UAV if unsafe { b.Anonymous.UAV.pResource.is_none() } => waits += 1,
                D3D12_RESOURCE_BARRIER_TYPE_UAV => uavs += 1,
                D3D12_RESOURCE_BARRIER_TYPE_TRANSITION => transitions += 1,
                _ => aliasings += 1,
            }
        }
    }
    TALLY.barriers(this as usize, waits, uavs, transitions, aliasings);
    unsafe { (was().barrier)(this, count, barriers) }
}
unsafe extern "system" fn heaps(this: Raw, count: u32, heaps: *const Raw) {
    let mut set = count as usize;
    if !heaps.is_null() {
        // SAFETY: `count` heaps, as the method takes them.
        for h in unsafe { std::slice::from_raw_parts(heaps, count as usize) } {
            set = set.rotate_left(17) ^ *h as usize;
        }
    }
    TALLY.heaps(this as usize, set);
    unsafe { (was().heaps)(this, count, heaps) }
}
unsafe extern "system" fn clear_depth(
    this: Raw,
    view: D3D12_CPU_DESCRIPTOR_HANDLE,
    flags: D3D12_CLEAR_FLAGS,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const windows::Win32::Foundation::RECT,
) {
    TALLY.clear(this as usize);
    unsafe { (was().clear_depth)(this, view, flags, depth, stencil, count, rects) }
}
unsafe extern "system" fn clear_target(
    this: Raw,
    view: D3D12_CPU_DESCRIPTOR_HANDLE,
    colour: *const f32,
    count: u32,
    rects: *const windows::Win32::Foundation::RECT,
) {
    TALLY.clear(this as usize);
    unsafe { (was().clear_target)(this, view, colour, count, rects) }
}
unsafe extern "system" fn indirect(this: Raw, signature: Raw, most: u32, args: Raw, at: u64, count: Raw, count_at: u64) {
    TALLY.work(this as usize, INDIRECT);
    unsafe { (was().indirect)(this, signature, most, args, at, count, count_at) }
}
unsafe extern "system" fn execute(this: Raw, count: u32, lists: *const Raw) {
    if !lists.is_null() {
        // SAFETY: `count` lists, as the method takes them.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        TALLY.sent(this as usize, lists.iter().map(|l| *l as usize));
    }
    unsafe { (was().execute)(this, count, lists) }
}

/// With the switch set, once: the method tables of `list`'s and `queue`'s
/// classes count from here on (every list and queue of those classes, which
/// is every one the device makes of these types). Whether they do.
pub(crate) fn hook(list: &ID3D12GraphicsCommandList, queue: &ID3D12CommandQueue) -> bool {
    if !on() {
        return false;
    }
    static DONE: OnceLock<bool> = OnceLock::new();
    *DONE.get_or_init(|| install(list, queue))
}

/// The tables' entries are written through `f` while they may be.
///
/// # Safety
/// `at` is `len` bytes of a method table that stays mapped.
unsafe fn writable(at: *mut u8, len: usize, f: &mut dyn FnMut()) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let mut old = 0;
    // SAFETY: the caller's; the protection is put back as it was.
    unsafe {
        if VirtualProtect(at.cast(), len, PAGE_READWRITE, &mut old) == 0 {
            return false;
        }
        f();
        VirtualProtect(at.cast(), len, old, &mut old);
    }
    true
}

/// [`hook`]'s doing. (Done again, it keeps the methods that were there the
/// first time.)
fn install(list: &ID3D12GraphicsCommandList, queue: &ID3D12CommandQueue) -> bool {
    {
        // SAFETY: an interface pointer points at a pointer to its method
        // table, laid out as the bindings say; the tables stay mapped while
        // the device's library is loaded. Each entry is a pointer, written
        // whole; the old entries are kept before any new one can be called.
        unsafe {
            let lv = *(list.as_raw() as *mut *mut ID3D12GraphicsCommandList_Vtbl);
            let qv = *(queue.as_raw() as *mut *mut ID3D12CommandQueue_Vtbl);
            // (Never our own, if the tables count already.)
            if std::ptr::fn_addr_eq((*lv).Dispatch, dispatch as unsafe extern "system" fn(Raw, u32, u32, u32)) {
                return true;
            }
            let _ = WAS.set(Was {
                reset: (*lv).Reset,
                draw: (*lv).DrawInstanced,
                draw_indexed: (*lv).DrawIndexedInstanced,
                dispatch: (*lv).Dispatch,
                copy_buffer: (*lv).CopyBufferRegion,
                copy_texture: (*lv).CopyTextureRegion,
                copy_resource: (*lv).CopyResource,
                resolve: (*lv).ResolveSubresource,
                barrier: (*lv).ResourceBarrier,
                heaps: (*lv).SetDescriptorHeaps,
                clear_depth: (*lv).ClearDepthStencilView,
                clear_target: (*lv).ClearRenderTargetView,
                indirect: (*lv).ExecuteIndirect,
                execute: (*qv).ExecuteCommandLists,
            });
            let lists = writable(lv.cast(), size_of::<ID3D12GraphicsCommandList_Vtbl>(), &mut || {
                (*lv).Reset = reset;
                (*lv).DrawInstanced = draw;
                (*lv).DrawIndexedInstanced = draw_indexed;
                (*lv).Dispatch = dispatch;
                (*lv).CopyBufferRegion = copy_buffer;
                (*lv).CopyTextureRegion = copy_texture;
                (*lv).CopyResource = copy_resource;
                (*lv).ResolveSubresource = resolve;
                (*lv).ResourceBarrier = barrier;
                (*lv).SetDescriptorHeaps = heaps;
                (*lv).ClearDepthStencilView = clear_depth;
                (*lv).ClearRenderTargetView = clear_target;
                (*lv).ExecuteIndirect = indirect;
            });
            let queues = lists && writable(qv.cast(), size_of::<ID3D12CommandQueue_Vtbl>(), &mut || (*qv).ExecuteCommandLists = execute);
            if !queues {
                tracing::warn!("no counts of the kinds of work (VICTORY_KIND_COUNTS): the method tables could not be written");
            }
            queues
        }
    }
}

/// The method tables as they were before [`install`].
#[cfg(test)]
fn remove(list: &ID3D12GraphicsCommandList, queue: &ID3D12CommandQueue) {
    let Some(was) = WAS.get() else {
        return;
    };
    // SAFETY: as in `install`; the entries are the ones that were there.
    unsafe {
        let lv = *(list.as_raw() as *mut *mut ID3D12GraphicsCommandList_Vtbl);
        let qv = *(queue.as_raw() as *mut *mut ID3D12CommandQueue_Vtbl);
        writable(lv.cast(), size_of::<ID3D12GraphicsCommandList_Vtbl>(), &mut || {
            (*lv).Reset = was.reset;
            (*lv).DrawInstanced = was.draw;
            (*lv).DrawIndexedInstanced = was.draw_indexed;
            (*lv).Dispatch = was.dispatch;
            (*lv).CopyBufferRegion = was.copy_buffer;
            (*lv).CopyTextureRegion = was.copy_texture;
            (*lv).CopyResource = was.copy_resource;
            (*lv).ResolveSubresource = was.resolve;
            (*lv).ResourceBarrier = was.barrier;
            (*lv).SetDescriptorHeaps = was.heaps;
            (*lv).ClearDepthStencilView = was.clear_depth;
            (*lv).ClearRenderTargetView = was.clear_target;
            (*lv).ExecuteIndirect = was.indirect;
        });
        writable(qv.cast(), size_of::<ID3D12CommandQueue_Vtbl>(), &mut || (*qv).ExecuteCommandLists = was.execute);
    }
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let calls = TALLY.counts[0][N::Calls as usize].swap(0, Relaxed) as f64 / frames;
    tracing::info!(
        "kinds of work recorded per frame (VICTORY_KIND_COUNTS=1), in the title's lists (translated, with what we put \
         between its work): {}",
        TALLY.text(1, frames)
    );
    tracing::info!(
        "kinds of work recorded per frame (VICTORY_KIND_COUNTS=1), in lists of ours: {}; {calls:.0} calls of \
         ExecuteCommandLists for all of them",
        TALLY.text(0, frames)
    );
    crate::shadow::bound_line(frames);
}

/// With the switch set: what was asked of each of `lists` since it began,
/// summed (as [`PER`]: draws, dispatches, copies, indirect executions, waits
/// for every earlier shader write, barrier calls).
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn of_lists(lists: impl Iterator<Item = usize>) -> [u32; PER] {
    let mut sum = [0u32; PER];
    for this in lists {
        for (s, c) in sum.iter_mut().zip(TALLY.of_list(this)) {
            *s += c;
        }
    }
    sum
}

/// With the switch set: `list` is one of the title's (a host list).
pub(crate) fn title(list: &ID3D12GraphicsCommandList) {
    if on() {
        TALLY.title(list.as_raw() as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(t: &Tally, n: N) -> u64 {
        t.counts[0][n as usize].load(Relaxed) + t.counts[1][n as usize].load(Relaxed)
    }

    #[test]
    fn a_change_of_kind_inside_a_list_is_counted_and_a_lists_first_work_is_not() {
        let t = Tally::new();
        let (a, b) = (0x1000, 0x2340);
        // a: copy copy dispatch dispatch draw; b, between: draw draw.
        t.work(a, COPY);
        t.work(b, DRAW);
        t.work(a, COPY);
        t.dispatch(a, 1);
        t.work(b, DRAW);
        t.dispatch(a, 700);
        t.work(a, DRAW);
        assert_eq!([count(&t, N::Copies), count(&t, N::Dispatches), count(&t, N::Draws)], [2, 2, 3]);
        assert_eq!([count(&t, N::ToDispatch), count(&t, N::ToDraw), count(&t, N::ToCopy)], [1, 1, 0]);
        assert_eq!([count(&t, N::OneGroup), count(&t, N::FewGroups)], [1, 1]);
        // Begun again, the list's first work is no change.
        t.reset(a);
        t.work(a, COPY);
        assert_eq!(count(&t, N::ToCopy), 0);
        t.work(a, INDIRECT);
        assert_eq!([count(&t, N::ToIndirect), count(&t, N::Indirects)], [1, 1]);
    }

    #[test]
    fn a_border_counts_when_the_next_list_begins_with_another_kind() {
        let t = Tally::new();
        let (q, sync, stamp, title, compare) = (0x9000, 0x1000, 0x1100, 0x1200, 0x1300);
        t.work(sync, COPY);
        t.dispatch(sync, 40);
        t.work(title, DRAW);
        t.work(title, DRAW);
        t.dispatch(compare, 30);
        // sync ends in dispatches, the title's list is draws: one border;
        // the stamp between has no work; draws to the compare: another.
        t.sent(q, [sync, stamp, title].into_iter());
        t.sent(q, [compare].into_iter());
        assert_eq!(count(&t, N::Borders), 2);
        assert_eq!([count(&t, N::Calls), count(&t, N::Lists), count(&t, N::ListsWithWork)], [2, 4, 3]);
        // The next submission's sync list begins with copies: from the
        // compare's dispatches, a border; another queue has its own last.
        t.reset(sync);
        t.work(sync, COPY);
        t.sent(0xA000, [sync].into_iter());
        assert_eq!(count(&t, N::Borders), 2);
        t.sent(q, [sync].into_iter());
        assert_eq!(count(&t, N::Borders), 3);
        // Dispatches behind dispatches: none.
        t.reset(compare);
        t.dispatch(compare, 1);
        t.reset(title);
        t.dispatch(title, 1);
        t.sent(q, [compare, title].into_iter());
        assert_eq!(count(&t, N::Borders), 4, "copies to dispatches");
        t.sent(q, [compare].into_iter());
        assert_eq!(count(&t, N::Borders), 4);
    }

    /// On the card: a list of ours is counted through the method tables,
    /// runs as before, and is counted no more once the tables are as they were.
    #[test]
    fn the_cards_lists_are_counted_through_their_method_tables_and_run_as_before() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let Some((allocator, list)) = host.make_prelude() else {
            return;
        };
        let (Some(from), Some(to)) = (host.shared_buffer(4096), host.read_back_buffer(4096)) else {
            return;
        };
        // SAFETY: the mapping is the buffer's 4096 bytes.
        unsafe { std::slice::from_raw_parts_mut(from.1, 4096) }.iter_mut().enumerate().for_each(|(i, b)| *b = (i % 251) as u8);
        let run = || {
            // SAFETY: an own list, its last run finished; live buffers; the
            // fence and event are the host's, under its lock.
            unsafe {
                allocator.Reset().unwrap();
                list.Reset(&allocator, None).unwrap();
                list.CopyBufferRegion(&to, 0, &from.0, 0, 2048);
                list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(None),
                        }),
                    },
                }]);
                list.CopyBufferRegion(&to, 2048, &from.0, 2048, 2048);
                list.Close().unwrap();
                let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
                *submitted += 1;
                host.queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                host.queue.Signal(&host.fence, *submitted).unwrap();
                host.fence.SetEventOnCompletion(*submitted, host.event).unwrap();
                windows::Win32::System::Threading::WaitForSingleObject(host.event, windows::Win32::System::Threading::INFINITE);
                // What was copied is there, counted or not.
                let mut data = std::ptr::null_mut();
                to.Map(0, None, Some(&mut data)).unwrap();
                let got = std::slice::from_raw_parts(data.cast::<u8>(), 4096).to_vec();
                to.Unmap(0, None);
                assert!(got.iter().enumerate().all(|(i, &b)| b == (i % 251) as u8), "the copies ran");
            }
        };
        let this = list.as_raw() as usize;
        assert!(install(&list, &host.queue));
        assert!(install(&list, &host.queue), "twice is once");
        // The list's own counts (other threads of the host may send lists
        // of theirs meanwhile): two copies, a wait in one barrier call.
        run();
        assert_eq!(TALLY.of_list(this), [0, 0, 2, 0, 1, 1]);
        // Begun again, it counts from nothing.
        run();
        assert_eq!(TALLY.of_list(this), [0, 0, 2, 0, 1, 1]);
        // With the tables as they were, nothing more is counted.
        remove(&list, &host.queue);
        TALLY.work(this, COPY);
        run();
        assert_eq!(TALLY.of_list(this), [0, 0, 3, 0, 1, 1]);
    }

    #[test]
    fn a_list_has_its_own_counts_until_it_begins_again() {
        let t = Tally::new();
        let (a, b) = (0x1000, 0x2340);
        t.work(a, DRAW);
        t.work(a, DRAW);
        t.work(a, INDIRECT);
        t.barriers(a, 2, 0, 1, 0);
        t.barriers(a, 0, 0, 3, 0);
        t.dispatch(b, 8);
        t.work(b, COPY);
        assert_eq!(t.of_list(a), [2, 0, 0, 1, 2, 2]);
        assert_eq!(t.of_list(b), [0, 1, 1, 0, 0, 0]);
        assert_eq!(t.of_list(0x7770), [0; PER]);
        t.reset(a);
        assert_eq!(t.of_list(a), [0; PER]);
        t.barriers(a, 1, 0, 0, 0);
        assert_eq!(t.of_list(a), [0, 0, 0, 0, 1, 1]);
        assert_eq!(t.of_list(b), [0, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn heaps_set_again_inside_a_list_are_a_change_only_if_they_differ() {
        let t = Tally::new();
        t.heaps(0x1000, 7);
        t.heaps(0x1000, 7);
        t.heaps(0x1000, 9);
        t.heaps(0x2000, 7);
        assert_eq!([count(&t, N::HeapSets), count(&t, N::HeapChanges)], [4, 1]);
        t.reset(0x1000);
        t.heaps(0x1000, 7);
        assert_eq!(count(&t, N::HeapChanges), 1);
    }

    #[test]
    fn barriers_are_counted_by_what_they_are_and_the_line_is_per_frame() {
        let t = Tally::new();
        // 0x1000 is a list of the title's, 0x2000 one of ours.
        t.title(0x1000);
        t.barriers(0x1000, 2, 0, 6, 0);
        t.barriers(0x1000, 0, 4, 0, 2);
        t.barriers(0x2000, 1, 0, 0, 0);
        t.barriers(0x2000, 1, 0, 0, 0);
        for _ in 0..4 {
            t.work(0x1000, DRAW);
        }
        t.dispatch(0x2000, 9);
        t.dispatch(0x2000, 9);
        let title = t.text(1, 2.0);
        assert!(title.starts_with("2 draws, 0 dispatches"), "{title}");
        assert!(
            title.contains("barrier calls 1: 1 waits for every earlier shader write, 2 for one resource's, 3 state changes, 1 aliasings"),
            "{title}"
        );
        let ours = t.text(0, 2.0);
        assert!(ours.starts_with("0 draws, 1 dispatches (0 of one thread group, 1 of fewer than 64 groups)"), "{ours}");
        assert!(ours.contains("barrier calls 1: 1 waits for every earlier shader write, 0 for one resource's"), "{ours}");
        assert_eq!(count(&t, N::BarrierCalls), 0, "taken");
    }
}
