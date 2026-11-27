//! A host list's D3D12 list, with barrier calls joined
//! (`VICTORY_JOIN_BARRIERS=1`; off by default; A/B-able with
//! `VICTORY_AB=<s>,VICTORY_JOIN_BARRIERS`).
//!
//! The state changes the host asks for (`crate::list::ensure_state`, the
//! aliasing before a texture's first upload) and the title's own barriers
//! wait here instead of going to the list at once; they go, in the order
//! they were asked for, in one call, at the next use of the list for
//! anything else. Every use goes through [`Rec`]'s `Deref`, so nothing is
//! recorded before a barrier asked for earlier. In a replay of capture
//! replay5 joining the calls that had no work between them saved 2.4 ms of
//! 79.5 (2,490 calls joined, 1,619 of them a texture's state change after
//! its upload followed by the next texture's aliasing).

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;

/// Whether barriers are held now (`VICTORY_JOIN_BARRIERS=1`, or its phase
/// of `VICTORY_AB`).
pub(crate) fn on() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_JOIN_BARRIERS") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_BARRIERS").is_some_and(|v| v == "1"))
}

/// Barriers held. Their resources are borrowed: the list's states hold the
/// textures (`HostList::transition`), and the title's barriers name what
/// the title keeps alive until the list has run.
struct Held(Vec<D3D12_RESOURCE_BARRIER>);

// SAFETY: borrowed interface pointers of free-threaded objects.
unsafe impl Send for Held {}

/// Since the last frame-rate line: barrier calls asked for while held, and
/// calls made of them.
pub(crate) static STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether a wait for every earlier shader write (an unordered-access
/// barrier that names no resource) is left out when the last thing asked of
/// the list that was not a barrier came before another such wait
/// (`VICTORY_WAIT_ONCE=1`; off by default).
///
/// The title's barrier groups each become one such wait
/// (`crate::list::barrier_groups`), 800 a frame on the path, and the title
/// asks for several groups in a row. The second wait of two with only state
/// changes between them orders nothing: no shader ran since the first, so
/// every write it waits for was waited for already, and what comes after is
/// behind both. The card is told to let its unordered-access work run out
/// once instead of twice.
pub(crate) fn wait_once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAIT_ONCE").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line: waits for every earlier shader write
/// asked for, those of them right behind another (nothing but barriers
/// between), and those left out ([`wait_once`]).
pub(crate) static WAITS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Since the last frame-rate line: times the title's descriptor heaps were
/// bound in a host list (`crate::draw`): once a list at least, and again
/// after every pass of ours that binds a heap of its own. A change of heaps
/// can make the card wait for all it has under way, the more so with a heap
/// at the host's limit of 1,000,000 views, as ours is.
pub(crate) static HEAP_BINDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether `b` is a wait for every earlier shader write.
fn is_wait(b: &D3D12_RESOURCE_BARRIER) -> bool {
    // SAFETY: the union's member is the one `Type` says; a pointer read.
    b.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV && unsafe { b.Anonymous.UAV.pResource.is_none() }
}

/// Whether the list was asked for nothing but barriers since a wait for
/// every earlier shader write.
#[derive(Default)]
struct Waited(AtomicBool);

impl Waited {
    /// Barriers are asked for, `waits` of them such waits: how many of the
    /// waits are behind another one with nothing run between.
    fn asked(&self, waits: usize) -> usize {
        if waits == 0 {
            return 0;
        }
        if self.0.swap(true, Relaxed) { waits } else { waits - 1 }
    }

    /// The list is used for something else.
    fn used(&self) {
        if self.0.load(Relaxed) {
            self.0.store(false, Relaxed);
        }
    }
}

/// `barriers` without the waits for every earlier shader write but the
/// first `keep` (0 or 1) of them.
fn without_waits(barriers: &[D3D12_RESOURCE_BARRIER], mut keep: usize) -> Vec<D3D12_RESOURCE_BARRIER> {
    barriers
        .iter()
        .filter(|b| {
            if !is_wait(b) {
                return true;
            }
            let kept = keep > 0;
            keep = keep.saturating_sub(1);
            kept
        })
        // SAFETY: a plain copy of the descriptions; nothing is released
        // through them (their pointers are borrowed, `ManuallyDrop`).
        .map(|b| unsafe { std::ptr::read(b) })
        .collect()
}

pub(crate) struct Rec {
    list: ID3D12GraphicsCommandList,
    any: AtomicBool,
    held: Mutex<Held>,
    waited: Waited,
}

impl Rec {
    pub(crate) fn new(list: ID3D12GraphicsCommandList) -> Self {
        Self { list, any: AtomicBool::new(false), held: Mutex::new(Held(Vec::new())), waited: Waited::default() }
    }

    /// Records `barriers` now, or holds them for the next use of the list.
    ///
    /// # Safety
    /// As `ResourceBarrier`'s: an open list, live resources.
    pub(crate) unsafe fn barriers(&self, barriers: &[D3D12_RESOURCE_BARRIER]) {
        if barriers.is_empty() {
            return;
        }
        // Waits for every earlier shader write behind another one
        // ([`wait_once`]): counted, and left out if asked.
        let waits = barriers.iter().filter(|b| is_wait(b)).count();
        let again = self.waited.asked(waits);
        if waits > 0 {
            WAITS[0].fetch_add(waits as u64, Relaxed);
            WAITS[1].fetch_add(again as u64, Relaxed);
        }
        let lean;
        let barriers = if again > 0 && wait_once() {
            WAITS[2].fetch_add(again as u64, Relaxed);
            lean = without_waits(barriers, waits - again);
            if lean.is_empty() {
                return;
            }
            &lean[..]
        } else {
            barriers
        };
        if !on() {
            // (What was held while it was on goes first.)
            if self.any.load(Relaxed) {
                self.flush();
            }
            // SAFETY: the caller's.
            unsafe { self.list.ResourceBarrier(barriers) };
            return;
        }
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        // A resource whose state changes again: what is held goes first, so
        // that no call holds two changes of one resource.
        if barriers.iter().filter_map(named).any(|r| held.0.iter().filter_map(named).any(|h| h == r)) {
            // SAFETY: as in `flush`.
            unsafe { self.list.ResourceBarrier(&held.0) };
            held.0.clear();
            STATS[1].fetch_add(1, Relaxed);
        }
        // SAFETY: a plain copy of the descriptions; nothing is released
        // through them (their pointers are borrowed, `ManuallyDrop`).
        held.0.extend(barriers.iter().map(|b| unsafe { std::ptr::read(b) }));
        STATS[0].fetch_add(1, Relaxed);
        self.any.store(true, Relaxed);
    }

    /// What is held goes to the list, in one call.
    fn flush(&self) {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        self.any.store(false, Relaxed);
        if held.0.is_empty() {
            return;
        }
        // SAFETY: the barriers were asked for on this open list, in order;
        // their resources are alive (`Held`).
        unsafe { self.list.ResourceBarrier(&held.0) };
        held.0.clear();
        STATS[1].fetch_add(1, Relaxed);
    }
}

/// The resource a state change or aliasing names (its interface pointer).
fn named(b: &D3D12_RESOURCE_BARRIER) -> Option<usize> {
    // SAFETY: the union's member is the one `Type` says; a pointer read.
    unsafe {
        let p: Option<&ID3D12Resource> = match b.Type {
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION => b.Anonymous.Transition.pResource.as_ref(),
            D3D12_RESOURCE_BARRIER_TYPE_ALIASING => b.Anonymous.Aliasing.pResourceAfter.as_ref(),
            _ => None,
        };
        p.map(|r| windows::core::Interface::as_raw(r) as usize)
    }
}

impl std::ops::Deref for Rec {
    type Target = ID3D12GraphicsCommandList;

    /// The list, with what is held recorded first.
    fn deref(&self) -> &ID3D12GraphicsCommandList {
        if self.any.load(Relaxed) {
            self.flush();
        }
        self.waited.used();
        &self.list
    }
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    let [asked, made] = [0, 1].map(|i| STATS[i].swap(0, Relaxed) as f64 / frames);
    if asked > 0.0 {
        tracing::info!(
            "barrier calls per frame (VICTORY_JOIN_BARRIERS): {asked:.0} asked for, {made:.0} made of them"
        );
    }
    let binds = HEAP_BINDS.swap(0, Relaxed) as f64 / frames;
    if binds > 0.0 {
        tracing::info!("the title's descriptor heaps bound in host lists per frame: {binds:.0} (a list's first, and again after a pass of ours with its own heap)");
    }
    let [waits, again, left] = [0, 1, 2].map(|i| WAITS[i].swap(0, Relaxed) as f64 / frames);
    if waits > 0.0 {
        tracing::info!(
            "waits for every earlier shader write recorded per frame: {waits:.0}, {again:.0} of them behind another with \
             nothing but barriers between, {left:.0} left out (VICTORY_WAIT_ONCE=1 leaves those out: now {})",
            if wait_once() { "on" } else { "off" }
        );
    }
}

#[cfg(test)]
mod wait_tests {
    use super::*;

    fn wait() -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        }
    }

    fn change() -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                    Subresource: 0,
                    StateBefore: D3D12_RESOURCE_STATE_COMMON,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_DEST,
                }),
            },
        }
    }

    #[test]
    fn the_switch_is_off_unless_asked() {
        // (The suites run without it.)
        if std::env::var_os("VICTORY_WAIT_ONCE").is_none() {
            assert!(!wait_once());
        }
    }

    #[test]
    fn only_a_wait_behind_another_with_nothing_run_between_is_again() {
        let w = Waited::default();
        // The first wait of a list is needed: earlier lists' shaders.
        assert_eq!(w.asked(1), 0);
        // Right behind it, and behind a state change: again.
        assert_eq!(w.asked(1), 1);
        assert_eq!(w.asked(0), 0);
        assert_eq!(w.asked(1), 1);
        // Something else was recorded (a draw, a dispatch, a copy): needed.
        w.used();
        assert_eq!(w.asked(0), 0);
        assert_eq!(w.asked(1), 0);
        // Two in one call: the second is again; behind a wait, both.
        w.used();
        assert_eq!(w.asked(2), 1);
        assert_eq!(w.asked(2), 2);
    }

    #[test]
    fn state_changes_stay_when_waits_are_left_out() {
        assert!(is_wait(&wait()) && !is_wait(&change()));
        let all = [change(), wait(), change(), wait()];
        let kinds = |v: &[D3D12_RESOURCE_BARRIER]| v.iter().map(is_wait).collect::<Vec<_>>();
        assert_eq!(kinds(&without_waits(&all, 0)), [false, false]);
        assert_eq!(kinds(&without_waits(&all, 1)), [false, true, false]);
        assert!(without_waits(&[wait()], 0).is_empty());
    }

    /// On a list of the host's: the waits are counted, a use of the list
    /// makes the next one needed again, and the list closes as before.
    #[test]
    fn a_list_counts_its_waits() {
        let Some(host) = crate::host::host() else {
            return;
        };
        // SAFETY: plain object creation; an open list of our own.
        unsafe {
            let Ok(allocator) = host.device.CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT) else {
                return;
            };
            let list: ID3D12GraphicsCommandList =
                host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
            let rec = Rec::new(list);
            let before = WAITS[1].load(Relaxed);
            rec.barriers(&[wait()]);
            rec.barriers(&[wait()]);
            assert!(WAITS[1].load(Relaxed) > before);
            // A use of the list: the next wait is needed again.
            rec.SetGraphicsRootSignature(None);
            assert_eq!(rec.waited.asked(1), 0);
            rec.Close().unwrap();
        }
    }
}
