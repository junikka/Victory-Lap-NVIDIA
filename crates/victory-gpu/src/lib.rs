//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees. Planned:
//! `present` (swap chain).

#[cfg(windows)]
mod ambient;
#[cfg(windows)]
mod lightprobe;
#[cfg(windows)]
mod texlate;
#[cfg(windows)]
mod texstore;
#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
#[cfg(windows)]
mod compose;
pub mod console;
pub mod failed;
#[cfg(all(windows, feature = "diagnostics"))]
mod account;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;

/// Release builds with the `mimalloc` feature: the Rust side allocates
/// through mimalloc (per-thread caches) instead of the process heap, which
/// it shares with the title and the drivers. Diagnostics builds count every
/// block through `copystat`'s allocator, which then uses mimalloc too.
#[cfg(all(feature = "mimalloc", not(all(windows, feature = "diagnostics"))))]
#[global_allocator]
static MIMALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;
#[cfg(all(windows, feature = "diagnostics"))]
mod faultbook;
#[cfg(windows)]
mod alt;
#[cfg(windows)]
mod cores;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod draw;
#[cfg(all(windows, feature = "diagnostics"))]
mod draw_rank;
pub mod dxgi;
#[cfg(windows)]
pub mod external;
mod fallback;
pub mod fence;
pub mod frame;
#[cfg(all(windows, feature = "diagnostics"))]
mod book;
#[cfg(windows)]
mod byuse;
#[cfg(windows)]
mod census;
#[cfg(windows)]
mod heap_worker;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
#[cfg(windows)]
mod inplace;
#[cfg(windows)]
mod port;
#[cfg(windows)]
mod port_cbv;
#[cfg(windows)]
pub mod port_descriptors;
#[cfg(windows)]
mod port_journal;
#[cfg(windows)]
mod land;
#[cfg(windows)]
mod lazy_back;
pub mod list;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod picked;
#[cfg(windows)]
mod pool;
#[cfg(all(windows, feature = "diagnostics"))]
mod popwatch;
#[cfg(windows)]
mod primitive;
#[cfg(windows)]
mod priority;
pub mod remap;
pub mod query;
pub mod queue;
#[cfg(all(windows, feature = "diagnostics"))]
pub mod replay;
#[cfg(windows)]
mod residency;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod safecopy;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
mod serve;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod memory;
#[cfg(windows)]
mod readkinds;
#[cfg(windows)]
mod shadow;
#[cfg(all(windows, feature = "diagnostics"))]
mod shot;
mod settings_menu;
#[cfg(windows)]
mod fpsbox;
#[cfg(windows)]
mod stats;
#[cfg(windows)]
mod tiles;
pub mod signature;
mod switch;
#[cfg(all(windows, feature = "diagnostics"))]
mod ab;
#[cfg(windows)]
mod mapped_trim;
#[cfg(windows)]
mod tdr;
#[cfg(windows)]
mod texture;
pub mod timeline;
#[cfg(windows)]
pub mod titlewrite;
#[cfg(windows)]
mod upload;
#[cfg(windows)]
mod watch;
mod waitwhy;
#[cfg(windows)]
mod windowskip;
#[cfg(windows)]
mod texback;

#[cfg(windows)]
pub use cores::{
    affinity_mask_for_title, core_log_may, cpu_sets_for_title, group_mask_for_title, ideal_number_for_title,
    ideal_processor_for_title, keep_disk_on_slow_cores, note_audio_thread, place_title_current, place_title_handle,
    prepare_core_placement, thread_import, title_cores_may,
};
#[cfg(windows)]
pub use census::title_memory_reserved;
pub use waitwhy::note_disk_window;
#[cfg(windows)]
pub use host::on_failure;
pub use settings_menu::{SETTINGS_MENU_OPEN, menu_on as settings_menu_on};

/// Whether the memory account is kept (`VICTORY_MEMORY_ACCOUNT`,
/// diagnostics builds): the runtime then answers the title's plain
/// `VirtualAlloc` import itself and tells of each allocation
/// ([`title_host_allocation`]). Otherwise that import is the host's own
/// function, as it always was.
pub fn memory_account_on() -> bool {
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        account::on()
    }
    #[cfg(not(all(windows, feature = "diagnostics")))]
    {
        false
    }
}

/// The title made an allocation of `size` bytes at `at` with its plain
/// `VirtualAlloc` import (for the memory account; nothing else reads it).
pub fn title_host_allocation(at: usize, size: usize) {
    #[cfg(all(windows, feature = "diagnostics"))]
    account::title_plain(at, size as u64);
    #[cfg(not(all(windows, feature = "diagnostics")))]
    let _ = (at, size);
}

/// The title released the allocation at `at` it made with its plain
/// `VirtualAlloc` import (for the memory account).
pub fn title_host_released(at: usize) {
    #[cfg(all(windows, feature = "diagnostics"))]
    account::title_plain_gone(at);
    #[cfg(not(all(windows, feature = "diagnostics")))]
    let _ = at;
}

/// Whether `switch` is on now, if it is one of those `VICTORY_AB` turns on and
/// off while the title runs (`ab`).
#[cfg(all(windows, feature = "diagnostics"))]
pub fn switched(switch: &str) -> Option<bool> {
    ab::pick(switch)
}

/// What the frame is at (`copystat::phase`), for the sampling profiler.
#[cfg(all(windows, feature = "diagnostics"))]
pub fn frame_phase() -> u32 {
    copystat::phase()
}

/// Frames the title has begun (its frame waits that ended), for logs.
pub fn frames_begun() -> u64 {
    frame::current()
}
#[cfg(windows)]
pub use inplace::{title_memory_freeing, title_memory_section, title_memory_section_closed, title_memory_section_gone, title_memory_view};
#[cfg(windows)]
pub use watch::{
    kernel_writing, title_memory_allocated, title_memory_freed, title_memory_reading, title_memory_remapped,
    title_memory_remapping, title_memory_splitting, title_memory_unmapping, title_memory_written,
    writing as kernel_will_write,
};
