//! Page-mapping meter (`VICTORY_REMAP_METER`).
//!
//! Off unless the variable is set to something other than `0`. The base
//! mapping path counts nothing and reads no cycle counters, and this tree
//! has no frame log to pour counts into, so an unset variable keeps that
//! path. Set, the system's mapping calls and the queue's page-mapping
//! copies are counted and a line is written about once a second.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether the meter runs. Unset and `0` are off.
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REMAP_METER").is_some_and(|v| v != "0"))
}

/// `VICTORY_REMAP_SPLIT`: 0 unset (split a placeholder first, as before),
/// 1 `late` (map first, split only if that failed), 2 `alt`.
pub fn split_mode(value: Option<&str>) -> u8 {
    match value {
        Some("late") => 1,
        Some("alt") => 2,
        _ => 0,
    }
}

/// Whether page `at` of `section` continues the page mapped just before
/// (`prev_after` is the address after that page). A run is the next index
/// of the same section; the same index is that page again.
pub fn follows(
    prev_after: u64,
    prev_section: u64,
    prev_index: u64,
    at: u64,
    section: u64,
    index: u64,
) -> (bool, bool) {
    if prev_after == at && prev_section == section {
        if index == prev_index.wrapping_add(1) {
            return (true, false);
        }
        if index == prev_index {
            return (false, true);
        }
    }
    (false, false)
}

const PAGE: u64 = 64 << 10;

/// Calls, microseconds, thread cycles and timestamp-counter ticks, four
/// system calls (unmap, split, map, commit).
static SYSCALL: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
/// Queue page-mapping copies: calls, then ranges and pages of kinds 0, 1
/// and 2.
static PAGES: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
/// Alias pages, those that continued a run, those that were the same page,
/// and splits tried only because a mapping failed first.
static ALIAS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn note() {
    static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
    let Ok(mut last) = LAST.lock() else {
        return;
    };
    let now = std::time::Instant::now();
    if last.is_some_and(|t| now.duration_since(t).as_secs() < 1) {
        return;
    }
    *last = Some(now);
    let take = |c: &[AtomicU64]| -> Vec<u64> { c.iter().map(|n| n.swap(0, Relaxed)).collect() };
    let sys = take(&SYSCALL);
    let pages = take(&PAGES);
    let alias = take(&ALIAS);
    if sys.iter().chain(&pages).chain(&alias).all(|&n| n == 0) {
        return;
    }
    tracing::info!(
        "remapping, metered (VICTORY_REMAP_METER): unmap {} calls {} us, split {} {}, map {} {}, commit {} {}; \
         page copies {} (kind 0: {} ranges {} pages, kind 1: {} {}, kind 2: {} {}); \
         alias pages {} ({} continued, {} the same page), splits after a failed map {}",
        sys[0], sys[1], sys[4], sys[5], sys[8], sys[9], sys[12], sys[13],
        pages[0], pages[1], pages[2], pages[3], pages[4], pages[5], pages[6],
        alias[0], alias[1], alias[2], alias[3],
    );
}

/// One system call that changes a mapping. `kind` is [`crate`] xmem's
/// `Call` discriminant (unmap 7, split 9, map 11, commit 13). `cycles` is
/// the calling thread's processor cycles over the call, 0 when they were
/// not read.
pub fn syscall(kind: u8, us: u64, cycles: u64, tsc: u64) {
    if !on() {
        return;
    }
    let slot = match kind {
        7 => 0,
        9 => 4,
        11 => 8,
        13 => 12,
        _ => return,
    };
    SYSCALL[slot].fetch_add(1, Relaxed);
    SYSCALL[slot + 1].fetch_add(us, Relaxed);
    SYSCALL[slot + 2].fetch_add(cycles, Relaxed);
    SYSCALL[slot + 3].fetch_add(tsc, Relaxed);
    note();
}

/// One range of a queue page-mapping copy. Kinds 0, 1 and 2 are counted.
pub fn page_range(kind: u32, pages: u32) {
    if !on() {
        return;
    }
    PAGES[0].fetch_add(1, Relaxed);
    if let Some(slot) = match kind {
        0 => Some(1),
        1 => Some(3),
        2 => Some(5),
        _ => None,
    } {
        PAGES[slot].fetch_add(1, Relaxed);
        PAGES[slot + 1].fetch_add(u64::from(pages), Relaxed);
    }
    note();
}

thread_local! {
    static PREV: std::cell::Cell<(u64, u64, u64)> = const { std::cell::Cell::new((0, 0, 0)) };
}

/// One page mapped through [`crate`] xmem's `map_alias`.
pub fn alias_page(at: u64, section: u64, index: u64) {
    if !on() {
        return;
    }
    let (after, sec, idx) = PREV.with(|p| p.get());
    let (run, same) = follows(after, sec, idx, at, section, index);
    ALIAS[0].fetch_add(1, Relaxed);
    if run {
        ALIAS[1].fetch_add(1, Relaxed);
    }
    if same {
        ALIAS[2].fetch_add(1, Relaxed);
    }
    PREV.with(|p| p.set((at.wrapping_add(PAGE), section, index)));
    note();
}

/// A split made only after a mapping into the placeholder failed.
pub fn late_split() {
    if !on() {
        return;
    }
    ALIAS[3].fetch_add(1, Relaxed);
    note();
}

/// Page-mapping counters on the 5 s line (`VICTORY_MAP_COUNT=1`; unset or
/// anything else: nothing is counted and no clock is read for it).
///
/// Why: in the slow spells title threads wait for the host's address-space
/// lock inside the runtime's mapping calls, and one waits for the runtime's
/// own lock behind them. The line says, for the last five seconds and by
/// who called, the calls, their pages, the time in them, the waits for the
/// two locks of ours, the upkeep of what the GPU side notes of the pages,
/// and each kind of host call with its time; then the threads that spent
/// the most in mapping calls.
pub mod count {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::time::Instant;

    /// Whether the counters run.
    pub fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| value_on(std::env::var("VICTORY_MAP_COUNT").ok().as_deref()))
    }

    /// [`on`] for a setting: only `1`.
    pub fn value_on(value: Option<&str>) -> bool {
        value.is_some_and(|v| v.trim() == "1")
    }

    /// Who asked for a change of mapping.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Caller {
        /// The title's map call with pages (each page of a scatter call that
        /// goes page by page too).
        TitleMap = 0,
        /// The same without pages.
        TitleUnmap = 1,
        /// The queue's page-mapping copies, on the thread that submits them.
        Copies = 2,
        /// Nobody's call: a thread of ours (the mapping threads, views of
        /// our own).
        Ours = 3,
    }

    /// Host calls counted by kind.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Kernel {
        Unmap = 0,
        Split = 1,
        Map = 2,
        Commit = 3,
        /// Placeholders joined into one for a view of several pages.
        Join = 4,
    }

    /// What is counted by caller, beside the host calls.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Field {
        Calls = 0,
        Pages = 1,
        /// Nanoseconds from entry to return.
        Spent = 2,
        /// Nanoseconds waiting for the runtime's lock on the views.
        Lock = 3,
        /// Nanoseconds waiting for reads of title memory in place to end.
        Readers = 4,
        /// Nanoseconds in what the GPU side notes of a change.
        Upkeep = 5,
        /// Host calls of 1 ms or more.
        Slow = 6,
        /// Pages asked for that the address showed already.
        Same = 7,
        /// Pages that are the next page of the same section at the next
        /// address, within one range asked for.
        Continued = 8,
        /// Views of which a part stayed and was mapped again.
        Cut = 9,
        /// Pages left as they were (`VICTORY_MAP_SAME=1`).
        Left = 10,
        /// Pages mapped in a view of more than one (`VICTORY_MAP_RUNS=1`).
        Joined = 11,
    }

    const FIELDS: usize = 12;
    const KERNELS: usize = 5;
    const CALLERS: usize = 4;

    static COUNTS: [[AtomicU64; FIELDS]; CALLERS] = [const { [const { AtomicU64::new(0) }; FIELDS] }; CALLERS];
    /// Calls and nanoseconds.
    static KERNEL: [[[AtomicU64; 2]; KERNELS]; CALLERS] =
        [const { [const { [const { AtomicU64::new(0) }; 2] }; KERNELS] }; CALLERS];
    /// By thread: its id, then nanoseconds in calls, waiting for the lock
    /// on the views, and in host calls.
    static THREADS: std::sync::Mutex<Vec<(u32, [u64; 3])>> = std::sync::Mutex::new(Vec::new());

    thread_local! {
        static CALLER: Cell<usize> = const { Cell::new(Caller::Ours as usize) };
        /// Of the call under way on this thread: lock and host nanoseconds.
        static MINE: Cell<[u64; 2]> = const { Cell::new([0; 2]) };
    }

    /// A mapping call under way on this thread; counted when dropped.
    pub struct Call {
        started: Instant,
        before: usize,
    }

    /// Begins a call of `caller` over `pages` pages; `None` while the
    /// counters are off.
    pub fn call(caller: Caller, pages: usize) -> Option<Call> {
        if !on() {
            return None;
        }
        let before = CALLER.replace(caller as usize);
        MINE.set([0; 2]);
        let at = &COUNTS[caller as usize];
        at[Field::Calls as usize].fetch_add(1, Relaxed);
        at[Field::Pages as usize].fetch_add(pages as u64, Relaxed);
        Some(Call { started: Instant::now(), before })
    }

    impl Drop for Call {
        fn drop(&mut self) {
            let ns = self.started.elapsed().as_nanos() as u64;
            let caller = CALLER.replace(self.before);
            COUNTS[caller][Field::Spent as usize].fetch_add(ns, Relaxed);
            let [lock, kernel] = MINE.replace([0; 2]);
            thread_add([ns, lock, kernel]);
        }
    }

    fn thread_add(add: [u64; 3]) {
        // SAFETY: a plain query of the calling thread.
        let id = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
        let Ok(mut threads) = THREADS.lock() else {
            return;
        };
        let at = match threads.iter().position(|t| t.0 == id) {
            Some(at) => at,
            None => {
                threads.push((id, [0; 3]));
                threads.len() - 1
            }
        };
        for (sum, add) in threads[at].1.iter_mut().zip(add) {
            *sum += add;
        }
    }

    /// Adds to a field of the caller under way on this thread.
    pub fn add(field: Field, n: u64) {
        if on() && n != 0 {
            COUNTS[CALLER.get()][field as usize].fetch_add(n, Relaxed);
        }
    }

    /// Runs `f`, a wait for a lock of ours or an upkeep step, and adds its
    /// time to `field`.
    pub fn timed<R>(field: Field, f: impl FnOnce() -> R) -> R {
        if !on() {
            return f();
        }
        let started = Instant::now();
        let done = f();
        let ns = started.elapsed().as_nanos() as u64;
        COUNTS[CALLER.get()][field as usize].fetch_add(ns, Relaxed);
        if field == Field::Lock {
            let mut mine = MINE.get();
            mine[0] += ns;
            MINE.set(mine);
        }
        done
    }

    /// Runs `f`, a host call that changes a mapping, counted for the caller
    /// under way on this thread.
    pub fn kernel<R>(kind: Kernel, f: impl FnOnce() -> R) -> R {
        if !on() {
            return f();
        }
        let started = Instant::now();
        let done = f();
        let ns = started.elapsed().as_nanos() as u64;
        let caller = CALLER.get();
        let at = &KERNEL[caller][kind as usize];
        at[0].fetch_add(1, Relaxed);
        at[1].fetch_add(ns, Relaxed);
        if ns >= 1_000_000 {
            COUNTS[caller][Field::Slow as usize].fetch_add(1, Relaxed);
        }
        let mut mine = MINE.get();
        mine[1] += ns;
        MINE.set(mine);
        if caller == Caller::Ours as usize {
            // No call of the title's or the queue's ends for it.
            thread_add([ns, 0, ns]);
        }
        done
    }

    /// Frames presented since the last line.
    static FRAMES: AtomicU64 = AtomicU64::new(0);

    /// The frames of the last second (from the meter, each second).
    pub fn frames(n: u64) {
        if on() {
            FRAMES.fetch_add(n, Relaxed);
        }
    }

    /// What the runtime holds mapped now ([`held_by`]).
    #[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
    pub struct Held {
        /// Views the title sees, their bytes, and those of one page.
        pub views: u64,
        pub view_bytes: u64,
        pub single: u64,
        /// Placeholders kept as some view's own, and their bytes.
        pub holes: u64,
        pub hole_bytes: u64,
        /// Sections of physical pages alive, their bytes, and of those the
        /// bytes committed.
        pub sections: u64,
        pub section_bytes: u64,
        pub committed: u64,
    }

    /// Asks the runtime what it holds; `None` if its tables are busy.
    static HELD: std::sync::OnceLock<fn() -> Option<Held>> = std::sync::OnceLock::new();

    /// The runtime gives the way to ask what it holds.
    pub fn held_by(ask: fn() -> Option<Held>) {
        let _ = HELD.set(ask);
    }

    /// The per-frame part of the line: over `frames` frames, the host's map
    /// and unmap calls, the time in host calls of every kind, and the time
    /// waiting for the lock on the views.
    pub fn per_frame(frames: u64, maps_and_unmaps: u64, host_ns: u64, lock_ns: u64) -> String {
        if frames == 0 {
            return " per frame: no frame in that time;".to_string();
        }
        let f = frames as f64;
        format!(
            " per frame, over {frames} frames: {:.1} host map and unmap calls, {:.2} ms in host calls (where the \
             address-space lock is waited for), {:.2} ms waiting for the lock on the views;",
            maps_and_unmaps as f64 / f,
            host_ns as f64 / 1e6 / f,
            lock_ns as f64 / 1e6 / f
        )
    }

    /// The part of the line that says what is held.
    pub fn held_text(held: Option<Held>) -> String {
        let Some(h) = held else {
            return " held now: not read (the tables were busy);".to_string();
        };
        let mb = |bytes: u64| bytes as f64 / (1 << 20) as f64;
        format!(
            " held now: {} views of {:.1} MB of address space mapped ({} of one page), {} placeholders of {:.1} MB kept \
             reserved for views that went, {} sections of {:.1} MB of which {:.1} MB committed;",
            h.views,
            mb(h.view_bytes),
            h.single,
            h.holes,
            mb(h.hole_bytes),
            h.sections,
            mb(h.section_bytes),
            mb(h.committed)
        )
    }

    /// The line for the counts since the last one, and the counts cleared;
    /// `None` if nothing was counted. `settings` names the switches that
    /// cut host calls, as they are.
    pub fn take_line(settings: &str) -> Option<String> {
        let frames = FRAMES.swap(0, Relaxed);
        let counts: Vec<Vec<u64>> = COUNTS.iter().map(|c| c.iter().map(|n| n.swap(0, Relaxed)).collect()).collect();
        let kernel: Vec<Vec<[u64; 2]>> =
            KERNEL.iter().map(|c| c.iter().map(|k| [k[0].swap(0, Relaxed), k[1].swap(0, Relaxed)]).collect()).collect();
        let mut threads = THREADS.lock().map(|mut t| std::mem::take(&mut *t)).unwrap_or_default();
        let any = counts.iter().flatten().any(|&n| n != 0) || kernel.iter().flatten().any(|k| k[0] != 0);
        if !any {
            return None;
        }
        let ms = |ns: u64| ns as f64 / 1e6;
        let names = ["the title's maps", "the title's unmaps", "page-mapping copies", "threads of ours"];
        let kinds = ["unmap", "split", "map", "commit", "join"];
        let mut line = format!("page mapping, since the last line (VICTORY_MAP_COUNT=1; {settings}):");
        let (mut calls, mut kernel_ns) = (0, 0);
        for (i, name) in names.iter().enumerate() {
            let (c, k) = (&counts[i], &kernel[i]);
            if c.iter().all(|&n| n == 0) && k.iter().all(|k| k[0] == 0) {
                continue;
            }
            let host: Vec<String> =
                kinds.iter().zip(k).map(|(kind, k)| format!("{kind} {} {:.1} ms", k[0], ms(k[1]))).collect();
            calls += k.iter().map(|k| k[0]).sum::<u64>();
            kernel_ns += k.iter().map(|k| k[1]).sum::<u64>();
            let f = |field: Field| c[field as usize];
            line += &format!(
                " {name}: {} calls, {} pages, {:.1} ms (waiting for the lock on the views {:.1} ms, for reads in place {:.1} ms, \
                 upkeep {:.1} ms; host calls: {}; {} of 1 ms or more); {} pages the address showed already ({} left as they \
                 were), {} pages continuing a run ({} mapped in views of several), {} views cut;",
                f(Field::Calls),
                f(Field::Pages),
                ms(f(Field::Spent)),
                ms(f(Field::Lock)),
                ms(f(Field::Readers)),
                ms(f(Field::Upkeep)),
                host.join(", "),
                f(Field::Slow),
                f(Field::Same),
                f(Field::Left),
                f(Field::Continued),
                f(Field::Joined),
                f(Field::Cut),
            );
        }
        let of = |kind: Kernel| kernel.iter().map(|k| k[kind as usize][0]).sum::<u64>();
        let lock = counts.iter().map(|c| c[Field::Lock as usize]).sum::<u64>();
        line += &per_frame(frames, of(Kernel::Map) + of(Kernel::Unmap), kernel_ns, lock);
        line += &held_text(HELD.get().and_then(|ask| ask()));
        line += &format!(" host calls in all: {calls} in {:.1} ms; by thread:", ms(kernel_ns));
        threads.sort_by_key(|t| std::cmp::Reverse(t.1[0]));
        for (id, [spent, lock, host]) in threads.into_iter().take(4) {
            line += &format!(" {id} {:.1} ms (lock {:.1}, host calls {:.1});", ms(spent), ms(lock), ms(host));
        }
        Some(line)
    }

    /// The switches that cut host calls, as the runtime read them, for the
    /// line (set once by the runtime).
    static SETTINGS: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    /// The runtime says how its switches stand.
    pub fn settings(text: String) {
        let _ = SETTINGS.set(text);
    }

    /// Logs the line; called with the other 5 s lines.
    pub(crate) fn log_line() {
        if !on() {
            return;
        }
        if let Some(line) = take_line(SETTINGS.get().map_or("", |s| s.as_str())) {
            tracing::info!("{line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_page_run_is_the_next_index_of_the_same_section() {
        assert_eq!(follows(0, 0, 0, 0x1000, 1, 0), (false, false));
        assert_eq!(follows(0x1000, 7, 3, 0x1000, 7, 4), (true, false));
        assert_eq!(follows(0x1000, 7, 3, 0x1000, 7, 3), (false, true));
        assert_eq!(follows(0x1000, 7, 3, 0x1000, 8, 4), (false, false));
        assert_eq!(follows(0x2000, 7, 3, 0x1000, 7, 4), (false, false));
    }

    #[test]
    fn split_late_is_off_unless_asked() {
        assert_eq!(split_mode(None), 0);
        assert_eq!(split_mode(Some("")), 0);
        assert_eq!(split_mode(Some("0")), 0);
        assert_eq!(split_mode(Some("late")), 1);
        assert_eq!(split_mode(Some("alt")), 2);
    }

    #[test]
    fn the_mapping_counters_are_off_unless_asked() {
        assert!(!count::value_on(None));
        assert!(!count::value_on(Some("")));
        assert!(!count::value_on(Some("0")));
        assert!(!count::value_on(Some("on")));
        assert!(count::value_on(Some("1")));
        // (The tests run without the switch.) Off: nothing begins, and the
        // work is done as it is.
        if !count::on() {
            assert!(count::call(count::Caller::TitleMap, 3).is_none());
            assert_eq!(count::kernel(count::Kernel::Map, || 7), 7);
            assert_eq!(count::timed(count::Field::Lock, || 8), 8);
            count::add(count::Field::Same, 1);
            assert_eq!(count::take_line(""), None);
        }
    }

    #[test]
    fn the_mapping_line_gives_calls_and_waits_by_the_frame_and_what_is_held() {
        assert_eq!(count::per_frame(0, 5, 5, 5), " per frame: no frame in that time;");
        let line = count::per_frame(50, 925, 36_000_000, 4_000_000);
        assert!(line.contains("over 50 frames: 18.5 host map and unmap calls, 0.72 ms in host calls"), "{line}");
        assert!(line.contains("0.08 ms waiting for the lock on the views"), "{line}");
        assert!(count::held_text(None).contains("not read"));
        let held = count::Held {
            views: 3,
            view_bytes: 6 << 20,
            single: 2,
            holes: 4,
            hole_bytes: 1 << 19,
            sections: 1,
            section_bytes: 8 << 20,
            committed: 5 << 20,
        };
        let line = count::held_text(Some(held));
        assert!(line.contains("3 views of 6.0 MB of address space mapped (2 of one page)"), "{line}");
        assert!(line.contains("4 placeholders of 0.5 MB kept"), "{line}");
        assert!(line.contains("1 sections of 8.0 MB of which 5.0 MB committed"), "{line}");
    }
}
