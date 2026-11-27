//! Diagnostics: the process's memory by owner, from a walk of its address
//! space. `VICTORY_MEMORY_ACCOUNT=1`: once a minute; a larger number: every
//! that many seconds (10 at least); unset or `0`: nothing is kept and
//! nothing is walked.
//!
//! The memory lines name what this crate counts (mirrors, staging, scratch,
//! our heap); in a session of 20.9 GB private that left 9.8 GB with no
//! name. The memory map (`VICTORY_MEMORY_MAP`) sums by kind of region, and
//! says "write-combined 7,103 MB (1,626 MB in memory)" without saying whose
//! the buffers are, or whether the card's memory is counted there again.
//! Here every committed region goes to one owner, each byte once:
//!
//! - the title: private memory at console addresses (below 1 TiB), what
//!   the runtime reserved for it at any address, and what the title took
//!   with its own plain `VirtualAlloc` import ([`title_plain`]: that
//!   import is the host's own function, at host addresses, and nobody was
//!   told of it; it is summed by size in a line of its own);
//! - buffers of ours that are mapped: by the pointer `Map` gave
//!   ([`mapped`]), under the name given them ([`named`]);
//! - ranges of ours told by address ([`range`]: descriptor heaps);
//! - our own heap: the process heaps' segments and the big blocks the
//!   allocator made ([`block`]);
//! - write-combined memory that is not ours (the driver's mappings: of the
//!   card's memory, if the sums beside it say so);
//! - the rest ("driver and other"), with its biggest groups by size.
//!
//! Mapped memory (not private) is summed beside it by file, with how much
//! of it is shown more than once (and so counted more than once in the
//! working set).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// Console addresses lie below this.
const HOST: usize = 1 << 40;
/// A segment of a process heap, reserved parts included.
const SEGMENT: usize = 0xfd_0000;
/// One page asked about stands for this much.
const STEP: usize = 64 << 10;
/// Blocks of our allocator from this size have an allocation of their own.
const BLOCK: usize = 500 << 10;

/// What `VICTORY_MEMORY_ACCOUNT` says: seconds between walks, 0 for none.
pub(crate) fn period(value: Option<&str>) -> u64 {
    match value.map(str::trim) {
        None | Some("") | Some("0") => 0,
        Some(v) => match v.parse::<u64>() {
            Ok(1) | Err(_) => 60,
            Ok(n) => n.max(10),
        },
    }
}

/// Seconds between walks (0: the account is off).
fn every() -> u64 {
    static EVERY: OnceLock<u64> = OnceLock::new();
    *EVERY.get_or_init(|| {
        let every = period(std::env::var("VICTORY_MEMORY_ACCOUNT").ok().as_deref());
        BLOCKS_ON.store(every > 0, Relaxed);
        every
    })
}

/// Whether the account is kept.
pub(crate) fn on() -> bool {
    every() > 0
}

/// Something of ours at an address.
struct Held {
    at: usize,
    size: u64,
    kind: &'static str,
    name: Option<Box<str>>,
    /// A range inside an allocation (not a mapping of its own).
    inside: bool,
    /// The title's own memory, as the runtime told of it.
    title: bool,
    /// The title's own by its plain `VirtualAlloc` import ([`title_plain`]).
    plain: bool,
    seq: u64,
}

/// By the object's address (a range: by its own).
static HELD: Mutex<Option<(HashMap<usize, Held>, u64)>> = Mutex::new(None);

fn note(key: usize, at: usize, size: u64, kind: &'static str, inside: bool, title: bool) {
    note_as(key, at, size, kind, inside, title, false);
}

fn note_as(key: usize, at: usize, size: u64, kind: &'static str, inside: bool, title: bool, plain: bool) {
    if at == 0 || !on() {
        return;
    }
    let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    let (map, seq) = held.get_or_insert_with(Default::default);
    *seq += 1;
    map.insert(key, Held { at, size, kind, name: None, inside, title, plain, seq: *seq });
}

/// `object`, a buffer of ours of `size` bytes, is mapped at `at`.
pub(crate) fn mapped(object: usize, at: *const u8, size: u64, kind: &'static str) {
    note(object, at as usize, size, kind, false, false);
}

/// `size` bytes at `at` are ours (memory the host gave an address of).
pub(crate) fn range(at: usize, size: u64, kind: &'static str) {
    note(at, at, size, kind, true, false);
}

/// The title reserved `size` bytes at `at` (the runtime tells of each).
pub(crate) fn title(at: usize, size: u64) {
    note(at, at, size, "the title's own, told by the runtime", true, true);
}

/// The title made an allocation of `size` bytes at `at` with its plain
/// `VirtualAlloc` import (the host's own function: the runtime answers
/// only for `XMemVirtualAlloc`). It counts as the title's only while the
/// allocation at `at` is still one of that size ([`fits`]): an address
/// used again by somebody else is not taken for the title's.
pub(crate) fn title_plain(at: usize, size: u64) {
    note_as(at, at, size, "the title's own, by its plain VirtualAlloc", false, true, true);
}

/// The title released its plain allocation at `at`.
pub(crate) fn title_plain_gone(at: usize) {
    if at == 0 || !on() {
        return;
    }
    let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((map, _)) = held.as_mut()
        && map.get(&at).is_some_and(|h| h.plain)
    {
        map.remove(&at);
    }
}

/// How many plain allocations of the title's are noted now (tests).
#[cfg(test)]
fn plain_noted() -> usize {
    HELD.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, |(map, _)| map.values().filter(|h| h.plain).count())
}

/// A name with its numbers taken out, so that many of a kind are one.
pub(crate) fn plain(text: &str) -> String {
    let mut out = String::with_capacity(text.len().min(48));
    let mut digits = false;
    for c in text.chars() {
        if out.len() >= 48 {
            break;
        }
        if c.is_ascii_digit() {
            if !digits {
                out.push('#');
            }
            digits = true;
        } else {
            out.push(c);
            digits = false;
        }
    }
    out
}

/// `object` was named `text`.
pub(crate) fn named(object: usize, text: &str) {
    if !on() {
        return;
    }
    let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = held.as_mut().and_then(|(map, _)| map.get_mut(&object)) {
        h.name = Some(plain(text).into_boxed_str());
    }
}

/// Whether [`block`] keeps anything (set with the switch: the allocator
/// must not read the environment).
static BLOCKS_ON: AtomicBool = AtomicBool::new(false);
static BLOCKS: [AtomicUsize; 4096] = [const { AtomicUsize::new(0) }; 4096];

/// Our allocator made a block of `size` bytes at `at`, or released it.
/// (Called inside the allocator: nothing here allocates.)
pub(crate) fn block(at: *mut u8, size: usize, made: bool) {
    if size < BLOCK || !BLOCKS_ON.load(Relaxed) {
        return;
    }
    let at = at as usize;
    for place in &BLOCKS {
        if made {
            if place.compare_exchange(0, at, Relaxed, Relaxed).is_ok() {
                return;
            }
        } else if place.load(Relaxed) == at && place.compare_exchange(at, 0, Relaxed, Relaxed).is_ok() {
            return;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Type {
    Private,
    Mapped,
    Image,
}

/// A region of the address space, as the system tells of it.
#[derive(Clone, Debug)]
pub(crate) struct Region {
    /// Its allocation's start, its own start and bytes.
    pub base: usize,
    pub at: usize,
    pub len: usize,
    pub committed: bool,
    pub kind: Type,
    pub write_combined: bool,
    /// Bytes of it in the working set (from the pages asked about), and of
    /// those the part that is another showing of a page counted already.
    pub resident: u64,
    pub again: u64,
    /// The file it maps (an index; 0: none).
    pub file: usize,
}

/// Something of ours for the walk: where, how long, its label, whether it
/// is a range inside an allocation.
pub(crate) struct Holding {
    pub at: usize,
    pub size: u64,
    pub label: String,
    pub inside: bool,
    /// The title's own.
    pub title: bool,
    /// The title's own by its plain `VirtualAlloc` import.
    pub plain: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Sum {
    /// Allocations, bytes committed, bytes in the working set.
    pub n: u64,
    pub committed: u64,
    pub resident: u64,
}

impl Sum {
    fn add(&mut self, r: &Region, first: bool) {
        self.n += u64::from(first);
        self.committed += r.len as u64;
        self.resident += r.resident;
    }
}

/// The owners of private memory.
pub(crate) const OWNERS: [&str; 6] = [
    "the title's own (console addresses, or told of by the runtime)",
    "buffers of ours that are mapped",
    "ranges of ours (descriptor heaps)",
    "our heap",
    "write-combined, not ours",
    "driver and other",
];
const TITLE: usize = 0;
const BUFFERS: usize = 1;
const RANGES: usize = 2;
const HEAP: usize = 3;
const COMBINED: usize = 4;
const OTHER: usize = 5;

#[derive(Debug, Default)]
pub(crate) struct Account {
    /// Private memory by owner ([`OWNERS`]).
    pub owners: [Sum; 6],
    /// Ours by label (buffers and ranges).
    pub ours: BTreeMap<String, Sum>,
    /// What has no owner, by (write-combined, the allocation's whole
    /// size): the sum and the first allocation's address.
    pub groups: BTreeMap<(bool, usize), (Sum, usize)>,
    /// Mapped memory: views at console addresses, at host addresses by
    /// file, programs and libraries.
    pub views: Sum,
    pub files: BTreeMap<usize, Sum>,
    pub images: Sum,
    /// Of the title's own, what the runtime told of (at any address).
    pub above: Sum,
    /// Of the title's own, what it took with its plain `VirtualAlloc`
    /// import, by the allocation's whole size: the sum and the first
    /// allocation's address.
    pub plain: BTreeMap<usize, (Sum, usize)>,
    /// Working-set bytes that are a page shown again.
    pub again: u64,
    /// Holdings that met no allocation (by index).
    pub unmet: Vec<usize>,
    /// Of those, mapped buffers of ours whose pointer lies in an allocation
    /// of another size, by that allocation's whole size: how many, and
    /// their own bytes. A buffer the host cut out of a bigger mapping of
    /// its own is one (the allocation then stands without an owner), and
    /// so is the address of a buffer released since.
    pub within: BTreeMap<usize, (u64, u64)>,
}

impl Account {
    pub(crate) fn private(&self) -> Sum {
        let mut all = Sum::default();
        for o in &self.owners {
            all.n += o.n;
            all.committed += o.committed;
            all.resident += o.resident;
        }
        all
    }

    /// Per cent of `of` (committed, resident) that has an owner by name:
    /// everything but "driver and other" and "write-combined, not ours".
    pub(crate) fn named(&self) -> [u64; 2] {
        let all = self.private();
        let rest = [COMBINED, OTHER].map(|o| self.owners[o]);
        let pct = |all: u64, rest: u64| if all == 0 { 100 } else { (all - rest.min(all)) * 100 / all };
        [
            pct(all.committed, rest[0].committed + rest[1].committed),
            pct(all.resident, rest[0].resident + rest[1].resident),
        ]
    }
}

/// Whether a buffer of `size` bytes mapped at an allocation's start is
/// that allocation, `whole` bytes long (mappings are whole pages, or whole
/// 64 KiB): an address used again by something else of another size is
/// not taken for ours.
pub(crate) fn fits(size: u64, whole: usize) -> bool {
    let whole = whole as u64;
    whole >= size && whole - size < 128 << 10
}

/// Every committed region to one owner. `blocks`: where our allocator's
/// big blocks are; `heaps`: where the process heaps begin.
pub(crate) fn tally(regions: &[Region], held: &[Holding], blocks: &[usize], heaps: &[usize]) -> Account {
    // Each allocation's whole size, reserved parts included.
    let mut whole: BTreeMap<usize, usize> = BTreeMap::new();
    for r in regions {
        *whole.entry(r.base).or_default() += r.len;
    }
    // Ours by allocation: a mapping is its allocation; a range lies in one.
    let mut ours: HashMap<usize, usize> = HashMap::new();
    let mut unmet = Vec::new();
    let mut within: BTreeMap<usize, (u64, u64)> = BTreeMap::new();
    for (i, h) in held.iter().enumerate() {
        let allocation = whole.range(..=h.at).next_back().map(|(&base, &size)| (base, size));
        let met = match allocation {
            Some((base, size)) if h.inside => (h.at - base < size).then_some(base),
            Some((base, size)) => (base == h.at && fits(h.size, size)).then_some(base),
            None => None,
        };
        match met {
            Some(base) => {
                ours.entry(base).or_insert(i);
            }
            None => {
                if let Some((base, size)) = allocation
                    && !h.inside
                    && !h.title
                    && h.at - base < size
                {
                    let w = within.entry(size).or_default();
                    w.0 += 1;
                    w.1 += h.size;
                }
                unmet.push(i);
            }
        }
    }
    let blocks: HashSet<usize> = blocks.iter().map(|b| b & !0xffff).collect();
    let heaps: HashSet<usize> = heaps.iter().copied().collect();
    let mut account = Account { unmet, within, ..Default::default() };
    let mut seen: HashSet<usize> = HashSet::new();
    for r in regions.iter().filter(|r| r.committed) {
        let first = seen.insert(r.base);
        account.again += r.again;
        match r.kind {
            Type::Image => account.images.add(r, first),
            Type::Mapped if r.base < HOST => account.views.add(r, first),
            Type::Mapped => account.files.entry(r.file).or_default().add(r, first),
            Type::Private => {
                let size = whole.get(&r.base).copied().unwrap_or(0);
                let told = ours.get(&r.base).copied();
                let owner = if told.is_some_and(|i| held[i].title) {
                    if told.is_some_and(|i| held[i].plain) {
                        account.plain.entry(size).or_insert((Sum::default(), r.base)).0.add(r, first);
                    } else {
                        account.above.add(r, first && r.base >= HOST);
                    }
                    TITLE
                } else if r.base < HOST {
                    TITLE
                } else if let Some(i) = told {
                    account.ours.entry(held[i].label.clone()).or_default().add(r, first);
                    if held[i].inside { RANGES } else { BUFFERS }
                } else if size == SEGMENT || heaps.contains(&r.base) || blocks.contains(&r.base) {
                    HEAP
                } else {
                    let group = account.groups.entry((r.write_combined, size)).or_insert((Sum::default(), r.base));
                    group.0.add(r, first);
                    if r.write_combined { COMBINED } else { OTHER }
                };
                account.owners[owner].add(r, first);
            }
        }
    }
    account
}

fn mb(bytes: u64) -> u64 {
    bytes >> 20
}

/// The account's first line: private memory by owner.
pub(crate) fn owners_line(a: &Account, regions: usize, ms: u128) -> String {
    let all = a.private();
    let named = a.named();
    let parts: Vec<String> = (0..OWNERS.len())
        .filter(|&o| a.owners[o].committed > 0)
        .map(|o| {
            let s = a.owners[o];
            format!("{} {} MB ({} MB in memory, {} allocations)", OWNERS[o], mb(s.committed), mb(s.resident), s.n)
        })
        .collect();
    format!(
        "memory account (VICTORY_MEMORY_ACCOUNT; {regions} regions, {ms} ms): private {} MB committed, {} MB of it \
         in memory; with an owner by name: {} % of the committed, {} % of what is in memory; {}; of the title's own \
         the runtime told of {} MB ({} MB in memory)",
        mb(all.committed),
        mb(all.resident),
        named[0],
        named[1],
        parts.join("; "),
        mb(a.above.committed),
        mb(a.above.resident),
    )
}

/// The biggest `most` of `sums` by committed bytes, as text.
fn biggest<K>(sums: impl Iterator<Item = (K, Sum)>, most: usize, text: impl Fn(&K, &Sum) -> String) -> (usize, String) {
    let mut all: Vec<(K, Sum)> = sums.collect();
    all.sort_by_key(|(_, s)| std::cmp::Reverse(s.committed));
    let shown: Vec<String> = all.iter().take(most).map(|(k, s)| text(k, s)).collect();
    (all.len(), if shown.is_empty() { "none".into() } else { shown.join("; ") })
}

/// Ours by label.
pub(crate) fn ours_line(a: &Account) -> String {
    let (n, text) = biggest(a.ours.iter().map(|(k, s)| (k.clone(), *s)), 16, |k, s| {
        format!("{} x \"{k}\": {} MB ({} MB in memory)", s.n, mb(s.committed), mb(s.resident))
    });
    let mut within: Vec<(usize, (u64, u64))> = a.within.iter().map(|(k, v)| (*k, *v)).collect();
    within.sort_by_key(|(_, (_, bytes))| std::cmp::Reverse(*bytes));
    let within: Vec<String> = within
        .iter()
        .take(8)
        .map(|(size, (n, bytes))| format!("{n} ({} MB) in allocations of {:.2} MB", mb(*bytes), *size as f64 / (1 << 20) as f64))
        .collect();
    format!(
        "memory account, ours by name, the biggest of {n}: {text}; told of and met by no allocation: {}; of those, mapped          buffers told of since the walk before whose pointer lies in an allocation of another size (cut out of a bigger          mapping of the host's, which then stands without an owner; or released since): {}",
        a.unmet.len(),
        if within.is_empty() { "none".to_owned() } else { within.join(", ") },
    )
}

/// What has no owner, by size.
pub(crate) fn groups_line(a: &Account) -> String {
    let (n, text) = biggest(a.groups.iter().map(|(k, (s, at))| ((*k, *at), *s)), 16, |k, s| {
        let ((combined, size), at) = *k;
        format!(
            "{} of {:.2} MB{}: {} MB ({} MB in memory; the first at {at:#x})",
            s.n,
            size as f64 / (1 << 20) as f64,
            if combined { " write-combined" } else { "" },
            mb(s.committed),
            mb(s.resident),
        )
    });
    format!("memory account, without an owner, by the allocation's size, the biggest of {n} groups: {text}")
}

/// What the title took with its plain `VirtualAlloc` import, by size: the
/// groups that had no owner before (four of 1,024 MB and some forty of
/// 32 MB in every session so far) are the title's if they stand here, and
/// the driver's or the host runtime's if they still stand in
/// [`groups_line`].
pub(crate) fn plain_line(a: &Account) -> String {
    let mut all = Sum::default();
    for (s, _) in a.plain.values() {
        all.n += s.n;
        all.committed += s.committed;
        all.resident += s.resident;
    }
    let (n, text) = biggest(a.plain.iter().map(|(k, (s, at))| ((*k, *at), *s)), 16, |k, s| {
        let (size, at) = *k;
        format!(
            "{} of {:.2} MB: {} MB ({} MB in memory; the first at {at:#x})",
            s.n,
            size as f64 / (1 << 20) as f64,
            mb(s.committed),
            mb(s.resident),
        )
    });
    format!(
        "memory account, the title's own by its plain VirtualAlloc import (counted with the title's own above): {} \
         allocations, {} MB committed, {} MB in memory; by the allocation's size, the biggest of {n} groups: {text}",
        all.n,
        mb(all.committed),
        mb(all.resident),
    )
}

/// Mapped memory, which is not private: by file.
pub(crate) fn mapped_line(a: &Account, names: &[String]) -> String {
    let (n, text) = biggest(a.files.iter().map(|(k, s)| (*k, *s)), 10, |k, s| {
        let name = names.get(*k).map(String::as_str).filter(|n| !n.is_empty()).unwrap_or("no file (a section)");
        format!("{name}: {} MB ({} MB in memory, {} views)", mb(s.committed), mb(s.resident), s.n)
    });
    format!(
        "memory account, mapped (not private): views of the title's pages {} MB ({} MB in memory); programs and \
         libraries {} MB ({} MB in memory); at host addresses by file, the biggest of {n}: {text}; of all that is in \
         memory {} MB is a page shown again (counted in the working set once for each showing)",
        mb(a.views.committed),
        mb(a.views.resident),
        mb(a.images.committed),
        mb(a.images.resident),
        mb(a.again),
    )
}

/// What a working-set answer says of a page: bytes in memory, and of
/// those the bytes that are a showing beyond the first (a page shown `n`
/// times is in the working set `n` times: `n - 1` of them are no memory).
pub(crate) fn page(answer: usize, stands: u64) -> (u64, u64) {
    if answer & 1 == 0 {
        return (0, 0);
    }
    let shown = (answer >> 1 & 7) as u64;
    let shared = answer >> 15 & 1 != 0;
    let again = if shared && shown >= 2 { stands * (shown - 1) / shown } else { 0 };
    (stands, again)
}

#[cfg(windows)]
mod walk {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        GetProcessHeaps, MEM_COMMIT, MEM_FREE, MEM_IMAGE, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_WRITECOMBINE,
        VirtualQuery,
    };
    use windows_sys::Win32::System::ProcessStatus::{K32GetMappedFileNameW, K32QueryWorkingSetEx};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    /// Asks which of the pages in `asked` are in memory and adds what each
    /// stands for (region, bytes) to its region.
    fn in_memory(asked: &mut Vec<[usize; 2]>, stands: &mut Vec<(usize, u64)>, regions: &mut [Region]) {
        // SAFETY: our own process; `asked` holds that many entries of two
        // words each, as the call fills them in.
        let ok =
            unsafe { K32QueryWorkingSetEx(GetCurrentProcess(), asked.as_mut_ptr().cast(), (asked.len() * 16) as u32) }
                != 0;
        if ok {
            for (a, &(region, bytes)) in asked.iter().zip(stands.iter()) {
                let (resident, again) = page(a[1], bytes);
                regions[region].resident += resident;
                regions[region].again += again;
            }
        }
        asked.clear();
        stands.clear();
    }

    /// The file mapped at `at`: its name without the folders.
    fn file(at: usize) -> String {
        let mut name = [0u16; 520];
        // SAFETY: our own process and a buffer of that many characters.
        let n = unsafe { K32GetMappedFileNameW(GetCurrentProcess(), at as *const _, name.as_mut_ptr(), name.len() as u32) };
        let text = String::from_utf16_lossy(&name[..(n as usize).min(name.len())]);
        text.rsplit('\\').next().unwrap_or("").to_string()
    }

    /// The address space: every region that is not free, with what of it
    /// is in memory, and the files' names by index.
    pub(super) fn regions() -> (Vec<Region>, Vec<String>) {
        let mut regions: Vec<Region> = Vec::new();
        let mut names: Vec<String> = vec![String::new()];
        let mut by_name: HashMap<String, usize> = HashMap::new();
        let mut by_base: HashMap<usize, usize> = HashMap::new();
        let mut at = 0usize;
        loop {
            // SAFETY: plain data filled in by the query.
            let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
                break;
            }
            if m.State != MEM_FREE {
                let base = m.AllocationBase as usize;
                let kind = match m.Type {
                    MEM_IMAGE => Type::Image,
                    MEM_MAPPED => Type::Mapped,
                    _ => Type::Private,
                };
                let committed = m.State == MEM_COMMIT;
                let file = if kind == Type::Mapped && base >= HOST && committed {
                    *by_base.entry(base).or_insert_with(|| {
                        let name = file(base);
                        if name.is_empty() {
                            return 0;
                        }
                        *by_name.entry(name.clone()).or_insert_with(|| {
                            names.push(name);
                            names.len() - 1
                        })
                    })
                } else {
                    0
                };
                regions.push(Region {
                    base,
                    at: m.BaseAddress as usize,
                    len: m.RegionSize,
                    committed,
                    kind,
                    write_combined: m.Protect & PAGE_WRITECOMBINE != 0,
                    resident: 0,
                    again: 0,
                    file,
                });
            }
            let next = (m.BaseAddress as usize).saturating_add(m.RegionSize);
            if next <= at {
                break;
            }
            at = next;
        }
        let mut asked: Vec<[usize; 2]> = Vec::with_capacity(4096);
        let mut stands: Vec<(usize, u64)> = Vec::with_capacity(4096);
        for i in 0..regions.len() {
            if !regions[i].committed {
                continue;
            }
            let (start, len) = (regions[i].at, regions[i].len);
            let mut offset = 0;
            while offset < len {
                asked.push([start + offset, 0]);
                stands.push((i, STEP.min(len - offset) as u64));
                if asked.len() == 4096 {
                    in_memory(&mut asked, &mut stands, &mut regions);
                }
                offset += STEP;
            }
        }
        if !asked.is_empty() {
            in_memory(&mut asked, &mut stands, &mut regions);
        }
        (regions, names)
    }

    /// Where the process heaps begin.
    pub(super) fn heaps() -> Vec<usize> {
        let mut heaps = [std::ptr::null_mut(); 256];
        // SAFETY: a buffer of that many handles.
        let n = unsafe { GetProcessHeaps(heaps.len() as u32, heaps.as_mut_ptr()) } as usize;
        heaps[..n.min(heaps.len())].iter().map(|&h| h as usize).collect()
    }
}

/// The PC's physical memory in bytes: all of it, and what is available.
#[cfg(windows)]
fn physical() -> (u64, u64) {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    unsafe { GlobalMemoryStatusEx(&mut m) };
    (m.ullTotalPhys, m.ullAvailPhys)
}

/// The line that sets the account against the card's and the PC's own
/// numbers: whether what is committed and not in memory is the card's, and
/// whether memory is used that is in nobody's working set. `card`: budget,
/// in use, -, shared in use; `blocks`: MB of texture and staging blocks;
/// `first`: memory used outside our working set at the first walk.
pub(crate) fn check_line(a: &Account, card: [u64; 4], blocks: [u64; 2], pc: [u64; 3], first: u64) -> String {
    let [total, available, working] = pc;
    let elsewhere = total.saturating_sub(available).saturating_sub(working);
    let combined: u64 = a.groups.iter().filter(|(k, _)| k.0).map(|(_, (s, _))| s.committed - s.resident.min(s.committed)).sum();
    let block: u64 = a
        .groups
        .iter()
        .filter(|(k, _)| k.0 && k.1 == 64 << 20)
        .map(|(_, (s, _))| s.n)
        .sum();
    format!(
        "memory account, against the card and the PC: write-combined memory that is not ours and not in memory {} MB, \
         the card has {} MB in use and {} MB of PC memory shared with it; {} write-combined allocations of 64 MB \
         without an owner, texture and staging blocks alive {} (if these two pairs agree, the card's memory is \
         counted again as private bytes and is no RAM); the PC: {} MB in all, {} MB available, our working set {} MB, \
         used by everything else {} MB ({:+} MB since the first walk: if that grows as our buffers do, memory is \
         held for us outside the working set)",
        mb(combined),
        mb(card[1]),
        mb(card[3]),
        block,
        (blocks[0] + blocks[1]) / 64,
        mb(total),
        mb(available),
        mb(working),
        mb(elsewhere),
        (elsewhere as i64 - first as i64) >> 20,
    )
}

/// Walks and logs: on a thread of its own (the walk asks the system about
/// every region, 200,000 of them, 0.7 s, and holds up other threads' memory
/// calls meanwhile, as the memory map does).
#[cfg(windows)]
fn walk_and_log(working: u64) {
    let started = std::time::Instant::now();
    let (held, seq): (Vec<(usize, Holding)>, u64) = {
        let held = HELD.lock().unwrap_or_else(|e| e.into_inner());
        match held.as_ref() {
            Some((map, seq)) => (
                map.iter()
                    .map(|(&key, h)| {
                        let label = h.name.as_deref().unwrap_or(h.kind).to_string();
                        (key, Holding { at: h.at, size: h.size, label, inside: h.inside, title: h.title, plain: h.plain })
                    })
                    .collect(),
                *seq,
            ),
            None => (Vec::new(), 0),
        }
    };
    let blocks: Vec<usize> = BLOCKS.iter().map(|b| b.load(Relaxed)).filter(|&b| b != 0).collect();
    let (regions, names) = walk::regions();
    let (keys, holdings): (Vec<usize>, Vec<Holding>) = held.into_iter().unzip();
    let account = tally(&regions, &holdings, &blocks, &walk::heaps());
    // What met nothing is gone (its buffer was released): forgotten, unless
    // it was told of after the walk began.
    {
        let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((map, _)) = held.as_mut() {
            for &i in &account.unmet {
                if map.get(&keys[i]).is_some_and(|h| h.seq <= seq && h.at == holdings[i].at) {
                    map.remove(&keys[i]);
                }
            }
        }
    }
    tracing::info!("{}", owners_line(&account, regions.len(), started.elapsed().as_millis()));
    tracing::info!("{}", ours_line(&account));
    tracing::info!("{}", groups_line(&account));
    tracing::info!("{}", plain_line(&account));
    tracing::info!("{}", mapped_line(&account, &names));
    let card = crate::host::host().and_then(|h| h.video_memory()).unwrap_or_default();
    let (total, available) = physical();
    static FIRST: AtomicU64 = AtomicU64::new(u64::MAX);
    let elsewhere = total.saturating_sub(available).saturating_sub(working);
    let _ = FIRST.compare_exchange(u64::MAX, elsewhere, Relaxed, Relaxed);
    tracing::info!(
        "{}",
        check_line(&account, card, crate::pool::memory(), [total, available, working], FIRST.load(Relaxed))
    );
}

/// With every memory line: starts a walk if one is due. `working`: the
/// working set as the memory line has it.
#[cfg(windows)]
pub(crate) fn tick(working: u64) {
    let every = every();
    if every == 0 {
        return;
    }
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    static NEXT: AtomicU64 = AtomicU64::new(0);
    static WALKING: AtomicBool = AtomicBool::new(false);
    let now = EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs();
    if now < NEXT.load(Relaxed) || WALKING.swap(true, Relaxed) {
        return;
    }
    let first = NEXT.swap(now + every, Relaxed) == 0;
    let walk = std::thread::Builder::new().name("victory-memory-account".into()).spawn(move || {
        if first && let Some(line) = crate::host::host().and_then(|h| h.lists_cost()) {
            tracing::info!("{line}");
        }
        walk_and_log(working);
        WALKING.store(false, Relaxed);
    });
    if walk.is_err() {
        WALKING.store(false, Relaxed);
    }
}

/// What batches of lists made alone cost: `grew`, private bytes and
/// working-set bytes each batch of `each` lists added; `back`, private
/// bytes that came back when all were let go.
pub(crate) fn lists_line(grew: &[[i64; 2]], each: u64, back: i64) -> String {
    let per = |i: usize| -> Vec<i64> {
        let mut all: Vec<i64> = grew.iter().map(|g| g[i] / each.max(1) as i64 / 1024).collect();
        all.sort_unstable();
        all
    };
    let (private, working) = (per(0), per(1));
    let middle = |v: &[i64]| if v.is_empty() { 0 } else { v[v.len() / 2] };
    format!(
        "memory account, one host list alone ({} batches of {each} made on a thread that does nothing else; other \
         threads' allocations fall into a batch, so the lowest and the middle one count): private bytes grew by \
         {:?} KiB a list (the middle {} KiB), the working set by {:?} KiB a list (the middle {} KiB); when all were \
         let go {} KiB a list came back",
        grew.len(),
        private,
        middle(&private),
        working,
        middle(&working),
        back / (each.max(1) as i64 * grew.len().max(1) as i64) / 1024,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;
    const UP: usize = 0x200_0000_0000;

    fn region(base: usize, at: usize, len: usize, kind: Type) -> Region {
        Region { base, at, len, committed: true, kind, write_combined: false, resident: len as u64 / 2, again: 0, file: 0 }
    }

    #[test]
    fn the_switch_gives_a_period() {
        assert_eq!(period(None), 0);
        assert_eq!(period(Some("0")), 0);
        assert_eq!(period(Some("")), 0);
        assert_eq!(period(Some("1")), 60);
        assert_eq!(period(Some("on")), 60);
        assert_eq!(period(Some(" 300 ")), 300);
        assert_eq!(period(Some("2")), 10);
    }

    #[test]
    fn names_lose_their_numbers() {
        assert_eq!(plain("victory host list 4711"), "victory host list #");
        assert_eq!(plain("a window of 4096 KiB at 12"), "a window of # KiB at #");
        assert_eq!(plain(&"x".repeat(100)).len(), 48);
    }

    #[test]
    fn a_mapping_is_its_allocation_only_at_its_size() {
        assert!(fits(4 << 20, 4 << 20));
        assert!(fits((4 << 20) - 100, 4 << 20));
        assert!(fits(1, 64 << 10));
        assert!(!fits(4 << 20, 32 << 20), "an address used again by something bigger");
        assert!(!fits(8 << 20, 4 << 20));
    }

    #[test]
    fn every_committed_byte_goes_to_one_owner() {
        let mut combined = region(UP + 100 * MIB, UP + 100 * MIB, 64 * MIB, Type::Private);
        combined.write_combined = true;
        combined.resident = 0;
        let mut reserved = region(UP + 300 * MIB, UP + 301 * MIB, 3 * MIB, Type::Private);
        reserved.committed = false;
        let mut file = region(UP + 600 * MIB, UP + 600 * MIB, 8 * MIB, Type::Mapped);
        file.file = 1;
        file.again = MIB as u64;
        let regions = vec![
            // The title's heap, and a view of its pages.
            region(0x1_0000_0000, 0x1_0000_0000, 16 * MIB, Type::Private),
            region(0x2_0000_0000, 0x2_0000_0000, 2 * MIB, Type::Mapped),
            // A window of ours, in two regions; a buffer the pointer of
            // which now lies in an allocation of another size.
            region(UP, UP, MIB, Type::Private),
            region(UP, UP + MIB, 3 * MIB, Type::Private),
            region(UP + 32 * MIB, UP + 32 * MIB, 32 * MIB, Type::Private),
            // The card's memory as the driver maps it.
            combined,
            // A heap segment, a big block of the allocator, a heap's start.
            region(UP + 200 * MIB, UP + 200 * MIB, 5 * MIB, Type::Private),
            region(UP + 220 * MIB, UP + 220 * MIB, 2 * MIB, Type::Private),
            region(UP + 230 * MIB, UP + 230 * MIB, MIB, Type::Private),
            // A descriptor heap of ours inside an allocation, part reserved.
            region(UP + 300 * MIB, UP + 300 * MIB, MIB, Type::Private),
            reserved,
            // Nobody's.
            region(UP + 400 * MIB, UP + 400 * MIB, 1024 * MIB, Type::Private),
            file,
            region(UP + 700 * MIB, UP + 700 * MIB, 4 * MIB, Type::Image),
            region(UP + 2048 * MIB, UP + 2048 * MIB, 8 * MIB, Type::Private),
        ];
        // (The heap segment's reserved rest makes it a segment's size.)
        let mut regions = regions;
        let mut rest = region(UP + 200 * MIB, UP + 205 * MIB, SEGMENT - 5 * MIB, Type::Private);
        rest.committed = false;
        regions.push(rest);
        let held = vec![
            Holding { at: UP, size: 4 * MIB as u64, label: "a window".into(), inside: false, title: false, plain: false },
            Holding { at: UP + 32 * MIB, size: 4 * MIB as u64, label: "a window".into(), inside: false, title: false, plain: false },
            Holding { at: UP + 300 * MIB + 4096, size: 32 << 10, label: "a descriptor heap".into(), inside: true, title: false, plain: false },
            Holding { at: UP + 900 * MIB, size: 4096, label: "gone".into(), inside: false, title: false, plain: false },
            // A reservation of the title's above the console's addresses.
            Holding { at: UP + 2048 * MIB, size: 8 * MIB as u64, label: "the title's".into(), inside: true, title: true, plain: false },
        ];
        let a = tally(&regions, &held, &[UP + 220 * MIB + 64], &[UP + 230 * MIB]);
        let m = |n: usize| (n * MIB) as u64;
        assert_eq!(a.owners[TITLE], Sum { n: 2, committed: m(24), resident: m(12) });
        assert_eq!(a.above, Sum { n: 1, committed: m(8), resident: m(4) });
        assert_eq!(a.owners[BUFFERS], Sum { n: 1, committed: m(4), resident: m(2) });
        assert_eq!(a.owners[RANGES], Sum { n: 1, committed: m(1), resident: m(1) / 2 });
        assert_eq!(a.owners[HEAP], Sum { n: 3, committed: m(8), resident: m(4) });
        assert_eq!(a.owners[COMBINED], Sum { n: 1, committed: m(64), resident: 0 });
        assert_eq!(a.owners[OTHER], Sum { n: 2, committed: m(1056), resident: m(528) });
        assert_eq!(a.ours["a window"].committed, m(4));
        assert_eq!(a.ours["a descriptor heap"].n, 1);
        assert_eq!(a.unmet, [1, 3], "another size at its address; no allocation at all");
        assert_eq!(a.groups[&(true, 64 * MIB)].0.n, 1);
        assert_eq!(a.groups[&(false, 1024 * MIB)], (Sum { n: 1, committed: m(1024), resident: m(512) }, UP + 400 * MIB));
        assert_eq!(a.views.committed, m(2));
        assert_eq!(a.files[&1].committed, m(8));
        assert_eq!(a.images.committed, m(4));
        assert_eq!(a.again, m(1));
        // Each private byte once.
        assert_eq!(a.private().committed, m(24 + 4 + 1 + 8 + 64 + 1056));
        // 1120 of 1157 MB have no owner by name.
        assert_eq!(a.named(), [3, 3]);
        let line = owners_line(&a, regions.len(), 7);
        assert!(line.contains("private 1157 MB committed, 546 MB of it in memory"), "{line}");
        assert!(line.contains("the runtime told of 8 MB (4 MB in memory)"), "{line}");
        assert!(line.contains("write-combined, not ours 64 MB (0 MB in memory, 1 allocations)"), "{line}");
        assert!(ours_line(&a).contains("1 x \"a window\": 4 MB (2 MB in memory)"));
        // The window whose pointer now lies in a 32 MiB allocation is
        // named with that allocation's size; what lies in none is not.
        assert_eq!(a.within.len(), 1);
        assert_eq!(a.within[&(32 * MIB)], (1, 4 << 20));
        assert!(ours_line(&a).ends_with("or released since): 1 (4 MB) in allocations of 32.00 MB"), "{}", ours_line(&a));
        assert!(groups_line(&a).starts_with(
            "memory account, without an owner, by the allocation's size, the biggest of 3 groups: 1 of 1024.00 MB: 1024 MB (512 MB in memory; the first at 0x20019000000)"
        ));
        let mapped = mapped_line(&a, &["".into(), "library.bin".into()]);
        assert!(mapped.contains("library.bin: 8 MB (4 MB in memory, 1 views)"), "{mapped}");
        assert!(mapped.contains("1 MB is a page shown again"), "{mapped}");
        let check = check_line(&a, [9000 << 20, 5785 << 20, 0, 6364 << 20], [64, 0], [32 << 30, 4 << 30, 21 << 30], 6 << 30);
        assert!(check.contains("not ours and not in memory 64 MB, the card has 5785 MB in use"), "{check}");
        assert!(check.contains("1 write-combined allocations of 64 MB without an owner, texture and staging blocks alive 1"), "{check}");
        assert!(check.contains("used by everything else 7168 MB (+1024 MB since the first walk"), "{check}");
    }

    /// What the title took with its plain `VirtualAlloc` is the title's
    /// own, by size, in a line of its own; an address that now holds an
    /// allocation of another size is not, and what has no owner keeps to
    /// the rest.
    #[test]
    fn the_title_s_plain_allocations_are_its_own_by_size() {
        const GIB: usize = 1 << 30;
        let part = |base: usize, at: usize, len: usize, committed: bool, resident: u64| {
            let mut r = region(base, at, len, Type::Private);
            r.committed = committed;
            r.resident = if committed { resident } else { 0 };
            r
        };
        let regions = vec![
            // Two reservations of 1 GiB the title made, committed in part.
            part(UP, UP, 768 * MIB, true, 700 << 20),
            part(UP, UP + 768 * MIB, 256 * MIB, false, 0),
            part(UP + 2 * GIB, UP + 2 * GIB, 100 * MIB, true, 90 << 20),
            part(UP + 2 * GIB, UP + 2 * GIB + 100 * MIB, 924 * MIB, false, 0),
            // One of 32 MiB of the title's, one of 32 MiB that is nobody's.
            part(UP + 4 * GIB, UP + 4 * GIB, 32 * MIB, true, 32 << 20),
            part(UP + 5 * GIB, UP + 5 * GIB, 32 * MIB, true, 32 << 20),
            // Where the title once had 1 GiB: 8 MiB of somebody else's now.
            part(UP + 6 * GIB, UP + 6 * GIB, 8 * MIB, true, 8 << 20),
            // A 1 GiB allocation nobody told of.
            part(UP + 8 * GIB, UP + 8 * GIB, GIB, true, 1 << 30),
        ];
        let plain = |at: usize, size: usize| Holding {
            at,
            size: size as u64,
            label: "the title's own, by its plain VirtualAlloc".into(),
            inside: false,
            title: true,
            plain: true,
        };
        let held = vec![plain(UP, GIB), plain(UP + 2 * GIB, GIB), plain(UP + 4 * GIB, 32 * MIB), plain(UP + 6 * GIB, GIB)];
        let a = tally(&regions, &held, &[], &[]);
        // The title's own: 768 + 100 + 32 MiB committed, in three allocations.
        assert_eq!(a.owners[TITLE], Sum { n: 3, committed: 900 << 20, resident: (700 + 90 + 32) << 20 });
        assert_eq!(a.plain.len(), 2);
        assert_eq!(a.plain[&GIB], (Sum { n: 2, committed: 868 << 20, resident: 790 << 20 }, UP));
        assert_eq!(a.plain[&(32 * MIB)], (Sum { n: 1, committed: 32 << 20, resident: 32 << 20 }, UP + 4 * GIB));
        // Not what the runtime told of: that sum is for its own reservations.
        assert_eq!(a.above, Sum::default());
        // The address used again is not the title's; it and the two nobody
        // told of have no owner.
        assert_eq!(a.unmet, vec![3]);
        // (A plain allocation of the title's is no buffer of ours.)
        assert!(a.within.is_empty());
        assert_eq!(a.owners[OTHER], Sum { n: 3, committed: (32 + 8 + 1024) << 20, resident: (32 + 8 + 1024) << 20 });
        assert_eq!(a.groups.len(), 3);
        assert_eq!(a.groups[&(false, GIB)].0.n, 1);
        assert_eq!(a.groups[&(false, 32 * MIB)].0.n, 1);
        // Every committed byte once.
        assert_eq!(a.private().committed, (900 + 32 + 8 + 1024) << 20);
        let line = plain_line(&a);
        assert!(line.starts_with("memory account, the title's own by its plain VirtualAlloc import"), "{line}");
        assert!(line.contains("3 allocations, 900 MB committed, 822 MB in memory"), "{line}");
        assert!(line.contains("the biggest of 2 groups: 2 of 1024.00 MB: 868 MB (790 MB in memory; the first at 0x20000000000)"), "{line}");
        assert!(line.contains("1 of 32.00 MB: 32 MB (32 MB in memory"), "{line}");
        // Without any: the line says so, and the owners' line is as it was.
        let none = tally(&regions, &[], &[], &[]);
        assert!(none.plain.is_empty());
        assert!(ours_line(&none).ends_with("or released since): none"));
        assert!(plain_line(&none).ends_with("0 allocations, 0 MB committed, 0 MB in memory; by the allocation's size, the biggest of 0 groups: none"));
        assert_eq!(none.owners[TITLE], Sum::default());
    }

    #[test]
    fn a_page_shown_twice_is_memory_once() {
        assert_eq!(page(0, 65536), (0, 0));
        // In memory, private.
        assert_eq!(page(1, 65536), (65536, 0));
        // In memory, shared, shown once (another process may show it too).
        assert_eq!(page(1 | 1 << 1 | 1 << 15, 65536), (65536, 0));
        // Shown twice and three times.
        assert_eq!(page(1 | 2 << 1 | 1 << 15, 65536), (65536, 32768));
        assert_eq!(page(1 | 3 << 1 | 1 << 15, 60000), (60000, 40000));
    }

    #[test]
    fn a_list_s_cost_is_the_middle_batch() {
        let line = lists_line(&[[16 << 20, 8 << 20], [160 << 20, 8 << 20], [18 << 20, 9 << 20]], 16, 48 << 20);
        assert!(line.contains("private bytes grew by [1024, 1152, 10240] KiB a list (the middle 1152 KiB)"), "{line}");
        assert!(line.contains("the working set by [512, 512, 576] KiB a list (the middle 512 KiB)"), "{line}");
        assert!(line.contains("let go 1024 KiB a list came back"), "{line}");
    }

    #[test]
    fn nothing_is_kept_while_the_account_is_off() {
        // (The tests run without the switch.)
        if on() {
            return;
        }
        mapped(0x1000, 0x2000 as *const u8, 4096, "an upload buffer");
        range(0x3000, 4096, "a descriptor heap");
        named(0x1000, "a name");
        title_plain(0x5000_0000, 1 << 30);
        title_plain_gone(0x5000_0000);
        assert_eq!(plain_noted(), 0);
        let block_at = 0x7000_0000usize;
        block(block_at as *mut u8, 64 << 20, true);
        assert!(HELD.lock().unwrap().is_none());
        assert!(BLOCKS.iter().all(|b| b.load(Relaxed) != block_at));
    }
}
