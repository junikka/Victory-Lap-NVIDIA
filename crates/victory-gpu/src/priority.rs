//! What Windows is told of the card's memory (ticket 074): how much of it
//! the layer cannot go without, and which memory should leave the card last.
//!
//! Over its budget a process "will be intermittently frozen", and Windows
//! moves some of its memory off the card, into the PC's memory shared with
//! the card; "the wrong resource in system-memory can severely impact frame
//! rate" (Microsoft's residency page). In session 0152 the card stood at
//! 105 % of its budget from minute 18 on, and the submitting thread waited
//! 33 ms a frame for the card where it had waited 2 to 5.
//!
//! Which memory leaves first goes by priority, the lowest first. By the
//! documented default everything made as a target, a depth texture or with
//! unordered access is high (`D3D12_RESIDENCY_PRIORITY_HIGH`, and more by
//! its size: a 10 MB a step), and everything else is normal, all alike. The
//! card copies of title buffers, their "before" twins, the depth twins and
//! the textures the host renders to are all made with one of those flags:
//! they are high already. What is normal, all alike, is the texture pool's
//! blocks (5.4 GB of 64 MB blocks in session 0152, half of each empty), the
//! tile heaps and the view heaps: among those Windows takes any.
//!
//! Three switches, all off unless set; `=alt` turns one on and off every
//! twenty seconds (`crate::switch::alt_on`), and a diagnostics build's
//! `VICTORY_AB=<seconds>,<switch>` as long as it says (`crate::ab`):
//!
//! - `VICTORY_VRAM_RESERVE_MB=<n>` (`alt`: 4096 and none): asks the system
//!   to reserve so much of the card's own memory for the process
//!   (`IDXGIAdapter3::SetVideoMemoryReservation`), when the swap chain is
//!   made; at most what the system says can be reserved. What was asked and
//!   what the next query reports are logged. A reservation raises no
//!   budget: it "helps the OS kernel quickly minimize the impact of large
//!   memory pressure situations".
//! - `VICTORY_RESIDENCY_PRIORITY=1` (`alt`): the priorities are told
//!   (`ID3D12Device1::SetResidencyPriority`), not left to the default.
//!   High, by the default's own rule so that telling never lowers one
//!   ([`high`]): buffers on the card that shaders read or write, textures
//!   the host renders to or writes, depth twins. A block of
//!   the texture pool is normal, and low once none of its textures was
//!   shown by a slot, held by a list or asked for in [`IDLE_TICKS`] looks,
//!   a second or more apart (`crate::resource::blocks_in_use`); normal
//!   again at the next look after one is, or after a piece of it is taken.
//! - `VICTORY_PRIORITY_BY_FILL=1` (`alt`): a block of the texture pool that
//!   is not low is the lower the emptier it is, in the four bands of the
//!   pool line ([`BANDS`]): if Windows must take blocks, it takes those
//!   whose loss slows the fewest textures. Not in the ticket; on its own
//!   switch so that it is measured alone.
//!
//! While a switch in turn is off, the pool's blocks are normal again. What
//! was told high stays so: it is what the default says.
//!
//! Without a switch, [`tail`] ends the residency line every 5 s with the
//! seconds the card was over its budget, the lowest and highest budget
//! seen, the PC's shared memory in use (it grows as Windows moves ours off
//! the card), the megabytes reserved and the priorities told so far.
//!
//! Not done, and why: `Evict` of pool blocks. A block is read by slots
//! shaders pick by an index only the card knows; an evicted block that one
//! reads loses the device. A low priority leaves the choice to Windows,
//! which moves the memory and keeps it readable. `MakeResident` is already
//! waited for when a block is made (`crate::host::resident`).
//!
//! Nor a priority for the view heaps: for a shader-visible descriptor heap
//! the call is refused (0x80070057 on an RTX 3080) and the device is lost
//! with it (0x887a0001), so no view heap is ever named in one.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::{ID3D12Device1, ID3D12Pageable};
use windows::core::Interface;

/// `D3D12_RESIDENCY_PRIORITY_LOW`, `_NORMAL` and `_HIGH`.
pub(crate) const LOW: u32 = 0x5000_0000;
pub(crate) const NORMAL: u32 = 0x7800_0000;
const HIGH: u32 = 0xa001_0000;
const _: () = {
    use windows::Win32::Graphics::Direct3D12::*;
    assert!(D3D12_RESIDENCY_PRIORITY_LOW.0 as u32 == LOW);
    assert!(D3D12_RESIDENCY_PRIORITY_NORMAL.0 as u32 == NORMAL);
    assert!(D3D12_RESIDENCY_PRIORITY_HIGH.0 as u32 == HIGH);
};

/// A pool block's priority by how full it is (`VICTORY_PRIORITY_BY_FILL`):
/// under a quarter, to under a half, to under three quarters, and over.
/// All above [`LOW`]; the fullest is [`NORMAL`].
const BANDS: [u32; 4] = [0x5a00_0000, 0x6400_0000, 0x6e00_0000, NORMAL];

/// Looks ([`tick`]) without a sign of use after which a pool block is low.
pub(crate) const IDLE_TICKS: u64 = 30;
/// The pooled textures are looked at in so many parts, one a look.
const PARTS: usize = 8;
/// A block seen in use so few looks ago is not asked about again: one of
/// its textures is looked at within [`PARTS`] looks after that, well
/// before [`IDLE_TICKS`].
const FRESH_TICKS: u64 = 10;

/// The megabytes `VICTORY_VRAM_RESERVE_MB=alt` reserves in turn with none.
const USUAL_MB: u64 = 4096;

const OFF: u8 = 0;
const ON: u8 = 1;
const ALT: u8 = 2;

/// What an on/off switch's value asks for: `1` on, `alt` in turn, anything
/// else off.
fn mode_of(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        Some("1") => ON,
        Some("alt") => ALT,
        _ => OFF,
    }
}

/// What `VICTORY_VRAM_RESERVE_MB` asks for: so many megabytes all the time,
/// `alt` [`USUAL_MB`] in turn with none, anything else nothing.
fn megabytes_of(value: Option<&str>) -> (u8, u64) {
    match value.map(str::trim) {
        Some("alt") => (ALT, USUAL_MB),
        Some(v) => match v.parse::<u64>() {
            Ok(n) if n > 0 => (ON, n),
            _ => (OFF, 0),
        },
        None => (OFF, 0),
    }
}

/// A switch that is off, on, or on and off in turn; named in `VICTORY_AB`
/// (diagnostics builds) it goes by that. `number`: what it carries when on.
pub(crate) struct Switch {
    name: &'static str,
    what: &'static str,
    read: fn(Option<&str>) -> (u8, u64),
    /// The mode and the number, or its place in `VICTORY_AB`.
    asked: OnceLock<Result<(u8, u64), u32>>,
}

impl Switch {
    pub(crate) const fn new(name: &'static str, what: &'static str, read: fn(Option<&str>) -> (u8, u64)) -> Switch {
        Switch { name, what, read, asked: OnceLock::new() }
    }

    fn asked(&self) -> Result<(u8, u64), u32> {
        *self.asked.get_or_init(|| {
            #[cfg(feature = "diagnostics")]
            if let Some(bit) = crate::ab::bit(self.name) {
                tracing::info!("{} ({} goes by VICTORY_AB)", self.what, self.name);
                return Err(bit);
            }
            let (mode, number) = (self.read)(std::env::var(self.name).ok().as_deref());
            match mode {
                OFF => {}
                ON => tracing::info!("{} ({}={})", self.what, self.name, number),
                _ => tracing::info!("{} ({}=alt: 20 s off, 20 s on)", self.what, self.name),
            }
            Ok((mode, number))
        })
    }

    /// Whether it is set at all (on, or in turn).
    fn set(&self) -> bool {
        self.asked() != Ok((OFF, 0))
    }

    pub(crate) fn now(&self) -> bool {
        match self.asked() {
            Ok((OFF, _)) => false,
            Ok((ON, _)) => true,
            Ok(_) => crate::switch::alt_on(),
            #[cfg(feature = "diagnostics")]
            Err(bit) => crate::ab::on(bit),
            #[cfg(not(feature = "diagnostics"))]
            Err(_) => false,
        }
    }

    /// What it carries (in `VICTORY_AB`: `usual`).
    fn number(&self, usual: u64) -> u64 {
        self.asked().map_or(usual, |(_, number)| number)
    }
}

pub(crate) fn on_off(value: Option<&str>) -> (u8, u64) {
    match mode_of(value) {
        OFF => (OFF, 0),
        mode => (mode, 1),
    }
}

static RESERVE: Switch =
    Switch::new("VICTORY_VRAM_RESERVE_MB", "megabytes of the card's own memory are reserved with the system", megabytes_of);
static BY_USE: Switch = Switch::new(
    "VICTORY_RESIDENCY_PRIORITY",
    "Windows is told which memory stays on the card: buffers, targets and twins high, idle pool blocks low",
    on_off,
);
static BY_FILL: Switch =
    Switch::new("VICTORY_PRIORITY_BY_FILL", "Windows is told to move the emptier pool blocks off the card first", on_off);

/// Whether priorities are told at all (`VICTORY_RESIDENCY_PRIORITY` is on,
/// or in turn): the pooled textures are then kept track of.
pub(crate) fn by_use_set() -> bool {
    BY_USE.set()
}

/// What a kind of memory is to Windows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Kind {
    /// A buffer in the card's memory that shaders read or write at every
    /// draw: a title buffer's card copy, its "before" twin, a window.
    Buffer,
    /// A texture the host renders to or writes: a target, a depth texture,
    /// a depth twin.
    Written,
    /// A block of the texture pool: whether nothing used a texture of it
    /// for [`IDLE_TICKS`] looks, and its band of fullness ([`BANDS`]).
    Block { idle: bool, band: usize },
}

/// Which of the pool's rules hold now.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Rules {
    /// `VICTORY_RESIDENCY_PRIORITY`: idle blocks are low.
    pub(crate) by_use: bool,
    /// `VICTORY_PRIORITY_BY_FILL`: emptier blocks are lower.
    pub(crate) by_fill: bool,
}

/// The default's own high priority for `bytes` of memory: 10 MB a step, at
/// most 0xffff of them. Telling it never lowers what the default gave.
fn high(bytes: u64) -> u32 {
    HIGH | (bytes / (10 << 20)).min(0xffff) as u32
}

/// The priority of `bytes` of memory of `kind` under `rules`.
pub(crate) fn of(kind: Kind, bytes: u64, rules: Rules) -> u32 {
    match kind {
        Kind::Buffer | Kind::Written => high(bytes),
        Kind::Block { idle: true, .. } if rules.by_use => LOW,
        Kind::Block { band, .. } if rules.by_fill => BANDS[band.min(BANDS.len() - 1)],
        Kind::Block { .. } => NORMAL,
    }
}

/// Objects told a priority since the start: high, normal, low, by fill
/// (between low and normal); and calls the system refused.
static TOLD: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Pool blocks that are low now, and below normal by their fill.
static BLOCKS_NOW: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Which of [`TOLD`] a priority counts in.
fn class(priority: u32) -> usize {
    match priority {
        p if p >= HIGH => 0,
        NORMAL => 1,
        LOW => 2,
        _ => 3,
    }
}

fn device() -> Option<&'static ID3D12Device1> {
    static DEVICE: OnceLock<Option<ID3D12Device1>> = OnceLock::new();
    let host = crate::host::made_host()?;
    DEVICE.get_or_init(|| host.device.cast().ok()).as_ref()
}

/// Tells Windows the priority of each object: one call for all of them.
pub(crate) fn tell(objects: &[(ID3D12Pageable, u32)]) {
    if objects.is_empty() {
        return;
    }
    let Some(device) = device() else {
        return;
    };
    let pageables: Vec<Option<ID3D12Pageable>> = objects.iter().map(|(o, _)| Some(o.clone())).collect();
    let priorities: Vec<windows::Win32::Graphics::Direct3D12::D3D12_RESIDENCY_PRIORITY> = objects
        .iter()
        .map(|&(_, p)| windows::Win32::Graphics::Direct3D12::D3D12_RESIDENCY_PRIORITY(p as i32))
        .collect();
    // SAFETY: two arrays of the same length, of live objects of the device's.
    let told = unsafe { device.SetResidencyPriority(objects.len() as u32, pageables.as_ptr(), priorities.as_ptr()) };
    match told {
        Ok(()) => {
            for &(_, priority) in objects {
                TOLD[class(priority)].fetch_add(1, Relaxed);
            }
        }
        Err(e) => {
            if TOLD[4].fetch_add(1, Relaxed) == 0 {
                tracing::warn!(error = %e, objects = objects.len(), "the system took no residency priority");
            }
        }
    }
}

/// `object`, `bytes` of memory of `kind` just made, is told high while
/// `VICTORY_RESIDENCY_PRIORITY` is on.
pub(crate) fn keep(object: &impl Interface, kind: Kind, bytes: u64) {
    if !BY_USE.now() {
        return;
    }
    if let Ok(pageable) = object.cast::<ID3D12Pageable>() {
        tell(&[(pageable, of(kind, bytes, Rules::default()))]);
    }
}

/// Looks so far at the pooled textures' use ([`tick`]): the clock a pool
/// block's last sign of use is noted by.
static TICK: AtomicU64 = AtomicU64::new(0);

pub(crate) fn ticks() -> u64 {
    TICK.load(Relaxed)
}

/// The pool's blocks that are low, and below normal by their fill, as the
/// last look left them.
pub(crate) fn blocks_now(low: u64, by_fill: u64) {
    BLOCKS_NOW[0].store(low, Relaxed);
    BLOCKS_NOW[1].store(by_fill, Relaxed);
}

/// What was seen of the card's budget since the start.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Budget {
    /// Milliseconds the card's own memory in use was over the budget.
    over_ms: u64,
    /// The lowest and the highest budget (0: none seen yet).
    lowest: u64,
    highest: u64,
}

impl Budget {
    /// A look `ms` after the one before: over its budget now, the card
    /// counts as over for that long, two seconds at most (looks stop while
    /// nothing is submitted).
    fn look(&mut self, ms: u64, budget: u64, used: u64) {
        if budget == 0 {
            return;
        }
        self.lowest = if self.lowest == 0 { budget } else { self.lowest.min(budget) };
        self.highest = self.highest.max(budget);
        if used > budget {
            self.over_ms += ms.min(2000);
        }
    }
}

static BUDGET: Mutex<(Option<std::time::Instant>, Budget)> = Mutex::new((None, Budget { over_ms: 0, lowest: 0, highest: 0 }));

/// The megabytes reserved, as the query after the last asking said.
static RESERVED_MB: AtomicU64 = AtomicU64::new(0);

/// The bytes to ask the system to reserve of `wanted`: not more than it
/// says can be.
fn reservation(wanted: u64, available: u64) -> u64 {
    wanted.min(available)
}

/// Asks the system for the reservation `VICTORY_VRAM_RESERVE_MB` wants now,
/// if that is not what was asked last. Called when the swap chain is made
/// and once a second.
pub(crate) fn reserve(host: &crate::host::Host) {
    if !RESERVE.set() {
        return;
    }
    /// The bytes asked for last (`u64::MAX`: nothing yet).
    static ASKED: AtomicU64 = AtomicU64::new(u64::MAX);
    let megabytes = RESERVE.number(USUAL_MB);
    let wanted = if RESERVE.now() { megabytes << 20 } else { 0 };
    let last = ASKED.swap(wanted, Relaxed);
    // (None is what a process has before it asks.)
    if last == wanted || (last == u64::MAX && wanted == 0) {
        return;
    }
    let Some([own, _]) = host.video_memory_groups() else {
        tracing::warn!("no reservation of the card's memory: the system does not name the adapter");
        return;
    };
    let ask = reservation(wanted, own[2]);
    let answer = match host.reserve_video_memory(ask) {
        Some(Ok(())) => "taken".to_owned(),
        Some(Err(e)) => format!("refused ({e})"),
        None => "not asked (no adapter)".to_owned(),
    };
    let after = host.video_memory_groups().map_or([0; 4], |[own, _]| own);
    RESERVED_MB.store(after[3] >> 20, Relaxed);
    tracing::info!(
        "reservation of the card's own memory: asked for {} MB (VICTORY_VRAM_RESERVE_MB wants {} MB now; {} MB can be \
         reserved), {answer}; the next query: budget {} MB, in use {} MB, {} MB can be reserved, {} MB reserved (frame {})",
        ask >> 20,
        wanted >> 20,
        own[2] >> 20,
        after[0] >> 20,
        after[1] >> 20,
        after[2] >> 20,
        after[3] >> 20,
        crate::frame::current()
    );
}

/// Once a second, before a submission starts, with the card's memory as
/// the system counts it: notes the budget, asks for the reservation if it
/// changed, and tells the pool's blocks' priorities where they changed.
pub(crate) fn tick(memory: Option<[u64; 4]>) {
    {
        let now = std::time::Instant::now();
        let mut budget = BUDGET.lock().unwrap_or_else(|e| e.into_inner());
        let ms = budget.0.replace(now).map_or(0, |last| (now - last).as_millis() as u64);
        if let Some([of, used, ..]) = memory {
            budget.1.look(ms, of, used);
        }
    }
    if let Some(host) = crate::host::made_host() {
        reserve(host);
    }
    if !(BY_USE.set() || BY_FILL.set()) {
        return;
    }
    let rules = Rules { by_use: BY_USE.now(), by_fill: BY_FILL.now() };
    let mut seen = Vec::new();
    if rules.by_use {
        let tick = TICK.fetch_add(1, Relaxed) + 1;
        let fresh = crate::pool::fresh_blocks(tick, FRESH_TICKS);
        seen = crate::resource::blocks_in_use(PARTS, IDLE_TICKS * 1000, |block| {
            fresh.get(block as usize).copied().unwrap_or(false)
        });
    }
    crate::pool::prioritize(&seen, TICK.load(Relaxed), rules);
}

/// The end of the residency line. `budget`: [`Budget`]; `shared`: the PC's
/// memory shared with the card in use now and at the line before, in
/// bytes (`None`: not known); `reserved`: megabytes; `told`: [`TOLD`];
/// `blocks`: [`BLOCKS_NOW`].
fn tail_of(budget: Budget, shared: Option<[u64; 2]>, reserved: u64, told: [u64; 5], blocks: [u64; 2]) -> String {
    let shared = shared.map_or("unknown".to_owned(), |[now, before]| {
        format!("{} MB ({:+} since the line before)", now >> 20, (now >> 20) as i64 - (before >> 20) as i64)
    });
    format!(
        "; card over its budget {} s, budget from {} to {} MB, the PC's shared memory in use {shared}, reserved {reserved} MB; \
         priorities told: high {}, normal {}, low {}, by fill {}, refused calls {} (pool blocks low now {}, lower by fill {})",
        budget.over_ms / 1000,
        budget.lowest >> 20,
        budget.highest >> 20,
        told[0],
        told[1],
        told[2],
        told[3],
        told[4],
        blocks[0],
        blocks[1],
    )
}

/// The end of the residency line, every 5 s: see [`tail_of`]. `card`: the
/// card's memory as the system counts it.
pub(crate) fn tail(card: Option<[u64; 4]>) -> String {
    /// The PC's shared memory in use at the line before.
    static SHARED: AtomicU64 = AtomicU64::new(0);
    let shared = card.map(|memory| [memory[3], SHARED.swap(memory[3], Relaxed)]);
    let budget = BUDGET.lock().unwrap_or_else(|e| e.into_inner()).1;
    tail_of(
        budget,
        shared,
        RESERVED_MB.load(Relaxed),
        TOLD.each_ref().map(|t| t.load(Relaxed)),
        BLOCKS_NOW.each_ref().map(|b| b.load(Relaxed)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_kind_of_memory_has_its_priority() {
        let none = Rules::default();
        let by_use = Rules { by_use: true, by_fill: false };
        let by_fill = Rules { by_use: false, by_fill: true };
        let both = Rules { by_use: true, by_fill: true };
        // What shaders read at every draw and what the host writes: high,
        // whatever the pool's rules, and more by size as the default has it.
        for rules in [none, by_use, by_fill, both] {
            assert_eq!(of(Kind::Buffer, 0, rules), 0xa001_0000);
            assert_eq!(of(Kind::Buffer, 64 << 20, rules), 0xa001_0006);
            assert_eq!(of(Kind::Written, 100 << 20, rules), 0xa001_000a);
            assert_eq!(of(Kind::Written, 32 << 20, rules), 0xa001_0003);
            assert_eq!(of(Kind::Written, u64::MAX, rules), 0xa001_ffff);
        }
        // A pool block: normal while one of its textures is used, however
        // full; low once none was, with the switch.
        let used = Kind::Block { idle: false, band: 0 };
        let idle = Kind::Block { idle: true, band: 3 };
        assert_eq!(of(used, 64 << 20, by_use), NORMAL);
        assert_eq!(of(idle, 64 << 20, by_use), LOW);
        assert_eq!(of(idle, 64 << 20, none), NORMAL);
        // By fill: the emptier the lower, the fullest normal, none as low
        // as an idle one; idle wins where both rules hold.
        let filled = |band| of(Kind::Block { idle: false, band }, 64 << 20, by_fill);
        assert_eq!([filled(0), filled(1), filled(2), filled(3)], [0x5a00_0000, 0x6400_0000, 0x6e00_0000, NORMAL]);
        assert!(LOW < filled(0) && filled(0) < filled(1) && filled(1) < filled(2) && filled(2) < filled(3));
        assert_eq!(filled(9), NORMAL);
        assert_eq!(of(idle, 64 << 20, by_fill), NORMAL);
        assert_eq!(of(idle, 64 << 20, both), LOW);
        assert_eq!(of(Kind::Block { idle: false, band: 1 }, 64 << 20, both), 0x6400_0000);
        // High is above every block, and what is told high never falls
        // under the documented default.
        assert!(of(Kind::Buffer, 0, both) > NORMAL);
        assert_eq!([class(0xa001_0006), class(NORMAL), class(LOW), class(0x6400_0000)], [0, 1, 2, 3]);
    }

    #[test]
    fn switches_read_one_alt_and_megabytes() {
        assert_eq!(on_off(None), (OFF, 0));
        assert_eq!(on_off(Some("0")), (OFF, 0));
        assert_eq!(on_off(Some("yes")), (OFF, 0));
        assert_eq!(on_off(Some(" 1 ")), (ON, 1));
        assert_eq!(on_off(Some("alt")), (ALT, 1));
        assert_eq!(megabytes_of(None), (OFF, 0));
        assert_eq!(megabytes_of(Some("0")), (OFF, 0));
        assert_eq!(megabytes_of(Some("-4")), (OFF, 0));
        assert_eq!(megabytes_of(Some("many")), (OFF, 0));
        assert_eq!(megabytes_of(Some(" 4096 ")), (ON, 4096));
        assert_eq!(megabytes_of(Some("1")), (ON, 1));
        assert_eq!(megabytes_of(Some("alt")), (ALT, USUAL_MB));
    }

    #[test]
    fn no_more_is_asked_than_can_be_reserved() {
        // Session 0152: 4,770 MB could be reserved.
        assert_eq!(reservation(4096 << 20, 4770 << 20), 4096 << 20);
        assert_eq!(reservation(8192 << 20, 4770 << 20), 4770 << 20);
        assert_eq!(reservation(0, 4770 << 20), 0);
        assert_eq!(reservation(4096 << 20, 0), 0);
    }

    #[test]
    fn the_budget_is_followed_and_the_time_over_it_counted() {
        let mut b = Budget::default();
        // No budget known: nothing.
        b.look(1000, 0, 5 << 30);
        assert_eq!(b, Budget::default());
        b.look(0, 9285 << 20, 6000 << 20);
        b.look(1000, 9121 << 20, 9121 << 20);
        assert_eq!((b.over_ms, b.lowest >> 20, b.highest >> 20), (0, 9121, 9285));
        // Over: the time since the look before counts, two seconds at most.
        b.look(1000, 9121 << 20, 9362 << 20);
        b.look(1100, 9030 << 20, 9098 << 20);
        b.look(60_000, 9105 << 20, 9371 << 20);
        b.look(1000, 9300 << 20, 9000 << 20);
        assert_eq!((b.over_ms, b.lowest >> 20, b.highest >> 20), (4100, 9030, 9300));
    }

    #[test]
    fn the_tail_holds_every_count() {
        let budget = Budget { over_ms: 512_900, lowest: 8942 << 20, highest: 9285 << 20 };
        assert_eq!(
            tail_of(budget, Some([5961 << 20, 5949 << 20]), 4096, [1203, 12, 3, 40, 0], [3, 17]),
            "; card over its budget 512 s, budget from 8942 to 9285 MB, the PC's shared memory in use 5961 MB (+12 since \
             the line before), reserved 4096 MB; priorities told: high 1203, normal 12, low 3, by fill 40, refused calls 0 \
             (pool blocks low now 3, lower by fill 17)"
        );
        let none = tail_of(Budget::default(), None, 0, [0; 5], [0; 2]);
        assert!(none.contains("over its budget 0 s, budget from 0 to 0 MB, the PC's shared memory in use unknown, reserved 0 MB"));
        assert!(tail_of(budget, Some([100 << 20, 300 << 20]), 0, [0; 5], [0; 2]).contains("100 MB (-200 since the line before)"));
    }

    /// The real calls, on this machine's card: a priority is taken for a
    /// heap and a buffer, and a reservation is taken and given back. (No
    /// view heap: the call is refused for one and the device lost with it.)
    #[test]
    fn the_system_takes_priorities_and_a_reservation() {
        use windows::Win32::Graphics::Direct3D12::*;
        let Some(host) = crate::host::host() else { return };
        let Some(device) = device() else { return };
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: 1 << 20,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation.
        unsafe { host.device.CreateHeap(&desc, &mut heap) }.unwrap();
        let heap: ID3D12Pageable = heap.unwrap().cast().unwrap();
        let buffer: ID3D12Pageable = host.uav_buffer(1 << 16).unwrap().cast().unwrap();
        let rules = Rules { by_use: true, by_fill: true };
        for (object, kind) in [
            (&heap, Kind::Block { idle: true, band: 0 }),
            (&heap, Kind::Block { idle: false, band: 1 }),
            (&heap, Kind::Block { idle: false, band: 3 }),
            (&buffer, Kind::Buffer),
        ] {
            let priority = D3D12_RESIDENCY_PRIORITY(of(kind, 1 << 20, rules) as i32);
            // SAFETY: one live object and one priority.
            unsafe { device.SetResidencyPriority(1, &Some(object.clone()), &priority) }.unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        }
        // SAFETY: a plain call on a live device.
        unsafe { host.device.GetDeviceRemovedReason() }.expect("the device is kept");
        let before = TOLD.each_ref().map(|t| t.load(Relaxed));
        tell(&[(heap.clone(), LOW), (buffer.clone(), high(1 << 16)), (heap, NORMAL)]);
        let after = TOLD.each_ref().map(|t| t.load(Relaxed));
        assert!(after[0] > before[0] && after[1] > before[1] && after[2] > before[2]);
        assert_eq!(after[4], before[4]);
        // A reservation, where the system names the adapter and has room.
        let Some([own, _]) = host.video_memory_groups() else { return };
        let ask = reservation(64 << 20, own[2]);
        if ask == 0 {
            return;
        }
        host.reserve_video_memory(ask).unwrap().unwrap();
        let reserved = host.video_memory_groups().unwrap()[0][3];
        println!("reservation: asked {} MB of {} MB that can be reserved, the next query says {} MB", ask >> 20, own[2] >> 20, reserved >> 20);
        assert_eq!(reserved, ask);
        host.reserve_video_memory(0).unwrap().unwrap();
        assert_eq!(host.video_memory_groups().unwrap()[0][3], 0);
    }
}
