//! `gameinput.dll`: input devices from the host. Gamepads come from XInput
//! (host slots 0-3); while none is connected, the keyboard drives one
//! stand-in gamepad (`FALLBACK_BUTTONS`). The keyboard is a device of its own
//! too. `VICTORY_MOUSE=1` (off by default) makes the mouse that pad's right
//! stick and triggers, with a PC key layout. A poll thread turns host state
//! into readings (a new one per change) and runs device callbacks on connect
//! and disconnect.
#![allow(non_snake_case)]
// Title code calls these through the objects, passing pointers the real API
// would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, LazyLock, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use super::xasync::{E_INVALIDARG, E_NOTIMPL};
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleExport, HleModule};

mod record;

/// The documented errors.
pub const DEVICE_DISCONNECTED: HRESULT = 0x838A_0001_u32 as i32;
pub const DEVICE_NOT_FOUND: HRESULT = 0x838A_0002_u32 as i32;
pub const READING_NOT_FOUND: HRESULT = 0x838A_0003_u32 as i32;
pub const REFERENCE_TOO_OLD: HRESULT = 0x838A_0004_u32 as i32;
const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;

/// Input kinds (bits) our devices have.
const KEYBOARD_KIND: u32 = 0x10;
const GAMEPAD_KIND: u32 = 0x4_0000;
/// Device status of a connected device: connected, input enabled.
const STATUS_CONNECTED: u32 = 0x3;
const ASYNC_ENUMERATION: u32 = 1;

/// Gamepad buttons; the title maps each bit to one of its pad buttons.
const MENU: u32 = 0x1;
const VIEW: u32 = 0x2;
const A: u32 = 0x4;
const B: u32 = 0x8;
const X: u32 = 0x10;
const Y: u32 = 0x20;
const UP: u32 = 0x40;
const DOWN: u32 = 0x80;
const LEFT: u32 = 0x100;
const RIGHT: u32 = 0x200;
const LEFT_SHOULDER: u32 = 0x400;
const RIGHT_SHOULDER: u32 = 0x800;
const LEFT_STICK: u32 = 0x1000;
const RIGHT_STICK: u32 = 0x2000;

/// Devices: host pad slots, the keyboard's stand-in pad, the keyboard.
const PADS: usize = 4;
const FALLBACK: usize = PADS;
const KEYBOARD: usize = PADS + 1;
const DEVICES: usize = PADS + 2;

/// Readings kept per device for next/previous lookups.
const HISTORY: usize = 256;
const POLL: Duration = Duration::from_millis(8);

/// The keyboard's stand-in gamepad (virtual key, button); README lists it.
/// Sticks: W/A/S/D left, I/J/K/L right; triggers Z and C. This is the table
/// while `VICTORY_MOUSE` is unset; [`MOUSE_BUTTONS`] is the one while it is on.
const FALLBACK_BUTTONS: [(u8, u32); 14] = [
    (0x1B, MENU), // Esc
    (0x09, VIEW), // Tab
    (0x0D, A),    // Enter
    (0x08, B),    // Backspace
    (b'X', X),
    (b'Y', Y),
    (0x26, UP), // arrows
    (0x28, DOWN),
    (0x25, LEFT),
    (0x27, RIGHT),
    (b'Q', LEFT_SHOULDER),
    (b'E', RIGHT_SHOULDER),
    (b'F', LEFT_STICK),
    (b'H', RIGHT_STICK),
];

/// The stand-in gamepad of keyboard and mouse together (`VICTORY_MOUSE=1`),
/// laid out the way third-person titles are on a PC.
/// Left stick: W/A/S/D. Right stick: the mouse ([`Look`]). Triggers: the
/// right mouse button (left) and the left one (right); Z is the left trigger
/// too. Caps Lock switches to driving and back: W and S are then the right
/// and left trigger (the pedals) and Space the right shoulder; A and D still
/// steer. A pad has one set of buttons for walking and driving and a title
/// does not say which it is at, so the switch is the player's.
const MOUSE_BUTTONS: [(u8, u32); 22] = [
    (0x1B, MENU), // Esc
    (b'P', MENU),
    (b'V', VIEW),
    (0x0D, A), // Enter
    (0xA0, A), // left Shift
    (0x08, B), // Backspace
    (b'R', B),
    (SPACE, X),
    (b'X', X),
    (b'F', Y),
    (b'Y', Y),
    (0x26, UP), // arrows
    (0x28, DOWN),
    (0x25, LEFT),
    (0x27, RIGHT),
    (b'E', RIGHT),
    (0x09, LEFT_SHOULDER), // Tab
    (b'Q', RIGHT_SHOULDER),
    (0xA2, LEFT_STICK), // left Ctrl
    (b'C', RIGHT_STICK),
    (b'G', LEFT_STICK),
    (b'H', RIGHT_STICK),
];
const SPACE: u8 = 0x20;

/// What the mouse adds to the stand-in pad.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Pointer {
    /// The mouse is part of the pad: the keys are [`MOUSE_BUTTONS`].
    on: bool,
    right_x: f32,
    right_y: f32,
    /// The left and the right button.
    fire: bool,
    aim: bool,
    /// The keys are laid out for driving.
    drive: bool,
}

/// Mouse travel as a stick: a stick sets how fast the camera turns, a mouse
/// how far. Travel is kept and paid out as a deflection that turns the camera
/// at `speed` pixels a second when full, so the whole of a movement arrives,
/// a twentieth of a second late at most while the stick is not full.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Look {
    x: f32,
    y: f32,
}

impl Look {
    /// The time over which travel is paid out, in seconds.
    const LAG: f32 = 0.045;
    /// The most travel kept, in seconds of a full stick: a flick does not
    /// turn the camera for long after the hand has stopped.
    const KEPT: f32 = 0.25;

    /// Takes `(dx, dy)` pixels moved in the last `dt` seconds; gives the
    /// stick (right and up positive). `dead` is the share of the stick the
    /// title ignores around its middle: a deflection starts above it.
    fn turn(&mut self, (dx, dy): (f32, f32), dt: f32, speed: f32, dead: f32) -> (f32, f32) {
        let axis = |kept: &mut f32, moved: f32| {
            *kept = (*kept + moved).clamp(-speed * Self::KEPT, speed * Self::KEPT);
            let stick = (*kept / (speed * Self::LAG)).clamp(-1.0, 1.0);
            *kept -= stick * speed * dt.min(Self::LAG);
            if stick.abs() < 0.004 {
                0.0
            } else {
                stick.signum() * (dead + (1.0 - dead) * stick.abs())
            }
        };
        (axis(&mut self.x, dx), -axis(&mut self.y, dy))
    }
}

/// Whether the mouse is held for the camera right now: the pointer is not
/// shown over the title's picture then (`lifecycle`'s window procedure).
pub(crate) fn mouse_held() -> bool {
    MOUSE_HELD.load(Ordering::Relaxed)
}
static MOUSE_HELD: AtomicBool = AtomicBool::new(false);

/// A gamepad reading: triggers 0 to 1, sticks -1 to 1 (up positive).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PadState {
    buttons: u32,
    left_trigger: f32,
    right_trigger: f32,
    left_x: f32,
    left_y: f32,
    right_x: f32,
    right_y: f32,
}

/// One key held down: scan code (0xE0xx when extended), character, key.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct KeyState {
    scan_code: u32,
    code_point: u32,
    virtual_key: u8,
    dead_key: bool,
}

const _: () = assert!(size_of::<KeyState>() == 12 && size_of::<PadState>() == 28);

/// Host input, or a fake in tests.
trait Source: Send {
    /// The pad in host slot `slot`, `None` when the slot is empty.
    fn pad(&mut self, slot: usize) -> Option<PadState>;
    /// Keys held down; none while the title's window lacks focus.
    fn keys(&mut self) -> Vec<KeyState>;
    /// The mouse as part of the stand-in pad; nothing unless the host has one.
    fn pointer(&mut self) -> Pointer {
        Pointer::default()
    }
    /// Motor speeds: low frequency (left), high frequency (right).
    fn rumble(&mut self, slot: usize, low: u16, high: u16);
}

fn fallback(keys: &[KeyState], pointer: Pointer) -> PadState {
    let on = |vk: u8| -> f32 {
        if keys.iter().any(|k| k.virtual_key == vk) {
            1.0
        } else {
            0.0
        }
    };
    if !pointer.on {
        return PadState {
            buttons: FALLBACK_BUTTONS
                .iter()
                .filter(|(vk, _)| on(*vk) != 0.0)
                .fold(0, |all, (_, bit)| all | bit),
            left_trigger: on(b'Z'),
            right_trigger: on(b'C'),
            left_x: on(b'D') - on(b'A'),
            left_y: on(b'W') - on(b'S'),
            right_x: on(b'L') - on(b'J'),
            right_y: on(b'I') - on(b'K'),
        };
    }
    let held = |held: bool| -> f32 {
        if held {
            1.0
        } else {
            0.0
        }
    };
    // Driving: W and S are the pedals, Space the handbrake.
    let drive = pointer.drive;
    let (walk, gas, brake) = if drive {
        (0.0, on(b'W'), on(b'S'))
    } else {
        (on(b'W') - on(b'S'), 0.0, 0.0)
    };
    let handbrake = if drive && on(SPACE) != 0.0 { RIGHT_SHOULDER } else { 0 };
    PadState {
        buttons: MOUSE_BUTTONS
            .iter()
            .filter(|(vk, _)| on(*vk) != 0.0 && !(drive && *vk == SPACE))
            .fold(handbrake, |all, (_, bit)| all | bit),
        left_trigger: on(b'Z').max(held(pointer.aim)).max(brake),
        right_trigger: held(pointer.fire).max(gas),
        left_x: on(b'D') - on(b'A'),
        left_y: walk,
        right_x: pointer.right_x,
        right_y: pointer.right_y,
    }
}

/// Pad 0 as `VICTORY_INPUT_RECORD` stores it: while the mouse is on, its
/// right stick and its buttons' triggers go into that pad (a replayed pad 0
/// is merged with the stand-in, so they come back where they were); with it
/// off, or still, the host pad is stored as it was (`None` stays
/// disconnected).
fn recorded_pad(host: Option<record::Raw>, pointer: Pointer) -> Option<record::Raw> {
    let moved = pointer.right_x != 0.0 || pointer.right_y != 0.0 || pointer.fire || pointer.aim;
    if !pointer.on || !moved {
        return host;
    }
    let mut raw = host.unwrap_or_default();
    let axis = |v: f32| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
    let rx = axis(pointer.right_x);
    let ry = axis(pointer.right_y);
    if rx.unsigned_abs() > raw.rx.unsigned_abs() {
        raw.rx = rx;
    }
    if ry.unsigned_abs() > raw.ry.unsigned_abs() {
        raw.ry = ry;
    }
    if pointer.aim {
        raw.lt = 255;
    }
    if pointer.fire {
        raw.rt = 255;
    }
    Some(raw)
}

/// Two pads as one: buttons of both, and of each axis and trigger the one
/// pushed further.
fn merged(a: PadState, b: PadState) -> PadState {
    let further = |x: f32, y: f32| if y.abs() > x.abs() { y } else { x };
    PadState {
        buttons: a.buttons | b.buttons,
        left_trigger: further(a.left_trigger, b.left_trigger),
        right_trigger: further(a.right_trigger, b.right_trigger),
        left_x: further(a.left_x, b.left_x),
        left_y: further(a.left_y, b.left_y),
        right_x: further(a.right_x, b.right_x),
        right_y: further(a.right_y, b.right_y),
    }
}

/// Whether the title sees one pad that is always there, fed by the keyboard
/// and by every host pad (on by default; `VICTORY_ONE_PAD=0` shows each host
/// pad as a device of its own, and the stand-in only while there is none).
/// The title keeps to the pad it started with: with the stand-in taken away
/// when a pad was switched on, it showed its controller-disconnected screen
/// until the pad was off again. It also keeps a pad the host lists twice (its
/// own slot and a remapper's) from being two players.
fn one_pad() -> bool {
    std::env::var_os("VICTORY_ONE_PAD").is_none_or(|v| v != "0")
}

/// Diagnostics builds: which input calls the title makes. Each method of the
/// input object and of a reading is logged on its first call and on each
/// input kind it is first asked for; a reading's calls are counted and
/// reported every `REPORT` (`Hub::start`); trigger presses are logged where
/// `gamepad_state` hands them over. Release builds skip all of it.
mod probe {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
    use std::time::Duration;

    use super::{DEVICES, PadState};

    pub(super) const REPORT: Duration = Duration::from_secs(10);
    pub(super) const ON: bool = cfg!(feature = "diagnostics");

    #[derive(Clone, Copy)]
    pub(super) enum Table {
        Input,
        Reading,
    }

    const INPUT: [&str; 22] = [
        "QueryInterface", "AddRef", "Release", "GetCurrentTimestamp",
        "GetCurrentReading", "GetNextReading", "GetPreviousReading", "GetTemporalReading",
        "RegisterReadingCallback", "RegisterDeviceCallback", "RegisterSystemButtonCallback",
        "RegisterKeyboardLayoutCallback", "StopCallback", "UnregisterCallback",
        "CreateDispatcher", "CreateAggregateDevice", "FindDeviceFromId",
        "FindDeviceFromObject", "FindDeviceFromPlatformHandle", "FindDeviceFromPlatformString",
        "EnableOemDeviceSupport", "SetFocusPolicy",
    ];
    const READING: [&str; 25] = [
        "QueryInterface", "AddRef", "Release", "GetInputKind", "GetSequenceNumber",
        "GetTimestamp", "GetDevice", "GetRawReport (none)", "GetControllerAxisCount (none)",
        "GetControllerAxisState (none)", "GetControllerButtonCount (none)",
        "GetControllerButtonState (none)", "GetControllerSwitchCount (none)",
        "GetControllerSwitchState (none)", "GetKeyCount", "GetKeyState", "GetMouseState (none)",
        "GetTouchCount (none)", "GetTouchState (none)", "GetMotionState (none)",
        "GetArcadeStickState (none)", "GetFlightStickState (none)", "GetGamepadState",
        "GetRacingWheelState (none)", "GetUiNavigationState (none)",
    ];

    const SLOTS: usize = 25;
    static CALLS: [[AtomicU64; SLOTS]; 2] = [const { [const { AtomicU64::new(0) }; SLOTS] }; 2];
    /// Kinds each method was asked for, or'ed together.
    static KINDS: [[AtomicU32; SLOTS]; 2] = [const { [const { AtomicU32::new(0) }; SLOTS] }; 2];
    /// Kinds `GetCurrentReading` found no reading for.
    static MISSED: AtomicU32 = AtomicU32::new(0);

    thread_local! {
        /// The input method that last looked up a reading on this thread.
        static LOOKUP: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn name(table: Table, offset: usize) -> &'static str {
        let names: &[&str] = match table {
            Table::Input => &INPUT,
            Table::Reading => &READING,
        };
        names.get(offset / 8).copied().unwrap_or("?")
    }

    /// One call of the method at `offset`, asked for `kind` if it takes one.
    pub(super) fn call(table: Table, offset: usize, kind: Option<u32>) {
        if !ON {
            return;
        }
        let i = offset / 8;
        let first = CALLS[table as usize][i].fetch_add(1, Relaxed) == 0;
        let new_kind = kind.filter(|&k| KINDS[table as usize][i].fetch_or(k, Relaxed) & k != k);
        if first || new_kind.is_some() {
            let what = match table {
                Table::Input => "input object",
                Table::Reading => "reading",
            };
            tracing::info!(
                "input probe: first call of {what} {offset:#x} {} kind {}",
                name(table, offset),
                kind.map_or("-".to_owned(), |k| format!("{k:#x}"))
            );
        }
        if matches!(table, Table::Input) && (0x20..=0x38).contains(&offset) {
            LOOKUP.with(|l| l.set(offset));
        }
    }

    /// `GetCurrentReading` found nothing of `kind`; logged once per kind.
    pub(super) fn missed(kind: u32) {
        if ON && MISSED.fetch_or(kind, Relaxed) & kind != kind {
            tracing::info!("input probe: no current reading of kind {kind:#x}");
        }
    }

    /// Calls since `last`, for slots with any: one line, or `None`.
    pub(super) fn report(last: &mut [[u64; SLOTS]; 2]) -> Option<String> {
        let mut parts = Vec::new();
        for (table, tag) in [(Table::Reading, "reading"), (Table::Input, "input")] {
            let t = table as usize;
            for i in 0..SLOTS {
                let now = CALLS[t][i].load(Relaxed);
                if now > last[t][i] {
                    parts.push(format!("{tag} {:#x} {} {}", i * 8, name(table, i * 8), now - last[t][i]));
                }
                last[t][i] = now;
            }
        }
        (!parts.is_empty()).then(|| parts.join(", "))
    }

    /// Where a trigger crosses this, up or down, it is logged.
    pub(super) const EDGE: f32 = 0.1;

    /// Triggers newly past or back under `EDGE` between `before` and `now`:
    /// (left or right, pressed, value).
    pub(super) fn crossings(before: &PadState, now: &PadState) -> Vec<(&'static str, bool, f32)> {
        [("left", before.left_trigger, now.left_trigger), ("right", before.right_trigger, now.right_trigger)]
            .into_iter()
            .filter(|&(_, b, n)| (b >= EDGE) != (n >= EDGE))
            .map(|(side, _, n)| (side, n >= EDGE, n))
            .collect()
    }

    /// Per device: the newest sequence handed over and its triggers (bits).
    static SEEN: [(AtomicU64, AtomicU32, AtomicU32); DEVICES] =
        [const { (AtomicU64::new(0), AtomicU32::new(0), AtomicU32::new(0)) }; DEVICES];

    /// `GetGamepadState` handed over `state`; older readings than one seen
    /// before (the title looks back at the previous one) are skipped.
    pub(super) fn handed(device: usize, sequence: u64, timestamp: u64, state: &PadState) {
        if !ON || device >= DEVICES {
            return;
        }
        let (seen, left, right) = &SEEN[device];
        if seen.fetch_max(sequence, Relaxed) >= sequence {
            return;
        }
        let before = PadState {
            left_trigger: f32::from_bits(left.swap(state.left_trigger.to_bits(), Relaxed)),
            right_trigger: f32::from_bits(right.swap(state.right_trigger.to_bits(), Relaxed)),
            ..Default::default()
        };
        for (side, pressed, value) in crossings(&before, state) {
            let lookup = LOOKUP.with(Cell::get);
            tracing::info!(
                "input probe: {side} trigger {} ({value:.3}) on device {device}, reading {sequence} at {timestamp} us, looked up by {:#x} {}",
                if pressed { "pressed" } else { "released" },
                lookup,
                name(Table::Input, lookup)
            );
        }
    }
}

#[repr(C)]
struct Table<const N: usize>([*const (); N]);
// SAFETY: function addresses only.
unsafe impl<const N: usize> Sync for Table<N> {}

/// The input object; one per hub.
#[repr(C)]
pub struct Object {
    vtable: &'static Table<22>,
    hub: *const Hub,
}

#[repr(C)]
pub struct Device {
    vtable: &'static Table<21>,
    hub: *const Hub,
    index: usize,
    info: DeviceInfo,
}

/// What a device says about itself. The title reads `id` (+0x20); the rest
/// follows the public layout and is zero where we have nothing to say.
#[repr(C)]
#[derive(Default)]
#[allow(dead_code)] // read by title code
struct DeviceInfo {
    size: u32,
    vendor: u16,
    product: u16,
    revision: u16,
    interface: u8,
    collection: u8,
    usage: [u16; 2],
    versions: [u16; 8],
    id: [u8; 32],
    root_id: [u8; 32],
    family: i32,
    capabilities: u32,
    supported_input: u32,
    rumble_motors: u32,
    /// Counts and optional detail pointers.
    rest: [u64; 26],
}

const _: () = assert!(std::mem::offset_of!(DeviceInfo, id) == 0x20);

fn kind(index: usize) -> u32 {
    if index == KEYBOARD {
        KEYBOARD_KIND
    } else {
        GAMEPAD_KIND
    }
}

impl Device {
    fn new(index: usize) -> Device {
        let mut id = [0; 32];
        id[..7].copy_from_slice(b"victory");
        id[31] = index as u8 + 1;
        let info = DeviceInfo {
            size: size_of::<DeviceInfo>() as u32,
            id,
            root_id: id,
            supported_input: kind(index),
            // Low and high frequency.
            rumble_motors: if index <= FALLBACK { 0x3 } else { 0 },
            ..Default::default()
        };
        Device {
            vtable: &DEVICE,
            hub: std::ptr::null(),
            index,
            info,
        }
    }

    fn hub(&self) -> &'static Hub {
        // SAFETY: set by `Hub::new`; hubs are never freed.
        unsafe { &*self.hub }
    }
}

#[derive(PartialEq)]
enum Value {
    Pad(PadState),
    Keys(Vec<KeyState>),
}

#[repr(C)]
pub struct Reading {
    vtable: &'static Table<25>,
    refs: AtomicU32,
    device: *const Device,
    sequence: u64,
    timestamp: u64,
    value: Value,
}

impl Reading {
    fn kind(&self) -> u32 {
        match self.value {
            Value::Pad(_) => GAMEPAD_KIND,
            Value::Keys(_) => KEYBOARD_KIND,
        }
    }
}

/// A counted reference to a reading.
struct Ref(NonNull<Reading>);

impl Ref {
    fn new(reading: Reading) -> Ref {
        Ref(NonNull::from(Box::leak(Box::new(reading))))
    }

    /// Another reference, for the title.
    fn share(&self) -> *const Reading {
        reading_add_ref(self.0.as_ptr());
        self.0.as_ptr()
    }
}

impl Deref for Ref {
    type Target = Reading;
    fn deref(&self) -> &Reading {
        // SAFETY: we hold a reference, so it is alive.
        unsafe { self.0.as_ref() }
    }
}

impl Drop for Ref {
    fn drop(&mut self) {
        reading_release(self.0.as_ptr());
    }
}

type DeviceCallback = extern "system" fn(u64, *mut c_void, *const Device, u64, u32, u32);

#[derive(Clone, Copy)]
struct Registration {
    token: u64,
    /// One device's address, or 0 for any.
    device: usize,
    kind: u32,
    status: u32,
    context: usize,
    callback: DeviceCallback,
}

impl Registration {
    fn matches(&self, devices: &[Device; DEVICES], index: usize) -> bool {
        kind(index) & self.kind != 0
            && (self.device == 0 || self.device == &devices[index] as *const Device as usize)
    }

    fn event(&self, device: usize, timestamp: u64, current: u32, previous: u32) -> Event {
        Event {
            registration: *self,
            device,
            timestamp,
            current,
            previous,
        }
    }
}

struct Event {
    registration: Registration,
    device: usize,
    timestamp: u64,
    current: u32,
    previous: u32,
}

#[derive(Default)]
struct Slot {
    connected: bool,
    sequence: u64,
    history: VecDeque<Ref>,
}

struct State {
    slots: [Slot; DEVICES],
    callbacks: Vec<Registration>,
    /// Asynchronous enumerations, delivered by the next poll.
    pending: Vec<Event>,
    /// Device callbacks running now: token and thread.
    running: Vec<(u64, ThreadId)>,
}

pub struct Hub {
    object: Object,
    devices: [Device; DEVICES],
    /// Apart from the state: XInput can be slow, and the title reads
    /// readings every frame.
    source: Mutex<Box<dyn Source>>,
    /// The title sees one pad ([`one_pad`]).
    one_pad: bool,
    state: Mutex<State>,
    /// A device callback returned.
    returned: Condvar,
}

fn lock<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// SAFETY: the back pointers are set before the hub is shared and never
// change; what changes sits behind the mutex or is atomic.
unsafe impl Send for Hub {}
// SAFETY: as above.
unsafe impl Sync for Hub {}

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Microseconds, the clock of every timestamp we hand out.
fn now() -> u64 {
    START.elapsed().as_micros() as u64
}

impl Hub {
    fn new(source: Box<dyn Source>, one_pad: bool) -> &'static Hub {
        let hub = Box::into_raw(Box::new(Hub {
            object: Object {
                vtable: &INPUT,
                hub: std::ptr::null(),
            },
            devices: std::array::from_fn(Device::new),
            source: Mutex::new(source),
            one_pad,
            state: Mutex::new(State {
                slots: Default::default(),
                callbacks: Vec::new(),
                pending: Vec::new(),
                running: Vec::new(),
            }),
            returned: Condvar::new(),
        }));
        // SAFETY: just allocated and never freed; nothing else sees it yet.
        unsafe {
            (*hub).object.hub = hub;
            for device in &mut (*hub).devices {
                device.hub = hub;
            }
            &*hub
        }
    }

    /// A hub over the host's input, polled from now on.
    fn start(source: Box<dyn Source>) -> &'static Hub {
        let hub = Hub::new(source, one_pad());
        hub.poll();
        std::thread::spawn(move || {
            let mut counted = Default::default();
            let mut reported = Instant::now();
            loop {
                std::thread::sleep(POLL);
                if probe::ON && reported.elapsed() >= probe::REPORT {
                    reported = Instant::now();
                    if let Some(line) = probe::report(&mut counted) {
                        tracing::info!("input probe: calls in the last {}s: {line}", probe::REPORT.as_secs());
                    }
                }
                hub.poll();
            }
        });
        hub
    }

    fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// Reads the host once; device callbacks run on the calling thread.
    fn poll(&self) {
        // The pointer first: the pad recording and the stand-in share one sample.
        let (pads, keys, pointer) = {
            let mut source = lock(&self.source);
            let pointer = source.pointer();
            let pads: [_; PADS] = std::array::from_fn(|slot| source.pad(slot));
            (pads, source.keys(), pointer)
        };
        let events = self.state().poll(&self.devices, self.one_pad, pads, keys, pointer);
        self.deliver(events);
    }

    /// Runs callbacks, never under our lock: they call back into us. One
    /// unregistered since the event was taken does not run, and
    /// `UnregisterCallback` waits for one that runs.
    fn deliver(&self, events: Vec<Event>) {
        let me = std::thread::current().id();
        for e in events {
            let r = e.registration;
            {
                let mut state = self.state();
                if !state.callbacks.iter().any(|c| c.token == r.token) {
                    continue;
                }
                state.running.push((r.token, me));
            }
            (r.callback)(
                r.token,
                r.context as *mut c_void,
                &self.devices[e.device],
                e.timestamp,
                e.current,
                e.previous,
            );
            let mut state = self.state();
            if let Some(i) = state.running.iter().position(|&x| x == (r.token, me)) {
                state.running.swap_remove(i);
            }
            self.returned.notify_all();
        }
    }

    fn index(&self, device: *const Device) -> Option<usize> {
        self.devices.iter().position(|d| std::ptr::eq(d, device))
    }
}

impl State {
    fn poll(
        &mut self,
        devices: &[Device; DEVICES],
        one_pad: bool,
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        pointer: Pointer,
    ) -> Vec<Event> {
        let now = now();
        let mut events = std::mem::take(&mut self.pending);
        if one_pad {
            let pad = pads.into_iter().flatten().fold(fallback(&keys, pointer), merged);
            self.update(devices, FALLBACK, Some(Value::Pad(pad)), now, &mut events);
            self.update(devices, KEYBOARD, Some(Value::Keys(keys)), now, &mut events);
            return events;
        }
        let any_pad = pads.iter().any(Option::is_some);
        for (slot, pad) in pads.into_iter().enumerate() {
            self.update(devices, slot, pad.map(Value::Pad), now, &mut events);
        }
        let fallback = (!any_pad).then(|| Value::Pad(fallback(&keys, pointer)));
        self.update(devices, FALLBACK, fallback, now, &mut events);
        self.update(devices, KEYBOARD, Some(Value::Keys(keys)), now, &mut events);
        events
    }

    /// Records a device's state: a reading when it changed, callbacks when
    /// the device came or went.
    fn update(
        &mut self,
        devices: &[Device; DEVICES],
        index: usize,
        value: Option<Value>,
        now: u64,
        events: &mut Vec<Event>,
    ) {
        let slot = &mut self.slots[index];
        let was = slot.connected;
        slot.connected = value.is_some();
        if let Some(value) = value
            && (!was || slot.history.back().is_none_or(|r| r.value != value))
        {
            slot.sequence += 1;
            if slot.history.len() == HISTORY {
                slot.history.pop_front();
            }
            slot.history.push_back(Ref::new(Reading {
                vtable: &READING,
                refs: AtomicU32::new(1),
                device: &devices[index],
                sequence: slot.sequence,
                timestamp: now,
                value,
            }));
        }
        if was != slot.connected {
            tracing::info!(device = index, connected = !was, "input device");
            let (current, previous) = if was {
                (0, STATUS_CONNECTED)
            } else {
                (STATUS_CONNECTED, 0)
            };
            events.extend(
                self.callbacks
                    .iter()
                    .filter(|r| r.matches(devices, index) && r.status & STATUS_CONNECTED != 0)
                    .map(|r| r.event(index, now, current, previous)),
            );
        }
    }
}

#[cfg(windows)]
fn host() -> Box<dyn Source> {
    Box::new(host::Host::default())
}

#[cfg(not(windows))]
fn host() -> Box<dyn Source> {
    struct Nothing;
    impl Source for Nothing {
        fn pad(&mut self, _: usize) -> Option<PadState> {
            None
        }
        fn keys(&mut self) -> Vec<KeyState> {
            Vec::new()
        }
        fn rumble(&mut self, _: usize, _: u16, _: u16) {}
    }
    Box::new(Nothing)
}

static HUB: LazyLock<&'static Hub> = LazyLock::new(|| Hub::start(host()));

type This = *const Object;
type Out = *mut *const c_void;

/// The DLL's only export (ordinal 1): the input object, static and shared.
pub extern "system" fn GameInputCreate(out: *mut This) -> HRESULT {
    tracing::info!("GameInputCreate");
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes room for a pointer.
    unsafe { out.write(&HUB.object) };
    S_OK
}

fn hub(this: This) -> &'static Hub {
    // SAFETY: `this` is a hub's object; hubs are never freed.
    unsafe { &*(*this).hub }
}

/// Clears a COM out pointer and returns `hr`.
fn fail(out: Out, hr: HRESULT) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(std::ptr::null()) };
    }
    hr
}

/// Gives the title a reference to `reading`, or fails as with none.
fn hand(out: Out, reading: Option<&Ref>) -> HRESULT {
    match reading {
        _ if out.is_null() => E_INVALIDARG,
        Some(reading) => {
            // SAFETY: checked non-null; the title passes room for a pointer.
            unsafe { out.write(reading.share().cast()) };
            S_OK
        }
        None => fail(out, READING_NOT_FOUND),
    }
}

extern "system" fn no_interface(_: *const c_void, _iid: *const c_void, out: Out) -> HRESULT {
    tracing::warn!("input QueryInterface not implemented");
    fail(out, E_NOINTERFACE)
}
/// The input object and devices live as long as the process.
extern "system" fn add_ref(_: *const c_void) -> u32 {
    1
}

extern "system" fn timestamp(_: This) -> u64 {
    probe::call(probe::Table::Input, 0x18, None);
    now()
}

extern "system" fn current_reading(
    this: This,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    probe::call(probe::Table::Input, 0x20, Some(kind));
    let hub = hub(this);
    let state = hub.state();
    let newest = |i: usize| {
        state.slots[i]
            .history
            .back()
            .filter(|r| r.kind() & kind != 0)
    };
    let found = if device.is_null() {
        (0..DEVICES)
            .filter(|&i| state.slots[i].connected)
            .filter_map(newest)
            .max_by_key(|r| r.timestamp)
    } else {
        match hub.index(device) {
            None => return fail(out, E_INVALIDARG),
            Some(i) if !state.slots[i].connected => return fail(out, DEVICE_DISCONNECTED),
            Some(i) => newest(i),
        }
    };
    tracing::trace!(
        kind = format_args!("{kind:#x}"),
        device = ?hub.index(device),
        sequence = found.map(|r| r.sequence),
        "current reading"
    );
    if found.is_none() {
        probe::missed(kind);
    }
    hand(out, found)
}

/// The reading after (`forward`) or before `reference` on its device.
fn relative(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
    forward: bool,
) -> HRESULT {
    if reference.is_null() {
        return fail(out, E_INVALIDARG);
    }
    let hub = hub(this);
    // SAFETY: a reading we handed out and the title still holds.
    let reference = unsafe { &*reference };
    // shortcut: with no device given only the reference's device is walked,
    // not every device's readings in time order; the title walks
    // only keyboard readings (one device) and the current pad's previous one.
    let Some(index) = hub.index(reference.device) else {
        return fail(out, E_INVALIDARG);
    };
    if !device.is_null() && !std::ptr::eq(device, reference.device) {
        return fail(out, E_INVALIDARG);
    }
    let state = hub.state();
    let history = &state.slots[index].history;
    if history
        .front()
        .is_none_or(|oldest| reference.sequence < oldest.sequence)
    {
        return fail(out, REFERENCE_TOO_OLD);
    }
    let wanted = |r: &&Ref| r.kind() & kind != 0;
    let found = if forward {
        history
            .iter()
            .filter(|r| r.sequence > reference.sequence)
            .find(wanted)
    } else {
        history
            .iter()
            .rev()
            .filter(|r| r.sequence < reference.sequence)
            .find(wanted)
    };
    hand(out, found)
}
extern "system" fn next_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    probe::call(probe::Table::Input, 0x28, Some(kind));
    relative(this, reference, kind, device, out, true)
}
extern "system" fn previous_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    probe::call(probe::Table::Input, 0x30, Some(kind));
    relative(this, reference, kind, device, out, false)
}
/// The device's reading at `time`, among those still kept.
extern "system" fn temporal_reading(
    this: This,
    time: u64,
    device: *const Device,
    out: Out,
) -> HRESULT {
    probe::call(probe::Table::Input, 0x38, None);
    let hub = hub(this);
    let Some(index) = hub.index(device) else {
        return fail(out, E_INVALIDARG);
    };
    let state = hub.state();
    let history = &state.slots[index].history;
    hand(out, history.iter().rev().find(|r| r.timestamp <= time))
}

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

fn token(out: *mut u64) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(NEXT_TOKEN.fetch_add(1, Ordering::Relaxed)) };
    }
    S_OK
}

// shortcut: reading callbacks never run; the title registers none
// (it polls).
extern "system" fn register_reading(
    _: This,
    _device: *const c_void,
    kind: u32,
    _threshold: f32,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    probe::call(probe::Table::Input, 0x40, Some(kind));
    tracing::warn!("input reading callbacks are never called");
    token(out)
}

/// Device callbacks run on our poll thread when a device comes or goes.
/// Enumeration reports the devices already there: blocking on the caller's
/// thread before we return (the title registers holding a lock its callback
/// takes again), asynchronous on the next poll.
extern "system" fn register_device(
    this: This,
    device: *const Device,
    kind: u32,
    status: u32,
    enumeration: u32,
    context: *mut c_void,
    callback: Option<DeviceCallback>,
    out: *mut u64,
) -> HRESULT {
    probe::call(probe::Table::Input, 0x48, Some(kind));
    let Some(callback) = callback else {
        return E_INVALIDARG;
    };
    let hub = hub(this);
    let registration = Registration {
        token: NEXT_TOKEN.fetch_add(1, Ordering::Relaxed),
        device: device as usize,
        kind,
        status,
        context: context as usize,
        callback,
    };
    let mut found = Vec::new();
    {
        let mut state = hub.state();
        if enumeration != 0 && status & STATUS_CONNECTED != 0 {
            let now = now();
            found = (0..DEVICES)
                .filter(|&i| state.slots[i].connected && registration.matches(&hub.devices, i))
                .map(|i| registration.event(i, now, STATUS_CONNECTED, 0))
                .collect();
        }
        tracing::info!(
            kind = format_args!("{kind:#x}"),
            enumeration,
            devices = found.len(),
            "input device callback"
        );
        state.callbacks.push(registration);
        if enumeration == ASYNC_ENUMERATION {
            state.pending.append(&mut found);
        }
    }
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(registration.token) };
    }
    hub.deliver(found);
    S_OK
}
/// Guide button and keyboard layout callbacks: neither ever changes here.
extern "system" fn register_other<const SLOT: usize>(
    _: This,
    _device: *const c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    probe::call(probe::Table::Input, SLOT, None);
    token(out)
}

fn forget(this: This, token: u64) {
    let mut state = hub(this).state();
    state.callbacks.retain(|r| r.token != token);
    state.pending.retain(|e| e.registration.token != token);
}
extern "system" fn stop_callback(this: This, token: u64) {
    probe::call(probe::Table::Input, 0x60, None);
    forget(this, token);
}
/// Stops the callback, then waits up to `timeout` µs for a call of it that
/// is running; false on time-out, or from inside that callback (it cannot
/// wait for itself), as documented. Either way it is never called again.
extern "system" fn unregister_callback(this: This, token: u64, timeout: u64) -> bool {
    probe::call(probe::Table::Input, 0x68, None);
    forget(this, token);
    let hub = hub(this);
    let me = std::thread::current().id();
    let deadline = Instant::now().checked_add(Duration::from_micros(timeout));
    let mut state = hub.state();
    loop {
        let running = state.running.iter().filter(|r| r.0 == token);
        if running.clone().any(|r| r.1 == me) {
            return false;
        }
        if running.count() == 0 {
            return true;
        }
        state = match deadline {
            None => hub.returned.wait(state).unwrap_or_else(PoisonError::into_inner),
            Some(deadline) => {
                let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                    return false;
                };
                let waited = hub.returned.wait_timeout(state, left);
                waited.unwrap_or_else(PoisonError::into_inner).0
            }
        };
    }
}

/// Dispatcher and aggregate device: no title of ours asks for them.
extern "system" fn create_dispatcher(_: This, out: Out) -> HRESULT {
    probe::call(probe::Table::Input, 0x70, None);
    tracing::warn!("input dispatcher not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn create_aggregate(_: This, kind: u32, out: Out) -> HRESULT {
    probe::call(probe::Table::Input, 0x78, Some(kind));
    tracing::warn!("aggregate input device not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn find_by_id(this: This, id: *const [u8; 32], out: Out) -> HRESULT {
    probe::call(probe::Table::Input, 0x80, None);
    if id.is_null() || out.is_null() {
        return fail(out, E_INVALIDARG);
    }
    // SAFETY: checked non-null; the title passes a device id.
    let id = unsafe { &*id };
    match hub(this).devices.iter().find(|d| d.info.id == *id) {
        Some(device) => {
            // SAFETY: checked non-null.
            unsafe { out.write((device as *const Device).cast()) };
            S_OK
        }
        None => fail(out, DEVICE_NOT_FOUND),
    }
}
/// Lookup by object, platform handle or platform string: our devices have
/// none of these.
extern "system" fn find_device<const SLOT: usize>(_: This, _value: *const c_void, out: Out) -> HRESULT {
    probe::call(probe::Table::Input, SLOT, None);
    fail(out, DEVICE_NOT_FOUND)
}
extern "system" fn enable_oem_device(
    _: This,
    _vendor: u16,
    _product: u16,
    _interface: u8,
    _collection: u8,
) -> HRESULT {
    probe::call(probe::Table::Input, 0xa0, None);
    S_OK
}
extern "system" fn set_focus_policy(_: This, _policy: u32) {
    probe::call(probe::Table::Input, 0xa8, None);
}

static INPUT: Table<22> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    timestamp as *const (),           // 0x18
    current_reading as *const (),     // 0x20
    next_reading as *const (),        // 0x28
    previous_reading as *const (),    // 0x30
    temporal_reading as *const (),    // 0x38
    register_reading as *const (),    // 0x40
    register_device as *const (),     // 0x48
    register_other::<0x50> as *const (), // 0x50 guide button
    register_other::<0x58> as *const (), // 0x58 keyboard layout
    stop_callback as *const (),       // 0x60
    unregister_callback as *const (), // 0x68
    create_dispatcher as *const (),   // 0x70
    create_aggregate as *const (),    // 0x78
    find_by_id as *const (),          // 0x80
    find_device::<0x88> as *const (), // 0x88 by object
    find_device::<0x90> as *const (), // 0x90 by platform handle
    find_device::<0x98> as *const (), // 0x98 by platform string
    enable_oem_device as *const (),   // 0xa0
    set_focus_policy as *const (),    // 0xa8
]);

extern "system" fn device_info(this: *const Device) -> *const DeviceInfo {
    // SAFETY: one of our devices.
    unsafe { &(*this).info }
}
extern "system" fn device_status(this: *const Device) -> u32 {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    if device.hub().state().slots[device.index].connected {
        STATUS_CONNECTED
    } else {
        0
    }
}
// shortcut: trigger motors are dropped (XInput has none). The host API that
// drives them without a redistributable is Windows.Gaming.Input (WinRT
// `Gamepad.Vibration`), which needs WinRT bindings this crate lacks outside
// diagnostics builds; the title sends trigger levels in slot 0x50.
extern "system" fn set_rumble(this: *const Device, params: *const [f32; 4]) {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    let one_pad = device.hub().one_pad;
    if device.index >= PADS && !(one_pad && device.index == FALLBACK) {
        return;
    }
    let [low, high, ..] = if params.is_null() {
        [0.0; 4]
    } else {
        // SAFETY: checked non-null; four motor levels.
        unsafe { *params }
    };
    let speed = |level: f32| (level.clamp(0.0, 1.0) * f32::from(u16::MAX)) as u16;
    let mut source = lock(&device.hub().source);
    if one_pad {
        // The one pad is every host pad.
        for slot in 0..PADS {
            if source.pad(slot).is_some() {
                source.rumble(slot, speed(low), speed(high));
            }
        }
    } else {
        source.rumble(device.index, speed(low), speed(high));
    }
}
extern "system" fn unsupported<const SLOT: usize>(_: *const c_void) -> HRESULT {
    tracing::warn!(
        slot = format_args!("{SLOT:#x}"),
        "input device method not implemented"
    );
    E_NOTIMPL
}

static DEVICE: Table<21> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    device_info as *const (),         // 0x18
    device_status as *const (),       // 0x20
    unsupported::<0x28> as *const (), // battery
    unsupported::<0x30> as *const (), // force feedback
    unsupported::<0x38> as *const (), //
    unsupported::<0x40> as *const (), //
    unsupported::<0x48> as *const (), // haptics
    set_rumble as *const (),          // 0x50
    unsupported::<0x58> as *const (), // input synchronization
    unsupported::<0x60> as *const (), //
    unsupported::<0x68> as *const (), // power off
    unsupported::<0x70> as *const (), // raw device access
    unsupported::<0x78> as *const (), //
    unsupported::<0x80> as *const (), //
    unsupported::<0x88> as *const (), //
    unsupported::<0x90> as *const (), //
    unsupported::<0x98> as *const (), //
    unsupported::<0xa0> as *const (), //
]);

fn reading<'a>(this: *const Reading) -> &'a Reading {
    // SAFETY: a reading we handed out and the title still holds.
    unsafe { &*this }
}
extern "system" fn reading_add_ref(this: *const Reading) -> u32 {
    reading(this).refs.fetch_add(1, Ordering::Relaxed) + 1
}
extern "system" fn reading_release(this: *const Reading) -> u32 {
    let left = reading(this).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        // SAFETY: the last reference is gone; `Ref::new` boxed it.
        drop(unsafe { Box::from_raw(this.cast_mut()) });
    }
    left
}
extern "system" fn input_kind(this: *const Reading) -> u32 {
    probe::call(probe::Table::Reading, 0x18, None);
    reading(this).kind()
}
extern "system" fn sequence_number(this: *const Reading, kind: u32) -> u64 {
    probe::call(probe::Table::Reading, 0x20, Some(kind));
    let reading = reading(this);
    if reading.kind() & kind != 0 {
        reading.sequence
    } else {
        0
    }
}
extern "system" fn reading_timestamp(this: *const Reading) -> u64 {
    probe::call(probe::Table::Reading, 0x28, None);
    reading(this).timestamp
}
extern "system" fn reading_device(this: *const Reading, out: *mut *const Device) {
    probe::call(probe::Table::Reading, 0x30, None);
    if !out.is_null() {
        // SAFETY: checked non-null; devices need no reference.
        unsafe { out.write(reading(this).device) };
    }
}
extern "system" fn key_count(this: *const Reading) -> u32 {
    probe::call(probe::Table::Reading, 0x70, None);
    match &reading(this).value {
        Value::Keys(keys) => keys.len() as u32,
        Value::Pad(_) => 0,
    }
}
extern "system" fn key_state(this: *const Reading, count: u32, out: *mut KeyState) -> u32 {
    probe::call(probe::Table::Reading, 0x78, None);
    let Value::Keys(keys) = &reading(this).value else {
        return 0;
    };
    if out.is_null() {
        return 0;
    }
    let n = keys.len().min(count as usize);
    // SAFETY: the title passes room for `count` keys.
    unsafe { std::ptr::copy_nonoverlapping(keys.as_ptr(), out, n) };
    n as u32
}
extern "system" fn gamepad_state(this: *const Reading, out: *mut PadState) -> bool {
    probe::call(probe::Table::Reading, 0xb0, None);
    let reading = reading(this);
    match &reading.value {
        Value::Pad(state) if !out.is_null() => {
            // SAFETY: checked non-null.
            unsafe { out.write(*state) };
            if probe::ON {
                // SAFETY: one of our devices.
                let device = unsafe { (*reading.device).index };
                probe::handed(device, reading.sequence, reading.timestamp, state);
            }
            true
        }
        _ => false,
    }
}
/// Kinds our devices do not have: no count, no state.
extern "system" fn nothing<const SLOT: usize>(_: *const c_void) -> usize {
    probe::call(probe::Table::Reading, SLOT, None);
    0
}

static READING: Table<25> = Table([
    no_interface as *const (),      // 0x00
    reading_add_ref as *const (),   // 0x08
    reading_release as *const (),   // 0x10
    input_kind as *const (),        // 0x18
    sequence_number as *const (),   // 0x20
    reading_timestamp as *const (), // 0x28
    reading_device as *const (),    // 0x30
    nothing::<0x38> as *const (), // 0x38 raw report
    nothing::<0x40> as *const (), // 0x40 controller axes, buttons, switches
    nothing::<0x48> as *const (), // 0x48
    nothing::<0x50> as *const (), // 0x50
    nothing::<0x58> as *const (), // 0x58
    nothing::<0x60> as *const (), // 0x60
    nothing::<0x68> as *const (), // 0x68
    key_count as *const (),         // 0x70
    key_state as *const (),         // 0x78
    nothing::<0x80> as *const (), // 0x80 mouse
    nothing::<0x88> as *const (), // 0x88 touch
    nothing::<0x90> as *const (), // 0x90
    nothing::<0x98> as *const (), // 0x98 motion
    nothing::<0xa0> as *const (), // 0xa0 arcade stick
    nothing::<0xa8> as *const (), // 0xa8 flight stick
    gamepad_state as *const (),     // 0xb0
    nothing::<0xb8> as *const (), // 0xb8 racing wheel
    nothing::<0xc0> as *const (), // 0xc0 UI navigation
]);

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "gameinput",
        dlls: &["gameinput.dll"],
        // Titles import it by ordinal only.
        exports: vec![HleExport {
            name: "#1",
            address: GameInputCreate as *const () as usize,
        }],
    }
}

/// Host input: XInput pads, the keyboard while our process has focus.
#[cfg(windows)]
mod host {
    use std::time::{Duration, Instant};

    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, MAPVK_VK_TO_VSC_EX, MapVirtualKeyW,
    };
    use windows_sys::Win32::UI::Input::XboxController::{
        XINPUT_GAMEPAD, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_BACK,
        XINPUT_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT,
        XINPUT_GAMEPAD_DPAD_UP, XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB,
        XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_START,
        XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y, XINPUT_STATE, XINPUT_VIBRATION, XInputGetState,
        XInputSetState,
    };
    use windows_sys::Win32::Foundation::{HWND, POINT, RECT};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        ClipCursor, GetCursorPos, GetForegroundWindow, GetWindowRect, GetWindowThreadProcessId,
        SetCursorPos,
    };

    use super::*;

    /// `VICTORY_MOUSE=1`: the mouse turns the camera, its buttons are the
    /// triggers and the keys are [`MOUSE_BUTTONS`] (off by default: the
    /// mouse is left alone and the keys are as they were).
    fn mouse_on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_MOUSE").is_some_and(|v| v == "1"))
    }

    /// `VICTORY_MOUSE_SPEED`: the mouse speed, in pixels a second, that is a
    /// full stick (1400; smaller turns faster). `VICTORY_MOUSE_DEAD`: the
    /// share of the stick the title ignores around its middle (0.12).
    fn mouse_set() -> (f32, f32) {
        static SET: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
        *SET.get_or_init(|| {
            let number = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<f32>().ok());
            (
                number("VICTORY_MOUSE_SPEED").filter(|v| *v >= 100.0).unwrap_or(1400.0),
                number("VICTORY_MOUSE_DEAD").filter(|v| (0.0..0.9).contains(v)).unwrap_or(0.12),
            )
        })
    }

    /// What the mouse has been doing, between polls.
    #[derive(Default)]
    struct Mouse {
        look: Look,
        /// The window the pointer is held in, and when it was last read.
        held: Option<(usize, Instant)>,
        /// F8 lets the pointer go and takes it again.
        free: bool,
        drive: bool,
        /// Caps Lock and F8 were down at the last poll.
        was: (bool, bool),
    }

    const CAPS_LOCK: u8 = 0x14;
    const F8: u8 = 0x77;

    /// The foreground window if it is one of ours (the title's or our
    /// fallback).
    fn ours() -> Option<HWND> {
        let mut process = 0;
        // SAFETY: plain queries; a null window gives process 0.
        unsafe {
            let window = GetForegroundWindow();
            GetWindowThreadProcessId(window, &mut process);
            (process == GetCurrentProcessId() && !window.is_null()).then_some(window)
        }
    }

    impl Mouse {
        fn let_go(&mut self) {
            if self.held.take().is_some() {
                // SAFETY: lifts the limit this process set.
                unsafe { ClipCursor(std::ptr::null()) };
                MOUSE_HELD.store(false, Ordering::Relaxed);
                self.look = Look::default();
            }
        }

        /// Caps Lock is `down` now: a press switches the keys to driving
        /// and back.
        fn caps(&mut self, down: bool) {
            if down && !self.was.0 {
                self.drive = !self.drive;
                tracing::info!(driving = self.drive, "keys: Caps Lock switched the layout");
            }
            self.was.0 = down;
        }

        /// A replay's: the layout alone, switched by the replayed `keys`.
        /// The recorded mouse is in the replay's pad 0 ([`recorded_pad`]),
        /// and the host's is not read.
        fn replayed(&mut self, keys: &[u8]) -> Pointer {
            self.caps(keys.contains(&CAPS_LOCK));
            Pointer { on: true, drive: self.drive, ..Default::default() }
        }

        fn poll(&mut self) -> Pointer {
            // SAFETY: plain queries.
            let down = |vk: u8| unsafe { GetAsyncKeyState(i32::from(vk)) } < 0;
            let Some(window) = ours() else {
                self.let_go();
                self.was = (false, false);
                return Pointer { on: true, drive: self.drive, ..Default::default() };
            };
            self.caps(down(CAPS_LOCK));
            let f8 = down(F8);
            if f8 && !self.was.1 {
                self.free = !self.free;
                tracing::info!(free = self.free, "mouse: F8");
            }
            self.was.1 = f8;
            let mut rect = RECT::default();
            // SAFETY: room for the rectangle; a window of ours.
            if self.free || unsafe { GetWindowRect(window, &mut rect) } == 0 {
                self.let_go();
                return Pointer { on: true, drive: self.drive, ..Default::default() };
            }
            // Away from the frame: a click during a fast movement must not
            // land on the title bar or the border.
            let inset = 48_i32.min((rect.right - rect.left) / 4).min((rect.bottom - rect.top) / 4);
            let inner = RECT {
                left: rect.left + inset,
                top: rect.top + inset,
                right: rect.right - inset,
                bottom: rect.bottom - inset,
            };
            let centre = POINT { x: (rect.left + rect.right) / 2, y: (rect.top + rect.bottom) / 2 };
            let mut at = POINT::default();
            let time = Instant::now();
            // SAFETY: room for the point; our own limit and position.
            let moved = unsafe {
                let read = GetCursorPos(&mut at) != 0;
                ClipCursor(&inner);
                SetCursorPos(centre.x, centre.y);
                read
            };
            let (speed, dead) = mouse_set();
            let stick = match self.held.replace((window as usize, time)) {
                // The first look in this window: the pointer was anywhere.
                Some((was, then)) if was == window as usize && moved => {
                    let travel = ((at.x - centre.x) as f32, (at.y - centre.y) as f32);
                    self.look.turn(travel, (time - then).as_secs_f32(), speed, dead)
                }
                _ => (0.0, 0.0),
            };
            MOUSE_HELD.store(true, Ordering::Relaxed);
            Pointer {
                on: true,
                right_x: stick.0,
                right_y: stick.1,
                fire: down(0x01),
                aim: down(0x02),
                drive: self.drive,
            }
        }
    }

    /// Asking XInput about an empty slot is slow; look again this often.
    const EMPTY_SLOT_RECHECK: Duration = Duration::from_secs(1);

    const BUTTONS: [(u16, u32); 14] = [
        (XINPUT_GAMEPAD_START, MENU),
        (XINPUT_GAMEPAD_BACK, VIEW),
        (XINPUT_GAMEPAD_A, A),
        (XINPUT_GAMEPAD_B, B),
        (XINPUT_GAMEPAD_X, X),
        (XINPUT_GAMEPAD_Y, Y),
        (XINPUT_GAMEPAD_DPAD_UP, UP),
        (XINPUT_GAMEPAD_DPAD_DOWN, DOWN),
        (XINPUT_GAMEPAD_DPAD_LEFT, LEFT),
        (XINPUT_GAMEPAD_DPAD_RIGHT, RIGHT),
        (XINPUT_GAMEPAD_LEFT_SHOULDER, LEFT_SHOULDER),
        (XINPUT_GAMEPAD_RIGHT_SHOULDER, RIGHT_SHOULDER),
        (XINPUT_GAMEPAD_LEFT_THUMB, LEFT_STICK),
        (XINPUT_GAMEPAD_RIGHT_THUMB, RIGHT_STICK),
    ];

    /// Raw values, no dead zone: titles apply their own.
    pub(super) fn convert(pad: &XINPUT_GAMEPAD) -> PadState {
        let stick = |v: i16| (f32::from(v) / 32767.0).max(-1.0);
        PadState {
            buttons: BUTTONS
                .iter()
                .filter(|(host, _)| pad.wButtons & host != 0)
                .fold(0, |all, (_, bit)| all | bit),
            left_trigger: f32::from(pad.bLeftTrigger) / 255.0,
            right_trigger: f32::from(pad.bRightTrigger) / 255.0,
            left_x: stick(pad.sThumbLX),
            left_y: stick(pad.sThumbLY),
            right_x: stick(pad.sThumbRX),
            right_y: stick(pad.sThumbRY),
        }
    }

    #[derive(Default)]
    pub(super) struct Host {
        empty_until: [Option<Instant>; PADS],
        /// `VICTORY_INPUT_RECORD` and `VICTORY_INPUT_REPLAY`, set up at the
        /// first poll (their clocks start there).
        taps: Option<Taps>,
        mouse: Mouse,
        /// This poll's mouse (`Hub::poll` reads it first), for the recording.
        pointer: Pointer,
    }

    type Taps = (Option<record::Recorder>, Option<record::Replay>);

    fn taps(taps: &mut Option<Taps>) -> &mut Taps {
        taps.get_or_insert_with(|| (record::Recorder::from_env(), record::Replay::from_env()))
    }

    fn raw(pad: &XINPUT_GAMEPAD) -> record::Raw {
        record::Raw {
            buttons: pad.wButtons,
            lt: pad.bLeftTrigger,
            rt: pad.bRightTrigger,
            lx: pad.sThumbLX,
            ly: pad.sThumbLY,
            rx: pad.sThumbRX,
            ry: pad.sThumbRY,
        }
    }

    fn gamepad(raw: record::Raw) -> XINPUT_GAMEPAD {
        XINPUT_GAMEPAD {
            wButtons: raw.buttons,
            bLeftTrigger: raw.lt,
            bRightTrigger: raw.rt,
            sThumbLX: raw.lx,
            sThumbLY: raw.ly,
            sThumbRX: raw.rx,
            sThumbRY: raw.ry,
        }
    }

    impl Source for Host {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            let (recorder, replay) = taps(&mut self.taps);
            // A replay is pad 0 (host slot 0, merged with the stand-in like
            // any host pad); the host's pads and VICTORY_KEYS are not read.
            if let Some(replay) = replay {
                return if slot == 0 { replay.pad().map(|r| convert(&gamepad(r))) } else { None };
            }
            let recorder = recorder.as_mut().filter(|_| slot == 0);
            let host = if self.empty_until[slot].is_some_and(|until| Instant::now() < until) {
                None
            } else {
                let mut state = XINPUT_STATE::default();
                // SAFETY: room for the state.
                if unsafe { XInputGetState(slot as u32, &mut state) } != 0 {
                    self.empty_until[slot] = Some(Instant::now() + EMPTY_SLOT_RECHECK);
                    None
                } else {
                    self.empty_until[slot] = None;
                    Some(state.Gamepad)
                }
            };
            // Every poll, so the mouse is recorded while no pad is there
            // too (the recorder writes changes only).
            if let Some(r) = recorder {
                r.pad(recorded_pad(host.as_ref().map(raw), self.pointer));
            }
            let mut pad = convert(&host?);
            // The scripted keys (`VICTORY_KEYS`) also press the first
            // pad's buttons, by the stand-in pad's table: with a real pad
            // connected the stand-in is not there, and a title that listens
            // to the pad (the title's start screen) saw nothing.
            if slot == 0 {
                for vk in scripted() {
                    if let Some(&(_, button)) = super::FALLBACK_BUTTONS.iter().find(|(key, _)| *key == vk) {
                        pad.buttons |= button;
                    }
                }
            }
            Some(pad)
        }

        fn pointer(&mut self) -> Pointer {
            if !mouse_on() {
                return Pointer::default();
            }
            self.pointer = match &taps(&mut self.taps).1 {
                Some(replay) => self.mouse.replayed(&replay.keys()),
                None => self.mouse.poll(),
            };
            self.pointer
        }

        fn keys(&mut self) -> Vec<KeyState> {
            let replaying = taps(&mut self.taps).1.is_some();
            let mut down = if replaying { Vec::new() } else { scripted() };
            let held: Vec<u8> = if focused() {
                // Left/right modifier keys only, as devices report them; no
                // mouse buttons (below 0x08).
                (0x08..=0xFE_u8)
                    .filter(|vk| !(0x10..=0x12).contains(vk) && !down.contains(vk))
                    // SAFETY: plain queries.
                    .filter(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) } < 0)
                    .collect()
            } else {
                Vec::new()
            };
            match taps(&mut self.taps) {
                // The recording is the whole keyboard.
                (_, Some(replay)) => down = replay.keys(),
                (recorder, None) => {
                    if let Some(r) = recorder {
                        r.keys(&held);
                    }
                    down.extend(held);
                }
            }
            down.into_iter()
                .map(|vk| KeyState {
                    // SAFETY: plain query.
                    scan_code: unsafe { MapVirtualKeyW(u32::from(vk), MAPVK_VK_TO_VSC_EX) },
                    virtual_key: vk,
                    // shortcut: no characters (code point 0);
                    // the title reads scan codes and virtual keys only.
                    ..Default::default()
                })
                .collect()
        }

        fn rumble(&mut self, slot: usize, low: u16, high: u16) {
            let speeds = XINPUT_VIBRATION {
                wLeftMotorSpeed: low,
                wRightMotorSpeed: high,
            };
            // SAFETY: a valid vibration; an empty slot just fails.
            unsafe { XInputSetState(slot as u32, &speeds) };
        }
    }

    /// Debugging aid: `VICTORY_KEYS=<seconds>:<key>,...` holds each key for
    /// 0.2 s that long after input starts (`<from>-<to>:<key>`: all that
    /// time), with or without focus. A key is a
    /// letter, a digit, or `Enter`, `Esc`, `Tab`, `Back`, `Up`, `Down`,
    /// `Left`, `Right` (the stand-in pad's table maps them to buttons; W/A/S/D
    /// move the left stick, I/J/K/L the right).
    ///
    /// `VICTORY_KEYS_FILE=<path>` also presses the keys on each line of that
    /// file, read while the title runs, so a script can wait for the title
    /// (a screen, the world) and then append keys or a whole route (after
    /// sinan-cmyk's jman 05df1eaf). See [`KeyFeed`] for the lines.
    fn scripted() -> Vec<u8> {
        static SCRIPT: LazyLock<(Instant, Vec<(Duration, Duration, u8)>)> = LazyLock::new(|| {
            let list = std::env::var("VICTORY_KEYS").unwrap_or_default();
            let keys = list.split(',').filter_map(|entry| {
                let (at, key) = entry.split_once(':')?;
                let vk = key_code(key.trim())?;
                let time = |t: &str| Duration::try_from_secs_f64(t.trim().parse().ok()?).ok();
                let (from, to) = match at.split_once('-') {
                    Some((from, to)) => (time(from)?, time(to)?),
                    None => (time(at)?, time(at)? + Duration::from_millis(200)),
                };
                Some((from, to, vk))
            });
            (Instant::now(), keys.collect())
        });
        let (start, keys) = &*SCRIPT;
        let now = start.elapsed();
        let mut down: Vec<u8> = keys
            .iter()
            .filter(|&&(from, to, _)| (from..to).contains(&now))
            .map(|&(_, _, vk)| vk)
            .collect();
        for vk in key_file() {
            if !down.contains(&vk) {
                down.push(vk);
            }
        }
        down
    }

    /// How long a scripted key stays down unless its line says.
    const KEY_TAP: Duration = Duration::from_millis(200);
    /// The pause after each held line of `VICTORY_KEYS_FILE`, so the next
    /// reads as a separate press.
    const KEY_GAP: Duration = Duration::from_millis(200);

    /// Without diagnostics there is no key file.
    #[cfg(not(feature = "diagnostics"))]
    fn key_file() -> Vec<u8> {
        Vec::new()
    }

    /// The keys `VICTORY_KEYS_FILE` holds down now, reading the file at most
    /// every 0.1 s.
    #[cfg(feature = "diagnostics")]
    fn key_file() -> Vec<u8> {
        struct File {
            path: Option<std::path::PathBuf>,
            feed: KeyFeed,
            read_at: Option<Instant>,
        }
        static FILE: LazyLock<std::sync::Mutex<File>> = LazyLock::new(|| {
            std::sync::Mutex::new(File {
                path: std::env::var_os("VICTORY_KEYS_FILE").map(Into::into),
                feed: KeyFeed::default(),
                read_at: None,
            })
        });
        let mut file = FILE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(path) = file.path.clone() else {
            return Vec::new();
        };
        let now = Instant::now();
        if file.read_at.is_none_or(|at| now - at >= Duration::from_millis(100)) {
            file.read_at = Some(now);
            if let Ok(text) = std::fs::read_to_string(&path) {
                file.feed.feed(&text);
            }
        }
        file.feed.keys(now)
    }

    /// One line of a key file: keys held together for a time (none: a wait).
    #[derive(Clone, Debug, PartialEq)]
    struct Step {
        keys: Vec<u8>,
        hold: Duration,
    }

    /// Lines of a growing text, played one after another. A line is
    /// `<keys> [seconds]`: key names joined by `+` (`W`, `W+L`, `Enter`),
    /// held together that long (0.2 s when left out), then 0.2 s with
    /// nothing down; or `wait <seconds>`. `#` starts a comment. A line
    /// still being written waits for its newline. Each step starts where
    /// the one before ended, not where it was next looked at, so a route
    /// keeps its timing however often the title polls.
    #[derive(Default)]
    struct KeyFeed {
        /// Bytes of the text already read.
        read: usize,
        queue: std::collections::VecDeque<Step>,
        /// The step playing and when it began.
        playing: Option<(Instant, Step)>,
    }

    impl KeyFeed {
        fn feed(&mut self, text: &str) {
            let Some(new) = text.get(self.read..) else {
                return;
            };
            let whole = new.rfind('\n').map_or(0, |i| i + 1);
            self.read += whole;
            for line in new[..whole].lines() {
                let line = line.split('#').next().unwrap_or("").trim();
                if line.is_empty() {
                    continue;
                }
                match step(line) {
                    Some(s) => {
                        tracing::info!(line, "VICTORY_KEYS_FILE step queued");
                        self.queue.push_back(s);
                    }
                    None => tracing::warn!(line, "VICTORY_KEYS_FILE line not understood"),
                }
            }
        }

        /// The keys down at `now`.
        fn keys(&mut self, now: Instant) -> Vec<u8> {
            loop {
                let Some((at, s)) = &self.playing else {
                    let Some(s) = self.queue.pop_front() else {
                        return Vec::new();
                    };
                    self.playing = Some((now, s));
                    continue;
                };
                let since = now.saturating_duration_since(*at);
                if since < s.hold {
                    return s.keys.clone();
                }
                let gap = if s.keys.is_empty() { Duration::ZERO } else { KEY_GAP };
                if since < s.hold + gap {
                    return Vec::new();
                }
                let end = *at + s.hold + gap;
                self.playing = None;
                let Some(next) = self.queue.pop_front() else {
                    return Vec::new();
                };
                self.playing = Some((end, next));
            }
        }
    }

    /// A key file line as a step.
    fn step(line: &str) -> Option<Step> {
        let mut words = line.split_whitespace();
        let first = words.next()?;
        let secs = |w: &str| Duration::try_from_secs_f64(w.parse().ok()?).ok();
        let s = if first.eq_ignore_ascii_case("wait") {
            Step { keys: Vec::new(), hold: secs(words.next()?)? }
        } else {
            let keys = first.split('+').map(key_code).collect::<Option<Vec<u8>>>()?;
            let hold = match words.next() {
                Some(w) => secs(w)?,
                None => KEY_TAP,
            };
            Step { keys, hold }
        };
        words.next().is_none().then_some(s)
    }

    /// The virtual key a scripted key name stands for.
    fn key_code(name: &str) -> Option<u8> {
        Some(match name {
            "Enter" => 0x0D,
            "Esc" => 0x1B,
            "Tab" => 0x09,
            "Back" => 0x08,
            "Up" => 0x26,
            "Down" => 0x28,
            "Left" => 0x25,
            "Right" => 0x27,
            k if k.len() == 1 && k.as_bytes()[0].is_ascii_alphanumeric() => k.as_bytes()[0].to_ascii_uppercase(),
            _ => return None,
        })
    }

    #[test]
    fn key_file_plays_holds_chords_and_waits_in_order() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let mut feed = KeyFeed::default();
        // the last line has no newline yet; an unknown name is skipped
        let mut text = String::from("# route\nW 1.0\nNope\nwait 0.5\nW+L 0.4  # turn\nEnt");
        feed.feed(&text);
        assert_eq!(feed.keys(at(0)), vec![b'W']);
        assert_eq!(feed.keys(at(999)), vec![b'W']);
        assert_eq!(feed.keys(at(1000)), Vec::<u8>::new(), "the gap after a hold");
        assert_eq!(feed.keys(at(1250)), Vec::<u8>::new(), "the wait");
        // polled late: the step still starts where the wait ended (1.7 s)
        assert_eq!(feed.keys(at(1900)), vec![b'W', b'L']);
        assert_eq!(feed.keys(at(2099)), vec![b'W', b'L']);
        assert_eq!(feed.keys(at(2100)), Vec::<u8>::new());
        text.push_str("er\n");
        feed.feed(&text);
        assert_eq!(feed.keys(at(2400)), vec![0x0D], "a tap, once its line is whole");
        assert_eq!(feed.keys(at(2650)), Vec::<u8>::new());
        assert_eq!(feed.keys(at(5000)), Vec::<u8>::new(), "nothing queued");
        assert_eq!(step("wait x"), None);
        assert_eq!(step("W 1 2"), None);
        assert_eq!(step("W+J 2.5"), Some(Step { keys: vec![b'W', b'J'], hold: Duration::from_millis(2500) }));
    }

    fn focused() -> bool {
        ours().is_some()
    }

    #[test]
    fn host_pads_convert_to_console_pads() {
        let pad = XINPUT_GAMEPAD {
            wButtons: XINPUT_GAMEPAD_A | XINPUT_GAMEPAD_START | XINPUT_GAMEPAD_DPAD_LEFT,
            bLeftTrigger: 255,
            bRightTrigger: 51,
            sThumbLX: -32768,
            sThumbLY: 32767,
            ..Default::default()
        };
        let state = convert(&pad);
        assert_eq!(state.buttons, A | MENU | LEFT);
        assert_eq!(
            (
                state.left_trigger,
                state.left_x,
                state.left_y,
                state.right_x
            ),
            (1.0, -1.0, 1.0, 0.0)
        );
        assert!((state.right_trigger - 0.2).abs() < 1e-6);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::HleResolver;
    use crate::HostPolicy;
    use victory_loader::Provenance;
    use victory_pe::ImportSymbol;

    #[derive(Default)]
    struct FakeHost {
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        rumble: Vec<(usize, u16, u16)>,
    }

    struct Fake(Arc<Mutex<FakeHost>>);

    impl Source for Fake {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            self.0.lock().unwrap().pads[slot]
        }
        fn keys(&mut self) -> Vec<KeyState> {
            self.0.lock().unwrap().keys.clone()
        }
        fn rumble(&mut self, slot: usize, low: u16, high: u16) {
            self.0.lock().unwrap().rumble.push((slot, low, high));
        }
    }

    fn fake() -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        fake_with(false)
    }

    fn fake_with(one_pad: bool) -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        let host = Arc::new(Mutex::new(FakeHost::default()));
        let hub = Hub::new(Box::new(Fake(Arc::clone(&host))), one_pad);
        hub.poll();
        (hub, host)
    }

    fn key(virtual_key: u8) -> KeyState {
        KeyState {
            scan_code: u32::from(virtual_key) + 0x100,
            virtual_key,
            ..Default::default()
        }
    }

    fn slot<F: Copy, T>(object: *const T, offset: usize) -> F {
        // SAFETY: our objects start with their table, which has the slot.
        let f = unsafe { *(*(object as *const *const *const ())).add(offset / 8) };
        // SAFETY: the caller names the slot's signature.
        unsafe { std::mem::transmute_copy(&f) }
    }

    /// Device callbacks seen: device index, current and previous status.
    type Log = Mutex<Vec<(usize, u32, u32)>>;

    extern "system" fn record(
        _token: u64,
        context: *mut c_void,
        device: *const Device,
        _timestamp: u64,
        current: u32,
        previous: u32,
    ) {
        // SAFETY: the test passes its log and one of the hub's devices.
        let (log, index) = unsafe { (&*(context as *const Log), (*device).index) };
        log.lock().unwrap().push((index, current, previous));
    }

    fn register(hub: &Hub, kind: u32, enumeration: u32, log: &Log) -> u64 {
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let mut token = 0;
        let context = log as *const Log as *mut c_void;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let hr = register(
            &hub.object,
            std::ptr::null(),
            kind,
            1,
            enumeration,
            context,
            Some(record),
            &mut token,
        );
        assert_eq!(hr, S_OK);
        assert_ne!(token, 0);
        token
    }

    fn result(hr: HRESULT, out: *const c_void) -> Result<*const Reading, HRESULT> {
        if hr == S_OK {
            Ok(out.cast())
        } else {
            assert!(out.is_null());
            Err(hr)
        }
    }

    fn current(hub: &Hub, kind: u32, device: usize) -> Result<*const Reading, HRESULT> {
        type Current = extern "system" fn(This, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let hr = slot::<Current, _>(&hub.object, 0x20)(
            &hub.object,
            kind,
            &hub.devices[device],
            &mut out,
        );
        result(hr, out)
    }

    fn walk(hub: &Hub, offset: usize, from: *const Reading) -> Result<*const Reading, HRESULT> {
        type Walk = extern "system" fn(This, *const Reading, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let device = reading(from).device;
        let hr =
            slot::<Walk, _>(&hub.object, offset)(&hub.object, from, GAMEPAD_KIND, device, &mut out);
        result(hr, out)
    }

    fn pad(reading: *const Reading) -> PadState {
        let mut state = PadState::default();
        let get =
            slot::<extern "system" fn(*const Reading, *mut PadState) -> bool, _>(reading, 0xb0);
        assert!(get(reading, &mut state));
        state
    }

    fn stamp(reading: *const Reading) -> (u64, u64) {
        let time = slot::<extern "system" fn(*const Reading) -> u64, _>(reading, 0x28);
        let sequence = slot::<extern "system" fn(*const Reading, u32) -> u64, _>(reading, 0x20);
        (time(reading), sequence(reading, GAMEPAD_KIND))
    }

    fn release(reading: *const Reading) {
        slot::<extern "system" fn(*const Reading) -> u32, _>(reading, 0x10)(reading);
    }

    #[test]
    fn trigger_crossings_are_found_both_ways() {
        let at = |l: f32, r: f32| PadState {
            left_trigger: l,
            right_trigger: r,
            ..Default::default()
        };
        assert_eq!(probe::crossings(&at(0.0, 0.0), &at(1.0, 0.05)), [("left", true, 1.0)]);
        assert!(probe::crossings(&at(1.0, 0.0), &at(0.5, 0.0)).is_empty());
        assert_eq!(
            probe::crossings(&at(0.5, 0.2), &at(0.0, 0.09)),
            [("left", false, 0.0), ("right", false, 0.09)]
        );
    }

    #[test]
    fn ordinal_one_resolves_to_create() {
        let r = HleResolver::new(HostPolicy::none());
        assert_eq!(
            r.classify("GameInput.dll", &ImportSymbol::Ordinal(1)),
            Some(Provenance::Hle)
        );
        assert_eq!(r.classify("GameInput.dll", &ImportSymbol::Ordinal(2)), None);
    }

    #[test]
    fn pads_connect_report_and_disconnect() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        // Blocking enumeration ran before registration returned: with no
        // host pad the keyboard's stand-in is the only one.
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        host.lock().unwrap().pads[1] = Some(PadState {
            buttons: A,
            left_x: 0.5,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[1..],
            [(1, STATUS_CONNECTED, 0), (FALLBACK, 0, STATUS_CONNECTED)]
        );

        let first = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(first).buttons, A);
        assert_eq!(pad(first).left_x, 0.5);
        // No change, no new reading.
        hub.poll();
        let again = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(again, first);
        release(again);

        std::thread::sleep(Duration::from_millis(2));
        host.lock().unwrap().pads[1].as_mut().unwrap().buttons = B;
        hub.poll();
        let second = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(second).buttons, B);
        let (t1, s1) = stamp(first);
        let (t2, s2) = stamp(second);
        assert!(t2 > t1 && s2 == s1 + 1);
        assert!(t2 <= now());

        // The title's edge check: previous of current, and back again.
        assert_eq!(walk(hub, 0x30, second), Ok(first));
        assert_eq!(walk(hub, 0x28, first), Ok(second));
        assert_eq!(walk(hub, 0x28, second), Err(READING_NOT_FOUND));
        assert_eq!(walk(hub, 0x30, first), Err(READING_NOT_FOUND));
        for r in [first, first, second, second] {
            release(r);
        }

        host.lock().unwrap().pads[1] = None;
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[3..],
            [(1, 0, STATUS_CONNECTED), (FALLBACK, STATUS_CONNECTED, 0)]
        );
        assert_eq!(current(hub, GAMEPAD_KIND, 1), Err(DEVICE_DISCONNECTED));
        // The stand-in is back, at rest.
        let rest = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(rest), PadState::default());
        release(rest);
    }

    #[test]
    fn keys_feed_the_keyboard_and_the_stand_in_pad() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, KEYBOARD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(KEYBOARD, STATUS_CONNECTED, 0)]);

        let keys = vec![key(b'W'), key(0x0D), key(b'L'), key(b'C')];
        host.lock().unwrap().keys = keys.clone();
        hub.poll();
        let r = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        let count = slot::<extern "system" fn(*const Reading) -> u32, _>(r, 0x70);
        let state =
            slot::<extern "system" fn(*const Reading, u32, *mut KeyState) -> u32, _>(r, 0x78);
        assert_eq!(count(r), 4);
        let mut got = [KeyState::default(); 8];
        assert_eq!(state(r, 8, got.as_mut_ptr()), 4);
        assert_eq!(got[..4], keys[..]);
        release(r);

        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(
            pad(r),
            PadState {
                buttons: A,
                left_y: 1.0,
                right_x: 1.0,
                right_trigger: 1.0,
                ..Default::default()
            }
        );
        release(r);
        // A keyboard reading is no gamepad reading.
        assert_eq!(current(hub, GAMEPAD_KIND, KEYBOARD), Err(READING_NOT_FOUND));
    }

    #[test]
    fn the_mouse_is_the_right_stick_and_the_triggers() {
        // Without the mouse the keys are as they were: I/J/K/L the right
        // stick, C the right trigger, Space nothing.
        let keys = [key(b'W'), key(SPACE), key(b'L'), key(b'C'), key(b'Q')];
        assert_eq!(
            fallback(&keys, Pointer::default()),
            PadState {
                buttons: LEFT_SHOULDER,
                left_y: 1.0,
                right_x: 1.0,
                right_trigger: 1.0,
                ..Default::default()
            }
        );
        let pointer = Pointer { on: true, right_x: 0.5, right_y: -0.25, fire: true, aim: true, drive: false };
        let pad = fallback(&keys, pointer);
        assert_eq!(
            pad,
            PadState {
                buttons: X | RIGHT_STICK | RIGHT_SHOULDER,
                left_y: 1.0,
                left_trigger: 1.0,
                right_trigger: 1.0,
                right_x: 0.5,
                right_y: -0.25,
                ..Default::default()
            }
        );
    }

    #[test]
    fn driving_puts_the_pedals_on_w_and_s_and_the_handbrake_on_space() {
        let drive = Pointer { on: true, drive: true, ..Default::default() };
        let pad = fallback(&[key(b'W'), key(b'D'), key(SPACE)], drive);
        assert_eq!(
            pad,
            PadState { buttons: RIGHT_SHOULDER, right_trigger: 1.0, left_x: 1.0, ..Default::default() }
        );
        let pad = fallback(&[key(b'S')], drive);
        assert_eq!(pad, PadState { left_trigger: 1.0, ..Default::default() });
    }

    #[test]
    fn mouse_travel_arrives_whole_and_the_stick_comes_to_rest() {
        // 300 pixels to the right in 0.1 s, then nothing: the stick, summed
        // over time at 1400 pixels a second for a full one, pays out 300.
        let (speed, dt) = (1400.0, 0.008);
        let mut look = Look::default();
        let mut paid = 0.0;
        let mut last = (0.0, 0.0);
        for step in 0..200 {
            let moved = if step < 12 { (25.0, 0.0) } else { (0.0, 0.0) };
            last = look.turn(moved, dt, speed, 0.0);
            assert!(last.0 >= 0.0 && last.0 <= 1.0 && last.1 == 0.0);
            paid += last.0 * speed * dt;
        }
        assert!((paid - 300.0).abs() < 3.0, "{paid}");
        assert_eq!(last, (0.0, 0.0));
        // Up on the mouse (negative y) is up on the stick; a deflection
        // starts above the share the title ignores.
        let up = Look::default().turn((0.0, -10.0), dt, speed, 0.12);
        assert!(up.0 == 0.0 && up.1 > 0.12 && up.1 < 0.5, "{up:?}");
    }

    #[test]
    fn recorded_mouse_replays_through_the_one_pad() {
        let still = Pointer { on: true, ..Default::default() };
        let host = record::Raw { buttons: 0x1000, lx: 100, rx: 20_000, ..Default::default() };
        // Mouse off or still: the host pad is stored as it was.
        assert_eq!(recorded_pad(None, Pointer::default()), None);
        assert_eq!(recorded_pad(None, still), None);
        assert_eq!(recorded_pad(Some(host), still), Some(host));
        // Moved: its stick (where it is pushed further) and its triggers.
        let moved = Pointer { right_x: 0.5, right_y: -1.0, fire: true, ..still };
        let stored = recorded_pad(Some(host), moved).unwrap();
        assert_eq!((stored.rx, stored.ry, stored.lt, stored.rt), (20_000, -32_767, 0, 255));
        assert_eq!((stored.buttons, stored.lx), (0x1000, 100));
        let alone = recorded_pad(None, Pointer { aim: true, right_x: -0.25, ..still }).unwrap();
        // The line written and read back, then as the replay hands it over
        // (pad 0) and the one pad merges it with the replay's stand-in.
        let line = record::Line { ms: 5, frame: None, change: record::Change::Pad(Some(alone)) };
        assert_eq!(record::Line::parse(&line.format()), Some(line));
        let pad0 = PadState {
            left_trigger: f32::from(alone.lt) / 255.0,
            right_x: f32::from(alone.rx) / 32767.0,
            ..Default::default()
        };
        let (hub, host) = fake_with(true);
        host.lock().unwrap().pads[0] = Some(pad0);
        host.lock().unwrap().keys = vec![key(b'W')];
        hub.poll();
        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        let p = pad(r);
        assert_eq!((p.left_trigger, p.left_y), (1.0, 1.0));
        assert!((p.right_x + 0.25).abs() < 1e-4, "{}", p.right_x);
        release(r);
    }

    #[test]
    fn a_host_pad_feeds_the_one_pad_which_never_goes() {
        let (hub, host) = fake_with(true);
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);
        let pad = |hub: &Hub| match &hub.state().slots[FALLBACK].history.back().unwrap().value {
            Value::Pad(p) => *p,
            Value::Keys(_) => unreachable!(),
        };
        host.lock().unwrap().keys = vec![key(b'D')];
        hub.poll();
        assert_eq!(pad(hub).left_x, 1.0);
        // A pad is switched on, in any slot: its input arrives, beside the
        // keyboard's, and no device comes or goes.
        host.lock().unwrap().pads[2] = Some(PadState { buttons: A, left_y: -0.5, ..Default::default() });
        hub.poll();
        let p = pad(hub);
        assert_eq!((p.buttons, p.left_x, p.left_y), (A, 1.0, -0.5));
        let device: *const Device = &hub.devices[FALLBACK];
        set_rumble(device, &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(host.lock().unwrap().rumble, [(2, u16::MAX, 0)]);
        // And off again.
        host.lock().unwrap().pads[2] = None;
        host.lock().unwrap().keys.clear();
        hub.poll();
        assert_eq!(pad(hub), PadState::default());
        assert_eq!(log.lock().unwrap().len(), 1);
        assert!(!hub.state().slots[2].connected);
    }

    #[test]
    fn async_enumeration_waits_for_the_poll_and_unregistering_stops_callbacks() {
        let (hub, host) = fake();
        let log = Log::default();
        let token = register(hub, GAMEPAD_KIND, ASYNC_ENUMERATION, &log);
        assert!(log.lock().unwrap().is_empty());
        hub.poll();
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(unregister(&hub.object, token, 5000));
        host.lock().unwrap().pads[0] = Some(PadState::default());
        hub.poll();
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    #[test]
    fn unregistering_waits_for_a_running_callback() {
        use std::sync::atomic::AtomicBool;
        static INSIDE: AtomicBool = AtomicBool::new(false);
        static LEAVE: AtomicBool = AtomicBool::new(false);
        static FROM_INSIDE: AtomicU32 = AtomicU32::new(0);
        extern "system" fn slow(
            token: u64,
            context: *mut c_void,
            _: *const Device,
            _: u64,
            _: u32,
            _: u32,
        ) {
            // From inside its own callback, unregistering fails at once.
            let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(context, 0x68);
            if !unregister(context.cast(), token, 0) {
                FROM_INSIDE.fetch_add(1, Ordering::SeqCst);
            }
            INSIDE.store(true, Ordering::SeqCst);
            while !LEAVE.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let (hub, host) = fake();
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let mut token = 0;
        let context = (&raw const hub.object).cast_mut().cast();
        assert_eq!(
            register(&hub.object, std::ptr::null(), GAMEPAD_KIND, 1, 0, context, Some(slow), &mut token),
            S_OK
        );
        host.lock().unwrap().pads[1] = Some(PadState::default());
        let poller = std::thread::spawn(|| hub.poll());
        while !INSIDE.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(!unregister(&hub.object, token, 5000), "times out while it runs");
        let leave = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            LEAVE.store(true, Ordering::SeqCst);
        });
        let start = Instant::now();
        assert!(unregister(&hub.object, token, u64::MAX), "waits for it");
        assert!(start.elapsed() >= Duration::from_millis(40));
        assert_eq!(FROM_INSIDE.load(Ordering::SeqCst), 1);
        poller.join().unwrap();
        leave.join().unwrap();
    }

    #[test]
    fn devices_are_found_by_id_and_rumble_reaches_the_host() {
        let (hub, host) = fake();
        host.lock().unwrap().pads[2] = Some(PadState::default());
        hub.poll();
        let device = &hub.devices[2] as *const Device;
        let info = slot::<extern "system" fn(*const Device) -> *const u8, _>(device, 0x18);
        // SAFETY: the info has the id at +0x20.
        let id = unsafe { *(info(device).add(0x20) as *const [u8; 32]) };
        assert_ne!(id, [0; 32]);

        type Find = extern "system" fn(This, *const [u8; 32], Out) -> HRESULT;
        let find = slot::<Find, _>(&hub.object, 0x80);
        let mut out = std::ptr::null();
        assert_eq!(find(&hub.object, &id, &mut out), S_OK);
        assert_eq!(out, device.cast());
        assert_eq!(find(&hub.object, &[1; 32], &mut out), DEVICE_NOT_FOUND);
        assert!(out.is_null());

        type Rumble = extern "system" fn(*const Device, *const [f32; 4]);
        let rumble = slot::<Rumble, _>(device, 0x50);
        rumble(device, &[0.5, 1.0, 1.0, 1.0]);
        rumble(&hub.devices[FALLBACK], &[1.0; 4]);
        rumble(device, std::ptr::null());
        assert_eq!(host.lock().unwrap().rumble, [(2, 32767, 65535), (2, 0, 0)]);
    }
}
