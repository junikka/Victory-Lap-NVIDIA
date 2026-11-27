//! Diagnostics: the run's numbers on the picture (`VICTORY_PANEL=1`; off by
//! default; F3 shows and hides it while our window has the focus): white
//! text on a black box in the upper left corner of what is shown, so that a
//! screenshot carries them, with what it costs itself on the submitting
//! thread. Made again twice a second from the counters of
//! the frame-rate line, memory and textures; a frame itself only copies the
//! box out of a buffer in the card's memory. Copied onto the swap chain's
//! buffer, behind the title's picture: saved frames and the frame recorder
//! see the title's own.
//!
//! With the settings menu (`crate::settings_menu`, `VICTORY_SETTINGS_MENU=1`)
//! F3 cycles Off, Minimal (one line, made again once a second; nothing is
//! counted or drawn while Off) and Full (this panel), and F6 opens the menu.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_F3};
use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

use super::{Host, HostList, transition};
use crate::settings_menu::{self as menu, CELL, Dots, GLYPH, Hud, Key, Layout, Menu, ROWS, Start, glyph};

/// How often the numbers change.
const EVERY: Duration = Duration::from_millis(500);

/// Characters in a line, lines, and the box's rim in dots.
const COLUMNS: u32 = 40;
const LINES: u32 = 9;
const RIM: u32 = 3;

/// The texts are put into one of these in turn: the card reads the others,
/// at most half a second old.
const UPLOADS: usize = 4;

fn asked() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_PANEL"))
}

/// Whether it is shown now.
fn shown() -> &'static AtomicBool {
    static SHOWN: OnceLock<AtomicBool> = OnceLock::new();
    SHOWN.get_or_init(|| AtomicBool::new(asked()))
}

/// F3 was pressed since the last call, with one of our windows in front.
fn pressed() -> bool {
    static DOWN: AtomicBool = AtomicBool::new(false);
    // SAFETY: a plain query.
    let down = unsafe { GetAsyncKeyState(i32::from(VK_F3)) } < 0;
    if DOWN.swap(down, Ordering::Relaxed) || !down {
        return false;
    }
    let mut process = 0;
    // SAFETY: plain queries; room for the answer.
    unsafe { GetWindowThreadProcessId(GetForegroundWindow(), &mut process) };
    process == std::process::id()
}

/// The box's size in pixels, a dot being `scale` of them each way.
const fn size(scale: u32) -> (u32, u32) {
    ((COLUMNS * CELL.0 + 2 * RIM) * scale, (LINES * CELL.1 + 2 * RIM) * scale)
}

/// Pixels to a dot for a picture that high: a 720-line screenshot of it
/// shows one and a third.
fn scale(height: u32) -> u32 {
    (height / 540).max(1)
}

/// Whether a dot of the picture of `lines` is white.
fn lit(lines: &[String], x: u32, y: u32) -> bool {
    let (Some(x), Some(y)) = (x.checked_sub(RIM), y.checked_sub(RIM)) else {
        return false;
    };
    if x % CELL.0 >= GLYPH.0 || y % CELL.1 >= GLYPH.1 || x / CELL.0 >= COLUMNS {
        return false;
    }
    let Some(&c) = lines.get((y / CELL.1) as usize).and_then(|l| l.as_bytes().get((x / CELL.0) as usize)) else {
        return false;
    };
    (glyph(c.to_ascii_uppercase())[(y % CELL.1) as usize] >> (GLYPH.0 - 1 - x % CELL.0)) & 1 != 0
}

/// The bytes of a black and a white pixel of a swap chain's format.
fn colours(format: DXGI_FORMAT) -> Option<[&'static [u8]; 2]> {
    Some(match format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM => [&[0, 0, 0, 0xFF], &[0xFF; 4]],
        DXGI_FORMAT_R10G10B10A2_UNORM => [&[0, 0, 0, 0xC0], &[0xFF; 4]],
        // Halves of 0 and 1.
        DXGI_FORMAT_R16G16B16A16_FLOAT => [
            &[0, 0, 0, 0, 0, 0, 0, 0x3C],
            &[0, 0x3C, 0, 0x3C, 0, 0x3C, 0, 0x3C],
        ],
        _ => return None,
    })
}

/// The box as rows of `pitch` bytes.
fn painted(lines: &[String], scale: u32, colours: [&[u8]; 2], pitch: usize) -> Vec<u8> {
    let (width, height) = size(scale);
    let bytes = colours[0].len();
    let mut out = vec![0; pitch * height as usize];
    for y in 0..height / scale {
        let top = pitch * (y * scale) as usize;
        for x in 0..width / scale {
            let colour = colours[usize::from(lit(lines, x, y))];
            for i in 0..scale {
                let at = top + (x * scale + i) as usize * bytes;
                out[at..at + bytes].copy_from_slice(colour);
            }
        }
        for i in 1..scale as usize {
            out.copy_within(top..top + pitch, top + i * pitch);
        }
    }
    out
}

/// The counters read for it. They grow, and go back to 0 with the frame-rate
/// line (right behind a look of ours).
#[derive(Clone, Copy, Default)]
struct Counters([u64; COUNTERS]);

const COUNTERS: usize = 11;

impl Counters {
    fn now() -> Counters {
        use super::{CARD_CLOCK_US, COPY_BACK, PREPARING};
        use crate::copystat::{C, peek};
        let get = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);
        Counters([
            super::worked(),
            get(&CARD_CLOCK_US[0]),
            get(&CARD_CLOCK_US[1]),
            get(&CARD_CLOCK_US[2]),
            get(&PREPARING[3]),
            get(&COPY_BACK[0]),
            get(&COPY_BACK[1]),
            get(&crate::shadow::CARD_STATS[0]),
            crate::draw::dropped_now(),
            peek(C::RecordNs),
            crate::draw::drawn(),
        ])
    }

    /// What came since `before`, per frame.
    fn since(self, before: Counters, frames: f64) -> [f64; COUNTERS] {
        let mut out = [0.0; COUNTERS];
        for (i, out) in out.iter_mut().enumerate() {
            let (was, is) = (before.0[i], self.0[i]);
            *out = if is >= was { is - was } else { is } as f64 / frames;
        }
        out
    }
}

/// The text. `c`: [`Counters`] per frame, `ms` a frame's time, `worst` the
/// longest frame's, `cost` what this takes of a frame, in microseconds.
fn text(c: &[f64; COUNTERS], ms: f64, worst: f64, cost: f64, host: &Host) -> Vec<String> {
    let gb = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    let (working, private) = crate::copystat::process_memory();
    let [budget, used, _, shared] = host.video_memory().unwrap_or_default();
    let idle = (c[1] / 1e3).min(ms);
    vec![
        format!("FPS {:.1}  FRAME {ms:.1} MS  LONGEST {worst:.0}", 1e3 / ms.max(0.001)),
        format!("DRAWS {:.0}  DROPPED {:.0}", c[10], c[8]),
        format!("SUBMIT BUSY {:.1}  WAITS FOR GPU {:.1}", c[0] / 1e3, c[3] / 1e3),
        format!("GPU BUSY {:.1}  IDLE BEHIND US {:.1}", ms - idle, c[2] / 1e3),
        format!("RECORDING {:.0} MS  COPY BACK {:.1}", c[9] / 1e6, c[5] / 1e3),
        format!("MB UP {:.0}  TEXTURES {:.0}  BACK {:.0}", c[7] / 1e6, c[4] / 1e6, c[6] / 1e6),
        format!("RAM {:.1} GB  IN MEMORY {:.1}", gb(private), gb(working)),
        format!("VRAM {:.1}/{:.1} GB  SHARED {:.1}", gb(used), gb(budget), gb(shared)),
        format!("THIS PANEL {:.0} US A FRAME", cost),
    ]
}

/// The buffers of a box that size and format.
struct Buffers {
    format: DXGI_FORMAT,
    scale: u32,
    pitch: u32,
    /// The box's size in pixels.
    size: (u32, u32),
    uploads: Vec<(ID3D12Resource, *mut u8)>,
    card: ID3D12Resource,
    /// The upload written last.
    turn: usize,
    /// It is not in the card's buffer yet.
    fresh: bool,
}

// SAFETY: free-threaded resources, and memory only written under the presenter's lock.
unsafe impl Send for Buffers {}

impl Buffers {
    fn new(host: &Host, format: DXGI_FORMAT, scale: u32) -> Option<Buffers> {
        Buffers::sized(host, format, scale, size(scale))
    }

    /// For a box of that many pixels.
    fn sized(host: &Host, format: DXGI_FORMAT, scale: u32, (width, height): (u32, u32)) -> Option<Buffers> {
        let pitch = (width * colours(format)?[0].len() as u32).next_multiple_of(256);
        let bytes = u64::from(pitch) * u64::from(height);
        let uploads = (0..UPLOADS).map(|_| host.upload_buffer(bytes)).collect::<Option<Vec<_>>>()?;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut card: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a buffer.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut card)
                .ok()?;
        }
        Some(Buffers { format, scale, pitch, size: (width, height), uploads, card: card?, turn: 0, fresh: false })
    }

    fn write(&mut self, lines: &[String]) {
        let Some(colours) = colours(self.format) else { return };
        let picture = painted(lines, self.scale, colours, self.pitch as usize);
        self.put(&picture);
    }

    /// A picture of rows of the pitch, the box's height.
    fn put(&mut self, picture: &[u8]) {
        debug_assert_eq!(picture.len(), self.pitch as usize * self.size.1 as usize);
        self.turn = (self.turn + 1) % UPLOADS;
        // SAFETY: a mapped upload buffer of the picture's size, which the card read a second or more ago.
        unsafe { std::ptr::copy_nonoverlapping(picture.as_ptr(), self.uploads[self.turn].1, picture.len()) };
        self.fresh = true;
    }

    /// Records the copy of the box onto `back` at `(x, y)`, a buffer of the
    /// swap chain of that size and format in the state to be copied into.
    fn copy_onto(&mut self, list: &HostList, back: &ID3D12Resource, (x, y): (u32, u32), (width, height): (u32, u32)) {
        if x >= width || y >= height {
            return;
        }
        let (w, h) = self.size;
        let list = &list.list;
        // SAFETY: an open list and live resources; the box lies within both.
        unsafe {
            if std::mem::take(&mut self.fresh) {
                list.CopyResource(&self.card, &self.uploads[self.turn].0);
                list.ResourceBarrier(&[transition(
                    &self.card,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&self.card),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: self.format,
                            Width: w,
                            Height: h,
                            Depth: 1,
                            RowPitch: self.pitch,
                        },
                    },
                },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(back),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let part = D3D12_BOX {
                left: 0,
                top: 0,
                front: 0,
                right: w.min(width - x),
                bottom: h.min(height - y),
                back: 1,
            };
            list.CopyTextureRegion(&to, x, y, 0, &from, Some(&part));
        }
    }
}

/// A box of the menu's: its buffers, made for the picture's format and
/// scale, and whether its picture is to be made again.
#[derive(Default)]
struct Shown {
    buffers: Option<Buffers>,
    failed: bool,
    dirty: bool,
}

impl Shown {
    /// Records the copy of the picture `paint` makes (`dots` dots, each
    /// `scale` pixels) onto `back` at `at`; `paint` runs only when the
    /// picture changed or the buffers are new.
    #[allow(clippy::too_many_arguments)]
    fn draw(
        &mut self,
        host: &Host,
        list: &HostList,
        back: &ID3D12Resource,
        at: (u32, u32),
        screen: (u32, u32),
        format: DXGI_FORMAT,
        scale: u32,
        dots: (u32, u32),
        paint: impl FnOnce() -> Dots,
    ) {
        let Some(layout) = layout(format) else { return };
        if self.failed {
            return;
        }
        if self.buffers.as_ref().is_none_or(|b| (b.format, b.scale) != (format, scale)) {
            // (What the card may still read stays, as for the panel's own.)
            std::mem::forget(self.buffers.take());
            self.buffers = Buffers::sized(host, format, scale, (dots.0 * scale, dots.1 * scale));
            self.failed = self.buffers.is_none();
            self.dirty = true;
        }
        let Some(buffers) = &mut self.buffers else { return };
        if std::mem::take(&mut self.dirty) {
            let picture = paint().scaled(scale, &menu::pixels(layout), buffers.pitch as usize);
            buffers.put(&picture);
        }
        buffers.copy_onto(list, back, at, screen);
    }
}

/// How a swap chain's format lays out a pixel.
fn layout(format: DXGI_FORMAT) -> Option<Layout> {
    Some(match format {
        DXGI_FORMAT_R8G8B8A8_UNORM => Layout::Rgba8,
        DXGI_FORMAT_B8G8R8A8_UNORM => Layout::Bgra8,
        DXGI_FORMAT_R10G10B10A2_UNORM => Layout::Rgb10a2,
        DXGI_FORMAT_R16G16B16A16_FLOAT => Layout::Rgba16f,
        _ => return None,
    })
}

/// The panel of a presenter.
pub(super) struct Panel {
    since: Instant,
    frames: u32,
    last: Instant,
    worst: f64,
    /// What this took since the last text.
    cost: Duration,
    before: Counters,
    lines: Vec<String>,
    changed: bool,
    buffers: Option<Buffers>,
    failed: bool,
    /// The settings menu's boxes, with the menu on.
    ui: Option<Ui>,
}

/// The settings menu's state; one for the process, as a presenter made
/// again (a new swap chain) starts a new [`Panel`].
struct State {
    menu: Menu,
    edges: menu::Edges,
    hud: Hud,
    path: std::path::PathBuf,
    /// The output size was taken from the first picture shown.
    output_seen: bool,
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| {
        let path = menu::file_path();
        let file = std::fs::read_to_string(&path).map(|t| menu::parse(&t)).unwrap_or_default();
        let menu = Menu::new(starts(), &file);
        let hud = Hud::of(menu.value[menu::PANEL]);
        let live: Vec<&str> = (0..menu::COUNT).filter(|&i| menu.live[i]).map(|i| ROWS[i].key).collect();
        tracing::info!(
            file = %path.display(),
            lines = file.len(),
            ?live,
            "settings menu on (VICTORY_SETTINGS_MENU=1): F6 opens it, F3 cycles the panel; the other rows are for the next launch"
        );
        Mutex::new(State { menu, edges: menu::Edges::default(), hud, path, output_seen: false })
    })
}

/// What each row runs with, from the values the layer read at its start.
fn starts() -> [Start; menu::COUNT] {
    let other = |s: String| Start::Other(s.to_ascii_uppercase());
    let frame_cap = match crate::frame::asked() {
        // The title's own rate, 30.
        None => Start::Value(0),
        Some(d) if d.is_zero() => Start::Value(2),
        Some(d) if (d.as_secs_f64() * 1e3 - 16.667).abs() < 0.05 => Start::Value(1),
        Some(d) => other(format!("{:.0}", 1.0 / d.as_secs_f64())),
    };
    let texture_detail = match crate::resource::drop_mips() {
        1 => Start::Value(0),
        0 => Start::Value(1),
        n => other(format!("DROP {n}")),
    };
    let texture_cap = match crate::resource::texture_max_pixels() {
        Some(0) => Start::Value(0),
        Some(n) => match menu::number_value(menu::TEXTURE_CAP, n as f64) {
            Start::Other(_) => other(n.to_string()),
            known => known,
        },
        None => other("auto".into()),
    };
    let windows = match crate::shadow::window_mapped_most() >> 20 {
        0 => Start::Value(2),
        mb => match menu::number_value(menu::MEMORY_WINDOWS, mb as f64) {
            Start::Other(_) => other(format!("{mb} MB")),
            known => known,
        },
    };
    let ambient = match crate::ambient::start_level() {
        Some(level) => menu::number_value(menu::AMBIENT, f64::from(level)),
        None => other("off".into()),
    };
    let reflections = match crate::ambient::reflect_start() {
        p if p <= 0.0 => Start::Value(0),
        p => menu::number_value(menu::REFLECTIONS, f64::from(p)),
    };
    let panel = match (asked(), crate::switch::on("VICTORY_PANEL_MINIMAL")) {
        (false, _) => Start::Value(0),
        (true, true) => Start::Value(1),
        (true, false) => Start::Value(2),
    };
    [
        frame_cap,
        // Taken from the first picture shown.
        other("?".into()),
        texture_detail,
        texture_cap,
        windows,
        ambient,
        Start::Value(if crate::ambient::sky_on() { 0 } else { 1 }),
        reflections,
        panel,
    ]
}

/// The output size as the menu shows it.
fn output_start((width, height): (u32, u32)) -> Start {
    let named = format!("{width}x{height}");
    match menu::file_value(menu::OUTPUT, &named) {
        Some(v) => Start::Value(v),
        None => Start::Other(named.to_ascii_uppercase()),
    }
}

/// Our window is in front.
fn ours_in_front() -> bool {
    let mut process = 0;
    // SAFETY: plain queries; room for the answer.
    unsafe { GetWindowThreadProcessId(GetForegroundWindow(), &mut process) };
    process == std::process::id()
}

/// The menu's keys pressed since the last look, one bit per [`Key::ALL`],
/// with one of our windows in front.
fn menu_presses(edges: &mut menu::Edges) -> u8 {
    let (mut down, mut tapped) = (0u8, 0u8);
    for (i, key) in Key::ALL.iter().enumerate() {
        // SAFETY: a plain query.
        let state = unsafe { GetAsyncKeyState(i32::from(key.vk())) };
        down |= u8::from(state < 0) << i;
        tapped |= u8::from(state & 1 != 0) << i;
    }
    let pressed = edges.pressed(down, tapped);
    if pressed != 0 && !ours_in_front() { 0 } else { pressed }
}

/// The boxes of the menu and of the minimal panel, and the minimal panel's
/// counting.
struct Ui {
    menu: Shown,
    minimal: Shown,
    since: Instant,
    frames: u32,
    line: String,
    /// The menu's picture, made when it changed.
    picture: Option<Dots>,
}

impl Ui {
    fn new() -> Ui {
        Ui {
            menu: Shown::default(),
            minimal: Shown::default(),
            since: Instant::now(),
            frames: 0,
            line: String::new(),
            picture: None,
        }
    }
}

/// Where the menu goes on a picture of that size, its box being `w` pixels
/// wide: right of the middle, a quarter of the way down, as drawn in the
/// approved picture (194 of 1280 from the right, 169 of 720 from the top).
fn menu_at((width, height): (u32, u32), w: u32) -> (u32, u32) {
    (width.saturating_sub(w + width * 194 / 1280), height * 169 / 720)
}

impl Panel {
    pub(super) fn new() -> Panel {
        let now = Instant::now();
        Panel {
            since: now,
            frames: 0,
            last: now,
            worst: 0.0,
            cost: Duration::ZERO,
            before: Counters::default(),
            lines: Vec::new(),
            changed: false,
            buffers: None,
            failed: false,
            ui: None,
        }
    }

    /// A frame was presented; `line`: the counters go back to 0 right behind
    /// this call. (Not shown it asks for the key's state and no more.)
    pub(super) fn frame(&mut self, line: bool) {
        if menu::menu_on() {
            // The menu's keys and the minimal panel; Full is this one.
            self.menu_frame();
        } else if pressed() {
            shown().fetch_xor(true, Ordering::Relaxed);
        }
        if !shown().load(Ordering::Relaxed) {
            self.frames = 0;
            return;
        }
        let now = Instant::now();
        if self.frames == 0 && now - self.last > 2 * EVERY {
            // (Shown from here on.)
            (self.since, self.last, self.worst, self.before) = (now, now, 0.0, Counters::now());
            self.frames = 0;
        }
        self.frames += 1;
        self.worst = self.worst.max((now - self.last).as_secs_f64() * 1e3);
        self.last = now;
        let elapsed = now - self.since;
        if elapsed < EVERY && !(line && elapsed * 2 >= EVERY) {
            return;
        }
        let Some(host) = super::host() else { return };
        let frames = f64::from(self.frames);
        let counters = Counters::now();
        let c = counters.since(self.before, frames);
        self.before = counters;
        let cost = self.cost.as_secs_f64() * 1e6 / frames;
        self.lines = text(&c, elapsed.as_secs_f64() * 1e3 / frames, self.worst, cost, host);
        self.changed = true;
        (self.since, self.frames, self.worst) = (now, 0, 0.0);
        self.cost = now.elapsed();
    }

    /// Records the copy of the box onto `back`, a buffer of the swap chain of
    /// that size and format in the state to be copied into.
    pub(super) fn draw(
        &mut self,
        host: &Host,
        list: &HostList,
        back: &ID3D12Resource,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
    ) {
        if menu::menu_on() {
            self.menu_draw(host, list, back, (width, height), format);
        }
        if !shown().load(Ordering::Relaxed) || self.lines.is_empty() || self.failed {
            return;
        }
        let began = Instant::now();
        // Not the title's: the frame recorder leaves it out.
        let _ours = crate::replay::record::unrecorded();
        let scale = scale(height);
        if self.buffers.as_ref().is_none_or(|b| (b.format, b.scale) != (format, scale)) {
            // (What the card may still read stays: a picture's size changes once or twice in a run.)
            std::mem::forget(self.buffers.take());
            self.buffers = Buffers::new(host, format, scale);
            self.failed = self.buffers.is_none();
            self.changed = true;
        }
        let Some(buffers) = &mut self.buffers else { return };
        if std::mem::take(&mut self.changed) {
            buffers.write(&self.lines);
        }
        buffers.copy_onto(list, back, (0, 0), (width, height));
        self.cost += began.elapsed();
    }

    /// The settings menu's part of a frame presented: its keys, what they
    /// change, and the minimal panel's count (once a second; nothing while
    /// the panel is not Minimal).
    fn menu_frame(&mut self) {
        let ui = self.ui.get_or_insert_with(Ui::new);
        let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
        let pressed = menu_presses(&mut state.edges);
        for (i, key) in Key::ALL.iter().enumerate() {
            if pressed & (1 << i) == 0 {
                continue;
            }
            let outcome = state.menu.press(*key);
            if outcome.redraw {
                ui.picture = None;
            }
            if *key == Key::F6 {
                menu::SETTINGS_MENU_OPEN.store(state.menu.open, Ordering::Relaxed);
            }
            if let Some(row) = outcome.changed {
                changed(&mut state, row);
                ui.picture = None;
            }
        }
        let hud = state.hud;
        drop(state);
        // The full panel is this one's, shown as without the menu.
        shown().store(hud == Hud::Full, Ordering::Relaxed);
        if hud != Hud::Minimal {
            ui.frames = 0;
            return;
        }
        let now = Instant::now();
        if ui.frames == 0 {
            ui.since = now;
        }
        ui.frames += 1;
        let elapsed = now - ui.since;
        if elapsed < Duration::from_secs(1) {
            return;
        }
        let Some(host) = super::host() else { return };
        // The frames since the last line, so its first frame is not counted twice.
        let frames = f64::from(ui.frames - 1);
        let ms = elapsed.as_secs_f64() * 1e3 / frames.max(1.0);
        let gb = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        let (_, private) = crate::copystat::process_memory();
        let [budget, used, ..] = host.video_memory().unwrap_or_default();
        ui.line = menu::minimal_text(1e3 / ms.max(0.001), ms, gb(private), gb(used), gb(budget));
        ui.minimal.dirty = true;
        (ui.since, ui.frames) = (now, 1);
    }

    /// Records the minimal panel and the menu onto `back`, as [`Panel::draw`].
    fn menu_draw(
        &mut self,
        host: &Host,
        list: &HostList,
        back: &ID3D12Resource,
        screen: (u32, u32),
        format: DXGI_FORMAT,
    ) {
        let ui = self.ui.get_or_insert_with(Ui::new);
        let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
        if !state.output_seen {
            state.output_seen = true;
            state.menu.learnt(menu::OUTPUT, output_start(screen));
            ui.picture = None;
        }
        let (hud, open) = (state.hud, state.menu.open);
        if open && ui.picture.is_none() {
            ui.picture = Some(menu::menu_picture(&state.menu));
            ui.menu.dirty = true;
        }
        drop(state);
        if (hud != Hud::Minimal || ui.line.is_empty()) && !open {
            return;
        }
        // Not the title's: the frame recorder leaves it out.
        let _ours = crate::replay::record::unrecorded();
        let scale = scale(screen.1);
        if hud == Hud::Minimal && !ui.line.is_empty() {
            let line = &ui.line;
            let at = (5 * scale, 5 * scale);
            let dots = menu::minimal_size();
            ui.minimal.draw(host, list, back, at, screen, format, scale, dots, || menu::minimal_picture(line));
        }
        if open && let Some(picture) = &ui.picture {
            let dots = menu::menu_size();
            let at = menu_at(screen, dots.0 * scale);
            ui.menu.draw(host, list, back, at, screen, format, scale, dots, || picture.clone());
        }
    }
}

/// A row changed: a live one acts now; every change goes into the file.
fn changed(state: &mut State, row: usize) {
    let value = state.menu.value[row].map(|v| ROWS[row].values[v]);
    let live = state.menu.live[row];
    if live && let Some(value) = value {
        match row {
            menu::AMBIENT => crate::ambient::set_live_level(value.parse().unwrap_or(0.0)),
            menu::REFLECTIONS => crate::ambient::set_live_reflect(value.parse().unwrap_or(0.0)),
            menu::PANEL => state.hud = Hud::of(state.menu.value[row]),
            _ => {}
        }
    }
    let written = menu::write_file(&state.path, &state.menu.file_text());
    match written {
        Ok(()) => tracing::info!(
            row = ROWS[row].key,
            value = value.unwrap_or("?"),
            live,
            file = %state.path.display(),
            "settings menu: a setting changed ({})",
            if live { "now" } else { "at the next launch" }
        ),
        Err(error) => tracing::warn!(%error, file = %state.path.display(), "settings menu: the settings file was not written"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_menu_sits_right_of_the_middle_within_the_picture() {
        let w = menu::menu_size().0 * 2;
        let (x, y) = menu_at((1920, 1080), w);
        assert_eq!((x, y), (1920 - w - 291, 253));
        assert!(x + w <= 1920 && y + menu::menu_size().1 * 2 <= 1080);
        // A picture narrower than the box: at the left edge.
        assert_eq!(menu_at((400, 300), w).0, 0);
        assert_eq!(output_start((1920, 1080)), Start::Value(0));
        assert_eq!(output_start((2560, 1440)), Start::Other("2560X1440".into()));
        assert_eq!(layout(DXGI_FORMAT_B8G8R8A8_UNORM), Some(Layout::Bgra8));
        assert_eq!(layout(DXGI_FORMAT_R16G16B16A16_FLOAT), Some(Layout::Rgba16f));
        assert_eq!(layout(DXGI_FORMAT_R32G32B32A32_FLOAT), None);
    }

    #[test]
    fn a_box_is_painted() {
        let lines = vec!["I".to_owned(), " i".to_owned()];
        for scale in [1, 2, 4] {
            let (width, height) = size(scale);
            let pitch = (width * 4).next_multiple_of(256) as usize;
            let colours = colours(DXGI_FORMAT_R10G10B10A2_UNORM).unwrap();
            let picture = painted(&lines, scale, colours, pitch);
            assert_eq!(picture.len(), pitch * height as usize);
            let at = |x: u32, y: u32| &picture[y as usize * pitch + x as usize * 4..][..4];
            // The rim is black, the first letter's top row lit in its middle.
            assert_eq!(at(0, 0), colours[0]);
            assert_eq!(at((RIM + 2) * scale, RIM * scale), colours[1]);
            assert_eq!(at((RIM + 2) * scale + scale - 1, RIM * scale + scale - 1), colours[1]);
            assert_eq!(at(RIM * scale, RIM * scale), colours[0]);
            // Small letters are shown as capitals, a line and a cell on.
            assert_eq!(at((RIM + CELL.0 + 2) * scale, (RIM + CELL.1) * scale), colours[1]);
            // Nothing past a line's end.
            assert_eq!(at((RIM + CELL.0 * 2 + 2) * scale, (RIM + CELL.1) * scale), colours[0]);
        }
    }

    #[test]
    fn counters_that_went_back_to_nothing_count_from_there() {
        let (mut a, mut b) = (Counters::default(), Counters::default());
        (a.0[0], b.0[0]) = (10, 30);
        (a.0[1], b.0[1]) = (50, 20);
        let c = b.since(a, 10.0);
        assert_eq!((c[0], c[1]), (2.0, 2.0));
    }
}
