//! The frame rate and the machine's load, drawn into the top left corner of
//! every frame shown.
//!
//! The window's caption has the frame rate, but a capture of the window's
//! picture (a stream) does not carry the caption. Here the numbers are part
//! of the picture: a small box of text copied onto the swap chain's back
//! buffer right after the title's frame, before it is presented. The
//! title's own frame is not touched, so saved frames and everything read
//! back from it are as before.
//!
//! The box has four lines, new once a second:
//!
//! ```text
//! 58.0 FPS  17 MS              the frames shown a second and a frame's time
//! 1% LOW 41  PREP 4.2 MS       the 99th-percentile frame of the last 10 s, as
//!                              a rate; the submit thread's preparing a frame
//! CPU 31% TOP 97% GPU 41%      all processors; this process's busiest thread,
//!                              of one; the card's busiest engine, all programs
//! VRAM 9.0/10.0 FREE 3.1 GB    the card's memory this process uses / its
//!                              budget; the PC's free memory (red under 2 GB)
//! ```
//!
//! A thread near 100 % beside a card far from it is a frame held by that
//! thread; a card near 100 % holds it itself; neither busy is waiting (for
//! memory to cross between the two, mostly). The load comes from
//! `crate::stats` (its own thread, a look a second; its log line every
//! fifth second has this process's cores and card share too), the card's
//! memory from the host, the preparing from the frame-rate meter's line.
//!
//! The text is drawn on the CPU with a font of its own (5 by 7 dots a
//! glyph) into one of four buffers the card copies from, taken in turn: a
//! buffer is written again only three texts (three seconds) after it was
//! last recorded, so a present still in the air reads what it was recorded
//! with. Off by default (`VICTORY_HUD=1` shows it); the frame rate alone
//! is shown otherwise, `VICTORY_FPS_COUNTER=0` leaves the frame as the
//! title drew it.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The frame rate in tenths of a frame a second, as last measured (once a
/// second, by the presenter's meter); `u32::MAX` before the first.
static TENTHS: AtomicU32 = AtomicU32::new(u32::MAX);

/// The submit thread's preparing a frame, in hundredths of a millisecond,
/// as the meter last measured it; `u32::MAX` before the first.
static PREP: AtomicU32 = AtomicU32::new(u32::MAX);

/// What is drawn: nothing, the frame rate alone, or the whole box.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Off,
    Rate,
    All,
}

fn mode() -> Mode {
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let hud = std::env::var("VICTORY_HUD");
        match std::env::var("VICTORY_FPS_COUNTER").as_deref() {
            Ok("0") => Mode::Off,
            Ok("fps") => Mode::Rate,
            // The full box only when asked (the player asked: "the box can go as long as
            // you get logs"); the load figures are logged either way.
            _ if hud.as_deref() == Ok("1") => Mode::All,
            _ => Mode::Rate,
        }
    })
}

/// The frame rate just measured, and the ms a frame the submit thread
/// spent preparing over the same second.
pub(crate) fn set(fps: f64, preparing_ms: f64) {
    if fps.is_finite() && fps >= 0.0 {
        TENTHS.store((fps * 10.0).round().min(99_999.0) as u32, Ordering::Relaxed);
    }
    if preparing_ms.is_finite() && preparing_ms >= 0.0 {
        PREP.store((preparing_ms * 100.0).round().min(999_999.0) as u32, Ordering::Relaxed);
    }
}

/// How long the 1% low looks back, in seconds.
const LOW_WINDOW_S: f64 = 10.0;

/// The most frames the 1% low holds (10 s at 1,600 frames a second).
const LOW_MOST: usize = 16_384;

/// The frame times of the last [`LOW_WINDOW_S`] seconds, for the 1% low.
#[derive(Default)]
struct Lows {
    /// When each frame ended (seconds from any start) and its time in ms.
    frames: VecDeque<(f64, f32)>,
    /// Room for the times while the slowest are picked out.
    scratch: Vec<f32>,
}

impl Lows {
    /// A frame of `ms` ended at `at` seconds; frames older than the window
    /// (or past [`LOW_MOST`]) are let go.
    fn push(&mut self, at: f64, ms: f32) {
        self.frames.push_back((at, ms));
        while self.frames.front().is_some_and(|&(t, _)| t < at - LOW_WINDOW_S) || self.frames.len() > LOW_MOST {
            self.frames.pop_front();
        }
    }

    /// The 1% low in frames a second: the rate of the frame time 99 % of
    /// the window's frames are no slower than. None with under 10 frames.
    fn low_fps(&mut self) -> Option<f32> {
        let n = self.frames.len();
        if n < 10 {
            return None;
        }
        self.scratch.clear();
        self.scratch.extend(self.frames.iter().map(|&(_, ms)| ms));
        let at = ((n as f64 * 0.99).ceil() as usize).clamp(1, n) - 1;
        let (_, &mut ms, _) = self.scratch.select_nth_unstable_by(at, f32::total_cmp);
        (ms > 0.0).then(|| 1000.0 / ms)
    }
}

/// Whether the PC's free memory is short enough to draw in the warning
/// colour: under 2 GB.
fn ram_low(free_mb: u32) -> bool {
    free_mb < 2048
}

/// A glyph's seven rows, five dots each (bit 4 is the left one).
fn glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        '.' => [0, 0, 0, 0, 0, 0b01100, 0b01100],
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        '/' => [0b00001, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b10000],
        '-' => [0, 0, 0, 0b11111, 0, 0, 0],
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        _ => [0; 7],
    }
}

/// The text for a frame rate in tenths.
fn text(tenths: u32) -> String {
    format!("{}.{} FPS", tenths / 10, tenths % 10)
}

/// The glyphs a line of the whole box has room for.
const COLUMNS: usize = 28;

/// The lines of the whole box.
const ROWS: usize = 4;

/// The glyphs of the frame rate alone ("9999.9 FPS").
const RATE_COLUMNS: usize = 10;

/// A part of the text drawn in the warning colour: its line and columns.
type Warn = Option<(usize, std::ops::Range<usize>)>;

/// What the whole box shows, as it was last measured.
#[derive(Clone, Copy, Default)]
struct Figures {
    /// Frames a second in tenths.
    tenths: u32,
    /// The submit thread's preparing a frame, ms.
    prep_ms: Option<f32>,
    /// The 1% low of the last ten seconds, frames a second.
    low_fps: Option<f32>,
    /// The machine's load, if a look was taken.
    stats: Option<crate::stats::Stats>,
    /// The card's memory in bytes (budget, used), if the host answered.
    card: Option<(u64, u64)>,
}

/// The [`ROWS`] lines of the whole box (see the module's notes), each
/// exactly [`COLUMNS`] glyphs, and the part to draw in the warning colour.
/// A figure not there yet is "--".
fn lines(f: &Figures) -> (Vec<String>, Warn) {
    let gb = |mb: f64| mb / 1024.0;
    let none = || "--".to_owned();
    let tenths = f.tenths;
    let mut out = Vec::with_capacity(ROWS);
    out.push(if tenths > 0 { format!("{}  {:.0} MS", text(tenths), 10_000.0 / f64::from(tenths)) } else { text(tenths) });
    out.push(format!(
        "1% LOW {}  PREP {}",
        f.low_fps.map_or_else(none, |l| format!("{l:.0}")),
        f.prep_ms.map_or_else(none, |p| format!("{p:.1} MS")),
    ));
    out.push(match f.stats {
        Some(s) => format!(
            "CPU {:.0}% TOP {:.0}% GPU {}",
            s.cpu_all,
            s.top_thread,
            s.gpu_all.map_or_else(none, |g| format!("{g:.0}%")),
        ),
        None => "CPU -- TOP -- GPU --".to_owned(),
    });
    let vram = match f.card {
        Some((budget, used)) => format!("VRAM {:.1}/{:.1}", gb((used >> 20) as f64), gb((budget >> 20) as f64)),
        None => "VRAM --".to_owned(),
    };
    let free = match f.stats {
        Some(s) => format!("FREE {:.1} GB", gb(f64::from(s.ram_free_mb))),
        None => "FREE --".to_owned(),
    };
    let warn = f
        .stats
        .filter(|s| ram_low(s.ram_free_mb))
        .map(|_| (3, vram.len() + 1..vram.len() + 1 + free.len()));
    out.push(format!("{vram} {free}"));
    (out.into_iter().map(|line| fit(&line, COLUMNS)).collect(), warn)
}

/// `line` in exactly `columns` glyphs: cut, or filled with spaces.
fn fit(line: &str, columns: usize) -> String {
    let mut out: String = line.chars().take(columns).collect();
    out.extend(std::iter::repeat_n(' ', columns - out.chars().count()));
    out
}

/// Bytes a pixel of a swap chain format this layer presents in, or none.
fn pixel_bytes(format: DXGI_FORMAT) -> Option<usize> {
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_R10G10B10A2_UNORM => Some(4),
        DXGI_FORMAT_R16G16B16A16_FLOAT => Some(8),
        _ => None,
    }
}

/// What a pixel of the box shows: the box behind the text.
const INK_BOX: u8 = 0;
/// The text.
const INK_TEXT: u8 = 1;
/// The text of a figure that warns.
const INK_WARN: u8 = 2;

/// One pixel of `format`, of [`INK_BOX`], [`INK_TEXT`] or [`INK_WARN`].
fn pixel(format: DXGI_FORMAT, ink: u8, out: &mut [u8]) {
    // The text: a bright yellow; a warning: red. The box: nearly black.
    let (r, g, b): (f32, f32, f32) = match ink {
        INK_TEXT => (1.0, 0.92, 0.16),
        INK_WARN => (1.0, 0.2, 0.16),
        _ => (0.03, 0.03, 0.03),
    };
    let byte = |v: f32| (v * 255.0).round() as u8;
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM => out.copy_from_slice(&[byte(r), byte(g), byte(b), 255]),
        DXGI_FORMAT_B8G8R8A8_UNORM => out.copy_from_slice(&[byte(b), byte(g), byte(r), 255]),
        DXGI_FORMAT_R10G10B10A2_UNORM => {
            let ten = |v: f32| (v * 1023.0).round() as u32;
            out.copy_from_slice(&(ten(r) | ten(g) << 10 | ten(b) << 20 | 3 << 30).to_le_bytes());
        }
        _ => {
            for (i, v) in [r, g, b, 1.0].into_iter().enumerate() {
                out[i * 2..i * 2 + 2].copy_from_slice(&half(v).to_le_bytes());
            }
        }
    }
}

/// A value from 0 to 1 as a 16-bit float.
fn half(v: f32) -> u16 {
    let bits = v.clamp(0.0, 1.0).to_bits();
    if v <= 0.0 {
        return 0;
    }
    // Sign 0; the exponent moves from a bias of 127 to one of 15; the top
    // ten bits of the fraction stay. Values under 2^-14 (none of ours) are 0.
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    if exponent <= 0 {
        return 0;
    }
    ((exponent as u32) << 10 | (bits >> 13) & 0x3FF) as u16
}

/// The box for `text` (lines parted by `\n`) at `scale` dots a pixel: its
/// width and height in pixels and, row by row from the top, each pixel's
/// ink (the glyphs of `warn` in [`INK_WARN`]). A glyph takes six dots
/// across with its gap and a line nine down with its gap; a pad of two
/// dots lies all round.
fn bitmap(text: &str, scale: u32, warn: &Warn) -> (u32, u32, Vec<u8>) {
    let lines: Vec<Vec<char>> = text.split('\n').map(|l| l.chars().collect()).collect();
    let longest = lines.iter().map(Vec::len).max().unwrap_or(0).max(1) as u32;
    let pad = 2 * scale;
    let width = 2 * pad + longest * 6 * scale - scale;
    let height = 2 * pad + lines.len() as u32 * 9 * scale - 2 * scale;
    let mut lit = vec![INK_BOX; (width * height) as usize];
    for (line, chars) in lines.iter().enumerate() {
        for (i, &c) in chars.iter().enumerate() {
            let ink = match warn {
                Some((l, columns)) if *l == line && columns.contains(&i) => INK_WARN,
                _ => INK_TEXT,
            };
            let rows = glyph(c);
            for (row, bits) in rows.into_iter().enumerate() {
                for column in 0..5u32 {
                    if bits >> (4 - column) & 1 == 0 {
                        continue;
                    }
                    let x0 = pad + (i as u32 * 6 + column) * scale;
                    let y0 = pad + (line as u32 * 9 + row as u32) * scale;
                    for y in y0..y0 + scale {
                        for x in x0..x0 + scale {
                            lit[(y * width + x) as usize] = ink;
                        }
                    }
                }
            }
        }
    }
    (width, height, lit)
}

/// A buffer the card copies a box from, with its footprint and where its
/// bytes are mapped.
struct Made {
    buffer: ID3D12Resource,
    data: *mut u8,
    width: u32,
    height: u32,
    pitch: u32,
}

// SAFETY: a free-threaded D3D12 object, its mapping (written under
// `PANEL`'s lock only) and plain numbers.
unsafe impl Send for Made {}

/// The buffers a text is drawn into, in turn.
const RING: usize = 4;

/// The box as last drawn.
#[derive(Default)]
struct Panel {
    /// What the buffers were made for: the format, the scale, the glyphs
    /// across and the lines.
    made_for: Option<(i32, u32, usize, usize)>,
    /// The text in `ring[at]`: the frame rate, the preparing and the number
    /// of the look at the machine's load.
    shown: Option<(u32, u32, u64)>,
    ring: Vec<Made>,
    at: usize,
    /// The frame times for the 1% low, and when the last frame was drawn.
    lows: Lows,
    last: Option<std::time::Instant>,
}

static PANEL: Mutex<Option<Panel>> = Mutex::new(None);

/// Copies the box onto `back`, a back buffer of `width` by `height` pixels
/// of `format` that `list` holds as a copy's destination. Nothing if the
/// box is off, no rate was measured yet, the format is not one of ours or
/// the frame is too small for the box.
pub(crate) fn draw(host: &crate::host::Host, list: &ID3D12GraphicsCommandList, back: &ID3D12Resource, (width, height, format): (u32, u32, DXGI_FORMAT)) {
    let mode = mode();
    if mode == Mode::Off {
        return;
    }
    let mut panel = PANEL.lock().unwrap_or_else(|e| e.into_inner());
    let panel = panel.get_or_insert_with(Panel::default);
    if mode == Mode::All {
        // A frame's time, for the 1% low: from the last frame drawn.
        static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let now = std::time::Instant::now();
        if let Some(last) = panel.last {
            let at = (now - *EPOCH.get_or_init(|| last)).as_secs_f64();
            panel.lows.push(at, (now - last).as_secs_f32() * 1000.0);
        }
        panel.last = Some(now);
    }
    let tenths = TENTHS.load(Ordering::Relaxed);
    let Some(bytes) = pixel_bytes(format).filter(|_| tenths != u32::MAX) else {
        return;
    };
    // A third of the size it was (the player asked: "make it like 1/3 size"): 2 dots a
    // pixel at 2160 lines, 1 at 1080.
    let scale = (height / 1080).clamp(1, 4);
    let stats = (mode == Mode::All).then(|| {
        crate::stats::start();
        crate::stats::now()
    });
    let look = stats.flatten();
    let (columns, rows) = if mode == Mode::All { (COLUMNS, ROWS) } else { (RATE_COLUMNS, 1) };
    let made_for = Some((format.0, scale, columns, rows));
    if panel.made_for != made_for {
        // Another size of frame or another format: new buffers. A present
        // still in the air may read the old ones, so they are never let go
        // (a few hundred KiB each; this happens when the window changes).
        std::mem::forget(std::mem::take(&mut panel.ring));
        panel.made_for = made_for;
        panel.shown = None;
        panel.at = 0;
    }
    let prep = PREP.load(Ordering::Relaxed);
    let key = (tenths, prep, look.map_or(0, |(number, _)| number));
    if panel.shown != Some(key) {
        let (text, warn) = if mode == Mode::All {
            let figures = Figures {
                tenths,
                prep_ms: (prep != u32::MAX).then(|| prep as f32 / 100.0),
                low_fps: panel.lows.low_fps(),
                stats: look.map(|(_, s)| s),
                card: host.video_memory().map(|m| (m[0], m[1])),
            };
            let (lines, warn) = lines(&figures);
            (lines.join("\n"), warn)
        } else {
            (fit(&text(tenths), RATE_COLUMNS), None)
        };
        let (w, h, lit) = bitmap(&text, scale, &warn);
        let pitch = (w as usize * bytes).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as usize);
        // The next buffer in turn, made at its first use.
        let at = if panel.ring.len() < RING {
            let Some((buffer, data)) = host.shared_buffer((pitch * h as usize) as u64) else {
                return;
            };
            panel.ring.push(Made { buffer, data, width: w, height: h, pitch: pitch as u32 });
            panel.ring.len() - 1
        } else {
            (panel.at + 1) % RING
        };
        let m = &panel.ring[at];
        if (m.width, m.height, m.pitch) != (w, h, pitch as u32) {
            // (Every text of one `made_for` has one size: not reached.)
            return;
        }
        // SAFETY: the buffer's own mapped bytes, `pitch * h` of them; the
        // card last read them at least `RING - 1` texts ago.
        let out = unsafe { std::slice::from_raw_parts_mut(m.data, pitch * h as usize) };
        for y in 0..h as usize {
            for x in 0..w as usize {
                let to = y * pitch + x * bytes;
                pixel(format, lit[y * w as usize + x], &mut out[to..to + bytes]);
            }
        }
        panel.at = at;
        panel.shown = Some(key);
    }
    let Some(m) = panel.ring.get(panel.at) else {
        return;
    };
    let margin = 4 * scale;
    if m.width + 2 * margin > width || m.height + 2 * margin > height {
        return;
    }
    // SAFETY: an open list; the back buffer is a copy's destination in it
    // and the box lies inside it; our buffer holds `pitch * height` bytes
    // and is kept for as long as the process runs.
    unsafe {
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&m.buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: format, Width: m.width, Height: m.height, Depth: 1, RowPitch: m.pitch },
                },
            },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(back),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        list.CopyTextureRegion(&to, margin, margin, 0, &from, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_text_is_the_rate_to_a_tenth() {
        assert_eq!(text(103), "10.3 FPS");
        assert_eq!(text(7), "0.7 FPS");
        assert_eq!(text(300), "30.0 FPS");
    }

    #[test]
    fn the_box_holds_every_glyph_with_a_margin() {
        let (w, h, ink) = bitmap("10.3 FPS", 3, &None);
        let lit: Vec<bool> = ink.iter().map(|&i| i != INK_BOX).collect();
        assert!(ink.iter().all(|&i| i != INK_WARN), "no warning asked for");
        // Eight glyphs of five dots and seven gaps of one, a pad of two dots all round.
        assert_eq!((w, h), ((8 * 5 + 7 + 4) * 3, (7 + 4) * 3));
        assert_eq!(lit.len(), (w * h) as usize);
        // The pad is box, not text; the "1" has its stem lit.
        assert!(lit[..(w * 6) as usize].iter().all(|&l| !l), "the top pad");
        assert!((0..h).all(|y| !lit[(y * w) as usize] && !lit[(y * w + w - 1) as usize]), "the side pads");
        let at = |x: u32, y: u32| lit[(y * w + x) as usize];
        // Glyph 0 ("1"): dot column 2 of row 0 is lit, column 0 is not.
        assert!(at(6 + 2 * 3, 6) && !at(6, 6));
        // A space lights nothing: glyph 4.
        assert!((0..7 * 3).all(|y| (0..5 * 3).all(|x| !at(6 + 4 * 18 + x, 6 + y))));
        // Every glyph used has at least one dot, but the space.
        for c in "0123456789.FPS".chars() {
            assert!(glyph(c).iter().any(|&r| r != 0), "{c}");
        }
    }

    fn full_stats(ram_free_mb: u32) -> crate::stats::Stats {
        crate::stats::Stats {
            cpu_all: 100.0,
            cores: 28.0,
            top_thread: 100.0,
            gpu_all: Some(100.0),
            gpu_ours: Some(100.0),
            ram_ours_mb: 31 * 1024 + 512,
            ram_free_mb,
        }
    }

    #[test]
    fn the_whole_box_has_four_lines_of_one_length_and_a_glyph_for_every_letter() {
        let stats = full_stats(999 * 1024 + 900);
        // The longest figures there can be, the usual ones, and none.
        let longest = Figures {
            tenths: 99_999,
            prep_ms: Some(9_999.9),
            low_fps: Some(9_999.0),
            stats: Some(stats),
            card: Some((99_900 << 20, 99_900 << 20)),
        };
        let usual = Figures {
            tenths: 580,
            prep_ms: Some(4.2),
            low_fps: Some(41.4),
            stats: Some(crate::stats::Stats { cpu_all: 31.0, top_thread: 97.0, gpu_all: Some(41.0), ram_free_mb: 3_174, ..stats }),
            card: Some((10_240 << 20, 9_216 << 20)),
        };
        let no_gpu = Figures { stats: usual.stats.map(|s| crate::stats::Stats { gpu_all: None, ..s }), ..usual };
        let texts = [lines(&longest).0, lines(&usual).0, lines(&no_gpu).0, lines(&Figures::default()).0];
        assert_eq!(texts[0][0].trim_end(), "9999.9 FPS  0 MS");
        assert_eq!(texts[0][1].trim_end(), "1% LOW 9999  PREP 9999.9 MS");
        assert_eq!(texts[0][2].trim_end(), "CPU 100% TOP 100% GPU 100%");
        assert_eq!(texts[0][3].trim_end(), "VRAM 97.6/97.6 FREE 999.9 GB");
        assert_eq!(texts[1][0].trim_end(), "58.0 FPS  17 MS");
        assert_eq!(texts[1][1].trim_end(), "1% LOW 41  PREP 4.2 MS");
        assert_eq!(texts[1][2].trim_end(), "CPU 31% TOP 97% GPU 41%");
        assert_eq!(texts[1][3].trim_end(), "VRAM 9.0/10.0 FREE 3.1 GB");
        assert_eq!(texts[2][2].trim_end(), "CPU 31% TOP 97% GPU --");
        assert_eq!(texts[3][0].trim_end(), "0.0 FPS");
        assert_eq!(texts[3][1].trim_end(), "1% LOW --  PREP --");
        assert_eq!(texts[3][2].trim_end(), "CPU -- TOP -- GPU --");
        assert_eq!(texts[3][3].trim_end(), "VRAM -- FREE --");
        for lines in &texts {
            assert_eq!(lines.len(), ROWS);
            for line in lines {
                assert_eq!(line.chars().count(), COLUMNS, "{line:?}");
                for c in line.chars().filter(|c| *c != ' ') {
                    assert!(glyph(c).iter().any(|&r| r != 0), "no glyph for {c:?} in {line:?}");
                }
            }
        }
        // Four lines: nine dots a line, less the last gap, and the pad.
        let (w, h, ink) = bitmap(&texts[0].join("\n"), 3, &None);
        assert_eq!((w, h), ((COLUMNS as u32 * 6 - 1 + 4) * 3, (ROWS as u32 * 9 - 2 + 4) * 3));
        assert_eq!(ink.len(), (w * h) as usize);
        // The frame rate alone keeps its length whatever the rate.
        assert_eq!(fit(&text(7), RATE_COLUMNS).len(), fit(&text(99_999), RATE_COLUMNS).len());
    }

    #[test]
    fn free_memory_under_two_gb_is_drawn_in_the_warning_colour() {
        assert!(ram_low(0) && ram_low(2047));
        assert!(!ram_low(2048) && !ram_low(64 * 1024));
        let figures = |free| Figures { tenths: 600, stats: Some(full_stats(free)), card: Some((10_240 << 20, 9_216 << 20)), ..Default::default() };
        assert_eq!(lines(&figures(2048)).1, None);
        assert_eq!(lines(&figures(u32::MAX)).1, None);
        assert_eq!(lines(&Figures::default()).1, None, "nothing known: nothing warns");
        let (text, warn) = lines(&figures(1_500));
        let (line, columns) = warn.clone().expect("a warning");
        assert_eq!(line, 3);
        let warned: String = text[line].chars().skip(columns.start).take(columns.len()).collect();
        assert_eq!(warned, "FREE 1.5 GB");
        // In the box: the warned glyphs in the warning ink, the rest not.
        let (w, _, ink) = bitmap(&text.join("\n"), 1, &warn);
        let glyph_ink = |row: usize, column: usize| {
            let (x0, y0) = (2 + column as u32 * 6, 2 + row as u32 * 9);
            (y0..y0 + 7).flat_map(|y| (x0..x0 + 5).map(move |x| (x, y))).map(|(x, y)| ink[(y * w + x) as usize]).max().unwrap()
        };
        assert_eq!(glyph_ink(3, columns.start), INK_WARN, "the F of FREE");
        assert_eq!(glyph_ink(3, 0), INK_TEXT, "the V of VRAM");
        assert_eq!(glyph_ink(0, 0), INK_TEXT);
    }

    #[test]
    fn the_one_percent_low_is_the_slowest_frames_of_the_last_ten_seconds() {
        let mut lows = Lows::default();
        assert_eq!(lows.low_fps(), None);
        // 99 frames of 10 ms and one of 50 ms: the 99th percentile is 10 ms.
        let mut at = 0.0;
        for i in 0..100 {
            at += 0.01;
            lows.push(at, if i == 50 { 50.0 } else { 10.0 });
        }
        assert_eq!(lows.low_fps(), Some(100.0));
        // Two slow frames in a hundred and one: the slowest 1 % is 50 ms.
        at += 0.05;
        lows.push(at, 50.0);
        assert_eq!(lows.low_fps(), Some(20.0));
        // Ten seconds on, the slow frames are out of the window.
        for _ in 0..200 {
            at += 0.06;
            lows.push(at, 16.0);
        }
        assert!(lows.frames.iter().all(|&(t, _)| t >= at - LOW_WINDOW_S));
        assert_eq!(lows.low_fps(), Some(62.5));
        // Never more than the most it holds.
        for _ in 0..LOW_MOST + 10 {
            lows.push(at, 1.0);
        }
        assert_eq!(lows.frames.len(), LOW_MOST);
    }

    #[test]
    fn pixels_are_in_the_formats_the_layer_presents_in() {
        let mut four = [0u8; 4];
        pixel(DXGI_FORMAT_R8G8B8A8_UNORM, INK_TEXT, &mut four);
        assert_eq!(four, [255, 235, 41, 255]);
        pixel(DXGI_FORMAT_R8G8B8A8_UNORM, INK_WARN, &mut four);
        assert_eq!(four, [255, 51, 41, 255]);
        pixel(DXGI_FORMAT_B8G8R8A8_UNORM, INK_TEXT, &mut four);
        assert_eq!(four, [41, 235, 255, 255]);
        pixel(DXGI_FORMAT_R10G10B10A2_UNORM, INK_TEXT, &mut four);
        let v = u32::from_le_bytes(four);
        assert_eq!((v & 1023, v >> 10 & 1023, v >> 20 & 1023, v >> 30), (1023, 941, 164, 3));
        let mut eight = [0u8; 8];
        pixel(DXGI_FORMAT_R16G16B16A16_FLOAT, INK_TEXT, &mut eight);
        assert_eq!(u16::from_le_bytes([eight[0], eight[1]]), 0x3C00, "1.0 as a half");
        assert_eq!(u16::from_le_bytes([eight[6], eight[7]]), 0x3C00);
        assert_eq!(half(0.5), 0x3800);
        assert_eq!(half(0.0), 0);
        assert_eq!(pixel_bytes(DXGI_FORMAT_R16G16B16A16_FLOAT), Some(8));
        assert_eq!(pixel_bytes(DXGI_FORMAT_D32_FLOAT), None);
    }
}
