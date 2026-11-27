//! The settings menu (`VICTORY_SETTINGS_MENU=1`, ticket 174; off unless the
//! value is `1`, and only in diagnostics builds, which have the panel): F6
//! opens and closes a small box of settings over the picture while one of our
//! windows has the focus; Up/Down pick a row, Left/Right (or Enter) change it.
//! F3 cycles the performance panel Off, Minimal (one line: frame rate, frame
//! time, RAM, VRAM) and Full (the panel of `VICTORY_PANEL`).
//!
//! A row is live when the layer reads its value again every frame (the
//! stand-in's ambient level and reflection part are root constants of each
//! dispatch; the panel's mode is ours). Every other row is read once at the
//! start, so a change there is for the next launch and marked `*`: the play
//! launcher reads the settings file and sets the matching `VICTORY_*` values.
//! The file (`VICTORY_SETTINGS_FILE`, else `C:\victory-data\settings.txt`)
//! holds `key=value` lines; it is read at the start and written on every
//! change, with the rows it had and the rows changed since. Without it the
//! launcher's values stay.
//!
//! While the menu is open the title gets no keys and no mouse (after
//! trustno1-coder's 86a397600, "suppress mouse motion and buttons while the
//! F1 settings menu is open": `gameinput` reads [`SETTINGS_MENU_OPEN`]).
//!
//! What is here has no card in it: the rows, the file, the keys and the
//! pictures as dots. `host::panel` puts them on the screen.

#![cfg_attr(not(feature = "diagnostics"), allow(dead_code))]

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;

/// The menu is open: the title is given no keys, mouse motion or buttons.
pub static SETTINGS_MENU_OPEN: AtomicBool = AtomicBool::new(false);

/// Whether the menu is asked for (`VICTORY_SETTINGS_MENU=1`), in a build that
/// can show it. Read once.
pub fn menu_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_SETTINGS_MENU").is_some_and(|v| v == "1")
    })
}

/// The settings file.
pub(crate) fn file_path() -> PathBuf {
    std::env::var_os("VICTORY_SETTINGS_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\victory-data\settings.txt"))
}

/// When a row's change takes effect.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Takes {
    /// At once, always.
    Now,
    /// At once if the run started with a value of the row's own (not off,
    /// not something else); otherwise at the next launch.
    NowIfStarted,
    /// At the next launch.
    NextLaunch,
}

/// One row: its key in the file, what is shown, the values (shown, and in
/// the file), when a change takes effect and the hint at the bottom.
pub(crate) struct Row {
    pub(crate) key: &'static str,
    pub(crate) label: &'static str,
    pub(crate) shown: &'static [&'static str],
    pub(crate) values: &'static [&'static str],
    pub(crate) takes: Takes,
    pub(crate) hint: &'static str,
}

pub(crate) const FRAME_CAP: usize = 0;
pub(crate) const OUTPUT: usize = 1;
pub(crate) const TEXTURE_DETAIL: usize = 2;
pub(crate) const TEXTURE_CAP: usize = 3;
pub(crate) const MEMORY_WINDOWS: usize = 4;
pub(crate) const AMBIENT: usize = 5;
pub(crate) const SKY: usize = 6;
pub(crate) const REFLECTIONS: usize = 7;
pub(crate) const PANEL: usize = 8;
pub(crate) const COUNT: usize = 9;

/// The rows, in their order on the screen. The launcher's mapping to
/// switches is `tools\play-launchers\settings-lib.ps1` in victory-lap.
pub(crate) const ROWS: [Row; COUNT] = [
    Row {
        key: "frame_cap",
        label: "FRAME CAP",
        shown: &["30", "60", "OFF"],
        values: &["30", "60", "off"],
        // The title's rate is written into its code at load (frame_rate).
        takes: Takes::NextLaunch,
        hint: "FRAMES A SECOND THE TITLE ASKS FOR",
    },
    Row {
        key: "output",
        label: "OUTPUT SIZE",
        shown: &["1920X1080", "1600X900", "1280X720"],
        values: &["1920x1080", "1600x900", "1280x720"],
        takes: Takes::NextLaunch,
        hint: "THE PICTURE AND WINDOW SIZE",
    },
    Row {
        key: "texture_detail",
        label: "TEXTURE DETAIL",
        shown: &["SAVER", "FULL"],
        values: &["saver", "full"],
        takes: Takes::NextLaunch,
        hint: "SAVER LEAVES OUT EACH TEXTURE'S LARGEST LEVEL",
    },
    Row {
        key: "texture_cap",
        label: "TEXTURE SIZE CAP",
        shown: &["OFF", "2048", "1024"],
        values: &["off", "2048", "1024"],
        takes: Takes::NextLaunch,
        hint: "LARGEST TEXTURE SIDE IN PIXELS",
    },
    Row {
        key: "memory_windows",
        label: "MEMORY WINDOWS",
        shown: &["2048 MB", "1024 MB", "OFF"],
        values: &["2048", "1024", "off"],
        takes: Takes::NextLaunch,
        hint: "WINDOWS IN CARD MEMORY  LESS RAM",
    },
    Row {
        key: "ambient",
        label: "AMBIENT LIGHT",
        shown: &["0.6", "0.8", "1.0"],
        values: &["0.6", "0.8", "1.0"],
        takes: Takes::NowIfStarted,
        hint: "FILL LIGHT WITHOUT RAY TRACING",
    },
    Row {
        key: "sky",
        label: "SKY LIGHT",
        shown: &["ON", "OFF"],
        values: &["on", "off"],
        // The sky's shaders are picked when pipelines are made.
        takes: Takes::NextLaunch,
        hint: "THE FILL LIGHT FOLLOWS THE SKY",
    },
    Row {
        key: "reflections",
        label: "REFLECTIONS",
        shown: &["OFF", "0.5", "1.0"],
        values: &["0", "0.5", "1.0"],
        takes: Takes::NowIfStarted,
        hint: "HELD TO THE LIGHT AROUND",
    },
    Row {
        key: "panel",
        label: "PERFORMANCE PANEL",
        shown: &["OFF", "MINIMAL", "FULL"],
        values: &["off", "minimal", "full"],
        takes: Takes::Now,
        hint: "F3 ALSO CYCLES IT",
    },
];

/// What a row runs with at the start: one of its values, or something
/// else, as shown.
#[derive(Clone, PartialEq, Debug)]
pub(crate) enum Start {
    Value(usize),
    Other(String),
}

/// The row's value for a number, if it is one of them.
pub(crate) fn number_value(row: usize, v: f64) -> Start {
    ROWS[row]
        .values
        .iter()
        .position(|s| s.parse::<f64>().is_ok_and(|n| (n - v).abs() < 1e-3))
        .map_or_else(|| Start::Other(format!("{v:.2}")), Start::Value)
}

/// The row's value for a word of the file.
pub(crate) fn file_value(row: usize, value: &str) -> Option<usize> {
    let value = value.trim();
    ROWS[row].values.iter().position(|v| v.eq_ignore_ascii_case(value)).or_else(|| {
        // Numbers as numbers: `1` is `1.0`, `0.50` is `0.5`.
        let n = value.parse::<f64>().ok()?;
        ROWS[row].values.iter().position(|v| v.parse::<f64>().is_ok_and(|m| (m - n).abs() < 1e-6))
    })
}

/// `key=value` lines; blank lines and `#` notes left out, spaces around
/// either side trimmed.
pub(crate) fn parse(text: &str) -> Vec<(String, String)> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_owned()))
        .filter(|(k, _)| !k.is_empty())
        .collect()
}

/// A key of the menu.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Key {
    F3,
    F6,
    Up,
    Down,
    Left,
    Right,
    Enter,
}

impl Key {
    pub(crate) const ALL: [Key; 7] = [Key::F3, Key::F6, Key::Up, Key::Down, Key::Left, Key::Right, Key::Enter];

    /// Its virtual key.
    pub(crate) const fn vk(self) -> u16 {
        match self {
            Key::F3 => 0x72,
            Key::F6 => 0x75,
            Key::Up => 0x26,
            Key::Down => 0x28,
            Key::Left => 0x25,
            Key::Right => 0x27,
            Key::Enter => 0x0D,
        }
    }
}

/// Presses from the keys' states, one bit per [`Key::ALL`]: a key counts
/// when it goes down, or was tapped between two looks (the low bit of
/// `GetAsyncKeyState`), not again while it is held.
#[derive(Default)]
pub(crate) struct Edges {
    was: u8,
}

impl Edges {
    pub(crate) fn pressed(&mut self, down: u8, tapped: u8) -> u8 {
        let new = (down | tapped) & !self.was;
        self.was = down;
        new
    }
}

/// What a press did.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) struct Outcome {
    /// The menu's picture changed.
    pub(crate) redraw: bool,
    /// This row's value changed.
    pub(crate) changed: Option<usize>,
}

/// The menu's state.
pub(crate) struct Menu {
    pub(crate) open: bool,
    pub(crate) picked: usize,
    /// Each row's value (index into its values); `None`: what it started
    /// with is none of them ([`Menu::other`]).
    pub(crate) value: [Option<usize>; COUNT],
    other: [String; COUNT],
    /// Changes to the row act now.
    pub(crate) live: [bool; COUNT],
    /// The row goes into the file: it was there, or was changed.
    touched: [bool; COUNT],
    /// Lines of the file that are none of the rows, kept as they were.
    kept: Vec<(String, String)>,
}

impl Menu {
    /// From what the run started with and the file's lines. A next-launch
    /// row shows the file's value (what the next launch takes); a live row
    /// what runs.
    pub(crate) fn new(start: [Start; COUNT], file: &[(String, String)]) -> Menu {
        let mut menu = Menu {
            open: false,
            picked: 0,
            value: [None; COUNT],
            other: Default::default(),
            live: [false; COUNT],
            touched: [false; COUNT],
            kept: Vec::new(),
        };
        for (i, s) in start.into_iter().enumerate() {
            // With one of its values, and not off.
            let started = matches!(s, Start::Value(v) if ROWS[i].values[v].parse::<f64>().is_ok_and(|n| n > 0.0));
            match s {
                Start::Value(v) => menu.value[i] = Some(v),
                Start::Other(text) => menu.other[i] = text,
            }
            menu.live[i] = match ROWS[i].takes {
                Takes::Now => true,
                Takes::NowIfStarted => started,
                Takes::NextLaunch => false,
            };
        }
        for (key, value) in file {
            match ROWS.iter().position(|r| r.key == key) {
                Some(i) => {
                    menu.touched[i] = true;
                    if !menu.live[i]
                        && let Some(v) = file_value(i, value)
                    {
                        menu.value[i] = Some(v);
                    }
                }
                None => menu.kept.push((key.clone(), value.clone())),
            }
        }
        menu
    }

    pub(crate) fn press(&mut self, key: Key) -> Outcome {
        match key {
            Key::F6 => {
                self.open = !self.open;
                return Outcome { redraw: true, changed: None };
            }
            Key::F3 => return self.step(PANEL, 1),
            _ if !self.open => return Outcome::default(),
            Key::Up => self.picked = (self.picked + COUNT - 1) % COUNT,
            Key::Down => self.picked = (self.picked + 1) % COUNT,
            Key::Left => return self.step(self.picked, -1),
            Key::Right | Key::Enter => return self.step(self.picked, 1),
        }
        Outcome { redraw: true, changed: None }
    }

    /// The row's value one on (or back), round.
    fn step(&mut self, row: usize, by: i32) -> Outcome {
        let n = ROWS[row].values.len();
        self.value[row] = Some(match (self.value[row], by > 0) {
            (Some(v), true) => (v + 1) % n,
            (Some(v), false) => (v + n - 1) % n,
            (None, true) => 0,
            (None, false) => n - 1,
        });
        self.touched[row] = true;
        Outcome { redraw: self.open, changed: Some(row) }
    }

    /// What a next-launch row runs with, learnt after the start (the output
    /// size, from the first picture shown). A value the file has for the
    /// next launch stays shown.
    pub(crate) fn learnt(&mut self, row: usize, start: Start) {
        if self.touched[row] {
            return;
        }
        match start {
            Start::Value(v) => self.value[row] = Some(v),
            Start::Other(text) => {
                self.value[row] = None;
                self.other[row] = text;
            }
        }
    }

    /// The row's value as shown.
    pub(crate) fn shown(&self, row: usize) -> &str {
        match self.value[row] {
            Some(v) => ROWS[row].shown[v],
            None => &self.other[row],
        }
    }

    /// The text of the settings file.
    pub(crate) fn file_text(&self) -> String {
        let mut text = String::from("# VIctory settings (F6 menu, ticket 174): the play launcher reads them at the start.\n");
        for (i, row) in ROWS.iter().enumerate() {
            if let (true, Some(v)) = (self.touched[i], self.value[i]) {
                text += &format!("{}={}\n", row.key, row.values[v]);
            }
        }
        for (k, v) in &self.kept {
            text += &format!("{k}={v}\n");
        }
        text
    }

    /// The lines of the box: label, value (between `< >` when picked, `*`
    /// when it is for the next launch), whether picked; and the hint.
    pub(crate) fn lines(&self) -> (Vec<(&'static str, String, bool)>, String) {
        let rows = ROWS
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let picked = i == self.picked;
                let v = self.shown(i);
                let v = if picked { format!("< {v} >") } else { v.to_owned() };
                let star = if self.live[i] { "" } else { " *" };
                (row.label, format!("{v}{star}"), picked)
            })
            .collect();
        let row = &ROWS[self.picked];
        let hint = if self.live[self.picked] {
            format!("{}  LIVE", row.hint)
        } else if row.takes == Takes::NowIfStarted {
            format!("{}  OFF AT START", row.hint)
        } else {
            row.hint.to_owned()
        };
        (rows, hint)
    }
}

/// Writes the file next to itself and moves it over the old one.
pub(crate) fn write_file(path: &std::path::Path, text: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let fresh = path.with_extension("txt.new");
    std::fs::write(&fresh, text)?;
    std::fs::rename(&fresh, path)
}

/// The panel's modes, as F3 cycles them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Hud {
    Off,
    Minimal,
    Full,
}

impl Hud {
    pub(crate) fn of(value: Option<usize>) -> Hud {
        match value {
            Some(1) => Hud::Minimal,
            Some(2) => Hud::Full,
            _ => Hud::Off,
        }
    }
}

/// The minimal panel's one line.
pub(crate) fn minimal_text(fps: f64, ms: f64, ram_gb: f64, vram_gb: f64, budget_gb: f64) -> String {
    format!("{fps:.1} FPS  {ms:.0} MS  RAM {ram_gb:.1}  VRAM {vram_gb:.1}/{budget_gb:.1}")
}

// The pictures, in dots: a dot is `scale` pixels each way on the screen.

/// A character's dots, and with the space beside and below it.
pub(crate) const GLYPH: (u32, u32) = (5, 7);
pub(crate) const CELL: (u32, u32) = (6, 9);

/// Seven rows of five dots, the leftmost the highest bit.
#[rustfmt::skip]
pub(crate) const fn glyph(c: u8) -> [u8; 7] {
    match c {
        b'0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        b'1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        b'2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        b'3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        b'4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        b'5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        b'6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        b'7' => [0b11111, 0b00001, 0b00010, 0b00010, 0b00100, 0b00100, 0b00100],
        b'8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        b'9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        b'A' => [0b00100, 0b01010, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001],
        b'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        b'C' => [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
        b'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        b'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        b'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        b'G' => [0b01111, 0b10000, 0b10000, 0b10011, 0b10001, 0b10001, 0b01111],
        b'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        b'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        b'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        b'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        b'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        b'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        b'N' => [0b10001, 0b11001, 0b11001, 0b10101, 0b10011, 0b10011, 0b10001],
        b'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        b'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        b'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        b'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        b'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        b'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        b'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        b'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        b'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        b'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        b'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        b'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        b'.' => [0, 0, 0, 0, 0, 0b01100, 0b01100],
        b'/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        b'-' => [0, 0, 0, 0b11111, 0, 0, 0],
        b'<' => [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
        b'>' => [0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000],
        b'*' => [0, 0b00100, 0b10101, 0b01110, 0b10101, 0b00100, 0],
        b'\'' => [0b00100, 0b00100, 0b01000, 0, 0, 0, 0],
        b' ' => [0; 7],
        _ => [0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111],
    }
}

/// Colours of the boxes (sRGB). [`Dots`] holds their places in this list.
pub(crate) const PALETTE: [[u8; 3]; 9] = [
    [0x12, 0x08, 0x1f], // the box
    [0xff, 0x3e, 0x8a], // its edge, the picked row's bar
    [0xff, 0x5f, 0xa2], // the header
    [0x4a, 0x15, 0x46], // the picked row
    [0xff, 0xd1, 0x66], // the picked value, the hint
    [0xf2, 0xf2, 0xf2], // text
    [0xc9, 0xa8, 0xd6], // the footer
    [0x5a, 0x2a, 0x5e], // rules
    [0xff, 0xff, 0xff], // the minimal panel's text
];
const BOX: u8 = 0;
const EDGE: u8 = 1;
const HEADER: u8 = 2;
const PICKED: u8 = 3;
const GOLD: u8 = 4;
const TEXT: u8 = 5;
const FOOTER: u8 = 6;
const RULE: u8 = 7;
const WHITE: u8 = 8;

/// A picture of `w` by `h` dots, each a place in [`PALETTE`].
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Dots {
    pub(crate) w: u32,
    pub(crate) h: u32,
    dots: Vec<u8>,
}

impl Dots {
    fn new(w: u32, h: u32, colour: u8) -> Dots {
        Dots { w, h, dots: vec![colour; (w * h) as usize] }
    }

    pub(crate) fn at(&self, x: u32, y: u32) -> u8 {
        self.dots[(y * self.w + x) as usize]
    }

    fn rect(&mut self, x: u32, y: u32, w: u32, h: u32, colour: u8) {
        for y in y..(y + h).min(self.h) {
            for x in x..(x + w).min(self.w) {
                self.dots[(y * self.w + x) as usize] = colour;
            }
        }
    }

    /// Capitals; what does not fit is cut.
    fn text(&mut self, x: u32, y: u32, text: &str, colour: u8) {
        for (i, c) in text.bytes().enumerate() {
            let rows = glyph(c.to_ascii_uppercase());
            let left = x + i as u32 * CELL.0;
            for (dy, row) in rows.iter().enumerate() {
                for dx in 0..GLYPH.0 {
                    let (px, py) = (left + dx, y + dy as u32);
                    if (row >> (GLYPH.0 - 1 - dx)) & 1 != 0 && px < self.w && py < self.h {
                        self.dots[(py * self.w + px) as usize] = colour;
                    }
                }
            }
        }
    }

    /// As rows of `pitch` bytes, a dot `scale` pixels each way, a pixel of
    /// colour `c` being `pixels[c]`.
    pub(crate) fn scaled(&self, scale: u32, pixels: &[Vec<u8>], pitch: usize) -> Vec<u8> {
        let bytes = pixels[0].len();
        let (width, height) = (self.w * scale, self.h * scale);
        let mut out = vec![0; pitch * height as usize];
        for y in 0..self.h {
            let top = pitch * (y * scale) as usize;
            for x in 0..self.w {
                let colour = &pixels[usize::from(self.at(x, y))];
                for i in 0..scale {
                    let at = top + (x * scale + i) as usize * bytes;
                    out[at..at + bytes].copy_from_slice(colour);
                }
            }
            for i in 1..scale as usize {
                out.copy_within(top..top + width as usize * bytes, top + i * pitch);
            }
        }
        out
    }
}

/// The menu box's characters in a line.
const MENU_COLUMNS: u32 = 46;
const MENU_PAD: u32 = 4;
const ROW_HEIGHT: u32 = 11;

/// The menu box's size in dots.
pub(crate) const fn menu_size() -> (u32, u32) {
    let w = 2 + 2 * MENU_PAD + MENU_COLUMNS * CELL.0;
    // Edge, pad, header, rule, rows, rule, hint and two footer lines, pad, edge.
    let h = 1 + 3 + CELL.1 + 2 + 1 + 2 + COUNT as u32 * ROW_HEIGHT + 2 + 1 + 3 + 3 * CELL.1 + 2 + 1;
    (w, h)
}

/// The menu as dots.
pub(crate) fn menu_picture(menu: &Menu) -> Dots {
    let (w, h) = menu_size();
    let mut d = Dots::new(w, h, BOX);
    // The 1-dot edge (2 pixels at 1080 lines).
    d.rect(0, 0, w, 1, EDGE);
    d.rect(0, h - 1, w, 1, EDGE);
    d.rect(0, 0, 1, h, EDGE);
    d.rect(w - 1, 0, 1, h, EDGE);
    let left = 1 + MENU_PAD;
    let right = w - 1 - MENU_PAD;
    let mut y = 1 + 3;
    d.text(left, y, "VICTORY SETTINGS", HEADER);
    y += CELL.1 + 2;
    d.rect(1 + 2, y, w - 6, 1, RULE);
    y += 1 + 2;
    let (rows, hint) = menu.lines();
    for (label, value, picked) in rows {
        let (text, value_colour) = if picked {
            d.rect(1, y, w - 2, ROW_HEIGHT, PICKED);
            // The 2-dot bar (4 pixels at 1080 lines).
            d.rect(1, y, 2, ROW_HEIGHT, EDGE);
            (TEXT, GOLD)
        } else {
            (TEXT, TEXT)
        };
        d.text(left, y + 2, label, text);
        let width = value.len() as u32 * CELL.0 - 1;
        d.text(right.saturating_sub(width), y + 2, &value, value_colour);
        y += ROW_HEIGHT;
    }
    y += 2;
    d.rect(1 + 2, y, w - 6, 1, RULE);
    y += 1 + 3;
    d.text(left, y, &hint, GOLD);
    d.text(left, y + CELL.1, "UP/DOWN PICK  LEFT/RIGHT CHANGE  F6 CLOSE", FOOTER);
    d.text(left, y + 2 * CELL.1, "* AT THE NEXT LAUNCH", FOOTER);
    d
}

/// The minimal panel's characters.
const MINIMAL_COLUMNS: u32 = 44;

pub(crate) const fn minimal_size() -> (u32, u32) {
    (2 + 3 + MINIMAL_COLUMNS * CELL.0 + 2, 2 + GLYPH.1 + 2)
}

/// The minimal panel as dots: a pink bar on the left, the line in white.
pub(crate) fn minimal_picture(line: &str) -> Dots {
    let (w, h) = minimal_size();
    let mut d = Dots::new(w, h, BOX);
    d.rect(0, 0, 2, h, EDGE);
    d.text(2 + 3, 2, line, WHITE);
    d
}

/// How the bytes of a pixel are laid out.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Layout {
    Rgba8,
    Bgra8,
    Rgb10a2,
    /// Half floats, linear (scRGB: 1 is the white of sRGB).
    Rgba16f,
}

/// A colour's bytes, opaque.
pub(crate) fn pixel(layout: Layout, [r, g, b]: [u8; 3]) -> Vec<u8> {
    match layout {
        Layout::Rgba8 => vec![r, g, b, 0xFF],
        Layout::Bgra8 => vec![b, g, r, 0xFF],
        Layout::Rgb10a2 => {
            let ten = |c: u8| (u32::from(c) * 1023 + 127) / 255;
            (ten(r) | ten(g) << 10 | ten(b) << 20 | 3 << 30).to_le_bytes().to_vec()
        }
        Layout::Rgba16f => {
            let mut out = Vec::with_capacity(8);
            for c in [linear(r), linear(g), linear(b), 1.0] {
                out.extend_from_slice(&half(c).to_le_bytes());
            }
            out
        }
    }
}

/// The palette's pixels.
pub(crate) fn pixels(layout: Layout) -> Vec<Vec<u8>> {
    PALETTE.iter().map(|&c| pixel(layout, c)).collect()
}

fn linear(c: u8) -> f32 {
    let c = f32::from(c) / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// A half float of `v` from 0 to 65504, rounded down.
fn half(v: f32) -> u16 {
    let v = v.clamp(0.0, 65504.0);
    if v < 6.103_515_6e-5 {
        // Below the smallest normal half: steps of 2^-24.
        return (v * 16_777_216.0).round() as u16;
    }
    let bits = v.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    ((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn starts() -> [Start; COUNT] {
        // As play29 starts: 30 (the title's), 1920x1080, saver, no cap, 2048 MB,
        // 0.8, sky on, reflections 0.5, panel full.
        [0, 0, 0, 0, 0, 1, 0, 1, 2].map(Start::Value)
    }

    #[test]
    fn the_file_is_read_with_notes_blanks_and_spaces() {
        let lines = parse("# a note\n\n Output = 1600x900 \nambient=1\nnonsense\n=x\nfuture_key=7\r\n");
        assert_eq!(
            lines,
            [("output", "1600x900"), ("ambient", "1"), ("future_key", "7")].map(|(k, v)| (k.to_owned(), v.to_owned()))
        );
        assert_eq!(file_value(OUTPUT, "1600X900"), Some(1));
        assert_eq!(file_value(AMBIENT, "1"), Some(2));
        assert_eq!(file_value(REFLECTIONS, "0.50"), Some(1));
        assert_eq!(file_value(REFLECTIONS, "0"), Some(0));
        assert_eq!(file_value(MEMORY_WINDOWS, "4096"), None);
    }

    #[test]
    fn without_a_file_nothing_is_written_until_a_row_changes() {
        let mut menu = Menu::new(starts(), &[]);
        assert_eq!(menu.file_text().lines().filter(|l| !l.starts_with('#')).count(), 0);
        menu.press(Key::F6);
        menu.press(Key::Down);
        menu.press(Key::Right);
        assert_eq!(parse(&menu.file_text()), [("output".to_owned(), "1600x900".to_owned())]);
    }

    #[test]
    fn the_file_keeps_its_rows_and_lines_it_does_not_know() {
        let file = parse("texture_cap=1024\nfuture_key=7\nambient=0.6\n");
        let mut menu = Menu::new(starts(), &file);
        // A next-launch row shows the file's value; a live row what runs.
        assert_eq!(menu.shown(TEXTURE_CAP), "1024");
        assert_eq!(menu.shown(AMBIENT), "0.8");
        menu.press(Key::F3);
        let written = parse(&menu.file_text());
        assert!(written.contains(&("texture_cap".into(), "1024".into())));
        assert!(written.contains(&("ambient".into(), "0.8".into())));
        assert!(written.contains(&("panel".into(), "off".into())));
        assert!(written.contains(&("future_key".into(), "7".into())));
        // Read back by a run the launcher started with it, the same.
        let mut again = starts();
        again[PANEL] = Start::Value(0);
        assert_eq!(parse(&Menu::new(again, &written).file_text()), written);
    }

    #[test]
    fn keys_pick_and_change_only_while_open() {
        let mut menu = Menu::new(starts(), &[]);
        assert_eq!(menu.press(Key::Down), Outcome::default());
        assert_eq!(menu.press(Key::Right), Outcome::default());
        assert_eq!(menu.picked, 0);
        assert_eq!(menu.press(Key::F6), Outcome { redraw: true, changed: None });
        assert!(menu.open);
        // Up from the first row is the last.
        menu.press(Key::Up);
        assert_eq!(menu.picked, PANEL);
        menu.press(Key::Down);
        assert_eq!(menu.picked, FRAME_CAP);
        assert_eq!(menu.press(Key::Left), Outcome { redraw: true, changed: Some(FRAME_CAP) });
        assert_eq!(menu.shown(FRAME_CAP), "OFF");
        menu.press(Key::Enter);
        assert_eq!(menu.shown(FRAME_CAP), "30");
        menu.press(Key::F6);
        assert!(!menu.open);
    }

    #[test]
    fn f3_cycles_the_panel_open_or_closed() {
        let mut menu = Menu::new(starts(), &[]);
        assert_eq!(Hud::of(menu.value[PANEL]), Hud::Full);
        assert_eq!(menu.press(Key::F3), Outcome { redraw: false, changed: Some(PANEL) });
        assert_eq!(Hud::of(menu.value[PANEL]), Hud::Off);
        menu.press(Key::F3);
        assert_eq!(Hud::of(menu.value[PANEL]), Hud::Minimal);
        menu.press(Key::F6);
        assert_eq!(menu.press(Key::F3), Outcome { redraw: true, changed: Some(PANEL) });
        assert_eq!(Hud::of(menu.value[PANEL]), Hud::Full);
    }

    #[test]
    fn rows_are_live_only_where_the_value_is_read_each_frame() {
        let menu = Menu::new(starts(), &[]);
        let live: Vec<usize> = (0..COUNT).filter(|&i| menu.live[i]).collect();
        assert_eq!(live, [AMBIENT, REFLECTIONS, PANEL]);
        // Reflections off at the start: their programs were never marked, so for the next launch.
        let mut start = starts();
        start[REFLECTIONS] = Start::Value(0);
        start[AMBIENT] = Start::Other("OFF".into());
        let menu = Menu::new(start, &[]);
        assert!(!menu.live[REFLECTIONS] && !menu.live[AMBIENT]);
        assert_eq!(menu.shown(AMBIENT), "OFF");
        let (rows, _) = menu.lines();
        assert_eq!(rows[AMBIENT].1, "OFF *");
        assert_eq!(rows[FRAME_CAP].1, "< 30 > *");
        assert_eq!(rows[PANEL].1, "FULL");
    }

    #[test]
    fn a_value_none_of_the_rows_has_is_shown_and_stepped_from() {
        assert_eq!(number_value(AMBIENT, 0.8), Start::Value(1));
        assert_eq!(number_value(MEMORY_WINDOWS, 1024.0), Start::Value(1));
        let mut start = starts();
        start[TEXTURE_CAP] = Start::Other("512".into());
        let mut menu = Menu::new(start, &[]);
        assert_eq!(menu.shown(TEXTURE_CAP), "512");
        menu.press(Key::F6);
        (0..TEXTURE_CAP).for_each(|_| _ = menu.press(Key::Down));
        menu.press(Key::Right);
        assert_eq!(menu.shown(TEXTURE_CAP), "OFF");
    }

    #[test]
    fn the_output_size_learnt_later_gives_way_to_the_file() {
        let mut start = starts();
        start[OUTPUT] = Start::Other("?".into());
        let mut menu = Menu::new(start.clone(), &[]);
        menu.learnt(OUTPUT, Start::Value(2));
        assert_eq!(menu.shown(OUTPUT), "1280X720");
        let mut menu = Menu::new(start, &parse("output=1600x900"));
        menu.learnt(OUTPUT, Start::Other("2560X1440".into()));
        assert_eq!(menu.shown(OUTPUT), "1600X900");
    }

    #[test]
    fn a_key_counts_once_when_it_goes_down_or_was_tapped() {
        let mut e = Edges::default();
        assert_eq!(e.pressed(0b01, 0), 0b01);
        assert_eq!(e.pressed(0b01, 0b01), 0, "held");
        assert_eq!(e.pressed(0, 0), 0);
        // Pressed and let go between two looks.
        assert_eq!(e.pressed(0, 0b10), 0b10);
        assert_eq!(e.pressed(0b11, 0), 0b11);
    }

    #[test]
    fn the_file_is_written_whole() {
        let dir = std::env::temp_dir().join(format!("victory-settings-{}", std::process::id()));
        let path = dir.join("settings.txt");
        write_file(&path, "a=1\n").unwrap();
        write_file(&path, "b=2\n").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "b=2\n");
        assert!(!path.with_extension("txt.new").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn characters_differ() {
        let used: Vec<u8> = (b'0'..=b'9').chain(b'A'..=b'Z').chain(*b"./-<>*' ").collect();
        for (i, a) in used.iter().enumerate() {
            assert_ne!(glyph(*a), glyph(b'#'), "{} has a picture", *a as char);
            assert!(glyph(*a).iter().all(|row| row >> GLYPH.0 == 0));
            for b in &used[i + 1..] {
                assert_ne!(glyph(*a), glyph(*b), "{} {}", *a as char, *b as char);
            }
        }
    }

    #[test]
    fn every_line_of_the_menu_fits_its_box() {
        let mut menu = Menu::new(starts(), &[]);
        let (w, _) = menu_size();
        let inner = (w - 2 - 2 * MENU_PAD) / CELL.0;
        for picked in 0..COUNT {
            menu.picked = picked;
            let (rows, hint) = menu.lines();
            assert!(hint.len() as u32 <= inner, "{hint}");
            for (label, value, _) in rows {
                assert!(label.len() + value.len() + 2 <= inner as usize, "{label} {value}");
            }
        }
        assert!(minimal_text(99.9, 999.0, 99.9, 99.9, 99.9).len() as u32 <= MINIMAL_COLUMNS);
        // About 560 pixels wide at 1080 lines (2 pixels a dot).
        assert!((540..=580).contains(&(w * 2)), "{}", w * 2);
    }

    #[test]
    fn the_menu_is_drawn_in_its_colours() {
        let mut menu = Menu::new(starts(), &[]);
        menu.picked = AMBIENT;
        let d = menu_picture(&menu);
        let (w, h) = menu_size();
        assert_eq!((d.w, d.h), (w, h));
        assert_eq!([d.at(0, 0), d.at(w - 1, h - 1)], [EDGE, EDGE]);
        assert_eq!(d.at(w / 2, 2), BOX);
        // The header's V, top left dot, in pink.
        assert_eq!(d.at(1 + MENU_PAD, 4), HEADER);
        // The picked row: its bar, its ground, and gold in its value.
        let top = 1 + 3 + CELL.1 + 2 + 1 + 2 + AMBIENT as u32 * ROW_HEIGHT;
        assert_eq!(d.at(1, top + 5), EDGE);
        assert_eq!(d.at(w / 2, top), PICKED);
        let row: Vec<u8> = (0..w).flat_map(|x| (top..top + ROW_HEIGHT).map(move |y| (x, y))).map(|(x, y)| d.at(x, y)).collect();
        assert!(row.contains(&GOLD) && row.contains(&TEXT));
        // A row not picked has no gold.
        let other = top + ROW_HEIGHT;
        assert!((0..w).flat_map(|x| (other..other + ROW_HEIGHT).map(move |y| (x, y))).all(|(x, y)| d.at(x, y) != GOLD));
        let m = minimal_picture(&minimal_text(12.6, 79.0, 18.4, 6.9, 9.0));
        assert_eq!((m.w, m.h), minimal_size());
        assert_eq!((m.at(0, 0), m.at(5, 0)), (EDGE, BOX));
        assert!((0..m.w).any(|x| m.at(x, 2) == WHITE));
    }

    #[test]
    fn pixels_in_every_layout() {
        assert_eq!(pixel(Layout::Rgba8, [1, 2, 3]), [1, 2, 3, 0xFF]);
        assert_eq!(pixel(Layout::Bgra8, [1, 2, 3]), [3, 2, 1, 0xFF]);
        // Black and white as the panel has them.
        assert_eq!(pixel(Layout::Rgb10a2, [0, 0, 0]), [0, 0, 0, 0xC0]);
        assert_eq!(pixel(Layout::Rgb10a2, [255; 3]), [0xFF; 4]);
        assert_eq!(pixel(Layout::Rgba16f, [255; 3]), [0, 0x3C, 0, 0x3C, 0, 0x3C, 0, 0x3C]);
        assert_eq!(pixel(Layout::Rgba16f, [0; 3]), [0, 0, 0, 0, 0, 0, 0, 0x3C]);
        // 0.5 is 0x3800.
        assert_eq!(half(0.5), 0x3800);
        assert!(half(linear(0x12)) > 0 && half(linear(0x12)) < 0x2400);
    }

    #[test]
    fn dots_are_scaled_into_rows_of_the_pitch() {
        let d = minimal_picture("1");
        let pixels = pixels(Layout::Rgba8);
        for scale in [1, 2, 3] {
            let pitch = (d.w * scale * 4).next_multiple_of(256) as usize;
            let out = d.scaled(scale, &pixels, pitch);
            assert_eq!(out.len(), pitch * (d.h * scale) as usize);
            let at = |x: u32, y: u32| &out[y as usize * pitch + x as usize * 4..][..4];
            assert_eq!(at(0, 0), pixels[EDGE as usize]);
            assert_eq!(at(scale * 2 - 1, scale * d.h - 1), pixels[EDGE as usize]);
            assert_eq!(at(scale * 2, 0), pixels[BOX as usize]);
            // The 1's top dot (third column of its glyph).
            assert_eq!(at((5 + 2) * scale, 2 * scale), pixels[WHITE as usize]);
            assert_eq!(at((5 + 2) * scale + scale - 1, 2 * scale + scale - 1), pixels[WHITE as usize]);
        }
    }
}
