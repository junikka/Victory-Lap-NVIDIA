//! Recording a person's input and playing it back.
//!
//! `VICTORY_INPUT_RECORD=<path>` writes host pad 0's raw state (and the
//! host keyboard's keys) to a text file each time it changes, and again
//! each second it holds; `VICTORY_INPUT_REPLAY=<path>` hands such a file
//! back as pad 0 and the keyboard instead of the host's pads, keys and
//! `VICTORY_KEYS` (`VICTORY_INPUT_REPLAY_BY=frame`
//! keys it by presented frame rather than by time). Both clocks start at
//! the first input poll, as `VICTORY_KEYS` does.
//!
//! The file is one line per change, fields split by spaces:
//!
//! ```text
//! # comment
//! <ms> <frame> <buttons> <lt> <rt> <lx> <ly> <rx> <ry>   pad 0's state
//! <ms> <frame> off                                       pad 0 gone
//! <ms> <frame> keys <vk>,<vk>,...                        keys held (`-`: none)
//! ```
//!
//! `ms` counts from the first poll, `frame` is the presented-frame count then
//! (`-` when unknown), `buttons` the XInput button bits in hex, triggers
//! 0-255, stick axes -32768 to 32767 (up positive), virtual keys in hex.

use std::io::Write;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

/// A raw XInput pad state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Raw {
    pub buttons: u16,
    pub lt: u8,
    pub rt: u8,
    pub lx: i16,
    pub ly: i16,
    pub rx: i16,
    pub ry: i16,
}

/// What a line says changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Change {
    /// Pad 0's state; `None` when it is not connected.
    Pad(Option<Raw>),
    /// Virtual keys held.
    Keys(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Line {
    pub ms: u64,
    pub frame: Option<u64>,
    pub change: Change,
}

impl Line {
    pub fn format(&self) -> String {
        let frame = self.frame.map_or("-".to_owned(), |f| f.to_string());
        let rest = match &self.change {
            Change::Pad(None) => "off".to_owned(),
            Change::Pad(Some(p)) => {
                format!("{:04x} {} {} {} {} {} {}", p.buttons, p.lt, p.rt, p.lx, p.ly, p.rx, p.ry)
            }
            Change::Keys(k) if k.is_empty() => "keys -".to_owned(),
            Change::Keys(k) => {
                let list: Vec<String> = k.iter().map(|vk| format!("{vk:02x}")).collect();
                format!("keys {}", list.join(","))
            }
        };
        format!("{} {frame} {rest}", self.ms)
    }

    /// A line as written by [`Line::format`]; `None` for anything else.
    pub fn parse(text: &str) -> Option<Line> {
        let mut words = text.split_whitespace();
        let ms = words.next()?.parse().ok()?;
        let frame = match words.next()? {
            "-" => None,
            f => Some(f.parse().ok()?),
        };
        let change = match words.next()? {
            "off" => Change::Pad(None),
            "keys" => match words.next()? {
                "-" => Change::Keys(Vec::new()),
                list => Change::Keys(
                    list.split(',').map(|k| u8::from_str_radix(k, 16).ok()).collect::<Option<_>>()?,
                ),
            },
            buttons => {
                let mut n = || words.next();
                Change::Pad(Some(Raw {
                    buttons: u16::from_str_radix(buttons, 16).ok()?,
                    lt: n()?.parse().ok()?,
                    rt: n()?.parse().ok()?,
                    lx: n()?.parse().ok()?,
                    ly: n()?.parse().ok()?,
                    rx: n()?.parse().ok()?,
                    ry: n()?.parse().ok()?,
                }))
            }
        };
        words.next().is_none().then_some(Line { ms, frame, change })
    }
}

/// The presented-frame count, when the GPU side has one.
pub(super) fn frame() -> Option<u64> {
    Some(victory_gpu::queue::presented())
}

/// How often a held state is written again, so a replay knows where the
/// recording ended and can check its drift on a still pad.
pub(super) const REPEAT: Duration = Duration::from_secs(1);

/// How often the writer thread flushes the file.
const FLUSH: Duration = Duration::from_secs(1);

/// A recording being written: lines go over a channel to a writer thread,
/// which writes them buffered and flushes every [`FLUSH`].
pub(super) struct Recorder {
    start: Instant,
    send: Sender<String>,
    last_pad: Option<(Option<Raw>, Instant)>,
    last_keys: Option<Vec<u8>>,
}

impl Recorder {
    /// From `VICTORY_INPUT_RECORD`; `None` when unset or the file cannot be
    /// made.
    pub fn from_env() -> Option<Recorder> {
        let path = std::path::PathBuf::from(std::env::var_os("VICTORY_INPUT_RECORD")?);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let file = match std::fs::File::create(&path) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("VICTORY_INPUT_RECORD: cannot create {}: {e}", path.display());
                return None;
            }
        };
        tracing::info!("recording pad 0 and the keyboard to {} (VICTORY_INPUT_RECORD)", path.display());
        let (send, receive) = std::sync::mpsc::channel();
        let _ = send.send(
            "# input recording: ms frame buttons lt rt lx ly rx ry | ms frame off | ms frame keys vk,..".to_owned(),
        );
        std::thread::Builder::new()
            .name("input recorder".into())
            .spawn(move || write_lines(receive, std::io::BufWriter::new(file)))
            .ok()?;
        Some(Recorder { start: Instant::now(), send, last_pad: None, last_keys: None })
    }

    fn line(&self, now: Instant, change: Change) {
        let ms = now.saturating_duration_since(self.start).as_millis() as u64;
        let _ = self.send.send(Line { ms, frame: frame(), change }.format());
    }

    /// Pad 0 now; written when it changed or held for [`REPEAT`].
    pub fn pad(&mut self, pad: Option<Raw>) {
        let now = Instant::now();
        if self.last_pad.is_some_and(|(p, at)| p == pad && now - at < REPEAT) {
            return;
        }
        self.last_pad = Some((pad, now));
        self.line(now, Change::Pad(pad));
    }

    /// The host keys held now; written when they changed.
    pub fn keys(&mut self, keys: &[u8]) {
        let now = Instant::now();
        if self.last_keys.as_deref() == Some(keys) {
            return;
        }
        self.last_keys = Some(keys.to_vec());
        self.line(now, Change::Keys(keys.to_vec()));
    }
}

fn write_lines(receive: Receiver<String>, mut out: impl Write) {
    let mut dirty = false;
    let mut flushed = Instant::now();
    loop {
        let wait = FLUSH.saturating_sub(flushed.elapsed());
        match receive.recv_timeout(wait) {
            Ok(line) => {
                dirty = true;
                if writeln!(out, "{line}").is_err() {
                    return;
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                let _ = out.flush();
                return;
            }
        }
        if flushed.elapsed() >= FLUSH {
            if dirty && out.flush().is_err() {
                return;
            }
            dirty = false;
            flushed = Instant::now();
        }
    }
}

/// Where a replay stands: by time or by frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum At {
    Ms(u64),
    Frame(u64),
}

/// A recording read back, its pad and key lines apart.
#[derive(Debug, Default)]
pub(super) struct Recording {
    pads: Vec<(u64, Option<u64>, Option<Raw>)>,
    keys: Vec<(u64, Option<u64>, Vec<u8>)>,
}

/// The entry in force at `at`: the last one at or before it.
fn last_at<T>(list: &[(u64, Option<u64>, T)], at: At) -> Option<&(u64, Option<u64>, T)> {
    let n = match at {
        At::Ms(ms) => list.partition_point(|e| e.0 <= ms),
        At::Frame(f) => list.partition_point(|e| e.1.is_some_and(|x| x <= f)),
    };
    n.checked_sub(1).map(|i| &list[i])
}

impl Recording {
    /// The lines of a recording, and how many lines were not understood.
    pub fn parse(text: &str) -> (Recording, usize) {
        let mut r = Recording::default();
        let mut bad = 0;
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            match Line::parse(line) {
                Some(Line { ms, frame, change: Change::Pad(p) }) => r.pads.push((ms, frame, p)),
                Some(Line { ms, frame, change: Change::Keys(k) }) => r.keys.push((ms, frame, k)),
                None => bad += 1,
            }
        }
        // Written in order; sort anyway, so a lookup's search holds.
        r.pads.sort_by_key(|e| e.0);
        r.keys.sort_by_key(|e| e.0);
        (r, bad)
    }

    pub fn lines(&self) -> usize {
        self.pads.len() + self.keys.len()
    }

    /// Every pad line has a frame, so the recording can be keyed by frame.
    pub fn has_frames(&self) -> bool {
        !self.pads.is_empty() && self.pads.iter().all(|e| e.1.is_some())
    }

    /// The last line's time and frame.
    pub fn end(&self) -> (u64, Option<u64>) {
        let pad = self.pads.last().map(|e| (e.0, e.1));
        let key = self.keys.last().map(|e| (e.0, e.1));
        pad.max(key).unwrap_or((0, None))
    }

    /// Past the last line.
    pub fn past_end(&self, at: At) -> bool {
        let (ms, frame) = self.end();
        match at {
            At::Ms(t) => t > ms,
            At::Frame(f) => frame.is_none_or(|e| f > e),
        }
    }

    /// Pad 0 at `at`: a neutral connected pad before the first line and past
    /// the last, `None` where the recording had no pad.
    pub fn pad_at(&self, at: At) -> Option<Raw> {
        if self.past_end(at) {
            return Some(Raw::default());
        }
        last_at(&self.pads, at).map_or(Some(Raw::default()), |e| e.2)
    }

    /// Keys held at `at`; none past the last line.
    pub fn keys_at(&self, at: At) -> &[u8] {
        if self.past_end(at) {
            return &[];
        }
        last_at(&self.keys, at).map_or(&[], |e| &e.2)
    }

    /// The recording's frame at a time, from the pad line in force then.
    pub fn frame_at(&self, ms: u64) -> Option<u64> {
        last_at(&self.pads, At::Ms(ms)).and_then(|e| e.1)
    }

    /// The recording's time at a frame, from the pad line in force then.
    pub fn ms_at(&self, frame: u64) -> Option<u64> {
        last_at(&self.pads, At::Frame(frame)).map(|e| e.0)
    }
}

/// How often a replay logs where it is.
pub(super) const REPLAY_REPORT: Duration = Duration::from_secs(10);

/// A recording played back as pad 0.
pub(super) struct Replay {
    recording: Recording,
    by_frame: bool,
    start: Instant,
    reported: Instant,
    finished: bool,
}

impl Replay {
    /// From `VICTORY_INPUT_REPLAY`; `None` when unset or unreadable.
    pub fn from_env() -> Option<Replay> {
        let path = std::env::var_os("VICTORY_INPUT_REPLAY")?;
        let path = std::path::Path::new(&path);
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("VICTORY_INPUT_REPLAY: cannot read {}: {e}; the host's pads are used", path.display());
                return None;
            }
        };
        let (recording, bad) = Recording::parse(&text);
        let mut by_frame = std::env::var("VICTORY_INPUT_REPLAY_BY").is_ok_and(|v| v == "frame");
        if by_frame && !recording.has_frames() {
            tracing::warn!("VICTORY_INPUT_REPLAY_BY=frame: the recording lacks frame numbers; replaying by time");
            by_frame = false;
        }
        let (end_ms, end_frame) = recording.end();
        tracing::info!(
            "replaying {} as pad 0 by {} (VICTORY_INPUT_REPLAY): {} lines ({bad} not understood), {:.1} s, {} frames; the host's pads are ignored",
            path.display(),
            if by_frame { "frame" } else { "time" },
            recording.lines(),
            end_ms as f64 / 1000.0,
            end_frame.map_or("?".to_owned(), |f| f.to_string()),
        );
        let now = Instant::now();
        Some(Replay { recording, by_frame, start: now, reported: now, finished: false })
    }

    fn at(&self, now: Instant, frame: Option<u64>) -> At {
        match (self.by_frame, frame) {
            (true, Some(f)) => At::Frame(f),
            _ => At::Ms(now.saturating_duration_since(self.start).as_millis() as u64),
        }
    }

    /// Pad 0 now, logging the replay's place every [`REPLAY_REPORT`] and
    /// once when it ends.
    pub fn pad(&mut self) -> Option<Raw> {
        let now = Instant::now();
        let frame = frame();
        let at = self.at(now, frame);
        let ms = now.saturating_duration_since(self.start).as_millis() as u64;
        if !self.finished && self.recording.past_end(at) {
            self.finished = true;
            tracing::info!("input replay finished: {}; pad 0 rests from now", self.place(ms, frame));
        } else if !self.finished && now - self.reported >= REPLAY_REPORT {
            self.reported = now;
            tracing::info!("input replay: {}", self.place(ms, frame));
        }
        self.recording.pad_at(at)
    }

    /// The keys held now.
    pub fn keys(&self) -> Vec<u8> {
        self.recording.keys_at(self.at(Instant::now(), frame())).to_vec()
    }

    /// Where the replay is, and how far time and frames have drifted apart:
    /// frames now less the recording's frames at this time, and by frame,
    /// time now less the recording's time at this frame.
    fn place(&self, ms: u64, frame: Option<u64>) -> String {
        let (end_ms, _) = self.recording.end();
        let frames = match (frame, self.recording.frame_at(ms)) {
            (Some(now), Some(then)) => format!("{:+} frames", now as i64 - then as i64),
            _ => "? frames".to_owned(),
        };
        let time = match frame.and_then(|f| self.recording.ms_at(f)) {
            Some(then) => format!("{:+} ms", ms as i64 - then as i64),
            None => "? ms".to_owned(),
        };
        format!(
            "{:.1} s of {:.1} s, frame {}, drift {frames} at this time, {time} at this frame",
            ms as f64 / 1000.0,
            end_ms as f64 / 1000.0,
            frame.map_or("?".to_owned(), |f| f.to_string()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(buttons: u16, lx: i16) -> Raw {
        Raw { buttons, lt: 255, rt: 3, lx, ly: -32768, rx: 32767, ry: 0 }
    }

    #[test]
    fn lines_round_trip() {
        let lines = [
            Line { ms: 0, frame: Some(0), change: Change::Pad(Some(Raw::default())) },
            Line { ms: 1234, frame: Some(77), change: Change::Pad(Some(raw(0xf0ff, -12))) },
            Line { ms: 99, frame: None, change: Change::Pad(None) },
            Line { ms: 5, frame: Some(1), change: Change::Keys(vec![0x57, 0x0d, 0xa0]) },
            Line { ms: 6, frame: Some(1), change: Change::Keys(Vec::new()) },
        ];
        for line in lines {
            let text = line.format();
            assert_eq!(Line::parse(&text), Some(line), "{text}");
        }
        assert_eq!(
            Line { ms: 1234, frame: Some(77), change: Change::Pad(Some(raw(0x1003, 5))) }.format(),
            "1234 77 1003 255 3 5 -32768 32767 0"
        );
        assert_eq!(Line::parse("1 2 0000 0 0 0 0 0"), None, "an axis short");
        assert_eq!(Line::parse("1 2 0000 0 0 0 0 0 0 9"), None, "one too many");
        assert_eq!(Line::parse("1 2 0000 256 0 0 0 0 0"), None, "trigger past 255");
        assert_eq!(Line::parse("1 2 keys 5g"), None);
        assert_eq!(Line::parse("x 2 off"), None);
    }

    #[test]
    fn replay_looks_up_the_state_in_force() {
        let text = "# header\n\
                    100 6 0004 0 0 0 0 0 0\n\
                    150 9 keys 57\n\
                    200 12 0000 0 0 1000 0 0 0\n\
                    garbage\n\
                    300 18 off\n\
                    400 24 keys -\n\
                    500 30 0000 0 0 0 0 0 0\n";
        let (r, bad) = Recording::parse(text);
        assert_eq!(bad, 1);
        assert_eq!(r.lines(), 6);
        assert!(r.has_frames());
        assert_eq!(r.end(), (500, Some(30)));
        let a = Some(Raw { buttons: 4, ..Default::default() });
        let stick = Some(Raw { lx: 1000, ..Default::default() });
        assert_eq!(r.pad_at(At::Ms(0)), Some(Raw::default()), "before the first line: at rest");
        assert_eq!(r.pad_at(At::Ms(100)), a);
        assert_eq!(r.pad_at(At::Ms(199)), a);
        assert_eq!(r.pad_at(At::Ms(200)), stick);
        assert_eq!(r.pad_at(At::Ms(350)), None, "pad gone");
        assert_eq!(r.pad_at(At::Ms(501)), Some(Raw::default()), "past the end: at rest");
        assert_eq!(r.pad_at(At::Frame(11)), a);
        assert_eq!(r.pad_at(At::Frame(12)), stick);
        assert_eq!(r.pad_at(At::Frame(31)), Some(Raw::default()));
        assert_eq!(r.keys_at(At::Ms(149)), &[] as &[u8]);
        assert_eq!(r.keys_at(At::Ms(160)), &[0x57]);
        assert_eq!(r.keys_at(At::Ms(450)), &[] as &[u8]);
        assert_eq!(r.frame_at(250), Some(12));
        assert_eq!(r.ms_at(20), Some(300));
        assert_eq!(r.frame_at(50), None);
        let (timeless, _) = Recording::parse("10 - 0000 0 0 0 0 0 0\n");
        assert!(!timeless.has_frames());
    }

    #[test]
    fn the_writer_writes_every_line() {
        let (send, receive) = std::sync::mpsc::channel();
        send.send("a".to_owned()).unwrap();
        send.send("b".to_owned()).unwrap();
        drop(send);
        let mut out = Vec::new();
        write_lines(receive, &mut out);
        assert_eq!(out, b"a\nb\n");
    }
}
