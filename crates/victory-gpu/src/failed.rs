//! The run's failed list: `victory-<run>-failed.csv`, one row per distinct
//! item that did not work, with how often it cost the title something.
//!
//! - `shader`: a shader that does not translate (once per container).
//! - `pipeline`: a pipeline the title made through a device method we lack
//!   (slots 0x1c8 and 0x218, [`crate::device`]), counted per call.
//! - `dropped`: draws and dispatches dropped, by the reason the frame-rate
//!   lines give (`dropped N/frame without a host pipeline: <why>`), counted
//!   per draw. With `VICTORY_DROP_REASONS=1`, a pipeline that has no host
//!   pipeline yet is `queued` (in the compile queue), `never started` (no
//!   request made) or `left out`: the row's id is the pipeline hash and its
//!   stage a shader name. They are taken from [`crate::pipeline::take_drops`]
//!   once a second, so nothing is added on the draw path.
//!
//! Columns: `kind,id,stage,reason,count,first_seen_s` (seconds since the
//! log started), most counted first. The file is written whole every
//! [`EVERY`] and at exit (the title's own end and the CRT's `exit`; a run
//! killed from outside keeps the last write).
//!
//! Where: `VICTORY_FAILED_LOG=<file>`; else beside the harness's draw rank
//! (`VICTORY_DRAW_RANK=<dir>\victory-<run>-rank.txt` gives
//! `<dir>\victory-<run>-failed.csv`); else the data folder's `runs`
//! (`victory-<unix seconds>-<pid>-failed.csv`). `VICTORY_FAILED_LOG=0`:
//! off.

use std::collections::HashMap;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How often the file is written.
const EVERY: Duration = Duration::from_secs(30);

pub(crate) const HEADER: &str = "kind,id,stage,reason,count,first_seen_s";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Kind {
    Shader,
    Pipeline,
    Dropped,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Shader => "shader",
            Kind::Pipeline => "pipeline",
            Kind::Dropped => "dropped",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Row {
    id: String,
    stage: String,
    reason: String,
    count: u64,
    first_seen_s: f64,
}

/// The distinct items, by kind and key (a shader's hash, a method's slot,
/// a drop's reason as logged).
#[derive(Default)]
pub(crate) struct Table {
    rows: [HashMap<String, Row>; 3],
    /// Something changed since the last [`Table::take_changed`].
    changed: bool,
}

impl Table {
    /// Counts `n` more of the item `key`; `describe` gives its id, stage and
    /// reason the first time only.
    pub(crate) fn add(
        &mut self,
        kind: Kind,
        key: &str,
        n: u64,
        at_s: f64,
        describe: impl FnOnce() -> (String, String, String),
    ) {
        if n == 0 {
            return;
        }
        self.changed = true;
        let rows = &mut self.rows[kind as usize];
        // A known item allocates nothing.
        if let Some(row) = rows.get_mut(key) {
            row.count += n;
            return;
        }
        let (id, stage, reason) = describe();
        rows.insert(
            key.to_owned(),
            Row { id, stage, reason, count: n, first_seen_s: at_s },
        );
    }

    fn take_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }

    /// The file's text: the header, then the rows, most counted first.
    pub(crate) fn csv(&self) -> String {
        let kinds = [Kind::Shader, Kind::Pipeline, Kind::Dropped];
        let mut rows: Vec<(Kind, &String, &Row)> = kinds
            .into_iter()
            .flat_map(|kind| self.rows[kind as usize].iter().map(move |(key, row)| (kind, key, row)))
            .collect();
        rows.sort_by(|a, b| b.2.count.cmp(&a.2.count).then_with(|| (a.0, a.1).cmp(&(b.0, b.1))));
        let mut out = String::with_capacity(64 * (rows.len() + 1));
        out.push_str(HEADER);
        out.push('\n');
        for (kind, _, row) in rows {
            let _ = writeln!(
                out,
                "{},{},{},{},{},{:.1}",
                kind.name(),
                field(&row.id),
                field(&row.stage),
                field(&row.reason),
                row.count,
                row.first_seen_s
            );
        }
        out
    }
}

/// A CSV field: quoted when it holds a comma, a quote or a line break.
fn field(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\"").replace(['\n', '\r'], " "))
    } else {
        text.to_owned()
    }
}

/// A drop of a pipeline left out on purpose (`pipeline::make`'s
/// `left_out`: ray-tracing shaders without the BVH builder) is this, then
/// the name of the shader it is left out for, when `VICTORY_DROP_REASONS`
/// is unset. Such a pipeline is never made; it is not "being made". With
/// the switch on, the row is `left out` plus the pipeline hash and that name
/// ([`making_drop`]).
pub(crate) const LEFT_OUT: &str = "left out on purpose (ray tracing): ";

/// Whether `text` is a pipeline hash ([`crate::pipeline::make`]'s key): 32 hex digits.
#[cfg(any(test, feature = "diagnostics"))]
fn is_pipeline(text: &str) -> bool {
    text.len() == 32 && text.bytes().all(|b| b.is_ascii_hexdigit())
}

/// `queued` / `never started` / `left out`, then the pipeline hash and a
/// shader name. `None` when `why` is not that line (the older left-out line
/// has no hash, so it is not this).
#[cfg(any(test, feature = "diagnostics"))]
fn split_making(why: &str) -> Option<(String, String, String)> {
    let (reason, rest) = if let Some(rest) = why.strip_prefix("never started ") {
        ("never started", rest)
    } else if let Some(rest) = why.strip_prefix("queued ") {
        ("queued", rest)
    } else if let Some(rest) = why.strip_prefix("left out ") {
        ("left out", rest)
    } else {
        return None;
    };
    let (pipeline, shader) = rest.split_once(' ')?;
    if !is_pipeline(pipeline) || shader.is_empty() {
        return None;
    }
    Some((pipeline.into(), shader.into(), reason.into()))
}

/// The drop line for a pipeline with no host pipeline yet. `left_out` wins:
/// a pipeline left out on purpose was never asked for, and is not queued.
#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn making_drop(queued: bool, left_out: bool, pipeline: &str, shader: &str) -> String {
    let reason = if left_out {
        "left out"
    } else if queued {
        "queued"
    } else {
        "never started"
    };
    format!("{reason} {pipeline} {shader}")
}

/// How many of `drops` are queued, never started, and left out.
#[cfg(any(test, feature = "diagnostics"))]
fn making_counts(drops: &[(String, u64)]) -> [u64; 3] {
    let mut n = [0; 3];
    for (why, count) in drops {
        match parse_drop(why).2.as_str() {
            "queued" => n[0] += *count,
            "never started" => n[1] += *count,
            "left out" | "left out on purpose (ray tracing)" => n[2] += *count,
            _ => {}
        }
    }
    n
}

/// The per-frame count line, or nothing when none of the three reasons occurred.
#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn making_line(drops: &[(String, u64)], frames: f64) -> Option<String> {
    if frames <= 0.0 {
        return None;
    }
    let [queued, never, left] = making_counts(drops);
    if queued + never + left == 0 {
        return None;
    }
    Some(format!(
        "dropped per frame without a host pipeline: {:.1} queued, {:.1} never started, {:.1} left out (VICTORY_DROP_REASONS=1)",
        queued as f64 / frames,
        never as f64 / frames,
        left as f64 / frames
    ))
}

fn is_hash(text: &str) -> bool {
    text.len() == 16 && text.bytes().all(|b| b.is_ascii_hexdigit())
}

/// A drop's reason as [`crate::pipeline::note_why`] has it, as id, stage
/// and reason.
pub(crate) fn parse_drop(why: &str) -> (String, String, String) {
    #[cfg(any(test, feature = "diagnostics"))]
    if let Some(row) = split_making(why) {
        return row;
    }
    let shaders = |list: &str| -> Option<(String, String)> {
        let (mut stages, mut hashes) = (Vec::new(), Vec::new());
        for part in list.split([',', '+']).map(str::trim).filter(|p| !p.is_empty()) {
            let (stage, hash) = part.split_once(' ')?;
            if !is_hash(hash) {
                return None;
            }
            stages.push(stage);
            hashes.push(hash);
        }
        (!hashes.is_empty()).then(|| (hashes.join("+"), stages.join("+")))
    };
    if let Some(list) = why.strip_prefix("refused [").and_then(|r| r.strip_suffix(']'))
        && let Some((id, stage)) = shaders(list)
    {
        return (id, stage, "the host refused the pipeline".into());
    }
    if let Some(hash) = why.strip_prefix("skipped shader ")
        && is_hash(hash)
    {
        return (hash.into(), String::new(), "skipped (VICTORY_SKIP_PIPELINES)".into());
    }
    if let Some(name) = why.strip_prefix(LEFT_OUT)
        && !name.is_empty()
    {
        return (name.into(), "compute".into(), LEFT_OUT.trim_end_matches([':', ' ']).into());
    }
    if let Some((id, stage)) = shaders(why) {
        return (id, stage, "a shader does not translate".into());
    }
    (String::new(), String::new(), why.into())
}

fn cell() -> &'static Mutex<Table> {
    static TABLE: OnceLock<Mutex<Table>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(Table::default()))
}

fn table() -> std::sync::MutexGuard<'static, Table> {
    cell().lock().unwrap_or_else(|e| e.into_inner())
}

/// Seconds since the log started (the first use, if earlier).
fn now_s() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// `VICTORY_FAILED_LOG=0` turns the list off.
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FAILED_LOG").is_none_or(|v| v != "0"))
}

/// A shader that does not translate.
pub(crate) fn shader(hash: &str, stage: &str, error: &str) {
    if !on() {
        return;
    }
    let at = now_s();
    table().add(Kind::Shader, hash, 1, at, || (hash.into(), stage.into(), error.into()));
}

/// A pipeline the title made through device method `slot`, which we lack.
pub(crate) fn pipeline_method(slot: usize) {
    if !on() {
        return;
    }
    let at = now_s();
    let key = if slot == 0x218 { "0x218" } else { "0x1c8" };
    table().add(Kind::Pipeline, key, 1, at, || {
        (format!("device+{slot:#x}"), String::new(), "a pipeline creation method that is not implemented".into())
    });
}

/// The drops of the last second, by reason.
pub(crate) fn dropped(drops: &[(String, u64)]) {
    if !on() || drops.is_empty() {
        return;
    }
    let at = now_s();
    let mut t = table();
    for (why, n) in drops {
        t.add(Kind::Dropped, why, *n, at, || parse_drop(why));
    }
}

/// Where the list goes: `failed` is `VICTORY_FAILED_LOG`, `rank`
/// `VICTORY_DRAW_RANK`, `runs` the data folder's runs folder.
fn resolve(failed: Option<OsString>, rank: Option<OsString>, runs: Option<&Path>, stamp: &str) -> Option<PathBuf> {
    match failed {
        Some(v) if v == "0" => return None,
        Some(v) if !v.is_empty() && v != "1" => return Some(PathBuf::from(v)),
        _ => {}
    }
    if let Some(rank) = rank.map(PathBuf::from)
        && let Some(name) = rank.file_name().and_then(|n| n.to_str())
        && let Some(run) = name.strip_suffix("-rank.txt")
    {
        return Some(rank.with_file_name(format!("{run}-failed.csv")));
    }
    runs.map(|dir| dir.join(format!("victory-{stamp}-failed.csv")))
}

fn path() -> &'static OnceLock<Option<PathBuf>> {
    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    &PATH
}

/// Writes `text` to `to` through a file beside it, so a reader never sees
/// half a list.
fn write_file(to: &Path, text: &str) -> std::io::Result<()> {
    let mut part = to.as_os_str().to_owned();
    part.push(".part");
    std::fs::write(&part, text)?;
    std::fs::rename(&part, to)
}

/// Writes the list if it changed (always with `force`).
fn write(force: bool) {
    let Some(Some(to)) = path().get() else { return };
    let text = {
        let Ok(mut t) = table_try() else { return };
        if !t.take_changed() && !force {
            return;
        }
        t.csv()
    };
    if let Err(e) = write_file(to, &text) {
        tracing::warn!(path = %to.display(), %e, "the failed list could not be written");
    }
}

/// The table without waiting: at exit a thread that ended can hold it.
fn table_try() -> Result<std::sync::MutexGuard<'static, Table>, ()> {
    cell().try_lock().or_else(|e| match e {
        std::sync::TryLockError::Poisoned(p) => Ok(p.into_inner()),
        std::sync::TryLockError::WouldBlock => Err(()),
    })
}

unsafe extern "C" {
    fn atexit(f: extern "C" fn()) -> i32;
}

extern "C" fn at_exit() {
    write(false);
}

/// Starts the list for a run, with `runs` the data folder's runs folder:
/// the file (header only) at once, then every [`EVERY`] and at exit.
pub fn start(runs: Option<PathBuf>) {
    if !on() {
        return;
    }
    let _ = now_s();
    let stamp = format!(
        "{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        std::process::id()
    );
    let to = resolve(
        std::env::var_os("VICTORY_FAILED_LOG"),
        std::env::var_os("VICTORY_DRAW_RANK"),
        runs.as_deref(),
        &stamp,
    );
    if path().set(to.clone()).is_err() {
        return;
    }
    let Some(to) = to else { return };
    if let Some(dir) = to.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    tracing::info!(path = %to.display(), "failed list (VICTORY_FAILED_LOG=0: off)");
    write(true);
    // SAFETY: registers a function with the host CRT's exit table.
    unsafe { atexit(at_exit) };
    let spawned = std::thread::Builder::new().name("victory-failed".into()).spawn(|| {
        loop {
            std::thread::sleep(EVERY);
            write(false);
        }
    });
    if let Err(e) = spawned {
        tracing::warn!(%e, "the failed list has no thread: it is written at exit only");
    }
}

/// Writes the list now (the title's own end).
pub fn flush() {
    write(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_each_item_once_and_describes_it_once() {
        let mut t = Table::default();
        let mut described = 0;
        for at in [1.0, 2.0, 3.0] {
            t.add(Kind::Dropped, "being made", 2, at, || {
                described += 1;
                parse_drop("being made")
            });
        }
        t.add(Kind::Shader, "00112233aabbccdd", 1, 5.0, || ("00112233aabbccdd".into(), "pixel".into(), "bad".into()));
        t.add(Kind::Dropped, "nothing", 0, 6.0, || unreachable!());
        assert_eq!(described, 1);
        assert_eq!(t.rows.iter().map(HashMap::len).sum::<usize>(), 2);
        let row = &t.rows[Kind::Dropped as usize]["being made"];
        assert_eq!((row.count, row.first_seen_s), (6, 1.0));
        assert!(t.take_changed());
        assert!(!t.take_changed());
    }

    #[test]
    fn csv_lists_most_counted_first_and_quotes_fields() {
        let mut t = Table::default();
        t.add(Kind::Pipeline, "0x218", 3, 0.5, || ("device+0x218".into(), String::new(), "not implemented".into()));
        t.add(Kind::Shader, "aa", 1, 12.34, || ("aa".into(), "compute".into(), "bad op, \"x\"\nline".into()));
        t.add(Kind::Dropped, "why", 40, 2.0, || parse_drop("vertex 0011223344556677 + pixel 8899aabbccddeeff"));
        assert_eq!(
            t.csv(),
            "kind,id,stage,reason,count,first_seen_s\n\
             dropped,0011223344556677+8899aabbccddeeff,vertex+pixel,a shader does not translate,40,2.0\n\
             pipeline,device+0x218,,not implemented,3,0.5\n\
             shader,aa,compute,\"bad op, \"\"x\"\" line\",1,12.3\n"
        );
    }

    #[test]
    fn drop_reasons_split_into_id_stage_and_reason() {
        assert_eq!(
            parse_drop("refused [compute 0123456789abcdef]"),
            ("0123456789abcdef".into(), "compute".into(), "the host refused the pipeline".into())
        );
        assert_eq!(
            parse_drop("refused [vertex 0123456789abcdef, pixel fedcba9876543210]"),
            ("0123456789abcdef+fedcba9876543210".into(), "vertex+pixel".into(), "the host refused the pipeline".into())
        );
        assert_eq!(parse_drop("pixel fedcba9876543210").2, "a shader does not translate");
        assert_eq!(parse_drop("skipped shader fedcba9876543210").0, "fedcba9876543210");
        assert_eq!(parse_drop("being made"), (String::new(), String::new(), "being made".into()));
        assert_eq!(
            parse_drop("left out on purpose (ray tracing): CS_RaytracingResolve"),
            ("CS_RaytracingResolve".into(), "compute".into(), "left out on purpose (ray tracing)".into())
        );
        assert_eq!(parse_drop(&format!("{LEFT_OUT}CS_ScreenProbes_ActivateRays")).0, "CS_ScreenProbes_ActivateRays");
        assert_eq!(parse_drop("indirect kind 2: no arguments").2, "indirect kind 2: no arguments");
    }

    #[test]
    fn the_three_drop_reasons_carry_the_pipeline_hash_and_a_shader_name() {
        let hash = "0123456789abcdef0123456789abcdef";
        let name = "mod:CS_KeepThisName";
        assert_eq!(
            making_drop(true, false, hash, name),
            "queued 0123456789abcdef0123456789abcdef mod:CS_KeepThisName"
        );
        assert_eq!(
            making_drop(false, false, hash, name),
            "never started 0123456789abcdef0123456789abcdef mod:CS_KeepThisName"
        );
        // Left out wins over queued: it was never put in the compile queue.
        assert_eq!(
            making_drop(true, true, hash, "CS_RaytracingResolve"),
            "left out 0123456789abcdef0123456789abcdef CS_RaytracingResolve"
        );
        assert_eq!(
            parse_drop("queued 0123456789abcdef0123456789abcdef mod:CS_KeepThisName"),
            (hash.into(), name.into(), "queued".into())
        );
        assert_eq!(
            parse_drop("never started 0123456789abcdef0123456789abcdef mod:CS_KeepThisName"),
            (hash.into(), name.into(), "never started".into())
        );
        assert_eq!(
            parse_drop("left out 0123456789abcdef0123456789abcdef CS_RaytracingResolve"),
            (hash.into(), "CS_RaytracingResolve".into(), "left out".into())
        );
        // A shader hash is 16 digits: that line is not a pipeline row.
        assert_eq!(
            parse_drop("queued 0123456789abcdef mod:CS_KeepThisName").2,
            "queued 0123456789abcdef mod:CS_KeepThisName"
        );
    }

    #[test]
    fn each_reason_is_its_own_row() {
        let hash = "0123456789abcdef0123456789abcdef";
        let other = "fedcba9876543210fedcba9876543210";
        let mut t = Table::default();
        for (why, n) in [
            (making_drop(true, false, hash, "CS_QueuedOne"), 10u64),
            (making_drop(true, false, other, "CS_QueuedOne"), 4),
            (making_drop(false, false, hash, "mod:CS_Never"), 7),
            (making_drop(false, true, hash, "CS_RaytracingResolve"), 3),
        ] {
            t.add(Kind::Dropped, &why, n, 4.0, || parse_drop(&why));
        }
        let again = making_drop(true, false, hash, "CS_QueuedOne");
        t.add(Kind::Dropped, &again, 2, 9.0, || parse_drop(&again));
        assert_eq!(
            t.csv(),
            "kind,id,stage,reason,count,first_seen_s\n\
             dropped,0123456789abcdef0123456789abcdef,CS_QueuedOne,queued,12,4.0\n\
             dropped,0123456789abcdef0123456789abcdef,mod:CS_Never,never started,7,4.0\n\
             dropped,fedcba9876543210fedcba9876543210,CS_QueuedOne,queued,4,4.0\n\
             dropped,0123456789abcdef0123456789abcdef,CS_RaytracingResolve,left out,3,4.0\n"
        );
    }

    #[test]
    fn the_frame_line_counts_the_three_reasons() {
        let hash = "0123456789abcdef0123456789abcdef";
        let drops = vec![
            (making_drop(true, false, hash, "CS_QueuedOne"), 8u64),
            (making_drop(false, false, hash, "mod:CS_Never"), 2),
            ("left out on purpose (ray tracing): CS_RaytracingResolve".into(), 4),
            ("being made".into(), 100),
        ];
        assert_eq!(
            making_line(&drops, 4.0).as_deref(),
            Some(
                "dropped per frame without a host pipeline: 2.0 queued, 0.5 never started, 1.0 left out (VICTORY_DROP_REASONS=1)"
            )
        );
        assert!(making_line(&[("being made".into(), 9)], 1.0).is_none());
        assert!(making_line(&drops, 0.0).is_none());
    }

    #[test]
    fn the_file_goes_where_the_switches_say() {
        let runs = Path::new("C:/data/runs");
        assert_eq!(resolve(Some("0".into()), Some("x-rank.txt".into()), Some(runs), "s"), None);
        assert_eq!(
            resolve(Some("C:/a/f.csv".into()), None, Some(runs), "s"),
            Some(PathBuf::from("C:/a/f.csv"))
        );
        assert_eq!(
            resolve(Some("1".into()), Some("C:/r/victory-route7-rank.txt".into()), Some(runs), "s"),
            Some(PathBuf::from("C:/r/victory-route7-failed.csv"))
        );
        assert_eq!(
            resolve(None, Some("C:/r/other.txt".into()), Some(runs), "12-34"),
            Some(runs.join("victory-12-34-failed.csv"))
        );
        assert_eq!(resolve(None, None, None, "s"), None);
    }

    #[test]
    fn written_through_a_part_file() {
        let dir = std::env::temp_dir().join(format!("victory-failed-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let to = dir.join("victory-t-failed.csv");
        write_file(&to, "a\n").unwrap();
        write_file(&to, HEADER).unwrap();
        assert_eq!(std::fs::read_to_string(&to).unwrap(), HEADER);
        assert!(!dir.join("victory-t-failed.csv.part").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
