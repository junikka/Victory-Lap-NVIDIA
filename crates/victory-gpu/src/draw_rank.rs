//! `VICTORY_DRAW_RANK=<file>` (diagnostics): draws per host pipeline. Every
//! [`EVERY`] frames, the pipelines with the most draws since the last
//! report are appended to the file, as `f<frame> <draws per frame> <name>`,
//! to find the few that dominate a world frame.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

const EVERY: u64 = 300;
const TOP: usize = 60;

fn file() -> Option<&'static PathBuf> {
    static F: OnceLock<Option<PathBuf>> = OnceLock::new();
    F.get_or_init(|| std::env::var_os("VICTORY_DRAW_RANK").map(PathBuf::from)).as_ref()
}

fn counts() -> &'static Mutex<HashMap<String, u64>> {
    static C: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A draw or mesh dispatch with host pipeline `name`.
pub(crate) fn note(name: &str) {
    if file().is_none() {
        return;
    }
    let mut c = counts().lock().unwrap_or_else(|e| e.into_inner());
    match c.get_mut(name) {
        Some(n) => *n += 1,
        None => {
            c.insert(name.to_owned(), 1);
        }
    }
}

/// The report lines for `counts` over `frames` frames, most draws first.
fn lines(frame: u64, frames: u64, counts: HashMap<String, u64>) -> Vec<String> {
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v.into_iter()
        .take(TOP)
        .map(|(name, n)| format!("f{frame} {:.1} {name}", n as f64 / frames as f64))
        .collect()
}

/// At present number `frame`.
pub(crate) fn at_present(frame: u64) {
    let Some(path) = file() else { return };
    if frame % EVERY != 0 {
        return;
    }
    let taken = std::mem::take(&mut *counts().lock().unwrap_or_else(|e| e.into_inner()));
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        for l in lines(frame, EVERY, taken) {
            let _ = writeln!(f, "{l}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_rank_by_draws_per_frame() {
        let c = HashMap::from([("a".to_owned(), 600), ("b".to_owned(), 3000), ("c".to_owned(), 600)]);
        assert_eq!(lines(900, 300, c), ["f900 10.0 b", "f900 2.0 a", "f900 2.0 c"]);
    }
}
