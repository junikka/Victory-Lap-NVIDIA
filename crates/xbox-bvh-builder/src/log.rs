//! Append-only log. No network APIs.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

#[link(name = "kernel32")]
extern "system" {
    fn GetSystemTime(time: *mut SystemTime);
}

#[repr(C)]
struct SystemTime {
    year: u16,
    month: u16,
    day_of_week: u16,
    day: u16,
    hour: u16,
    minute: u16,
    second: u16,
    milliseconds: u16,
}

fn log_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn log_path() -> Option<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")?;
    Some(
        PathBuf::from(base)
            .join("victory-lap")
            .join("xbox-bvh-builder.log"),
    )
}

fn timestamp() -> String {
    let mut time = SystemTime {
        year: 0,
        month: 0,
        day_of_week: 0,
        day: 0,
        hour: 0,
        minute: 0,
        second: 0,
        milliseconds: 0,
    };
    unsafe { GetSystemTime(&mut time) }
    let year = time.year;
    let month = time.month;
    let day = time.day;
    let hour = time.hour;
    let minute = time.minute;
    let second = time.second;
    let milliseconds = time.milliseconds;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{milliseconds:03}Z")
}

fn append(line: &str) {
    let Some(path) = log_path() else {
        return;
    };
    let lock = log_lock();
    let _guard = lock.lock().unwrap_or_else(|err| err.into_inner());
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut file = match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(file) => file,
        Err(_) => return,
    };
    let _ = file.write_all(line.as_bytes());
    let _ = file.flush();
}

/// `VICTORY_BVH_LOG_ONCE=1` (ticket 186, off unless 1): `flag` as the call's
/// `once`, so a call logged on every use logs only its first. Read once.
pub(crate) fn once_if_switched(flag: &'static AtomicBool) -> Option<&'static AtomicBool> {
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| log_once_set(std::env::var("VICTORY_BVH_LOG_ONCE").ok().as_deref()));
    if on { Some(flag) } else { None }
}

pub(crate) fn log_once_set(v: Option<&str>) -> bool {
    v.map(|v| v.trim() == "1").unwrap_or(false)
}

/// Log one call. `once` skips every call after the first.
pub(crate) fn log_event(name: &str, once: Option<&'static AtomicBool>, detail: &str) {
    if let Some(flag) = once {
        if flag.swap(true, Ordering::AcqRel) {
            return;
        }
    }
    let ra = crate::caller::caller_return_address();
    let ts = timestamp();
    let line = format!("{ts} {name} ra={ra:#x} {detail}\r\n");
    append(&line);
}

#[cfg(test)]
mod log_once_tests {
    #[test]
    fn the_switch_is_on_only_with_1() {
        assert!(super::log_once_set(Some("1")));
        assert!(super::log_once_set(Some(" 1 ")));
        assert!(!super::log_once_set(None));
        assert!(!super::log_once_set(Some("0")));
        assert!(!super::log_once_set(Some("on")));
    }
}
