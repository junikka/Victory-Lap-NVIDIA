//! Switches that can be set to `alt`: twenty seconds off, twenty on, in turn,
//! so that one run measures both sides in the same scene. (`VICTORY_AB`,
//! `crate::ab`, does the same for several switches at once, with the frame
//! times of each phase.)

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering::Relaxed};

/// What such a switch is set to: 0 off, 1 on, 2 in turn. `default`: what it
/// is unset, or set to anything else.
pub(crate) fn mode(setting: Option<&str>, default: u8) -> u8 {
    match setting.map(str::trim) {
        Some("0") => 0,
        Some("1") => 1,
        Some("alt") => 2,
        _ => default,
    }
}

/// For a switch set to `alt`: whether it is on now, each change said in the
/// log (`last`: the switch's own note of what was said, 2 at first).
pub(crate) fn in_turn(last: &AtomicU8, switch: &str) -> bool {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
    if last.swap(u8::from(on), Relaxed) != u8::from(on) {
        tracing::info!("{switch}=alt: {} from now", if on { "ON" } else { "off" });
    }
    on
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_switch_that_can_alternate_is_read() {
        assert_eq!(mode(None, 0), 0);
        assert_eq!(mode(None, 1), 1);
        assert_eq!(mode(Some("0"), 1), 0);
        assert_eq!(mode(Some("1"), 0), 1);
        assert_eq!(mode(Some(" alt "), 0), 2);
        assert_eq!(mode(Some("yes"), 0), 0);
        assert_eq!(mode(Some("yes"), 1), 1);
        // In turn: off at first, and said once.
        let last = AtomicU8::new(2);
        assert!(!in_turn(&last, "VICTORY_A_TEST"));
        assert_eq!(last.load(Relaxed), 0);
    }
}
