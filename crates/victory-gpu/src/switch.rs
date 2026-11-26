//! The `VICTORY_*` on/off switches (`VICTORY_D3D12_DEBUG=1` and its siblings).

use std::ffi::OsStr;

/// Whether a switch's value turns it on: it is set, and is not `0`.
///
/// A switch used to be on whenever its variable existed, so `VICTORY_D3D12_WARP=0` (typed to switch it off, or left
/// over from comparing two runs) still selected the software adapter. Any other value keeps the old meaning.
fn value_is_on(value: Option<&OsStr>) -> bool {
    value.is_some_and(|v| v != "0")
}

/// Whether the environment variable `name` switches something on.
pub(crate) fn on(name: &str) -> bool {
    value_is_on(std::env::var_os(name).as_deref())
}

/// A switch that is on unless its variable is `0`, read once; with
/// `VICTORY_AB` naming it (diagnostics builds, `crate::ab`), on and off while
/// the title runs.
#[cfg(windows)]
#[macro_export]
#[doc(hidden)]
macro_rules! toggled {
    ($name:literal) => {{
        #[cfg(feature = "diagnostics")]
        {
            static BIT: std::sync::OnceLock<Result<bool, u32>> = std::sync::OnceLock::new();
            match BIT.get_or_init(|| match $crate::ab::bit($name) {
                Some(bit) => Err(bit),
                None => Ok(std::env::var_os($name).is_none_or(|v| v != "0")),
            }) {
                Ok(on) => *on,
                Err(bit) => $crate::ab::on(*bit),
            }
        }
        #[cfg(not(feature = "diagnostics"))]
        {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var_os($name).is_none_or(|v| v != "0"))
        }
    }};
}

/// What a switch that gives a number of microseconds says.
#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Micros {
    /// That many, all the time (0: the switch is off).
    Fixed(u64),
    /// That many and none in turn ([`alt_on`]), so that one run measures both.
    Alt(u64),
}

/// What `value`, such a switch's variable, says: unset, `0` or something not
/// understood, off; `1`, `usual` microseconds; a larger number, that many;
/// `alt`, `usual` in turn with none.
#[cfg(windows)]
pub(crate) fn micros(value: Option<&str>, usual: u64) -> Micros {
    match value.map(str::trim) {
        Some("alt") => Micros::Alt(usual),
        Some("1") => Micros::Fixed(usual),
        Some(v) => Micros::Fixed(v.parse().unwrap_or(0)),
        None => Micros::Fixed(0),
    }
}

/// A switch of microseconds ([`micros`]), read once; with `VICTORY_AB` naming
/// it (diagnostics builds, `crate::ab`), its usual number and none while the
/// title runs.
#[cfg(windows)]
pub(crate) struct MicrosSwitch {
    name: &'static str,
    usual: u64,
    asked: std::sync::OnceLock<Result<Micros, u32>>,
}

#[cfg(windows)]
impl MicrosSwitch {
    pub(crate) const fn new(name: &'static str, usual: u64) -> MicrosSwitch {
        MicrosSwitch { name, usual, asked: std::sync::OnceLock::new() }
    }

    /// The microseconds it gives now.
    pub(crate) fn now(&self) -> u64 {
        let asked = self.asked.get_or_init(|| {
            #[cfg(feature = "diagnostics")]
            if let Some(bit) = crate::ab::bit(self.name) {
                return Err(bit);
            }
            Ok(micros(std::env::var(self.name).ok().as_deref(), self.usual))
        });
        match *asked {
            Ok(Micros::Fixed(us)) => us,
            Ok(Micros::Alt(us)) => {
                if alt_on() {
                    us
                } else {
                    0
                }
            }
            #[cfg(feature = "diagnostics")]
            Err(bit) => {
                if crate::ab::on(bit) {
                    self.usual
                } else {
                    0
                }
            }
            #[cfg(not(feature = "diagnostics"))]
            Err(_) => 0,
        }
    }
}

/// How long each turn of a switch set to `alt` lasts.
#[cfg(windows)]
const ALT_SECONDS: u64 = 20;

/// Whether the switches set to `alt` are on now: twenty seconds off, twenty
/// on, in turn, counted from the first that asks (all of them change
/// together; `VICTORY_AB` takes them apart).
#[cfg(windows)]
pub(crate) fn alt_on() -> bool {
    static BEGAN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    alt_turn(BEGAN.get_or_init(std::time::Instant::now).elapsed())
}

/// [`alt_on`], that long after the first asking.
#[cfg(windows)]
fn alt_turn(since: std::time::Duration) -> bool {
    since.as_secs() / ALT_SECONDS % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::value_is_on;
    use std::ffi::OsStr;

    #[test]
    fn unset_is_off() {
        assert!(!value_is_on(None));
    }

    #[test]
    fn zero_is_off() {
        assert!(!value_is_on(Some(OsStr::new("0"))));
    }

    #[test]
    fn one_is_on() {
        assert!(value_is_on(Some(OsStr::new("1"))));
    }

    #[test]
    fn other_values_keep_their_old_meaning() {
        // Documented as `=1`; anything else that was on before stays on: only `0` is new.
        for value in ["", "true", "yes", "2", "00", " 0", "0 "] {
            assert!(value_is_on(Some(OsStr::new(value))), "{value:?}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn microseconds_are_read_with_one_as_the_usual_and_alt_in_turn() {
        use super::{Micros, alt_turn, micros};
        assert_eq!(micros(None, 200), Micros::Fixed(0));
        assert_eq!(micros(Some("0"), 200), Micros::Fixed(0));
        assert_eq!(micros(Some("1"), 200), Micros::Fixed(200));
        assert_eq!(micros(Some(" 350 "), 200), Micros::Fixed(350));
        assert_eq!(micros(Some("alt"), 200), Micros::Alt(200));
        // Not understood: off, as unset.
        assert_eq!(micros(Some("yes"), 200), Micros::Fixed(0));
        assert_eq!(micros(Some("-5"), 200), Micros::Fixed(0));
        // Off first, then on, twenty seconds each.
        let at = |s: u64| alt_turn(std::time::Duration::from_secs(s));
        assert!(!at(0) && !at(19) && at(20) && at(39) && !at(40) && at(60));
    }
}
