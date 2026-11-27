//! A fixed game step for test runs (`VICTORY_FIXED_STEP=1`, off without it):
//! every frame moves the title's simulation by the same time, whatever the
//! real time since the last frame, so a replay keyed by frame sees the same
//! game time on the same frame in every run.
//!
//! What the title does (`frame_rate`): +0x22264d0 hands each of its timers
//! the real time since the last frame (`QueryPerformanceCounter`), and
//! +0x2226040 makes the step of it, `clamp(real, 1 / rate, 1/15 s)`, the
//! shortest step's rate coming from +0x2241fd0. At our 7-15 fps the real time
//! lies between the two bounds, so each frame's step is a length of its own.
//!
//! The cap (1/15 s) is a constant inside +0x2226040 whose bytes we do not
//! have; the floor is the `SimRate` site of `frame_rate`, which we do. So the
//! floor is raised to the cap: +0x2241fd0 returns 15, the shortest step is
//! 1/15 s, and `clamp(real, 1/15, 1/15)` is 1/15 s for every frame, in either
//! order the two bounds are taken. Game time moves 66.667 ms per frame: slow
//! motion below 15 fps, fast above it (the title's own limiter keeps frames
//! at 30 or under). A step of `1 / rate` at a rate above 15 (33.333 ms at 30)
//! needs the cap changed too, which needs its bytes.
//!
//! The site's bytes are compared before anything is written (`frame_rate::
//! apply`); a site whose bytes differ is left alone and said so.

use crate::frame_rate::{Kind, SITES, Site};

/// The simulation's step rate with the fixed step: the cap's rate, 15.
pub const STEP_RATE: u32 = 15;
/// The title's longest step, in seconds (1/15 s).
pub const CAP_S: f32 = 1.0 / 15.0;

/// The switch: `VICTORY_FIXED_STEP=1`.
pub fn asked() -> bool {
    on(std::env::var("VICTORY_FIXED_STEP").ok().as_deref())
}

/// `1` turns it on; anything else (or nothing) leaves the title's step alone.
pub fn on(spec: Option<&str>) -> bool {
    spec.is_some_and(|v| v.trim() == "1")
}

/// The site the fixed step patches: the shortest step's rate (+0x2241fd0).
pub fn site() -> &'static Site {
    SITES.iter().find(|s| s.kind == Kind::SimRate).expect("frame_rate has its SimRate site")
}

/// The step the title makes of `real` seconds since the last frame with its
/// shortest step at `1 / floor_rate`, as its code does (`frame_rate`): at
/// most the cap, at least the floor. `floor_first` takes the bounds in the
/// other order (which one the code takes first is not known; with the fixed
/// step it does not matter).
pub fn title_step(real: f32, floor_rate: u32, floor_first: bool) -> f32 {
    let floor = 1.0 / floor_rate.max(1) as f32;
    if floor_first { real.max(floor).min(CAP_S) } else { real.min(CAP_S).max(floor) }
}

/// The fixed step, in milliseconds.
pub fn step_ms() -> f64 {
    1000.0 / f64::from(STEP_RATE)
}

/// Patches the site in the image of `size` bytes at `base`.
///
/// # Safety
/// The image stays mapped and no thread runs its code yet.
#[cfg(windows)]
pub unsafe fn apply(base: usize, size: usize) -> crate::frame_rate::Patched {
    // SAFETY: as the caller promises.
    let done = unsafe { crate::frame_rate::patch(base, size, STEP_RATE, std::slice::from_ref(site()), "VICTORY_FIXED_STEP") };
    done[0]
}

/// What the title's timers held at one look (`frame_rate::TIMERS`): the first
/// timer's time (ms) and frames, and the steps of all of them (seconds).
#[derive(Debug, Clone, PartialEq)]
pub struct Look {
    pub time_ms: u32,
    pub frames: u32,
    pub steps: Vec<f32>,
    pub presented: u64,
}

/// The line logged every 10 s: the step asked for, the steps the title's
/// timers hold, and the frames stepped since the last look.
pub fn line(before: &Look, now: &Look) -> String {
    let (lo, hi) = now.steps.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &s| (lo.min(s), hi.max(s)));
    let frames = now.frames.wrapping_sub(before.frames);
    let game_ms = now.time_ms.wrapping_sub(before.time_ms);
    let per = if frames == 0 { 0.0 } else { f64::from(game_ms) / f64::from(frames) };
    format!(
        "fixed step: {:.3} ms asked; the title's timers step {:.3}-{:.3} ms; {frames} frames stepped (game time +{game_ms} ms, {per:.3} ms a frame), {} presented",
        step_ms(),
        lo * 1000.0,
        hi * 1000.0,
        now.presented.wrapping_sub(before.presented),
    )
}

/// Logs `line` every 10 s while the title runs. `presented` counts our
/// presented frames.
#[cfg(windows)]
pub fn log_every_10s(base: usize, size: usize, presented: fn() -> u64) {
    use crate::frame_rate::TIMERS;
    if TIMERS.iter().any(|&rva| rva + 0x14 > size) {
        return;
    }
    let look = move || {
        // SAFETY: inside the mapped image (checked above), title data.
        let u32_at = |rva: usize| unsafe { ((base + rva) as *const u32).read_volatile() };
        Look {
            time_ms: u32_at(TIMERS[0]),
            frames: u32_at(TIMERS[0] + 4),
            steps: TIMERS.iter().map(|&t| f32::from_bits(u32_at(t + 0xc))).collect(),
            presented: presented(),
        }
    };
    let _ = std::thread::Builder::new().name("fixed step log".into()).spawn(move || {
        let mut before = look();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            let now = look();
            tracing::info!("{}", line(&before, &now));
            before = now;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch() {
        assert!(on(Some("1")));
        assert!(on(Some(" 1 ")));
        assert!(!on(Some("0")));
        assert!(!on(Some("yes")));
        assert!(!on(None));
    }

    /// The site is the shortest step's rate, and the fixed step's code
    /// returns 15 there.
    #[test]
    fn the_site_and_its_code() {
        let s = site();
        assert_eq!(s.at, 0x2241fdb);
        assert_eq!(&s.bytes[..7], &[0x83, 0xb8, 0x10, 0x02, 0x00, 0x00, 0x00]);
        let code = crate::frame_rate::replacement(Kind::SimRate, STEP_RATE);
        assert_eq!(code, [0xb8, 15, 0, 0, 0, 0xeb, 0x0d]);
        assert_eq!(code.len() + usize::from(code[6]), 20, "the jump lands on the vcvtsi2ss");
    }

    /// With the floor at the cap, every real frame time gives the same step,
    /// whichever bound is taken first; with the title's own 30 it does not.
    #[test]
    fn step_per_rate() {
        let frames = [0.005, 0.0333, 0.05, 0.0667, 0.071, 0.1, 0.143, 0.5];
        for floor_first in [false, true] {
            for real in frames {
                assert_eq!(title_step(real, STEP_RATE, floor_first), CAP_S, "{real} s");
            }
            // The title's 30: real time between 33.3 and 66.7 ms.
            assert_eq!(title_step(0.05, 30, floor_first), 0.05);
            assert_eq!(title_step(0.01, 30, floor_first), 1.0 / 30.0);
            assert_eq!(title_step(0.1, 30, floor_first), CAP_S);
            // Below 15 the floor passes the cap: the step is no longer known.
            assert!(title_step(0.05, 10, floor_first) >= CAP_S);
        }
        assert!((step_ms() - 66.667).abs() < 0.001);
    }

    #[test]
    fn the_log_line() {
        let before = Look { time_ms: 1000, frames: 10, steps: vec![CAP_S; 7], presented: 12 };
        let now = Look { time_ms: 11000, frames: 160, steps: vec![CAP_S; 7], presented: 162 };
        assert_eq!(
            line(&before, &now),
            "fixed step: 66.667 ms asked; the title's timers step 66.667-66.667 ms; 150 frames stepped (game time +10000 ms, 66.667 ms a frame), 150 presented"
        );
        let idle = Look { frames: 10, ..now.clone() };
        assert!(line(&before, &idle).contains("0 frames stepped"));
    }

    /// The real code: the shortest step's window, made a function over
    /// settings of 30, returns 15 once patched; other bytes are left alone.
    #[cfg(windows)]
    #[test]
    fn the_site_returns_15_and_others_are_left_alone() {
        use crate::frame_rate::{Patched, patch};
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc, VirtualFree};
        // SAFETY: fresh memory of our own, written and run inside it.
        unsafe {
            let place = VirtualAlloc(std::ptr::null(), 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
            assert!(!place.is_null());
            let settings = [30u32; 0x100];
            let sim = site().bytes;
            let mut f = vec![0x48, 0xb8];
            f.extend((settings.as_ptr() as u64).to_le_bytes());
            let window_at = f.len();
            f.extend(sim);
            f.push(0xc3);
            std::ptr::copy_nonoverlapping(f.as_ptr(), place.add(0x100), f.len());
            let mut other = f.clone();
            other[window_at + 2] = 0x14; // [rax+214h]
            std::ptr::copy_nonoverlapping(other.as_ptr(), place.add(0x200), other.len());
            // Not run before the patch: the window's `call` goes outside it.
            let rate: extern "system" fn() -> f32 = std::mem::transmute(place.add(0x100));
            let mine = [Site { at: 0x100 + window_at, kind: Kind::SimRate, bytes: sim }, Site { at: 0x200 + window_at, kind: Kind::SimRate, bytes: sim }];
            let done = patch(place as usize, 0x1000, STEP_RATE, &mine, "VICTORY_FIXED_STEP");
            assert_eq!(done, [Patched::Done, Patched::Differs]);
            assert_eq!(rate(), 15.0);
            assert_eq!(std::slice::from_raw_parts(place.add(0x200), other.len()), &other[..]);
            VirtualFree(place.cast(), 0, MEM_RELEASE);
        }
    }
}
