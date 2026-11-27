//! Diagnostics: a least-squares fit of the card's time of the title's runs
//! of lists against what is in them (`VICTORY_PASS_TIMES=1` with
//! `VICTORY_KIND_COUNTS=1`).
//!
//! The stamps between lists say how long the card took for each run of the
//! title's lists; the counts say how many draws, dispatches, copies,
//! indirect executions, waits for every earlier shader write and barrier
//! calls each run holds. Over a few thousand runs the fit says what a run
//! costs by itself and what each of those adds: which of them the time goes
//! with. It is a fit, not a measurement of any one command: what goes
//! together in every run (a draw and the memory it reads) cannot be told
//! apart by it.

/// What a run's time is fitted against, after the run itself.
pub(crate) const WHAT: [&str; 6] =
    ["draws", "dispatches", "copies", "indirect executions", "waits for every earlier shader write", "barrier calls"];
/// The run itself and [`WHAT`].
const TERMS: usize = WHAT.len() + 1;

/// The sums a fit is made of.
#[derive(Default)]
pub(super) struct Fit {
    runs: u64,
    xtx: [[f64; TERMS]; TERMS],
    xty: [f64; TERMS],
    y: f64,
    yy: f64,
}

impl Fit {
    /// A run took `ticks` and held `counts` (as [`WHAT`]).
    pub(super) fn add(&mut self, ticks: u64, counts: &[u32; WHAT.len()]) {
        let y = ticks as f64;
        let mut x = [1.0; TERMS];
        for (x, &c) in x[1..].iter_mut().zip(counts) {
            *x = f64::from(c);
        }
        self.runs += 1;
        self.y += y;
        self.yy += y * y;
        for i in 0..TERMS {
            self.xty[i] += x[i] * y;
            for j in 0..TERMS {
                self.xtx[i][j] += x[i] * x[j];
            }
        }
    }

    /// The ticks a run costs by itself and each of [`WHAT`] adds, and how
    /// much of the runs' differences that explains (R squared), if the runs
    /// say: more runs than terms, and no term that is another's multiple in
    /// all of them. A term no run has costs nothing.
    pub(super) fn solve(&self) -> Option<([f64; TERMS], f64)> {
        let used: Vec<usize> = (0..TERMS).filter(|&i| self.xtx[i][i] > 0.0).collect();
        if self.runs as usize <= used.len() + 1 {
            return None;
        }
        let n = used.len();
        // Scaled, so that counts of thousands and the run's 1 weigh alike.
        let scale: Vec<f64> = used.iter().map(|&i| self.xtx[i][i].sqrt()).collect();
        let mut a: Vec<Vec<f64>> = (0..n)
            .map(|r| {
                let mut row: Vec<f64> = (0..n).map(|c| self.xtx[used[r]][used[c]] / (scale[r] * scale[c])).collect();
                row.push(self.xty[used[r]] / scale[r]);
                row
            })
            .collect();
        for col in 0..n {
            let pivot = (col..n).max_by(|&p, &q| a[p][col].abs().total_cmp(&a[q][col].abs()))?;
            if a[pivot][col].abs() < 1e-9 {
                return None;
            }
            a.swap(col, pivot);
            for row in 0..n {
                if row != col {
                    let f = a[row][col] / a[col][col];
                    for k in col..=n {
                        a[row][k] -= f * a[col][k];
                    }
                }
            }
        }
        let mut out = [0.0; TERMS];
        let mut explained = 0.0;
        for (r, &i) in used.iter().enumerate() {
            out[i] = a[r][n] / a[r][r] / scale[r];
            explained += out[i] * self.xty[i];
        }
        let total = self.yy - self.y * self.y / self.runs as f64;
        let left = self.yy - explained;
        let r2 = if total > 0.0 { (1.0 - left / total).clamp(0.0, 1.0) } else { 0.0 };
        Some((out, r2))
    }

    /// The line's text: per frame, the milliseconds that go with the runs
    /// themselves and with each of [`WHAT`], and what one of each costs.
    pub(super) fn text(&self, ticks_per_ms: f64, frames: f64) -> String {
        let Some((terms, r2)) = self.solve() else {
            return format!("{} runs: too few, or too much alike, to say", self.runs);
        };
        let mut parts = Vec::new();
        for (i, &t) in terms.iter().enumerate() {
            // (How many of them there were: the sum of the term over the runs.)
            let count = self.xtx[0][i];
            if count == 0.0 {
                continue;
            }
            let name = if i == 0 { "the runs themselves" } else { WHAT[i - 1] };
            parts.push(format!(
                "{:.2} ms with {name} ({:.1} us each of {:.0})",
                t * count / ticks_per_ms / frames,
                t / ticks_per_ms * 1000.0,
                count / frames
            ));
        }
        format!(
            "{:.1} ms in {:.0} runs a frame; least squares over {} runs, R2 {r2:.2}: {}",
            self.y / ticks_per_ms / frames,
            self.runs as f64 / frames,
            self.runs,
            parts.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fit_finds_what_each_thing_in_a_run_costs() {
        // Made-up runs: 50 ticks each, 3 a draw, 40 a wait, 7 an indirect
        // execution; no run has a copy.
        let mut fit = Fit::default();
        let mut seed = 12345u32;
        let mut next = |most: u32| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 16) % most
        };
        let mut all = 0;
        for _ in 0..500 {
            let counts = [next(400), next(5), 0, next(60), next(12), next(30)];
            let ticks = 50 + 3 * u64::from(counts[0]) + 7 * u64::from(counts[3]) + 40 * u64::from(counts[4]);
            all += ticks;
            fit.add(ticks, &counts);
        }
        let (terms, r2) = fit.solve().expect("500 runs that differ");
        for (found, meant) in terms.iter().zip([50.0, 3.0, 0.0, 0.0, 7.0, 40.0, 0.0]) {
            assert!((found - meant).abs() < 1e-6, "{terms:?}");
        }
        assert!(r2 > 0.999_999, "{r2}");
        // Per frame over 10 frames, at 1,000 ticks a millisecond.
        let text = fit.text(1000.0, 10.0);
        assert!(text.starts_with(&format!("{:.1} ms in 50 runs a frame", all as f64 / 10_000.0)), "{text}");
        assert!(text.contains("2.50 ms with the runs themselves (50.0 us each of 50)"), "{text}");
        assert!(text.contains("(40.0 us each of"), "{text}");
        assert!(!text.contains("copies"), "{text}");
    }

    #[test]
    fn runs_that_are_all_alike_say_nothing() {
        let mut fit = Fit::default();
        assert!(fit.solve().is_none());
        for _ in 0..100 {
            // Twice as many barrier calls as waits, always: neither can be
            // told from the other.
            fit.add(1000, &[10, 0, 0, 0, 4, 8]);
        }
        assert!(fit.solve().is_none());
        assert!(fit.text(1000.0, 1.0).starts_with("100 runs: too few"));
    }
}
