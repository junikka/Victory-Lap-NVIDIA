//! The free runs of a row of slots, kept by length: the smallest run that
//! holds what is asked for is taken (best fit), and what is given back joins
//! the free runs beside it.
//!
//! Port of miamiwalk 9c4e9690 (`victory-unimem/src/offset.rs`), as its
//! ebcf56f1 uses it for the dynamic descriptor slots: without the alignment
//! and the count of used bytes that nothing here asks for, and in slots.
//! [`super::host`] keeps one beside its table of taken slots with
//! `VICTORY_SLOTS_RANGES=1`.

use std::collections::{BTreeMap, BTreeSet};

/// The free runs of `size` slots.
#[derive(Debug)]
pub(crate) struct FreeRuns {
    /// By where they start: start -> length.
    by_start: BTreeMap<u32, u32>,
    /// The same runs by (length, start), for the smallest that fits.
    by_len: BTreeSet<(u32, u32)>,
}

impl FreeRuns {
    /// All of `size` slots free.
    pub(crate) fn new(size: u32) -> FreeRuns {
        let mut all = FreeRuns { by_start: BTreeMap::new(), by_len: BTreeSet::new() };
        if size > 0 {
            all.insert(0, size);
        }
        all
    }

    /// The longest free run.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn longest(&self) -> u32 {
        self.by_len.last().map_or(0, |&(len, _)| len)
    }

    fn insert(&mut self, start: u32, len: u32) {
        self.by_start.insert(start, len);
        self.by_len.insert((len, start));
    }

    fn remove(&mut self, start: u32, len: u32) {
        self.by_start.remove(&start);
        self.by_len.remove(&(len, start));
    }

    /// `n` slots in a row out of the smallest free run that holds them (of
    /// several that small, the first): the first's index. `None` if no free
    /// run is that long.
    pub(crate) fn take(&mut self, n: u32) -> Option<u32> {
        if n == 0 {
            return None;
        }
        let &(len, start) = self.by_len.range((n, 0)..).next()?;
        self.remove(start, len);
        if len > n {
            self.insert(start + n, len - n);
        }
        Some(start)
    }

    /// Gives back `n` slots from `at` that were taken, all of them (the
    /// caller knows which are): they join the free runs that touch them.
    pub(crate) fn give(&mut self, at: u32, n: u32) {
        if n == 0 {
            return;
        }
        let (mut start, mut end) = (at, at + n);
        if let Some((&before, &len)) = self.by_start.range(..at).next_back() {
            debug_assert!(before + len <= at, "slots given back twice");
            if before + len == at {
                self.remove(before, len);
                start = before;
            }
        }
        debug_assert!(self.by_start.range(at..end).next().is_none(), "slots given back twice");
        if let Some(&len) = self.by_start.get(&end) {
            self.remove(end, len);
            end += len;
        }
        self.insert(start, end - start);
    }

    /// The free runs in order, as (start, length).
    #[cfg(test)]
    fn all(&self) -> Vec<(u32, u32)> {
        assert_eq!(self.by_start.len(), self.by_len.len());
        assert!(self.by_start.iter().all(|(&start, &len)| self.by_len.contains(&(len, start))));
        self.by_start.iter().map(|(&start, &len)| (start, len)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::FreeRuns;

    #[test]
    fn what_is_given_back_joins_into_one_run() {
        let mut runs = FreeRuns::new(1 << 18);
        let x = runs.take(100).unwrap();
        let y = runs.take(4096).unwrap();
        let z = runs.take(64).unwrap();
        assert_eq!((x, y, z), (0, 100, 4196));
        runs.give(y, 4096);
        assert_eq!(runs.all(), [(100, 4096), (4260, (1 << 18) - 4260)]);
        runs.give(x, 100);
        runs.give(z, 64);
        assert_eq!(runs.all(), [(0, 1 << 18)], "the pieces joined again");
        assert_eq!(runs.longest(), 1 << 18);
    }

    #[test]
    fn a_full_row_gives_nothing_and_the_smallest_run_that_fits_is_taken() {
        let mut runs = FreeRuns::new(1000);
        let big = runs.take(600).unwrap();
        let small = runs.take(100).unwrap();
        assert_eq!(runs.take(400), None);
        assert_eq!(runs.take(0), None);
        runs.give(big, 600);
        // 600 free at the front, 300 at the back: 250 goes into the back.
        assert_eq!(runs.take(250), Some(small + 100));
        // Exactly what is left at the back, and then all of the front.
        assert_eq!(runs.take(50), Some(950));
        assert_eq!(runs.take(600), Some(0));
        assert_eq!((runs.longest(), runs.take(1)), (0, None));
        assert_eq!(FreeRuns::new(0).take(1), None);
    }

    /// What the source's commit says of the walk from a moving place: it
    /// takes a small table out of the long run it stands in, though a hole
    /// of just that size lies free, and the next long table finds no run.
    /// Here the hole is taken and the long table still fits.
    #[test]
    fn a_small_table_goes_into_the_hole_and_leaves_the_long_run_whole() {
        let mut runs = FreeRuns::new(16);
        let (a, b, c) = (runs.take(4).unwrap(), runs.take(2).unwrap(), runs.take(4).unwrap());
        assert_eq!((a, b, c), (0, 4, 6));
        runs.give(b, 2);
        assert_eq!(runs.all(), [(4, 2), (10, 6)]);
        assert_eq!(runs.take(2), Some(4));
        assert_eq!(runs.take(6), Some(10));
        // (The walk, from where it last stopped: the 2 out of the 6.)
        let mut taken = [true; 16];
        taken[4..6].fill(false);
        taken[10..16].fill(false);
        let walked = super::super::host::free_run(&taken, (10..16).chain(0..11), 2, false).unwrap();
        assert_eq!(walked, 10);
        taken[10..12].fill(true);
        assert_eq!(super::super::host::free_run(&taken, (12..16).chain(0..16), 6, false), None);
    }

    /// Many random takes and returns against a table of taken slots: what is
    /// taken was free, and the free runs are always the table's.
    #[test]
    fn random_use_keeps_the_runs_those_of_a_table_of_taken_slots() {
        const SIZE: u32 = 4096;
        let mut runs = FreeRuns::new(SIZE);
        let mut taken = vec![false; SIZE as usize];
        let mut live: Vec<(u32, u32)> = Vec::new();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 20) as u32
        };
        let runs_of = |taken: &[bool]| {
            let mut out: Vec<(u32, u32)> = Vec::new();
            for (at, _) in taken.iter().enumerate().filter(|(_, held)| !**held) {
                match out.last_mut() {
                    Some(last) if last.0 + last.1 == at as u32 => last.1 += 1,
                    _ => out.push((at as u32, 1)),
                }
            }
            out
        };
        for step in 0..20_000 {
            if live.len() > 200 || (next() % 3 == 0 && !live.is_empty()) {
                let (at, n) = live.swap_remove(next() as usize % live.len());
                // (Now and then in two parts, as a list gives a part back.)
                let part = if n > 1 && next() % 4 == 0 { 1 + next() % (n - 1) } else { n };
                runs.give(at, part);
                runs.give(at + part, n - part);
                taken[at as usize..(at + n) as usize].fill(false);
            } else {
                let n = 1 + if next() % 8 == 0 { next() % 512 } else { next() % 12 };
                let free = runs_of(&taken);
                let smallest = free.iter().filter(|r| r.1 >= n).min_by_key(|r| (r.1, r.0)).map(|r| r.0);
                let got = runs.take(n);
                assert_eq!(got, smallest, "the smallest run that holds {n}, the first of those");
                if let Some(at) = got {
                    assert!(taken[at as usize..(at + n) as usize].iter().all(|held| !held), "taken twice");
                    taken[at as usize..(at + n) as usize].fill(true);
                    live.push((at, n));
                }
            }
            if step % 64 == 0 {
                assert_eq!(runs.all(), runs_of(&taken));
            }
        }
        for (at, n) in live {
            runs.give(at, n);
        }
        assert_eq!(runs.all(), [(0, SIZE)]);
    }
}
