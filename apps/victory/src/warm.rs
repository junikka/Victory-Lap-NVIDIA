//! `victory warm`: host pipelines made ahead of a run (`known_pipelines`), in
//! processes of their own: a compiler that crashes or hangs costs one pipeline.
//! `victory warm --used` keeps the ones earlier runs drew with in the host
//! pipeline library, in one process, so a run loads them instead of making
//! them.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Mutex, OnceLock, mpsc};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use victory_hle::known_pipelines as known;

/// Pipelines a process makes before another takes its place: the compiler's
/// and the translations' memory goes with it.
const AT_ONCE: usize = 40;
/// A pipeline that takes longer is taken for one the compiler hangs on.
const LONGEST: Duration = Duration::from_secs(180);

/// As the runs count one that takes long.
const SLOW_MS: f64 = 1000.0;

fn folder() -> Result<PathBuf> {
    let cfg = victory_core::Config::load()?;
    Ok(cfg.cache_subdir("shaders")?.with_file_name("pipelines"))
}

pub fn run(
    jobs: Option<usize>,
    limit: Option<usize>,
    again: bool,
    used: bool,
    these: Option<String>,
    these_file: Option<PathBuf>,
    into_library: bool,
) -> Result<()> {
    let dir = folder()?;
    if these.is_some() || these_file.is_some() {
        return make(&dir, these.as_deref(), these_file.as_deref(), jobs, into_library);
    }
    let began = Instant::now();
    let keys = known::keys(&dir);
    println!("pipeline book {} ({} pipelines)", known::book_path(&dir).display(), keys.len());
    victory_hle::known_pipelines::use_translations(dir.with_file_name("shaders"));
    let fingerprint = known::fingerprint();
    let done_file = dir.join("warm-done.txt");
    let bad_file = dir.join("warm-bad.txt");
    let listed = std::fs::read_to_string(&done_file).unwrap_or_default();
    let mut done: HashSet<&str> = HashSet::new();
    for line in listed.lines() {
        if let Some((of, key)) = line.split_once(' ')
            && of == fingerprint
        {
            done.insert(key);
        }
    }
    let bad = std::fs::read_to_string(&bad_file).unwrap_or_default();
    let bad: HashSet<&str> = bad.lines().filter_map(|l| l.split(' ').next()).collect();
    let early = std::fs::read_to_string(dir.join("early-pipelines.txt")).unwrap_or_default();
    let early: HashSet<&str> = early.lines().collect();
    // Those earlier runs drew with first, the others in no order (by key).
    let mut all: Vec<usize> = (0..keys.len())
        .filter(|&n| {
            let key = keys[n].as_str();
            (again || !done.contains(key)) && !bad.contains(key) && (!used || early.contains(key))
        })
        .collect();
    all.sort_by_key(|&n| (!early.contains(keys[n].as_str()), &keys[n]));
    all.truncate(limit.unwrap_or(usize::MAX));
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let jobs = jobs.unwrap_or(cores * 2 / 3).max(1);
    println!(
        "{} pipelines known, {} made before ({fingerprint}), {} bad; {} to make, {jobs} {}",
        keys.len(),
        done.len(),
        bad.len(),
        all.len(),
        if used { "threads" } else { "processes" }
    );
    let state = Mutex::new((Vec::<f64>::new(), 0usize, 0usize));
    let done_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&done_file)?);
    // (Read back by the runs: these are made beside the others, not before.)
    let slow_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(dir.join("slow-pipelines.txt"))?);
    let bad_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&bad_file)?);
    let strict_exit = std::env::var("VICTORY_WARM_STRICT_EXIT").is_ok_and(|v| v == "1");
    if used {
        // One process holds the pipeline library: several would each load it
        // and the last save would throw the others' pipelines away.
        let notes = Notes { fingerprint: &fingerprint, keys: &keys, done: &done_out, slow: &slow_out, bad: &bad_out };
        library_run(&dir, &all, jobs, &began, &state, &notes)?;
    } else {
        run_batches(
            &all,
            jobs,
            strict_exit,
            start_worker,
            |n, ms, there| {
                if ms >= SLOW_MS {
                    writeln!(lock(&slow_out), "\n{} {ms:.0}", keys[n]).context("writing slow-pipelines.txt")?;
                }
                let mut state = lock(&state);
                state.0.push(ms);
                state.1 += usize::from(!there);
                let count = state.0.len();
                // A short run (--limit) shows each one.
                if limit.is_some() || count.is_multiple_of(1000) {
                    println!("{count} of {}, {:.0} s", all.len(), began.elapsed().as_secs_f64());
                }
                drop(state);
                writeln!(lock(&done_out), "{fingerprint} {}", keys[n]).context("writing warm-done.txt")
            },
            |n, why| {
                println!("pipeline {n} {}: {why}", keys[n]);
                writeln!(lock(&bad_out), "{} {why}", keys[n]).context("writing warm-bad.txt")?;
                lock(&state).2 += 1;
                Ok(())
            },
        )?;
    }
    let (mut times, refused, crashed) = state.into_inner().unwrap_or_else(|e| e.into_inner());
    times.sort_by(f64::total_cmp);
    let at = |q: f64| times.get((times.len() as f64 * q) as usize).copied().unwrap_or(0.0);
    println!(
        "{} made in {:.0} s ({refused} have no host pipeline, {crashed} ended their process); ms each: median {:.1}, p90 {:.1}, p99 {:.1}, most {:.0}, sum {:.0} s",
        times.len(),
        began.elapsed().as_secs_f64(),
        at(0.5),
        at(0.9),
        at(0.99),
        times.last().copied().unwrap_or(0.0),
        times.iter().sum::<f64>() / 1e3
    );
    Ok(())
}

fn run_batches(
    all: &[usize],
    jobs: usize,
    strict_exit: bool,
    start: impl Fn(&[usize]) -> Result<Child> + Sync,
    each: impl Fn(usize, f64, bool) -> Result<()> + Sync,
    bad: impl Fn(usize, String) -> Result<()> + Sync,
) -> Result<()> {
    let chunks: Vec<&[usize]> = all.chunks(AT_ONCE).collect();
    let next = AtomicUsize::new(0);
    // Stop launching workers after a setup failure. In-flight workers can still report.
    let failure = OnceLock::new();
    std::thread::scope(|s| {
        for _ in 0..jobs.min(chunks.len()) {
            s.spawn(|| {
                while let Some(&chunk) = chunks.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let mut rest = chunk;
                    while !rest.is_empty() {
                        if failure.get().is_some() {
                            return;
                        }
                        let outcome = start(rest)
                            .context("could not start warm worker")
                            .and_then(|process| child(process, rest, strict_exit, &each));
                        match outcome {
                            Ok(Some((made, why))) => {
                                // child only returns a failure index after a validated start.
                                let Some((&n, after)) = rest.get(made..).and_then(<[usize]>::split_first) else {
                                    break;
                                };
                                if let Err(error) = bad(n, why) {
                                    let _ = failure.set(error);
                                    return;
                                }
                                rest = after;
                            }
                            Ok(None) => break,
                            Err(error) => {
                                let _ = failure.set(error);
                                return;
                            }
                        }
                    }
                }
            });
        }
    });
    failure.into_inner().map_or(Ok(()), Err)
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn start_worker(these: &[usize]) -> Result<Child> {
    use std::os::windows::process::CommandExt;
    const IDLE_PRIORITY_CLASS: u32 = 0x40;
    let exe = std::env::current_exe().context("could not locate the warm worker executable")?;
    let list: Vec<String> = these.iter().map(usize::to_string).collect();
    Command::new(&exe)
        .args(["warm", "--these", &list.join(",")])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .creation_flags(IDLE_PRIORITY_CLASS)
        .spawn()
        .with_context(|| format!("could not spawn {}", exe.display()))
}

/// The worker reports a start and a completion for each requested index, in order.
struct Progress<'a> {
    expected: &'a [usize],
    made: usize,
    active: bool,
}

impl<'a> Progress<'a> {
    fn new(expected: &'a [usize]) -> Self {
        Self { expected, made: 0, active: false }
    }

    fn line(&mut self, line: &str) -> Result<Option<(usize, f64, bool)>> {
        let mut words = line.split_whitespace();
        match words.next() {
            Some("at") => {
                let n: usize = words.next().context("missing pipeline index")?.parse()?;
                anyhow::ensure!(
                    !self.active && self.expected.get(self.made) == Some(&n) && words.next().is_none(),
                    "unexpected pipeline start: {line}"
                );
                self.active = true;
                Ok(None)
            }
            Some("made") => {
                let n: usize = words.next().context("missing pipeline index")?.parse()?;
                let ms: f64 = words.next().context("missing pipeline duration")?.parse()?;
                let there = match words.next() {
                    Some("0") => false,
                    Some("1") => true,
                    _ => anyhow::bail!("invalid pipeline result: {line}"),
                };
                anyhow::ensure!(
                    self.active && self.expected.get(self.made) == Some(&n)
                        && ms.is_finite() && ms >= 0.0 && words.next().is_none(),
                    "unexpected pipeline completion: {line}"
                );
                self.made += 1;
                self.active = false;
                Ok(Some((n, ms, there)))
            }
            _ => Ok(None),
        }
    }

    /// `strict_exit` (`VICTORY_WARM_STRICT_EXIT=1`): only a hang or a device
    /// loss costs the active pipeline, and any other abnormal exit stops the
    /// warm-up; unset, any abnormal exit during a pipeline costs that one.
    fn finish(self, status: ExitStatus, timed_out: bool, strict_exit: bool) -> Result<Option<(usize, String)>> {
        if status.success() && !timed_out {
            anyhow::ensure!(
                self.made == self.expected.len() && !self.active,
                "warm worker exited successfully after reporting only {} of {} pipelines",
                self.made, self.expected.len()
            );
            return Ok(None);
        }
        let why = if timed_out {
            format!("hung for {} s", LONGEST.as_secs())
        } else {
            status.to_string()
        };
        // Only a pipeline that was started is blamed. Strict: for process
        // exits, a DXGI device-loss HRESULT, not any negative exit code
        // (which includes unrelated NTSTATUS errors).
        if self.active && (timed_out || !strict_exit || device_loss(status)) {
            return Ok(Some((self.made, why)));
        }
        anyhow::bail!("warm worker failed after {} of {} pipelines: {why}", self.made, self.expected.len())
    }
}

fn device_loss(status: ExitStatus) -> bool {
    // DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_RESET.
    matches!(status.code().map(|code| code as u32), Some(0x887A0005..=0x887A0007))
}

/// Only return a bad pipeline for a failure attributable to an active pipeline.
fn child(
    mut process: Child,
    these: &[usize],
    strict_exit: bool,
    mut each: impl FnMut(usize, f64, bool) -> Result<()>,
) -> Result<Option<(usize, String)>> {
    let Some(out) = process.stdout.take() else {
        let _ = process.kill();
        process.wait().context("warm worker stdout unavailable, failed to reap worker")?;
        anyhow::bail!("warm worker stdout unavailable");
    };
    let process = Mutex::new(process);
    let heard = Mutex::new(Instant::now());
    let hung = AtomicBool::new(false);
    let mut progress = Progress::new(these);
    let status = std::thread::scope(|s| -> Result<ExitStatus> {
        let (finished, wake) = mpsc::channel::<()>();
        let (watch_heard, watch_hung, watch_process) = (&heard, &hung, &process);
        s.spawn(move || {
            while let Err(mpsc::RecvTimeoutError::Timeout) = wake.recv_timeout(Duration::from_millis(500)) {
                if lock(watch_heard).elapsed() > LONGEST {
                    let mut worker = lock(watch_process);
                    if matches!(worker.try_wait(), Ok(None)) && worker.kill().is_ok() {
                        watch_hung.store(true, Ordering::Relaxed);
                    }
                    return;
                }
            }
        });
        let result = (|| -> Result<ExitStatus> {
            for line in BufReader::new(out).lines() {
                let line = line.context("reading warm worker output")?;
                if let Some((n, ms, there)) = progress.line(&line)? {
                    each(n, ms, there)?;
                }
                *lock(&heard) = Instant::now();
            }
            // Keep the watchdog alive even if stdout closes before the process exits.
            loop {
                if let Some(status) = lock(&process).try_wait().context("waiting for warm worker")? {
                    return Ok(status);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        })();
        if result.is_err() {
            let mut worker = lock(&process);
            let _ = worker.kill();
            let _ = worker.wait();
        }
        drop(finished);
        result
    })?;
    progress.finish(status, hung.load(Ordering::Relaxed), strict_exit)
}

struct Batch {
    made: Vec<(usize, f64, bool)>,
    inflight: Vec<usize>,
    failed: Option<String>,
    /// The child said the library could not be opened or saved.
    saw_error: bool,
}

/// What [`library_run`] writes its lines with: the pipelines' keys, the
/// fingerprint the done lines carry, and the done, slow and bad lists.
struct Notes<'a> {
    fingerprint: &'a str,
    keys: &'a [String],
    done: &'a Mutex<std::fs::File>,
    slow: &'a Mutex<std::fs::File>,
    bad: &'a Mutex<std::fs::File>,
}

/// `--used`: one process, the host pipeline library open, `jobs` threads.
/// Pipelines are noted done only after that process has saved the library.
fn library_run(
    dir: &Path,
    all: &[usize],
    jobs: usize,
    began: &Instant,
    state: &Mutex<(Vec<f64>, usize, usize)>,
    notes: &Notes,
) -> Result<()> {
    let Notes { fingerprint, keys, done: done_out, slow: slow_out, bad: bad_out } = *notes;
    let mut todo: Vec<usize> = all.to_vec();
    let list_path = dir.join(format!("warm-list-{}.txt", std::process::id()));
    let mut passes = 0usize;
    while !todo.is_empty() {
        passes += 1;
        if passes > all.len() + 1 {
            let _ = std::fs::remove_file(&list_path);
            anyhow::bail!("warm stopped before the list was done");
        }
        let body = todo.iter().map(usize::to_string).collect::<Vec<_>>().join("\n");
        std::fs::write(&list_path, body).with_context(|| format!("writing {}", list_path.display()))?;
        let batch = run_batch(&list_path, jobs, |n, ms, there| {
            if ms >= SLOW_MS {
                let _ = writeln!(lock(slow_out), "\n{} {ms:.0}", keys[n]);
            }
            let mut state = lock(state);
            state.0.push(ms);
            state.1 += usize::from(!there);
            let count = state.0.len();
            println!("{count} of {}, {:.0} s", all.len(), began.elapsed().as_secs_f64());
        })?;
        if batch.saw_error {
            let _ = std::fs::remove_file(&list_path);
            anyhow::bail!("{}", batch.failed.unwrap_or_else(|| "the pipeline library was not saved".into()));
        }
        if batch.failed.is_none() {
            for (n, _, _) in &batch.made {
                let _ = writeln!(lock(done_out), "{fingerprint} {}", keys[*n]);
            }
            break;
        }
        let why = batch.failed.clone().unwrap_or_else(|| "ended".into());
        let hung: Vec<usize> = if batch.inflight.is_empty() { todo.first().copied().into_iter().collect() } else { batch.inflight.clone() };
        for n in &hung {
            println!("pipeline {n} {}: {why}", keys[*n]);
            let _ = writeln!(lock(bad_out), "{} {why}", keys[*n]);
            lock(state).2 += 1;
        }
        let hung: HashSet<usize> = hung.into_iter().collect();
        todo.retain(|n| !hung.contains(n));
    }
    let _ = std::fs::remove_file(&list_path);
    Ok(())
}

/// One process making `list` (indices, one per line) into the pipeline library.
fn run_batch(list: &Path, jobs: usize, mut each: impl FnMut(usize, f64, bool)) -> Result<Batch> {
    let jobs_s = jobs.to_string();
    let list_s = list.display().to_string();
    let spawned = std::env::current_exe().and_then(|exe| {
        use std::os::windows::process::CommandExt;
        const IDLE_PRIORITY_CLASS: u32 = 0x40;
        Command::new(exe)
            .args(["warm", "--jobs", &jobs_s, "--into-library", "--these-file", &list_s])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .creation_flags(IDLE_PRIORITY_CLASS)
            .spawn()
    });
    let mut process = spawned.context("starting a warm process")?;
    let out = process.stdout.take().context("warm process output")?;
    let process = Mutex::new(process);
    let (heard, over) = (Mutex::new(Instant::now()), AtomicBool::new(false));
    let hung = AtomicBool::new(false);
    let mut made = Vec::new();
    let mut inflight = HashSet::new();
    let mut failed_line = None;
    std::thread::scope(|s| {
        s.spawn(|| {
            while !over.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(500));
                if lock(&heard).elapsed() > LONGEST {
                    hung.store(true, Ordering::Relaxed);
                    let _ = lock(&process).kill();
                    return;
                }
            }
        });
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            *lock(&heard) = Instant::now();
            let mut words = line.split(' ');
            match words.next() {
                Some("at") => {
                    if let Ok(n) = words.next().unwrap_or("").parse() {
                        inflight.insert(n);
                    }
                }
                Some("made") => {
                    if let (Some(n), Some(ms), Some(there)) = (words.next(), words.next(), words.next())
                        && let (Ok(n), Ok(ms)) = (n.parse::<usize>(), ms.parse::<f64>())
                    {
                        inflight.remove(&n);
                        made.push((n, ms, there == "1"));
                        each(n, ms, there == "1");
                    }
                }
                Some("loading") | Some("saving") | Some("error") => {
                    if line.starts_with("error") {
                        failed_line = Some(line.clone());
                    }
                    println!("{line}");
                }
                _ => {}
            }
        }
        over.store(true, Ordering::Relaxed);
    });
    let status = lock(&process).wait();
    let hung = hung.load(Ordering::Relaxed);
    let clean = status.as_ref().is_ok_and(|s| s.success()) && !hung && failed_line.is_none() && inflight.is_empty();
    let failed = if clean {
        None
    } else if let Some(line) = failed_line.clone() {
        Some(line)
    } else if hung {
        Some(format!("hung for {} s", LONGEST.as_secs()))
    } else if status.as_ref().is_ok_and(|s| s.success()) {
        Some("stopped before the rest".into())
    } else {
        Some(status.map_or_else(|e| e.to_string(), |s| format!("{s}")))
    };
    Ok(Batch { made, inflight: inflight.into_iter().collect(), failed, saw_error: failed_line.is_some() })
}

fn tell(out: &Mutex<std::io::Stdout>, n: usize, came: Option<(f64, bool)>) {
    let mut out = lock(out);
    let told = match came {
        None => writeln!(out, "at {n}"),
        Some((ms, there)) => writeln!(out, "made {n} {ms:.2} {}", u8::from(there)),
    };
    if told.and_then(|()| out.flush()).is_err() {
        // Whoever asked is gone.
        std::process::exit(1);
    }
}

/// What such a process does.
fn make(dir: &Path, these: Option<&str>, file: Option<&Path>, jobs: Option<usize>, into_library: bool) -> Result<()> {
    let text = if let Some(path) = file {
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?
    } else {
        these.unwrap_or("").to_owned()
    };
    let these: Vec<usize> = text
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|n| n.parse().context("pipeline number"))
        .collect::<Result<_>>()?;
    let cfg = victory_core::Config::load()?;
    victory_hle::known_pipelines::use_translations(cfg.cache_subdir("shaders")?);
    if into_library {
        println!("loading the pipeline library");
        let _ = std::io::stdout().flush();
        if let Err(e) = victory_hle::known_pipelines::open_library(dir.to_path_buf()) {
            println!("error {e}");
            let _ = std::io::stdout().flush();
            std::process::exit(1);
        }
    }
    if !these.is_empty() {
        let threads = if into_library { jobs.unwrap_or(1).max(1).min(these.len()) } else { 1 };
        let out = Mutex::new(std::io::stdout());
        if threads == 1 {
            known::make(dir, &these, |n, came| tell(&out, n, came));
        } else {
            let size = these.len().div_ceil(threads);
            std::thread::scope(|s| {
                for shard in these.chunks(size) {
                    s.spawn(|| known::make(dir, shard, |n, came| tell(&out, n, came)));
                }
            });
        }
    }
    if into_library {
        println!("saving the pipeline library");
        let _ = std::io::stdout().flush();
        if let Err(e) = victory_hle::known_pipelines::save_library() {
            println!("error {e}");
            let _ = std::io::stdout().flush();
            std::process::exit(1);
        }
    }
    // Without what a run's end does (nothing to save, the host's threads).
    let _ = std::io::stdout().flush();
    std::process::exit(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::windows::process::CommandExt;

    fn fake_child(script: &str, stdout: Stdio) -> Result<Child> {
        let cmd = PathBuf::from(std::env::var_os("SystemRoot").unwrap()).join(r"System32\cmd.exe");
        Ok(Command::new(cmd)
            .args(["/d", "/c"])
            .raw_arg(script)
            .stdin(Stdio::null())
            .stdout(stdout)
            .stderr(Stdio::null())
            .spawn()?)
    }

    #[test]
    fn missing_executable_leaves_the_batch_pending() {
        let missing = std::env::temp_dir().join(format!("victory-missing-worker-{}.exe", std::process::id()));
        assert!(!missing.try_exists().unwrap());
        let starts = AtomicUsize::new(0);
        let done = Mutex::new(Vec::new());
        let bad = Mutex::new(Vec::new());
        let result = run_batches(
            &[7, 8, 9],
            1,
            false,
            |_| {
                starts.fetch_add(1, Ordering::Relaxed);
                Ok(Command::new(&missing).spawn()?)
            },
            |n, _, _| { lock(&done).push(n); Ok(()) },
            |n, _| { lock(&bad).push(n); Ok(()) },
        );
        assert_eq!(starts.load(Ordering::Relaxed), 1);
        assert!(lock(&done).is_empty());
        assert!(lock(&bad).is_empty());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("start warm worker"));
        assert_eq!(error.downcast_ref::<std::io::Error>().unwrap().kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn missing_stdout_is_a_startup_failure() {
        let error = run_batches(
            &[1, 2],
            1,
            false,
            |_| fake_child("exit /b 0", Stdio::null()),
            |_, _, _| panic!("unexpected completion"),
            |_, _| panic!("unexpected bad pipeline"),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("stdout"));
    }

    #[test]
    fn device_loss_records_only_the_active_pipeline_and_resumes() {
        let done = Mutex::new(Vec::new());
        let bad = Mutex::new(Vec::new());
        run_batches(
            &[7, 8, 9],
            1,
            false,
            |rest| match rest {
                [7, 8, 9] => fake_child("echo at 7&echo made 7 3.5 1&echo at 8&exit /b -2005270523", Stdio::piped()),
                [9] => fake_child("echo at 9&echo made 9 2.0 0", Stdio::piped()),
                _ => panic!("unexpected batch {rest:?}"),
            },
            |n, ms, there| { lock(&done).push((n, ms, there)); Ok(()) },
            |n, why| { lock(&bad).push((n, why)); Ok(()) },
        )
        .unwrap();
        assert_eq!(*lock(&done), [(7, 3.5, true), (9, 2.0, false)]);
        let bad = lock(&bad);
        assert_eq!(bad.len(), 1);
        assert_eq!(bad[0].0, 8);
        assert!(!bad[0].1.is_empty());
    }

    #[test]
    fn startup_failure_preserves_another_workers_completion() {
        // Hold the failed start until its peer has reported the only key in its chunk.
        let keys: Vec<_> = (0..=AT_ONCE).collect();
        let (confirmed, received) = std::sync::mpsc::channel();
        let received = Mutex::new(received);
        let error = run_batches(
            &keys,
            2,
            false,
            |chunk| {
                if chunk[0] == 0 {
                    assert_eq!(lock(&received).recv_timeout(Duration::from_secs(5)).unwrap(), AT_ONCE);
                    anyhow::bail!("worker unavailable")
                }
                fake_child(&format!("echo at {AT_ONCE}&echo made {AT_ONCE} 1 1"), Stdio::piped())
            },
            |n, _, _| { confirmed.send(n).unwrap(); Ok(()) },
            |_, _| panic!("unexpected bad pipeline"),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("worker unavailable"));
    }

    #[test]
    fn strict_generic_exit_leaves_active_pipeline_pending() {
        for code in [7, -1073741510, -1073741819] {
            let done = Mutex::new(Vec::new());
            let error = run_batches(
                &[7, 8, 9],
                1,
                true,
                |_| fake_child(&format!("echo at 7&echo made 7 3.5 1&echo at 8&exit /b {code}"), Stdio::piped()),
                |n, _, _| { lock(&done).push(n); Ok(()) },
                |_, _| panic!("generic failures must not blacklist a pipeline"),
            ).unwrap_err();
            assert!(error.to_string().contains("after 1 of 3 pipelines"));
            assert_eq!(*lock(&done), [7]);
        }
    }

    #[test]
    fn generic_exit_records_the_active_pipeline_and_resumes() {
        let done = Mutex::new(Vec::new());
        let bad = Mutex::new(Vec::new());
        run_batches(
            &[7, 8, 9],
            1,
            false,
            |rest| match rest {
                [7, 8, 9] => fake_child("echo at 7&echo made 7 3.5 1&echo at 8&exit /b 7", Stdio::piped()),
                [9] => fake_child("echo at 9&echo made 9 2.0 0", Stdio::piped()),
                _ => panic!("unexpected batch {rest:?}"),
            },
            |n, _, _| { lock(&done).push(n); Ok(()) },
            |n, _| { lock(&bad).push(n); Ok(()) },
        )
        .unwrap();
        assert_eq!(*lock(&done), [7, 9]);
        assert_eq!(*lock(&bad), [8]);
    }

    #[test]
    fn incomplete_success_leaves_unreported_pipelines_pending() {
        for (script, completed) in [
            ("exit /b 0", vec![]),
            ("echo at 7&echo made 7 1 1&exit /b 0", vec![7]),
            ("echo at 7&exit /b 0", vec![]),
        ] {
            let done = Mutex::new(Vec::new());
            let error = run_batches(
                &[7, 8], 1, false,
                |_| fake_child(script, Stdio::piped()),
                |n, _, _| { lock(&done).push(n); Ok(()) },
                |_, _| panic!("incomplete success must not blacklist a pipeline"),
            ).unwrap_err();
            assert!(error.to_string().contains("exited successfully after reporting only"));
            assert_eq!(*lock(&done), completed);
        }
    }

    #[test]
    fn device_loss_and_timeout_require_an_active_pipeline() {
        use std::os::windows::process::ExitStatusExt;
        for code in [0x887A0005, 0x887A0006, 0x887A0007] {
            let status = ExitStatus::from_raw(code);
            assert!(Progress::new(&[7]).finish(status, false, true).is_err());
            let mut progress = Progress::new(&[7]);
            assert_eq!(progress.line("at 7").unwrap(), None);
            assert_eq!(progress.finish(status, false, true).unwrap().unwrap().0, 0);
        }
        let status = ExitStatus::from_raw(1);
        assert!(Progress::new(&[7]).finish(status, true, true).is_err());
        let mut progress = Progress::new(&[7]);
        assert_eq!(progress.line("at 7").unwrap(), None);
        assert_eq!(progress.finish(status, true, true).unwrap().unwrap().0, 0);
        let mut finished = Progress::new(&[7]);
        assert_eq!(finished.line("at 7").unwrap(), None);
        assert!(finished.line("made 7 0 1").unwrap().is_some());
        assert!(finished.finish(ExitStatus::from_raw(0x887A0005), false, true).is_err());
    }

    #[test]
    fn malformed_reports_do_not_complete_or_blacklist_pipelines() {
        for script in [
            "echo at 8",
            "echo made 7 1 1",
            "echo at 7&echo at 7",
            "echo at 7&echo made 8 1 1",
            "echo at 7&echo made 7 NaN 1",
            "echo at 7&echo made 7 1 2",
        ] {
            assert!(run_batches(
                &[7, 8], 1, false,
                |_| fake_child(script, Stdio::piped()),
                |_, _, _| panic!("invalid report"),
                |_, _| panic!("invalid report"),
            ).is_err());
        }
    }

    #[test]
    fn journal_failure_reaches_the_caller() {
        let error = run_batches(
            &[7, 8], 1, false,
            |_| fake_child("echo at 7&echo made 7 1 1", Stdio::piped()),
            |_, _, _| anyhow::bail!("journal unavailable"),
            |_, _| panic!("journal error must not blacklist a pipeline"),
        ).unwrap_err();
        assert_eq!(error.to_string(), "journal unavailable");
    }
}
