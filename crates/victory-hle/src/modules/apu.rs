//! The console's audio decoder (`xapu.dll`): Opus packets in, float PCM out.
//!
//! A title connects once for one kind of processing, gets a block of memory
//! for its packets and output, and then queues per-stream commands
//! (activate, process, reset, deactivate) and dequeues one result per
//! command. We decode each command when it is queued, so its result is ready
//! at once; the order of results is the order of commands.
//!
//! That decoding runs on the title's own thread, its time-critical mixer,
//! which has 5.3 ms for a block of 256 frames and on the console only
//! queues: the hardware decodes beside it. `VICTORY_APU_DECODE=alt` does
//! the same here: threads of ours take the commands, one stream's in their
//! order and different streams' side by side, and a result is there to
//! dequeue once it and all before it are done (until then the title gets
//! "pending", which it polls for). `audio::meter` logs what either way
//! costs the title's thread.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::Layout;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use opus_pure::OpusDecoder;
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::Threading::{CreateEventW, SetEvent};

use super::audio::meter::{self, meter};
use super::audio::{Raise, decoder_raise, opus_workers, raise_our_thread, title_audio_thread};
use super::xasync::E_INVALIDARG;
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleModule, hle_export};

const E_NOTIMPL: HRESULT = 0x8000_4001_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;
const QUEUE_FULL: HRESULT = 0x8AC8_1000_u32 as i32;
/// No result yet, which titles poll for.
pub(crate) const PENDING_RESULTS: HRESULT = 0x8AC8_1001_u32 as i32;
const INVALID_COMMAND: HRESULT = 0x8AC8_1002_u32 as i32;
const INVALID_STREAM: HRESULT = 0x8AC8_1003_u32 as i32;
const ALREADY_ACTIVE: HRESULT = 0x8AC8_1005_u32 as i32;
const BAD_PACKET: HRESULT = 0x8AC8_1008_u32 as i32;

/// Processing kinds; only plain decoding is implemented.
const DECODE_OPUS: u32 = 1;

const ACTIVATE: u32 = 1;
const DEACTIVATE: u32 = 2;
const PROCESS: u32 = 3;
const NOTIFY: u32 = 4;
const RESET: u32 = 5;

/// Command option: signal the connection's event when the result is ready.
const SIGNAL: u32 = 1;
/// Connect options.
const DEINTERLEAVED_OUTPUT: u32 = 4;
const RESULT_EVENT: u32 = 8;

/// The longest Opus packet duration at 48 kHz (120 ms).
const MAX_FRAMES: usize = 5760;

#[repr(C)]
pub struct ConnectInput {
    process: u32,
    streams: u32,
    commands_per_stream: u32,
    memory: u32,
    frames_per_output: u32,
    options: u32,
}

#[repr(C)]
pub struct ConnectOutput {
    memory: *mut c_void,
    result_event: HANDLE,
}

/// What every command starts with, and what its result echoes.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct CommandId {
    kind: u32,
    stream: u32,
    sequence: u32,
    options: u32,
    context: u64,
}

#[repr(C, packed)]
struct Activate {
    id: CommandId,
    channels: u32,
    init_length: u32,
    init: *const c_void,
    start_pitch: f32,
    processing_length: u32,
    processing: *mut c_void,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Process {
    id: CommandId,
    pitch_ramp: f32,
    target_pitch: f32,
    first_frame: u32,
    frame_count: u32,
    input_length: u32,
    input: *const u8,
    output_length: u32,
    output: *mut u8,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct ApuResult {
    id: CommandId,
    state: u32,
    state_data: u64,
    usec: u64,
    error: HRESULT,
    frames: u32,
    output_length: u32,
    output: *mut u8,
    pitch: f32,
    input_length: u32,
    input: *const u8,
}

// SAFETY: the raw pointers are title addresses carried through, never
// dereferenced after the result is made.
unsafe impl Send for ApuResult {}

pub struct Connection {
    memory: *mut u8,
    layout: Layout,
    per_stream: u32,
    shared: Arc<Shared>,
    /// Our decoding threads; none: commands are done where they are queued.
    workers: Vec<std::thread::JoinHandle<()>>,
}

/// What the title's calls and our decoding threads share.
struct Shared {
    /// The result event's handle, or 0.
    event: usize,
    options: u32,
    state: Mutex<State>,
    /// Wakes a decoding thread: a command queued, a stream free again, or
    /// the connection closing.
    work: Condvar,
}

struct State {
    streams: Vec<Option<OpusDecoder>>,
    /// Results not yet dequeued, per stream.
    queued: Vec<u32>,
    /// One per command not yet dequeued, in the order of commands.
    results: VecDeque<Slot>,
    /// How many at the front are done: what the title can dequeue.
    ready: usize,
    commands: u32,
    dequeued: u32,
    /// Commands for our decoding threads, in the order queued.
    jobs: VecDeque<Job>,
    /// Streams a decoding thread is working on: their next command waits.
    busy: Vec<bool>,
    closing: bool,
}

struct Slot {
    result: ApuResult,
    done: bool,
    queued: Instant,
}

/// A command, copied when queued: the title's struct may be gone later.
enum Work {
    Activate { channels: u32 },
    Deactivate,
    Reset,
    Notify,
    Process(Process),
}

struct Job {
    /// The command's number, counted like `State::commands`.
    number: u32,
    stream: usize,
    work: Work,
}

// SAFETY: the raw pointers are the title's packet and output in the
// connection's memory, which the title leaves alone until it dequeues the
// result and which outlives the decoding threads (`XApuDisconnect` joins
// them before freeing it).
unsafe impl Send for Job {}

fn lock(shared: &Shared) -> std::sync::MutexGuard<'_, State> {
    shared.state.lock().unwrap_or_else(|e| e.into_inner())
}

fn signal(shared: &Shared) {
    if shared.event != 0 {
        // SAFETY: our event, open until the connection is dropped.
        unsafe { SetEvent(shared.event as HANDLE) };
    }
}

pub extern "system" fn XApuConnect(
    input: *const ConnectInput,
    output: *mut ConnectOutput,
    handle: *mut *mut Connection,
) -> HRESULT {
    connect(input, output, handle, opus_workers())
}

/// `workers`: how many threads of ours decode; 0: the calling thread does.
fn connect(
    input: *const ConnectInput,
    output: *mut ConnectOutput,
    handle: *mut *mut Connection,
    workers: usize,
) -> HRESULT {
    if input.is_null() || output.is_null() || handle.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    let input = unsafe { &*input };
    if !(1..=500).contains(&input.streams)
        || !(1..=16).contains(&input.commands_per_stream)
        || input.memory == 0
    {
        return E_INVALIDARG;
    }
    if input.process != DECODE_OPUS {
        // shortcut: rate conversion (decode-convert, PCM convert) is not
        // implemented; add a resampler if a title connects for it.
        tracing::warn!(
            process = input.process,
            "audio decoder: processing kind not implemented"
        );
        return E_NOTIMPL;
    }
    if input.frames_per_output != 0 {
        return E_INVALIDARG;
    }
    let Ok(layout) = Layout::from_size_align(input.memory as usize, 16) else {
        return E_INVALIDARG;
    };
    // SAFETY: non-zero size.
    let memory = unsafe { std::alloc::alloc_zeroed(layout) };
    if memory.is_null() {
        return E_OUTOFMEMORY;
    }
    let event = if input.options & RESULT_EVENT != 0 {
        // SAFETY: plain auto-reset event, no name.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        if event.is_null() {
            // SAFETY: allocated above with `layout`.
            unsafe { std::alloc::dealloc(memory, layout) };
            return E_OUTOFMEMORY;
        }
        event
    } else {
        std::ptr::null_mut()
    };
    let streams = input.streams as usize;
    let shared = Arc::new(Shared {
        event: event as usize,
        options: input.options,
        state: Mutex::new(State {
            streams: (0..streams).map(|_| None).collect(),
            queued: vec![0; streams],
            results: VecDeque::new(),
            ready: 0,
            commands: 0,
            dequeued: 0,
            jobs: VecDeque::new(),
            busy: vec![false; streams],
            closing: false,
        }),
        work: Condvar::new(),
    });
    meter::start();
    let workers: Vec<_> = (0..workers)
        .filter_map(|i| {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name(format!("victory Opus {i}"))
                .spawn(move || {
                    // In place of the title's mixer, which is time critical
                    // (or Pro Audio like it, with `VICTORY_AUDIO_MMCSS=1`).
                    let raise = match decoder_raise() {
                        Raise::ProAudio => Raise::ProAudio,
                        _ => Raise::TimeCritical,
                    };
                    raise_our_thread(raise, Raise::TimeCritical);
                    work(&shared)
                })
                .ok()
        })
        .collect();
    tracing::info!(
        streams,
        memory = input.memory,
        options = input.options,
        workers = workers.len(),
        "audio decoder connected (VICTORY_APU_DECODE=alt: threads of ours decode, \
         VICTORY_APU_WORKERS of them; workers=0: the title's thread decodes)"
    );
    let connection = Box::new(Connection {
        memory,
        layout,
        per_stream: input.commands_per_stream,
        shared,
        workers,
    });
    // SAFETY: checked non-null.
    unsafe {
        output.write(ConnectOutput {
            memory: memory.cast(),
            result_event: event,
        });
        handle.write(Box::into_raw(connection));
    }
    S_OK
}

pub extern "system" fn XApuDisconnect(handle: *mut Connection) -> HRESULT {
    if handle.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XApuConnect`; disconnected once.
    let mut connection = unsafe { Box::from_raw(handle) };
    // Our threads finish the command they are on and leave the rest.
    lock(&connection.shared).closing = true;
    connection.shared.work.notify_all();
    for worker in connection.workers.drain(..) {
        let _ = worker.join();
    }
    for _ in lock(&connection.shared).streams.iter().flatten() {
        meter().stream(false);
    }
    if connection.shared.event != 0 {
        // SAFETY: our event; the threads that signalled it are gone.
        unsafe { CloseHandle(connection.shared.event as HANDLE) };
    }
    // SAFETY: allocated in `XApuConnect` with this layout.
    unsafe { std::alloc::dealloc(connection.memory, connection.layout) };
    S_OK
}

pub extern "system" fn XApuEnqueueCommand(
    handle: *mut Connection,
    command: *const CommandId,
    total: *mut u32,
) -> HRESULT {
    if handle.is_null() || command.is_null() {
        return E_INVALIDARG;
    }
    let start = Instant::now();
    // SAFETY: a connection from `XApuConnect`.
    let connection = unsafe { &*handle };
    let hr = enqueue(connection, command, total, start);
    meter().caller(start);
    hr
}

fn enqueue(
    connection: &Connection,
    command: *const CommandId,
    total: *mut u32,
    now: Instant,
) -> HRESULT {
    let shared = &*connection.shared;
    // SAFETY: every command starts with an id.
    let id = unsafe { command.read_unaligned() };
    if id.kind == PROCESS {
        title_audio_thread();
    }
    let mut state = lock(shared);
    let stream = id.stream as usize;
    if stream >= state.streams.len() {
        return INVALID_STREAM;
    }
    if state.queued[stream] >= connection.per_stream {
        return QUEUE_FULL;
    }
    let work = match id.kind {
        // SAFETY: an activate command is this struct.
        ACTIVATE => Work::Activate {
            channels: unsafe { command.cast::<Activate>().read_unaligned() }.channels,
        },
        DEACTIVATE => Work::Deactivate,
        RESET => Work::Reset,
        NOTIFY => Work::Notify,
        // SAFETY: a process command is this struct.
        PROCESS => Work::Process(unsafe { command.cast::<Process>().read_unaligned() }),
        _ => return INVALID_COMMAND,
    };
    let mut result = ApuResult {
        id,
        state: 0,
        state_data: 0,
        usec: 0,
        error: S_OK,
        frames: 0,
        output_length: 0,
        output: std::ptr::null_mut(),
        pitch: 1.0,
        input_length: 0,
        input: std::ptr::null(),
    };
    let number = state.commands;
    state.queued[stream] += 1;
    state.commands = state.commands.wrapping_add(1);
    if !total.is_null() {
        // SAFETY: optional out parameter.
        unsafe { total.write(state.commands) };
    }
    if connection.workers.is_empty() {
        run(&mut state.streams[stream], shared.options, stream, &work, &mut result);
        state.results.push_back(Slot { result, done: true, queued: now });
        state.ready += 1;
        if id.options & SIGNAL != 0 {
            signal(shared);
        }
    } else {
        state.results.push_back(Slot { result, done: false, queued: now });
        state.jobs.push_back(Job { number, stream, work });
        drop(state);
        shared.work.notify_one();
    }
    S_OK
}

/// Does one command on its stream's decoder; `result.error` says how it went.
fn run(
    decoder: &mut Option<OpusDecoder>,
    options: u32,
    stream: usize,
    work: &Work,
    result: &mut ApuResult,
) {
    match work {
        Work::Activate { channels } => {
            tracing::debug!(stream, channels, "Opus stream activated");
            result.error = if decoder.is_some() {
                ALREADY_ACTIVE
            } else if !(1..=2).contains(channels) {
                E_INVALIDARG
            } else {
                match OpusDecoder::new(48_000, *channels as usize) {
                    Ok(d) => {
                        *decoder = Some(d);
                        meter().stream(true);
                        S_OK
                    }
                    Err(_) => E_INVALIDARG,
                }
            };
        }
        Work::Deactivate => {
            if decoder.take().is_some() {
                meter().stream(false);
            }
        }
        Work::Reset => {
            if let Some(d) = decoder {
                let _ = d.reset_state();
            }
        }
        Work::Notify => {}
        Work::Process(command) => match decoder {
            Some(d) => {
                let start = Instant::now();
                process(d, options, command, result);
                meter().decoded(start.elapsed());
            }
            None => result.error = INVALID_STREAM,
        },
    }
}

/// A decoding thread: takes the oldest command whose stream no other
/// thread is on, so one stream's commands run in their order.
fn work(shared: &Shared) {
    let mut state = lock(shared);
    loop {
        if state.closing {
            return;
        }
        let next = state.jobs.iter().position(|job| !state.busy[job.stream]);
        let Some(job) = next.and_then(|at| state.jobs.remove(at)) else {
            state = shared.work.wait(state).unwrap_or_else(|e| e.into_inner());
            continue;
        };
        state.busy[job.stream] = true;
        let mut decoder = state.streams[job.stream].take();
        // Results leave from the front only once done: this command's is
        // still there, as far from the front as commands were dequeued.
        let at = job.number.wrapping_sub(state.dequeued) as usize;
        let mut result = state.results[at].result;
        drop(state);
        run(&mut decoder, shared.options, job.stream, &job.work, &mut result);
        state = lock(shared);
        state.streams[job.stream] = decoder;
        state.busy[job.stream] = false;
        let at = job.number.wrapping_sub(state.dequeued) as usize;
        let slot = &mut state.results[at];
        slot.result = result;
        slot.done = true;
        meter().waited(slot.queued.elapsed());
        let mut signalled = false;
        while state.ready < state.results.len() && state.results[state.ready].done {
            let options = state.results[state.ready].result.id.options;
            signalled |= options & SIGNAL != 0;
            state.ready += 1;
        }
        if signalled {
            signal(shared);
        }
        // This stream's next command may have waited for this one.
        if !state.jobs.is_empty() {
            shared.work.notify_one();
        }
    }
}

/// Decodes one packet into the command's output buffer, keeping the frames
/// from `first_frame` on, at most `frame_count` of them.
fn process(decoder: &mut OpusDecoder, options: u32, command: &Process, result: &mut ApuResult) {
    let channels = decoder.channels();
    let (input, input_length) = (command.input, command.input_length);
    let (output, output_length) = (command.output, command.output_length);
    result.input = input;
    result.input_length = input_length;
    result.output = output;
    if input.is_null() || output.is_null() || input_length == 0 {
        result.error = E_INVALIDARG;
        return;
    }
    // SAFETY: the title's packet in its connection memory.
    let packet = unsafe { std::slice::from_raw_parts(input, input_length as usize) };
    let mut pcm = vec![0f32; MAX_FRAMES * channels];
    let decoded = match decoder.decode(packet, MAX_FRAMES, &mut pcm) {
        Ok(n) => n,
        Err(_) => {
            result.error = BAD_PACKET;
            return;
        }
    };
    let first = (command.first_frame as usize).min(decoded);
    let frames = (decoded - first)
        .min(command.frame_count as usize)
        .min(output_length as usize / (4 * channels));
    let kept = &pcm[first * channels..(first + frames) * channels];
    let bytes = frames * channels * 4;
    if bytes > 0 {
        // SAFETY: `output` is non-null and holds at least `bytes <= output_length` bytes;
        // writing raw bytes avoids requiring 4-byte alignment on the title's `*mut u8`.
        let out = unsafe { std::slice::from_raw_parts_mut(output, bytes) };
        if options & DEINTERLEAVED_OUTPUT != 0 && channels == 2 {
            let (left, right) = out.split_at_mut(frames * 4);
            let (left, _) = left.as_chunks_mut::<4>();
            let (right, _) = right.as_chunks_mut::<4>();
            for (i, frame) in kept.as_chunks::<2>().0.iter().enumerate() {
                left[i] = frame[0].to_le_bytes();
                right[i] = frame[1].to_le_bytes();
            }
        } else {
            for (dst, &sample) in out.as_chunks_mut::<4>().0.iter_mut().zip(kept) {
                *dst = sample.to_le_bytes();
            }
        }
    }
    result.frames = frames as u32;
    result.output_length = bytes as u32;
}

pub extern "system" fn XApuDequeueResult(handle: *mut Connection, out: *mut ApuResult) -> HRESULT {
    if handle.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a connection from `XApuConnect`.
    let mut state = lock(unsafe { &(*handle).shared });
    if state.ready == 0 {
        if !state.results.is_empty() {
            meter().not_ready();
        }
        return PENDING_RESULTS;
    }
    let Some(Slot { result, .. }) = state.results.pop_front() else {
        return PENDING_RESULTS;
    };
    state.ready -= 1;
    let stream = result.id.stream as usize;
    state.queued[stream] -= 1;
    state.dequeued = state.dequeued.wrapping_add(1);
    // SAFETY: the title passes room for one result.
    unsafe { out.write_unaligned(result) };
    S_OK
}

pub extern "system" fn XApuGetResultCount(handle: *mut Connection, total: *mut u32) -> u32 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: a connection from `XApuConnect`.
    let state = lock(unsafe { &(*handle).shared });
    if !total.is_null() {
        // SAFETY: optional out parameter.
        unsafe { total.write(state.dequeued) };
    }
    state.ready as u32
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "apu",
        dlls: &["xapu.dll"],
        exports: vec![
            hle_export!(XApuConnect),
            hle_export!(XApuDisconnect),
            hle_export!(XApuEnqueueCommand),
            hle_export!(XApuDequeueResult),
            hle_export!(XApuGetResultCount),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opus_pure::{Application, OpusEncoder};

    fn id(kind: u32, stream: u32) -> CommandId {
        CommandId {
            kind,
            stream,
            sequence: 7,
            options: 0,
            context: 0x1234,
        }
    }

    #[test]
    fn decodes_opus_as_the_title_drives_it() {
        assert_eq!(size_of::<ApuResult>(), 0x50);
        // 256 mono/stereo streams, two commands each, as the title connects.
        let input = ConnectInput {
            process: DECODE_OPUS,
            streams: 256,
            commands_per_stream: 2,
            memory: 0x10_4000,
            frames_per_output: 0,
            options: 1,
        };
        let mut output = ConnectOutput {
            memory: std::ptr::null_mut(),
            result_event: std::ptr::null_mut(),
        };
        let mut handle = std::ptr::null_mut();
        assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
        let base = output.memory.cast::<u8>();

        let activate = Activate {
            id: id(ACTIVATE, 3),
            channels: 2,
            init_length: 0,
            init: std::ptr::null(),
            start_pitch: 1.0,
            processing_length: 0,
            processing: std::ptr::null_mut(),
        };
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const activate).cast(), std::ptr::null_mut()),
            S_OK
        );

        // A 20 ms stereo packet of a 440 Hz tone.
        let tone: Vec<f32> = (0..960 * 2)
            .map(|i| ((i / 2) as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5)
            .collect();
        let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
        let mut packet = [0u8; 1500];
        let n = encoder.encode(&tone, 960, &mut packet).unwrap();
        // SAFETY: inside the connection memory.
        unsafe { std::ptr::copy_nonoverlapping(packet.as_ptr(), base, n) };
        let process = Process {
            id: id(PROCESS, 3),
            pitch_ramp: 0.0,
            target_pitch: 1.0,
            first_frame: 0,
            frame_count: 960,
            input_length: n as u32,
            input: base,
            output_length: 0xf00 * 2,
            output: base.wrapping_add(0x14000),
        };
        let mut total = 0;
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const process).cast(), &mut total),
            S_OK
        );
        assert_eq!(total, 2);
        // Two results queued for stream 3: its queue is full.
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const process).cast(), &mut total),
            QUEUE_FULL
        );
        assert_eq!(XApuGetResultCount(handle, std::ptr::null_mut()), 2);

        let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let first = unsafe { result.assume_init() };
        let (kind, error, context) = (first.id.kind, first.error, first.id.context);
        assert_eq!((kind, error, context), (ACTIVATE, S_OK, 0x1234));
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let decoded = unsafe { result.assume_init() };
        let (kind, error, frames, length) = (
            decoded.id.kind,
            decoded.error,
            decoded.frames,
            decoded.output_length,
        );
        assert_eq!((kind, error, frames, length), (PROCESS, S_OK, 960, 960 * 8));
        // SAFETY: the decoder wrote 960 stereo frames there.
        let pcm =
            unsafe { std::slice::from_raw_parts(base.wrapping_add(0x14000).cast::<f32>(), 1920) };
        let energy: f32 = pcm.iter().map(|s| s * s).sum::<f32>() / 1920.0;
        assert!(energy > 0.01, "decoded a tone, energy {energy}");
        assert_eq!(
            XApuDequeueResult(handle, result.as_mut_ptr()),
            PENDING_RESULTS
        );
        let mut dequeued = 0;
        assert_eq!(XApuGetResultCount(handle, &mut dequeued), 0);
        assert_eq!(dequeued, 2);

        // A process on a stream that was never activated fails in its result.
        let mut stray = process;
        stray.id.stream = 4;
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const stray).cast(), &mut total),
            S_OK
        );
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let error = unsafe { result.assume_init() }.error;
        assert_eq!(error, INVALID_STREAM);
        assert_eq!(XApuDisconnect(handle), S_OK);
    }

    #[test]
    fn decodes_into_unaligned_output_interleaved_and_deinterleaved() {
        let tone: Vec<f32> = (0..960 * 2)
            .map(|i| {
                let phase = (i / 2) as f32 * std::f32::consts::TAU / 48_000.0;
                if i % 2 == 0 {
                    (phase * 440.0).sin() * 0.5
                } else {
                    (phase * 880.0).sin() * 0.25
                }
            })
            .collect();
        let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
        let mut packet = [0u8; 1500];
        let n = encoder.encode(&tone, 960, &mut packet).unwrap();

        for (options, deinterleaved) in [(0u32, false), (DEINTERLEAVED_OUTPUT, true)] {
            let input = ConnectInput {
                process: DECODE_OPUS,
                streams: 4,
                commands_per_stream: 4,
                memory: 0x4_0000,
                frames_per_output: 0,
                options,
            };
            let mut output = ConnectOutput {
                memory: std::ptr::null_mut(),
                result_event: std::ptr::null_mut(),
            };
            let mut handle = std::ptr::null_mut();
            assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
            let base = output.memory.cast::<u8>();

            // Invalid channel counts fail cleanly without activating the stream.
            for bad_channels in [0u32, 3, u32::MAX] {
                let bad = Activate {
                    id: id(ACTIVATE, 0),
                    channels: bad_channels,
                    init_length: 0,
                    init: std::ptr::null(),
                    start_pitch: 1.0,
                    processing_length: 0,
                    processing: std::ptr::null_mut(),
                };
                assert_eq!(
                    XApuEnqueueCommand(handle, (&raw const bad).cast(), std::ptr::null_mut()),
                    S_OK
                );
                let mut res = std::mem::MaybeUninit::<ApuResult>::zeroed();
                assert_eq!(XApuDequeueResult(handle, res.as_mut_ptr()), S_OK);
                // SAFETY: written by `XApuDequeueResult`.
                let err = unsafe { res.assume_init() }.error;
                assert_eq!(err, E_INVALIDARG, "channels={bad_channels}");
            }

            // Activate two identical stereo decoders so aligned and unaligned outputs match bit-for-bit.
            for stream in [0u32, 1] {
                let activate = Activate {
                    id: id(ACTIVATE, stream),
                    channels: 2,
                    init_length: 0,
                    init: std::ptr::null(),
                    start_pitch: 1.0,
                    processing_length: 0,
                    processing: std::ptr::null_mut(),
                };
                assert_eq!(
                    XApuEnqueueCommand(handle, (&raw const activate).cast(), std::ptr::null_mut()),
                    S_OK
                );
                let mut res = std::mem::MaybeUninit::<ApuResult>::zeroed();
                assert_eq!(XApuDequeueResult(handle, res.as_mut_ptr()), S_OK);
                // SAFETY: written by `XApuDequeueResult`.
                let err = unsafe { res.assume_init() }.error;
                assert_eq!(err, S_OK);
            }

            // SAFETY: inside the connection memory.
            unsafe { std::ptr::copy_nonoverlapping(packet.as_ptr(), base, n) };
            let aligned_out = base.wrapping_add(0x10000);
            let unaligned_out = base.wrapping_add(0x20001); // 1 mod 4: misaligned for f32

            for (stream, out_ptr, frame_count) in [
                (0u32, aligned_out, 960u32),
                (1u32, unaligned_out, 960u32),
                (1u32, base.wrapping_add(0x30003), 0u32), // zero frames at misaligned ptr
            ] {
                let proc_cmd = Process {
                    id: id(PROCESS, stream),
                    pitch_ramp: 0.0,
                    target_pitch: 1.0,
                    first_frame: 0,
                    frame_count,
                    input_length: n as u32,
                    input: base,
                    output_length: 960 * 8,
                    output: out_ptr,
                };
                assert_eq!(
                    XApuEnqueueCommand(handle, (&raw const proc_cmd).cast(), std::ptr::null_mut()),
                    S_OK
                );
                let mut res = std::mem::MaybeUninit::<ApuResult>::zeroed();
                assert_eq!(XApuDequeueResult(handle, res.as_mut_ptr()), S_OK);
                // SAFETY: written by `XApuDequeueResult`.
                let decoded = unsafe { res.assume_init() };
                let (err, frames, len) = (decoded.error, decoded.frames, decoded.output_length);
                assert_eq!((err, frames, len), (S_OK, frame_count, frame_count * 8));
            }

            // SAFETY: both output regions are inside connection memory and hold 960 * 8 bytes.
            let (aligned_bytes, unaligned_bytes) = unsafe {
                (
                    std::slice::from_raw_parts(aligned_out, 960 * 8),
                    std::slice::from_raw_parts(unaligned_out, 960 * 8),
                )
            };
            assert_eq!(
                unaligned_bytes, aligned_bytes,
                "unaligned output must match aligned output (deinterleaved={deinterleaved})"
            );
            let samples: Vec<f32> = unaligned_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
            assert!(
                energy > 0.01,
                "expected decoded tone (deinterleaved={deinterleaved}), energy={energy}"
            );
            assert_eq!(XApuDisconnect(handle), S_OK);
        }
    }
    /// Connects as the title does, with `workers` decoding threads of ours.
    fn connected(
        streams: u32,
        per_stream: u32,
        options: u32,
        workers: usize,
    ) -> (*mut Connection, *mut u8, HANDLE) {
        let input = ConnectInput {
            process: DECODE_OPUS,
            streams,
            commands_per_stream: per_stream,
            memory: 0x10_0000,
            frames_per_output: 0,
            options,
        };
        let mut output = ConnectOutput {
            memory: std::ptr::null_mut(),
            result_event: std::ptr::null_mut(),
        };
        let mut handle = std::ptr::null_mut();
        assert_eq!(connect(&input, &mut output, &mut handle, workers), S_OK);
        // SAFETY: just made.
        assert_eq!(unsafe { &*handle }.workers.len(), workers);
        (handle, output.memory.cast(), output.result_event)
    }

    fn queue_activate(handle: *mut Connection, stream: u32, sequence: u32) -> HRESULT {
        let activate = Activate {
            id: CommandId { kind: ACTIVATE, stream, sequence, options: 0, context: 0 },
            channels: 2,
            init_length: 0,
            init: std::ptr::null(),
            start_pitch: 1.0,
            processing_length: 0,
            processing: std::ptr::null_mut(),
        };
        XApuEnqueueCommand(handle, (&raw const activate).cast(), std::ptr::null_mut())
    }

    fn queue_plain(handle: *mut Connection, kind: u32, stream: u32, sequence: u32) -> HRESULT {
        let id = CommandId { kind, stream, sequence, options: 0, context: 0 };
        XApuEnqueueCommand(handle, &raw const id, std::ptr::null_mut())
    }

    #[allow(clippy::too_many_arguments)]
    fn queue_process(
        handle: *mut Connection,
        stream: u32,
        sequence: u32,
        options: u32,
        input: *const u8,
        input_length: usize,
        output: *mut u8,
    ) -> HRESULT {
        let process = Process {
            id: CommandId { kind: PROCESS, stream, sequence, options, context: 0 },
            pitch_ramp: 0.0,
            target_pitch: 1.0,
            first_frame: 0,
            frame_count: 960,
            input_length: input_length as u32,
            input,
            output_length: 960 * 8,
            output,
        };
        XApuEnqueueCommand(handle, (&raw const process).cast(), std::ptr::null_mut())
    }

    /// The next result, polled for as a title does; our threads get 20 s.
    fn next_result(handle: *mut Connection) -> ApuResult {
        let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
        let start = Instant::now();
        loop {
            match XApuDequeueResult(handle, result.as_mut_ptr()) {
                S_OK => break,
                PENDING_RESULTS => {
                    assert!(start.elapsed().as_secs() < 20, "no result in 20 s");
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                other => panic!("dequeue failed: {other:#x}"),
            }
        }
        // SAFETY: written by `XApuDequeueResult`.
        unsafe { result.assume_init() }
    }

    /// Three consecutive 20 ms stereo packets of a tone.
    fn tone_packets() -> Vec<Vec<u8>> {
        let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
        (0..3)
            .map(|p| {
                let tone: Vec<f32> = (0..960 * 2)
                    .map(|i| {
                        let t = (p * 960 + i / 2) as f32 / 48_000.0;
                        (t * 440.0 * std::f32::consts::TAU).sin() * 0.5
                    })
                    .collect();
                let mut packet = [0u8; 1500];
                let n = encoder.encode(&tone, 960, &mut packet).unwrap();
                packet[..n].to_vec()
            })
            .collect()
    }

    #[test]
    fn our_threads_decode_what_the_calling_thread_would_in_command_order() {
        const STREAMS: u32 = 6;
        let packets = tone_packets();
        let mut decoded: Vec<Vec<u8>> = Vec::new();
        for workers in [0usize, 3] {
            let (handle, base, _) = connected(8, 4, 0, workers);
            let mut at = 0usize;
            let mut inputs = Vec::new();
            for packet in &packets {
                // SAFETY: inside the connection memory.
                unsafe { std::ptr::copy_nonoverlapping(packet.as_ptr(), base.add(at), packet.len()) };
                inputs.push((base.wrapping_add(at), packet.len()));
                at += 0x800;
            }
            let output = |stream: u32, packet: usize| {
                base.wrapping_add(0x1_0000 + (stream as usize * 3 + packet) * 0x2000)
            };
            // All of a stream's commands at once, stream after stream, as a
            // mixer block that refills many voices queues them.
            let mut sequence = 0;
            let mut expected = Vec::new();
            for stream in 0..STREAMS {
                assert_eq!(queue_activate(handle, stream, sequence), S_OK);
                expected.push((ACTIVATE, stream, sequence));
                sequence += 1;
                for (packet, &(input, length)) in inputs.iter().enumerate() {
                    let out = output(stream, packet);
                    assert_eq!(queue_process(handle, stream, sequence, 0, input, length, out), S_OK);
                    expected.push((PROCESS, stream, sequence));
                    sequence += 1;
                }
            }
            // Four results a stream are not dequeued yet: its queue is full.
            assert_eq!(queue_plain(handle, NOTIFY, 0, 99), QUEUE_FULL);
            for &(kind, stream, sequence) in &expected {
                let result = next_result(handle);
                let (id, error, frames) = (result.id, result.error, result.frames);
                assert_eq!((id.kind, id.stream, id.sequence), (kind, stream, sequence));
                assert_eq!(error, S_OK, "workers={workers}");
                assert_eq!(frames, if kind == PROCESS { 960 } else { 0 });
            }
            let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
            assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), PENDING_RESULTS);
            let mut dequeued = 0;
            assert_eq!(XApuGetResultCount(handle, &mut dequeued), 0);
            assert_eq!(dequeued, expected.len() as u32);
            // SAFETY: every output holds 960 stereo frames; nothing decodes now.
            let all = unsafe {
                std::slice::from_raw_parts(output(0, 0), STREAMS as usize * 3 * 0x2000).to_vec()
            };
            let energy: f32 = all[2 * 0x2000..][..960 * 8]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c).powi(2))
                .sum::<f32>()
                / 1920.0;
            assert!(energy > 0.01, "decoded a tone, energy {energy} (workers={workers})");
            decoded.push(all);
            assert_eq!(XApuDisconnect(handle), S_OK);
        }
        assert!(decoded[0] == decoded[1], "our threads decode the same samples");
    }

    #[test]
    fn our_threads_keep_one_streams_commands_in_order() {
        let packets = tone_packets();
        let (handle, base, _) = connected(4, 8, 0, 4);
        // SAFETY: inside the connection memory.
        unsafe { std::ptr::copy_nonoverlapping(packets[0].as_ptr(), base, packets[0].len()) };
        let out = base.wrapping_add(0x1_0000);
        let n = packets[0].len();
        // Each depends on the one before it having been done.
        assert_eq!(queue_process(handle, 2, 0, 0, base, n, out), S_OK);
        assert_eq!(queue_activate(handle, 2, 1), S_OK);
        assert_eq!(queue_activate(handle, 2, 2), S_OK);
        assert_eq!(queue_process(handle, 2, 3, 0, base, n, out), S_OK);
        assert_eq!(queue_plain(handle, RESET, 2, 4), S_OK);
        assert_eq!(queue_plain(handle, DEACTIVATE, 2, 5), S_OK);
        assert_eq!(queue_process(handle, 2, 6, 0, base, n, out), S_OK);
        assert_eq!(queue_plain(handle, 77, 2, 7), INVALID_COMMAND);
        assert_eq!(queue_plain(handle, NOTIFY, 9, 7), INVALID_STREAM);
        let errors: Vec<(u32, HRESULT, u32)> = (0..7)
            .map(|_| {
                let r = next_result(handle);
                let (id, error, frames) = (r.id, r.error, r.frames);
                (id.sequence, error, frames)
            })
            .collect();
        assert_eq!(
            errors,
            [
                (0, INVALID_STREAM, 0),
                (1, S_OK, 0),
                (2, ALREADY_ACTIVE, 0),
                (3, S_OK, 960),
                (4, S_OK, 0),
                (5, S_OK, 0),
                (6, INVALID_STREAM, 0),
            ]
        );
        assert_eq!(XApuDisconnect(handle), S_OK);
    }

    #[test]
    fn our_threads_signal_once_the_result_can_be_dequeued() {
        use windows_sys::Win32::System::Threading::WaitForSingleObject;
        let packets = tone_packets();
        let (handle, base, event) = connected(4, 4, RESULT_EVENT, 2);
        assert!(!event.is_null());
        // SAFETY: inside the connection memory.
        unsafe { std::ptr::copy_nonoverlapping(packets[0].as_ptr(), base, packets[0].len()) };
        let n = packets[0].len();
        assert_eq!(queue_activate(handle, 0, 0), S_OK);
        assert_eq!(queue_activate(handle, 1, 1), S_OK);
        // Stream 1's quick command asks for the signal behind stream 0's
        // packet: the signal comes when both can be dequeued.
        assert_eq!(queue_process(handle, 0, 2, 0, base, n, base.wrapping_add(0x1_0000)), S_OK);
        let reset = CommandId { kind: RESET, stream: 1, sequence: 3, options: SIGNAL, context: 0 };
        assert_eq!(XApuEnqueueCommand(handle, &raw const reset, std::ptr::null_mut()), S_OK);
        // SAFETY: the connection's event.
        assert_eq!(unsafe { WaitForSingleObject(event, 20_000) }, 0);
        assert_eq!(XApuGetResultCount(handle, std::ptr::null_mut()), 4);
        let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
        for sequence in 0..4 {
            assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
            // SAFETY: written above.
            let id = unsafe { result.assume_init() }.id;
            assert_eq!({ id.sequence }, sequence);
        }
        assert_eq!(XApuDisconnect(handle), S_OK);
    }

    #[test]
    fn disconnecting_with_commands_queued_stops_our_threads() {
        let packets = tone_packets();
        let (handle, base, _) = connected(64, 8, 0, 2);
        // SAFETY: inside the connection memory.
        unsafe { std::ptr::copy_nonoverlapping(packets[0].as_ptr(), base, packets[0].len()) };
        let n = packets[0].len();
        for stream in 0..64 {
            assert_eq!(queue_activate(handle, stream, 0), S_OK);
            for i in 0..7 {
                let out = base.wrapping_add(0x1_0000 + stream as usize * 0x2000);
                assert_eq!(queue_process(handle, stream, i, 0, base, n, out), S_OK);
            }
        }
        // Most are still queued; the memory is freed only after the threads
        // have left.
        assert_eq!(XApuDisconnect(handle), S_OK);
    }

    /// What decoding costs, for the mixer's 5.3 ms block: prints the time a
    /// 20 ms packet takes on this machine.
    /// `cargo test --release -p victory-hle --lib opus_packet_cost -- --ignored --nocapture`
    #[test]
    #[ignore = "a measurement, not a check"]
    fn opus_packet_cost() {
        for channels in [1usize, 2] {
            let mut encoder = OpusEncoder::new(48_000, channels, Application::Audio).unwrap();
            let packets: Vec<Vec<u8>> = (0..200)
                .map(|p| {
                    // A chord with some noise, so the bands carry something.
                    let mut seed = 12345u32 + p as u32;
                    let pcm: Vec<f32> = (0..960 * channels)
                        .map(|i| {
                            let t = (p * 960 + i / channels) as f32 / 48_000.0;
                            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            let noise = (seed >> 16) as f32 / 65_536.0 - 0.5;
                            (t * 220.0 * std::f32::consts::TAU).sin() * 0.3
                                + (t * 1870.0 * std::f32::consts::TAU).sin() * 0.2
                                + noise * 0.2
                        })
                        .collect();
                    let mut packet = [0u8; 1500];
                    let n = encoder.encode(&pcm, 960, &mut packet).unwrap();
                    packet[..n].to_vec()
                })
                .collect();
            let mut decoder = OpusDecoder::new(48_000, channels).unwrap();
            let mut pcm = vec![0f32; MAX_FRAMES * channels];
            let mut times: Vec<u128> = Vec::new();
            for _ in 0..5 {
                for packet in &packets {
                    let start = Instant::now();
                    decoder.decode(packet, MAX_FRAMES, &mut pcm).unwrap();
                    times.push(start.elapsed().as_nanos());
                }
            }
            times.sort_unstable();
            let bytes = packets.iter().map(Vec::len).sum::<usize>() / packets.len();
            println!(
                "Opus {channels} ch, 20 ms packets of about {bytes} bytes: median {:.0} us, \
                 95th {:.0} us, longest {:.0} us a packet",
                times[times.len() / 2] as f64 / 1e3,
                times[times.len() * 95 / 100] as f64 / 1e3,
                times[times.len() - 1] as f64 / 1e3,
            );
        }
    }
}
