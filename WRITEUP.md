# VIctory on an RTX 3080: what we did and where it stands

*Status: 27 November 2026, build "play30" (play29 source + one launcher switch). First published as play13 on 26 November. Measured on one PC: i9-12900F, RTX 3080 10 GB, 32 GB RAM, Windows 11, 1920x1080, Series S profile.*

## Is this our own build?

Partly. It is **not written from scratch**. The base is the open-source [VIctory](https://github.com/Marvisak/VIctory) layer (GPL-3.0) as carried by the cool_fox fork. On top of that base:

- **Ported:** fixes and features from about a dozen public forks, each one reviewed, tested and usually put behind a switch. The forks are VIctory upstream, avatarmanz, Belobug, mavis, trustno1, xeltax, jman, Vice-Tay, VI-BR-PORT and others.
- **Our own work:** about 70 commits. They cover NVIDIA-specific memory and residency tuning, the measurement tooling, the merges and fixes needed to make the ports work together, and the finding below that unlocked the frame rate.

Roughly 200 commits sit on top of the fox base, across play builds 3 to 13. Every play build passes the full workspace test suite (about 360 shader tests and 550 GPU tests, 0 failures) before it is played.

## Performance achieved

Same route in the open world, comparable load (6,000-9,500 draws a frame):

| | Before (play12, default play set) | **Now (play13, tuned set)** |
|---|---|---|
| Average frame rate | 3-4 fps | **13.1-13.4 fps** |
| Frame time p50 / p95 / p99 | 208 / 374 / 510 ms | **70-74 / 104-121 / 140-146 ms** |
| 1% low | 1.8 fps | **4.9-5.4 fps** |
| Cost per 1,000 draws | 36-42 ms | **10.9-11.6 ms (3.5x better)** |
| Card memory used | 5.5-6.0 GB of 9.3 | 5.1-6.3 GB of 9.3 |
| Lowest free PC memory in play | 3-4 GB | 2.5-2.8 GB |

The run with the panel on shows up to 18 fps at 5,000 draws. **It is playable but not smooth yet:** the goal is a stable 30 fps.

## Update, 27 November: play29 / play30

Measured on a fixed camera-spin route (480 s, scored from world entry + 150 s in 5 s windows), so the numbers are not the same route as the play13 table above:

| | play29 | **play30** |
|---|---|---|
| Average frame rate | 12.9-13.0 fps | **17.6-19.1 fps** |
| Frame time (median of 5 s windows) | 77-78 ms | **52-57 ms** |
| Draws a frame | 5,100-5,500 | 5,100-6,500 |
| Lowest free PC memory | 5.5 GB | 4.1-5.3 GB |
| Card memory | 75-77 % of 9.2 GB | 75-77 % |

- **play29** adds about 130 commits over play13: a memory set (card windows behind a 2 GB bar, mapped-memory trims, port guards and constants, dirty lists) and submit-side fixes, each turned on only after its own on/off runs on this PC (working set -1.4 GB, submitting thread -10 to -12 ms, slow windows 26 -> 2).
- **play30** is the same code with `VICTORY_BVH_LOG_ONCE=1`. Our Windows stand-in for the title's BVH-builder DLL wrote a log line on every call, about 110 a frame. Logging once took 21-24 ms off every frame in two on/off pairs. The stand-in is now in this repository as `crates/xbox-bvh-builder` (build it and place `XboxBvhBuilder.dll` beside the title's executable; see `scripts/play-nvidia.ps1`). Without `VICTORY_BVH_LOG_ONCE=1` it still logs every call, as it did when play13 was measured.
- **Where the frame goes now:** with no draws at all the title runs at its own 30 fps cap, so its CPU turn fits. What is left is our side: preparing the title's recorded draws (33-38 ms), the submitting thread (42-50 ms) and the card (43-49 ms). With the card's work removed the frame is still about 46 ms.
- **Tried tonight and not kept:** moving heap-record upkeep after the submission's signal (the wait fell a third, not half), a per-thread lookup cache (worse in its first pair), 14 recording threads (less free memory and a 1.8 s stall).

## The technical approach

1. **Measure everything, at equal load.** Every session records:
   - a per-5-second frame log (preparing, recording, card time, copies up and back);
   - card memory, PC memory, paging, per-thread CPU and nvidia-smi;
   - the draws ranked by cost, failed and dropped work, and screenshots.

   Runs are compared per 1,000 draws, because comparing raw fps between scenes misled us more than once.
2. **Find the wall, not the symptom.** The card looked "100 % busy" at low power. The real cause was one of our own correctness fixes (`VICTORY_TWIN_BYTES`, made for stretched triangles): it created 300-750 host buffers a frame under a single global lock, so the title's 8 recording threads ran one at a time. That cost about 300 of the 345 ms in a frame. Turning it off took the frame from 345 ms to 74 ms, with no stretched triangles seen in 17 world shots. A lock-free version of that fix is in progress.
3. **Port before inventing.** Upstream and fork fixes are adopted first. Each one is:
   - reviewed;
   - kept behind a switch that is off by default unless it is a pure fix;
   - checked by a script that fails a merge if it changes any existing default.
4. **Desktop memory, not console memory.** The console has one memory pool shared by CPU and GPU. A PC has 32 GB of RAM and a 10 GB card. Today the layer still mirrors much of the title's memory: the card reads about 6 GB of PC memory over the bus while 3-4 GB of its own sits free. The roadmap's main track:
   - classify the title's memory by use;
   - keep GPU-only data on the card only;
   - stop copying back what the CPU never reads;
   - keep one copy instead of two.
5. **Pre-translate and pre-build.** Shaders are translated ahead of play (about 25,800 in the cache) and about 83,700 pipelines are built from a pipeline book before the session, so turning the camera does not compile.
6. **Off-machine builds.** Compiles and the CPU test suites run on a separate 28-core build VM. Only GPU tests and game runs use the gaming PC.

## What limits it now (next work)

- **The submitting thread** works about 64 ms a frame; the card is done in about 53 ms and waits. Preparing doubled to 50 ms once the 8 recording threads ran in parallel and started contending on a guard lock.
- **Memory churn:** two eviction rules drop and remake textures and card windows: 35 % of what is dropped is remade, and 10.9 GB of windows are remade in 10 minutes.
- **Copy-back:** about 26 MB a frame is fetched back from the card; only about 9 MB of it changes title memory.
- **Visual:** the minimap's map goes flat at two or more wanted stars. Ray tracing is left out on purpose.

## How to run it

You need your own legally obtained copy of the title. **No game files are included or provided.** Build with:

```
cargo build --release -p victory --features diagnostics
```

Then use `scripts/play-nvidia.ps1`, which sets the tuned switch set measured above. Set `VICTORY_GAME_DIR` to your own files first.
