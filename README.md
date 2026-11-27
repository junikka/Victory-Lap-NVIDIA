# VIctory on an RTX 3080 (NVIDIA)

**Current state (27 Nov 2026, play31):** on an RTX 3080 10 GB, i9-12900F and 32 GB RAM at 1080p (Series S profile), our camera-spin test route now runs at 17.6-19.1 fps (52-57 ms a frame), up from about 13 fps, mostly because a debug log in our BVH-builder stand-in fired about 110 times a frame and now logs once. With no draws the title already runs at its own 30 fps cap, so what remains is our submitting thread (its own preparation, flat in draws) and the card, both measured in [docs/fps-findings-rtx3080.md](docs/fps-findings-rtx3080.md) with what we tried and did not keep. This build also adds an in-game settings menu (F6) and a compact HUD (F3), and every test suite now passes with the play settings.

- **What this is:** the open-source VIctory layer (GPL-3.0), on the cool_fox fork, with about 200 commits on top: ported fixes from about a dozen public forks plus our own NVIDIA memory and residency work and measurement tooling.
- **The approach, the numbers and what limits it now:** [WRITEUP.md](WRITEUP.md).
- **Run it:** `cargo build --release -p victory --features diagnostics`, set `VICTORY_GAME_DIR` to your own copy of the title (no game files are included), then run `scripts\play-nvidia.ps1`.

**The goal:** a steady 30 fps on the Series S profile at 1080p on this RTX 3080 10 GB PC, keeping RAM out of pressure (memory first), then full texture detail; ray tracing later as a setting.

**This repository contains no game files, code or assets.** You need your own
copy of the game. It is not affiliated with or endorsed by the game's developer
or publisher.

What is known to be wrong:

- Trees, signs and far objects flicker in the distance in some places.
- Lighting is too bright or too dark after walking indoors, until the title has
  run for a while; rooms with little direct light are too dark (ray tracing is off).
- Textures of what just came into view appear late while the title streams.
- The first run of a new build stutters while the driver makes pipelines
  (`victory warm` makes them ahead).
- Recording a frame in the world makes the title's own watchdog end it.


## How to run it

You need Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with shader
model 6.6 and mesh shaders (tested on an RTX 3080 10 GB), Rust (the toolchain
is pinned), the MSVC build tools, and your own copy of the title.

```powershell
cargo build --release -p victory --features diagnostics
$env:VICTORY_GAME_DIR = '<folder of your own copy>'
scripts\play-nvidia.ps1
```

`scripts\play-nvidia.ps1` starts the title with the settings this build is
measured with (Series S profile, 1080p). It also says how to build and place
the BVH-builder stand-in (`crates/xbox-bvh-builder`). F6 opens the settings
menu in game, F3 cycles the HUD.

Commands and debugging switches: [docs/README-fxceless.md](docs/README-fxceless.md).
Every `VICTORY_...` switch is described in a comment where it is read.

## What is different here

In short; the measurements are in [WRITEUP.md](WRITEUP.md) and [docs/fps-findings-rtx3080.md](docs/fps-findings-rtx3080.md).

- **List calls are written down, not carried out, on the title's threads**
  (`VICTORY_THIN`): eight threads of ours translate them.
- **Submissions are prepared by a thread of ours and joined**
  (`VICTORY_ASYNC_SUBMIT`, `VICTORY_JOIN_PARTS`), with what can be spread over
  threads spread: refreshes, compares (AVX2), descriptor records, landings.
- **What shaders wrote comes back beside the write guards**, by the dwords that
  changed, on a thread of its own (`VICTORY_RELAY`, `VICTORY_LAND`).
- **Buffer fills that need bytes still on the card are done on the card**
  (`VICTORY_CARD_STEPS`) instead of waiting for it.
- **The title's writes are caught more cheaply**: one view opened at a fault,
  blocks that rest, places in the title's code that look before they write.
- **Textures go up on a copy queue of their own** and a slot shows one only after
  its upload ran, which also ended a device loss that came every 15 minutes or so.
- **Boats float**: textures that shaders write and the title's CPU reads go back
  to title memory (from fxceless and Geckomacabre, on by default here).
- **Frame record and replay** with per-pass times and shader variants compared
  turn by turn ([docs/replay.md](docs/replay.md)).

## Credits

This is a tree of other people's work with more on top. The history is kept, and
commits taken from elsewhere name the commit and its author.

- **[VIctory](https://github.com/Marvisak/VIctory)** by Marvisak and its
  contributors: the loader, the runtime services, the Direct3D 12 back end and
  the shader recompiler.
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)**: the fork
  this one started from, and the people whose pull requests it carries.
- **[DexNom/VIctory](https://github.com/DexNom/VIctory)**: ray tracing fixes,
  cube gradients, sparse textures.
- **[Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD)**:
  the ambient light idea, pipeline workers, and work by romainhedouin.
- **[FQY55kai/ZJOU-VICTORY](https://github.com/FQY55kai/ZJOU-VICTORY)**.
- **[myokex/victory-amd](https://github.com/myokex/victory-amd)**: the guard over
  every view of a page (with Spexivus, joroyt and tightfall), and Joshua Odom's
  reads into title memory.
- **Geckomacabre** and **cxy0072019-sketch**: the texture read-back.
- **[XCL](https://github.com/ivall/xcl)** (GPL-2.0-only) was read for how the same
  title behaves under another layer; no code of it is here.
- This fork's changes were made with [Claude Code](https://claude.com/claude-code).

## License

GPL-3.0-or-later, as VIctory. See [LICENSE](LICENSE); the one file that is not
covered is named in [docs/asset-policy.md](docs/asset-policy.md).
