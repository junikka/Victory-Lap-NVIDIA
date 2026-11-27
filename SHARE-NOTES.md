# About this copy

Source of play31 (2026-11-27: play29 + the F6 settings menu + a test fix; run with the play30 set): the VIctory D3D12 layer, base cool_fox, plus our ports and fixes. First shared as play13 on 2026-11-26.
Exported with `git archive` from commit efbcb2b8, without history.

- No game files, keys, saves or build output.
- Licence: GPL-3.0 (see LICENSE); keep it when sharing.
- Some upstream source files and docs name the title; they came from upstream as-is.
- Build: `cargo build --release -p victory --features diagnostics` (Rust 1.98.1, MSVC).
- Play-tuned switches: scripts/play-nvidia.ps1 (the measured play30 set).
- crates/xbox-bvh-builder: our Windows stand-in for the title's XboxBvhBuilder.dll (no title code); build it and place the DLL beside the title's executable yourself.
