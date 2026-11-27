# About this copy

Source of play29 (2026-11-27; run as play30 with VICTORY_BVH_LOG_ONCE=1): the VIctory D3D12 layer, base cool_fox, plus our ports and fixes. First shared as play13 on 2026-11-26.
Exported with `git archive` from commit f321192d, without history.

- No game files, keys, saves or build output.
- Licence: GPL-3.0 (see LICENSE); keep it when sharing.
- Some upstream source files and docs name the title; they came from upstream as-is.
- Build: `cargo build --release -p victory --features diagnostics` (Rust 1.98.1, MSVC).
- Play-tuned switches: see the launchers in the victory-lap repo (not included here).
