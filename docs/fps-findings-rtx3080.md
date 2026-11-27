# fps findings, 2026-11-27 (spot-spin, RTX 3080 10 GB, i9-12900F, 32 GB, Series S profile, 1080p)

Base: play30 = play29 exe + VICTORY_BVH_LOG_ONCE=1 (launcher scripts/play-nvidia.ps1).
Spin steady (world + 150 s, 5 s windows): 16.5-19.1 fps, 52-60 ms a frame at 5,100-6,500 draws; free RAM low 3.5-5.3 GB; card memory 75-77 % of a 9.2 GB budget. Base runs alone spread 53-60 ms (mapping-stall spells and draw count).

## Measured walls (5 s lines, p50 a frame; corrected 11:47 after an independent review)
These are three stages side by side, not parts of one sum.
- The submitting thread worked 42-50 ms. Inside it, "preparing" (33-38 ms) is that thread's own preparation (SUBMISSION_US[0] in host.rs), not the title's recording threads; it is flat in draws. Within it: asking the write watch ~5.5 ms, the lists' steps ~20 ms (about 8,450 refreshes a frame, ~84 % copying nothing).
- The card running our lists: 43-49 ms.
- Waited for the card: 1-10 ms.
- Mapping-stall spells: ~25 % of steady 5 s windows (1,000-1,700 map calls of 1 ms+ in a 5 s line against 5-30 when clear), ~7 ms on the mean frame; trigger not found yet. Single A/B pairs mislead unless these windows are scored out.
- Fit after 186 (clear windows, seven bases): frame = 33.0 ms + 4.10 ms per 1,000 draws; card = 24.0 + 3.84; submitting thread = 42.4 + 0.88. Below ~4,500 draws the frame is the submitting thread; above ~5,500 it is card time plus ~10 ms of card idle. 30 fps at 5,400 draws needs about -20 ms on both.

## Settled
- 186 KEPT: our BVH stand-in DLL logged every prebuild call (~110 a frame); logging once: -21 to -24 ms, 77 -> 54-57 ms.
- No-draws with 186 on: the title runs at its own 30 fps cap (33.3 ms; preparing 2, submit 5 ms). The title's own CPU turn fits 30 fps; what remains is our submitting thread and the card (~21 ms over at 5,000+ draws).
- E1 (card does nothing, 186 on): 46.0 ms at 4,927 draws vs 54.2 at 5,354: the card is worth ~6-8 ms; without it the frame is still 46 ms, set by the submitting thread (41, of which preparing 37).
- 188 REJECTED: heap records' maintenance after the done signal; the submission's heap-record wait fell 24-34 %, never half; frame mixed. The remaining ~3 ms is the marked part itself.
- 187 REJECTED (per-thread buffer-span + staging caches): recording time did not fall in either on-run (span hits 78-80 %); pair 1's +5.4 ms was a mapping-stall spell.
- 216 (10-19 GB NVIDIA card-memory profile: CARD_RESERVE_BUDGETS=1, CARD_BUFFERS_MB=1536, CARD_WINDOWS_MB=1024): pair 1 put +0.65 GB on the card, PC working set unchanged, free RAM low -0.2 GB, frame not decidable (mapping-stall spells). Pair 2: same. REJECTED (no PC memory gain).
- 182 PROMISING: constants copied to card memory (card -10 to -11 ms, frame held).
- 183 REJECTED: 14 thin threads (free RAM low -2.3 GB, 1.8 s stall).
- 176 NOT promising: page sections (the gain was the constants copy).
- sRGB-twin count (a ticket build's counter, not in this source): 0 views needing the sRGB twin along spin.
- Heap pass line: ~6-7 ms looking at slots, ~3 copies' spans, ~5-6 textures a frame.

## Wanted from other forks
Changes that cut, at 5,000+ draws a frame on a 10 GB NVIDIA card: (a) the submitting thread's per-frame work (its flat ~35 ms of preparing: write-watch asking, list steps, heap records), (b) per-draw CPU cost on the recording threads, (c) card time for our lists, (d) PC RAM by moving title memory to the card without raising stalls.
