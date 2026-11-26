# play-nvidia.ps1: run VIctory with the switch set measured on an RTX 3080 10 GB / 32 GB RAM (WRITEUP.md).
# Set VICTORY_GAME_DIR to your own copy of the title first. No game files are included.
param([string]$Exe = "$PSScriptRoot\..\target\release\victory.exe")
if (-not $env:VICTORY_GAME_DIR) { throw 'Set VICTORY_GAME_DIR to the folder with your own copy of the title.' }
$env:XCL_NETWORK = '0'
$set = @(
    # Play tuning
    'VICTORY_PIPELINE_WAIT_MS=24', 'VICTORY_LATE_WAIT=3', 'VICTORY_WIDE_BACK=1',
    'VICTORY_CARD_BUFFERS_MIN_MB=1', 'VICTORY_CARD_BUFFERS_MB=1024', 'VICTORY_RT_MISS=1', 'VICTORY_RT_MISS_HIT=0.5',
    # The measured play13 set (VICTORY_TWIN_BYTES deliberately off: it serialised recording)
    'VICTORY_AMBIENT_REFLECT=0.5', 'VICTORY_AMBIENT_SKY=on', 'VICTORY_POOL_SMALL=1', 'VICTORY_RECHECK_LONG=1',
    'VICTORY_AMBIENT_AIR=1.5', 'VICTORY_TEXTURE_NO_WAIT=1', 'VICTORY_RELAY=2', 'VICTORY_CARD_REHASH_RUNS=1',
    'VICTORY_JOIN_COPIES=1', 'VICTORY_INDIRECT_WINDOW=1', 'VICTORY_RELAY_COPY=1', 'VICTORY_CARD_STEPS=2',
    'VICTORY_INDIRECT_BATCH=2', 'VICTORY_MAP_THREADS=1', 'VICTORY_CARD_AWAKE=1', 'VICTORY_SHOW_AFTER=1',
    'VICTORY_AMBIENT=0.8', 'VICTORY_THIN=1', 'VICTORY_JOIN_PARTS=8', 'VICTORY_UPLOAD_AHEAD=1',
    'VICTORY_CARD_BUFFERS_MIN_KB=64', 'VICTORY_RESOLUTION=1920x1080', 'VICTORY_TEXTURE_DROP_MIPS=1',
    'VICTORY_INDIRECT_WIDE=1', 'VICTORY_PIPELINE_BOOK=1',
    # On-screen performance panel (top left); remove to hide it
    'VICTORY_PANEL=1'
)
foreach ($kv in $set) { $k, $v = $kv.Split('=', 2); Set-Item "env:$k" $v }
& $Exe --console series-s
