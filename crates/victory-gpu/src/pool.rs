//! Host memory for the textures the title fills, and for their staging
//! buffers, cut from blocks made once.
//!
//! Each such texture had an allocation of its own and its staging buffer
//! another. On an RX 7900 XTX an allocation takes about 310 microseconds
//! whatever its size, on one thread (`texture::allocation_costs`), and a
//! texture cut from a block 6: the pass that makes the textures shaders
//! pick by index (`draw.rs` `heap_textures`) got through 350-500 of the
//! 24,000 queued when the world loads in each of its 250 ms turns, some
//! forty seconds of a world without its textures.
//! `VICTORY_TEXTURE_POOL=0` makes every one an allocation of its own again.
//!
//! A piece given back is used again only two seconds later (nothing the card
//! was told before that is still running: Windows resets a card that takes
//! that long over one submission), and by then no list names it (the owners' rule, as for allocations of their own): a
//! texture's piece comes back when the last reference to the host texture
//! goes (an object of ours the texture holds says so), a staging buffer's
//! when the last list that recorded its copy is reset.

use std::collections::{BTreeMap, VecDeque};
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{GUID, HRESULT, IUnknown, Interface};

/// Bytes of a block; pieces of more than half of one are not pooled (with
/// blocks made ahead, [`ahead`]: of more than one).
const BLOCK: u64 = 64 << 20;
/// Blocks with nothing in them that are kept.
const SPARE: usize = 2;
/// The same while the card has room ([`room`]): 4 GiB of them.
const SPARE_KEPT: usize = 64;
/// The same of blocks of staging buffers, which are the PC's memory: run
/// crash-1 held 5.2 GB of them.
const STAGING_KEPT: usize = 8;
/// How many are kept now.
static SPARES: AtomicUsize = AtomicUsize::new(SPARE);
/// Pieces of this many bytes and less lie in blocks of their own while the
/// pool is packed ([`pack`]): a small texture in use then does not keep a
/// block of large ones from emptying.
const SMALL_PIECE: u64 = 64 << 10;
/// Milliseconds for which nothing new is placed in a block chosen to empty
/// ([`Pool::empty_out`]): its textures go once the lists that ran have
/// finished (`crate::host::retire`), and their pieces rest ([`REST`]).
const DRAIN_MS: u64 = 30_000;
/// Blocks chosen to empty at one look (once a second), at most.
const MOST_A_LOOK: u64 = 8;

static PACK: crate::priority::Switch = crate::priority::Switch::new(
    "VICTORY_POOL_PACK",
    "the texture pool is packed: a piece goes to the fullest block of its kind, and over its budget the card gets whole blocks back",
    crate::priority::on_off,
);

/// Whether the texture pool is packed now (`VICTORY_POOL_PACK=1`, `alt`: in
/// turn; ticket 075). Session 0152 held 84 blocks, 5.4 GB, for 2.6-3.1 GB
/// of textures and gave none back while 810,239 textures were let go for
/// the budget: a piece went to the first block with room, and the sweep
/// let go the longest idle whatever block it lay in. Packed,
/// - a piece goes to the fullest block with room for it, small pieces
///   ([`SMALL_PIECE`]) and large ones each to blocks of their own;
/// - while the card is over its line as the system counts it
///   ([`over_line`]), the blocks whose every texture is idle are emptied,
///   those with the fewest bytes first, and given back whatever the spare
///   blocks (`crate::resource::let_go_idle` lets their textures go; their
///   next use makes them again, in a fuller block);
/// - the room rule (`crate::resource::room`) then counts the blocks whole.
pub(crate) fn pack() -> bool {
    PACK.now()
}

/// Whether the card's memory in use is at 95 % of its budget or over, both
/// as the system counts them.
pub(crate) fn over_line(budget: u64, used: u64) -> bool {
    budget > 0 && u128::from(used) * 100 >= u128::from(budget) * 95
}

/// The bytes of blocks to give back: over the line ([`over_line`]) what
/// brings the card down to 90 % of its budget; one block's when the system
/// has just refused one (`refused`); else none.
fn to_return(memory: Option<[u64; 4]>, refused: bool) -> u64 {
    match memory {
        Some([budget, used, ..]) if over_line(budget, used) => used.saturating_sub(budget / 100 * 90),
        _ if refused => BLOCK,
        _ => 0,
    }
}

/// Sets how many empty blocks stay, by the card's memory (budget and use):
/// [`SPARE_KEPT`] below 85 % ([`crate::host::keep_freed`]), else [`SPARE`].
/// Asked once a second.
pub(crate) fn room(memory: Option<[u64; 4]>) {
    if !crate::residency::on() {
        SPARES.store(spares(memory, crate::host::keep_freed()), Ordering::Relaxed);
        return;
    }
    // (`VICTORY_RESIDENCY=1`, codec-d/cool_fox c64fc9b4/e94d277a: fewer
    // spares while the system is short of memory, and idle blocks let go now,
    // not at the pool's next allocation.)
    let pressure = crate::residency::pressured();
    let limit = spares(memory, crate::host::keep_freed() && !pressure);
    SPARES.store(limit, Ordering::Relaxed);
    let tick = now();
    // A full pool can become idle while another subsystem needs the VRAM.
    // A busy allocator observes the new limit on its next allocation or tick.
    if let Ok(mut pool) = TEXTURES.try_lock() {
        pool.reclaim(tick, limit, true);
    }
    if let Ok(mut pool) = STAGING.try_lock() {
        let staging_limit = if crate::residency::critical_commit() {
            0
        } else if pressure {
            // (Ticket 124: all of them, before any texture goes.)
            usize::from(!crate::residency::pressure_keep())
        } else {
            limit
        };
        pool.reclaim(tick, staging_limit, true);
    }
}

fn spares(memory: Option<[u64; 4]>, keep: bool) -> usize {
    match memory {
        Some([budget, used, ..]) if keep && u128::from(used) * 100 < u128::from(budget) * 85 => SPARE_KEPT,
        _ => SPARE,
    }
}

/// Milliseconds until a piece given back is used again. (It was three
/// frames, a tenth of a second while the title shows a loading screen at 30
/// a second, with up to 18 submissions in the air.)
const REST: u64 = 2500;

/// How long a piece of staging memory rests: `VICTORY_STAGING_REST_MS`
/// (Op-lite's memory audit, item 1; unset: [`REST`], as for textures). A
/// staging piece is given back only after the list that copies from it
/// has run and was reset, so it needs no rest against a device loss as a
/// texture's piece does; while it rests its block cannot empty, and the
/// staging blocks follow the uploads of the last 2.5 s (theirs: 0.7 to
/// 3.4 GB of blocks for 60 to 240 MB of live staging; ours 320 MB for 50).
/// Try 250.
fn staging_rest() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        let ms = rest_ms(std::env::var("VICTORY_STAGING_REST_MS").ok().as_deref());
        if ms != REST {
            tracing::info!(ms, "staging pieces rest {ms} ms before they are used again (VICTORY_STAGING_REST_MS; usually {REST})");
        }
        ms
    })
}

/// What `VICTORY_STAGING_REST_MS` says: milliseconds, [`REST`] if it is
/// unset or no number.
fn rest_ms(value: Option<&str>) -> u64 {
    value.and_then(|v| v.trim().parse().ok()).unwrap_or(REST)
}

fn now() -> u64 {
    crate::host::now_ms()
}

/// `VICTORY_TEXTURE_POOL` as (textures cut from blocks in the card's
/// memory, staging cut from blocks of upload memory): unset or anything
/// else both, as before; `0` neither; `staging` (after
/// Voltexs/VIctory-RTXNvidia 295fd537) the staging only: every texture is
/// a resource of its own in the card's memory, so no 64 MiB block is held
/// by a few small textures in use (session 0152: 84 blocks, 5.4 GB, for
/// 2.6-3.1 GB of textures), while uploads still come from pooled staging.
fn pool_value(value: Option<&str>) -> (bool, bool) {
    match value {
        Some("0") => (false, false),
        Some(v) if v.eq_ignore_ascii_case("staging") => (false, true),
        _ => (true, true),
    }
}

fn pool_setting() -> (bool, bool) {
    static ON: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pool_value(std::env::var("VICTORY_TEXTURE_POOL").ok().as_deref()))
}

/// Whether textures are cut from blocks in the card's memory
/// ([`pool_value`]).
pub(crate) fn on() -> bool {
    pool_setting().0
}

/// Whether staging memory for uploads is cut from blocks ([`pool_value`]).
fn staging_on() -> bool {
    pool_setting().1
}

/// How many blocks of textures are made ahead, so that one has been there
/// for a while when its first piece is taken (`VICTORY_RIPE_BLOCKS`, 12;
/// 0: none, and new blocks are used at once, as before).
///
/// A page fault of lists already on the card, in a texture made the same
/// tenth of a second as the first piece of a block made for it, lost the
/// device in runs n08 and n11 (2026-11-25): the title gives a slot that
/// those lists read by index a new texture. Why the card does not have the
/// new block's addresses yet is not known (n11 waited for `MakeResident`;
/// `indirect::tests::new_blocks_under_readers` never loses it). A block that
/// every submission begun before its making has left behind ([`Ripe`]) has
/// no such lists: pieces are cut from those first.
fn ahead() -> usize {
    static AHEAD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *AHEAD.get_or_init(|| std::env::var("VICTORY_RIPE_BLOCKS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(12))
}

/// Submissions begun and finished ([`crate::host::submissions`]), and how
/// many blocks to make ahead (0: every block is ripe at once).
#[derive(Clone, Copy)]
struct Ripe {
    started: u64,
    finished: u64,
    ahead: usize,
}

impl Ripe {
    const ALL: Ripe = Ripe { started: 0, finished: 0, ahead: 0 };

    fn now() -> Ripe {
        let (started, finished) = crate::host::submissions();
        // Not more than are kept ([`room`]): they would go and be made again.
        Ripe { started, finished, ahead: ahead().min(SPARES.load(Ordering::Relaxed)) }
    }

    fn is<B>(self, block: &Block<B>) -> bool {
        self.ahead == 0 || block.made <= self.finished
    }
}

/// Pieces cut from blocks that were not ripe, for want of a ripe one.
static UNRIPE: AtomicU64 = AtomicU64::new(0);

/// The free ranges of a block: start and length, never two that touch;
/// and their bytes.
struct Ranges(BTreeMap<u64, u64>, u64);

impl Ranges {
    fn new(size: u64) -> Ranges {
        Ranges(BTreeMap::from([(0, size)]), size)
    }

    /// The lowest place for `len` bytes at a multiple of `align`, and the
    /// free range it lies in (start and length).
    fn place(&self, len: u64, align: u64) -> Option<(u64, u64, u64)> {
        self.0.iter().find_map(|(&start, &size)| {
            let at = start.next_multiple_of(align);
            (at + len <= start + size).then_some((start, size, at))
        })
    }

    /// Takes that place.
    fn take(&mut self, len: u64, align: u64) -> Option<u64> {
        let (start, size, at) = self.place(len, align)?;
        self.0.remove(&start);
        if at > start {
            self.0.insert(start, at - start);
        }
        if at + len < start + size {
            self.0.insert(at + len, start + size - at - len);
        }
        self.1 -= len;
        Some(at)
    }

    fn give(&mut self, mut start: u64, mut len: u64) {
        self.1 += len;
        if let Some((&before, &size)) = self.0.range(..start).next_back()
            && before + size == start
        {
            self.0.remove(&before);
            (start, len) = (before, len + size);
        }
        if let Some(size) = self.0.remove(&(start + len)) {
            len += size;
        }
        self.0.insert(start, len);
    }

    fn whole(&self, size: u64) -> bool {
        self.0.get(&0) == Some(&size)
    }

    fn free_bytes(&self) -> u64 {
        self.1
    }

    fn largest(&self) -> u64 {
        self.0.values().copied().max().unwrap_or(0)
    }
}

/// A part of a block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Piece {
    block: usize,
    start: u64,
    len: u64,
    /// The texture's own size, when this piece holds one. 0 for a staging
    /// piece, and until [`Pool::note_texture`] counts it. Subtracted when
    /// the piece is given back.
    real: u64,
}

struct Block<B> {
    memory: B,
    free: Ranges,
    /// The end of the highest piece ever taken: above it the memory is as
    /// it was made (zeros).
    used_to: u64,
    /// The newest submission begun when it was made ([`Ripe`]).
    made: u64,
    /// When that was (ms).
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    since: u64,
    /// The look (`crate::priority::ticks`) at which it last showed a sign
    /// of use: it was made, a piece of it was taken, or one of its textures
    /// was found in use ([`Pool::told`]).
    seen: u64,
    /// The residency priority Windows was last told for it (normal, as
    /// every heap begins).
    told: u32,
    /// Whether its pieces are small ones ([`SMALL_PIECE`]): what the first
    /// piece taken from it, empty, was.
    small: bool,
    /// Until when (ms) nothing new is placed in it, as it was chosen to
    /// empty ([`Pool::empty_out`]); 0: never chosen.
    drain: u64,
}

struct Pool<B> {
    /// `None`: a block let go, its place to be used again.
    blocks: Vec<Option<Block<B>>>,
    /// Pieces given back, and when each may be used again.
    resting: VecDeque<(u64, Piece)>,
    /// The empty blocks it was last told to keep, and the most it keeps.
    spare: usize,
    most: usize,
    /// No new block is asked for before this time after one was refused
    /// ([`Pool::grow`]).
    retry_growth_at: u64,
    /// Sum of [`Piece::real`] for pieces taken and not yet given back.
    textures: u64,
    /// Blocks let go since the start, and those of them that were chosen
    /// to empty ([`Pool::empty_out`]).
    returned: u64,
    emptied: u64,
    /// Milliseconds a piece given back rests ([`REST`]; the staging pool's
    /// may be set, [`staging_rest`]).
    rest: u64,
}

impl<B: Clone> Pool<B> {
    const fn new(most: usize) -> Pool<B> {
        Pool {
            blocks: Vec::new(),
            resting: VecDeque::new(),
            spare: 0,
            most,
            retry_growth_at: 0,
            textures: 0,
            returned: 0,
            emptied: 0,
            rest: REST,
        }
    }

    /// A piece of `len` bytes and the memory it lies in, and whether that
    /// part was ever used before; `make` makes a block when none has room.
    fn take(&mut self, len: u64, align: u64, now: u64, make: impl FnMut() -> Option<B>) -> Option<(Piece, B, bool)> {
        self.take_ripe(len, align, now, Ripe::ALL, false, make)
    }

    /// The same from a block that is ripe, if one has room; `make` makes one
    /// ahead too, when few are empty. `pack` ([`pack`]): from the fullest
    /// such block that holds pieces of its kind (small or large) or none,
    /// not from the first with room.
    fn take_ripe(
        &mut self,
        len: u64,
        align: u64,
        now: u64,
        ripe: Ripe,
        pack: bool,
        mut make: impl FnMut() -> Option<B>,
    ) -> Option<(Piece, B, bool)> {
        // With blocks made ahead, what would be an allocation of its own (and
        // new when the card first reads it) lies in one of those too.
        if len == 0 || len > if ripe.ahead > 0 { BLOCK } else { BLOCK / 2 } {
            return None;
        }
        self.rested(now);
        let small = len <= SMALL_PIECE;
        // Not a block chosen to empty; packed, not one of the other kind.
        let open = |block: &Block<B>| block.drain <= now && (!pack || block.small == small || block.free.whole(BLOCK));
        let mut found = None;
        if pack {
            let mut fullest: Option<(u64, usize)> = None;
            for (i, block) in self.blocks.iter().enumerate() {
                if let Some(block) = block
                    && ripe.is(block)
                    && open(block)
                    && fullest.is_none_or(|(free, _)| block.free.free_bytes() < free)
                    && block.free.place(len, align).is_some()
                {
                    fullest = Some((block.free.free_bytes(), i));
                }
            }
            found = fullest.and_then(|(_, i)| Some((i, self.blocks[i].as_mut()?.free.take(len, align)?)));
        } else {
            for (i, block) in self.blocks.iter_mut().enumerate() {
                if let Some(block) = block
                    && ripe.is(block)
                    && open(block)
                    && let Some(start) = block.free.take(len, align)
                {
                    found = Some((i, start));
                    break;
                }
            }
        }
        if found.is_none() && ripe.ahead > 0 {
            // The one that has been there the longest.
            let mut unripe: Vec<(u64, usize)> = self
                .blocks
                .iter()
                .enumerate()
                .filter_map(|(i, b)| b.as_ref().filter(|b| open(b)).map(|b| (b.made, i)))
                .collect();
            unripe.sort_unstable();
            found = unripe.into_iter().find_map(|(_, i)| Some((i, self.blocks[i].as_mut()?.free.take(len, align)?)));
        }
        // A failed 64 MiB block must not trigger another driver call for each
        // small texture. Existing free ranges remain usable during backoff.
        let (i, start) = match found {
            Some(found) => found,
            None => {
                let memory = self.grow(now, &mut make)?;
                let i = self.add(memory, ripe.started);
                (i, self.blocks[i].as_mut()?.free.take(len, align)?)
            }
        };
        if ripe.ahead > 0 {
            if !ripe.is(self.blocks[i].as_ref()?) {
                let n = UNRIPE.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::info!("textures cut from blocks made while lists ran that still run: {n}");
                }
            }
            let empty = self.blocks.iter().flatten().filter(|b| b.free.whole(BLOCK)).count();
            if empty < ripe.ahead
                && let Some(memory) = self.grow(now, &mut make)
            {
                self.add(memory, ripe.started);
            }
        }
        let block = self.blocks[i].as_mut()?;
        // (Its first piece: it was whole.)
        if block.free.free_bytes() + len == BLOCK {
            block.small = small;
        }
        let again = start < block.used_to;
        block.used_to = block.used_to.max(start + len);
        block.seen = crate::priority::ticks();
        Some((Piece { block: i, start, len, real: 0 }, block.memory.clone(), again))
    }

    /// A new block from `make`, unless one was refused less than 500 ms ago
    /// (codec-d/cool_fox 08e4574d: a refused 64 MiB block is not asked for
    /// again for each small texture; the blocks there are still used).
    fn grow(&mut self, now: u64, make: &mut impl FnMut() -> Option<B>) -> Option<B> {
        if now < self.retry_growth_at {
            return None;
        }
        let block = make();
        if block.is_none() {
            self.retry_growth_at = now.saturating_add(500);
        }
        block
    }

    fn add(&mut self, memory: B, made: u64) -> usize {
        let i = match self.blocks.iter().position(Option::is_none) {
            Some(i) => i,
            None => {
                self.blocks.push(None);
                self.blocks.len() - 1
            }
        };
        self.blocks[i] = Some(Block {
            memory,
            free: Ranges::new(BLOCK),
            used_to: 0,
            made,
            since: crate::host::now_ms(),
            seen: crate::priority::ticks(),
            told: crate::priority::NORMAL,
            small: false,
            drain: 0,
        });
        i
    }

    fn give(&mut self, piece: Piece, now: u64) {
        self.textures = self.textures.saturating_sub(piece.real);
        self.resting.push_back((now + self.rest, piece));
    }

    /// Counts `real` as the texture's own size inside `piece` (the piece's
    /// length is what the block reserved, alignment included).
    fn note_texture(&mut self, piece: &mut Piece, real: u64) {
        self.textures = self.textures.saturating_sub(piece.real).saturating_add(real);
        piece.real = real;
    }

    /// One look at time `now` (ms), for the lines every 5 s. Not per texture.
    fn account(&self, now: u64) -> Account {
        let mut blocks = 0u64;
        let mut free = 0u64;
        let mut empty = 0u64;
        let mut largest_free = 0u64;
        let mut bands = [0u64; 4];
        let mut fills = Vec::new();
        let mut draining = 0u64;
        for block in self.blocks.iter().flatten() {
            blocks += 1;
            let free_bytes = block.free.free_bytes();
            free += free_bytes;
            largest_free = largest_free.max(block.free.largest());
            if block.free.whole(BLOCK) {
                empty += 1;
            } else {
                fills.push(BLOCK.saturating_sub(free_bytes));
            }
            bands[band(BLOCK.saturating_sub(free_bytes))] += 1;
            draining += u64::from(block.drain > now);
        }
        fills.sort_unstable();
        let resting = self.resting.iter().map(|(_, piece)| piece.len).sum();
        // Pieces still held: everything not free, except pieces given back
        // that have not joined the free ranges yet.
        let held = blocks.saturating_mul(BLOCK).saturating_sub(free).saturating_sub(resting);
        Account {
            blocks,
            block_bytes: blocks.saturating_mul(BLOCK),
            held,
            textures: self.textures,
            empty,
            resting,
            largest_free,
            bands,
            fewest: fills.first().copied().unwrap_or(0),
            median: fills.get(fills.len() / 2).copied().unwrap_or(0),
            draining,
            returned: self.returned,
            emptied: self.emptied,
        }
    }

    /// The places of the blocks to empty so that `need` bytes of blocks
    /// come back. `idle`: by block, the bytes of the pieces of textures
    /// that may be let go now. A block can be emptied when those are all
    /// its pieces (none in use, none being made, none on its way out
    /// already). Of such blocks those with the fewest bytes are chosen (the
    /// least to make again), as many as `need` takes less those chosen
    /// before and still to go, [`MOST_A_LOOK`] at most. Nothing new is
    /// placed in a chosen block for [`DRAIN_MS`], and once it is empty it
    /// goes whatever the spare blocks ([`Pool::reclaim`]).
    fn empty_out(&mut self, need: u64, idle: &BTreeMap<u32, u64>, now: u64) -> Vec<u32> {
        let mut resting: BTreeMap<usize, u64> = BTreeMap::new();
        for (_, piece) in &self.resting {
            *resting.entry(piece.block).or_default() += piece.len;
        }
        let going = self.blocks.iter().flatten().filter(|b| b.drain > now).count() as u64;
        let wanted = need.div_ceil(BLOCK).saturating_sub(going).min(MOST_A_LOOK);
        let mut can: Vec<(u64, u32)> = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            let (Some(block), Ok(place)) = (block, u32::try_from(i)) else { continue };
            let pieces =
                BLOCK.saturating_sub(block.free.free_bytes()).saturating_sub(resting.get(&i).copied().unwrap_or(0));
            if block.drain <= now && pieces > 0 && idle.get(&place) == Some(&pieces) {
                can.push((pieces, place));
            }
        }
        can.sort_unstable();
        can.truncate(wanted as usize);
        for &(_, place) in &can {
            if let Some(Some(block)) = self.blocks.get_mut(place as usize) {
                block.drain = now.saturating_add(DRAIN_MS);
            }
        }
        can.into_iter().map(|(_, place)| place).collect()
    }

    /// Makes the pieces that rested long enough free, and lets go of empty
    /// blocks past [`SPARE`].
    fn rested(&mut self, now: u64) {
        self.reclaim(now, SPARES.load(Ordering::Relaxed), false);
    }

    /// [`Pool::rested`] with at most `limit` empty blocks kept; `sweep`: the
    /// empty blocks are looked at even if no piece came back. A block
    /// chosen to empty ([`Pool::empty_out`]) is not kept.
    fn reclaim(&mut self, now: u64, limit: usize, sweep: bool) {
        let mut any = false;
        while let Some(&(ready, piece)) = self.resting.front()
            && ready <= now
        {
            self.resting.pop_front();
            if let Some(block) = self.blocks[piece.block].as_mut() {
                block.free.give(piece.start, piece.len);
                any |= block.free.whole(BLOCK);
            }
        }
        // (Without a piece given back too, once there is less room.)
        let most = limit.min(self.most);
        let previous = std::mem::replace(&mut self.spare, most);
        if sweep || any || most < previous {
            let mut spare = 0;
            for block in &mut self.blocks {
                let Some(empty) = block.as_ref().filter(|b| b.free.whole(BLOCK)) else { continue };
                let emptied = empty.drain > now;
                spare += usize::from(!emptied);
                if emptied || spare > most {
                    *block = None;
                    self.returned += 1;
                    self.emptied += u64::from(emptied);
                }
            }
        }
    }

    fn blocks(&self) -> usize {
        self.blocks.iter().flatten().count()
    }

    /// For each place of a block, whether a block there showed a sign of
    /// use within the last `within` looks before `tick`.
    fn fresh(&self, tick: u64, within: u64) -> Vec<bool> {
        self.blocks.iter().map(|b| b.as_ref().is_some_and(|b| tick.saturating_sub(b.seen) < within)).collect()
    }

    /// The blocks whose residency priority is not what Windows was last
    /// told, each with its memory and the priority to tell
    /// (`crate::priority::of`); then how many blocks are low now, and how
    /// many below normal by their fill. `seen`: the places of blocks one
    /// of whose textures was found in use at look `tick`. A block is idle
    /// once it showed no sign of use for `crate::priority::IDLE_TICKS`
    /// looks; its fullness is what is not free of it ([`band`]).
    fn told(&mut self, seen: &[u32], tick: u64, rules: crate::priority::Rules) -> (Vec<(B, u32)>, [u64; 2]) {
        for &i in seen {
            if let Some(Some(block)) = self.blocks.get_mut(i as usize) {
                block.seen = tick;
            }
        }
        let mut change = Vec::new();
        let mut now = [0u64; 2];
        for block in self.blocks.iter_mut().flatten() {
            let kind = crate::priority::Kind::Block {
                idle: tick.saturating_sub(block.seen) >= crate::priority::IDLE_TICKS,
                band: band(BLOCK.saturating_sub(block.free.free_bytes())),
            };
            let want = crate::priority::of(kind, BLOCK, rules);
            if want != block.told {
                block.told = want;
                change.push((block.memory.clone(), want));
            }
            now[0] += u64::from(want == crate::priority::LOW);
            now[1] += u64::from(want != crate::priority::LOW && want < crate::priority::NORMAL);
        }
        (change, now)
    }
}

/// What one look at the texture pool sees. `held` is the bytes of pieces a
/// texture still has (what [`HELD`] counts, once the placing has finished).
/// `textures` is the sum of those textures' own sizes: the bytes the card
/// would give each one at a 4 KiB alignment, which is less than the piece
/// where the piece was rounded up to 64 KiB. `resting` has been given back
/// and is not free yet. A block's fullness counts pieces and resting both,
/// since neither is free.
pub(crate) struct Account {
    pub(crate) blocks: u64,
    pub(crate) block_bytes: u64,
    pub(crate) held: u64,
    pub(crate) textures: u64,
    pub(crate) empty: u64,
    pub(crate) resting: u64,
    pub(crate) largest_free: u64,
    /// Under 25%, 25% to under 50%, 50% to under 75%, 75% and over.
    pub(crate) bands: [u64; 4],
    /// The bytes not free of the emptiest block that is not empty, and of
    /// the median one of those.
    pub(crate) fewest: u64,
    pub(crate) median: u64,
    /// Blocks chosen to empty that have not gone yet.
    pub(crate) draining: u64,
    /// Blocks let go since the start, and those of them chosen to empty.
    pub(crate) returned: u64,
    pub(crate) emptied: u64,
}

/// Fullness of one block from the bytes that are not free. A block on a
/// quarter counts in the band that starts there, and a block at 75% counts
/// as 75% and over.
fn band(used: u64) -> usize {
    if used.saturating_mul(4) < BLOCK {
        0
    } else if used.saturating_mul(2) < BLOCK {
        1
    } else if used.saturating_mul(4) < BLOCK.saturating_mul(3) {
        2
    } else {
        3
    }
}

/// The texture pool, as of this moment. One pass under its lock.
#[cfg(windows)]
pub(crate) fn account() -> Account {
    lock(&TEXTURES).account(now())
}

fn mb(n: u64) -> u64 {
    n >> 20
}

/// The pool line beside the residency line.
pub(crate) fn fill_line(account: &Account) -> String {
    format!(
        "pool fill: blocks {}, block bytes {} MB, pieces hold {} MB, textures in them {} MB, \
         empty {}, resting {} MB, largest free range {} MB, full under 25% {}, \
         25% to under 50% {}, 50% to under 75% {}, 75% and over {}",
        account.blocks,
        mb(account.block_bytes),
        mb(account.held),
        mb(account.textures),
        account.empty,
        mb(account.resting),
        mb(account.largest_free),
        account.bands[0],
        account.bands[1],
        account.bands[2],
        account.bands[3],
    )
}

/// The packing line beside the pool line, whether the pool is packed or
/// not: what an A/B of `VICTORY_POOL_PACK` is judged by. `let_go`:
/// [`EMPTYING`].
pub(crate) fn pack_line(packed: bool, account: &Account, let_go: [u64; 4]) -> String {
    format!(
        "pool pack: VICTORY_POOL_PACK {} now, blocks held {}, of those not empty the emptiest holds {} MB and the median one {} MB, \
         blocks given back {} since the start ({} of them emptied for the budget), being emptied now {}, \
         textures let go to empty blocks {} ({} MB, counted with those let go for the budget) in {} looks of {:.2} ms",
        if packed { "on" } else { "off" },
        account.blocks,
        mb(account.fewest),
        mb(account.median),
        account.returned,
        account.emptied,
        account.draining,
        let_go[1],
        mb(let_go[2]),
        let_go[0],
        let_go[3] as f64 / 1e3 / let_go[0].max(1) as f64,
    )
}

/// Since the start: looks for blocks to empty, the textures let go to
/// empty them and their bytes, and the looks' microseconds.
static EMPTYING: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// [`EMPTYING`], now.
pub(crate) fn emptying() -> [u64; 4] {
    EMPTYING.each_ref().map(|n| n.load(Ordering::Relaxed))
}

/// When the system last refused a block of texture memory (ms).
static REFUSED_AT: AtomicU64 = AtomicU64::new(0);

/// The bytes of texture blocks to give back now ([`to_return`]; a block
/// refused within the last two seconds counts). `memory`: the card's
/// budget and use as the system counts them.
pub(crate) fn need(memory: Option<[u64; 4]>) -> u64 {
    let refused = REFUSED_AT.load(Ordering::Relaxed);
    to_return(memory, refused != 0 && now().saturating_sub(refused) < 2000)
}

/// Chooses the texture blocks to empty ([`Pool::empty_out`]).
pub(crate) fn empty_out(need: u64, idle: &BTreeMap<u32, u64>) -> Vec<u32> {
    lock(&TEXTURES).empty_out(need, idle, now())
}

/// Counts one look for blocks to empty: the textures let go, their bytes,
/// and the microseconds it took.
pub(crate) fn emptied(textures: u64, bytes: u64, us: u64) {
    for (count, n) in EMPTYING.iter().zip([1, textures, bytes, us]) {
        count.fetch_add(n, Ordering::Relaxed);
    }
}

/// Where a texture of 64 KiB and less is placed. `four_kib` is
/// [`small_placement`].
pub(crate) fn placement_line(four_kib: bool) -> String {
    let (step, flag) = if four_kib { (4, 1) } else { (64, 0) };
    format!("pool placement: small textures of 64 KiB and less are placed at {step} KiB multiples (VICTORY_POOL_SMALL={flag})")
}

/// The size to count for a texture whose piece is `placed_size` at
/// `placed_align`. `tight_*` is the same texture asked at 4 KiB. A piece
/// rounded up to 64 KiB counts as the 4 KiB size, so the line can show the
/// bytes the alignment added. Anything that is not a smaller 4 KiB answer
/// counts as the piece.
fn counted_texture_bytes(placed_size: u64, placed_align: u64, tight_size: u64, tight_align: u64) -> u64 {
    /// [`D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT`].
    const SMALL: u64 = 4096;
    const _: () = assert!(D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT == SMALL as u32);
    if placed_align > 0 && placed_align <= SMALL {
        return placed_size;
    }
    if tight_align == SMALL && tight_size > 0 && tight_size < placed_size {
        tight_size
    } else {
        placed_size
    }
}

/// A mapped upload buffer's memory.
#[derive(Clone)]
struct Upload(ID3D12Resource, *mut u8);
// SAFETY: the mapping is written by the owners of its pieces, each its own.
unsafe impl Send for Upload {}

static TEXTURES: Mutex<Pool<ID3D12Heap>> = Mutex::new(Pool::new(SPARE_KEPT));
static STAGING: Mutex<Pool<Upload>> = Mutex::new(Pool::new(STAGING_KEPT));

fn lock<T>(m: &'static Mutex<T>) -> std::sync::MutexGuard<'static, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The bytes of the blocks of textures and of staging given back since the
/// start (the residency line).
pub(crate) fn given_back_bytes() -> u64 {
    (lock(&TEXTURES).returned + lock(&STAGING).returned) * BLOCK
}

/// Bytes of the pieces textures hold now.
static HELD: AtomicU64 = AtomicU64::new(0);

/// Bytes of the texture blocks no texture holds: the card's memory in use
/// by us that new textures take first.
pub(crate) fn texture_room() -> u64 {
    (lock(&TEXTURES).blocks() as u64 * BLOCK).saturating_sub(HELD.load(Ordering::Relaxed))
}

/// For each place of a texture block, whether the block there showed a
/// sign of use within the last `within` looks before `tick`
/// (`crate::priority::tick`: such a block's textures are not asked about).
pub(crate) fn fresh_blocks(tick: u64, within: u64) -> Vec<bool> {
    lock(&TEXTURES).fresh(tick, within)
}

/// Tells Windows the residency priority of every texture block whose
/// priority changed ([`Pool::told`]), in one call, outside the pool's lock.
pub(crate) fn prioritize(seen: &[u32], tick: u64, rules: crate::priority::Rules) {
    let (change, now) = lock(&TEXTURES).told(seen, tick, rules);
    crate::priority::blocks_now(now[0], now[1]);
    let change: Vec<(ID3D12Pageable, u32)> =
        change.into_iter().filter_map(|(heap, priority)| Some((heap.cast().ok()?, priority))).collect();
    crate::priority::tell(&change);
}

/// The place of the block a pooled texture's piece lies in; `None` for a
/// texture that is not the pool's.
pub(crate) fn block_of(texture: &ID3D12Resource) -> Option<u32> {
    piece_of(texture).map(|(block, _)| block)
}

/// The same, and the piece's bytes.
pub(crate) fn piece_of(texture: &ID3D12Resource) -> Option<(u32, u64)> {
    let mut held: *mut c_void = std::ptr::null_mut();
    let mut size = size_of::<*mut c_void>() as u32;
    // SAFETY: what the key holds is the object `give_back_with` gave the
    // texture, handed out with a reference of its own, which is given back
    // here; the texture, alive, keeps another.
    unsafe {
        texture.GetPrivateData(&RECLAIM_KEY, &mut size, Some((&raw mut held).cast())).ok()?;
        if held.is_null() || size as usize != size_of::<*mut c_void>() {
            return None;
        }
        let piece = (*held.cast::<Reclaim>()).piece;
        release(held);
        Some((u32::try_from(piece.block).ok()?, piece.len))
    }
}

/// For the memory line: blocks of textures and of staging buffers, in MiB.
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 2] {
    [lock(&TEXTURES).blocks() as u64 * (BLOCK >> 20), lock(&STAGING).blocks() as u64 * (BLOCK >> 20)]
}

/// Part of a block of mapped upload memory, back in the pool when dropped.
pub(crate) struct StagingPiece(Piece);

impl Drop for StagingPiece {
    fn drop(&mut self) {
        lock(&STAGING).give(self.0, now());
    }
}

/// `bytes` of upload memory, all zeros: the buffer it lies in, where in
/// that buffer, and its mapping. `None`: not pooled.
pub(crate) fn staging(host: &crate::host::Host, bytes: u64) -> Option<(StagingPiece, ID3D12Resource, u64, *mut u8)> {
    if !staging_on() {
        return None;
    }
    // Copies from a buffer begin at multiples of 512.
    let bytes = bytes.next_multiple_of(512);
    let mut pool = lock(&STAGING);
    pool.rest = staging_rest();
    let taken = pool.take(bytes, 512, now(), || {
        // (`VICTORY_RESIDENCY=1`: no new block for a texture picked by index
        // while commit is short; those a draw samples now still get one.)
        let speculative = crate::texture::PICKING.get().is_some() && !crate::texture::BAKING.get();
        if speculative && !crate::residency::allow_staging_growth(false) {
            return None;
        }
        let (buffer, data) = host.upload_buffer(BLOCK)?;
        crate::host::name(&buffer, || "staging of textures (a block)".into());
        Some(Upload(buffer, data))
    });
    drop(pool);
    let (piece, memory, again) = taken?;
    let owned = StagingPiece(piece);
    // SAFETY: the piece lies in the block's mapping.
    let data = unsafe { memory.1.add(piece.start as usize) };
    if again {
        // What is not detiled into it (an unmapped texture) uploads zeros,
        // as from a buffer of its own.
        // SAFETY: the piece is ours alone.
        unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
    }
    Some((owned, memory.0, piece.start, data))
}

/// An object the host texture holds and releases when it goes: its piece
/// is free then.
#[repr(C)]
struct Reclaim {
    methods: &'static [*const (); 3],
    references: AtomicU32,
    piece: Piece,
}

struct Methods([*const (); 3]);
// SAFETY: function addresses only.
unsafe impl Sync for Methods {}

static RECLAIM: Methods = Methods([query as *const (), add_ref as *const (), release as *const ()]);

unsafe extern "system" fn query(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    const E_NOINTERFACE: HRESULT = HRESULT(0x8000_4002_u32 as i32);
    // SAFETY: COM's arguments.
    unsafe {
        if out.is_null() {
            return HRESULT(0x8000_4003_u32 as i32);
        }
        if iid.as_ref().is_some_and(|iid| *iid == IUnknown::IID) {
            add_ref(this);
            *out = this;
            return HRESULT(0);
        }
        *out = std::ptr::null_mut();
    }
    E_NOINTERFACE
}

unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
    // SAFETY: one of ours, alive while a reference to it is.
    unsafe { (*this.cast::<Reclaim>()).references.fetch_add(1, Ordering::Relaxed) + 1 }
}

unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    // SAFETY: as above; the last reference frees the box it was made as.
    unsafe {
        let left = (*this.cast::<Reclaim>()).references.fetch_sub(1, Ordering::AcqRel) - 1;
        if left == 0 {
            let gone = Box::from_raw(this.cast::<Reclaim>());
            HELD.fetch_sub(gone.piece.len, Ordering::Relaxed);
            lock(&TEXTURES).give(gone.piece, now());
        }
        left
    }
}

/// Whether textures of 64 KiB and less are placed at multiples of 4 KiB in
/// their block, where the host allows it (`VICTORY_POOL_SMALL=1`; otherwise
/// at multiples of 64 KiB, as every other texture).
///
/// The two device losses of the night of 2026-11-25 with the pool in the
/// report (runs steps-ab2 and start-3) faulted in the same kind of texture:
/// 256x256, block-compressed, 7 levels, placed at a 4 KiB multiple that is no
/// 64 KiB one (offsets 0x406000 and 0x3fe6000 of their blocks), never used
/// before, read by the card the moment its slot was made; the descriptor's
/// base of both lies 0xa000 into the piece, on a 64 KiB multiple, and its
/// first level of 32 KiB from there ends 0x2000 past the piece. Why the card
/// faulted there is not proven; textures at 64 KiB multiples have none of
/// that, for some memory (at most 60 KiB a small texture).
pub(crate) fn small_placement() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_POOL_SMALL") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_POOL_SMALL").is_some_and(|v| v != "0"))
}

/// Diagnostics: whether the card reads a pooled texture inside the piece it
/// was placed in. A block's address is taken as the lowest of its textures'
/// descriptor bases less their pieces' starts (a texture's first level may
/// lie past its piece's start, never before it); a texture whose first level
/// the card would read from past its piece's end is said loudly.
#[cfg(feature = "diagnostics")]
fn where_the_card_reads(heap: &ID3D12Heap, texture: &ID3D12Resource, piece: Piece) {
    static BLOCKS: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());
    static SAID: AtomicU64 = AtomicU64::new(0);
    let Some(base) = crate::book::address(texture) else { return };
    let Some(seen) = base.checked_sub(piece.start) else { return };
    let raw = heap.as_raw() as usize;
    let block = {
        let mut blocks = BLOCKS.lock().unwrap_or_else(|e| e.into_inner());
        match blocks.iter_mut().find(|b| b.0 == raw) {
            Some(b) => {
                b.1 = b.1.min(seen);
                b.1
            }
            None => {
                if blocks.len() >= 4096 {
                    blocks.remove(0);
                }
                blocks.push((raw, seen));
                seen
            }
        }
    };
    let into = base - block - piece.start;
    // SAFETY: a live texture.
    let desc = unsafe { texture.GetDesc() };
    // The first level's bytes, as a copy would have them.
    let mut level = 0u64;
    if let Some(host) = crate::host::made_host() {
        // SAFETY: plain query of a valid description.
        unsafe { host.device.GetCopyableFootprints(&desc, 0, 1, 0, None, None, None, Some(&raw mut level)) };
    }
    if into + level > piece.len {
        let n = SAID.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 50 || n.is_power_of_two() {
            // (Debug only: it assumes a texture's address is its block's
            // start and its piece's offset, which this driver does not
            // keep: 56 false alarms in run pool-1.)
            tracing::debug!(
                times = n,
                base = format_args!("{base:#x}"),
                piece_start = format_args!("{:#x}", piece.start),
                piece_len = format_args!("{:#x}", piece.len),
                into = format_args!("{into:#x}"),
                first_level_bytes = format_args!("{level:#x}"),
                width = desc.Width,
                height = desc.Height,
                levels = desc.MipLevels,
                format = desc.Format.0,
                "POOL: the card reads a pooled texture's first level past the end of the piece it was placed in"
            );
        }
    }
    // Every placement offset and where the first level lies, a few of them.
    static SHOWN: AtomicU64 = AtomicU64::new(0);
    if into > 0 && SHOWN.fetch_add(1, Ordering::Relaxed) < 40 {
        tracing::info!(
            into = format_args!("{into:#x}"),
            piece_start = format_args!("{:#x}", piece.start),
            piece_len = format_args!("{:#x}", piece.len),
            width = desc.Width,
            height = desc.Height,
            levels = desc.MipLevels,
            format = desc.Format.0,
            alignment = desc.Alignment,
            "pool: a texture's first level lies past its piece's start"
        );
    }
}

/// Textures made in the last second ([`fresh`]), and when.
static RECENT: Mutex<VecDeque<(usize, u64)>> = Mutex::new(VecDeque::new());

/// Whether `texture` was made in a block in the last second.
pub(crate) fn fresh(texture: &ID3D12Resource) -> bool {
    let raw = texture.as_raw() as usize;
    let since = now().saturating_sub(1000);
    RECENT.lock().unwrap_or_else(|e| e.into_inner()).iter().rev().take_while(|r| r.1 >= since).any(|r| r.0 == raw)
}

fn made_now(texture: &ID3D12Resource) {
    let now = now();
    let raw = texture.as_raw() as usize;
    {
        let mut recent = RECENT.lock().unwrap_or_else(|e| e.into_inner());
        while recent.front().is_some_and(|r| r.1 + 1000 < now) {
            recent.pop_front();
        }
        recent.push_back((raw, now));
    }
    let begun = crate::host::submissions().0;
    NEWEST_BEGUN.fetch_max(begun, Ordering::Relaxed);
    let mut unseen = UNSEEN.lock().unwrap_or_else(|e| e.into_inner());
    let finished = crate::host::submissions().1;
    // Those every submission begun before them has left behind are no
    // one's concern any more (and a raw pointer may come again).
    unseen.retain(|_, b| *b > finished);
    unseen.insert(raw, begun);
}

/// Pooled textures made while submissions begun before them may still run:
/// the submissions begun when each was made.
static UNSEEN: std::sync::LazyLock<Mutex<std::collections::HashMap<usize, u64>>> = std::sync::LazyLock::new(Default::default);
/// The newest of those numbers (nothing to look up once it has finished).
static NEWEST_BEGUN: AtomicU64 = AtomicU64::new(0);

/// Whether a texture was made in a block while a submission begun before
/// it may still run ([`begun_before`]).
pub(crate) fn any_unseen() -> bool {
    NEWEST_BEGUN.load(Ordering::Relaxed) > crate::host::submissions().1
}

/// The submissions begun when `texture` was made in a block, if one of
/// those may still be on the card: lists already sent then must not find it
/// in a slot they read by index ([`crate::descriptor::host::show_new_later`]).
/// Every device loss with the pool in the report (runs n08, n11, n15,
/// steps-ab2, start-3 and the user's play-1125-2107) was a page fault at the
/// first byte of a texture made that very millisecond, its title slot
/// changed "at once" while lists sent before it read that slot by index.
pub(crate) fn begun_before(texture: &ID3D12Resource) -> Option<u64> {
    if !any_unseen() {
        return None;
    }
    let finished = crate::host::submissions().1;
    let raw = texture.as_raw() as usize;
    let unseen = UNSEEN.lock().unwrap_or_else(|e| e.into_inner());
    unseen.get(&raw).copied().filter(|&begun| begun > finished)
}

/// What a pooled texture holds its [`Reclaim`] under.
const RECLAIM_KEY: GUID = GUID::from_u128(0x6f0e2f0c_5a59_4b1e_9a3f_7f4d2c9be713);

/// Makes `texture` give `piece` back when the last reference to it goes.
fn give_back_with(texture: &ID3D12Resource, piece: Piece) {
    let made = Box::into_raw(Box::new(Reclaim { methods: &RECLAIM.0, references: AtomicU32::new(1), piece }));
    // SAFETY: a COM object of ours with one reference, which this takes and
    // drops at the end; the texture takes its own.
    unsafe {
        let object = IUnknown::from_raw(made.cast());
        if texture.SetPrivateDataInterface(&RECLAIM_KEY, &object).is_err() {
            // Never given back: the piece stays taken, nothing is reused
            // under a texture that may be alive.
            std::mem::forget(object);
        }
    }
}

/// The texture's own size: what the card asks for it at 4 KiB, when that
/// is smaller than the piece. One query, and only when the piece's
/// alignment is coarser than 4 KiB. Where the piece is placed does not change.
fn real_bytes(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC, placed_size: u64, placed_align: u64) -> u64 {
    const SMALL: u64 = 4096;
    if placed_align > 0 && placed_align <= SMALL {
        return placed_size;
    }
    let mut tight = *desc;
    tight.Alignment = SMALL;
    // SAFETY: a plain query of a description we are about to place, or have
    // just placed. The answer is only counted.
    let tight = unsafe { device.GetResourceAllocationInfo(0, &[tight]) };
    counted_texture_bytes(placed_size, placed_align, tight.SizeInBytes, tight.Alignment)
}

/// A texture of `desc` (no target, depth or unordered access) in a block;
/// `None`: not pooled, or the host refused.
///
/// # Safety
/// `desc` is a valid description.
pub(crate) unsafe fn texture(
    device: &ID3D12Device,
    desc: *const D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> Option<ID3D12Resource> {
    if !on() {
        return None;
    }
    // Small textures lie at multiples of 4 KiB where the host allows it
    // (the top mip is 64 KiB at most) and [`small_placement`] says so: 64 KiB
    // each otherwise.
    // SAFETY: the caller's description.
    let mut d = unsafe { *desc };
    d.Alignment = if small_placement() { u64::from(D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT) } else { 0 };
    // SAFETY: plain queries.
    let mut info = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
    if info.SizeInBytes == u64::MAX || info.Alignment != d.Alignment {
        d.Alignment = 0;
        info = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
    }
    if info.SizeInBytes == u64::MAX {
        return None;
    }
    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
    let (piece, heap, again) = lock(&TEXTURES).take_ripe(info.SizeInBytes, info.Alignment.max(1), now(), Ripe::now(), pack(), || {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: BLOCK,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES | crate::host::not_resident(),
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation.
        if let Err(e) = unsafe { device.CreateHeap(&desc, &mut heap) } {
            crate::residency::allocation_error(&e);
            REFUSED_AT.store(now().max(1), Ordering::Relaxed);
            tracing::warn!(error = %e, "no block of texture memory");
        }
        if let Some(h) = &heap
            && let Err(e) = crate::host::resident(device, h).and_then(|()| crate::host::mapped(device))
        {
            tracing::warn!(error = %e, "a block of texture memory is not resident");
            return None;
        }
        heap
    })?;
    let mut made: Option<ID3D12Resource> = None;
    // SAFETY: the piece is no one else's; the heap lives while a piece of
    // it is taken.
    let result = unsafe { device.CreatePlacedResource(&heap, piece.start, &d, state, None, &mut made) };
    match (result, made) {
        (Ok(()), Some(made)) => {
            HELD.fetch_add(piece.len, Ordering::Relaxed);
            let mut piece = piece;
            let real = real_bytes(device, &d, info.SizeInBytes, info.Alignment);
            lock(&TEXTURES).note_texture(&mut piece, real);
            give_back_with(&made, piece);
            made_now(&made);
            #[cfg(feature = "diagnostics")]
            {
                crate::book::made(&made, || "a texture in a block".into());
                where_the_card_reads(&heap, &made, piece);
                crate::book::block(&heap, &made, piece.start, BLOCK);
                crate::book::note(made.as_raw() as usize, || {
                    let block = lock(&TEXTURES).blocks.get(piece.block).and_then(Option::as_ref).map(|b| (b.made, b.since));
                    let (submission, since) = block.unwrap_or_default();
                    format!(
                        "; piece {:#x}+{:#x} of block {} {:#x} made {:.1} s before it at submission {submission}, used before: {again}, made at submissions {:?}",
                        piece.start,
                        piece.len,
                        piece.block,
                        heap.as_raw() as usize,
                        now().saturating_sub(since) as f64 / 1e3,
                        crate::host::submissions()
                    )
                });
            }
            Some(made)
        }
        _ => {
            lock(&TEXTURES).give(piece, now());
            None
        }
    }
}

#[cfg(test)]
mod tests {
    /// `VICTORY_TEXTURE_POOL`: unset and other values pool both, `0` neither,
    /// `staging` only the staging.
    #[test]
    fn texture_pool_staging_keeps_only_the_staging_pooled() {
        use super::pool_value;
        assert_eq!(pool_value(None), (true, true));
        assert_eq!(pool_value(Some("1")), (true, true));
        assert_eq!(pool_value(Some("")), (true, true));
        assert_eq!(pool_value(Some("0")), (false, false));
        // Only `0` itself turns it off, as before.
        assert_eq!(pool_value(Some(" 0 ")), (true, true));
        assert_eq!(pool_value(Some("staging")), (false, true));
        assert_eq!(pool_value(Some("Staging")), (false, true));
    }

    use super::*;

    #[test]
    fn a_failed_block_is_not_retried_for_each_small_upload() {
        let mut pool: Pool<u32> = Pool::new(2);
        let calls = std::cell::Cell::new(0);
        let refuse = || {
            calls.set(calls.get() + 1);
            None
        };
        for size in [704008, 179720, 179720, 24080, 48648, 524288, 89616, 351760] {
            assert!(pool.take(size, 512, 12, refuse).is_none());
        }
        assert_eq!(calls.get(), 1);
        assert!(pool.take(24080, 512, 1012, || Some(1)).is_some());
    }

    #[test]
    fn budget_tick_reclaims_idle_blocks_without_reusing_live_or_resting_pieces() {
        let mut pool = Pool::new(STAGING_KEPT);
        let (held, _, _) = pool.take(BLOCK / 2, 1, 0, || Some(1)).unwrap();
        let (resting, _, _) = pool.take(BLOCK / 2, 1, 0, || Some(2)).unwrap();
        pool.give(resting, 0);
        for i in 0..12 {
            pool.add(i, 0);
        }
        pool.reclaim(0, SPARE_KEPT, true);
        assert_eq!(pool.blocks(), STAGING_KEPT + 1);
        pool.reclaim(REST - 1, SPARE, true);
        assert_eq!(pool.blocks(), SPARE + 1);
        assert_eq!(pool.resting.len(), 1);
        pool.reclaim(REST, 0, true);
        assert_eq!(pool.blocks(), 1);
        assert!(pool.resting.is_empty());
        pool.give(held, REST);
        pool.reclaim(2 * REST, 0, true);
        assert_eq!(pool.blocks(), 0);
    }

    #[test]
    fn ranges_are_cut_aligned_and_join_again() {
        let mut r = Ranges::new(1024);
        assert_eq!(r.take(100, 64), Some(0));
        assert_eq!(r.take(100, 64), Some(128));
        assert_eq!(r.take(28, 4), Some(100));
        assert_eq!(r.take(2000, 4), None);
        r.give(128, 100);
        r.give(0, 100);
        assert!(!r.whole(1024));
        r.give(100, 28);
        assert!(r.whole(1024));
    }

    #[test]
    fn empty_blocks_stay_while_the_card_has_room() {
        assert_eq!(spares(Some([100, 84, 0, 0]), true), SPARE_KEPT);
        assert_eq!(spares(Some([100, 85, 0, 0]), true), SPARE);
        assert_eq!(spares(Some([100, 10, 0, 0]), false), SPARE);
        assert_eq!(spares(None, true), SPARE);
    }

    #[test]
    fn pieces_never_overlap_and_rest_before_they_are_used_again() {
        let mut pool: Pool<u32> = Pool::new(SPARE);
        let made = std::cell::Cell::new(0);
        let make = || {
            made.set(made.get() + 1);
            Some(made.get())
        };
        let mut taken: Vec<Piece> = Vec::new();
        for i in 0..500u64 {
            let (p, _, again) = pool.take(65536 * (1 + i % 7), 65536, 0, make).unwrap();
            assert!(!again);
            assert!(taken.iter().all(|q| q.block != p.block || p.start + p.len <= q.start || q.start + q.len <= p.start));
            taken.push(p);
        }
        let blocks = pool.blocks();
        assert!(blocks >= 2);
        // Given back at 10: not used again before 10 + REST.
        let first = taken[0];
        pool.give(first, 10);
        let (p, _, _) = pool.take(first.len, 65536, 10 + REST - 1, make).unwrap();
        assert_ne!((p.block, p.start), (first.block, first.start));
        let (q, _, again) = pool.take(first.len, 65536, 10 + REST, make).unwrap();
        assert_eq!((q.block, q.start, again), (first.block, first.start, true));
        // Everything back: the empty blocks past the spare ones go.
        for p in taken.drain(1..).chain([p, q]) {
            pool.give(p, 20);
        }
        pool.rested(20 + REST);
        assert_eq!(pool.blocks(), blocks.min(SPARE));
        // Too big for a block, and a block that cannot be made.
        assert!(pool.take(BLOCK, 65536, 30, make).is_none());
        let mut none: Pool<u32> = Pool::new(SPARE);
        assert!(none.take(65536, 65536, 0, || None).is_none());
    }

    #[test]
    fn pieces_come_from_blocks_older_than_the_lists_that_run() {
        let mut pool: Pool<u64> = Pool::new(SPARE_KEPT);
        let made = std::cell::Cell::new(0);
        let make = || {
            made.set(made.get() + 1);
            Some(made.get())
        };
        let at = |started, finished| Ripe { started, finished, ahead: 2 };
        let take = |pool: &mut Pool<u64>, ripe| pool.take_ripe(BLOCK / 2, 65536, 0, ripe, false, make).unwrap().1;
        // Nothing runs: at once, and two are made ahead, one a time.
        assert_eq!(take(&mut pool, at(0, 0)), 1);
        assert_eq!(made.get(), 2);
        assert_eq!(take(&mut pool, at(0, 0)), 1);
        assert_eq!(made.get(), 3);
        // 4 is made while submission 5 runs.
        assert_eq!(take(&mut pool, at(5, 4)), 2);
        assert_eq!(made.get(), 4);
        assert_eq!(take(&mut pool, at(5, 4)), 2);
        assert_eq!(take(&mut pool, at(6, 4)), 3);
        assert_eq!(take(&mut pool, at(6, 4)), 3);
        // None is ripe: the oldest; it is ripe once 5 has finished.
        assert_eq!(take(&mut pool, at(7, 4)), 4);
        assert_eq!(take(&mut pool, at(7, 5)), 4);
        // A whole block's worth too.
        assert_eq!(pool.take_ripe(BLOCK, 65536, 0, at(8, 7), false, make).unwrap().1, 5);
        // Off: the first with room, whatever its age, none ahead.
        let n = made.get();
        assert_eq!(pool.take(BLOCK / 2, 65536, 0, make).unwrap().1, 6);
        assert_eq!(made.get(), n);
        assert!(pool.take(BLOCK, 65536, 0, make).is_none());
    }

    /// A texture placed in a heap holds no reference to it that its count
    /// shows (this host, 2026-11-25: 1 before placing, 1 with the texture):
    /// the pool must not count on one, and does not (a block goes only with
    /// every piece back and rested, [`Pool::rested`]). Said here, and the
    /// count is the same again once the texture went.
    #[test]
    fn a_placed_texture_holds_its_heap() {
        let Some(host) = crate::host::host() else { return };
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: 1 << 20,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation.
        unsafe { host.device.CreateHeap(&desc, &mut heap) }.unwrap();
        let heap = heap.unwrap();
        let before = crate::host::references(&heap);
        let texture_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let mut made: Option<ID3D12Resource> = None;
        // SAFETY: a live heap, a valid description.
        unsafe { host.device.CreatePlacedResource(&heap, 0, &texture_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut made) }.unwrap();
        let made = made.unwrap();
        let held = crate::host::references(&heap);
        println!("heap references: {before} before placing, {held} with a texture in it");
        assert!(held >= before);
        drop(made);
        assert_eq!(crate::host::references(&heap), before);
    }

    /// A pooled texture's piece comes back when its last reference goes, not
    /// before; its staging piece begins as zeros.
    #[test]
    fn a_texture_gives_its_piece_back_when_it_goes() {
        let Some(host) = crate::host::host() else { return };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        // SAFETY: a valid description.
        let Some(t) = (unsafe { texture(&host.device, &desc, D3D12_RESOURCE_STATE_COMMON) }) else { return };
        assert_eq!(crate::host::references(&t), 1, "what the texture holds is no reference to it");
        // Its block is known from the texture alone, and asking leaves the
        // texture's piece where it is; a texture of its own has no block.
        let block = block_of(&t).expect("a pooled texture's block");
        assert!(lock(&TEXTURES).blocks.get(block as usize).is_some_and(Option::is_some));
        assert_eq!(block_of(&t), Some(block));
        assert_eq!(crate::host::references(&t), 1);
        let own = host.uav_buffer(1 << 16).unwrap();
        assert_eq!(block_of(&own), None);
        let resting = || lock(&TEXTURES).resting.len();
        let before = resting();
        let other = t.clone();
        drop(t);
        assert_eq!(resting(), before);
        drop(other);
        assert_eq!(resting(), before + 1);
        // SAFETY: the piece's mapping, 4096 bytes of it ours.
        unsafe {
            let (piece, _, at, data) = staging(host, 4096).unwrap();
            assert!(at.is_multiple_of(512));
            std::ptr::write_bytes(data, 0xAB, 4096);
            drop(piece);
            lock(&STAGING).rested(u64::MAX);
            let (_piece, _, _, data) = staging(host, 4096).unwrap();
            assert!((0..4096).all(|i| *data.add(i) == 0));
            // SAFETY: plain query.
            assert!(host.device.GetDeviceRemovedReason().is_ok());
        }
    }

    /// Windows is told a block's priority when it changes, and only then:
    /// low once the block showed no sign of use for `IDLE_TICKS` looks,
    /// normal again when one of its textures is found in use; with the
    /// other rule, the lower the emptier it is.
    #[test]
    fn blocks_are_told_low_when_idle_and_lower_when_emptier() {
        use crate::priority::{IDLE_TICKS, LOW, NORMAL, Rules};
        let by_use = Rules { by_use: true, by_fill: false };
        let by_fill = Rules { by_use: false, by_fill: true };
        let mut pool: Pool<u32> = Pool::new(8);
        // The block at place 0 (memory 1) is full, the one at place 1
        // (memory 2) an eighth full.
        pool.take(BLOCK / 2, 1, 0, || Some(1)).unwrap();
        pool.take(BLOCK / 2, 1, 0, || Some(1)).unwrap();
        let (small, _, _) = pool.take(BLOCK / 8, 1, 0, || Some(2)).unwrap();
        assert_eq!(small.block, 1);
        // Every heap begins normal: nothing to tell.
        assert_eq!(pool.told(&[], 1, by_use), (vec![], [0, 0]));
        assert_eq!(pool.fresh(5, 10), [true, true]);
        // A texture of the first block is found in use at every look, none
        // of the second's ever.
        assert_eq!(pool.told(&[0], IDLE_TICKS - 1, by_use), (vec![], [0, 0]));
        assert_eq!(pool.told(&[0], IDLE_TICKS, by_use), (vec![(2, LOW)], [1, 0]));
        assert_eq!(pool.fresh(IDLE_TICKS + 5, 10), [true, false]);
        // Told once.
        assert_eq!(pool.told(&[0], IDLE_TICKS + 1, by_use), (vec![], [1, 0]));
        // One of its textures is found in use: normal again. A place
        // without a block is no one's.
        assert_eq!(pool.told(&[0, 1, 7], IDLE_TICKS + 2, by_use), (vec![(2, NORMAL)], [0, 0]));
        // Nothing of either for long: both low.
        let late = 3 * IDLE_TICKS;
        assert_eq!(pool.told(&[], late, by_use), (vec![(1, LOW), (2, LOW)], [2, 0]));
        // The switch, in turn, is off: both normal again.
        assert_eq!(pool.told(&[], late, Rules::default()), (vec![(1, NORMAL), (2, NORMAL)], [0, 0]));
        // By fill: the emptier block is lower, the full one stays normal.
        assert_eq!(pool.told(&[], late, by_fill), (vec![(2, 0x5a00_0000)], [0, 1]));
        // It fills to five eighths: its band, and its priority with it.
        pool.take(BLOCK / 2, 1, 0, || None).unwrap();
        assert_eq!(pool.told(&[], late, by_fill), (vec![(2, 0x6e00_0000)], [0, 1]));
        // Both rules: an idle block is low whatever its fill.
        assert_eq!(pool.told(&[], late, Rules { by_use: true, by_fill: true }), (vec![(1, LOW), (2, LOW)], [2, 0]));
    }

    #[test]
    fn fullness_bands_meet_at_the_quarters() {
        assert_eq!(band(0), 0);
        assert_eq!(band(BLOCK / 4 - 1), 0);
        assert_eq!(band(BLOCK / 4), 1);
        assert_eq!(band(BLOCK / 2 - 1), 1);
        assert_eq!(band(BLOCK / 2), 2);
        assert_eq!(band(BLOCK * 3 / 4 - 1), 2);
        assert_eq!(band(BLOCK * 3 / 4), 3);
        assert_eq!(band(BLOCK), 3);
    }

    #[test]
    fn a_texture_counted_at_4kib_is_not_the_64kib_piece() {
        assert_eq!(counted_texture_bytes(65536, 65536, 4096, 4096), 4096);
        assert_eq!(counted_texture_bytes(8192, 4096, 0, 0), 8192);
        assert_eq!(counted_texture_bytes(65536, 65536, u64::MAX, 65536), 65536);
        assert_eq!(counted_texture_bytes(65536, 65536, 70000, 4096), 65536);
        assert_eq!(counted_texture_bytes(65536, 65536, 0, 4096), 65536);
        assert_eq!(counted_texture_bytes(65536, 65536, 65536, 4096), 65536);
    }

    #[test]
    fn placement_line_names_4kib_or_64kib() {
        assert_eq!(
            placement_line(false),
            "pool placement: small textures of 64 KiB and less are placed at 64 KiB multiples (VICTORY_POOL_SMALL=0)"
        );
        assert_eq!(
            placement_line(true),
            "pool placement: small textures of 64 KiB and less are placed at 4 KiB multiples (VICTORY_POOL_SMALL=1)"
        );
    }

    /// Pieces taken, given back, rested and reclaimed, and the bytes of the
    /// textures in them. A gap left for alignment stays free, so it is not held.
    #[test]
    fn account_follows_take_give_rest_and_reclaim() {
        let mut pool: Pool<u32> = Pool::new(4);
        let make = || Some(1);
        let (mut small, _, _) = pool.take(BLOCK / 8, 1, 0, make).unwrap();
        pool.note_texture(&mut small, BLOCK / 16);
        let look = pool.account(0);
        assert_eq!(look.blocks, 1);
        assert_eq!(look.block_bytes, BLOCK);
        assert_eq!(look.held, BLOCK / 8);
        assert_eq!(look.textures, BLOCK / 16);
        assert_eq!(look.empty, 0);
        assert_eq!(look.resting, 0);
        assert_eq!(look.largest_free, BLOCK - BLOCK / 8);
        assert_eq!(look.bands, [1, 0, 0, 0]);
        assert_eq!(
            fill_line(&look),
            "pool fill: blocks 1, block bytes 64 MB, pieces hold 8 MB, textures in them 4 MB, \
             empty 0, resting 0 MB, largest free range 56 MB, full under 25% 1, \
             25% to under 50% 0, 50% to under 75% 0, 75% and over 0"
        );

        // 100 bytes at 256 leaves a free gap, which is not part of either piece.
        let mut gapped: Pool<u32> = Pool::new(1);
        let (mut first, _, _) = gapped.take(100, 256, 0, make).unwrap();
        gapped.note_texture(&mut first, 80);
        let (second, _, _) = gapped.take(100, 256, 0, make).unwrap();
        assert_eq!(second.start, 256);
        let look = gapped.account(0);
        assert_eq!(look.held, 200);
        assert_eq!(look.textures, 80);
        assert_eq!(look.largest_free, BLOCK - 356);

        let (mid, _, _) = pool.take(BLOCK / 8, 1, 0, make).unwrap();
        assert_eq!(pool.account(0).bands, [0, 1, 0, 0]);
        assert_eq!(pool.account(0).held, BLOCK / 4);
        assert_eq!(pool.account(0).textures, BLOCK / 16);

        let (half, _, _) = pool.take(BLOCK * 3 / 8, 1, 0, make).unwrap();
        let look = pool.account(0);
        assert_eq!(look.held, BLOCK * 5 / 8);
        assert_eq!(look.largest_free, BLOCK * 3 / 8);
        assert_eq!(look.bands, [0, 0, 1, 0]);

        let (tail, _, _) = pool.take(BLOCK / 4, 1, 0, make).unwrap();
        let look = pool.account(0);
        assert_eq!(look.held, BLOCK * 7 / 8);
        assert_eq!(look.largest_free, BLOCK / 8);
        assert_eq!(look.bands, [0, 0, 0, 1]);

        pool.give(half, 0);
        pool.give(small, 0);
        let look = pool.account(0);
        assert_eq!(look.resting, BLOCK / 2);
        assert_eq!(look.held, BLOCK * 3 / 8);
        assert_eq!(look.textures, 0);
        assert_eq!(look.bands, [0, 0, 0, 1]);
        pool.reclaim(REST - 1, 4, true);
        assert_eq!(pool.account(0).resting, BLOCK / 2);

        pool.reclaim(REST, 4, true);
        let look = pool.account(0);
        assert_eq!(look.resting, 0);
        assert_eq!(look.held, BLOCK * 3 / 8);
        assert_eq!(look.largest_free, BLOCK * 3 / 8);
        assert_eq!(look.blocks, 1);
        assert_eq!(look.empty, 0);
        assert_eq!(look.bands, [0, 1, 0, 0]);

        pool.give(mid, REST);
        pool.give(tail, REST);
        pool.reclaim(2 * REST, 2, true);
        let look = pool.account(0);
        assert_eq!(look.blocks, 1);
        assert_eq!(look.empty, 1);
        assert_eq!(look.held, 0);
        assert_eq!(look.resting, 0);
        assert_eq!(look.textures, 0);
        assert_eq!(look.largest_free, BLOCK);
        assert_eq!(look.bands, [1, 0, 0, 0]);
        pool.reclaim(2 * REST, 0, true);
        let look = pool.account(0);
        assert_eq!(look.blocks, 0);
        assert_eq!(look.empty, 0);
        assert_eq!(look.held, 0);
        assert_eq!(look.largest_free, 0);
    }

    /// Packed, a piece goes to the fullest block with room for it, small
    /// pieces and large ones to blocks of their own; an empty block takes
    /// either.
    #[test]
    fn packed_a_piece_goes_to_the_fullest_block_of_its_kind() {
        let mut pool: Pool<usize> = Pool::new(8);
        let made = std::cell::Cell::new(0usize);
        let make = || {
            made.set(made.get() + 1);
            Some(made.get())
        };
        let take = |pool: &mut Pool<usize>, len: u64, now: u64| pool.take_ripe(len, 4096, now, Ripe::ALL, true, make).unwrap().0;
        // Block 0 full, block 1 three quarters, block 2 half.
        let a = take(&mut pool, BLOCK / 2, 0);
        assert_eq!([a.block, take(&mut pool, BLOCK / 2, 0).block], [0, 0]);
        assert_eq!([take(&mut pool, BLOCK / 4, 0).block, take(&mut pool, BLOCK / 2, 0).block], [1, 1]);
        assert_eq!(take(&mut pool, BLOCK / 2, 0).block, 2);
        // Block 0 is half empty again: the first with room, not the fullest.
        pool.give(a, 0);
        assert_eq!(take(&mut pool, BLOCK / 4, REST).block, 1);
        // Of two as full, the first.
        assert_eq!(take(&mut pool, BLOCK / 4, REST).block, 0);
        // Small pieces: a block of their own, though blocks 0 and 2 have room.
        let small = [take(&mut pool, 4096, REST), take(&mut pool, SMALL_PIECE, REST)];
        assert_eq!([small[0].block, small[1].block, made.get()], [3, 3, 4]);
        // And no large piece beside them: a new block.
        assert_eq!(take(&mut pool, BLOCK / 2, REST).block, 2);
        assert_eq!([take(&mut pool, BLOCK / 2, REST).block, made.get()], [4, 5]);
        // Empty, the small pieces' block takes large ones.
        for piece in small {
            pool.give(piece, REST);
        }
        assert_eq!([take(&mut pool, BLOCK / 2, 2 * REST).block, take(&mut pool, BLOCK / 2, 2 * REST).block], [4, 3]);
        assert_eq!(made.get(), 5);
    }

    /// The blocks chosen to empty are those all of whose pieces may go,
    /// the fewest bytes first; nothing new is placed in one, and empty it
    /// is given back whatever the spare blocks.
    #[test]
    fn a_block_chosen_to_empty_takes_nothing_new_and_is_given_back() {
        let mut pool: Pool<u32> = Pool::new(SPARE_KEPT);
        let take = |pool: &mut Pool<u32>, len: u64, now: u64| pool.take(len, 1, now, || Some(1)).unwrap().0;
        // Three full blocks; then block 0 holds two pieces of a sixteenth,
        // block 1 two of a quarter, block 2 two halves.
        let fill0 = take(&mut pool, BLOCK / 2, 0);
        let small = [take(&mut pool, BLOCK / 16, 0), take(&mut pool, BLOCK / 16, 0)];
        let fill0b = take(&mut pool, BLOCK * 3 / 8, 0);
        let fill1 = take(&mut pool, BLOCK / 2, 0);
        let quarter = [take(&mut pool, BLOCK / 4, 0), take(&mut pool, BLOCK / 4, 0)];
        let half = [take(&mut pool, BLOCK / 2, 0), take(&mut pool, BLOCK / 2, 0)];
        assert_eq!([small[1].block, fill0b.block, quarter[1].block, half[0].block, half[1].block], [0, 0, 1, 2, 2]);
        for piece in [fill0, fill0b, fill1] {
            pool.give(piece, 0);
        }
        // One of block 1's textures is in use: only a quarter of it may go.
        let idle = BTreeMap::from([(0, BLOCK / 8), (1, BLOCK / 4), (2, BLOCK)]);
        // Nothing to give back: none chosen.
        assert_eq!(pool.empty_out(0, &idle, 0), []);
        // One block's worth: the one with the fewest bytes (pieces that
        // rest are no one's any more).
        assert_eq!(pool.empty_out(1, &idle, 0), [0]);
        // Three more blocks' worth: one is on its way, block 1 cannot be
        // emptied, so block 2 alone.
        assert_eq!(pool.empty_out(4 * BLOCK, &idle, 0), [2]);
        assert_eq!(pool.empty_out(4 * BLOCK, &idle, 0), []);
        assert_eq!(pool.account(0).draining, 2);
        // Nothing new goes into block 0, seven eighths free.
        let later = take(&mut pool, BLOCK / 2, REST);
        assert_eq!(later.block, 1);
        // Its textures let go and rested: it is given back, with 64 spare
        // blocks allowed.
        for piece in small {
            pool.give(piece, REST);
        }
        pool.reclaim(2 * REST, SPARE_KEPT, false);
        let look = pool.account(2 * REST);
        assert_eq!([look.blocks, look.returned, look.emptied, look.draining], [2, 1, 1, 1]);
        assert_eq!(
            pack_line(true, &look, [3, 2, BLOCK / 8, 4500]),
            "pool pack: VICTORY_POOL_PACK on now, blocks held 2, of those not empty the emptiest holds 64 MB and the median one 64 MB, \
             blocks given back 1 since the start (1 of them emptied for the budget), being emptied now 1, \
             textures let go to empty blocks 2 (8 MB, counted with those let go for the budget) in 3 looks of 1.50 ms"
        );
        // Block 2 did not empty in time (one of its textures came into
        // use): pieces go into it again.
        pool.give(half[0], REST);
        assert!(pool.take(BLOCK / 2, 1, DRAIN_MS - 1, || None).is_none());
        assert_eq!(take(&mut pool, BLOCK / 2, DRAIN_MS).block, 2);
        // A block that empties by itself is a spare one, kept or not by
        // their number.
        for piece in [quarter[0], quarter[1], later] {
            pool.give(piece, DRAIN_MS);
        }
        pool.reclaim(DRAIN_MS + REST, 0, true);
        let look = pool.account(DRAIN_MS + REST);
        assert_eq!([look.blocks, look.returned, look.emptied], [1, 2, 1]);
    }

    #[test]
    fn blocks_come_back_from_95_percent_of_the_budget_down_to_90() {
        assert!(!over_line(1000, 949));
        assert!(over_line(1000, 950));
        assert!(!over_line(0, 950));
        // Session 0152: 9.5 GB of a budget of 9.1.
        let mb = |n: u64| n << 20;
        assert_eq!(to_return(Some([mb(9100), mb(9500), 0, 0]), false), mb(9500) - mb(9100) / 100 * 90);
        assert_eq!(to_return(Some([mb(9100), mb(8600), 0, 0]), false), 0);
        // A block the system refused: one comes back.
        assert_eq!(to_return(Some([mb(9100), mb(8600), 0, 0]), true), BLOCK);
        assert_eq!(to_return(None, true), BLOCK);
        assert_eq!(to_return(None, false), 0);
    }
}

#[cfg(test)]
mod rest_tests {
    use super::*;

    /// The staging pool's own rest: unset it is the textures' rest; set
    /// shorter, a piece given back is used again that much sooner.
    #[test]
    fn a_pool_s_own_rest_lets_a_piece_back_sooner() {
        assert_eq!(rest_ms(None), REST);
        assert_eq!(rest_ms(Some("soon")), REST);
        assert_eq!(rest_ms(Some(" 250 ")), 250);
        let mut pool: Pool<u32> = Pool::new(STAGING_KEPT);
        assert_eq!(pool.rest, REST);
        pool.rest = 250;
        let make = || Some(7u32);
        let (a, _, _) = pool.take(BLOCK / 2, 512, 0, make).unwrap();
        let (full, _, _) = pool.take(BLOCK / 2, 512, 0, make).unwrap();
        assert_eq!((a.block, full.block), (0, 0));
        pool.give(a, 100);
        // Still resting: a second block is made.
        let (b, _, _) = pool.take(BLOCK / 2, 512, 100 + 249, make).unwrap();
        assert_eq!(b.block, 1);
        // Rested: the first block's memory is taken again.
        let (c, _, again) = pool.take(BLOCK / 2, 512, 100 + 250, make).unwrap();
        assert_eq!((c.block, again), (0, true));
    }
}
