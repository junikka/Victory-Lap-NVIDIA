use super::*;
use std::sync::atomic::AtomicU64;
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_WRITE, MEMORY_BASIC_INFORMATION, MapViewOfFile, PAGE_NOACCESS, PAGE_READONLY,
    PAGE_READWRITE, VirtualQuery,
};

/// What the card holds of each test copy (by the copy's address), in place
/// of a host: [`fetch`] takes its bytes from here.
static CARD: Mutex<Option<HashMap<usize, Vec<u8>>>> = Mutex::new(None);
/// Fetches made.
static FETCHES: AtomicU64 = AtomicU64::new(0);
/// Milliseconds a fetch takes.
static SLOW: AtomicU64 = AtomicU64::new(0);

pub(super) fn card(pieces: &[(Arc<Shadow>, usize, usize)]) -> Option<Vec<u8>> {
    FETCHES.fetch_add(1, Ordering::SeqCst);
    std::thread::sleep(std::time::Duration::from_millis(SLOW.load(Ordering::SeqCst)));
    let card = CARD.lock().unwrap();
    let card = card.as_ref().expect("a test's card");
    let mut bytes = Vec::new();
    for (copy, lo, hi) in pieces {
        bytes.extend_from_slice(&card[&(Arc::as_ptr(copy) as usize)][*lo..*hi]);
    }
    Some(bytes)
}

fn fetches() -> u64 {
    FETCHES.load(Ordering::SeqCst)
}

/// View memory as the runtime makes the title's: `views` views of one
/// section of `len` bytes, noted as it notes them, holding `fill`. Kept:
/// the journal knows its addresses.
fn title(len: usize, views: usize, fill: u8) -> Vec<usize> {
    // SAFETY: a pagefile section of our own and views of it.
    unsafe {
        let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, len as u32, std::ptr::null());
        assert!(!section.is_null());
        crate::inplace::title_memory_section(section as usize, len as u64, true);
        let at: Vec<usize> = (0..views)
            .map(|_| {
                let at = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize;
                assert_ne!(at, 0);
                crate::title_memory_remapped(at, len);
                crate::inplace::title_memory_view(at, len, section as usize, 0);
                at
            })
            .collect();
        std::ptr::write_bytes(at[0] as *mut u8, fill, len);
        at
    }
}

/// A copy in the card's memory of title memory `[at, at + len)`, holding
/// what is there on both sides; and where its CPU's side is.
fn copy(at: usize, len: usize) -> (Arc<Shadow>, usize) {
    // SAFETY: the view.
    let there = unsafe { std::slice::from_raw_parts(at as *const u8, len) }.to_vec();
    let side: &'static mut [u8] = Box::leak(there.clone().into_boxed_slice());
    let copy = Shadow::unbacked(at as u64, side.as_mut_ptr(), len as u64, true);
    CARD.lock().unwrap().get_or_insert_default().insert(Arc::as_ptr(&copy) as usize, there);
    (copy, side.as_ptr() as usize)
}

/// Shaders write dword `value` at offsets `at` of `copy`, on the card.
fn shade(copy: &Arc<Shadow>, at: &[usize], value: u32) {
    let mut card = CARD.lock().unwrap();
    let bytes = card.as_mut().unwrap().get_mut(&(Arc::as_ptr(copy) as usize)).unwrap();
    for &at in at {
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
}

/// As a refresh leaves view memory: guarded, nothing written.
fn guard(at: usize, len: usize) {
    assert!(crate::watch::writes(at, at + len, true).is_some());
    assert!(crate::watch::writes(at, at + len, true).is_some());
}

fn protection(at: usize) -> u32 {
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    assert_ne!(unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) }, 0);
    info.Protect
}

/// The title reads a dword.
fn read(at: usize) -> u32 {
    // SAFETY: title-like memory of the test's; a held block faults and is
    // landed by the handler.
    unsafe { std::ptr::read_volatile(at as *const u32) }
}

/// A piece no flight writes again and the title never reads is fetched once
/// only when it has been idle; a piece written again meanwhile is not
/// fetched for the second flight either; and the title's read gets what the
/// shaders wrote, both flights', with one fetch.
#[test]
fn a_piece_stays_on_the_card_until_the_title_reads_it() {
    assert!(crate::safecopy::on());
    let at = title(2 * BLOCK, 1, 0x11)[0];
    let (copy, side) = copy(at, 2 * BLOCK);
    guard(at, 2 * BLOCK);
    let asked = crate::watch::writes(at, at + BLOCK, true).expect("a view").0;
    let before = fetches();
    shade(&copy, &[0x100, 0x1ffc], 0xaaaa_0001);
    assert!(defer(&copy, 0x100, 0x2000, 1));
    assert!(active());
    assert_eq!(protection(at), PAGE_NOACCESS);
    assert_eq!(protection(at + BLOCK), PAGE_READONLY, "the block beside it is as it was");
    assert_eq!(held(&copy, at, at + 2 * BLOCK), vec![(at + 0x100, at + 0x2000)]);
    // A second flight: the same piece again, and another in the block.
    shade(&copy, &[0x100, 0x3000], 0xaaaa_0002);
    assert!(defer(&copy, 0x100, 0x2000, 2));
    assert!(defer(&copy, 0x3000, 0x3800, 2));
    assert_eq!(held(&copy, at, at + BLOCK), vec![(at + 0x100, at + 0x2000), (at + 0x3000, at + 0x3800)]);
    assert_eq!(held(&copy, at + BLOCK, at + 2 * BLOCK), vec![]);
    assert_eq!(fetches(), before, "nothing came over the bus");
    // The title looks.
    assert_eq!(read(at + 0x100), 0xaaaa_0002);
    assert_eq!(fetches(), before + 1);
    assert_eq!(read(at + 0x1ffc), 0xaaaa_0001);
    assert_eq!(read(at + 0x3000), 0xaaaa_0002);
    assert_eq!(read(at + 0x104), 0x1111_1111, "what no shader wrote is the title's");
    assert_eq!(fetches(), before + 1);
    assert_eq!(protection(at), PAGE_READONLY, "guarded as before");
    assert!(held(&copy, at, at + 2 * BLOCK).is_empty());
    assert!(!crate::watch::lazy_any(at, 2 * BLOCK));
    // The CPU's side took what the card wrote, as a landing gives it.
    // SAFETY: the copy's side, which nothing writes now.
    assert_eq!(unsafe { std::ptr::read_unaligned((side + 0x3000) as *const u32) }, 0xaaaa_0002);
    // The journal has the landing: a copy that took the block before takes
    // it again.
    let (_, written) = crate::watch::writes(at, at + BLOCK, true).expect("a view");
    assert!(written.iter().any(|&(page, noted)| page == at / BLOCK && noted > asked));
}

/// The race the ticket asks about: threads of the title's read a held block
/// while its bytes are on their way. None reads what was there before the
/// shaders wrote: each faults, one lands the block and the others wait for
/// that landing. A read of ours at the same time does not wait: it takes
/// the title's bytes as they are, as a read just ahead of a flight's
/// landing does, and the block's landing is noted for it to take again.
#[test]
fn a_read_that_races_the_landing_waits_for_it() {
    let at = title(BLOCK, 1, 0x22)[0];
    let (copy, _) = copy(at, BLOCK);
    guard(at, BLOCK);
    shade(&copy, &[0x40, 0x8000], 0xbbbb_0001);
    assert!(defer(&copy, 0, BLOCK, 1));
    let before = fetches();
    SLOW.store(300, Ordering::SeqCst);
    let began = std::time::Instant::now();
    std::thread::scope(|s| {
        let readers: Vec<_> = (0..6)
            .map(|i| {
                s.spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(10 * i));
                    (read(at + 0x40), read(at + 0x8000), began.elapsed())
                })
            })
            .collect();
        // One of ours, while the fetch is under way.
        let ours = s.spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(40));
            let mut bytes = [0u8; 8];
            let read = crate::host::read_into((at + 0x40) as *const u8, &mut bytes);
            (read, bytes, began.elapsed())
        });
        for reader in readers {
            let (one, two, took) = reader.join().unwrap();
            assert_eq!((one, two), (0xbbbb_0001, 0xbbbb_0001), "a title thread read what was there before");
            assert!(took.as_millis() >= 300, "it waited for the landing");
        }
        let (read, bytes, took) = ours.join().unwrap();
        assert!(read);
        assert_eq!(bytes, [0x22; 8], "our read took the title's bytes as they were");
        assert!(took.as_millis() < 300, "and did not wait");
    });
    SLOW.store(0, Ordering::SeqCst);
    assert_eq!(fetches(), before + 1, "one fetch for all of them");
    assert!(!active());
    assert_eq!(protection(at), PAGE_READONLY);
}

/// A write of the title's to a held block: what the shaders wrote lands
/// first, then the write meets the guard as any other, and both are there.
#[test]
fn a_write_of_the_title_lands_first_and_then_meets_the_guard() {
    let at = title(BLOCK, 1, 0x33)[0];
    let (copy, _) = copy(at, BLOCK);
    guard(at, BLOCK);
    shade(&copy, &[0x200], 0xcccc_0001);
    assert!(defer(&copy, 0x200, 0x204, 1));
    let before = fetches();
    // SAFETY: the test's view; the handler lands the block and opens it.
    unsafe { std::ptr::write_volatile((at + 0x208) as *mut u32, 0x7777_7777) };
    assert_eq!(fetches(), before + 1);
    assert_eq!(read(at + 0x200), 0xcccc_0001);
    assert_eq!(read(at + 0x208), 0x7777_7777);
    assert_eq!(protection(at), PAGE_READWRITE, "the write opened the guard");
    // Not guarded: fetched at once, as before.
    shade(&copy, &[0x200], 0xcccc_0002);
    assert!(!defer(&copy, 0x200, 0x204, 2));
    assert!(!active());
}

/// One page at two addresses: both are held, a read through either lands
/// the block, and both are as the guards left them afterwards.
#[test]
fn every_view_of_a_page_is_held() {
    let views = title(BLOCK, 2, 0x44);
    let (one, two) = (views[0], views[1]);
    let (copy, _) = copy(one, BLOCK);
    guard(one, BLOCK);
    assert_eq!(protection(two), PAGE_READONLY, "the guard covers the page's other view");
    shade(&copy, &[0x10], 0xdddd_0001);
    assert!(defer(&copy, 0, 0x1000, 1));
    assert_eq!((protection(one), protection(two)), (PAGE_NOACCESS, PAGE_NOACCESS));
    let before = fetches();
    assert_eq!(read(two + 0x10), 0xdddd_0001);
    assert_eq!(fetches(), before + 1);
    assert_eq!(read(one + 0x10), 0xdddd_0001);
    assert_eq!((protection(one), protection(two)), (PAGE_READONLY, PAGE_READONLY));
    assert!(!active());
}

/// A block the title reads in frame after frame is fetched at once again,
/// and held anew when it has not been read for a while.
#[test]
fn a_block_the_title_keeps_reading_is_fetched_at_once() {
    let at = title(BLOCK, 1, 0x55)[0];
    let (copy, _) = copy(at, BLOCK);
    guard(at, BLOCK);
    for round in 0..promote() {
        shade(&copy, &[0], 0xeeee_0000 + round);
        assert!(defer(&copy, 0, 0x100, u64::from(round)), "round {round}");
        assert_eq!(read(at), 0xeeee_0000 + round);
        crate::frame::test_advance();
    }
    assert!(!defer(&copy, 0, 0x100, 9), "hot");
    assert!(!active());
    for _ in 0..=DECAY {
        crate::frame::test_advance();
    }
    assert!(defer(&copy, 0, 0x100, 10), "not read for a while");
    land_range(at, BLOCK);
    assert!(!active());
}

/// What else lands a held piece: a kernel write into the block (which then
/// is not held again while it lasts), another copy's piece in the block, a
/// read of ours through our own view, and frames without a flight.
#[test]
fn writes_of_ours_other_copies_and_idle_frames_land_a_held_piece() {
    let at = title(BLOCK, 1, 0x66)[0];
    let (other, _) = copy(at, BLOCK);
    let (copy, _) = copy(at, BLOCK);
    guard(at, BLOCK);
    // The kernel is about to write there.
    shade(&copy, &[0x80], 0xffff_0001);
    assert!(defer(&copy, 0x80, 0x84, 1));
    let before = fetches();
    let writing = crate::watch::kernel_writing(at + 0x1000, 16);
    assert_eq!(fetches(), before + 1);
    assert_eq!(protection(at), PAGE_READWRITE);
    assert_eq!(read(at + 0x80), 0xffff_0001);
    guard(at, BLOCK);
    assert!(!defer(&copy, 0x80, 0x84, 2), "left alone while the kernel writes");
    drop(writing);
    // Another copy lands a piece in the block.
    guard(at, BLOCK);
    shade(&copy, &[0x80], 0xffff_0002);
    assert!(defer(&copy, 0x80, 0x84, 3));
    assert!(!defer(&other, 0x80, 0x84, 3), "one copy to a block");
    assert_eq!(fetches(), before + 1);
    tend();
    assert_eq!(fetches(), before + 2);
    assert!(!active());
    assert_eq!(read(at + 0x80), 0xffff_0002);
    // A read of ours: the title's bytes as they are, and the block wanted.
    for _ in 0..=DECAY {
        crate::frame::test_advance();
    }
    shade(&copy, &[0x80], 0xffff_0003);
    assert!(defer(&copy, 0x80, 0x84, 4));
    let mut bytes = [0u8; 4];
    assert!(crate::host::read_into((at + 0x80) as *const u8, &mut bytes));
    assert_eq!(u32::from_le_bytes(bytes), 0xffff_0002);
    assert_eq!(fetches(), before + 2);
    tend();
    assert_eq!(fetches(), before + 3);
    assert_eq!(read(at + 0x80), 0xffff_0003);
    // Idle.
    for _ in 0..=DECAY {
        crate::frame::test_advance();
    }
    shade(&copy, &[0x80], 0xffff_0004);
    assert!(defer(&copy, 0x80, 0x84, 5));
    tend();
    assert!(active(), "written this frame");
    for _ in 0..idle() {
        crate::frame::test_advance();
    }
    tend();
    assert_eq!(fetches(), before + 4);
    assert!(!active());
    assert_eq!(read(at + 0x80), 0xffff_0004);
}

/// A change of what the addresses show lands what is held of them before
/// the views go.
#[test]
fn a_change_of_mapping_lands_what_is_held() {
    let at = title(BLOCK, 1, 0x77)[0];
    let (copy, _) = copy(at, BLOCK);
    guard(at, BLOCK);
    shade(&copy, &[0xfff0], 0x1234_5678);
    assert!(defer(&copy, 0xff00, BLOCK, 1));
    let before = fetches();
    {
        let _changing = crate::watch::title_memory_remapping();
        crate::watch::title_memory_unmapping(at, BLOCK);
        assert_eq!(fetches(), before + 1);
        assert!(!active());
        assert_eq!(protection(at), PAGE_READONLY, "not unreadable any more");
    }
    crate::title_memory_remapped(at, BLOCK);
    assert_eq!(read(at + 0xfff0), 0x1234_5678);
}
