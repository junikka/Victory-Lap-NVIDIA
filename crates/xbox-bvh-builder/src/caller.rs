//! Caller return address, captured without a stack walk.
//!
//! Under XCL the game image can be mapped without OS unwind data, so
//! `RtlCaptureStackBackTrace` stops before the caller. On entry to an
//! `extern "system"` export, `[rsp]` is that return address. A naked thunk
//! stores it in a thread-local and jumps to the Rust body.

use std::cell::Cell;

thread_local! {
    static CALLER: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn caller_return_address() -> usize {
    CALLER.with(Cell::get)
}

/// `addr` is the return address read from `[rsp]` by the entry thunk.
#[inline(never)]
pub(crate) extern "system" fn store_caller(addr: usize) {
    CALLER.with(|slot| slot.set(addr));
}

macro_rules! asm_entry {
    ($export:literal => $body:ident) => {
        ::std::arch::global_asm!(
            concat!(
                ".globl \"",
                $export,
                "\"\n\"",
                $export,
                "\":\n",
                "mov rax, qword ptr [rsp]\n",
                "mov qword ptr [rsp + 8], rcx\n",
                "mov qword ptr [rsp + 16], rdx\n",
                "mov qword ptr [rsp + 24], r8\n",
                "mov qword ptr [rsp + 32], r9\n",
                "mov rcx, rax\n",
                "sub rsp, 40\n",
                "call {store}\n",
                "add rsp, 40\n",
                "mov rcx, qword ptr [rsp + 8]\n",
                "mov rdx, qword ptr [rsp + 16]\n",
                "mov r8, qword ptr [rsp + 24]\n",
                "mov r9, qword ptr [rsp + 32]\n",
                "jmp {body}\n"
            ),
            store = sym $crate::caller::store_caller,
            body = sym $body,
        );
    };
}
