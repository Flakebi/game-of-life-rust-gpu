#![allow(internal_features)]
#![feature(
    abi_gpu_kernel,
    core_intrinsics,
    link_llvm_intrinsics,
    repr_simd,
    simd_ffi,
    abi_unadjusted
)]
#![no_std]

extern crate alloc;

use core::sync::atomic::AtomicU32;
use core::sync::atomic::Ordering;

use amdgpu_device_libs::prelude::*;

#[derive(Clone, Copy)]
#[repr(simd)]
pub struct BufDesc([u32; 4]);

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    /// Returns the x coordinate of the workitem index within the workgroup.
    #[link_name = "llvm.amdgcn.struct.buffer.load"]
    pub fn struct_buffer_load_u8(
        buf: BufDesc,
        index: u32,
        offset: u32,
        soffset: u32,
        cache_policy: u32,
    ) -> u8;
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kernel(
    input: *const u8,
    output: *mut u32,
    width: u32,
    height: u32,
) {
    let dispatch = dispatch_ptr();

    // Compute global coordinates
    let x = workgroup_id_x() * dispatch.workgroup_size_x as u32 + workitem_id_x();
    let y = workgroup_id_y() * dispatch.workgroup_size_y as u32 + workitem_id_y();

    if x >= width || y >= height {
        return;
    }

    // Build a structured buffer descriptor that takes y coordinate as index and x as offset
    let mut desc = [0u32; 4];
    // addr
    desc[0] = input as usize as u32;
    // addr, stride
    desc[1] = (((input as usize) >> 32) as u32) | ((width + 1) << 16);
    // num records
    desc[2] = height;
    // dst_sel, format = 32_UINT, oob_select = 0 (check index and offset)
    desc[3] = 4 | (5 << 3) | (6 << 6) | (7 << 9) | (20 << 12) | (1 << 24);
    let desc = BufDesc(desc);
    // Read at coordinate, returns 0 if out of bounds
    let load = |x, y| unsafe { struct_buffer_load_u8(desc, y, x, 0, 0) };

    let atomic = unsafe { AtomicU32::from_ptr(output) };
    if load(x, y) == b'@' {
        let roll_num = |x, y| {
            if load(x, y) == b'@' { 1 } else { 0 }
        };
        let mut set_neighbors = 0;
        // Loop does not get unrolled for some reason
        /*for dx in -1..=1 {
            for dy in -1..=1 {
                if !(dx == 0 && dy == 0) {
                    set_neighbors += roll_num(x.wrapping_add_signed(dx), y.wrapping_add_signed(dy));
                }
            }
        }*/

        set_neighbors += roll_num(x.wrapping_add_signed(-1), y.wrapping_add_signed(-1));
        set_neighbors += roll_num(x.wrapping_add_signed(-1), y.wrapping_add_signed(0));
        set_neighbors += roll_num(x.wrapping_add_signed(-1), y.wrapping_add_signed(1));

        set_neighbors += roll_num(x.wrapping_add_signed(0), y.wrapping_add_signed(-1));
        set_neighbors += roll_num(x.wrapping_add_signed(0), y.wrapping_add_signed(1));

        set_neighbors += roll_num(x.wrapping_add_signed(1), y.wrapping_add_signed(-1));
        set_neighbors += roll_num(x.wrapping_add_signed(1), y.wrapping_add_signed(0));
        set_neighbors += roll_num(x.wrapping_add_signed(1), y.wrapping_add_signed(1));

        if set_neighbors < 4 {
            atomic.fetch_add(1, Ordering::Relaxed);
        }
    }
}
