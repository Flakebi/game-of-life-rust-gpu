#![allow(internal_features, improper_ctypes)]
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

use amdgpu_device_libs::prelude::*;

#[derive(Clone, Copy)]
#[repr(simd)]
pub struct ImageDesc([u32; 8]);

#[derive(Clone, Copy)]
#[repr(simd)]
pub struct SamplerDesc([u32; 4]);

#[derive(Clone, Copy)]
#[repr(simd)]
pub struct RGBA([f32; 4]);

unsafe extern "C" {
    safe fn __amdgpu_util_kernarg_segment_ptr() -> *const core::ffi::c_void;
}

unsafe extern "unadjusted" {
    #[link_name = "llvm.amdgcn.image.sample.lz.2d.f32.i32"]
    pub fn image_sample(
        dmask: u32,
        x: f32,
        y: f32,
        img: ImageDesc,
        samp: SamplerDesc,
        unorm: bool,
        tfe: u32,
        aux: u32,
    ) -> u32;

    #[link_name = "llvm.amdgcn.image.store.2d.f32.i32"]
    pub fn image_store(data: f32, dmask: u32, x: u32, y: u32, img: ImageDesc, tfe: u32, aux: u32);

    #[link_name = "llvm.amdgcn.image.store.2d.v4f32.i32"]
    pub fn image_store_color(
        data: RGBA,
        dmask: u32,
        x: u32,
        y: u32,
        img: ImageDesc,
        tfe: u32,
        aux: u32,
    );
}

#[allow(dead_code)]
struct KernelArgs {
    old_content_desc: ImageDesc,
    new_content_desc: ImageDesc,
    screen_desc: ImageDesc,
    sampler: SamplerDesc,
    width: u32,
    height: u32,
    screen_width: u32,
    screen_height: u32,
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kernel(
    /*_: u32,
    _: u32,
    _: u32,
    _: u32,
    _: u32,
    _: u32,
    _: u32,
    _: u32,

    _: u32,
    _: u32,
    _: u32,
    _: u32,
    _: u32,
    _: u32,
    _: u32,
    _: u32,

    _: u32,
    _: u32,
    _: u32,
    _: u32,
    _: u32,
    _: u32,
    _: u32,
    _: u32,

    _: u32,
    _: u32,
    _: u32,
    _: u32,

    _: u32,
    _: u32,*/
    old_content_desc: ImageDesc,
    new_content_desc: ImageDesc,
    screen_desc: ImageDesc,
    sampler: SamplerDesc,
    width: u32,
    height: u32,
    screen_width: u32,
    screen_height: u32,
) {
    // let args: &KernelArgs = unsafe { &*(__amdgpu_util_kernarg_segment_ptr() as *const _) };
    let dispatch = dispatch_ptr();

    // Compute global coordinates
    let x = workgroup_id_x() * dispatch.workgroup_size_x as u32 + workitem_id_x();
    let y = workgroup_id_y() * dispatch.workgroup_size_y as u32 + workitem_id_y();

    if x == 0 && y == 0 {
        // println!("Is running with {}x{}", screen_width, screen_height);
    }

    // if x >= args.width || y >= args.height {
    if x >= width || y >= height {
        return;
    }

    unsafe {
        let x_f = x as f32 / width as f32;
        let y_f = y as f32 / height as f32;
        let val = image_sample(1, x_f, y_f, old_content_desc, sampler, false, 0, 0);
        if x == 0 && y == 0 {
            // println!("Got {val}");
        }
        let mut sum = 0;
        for i in -1..=1 {
            for j in -1..=1 {
                if i == 0 && j == 0 {
                    continue;
                }

                if image_sample(
                    1,
                    (x as f32 + i as f32) / width as f32,
                    (y as f32 + j as f32) / height as f32,
                    old_content_desc,
                    sampler,
                    false,
                    0,
                    0,
                ) > 128
                {
                    sum += 1;
                }
            }
        }

        let new_val = if sum == 3 {
            // Becomes alive
            1.0
        } else if sum != 2 {
            // Dies
            0.0
        } else {
            if val > 128 { 1.0 } else { 0.0 }
        };

        image_store(new_val, 1, x, y, new_content_desc, 0, 0);

        // Write screen image
        let x_screen = (x_f * screen_width as f32) as u32;
        let next_x_screen = (((x + 1) as f32 / width as f32) * screen_width as f32) as u32;
        let y_screen = (y_f * screen_height as f32) as u32;
        let next_y_screen = (((y + 1) as f32 / height as f32) * screen_height as f32) as u32;

        /*if x == 0 && y == 0 {
            println!(
                "Write {val} to {x_screen}..{next_x_screen} x {y_screen}..{next_y_screen} ({width}x{height} to {screen_width}x{screen_height})"
            );
        }*/
        let col = if new_val > 0.5 {
            RGBA([0.2, 0.7, 1.0, 1.0])
        } else {
            RGBA([0.0, 0.0, 0.0, 1.0])
        };
        for i in x_screen..next_x_screen {
            for j in y_screen..next_y_screen {
                image_store_color(col, 0xf, i, j, screen_desc, 0, 0);
            }
        }
    }
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn clear(img: ImageDesc, width: u32, height: u32) {
    let dispatch = dispatch_ptr();

    // Compute global coordinates
    let x = workgroup_id_x() * dispatch.workgroup_size_x as u32 + workitem_id_x();
    let y = workgroup_id_y() * dispatch.workgroup_size_y as u32 + workitem_id_y();

    if x >= width || y >= height {
        return;
    }

    let val = if y % 2 == 0 && x % 4 != 1 { 1.0 } else { 0.0 };

    unsafe { image_store(val, 1, x, y, img, 0, 0) };
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn set(img: ImageDesc, x: u32, y: u32, val: f32) {
    if workitem_id_x() != 0 || workitem_id_y() != 0 {
        return;
    }

    unsafe { image_store(val, 1, x, y, img, 0, 0) };
}
