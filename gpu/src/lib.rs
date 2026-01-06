#![allow(
    internal_features,
    improper_ctypes,
    improper_ctypes_definitions,
    improper_gpu_kernel_arg
)]
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

    #[link_name = "llvm.amdgcn.image.load.2d.i32.i32"]
    pub fn image_load(dmask: u32, x: u32, y: u32, img: ImageDesc, tfe: u32, aux: u32) -> u32;

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

fn sample(img: ImageDesc, sampler: SamplerDesc, x: i32, y: i32, width: u32, height: u32) -> bool {
    let x_f = (x as f32 + 0.5) / width as f32;
    let y_f = (y as f32 + 0.5) / height as f32;
    unsafe { image_sample(1, x_f, y_f, img, sampler, false, 0, 0) > 128 }
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
    paused: u32,
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
        let val = sample(old_content_desc, sampler, x as i32, y as i32, width, height);
        if x == 0 && y == 0 {
            // println!("Got {val}");
        }
        let mut sum = 0;
        for i in -1..=1 {
            for j in -1..=1 {
                if i == 0 && j == 0 {
                    continue;
                }

                if sample(
                    old_content_desc,
                    sampler,
                    x as i32 + i,
                    y as i32 + j,
                    width,
                    height,
                ) {
                    sum += 1;
                }
            }
        }

        let new_val = if paused == 1 {
            val
        } else if sum == 3 {
            // Becomes alive
            true
        } else if sum != 2 {
            // Dies
            false
        } else {
            val
        };
        let new_val_f = if new_val { 1.0 } else { 0.0 };

        image_store(new_val_f, 1, x, y, new_content_desc, 0, 0);

        // Write screen image
        let x_screen = ((x as f32 / width as f32) * screen_width as f32) as u32;
        let next_x_screen = (((x + 1) as f32 / width as f32) * screen_width as f32) as u32;
        let y_screen = ((y as f32 / height as f32) * screen_height as f32) as u32;
        let next_y_screen = (((y + 1) as f32 / height as f32) * screen_height as f32) as u32;

        /*if x == 0 && y == 0 {
            println!(
                "Write {val} to {x_screen}..{next_x_screen} x {y_screen}..{next_y_screen} ({width}x{height} to {screen_width}x{screen_height})"
            );
        }*/
        // TODO Color based on direction and age
        let col = if new_val {
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

    unsafe { image_store(0.0, 1, x, y, img, 0, 0) };
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn set(img: ImageDesc, x: u32, y: u32, val: f32) {
    if workitem_id_x() != 0 || workitem_id_y() != 0 {
        return;
    }

    // println!("GPU setting {x},{y} to {val}");
    unsafe { image_store(val, 1, x, y, img, 0, 0) };
}

#[allow(unused_variables, clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn check(
    img: ImageDesc,
    sampler: SamplerDesc,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    val: f32,
) {
    /*if workitem_id_x() != 0 || workitem_id_y() != 0 {
        return;
    }

    let is_val = sample(img, sampler, x as i32, y as i32, width, height);
    let is_val2 = unsafe { image_load(1, x, y, img, 0, 0) };
    if is_val != (val == 1.0) {
        println!("Check sample: {x},{y} is {is_val} but expected {val}");
    }
    if is_val2 as f32 / 255.0 != val {
        println!("Check load: {x},{y} is {is_val2} but expected {val}");
    }*/
}
