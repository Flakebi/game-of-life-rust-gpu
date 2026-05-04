//! The GPU code to simulate game of life.
//!
//! Also contains some helper kernels.

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

/// An image descriptor
#[derive(Clone, Copy)]
#[repr(simd)]
pub struct ImageDesc([u32; 8]);

/// A sampler descriptor
#[derive(Clone, Copy)]
#[repr(simd)]
pub struct SamplerDesc([u32; 4]);

/// A color value
#[derive(Clone, Copy)]
#[repr(simd)]
pub struct RGBA([f32; 4]);

/// Declare the used LLVM intrinsics
unsafe extern "unadjusted" {
    /// Sample from an image at the coordinate using the descriptors.
    ///
    /// Coordinates are between 0 and 1.
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

    /// Store a value into the image.
    ///
    /// The value must be given as float, but can be arbitrary 32 bits, depending on the image format.
    ///
    /// Coordinates are between 0 and image size.
    #[link_name = "llvm.amdgcn.image.store.2d.f32.i32"]
    pub fn image_store(data: f32, dmask: u32, x: u32, y: u32, img: ImageDesc, tfe: u32, aux: u32);

    /// Store a value into the image.
    ///
    /// Coordinates are between 0 and image size.
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

/// Sample a value from an image.
///
/// Coordinates are between 0 and image size.
fn sample(img: ImageDesc, sampler: SamplerDesc, x: i32, y: i32, width: u32, height: u32) -> u32 {
    let x_f = (x as f32 + 0.5) / width as f32;
    let y_f = (y as f32 + 0.5) / height as f32;
    unsafe { image_sample(1, x_f, y_f, img, sampler, false, 0, 0) }
}

/// Store a value to an image.
///
/// Coordinates are between 0 and image size.
fn store(img: ImageDesc, x: u32, y: u32, val: u32) {
    let val_f = f32::from_bits(val);
    unsafe { image_store(val_f, 1, x, y, img, 0, 0) };
}

/// The kernel that simulates one step and renders the new field to the screen.
#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kernel(
    // The field we read from
    old_content_desc: ImageDesc,
    // The field we write to
    new_content_desc: ImageDesc,
    // The image we render to
    screen_desc: ImageDesc,
    // The sampler for reading the field
    sampler: SamplerDesc,
    // Field width
    width: u32,
    // Field height
    height: u32,
    // Screen image width
    screen_width: u32,
    // Screen image height
    screen_height: u32,
    // 1 if paused or 0 if not
    paused: u32,
) {
    let dispatch = dispatch_ptr();

    // Compute global coordinates
    let x = workgroup_id_x() * dispatch.workgroup_size_x as u32 + workitem_id_x();
    let y = workgroup_id_y() * dispatch.workgroup_size_y as u32 + workitem_id_y();

    // Do nothing if this thread is outside of the field
    if x >= width || y >= height {
        return;
    }

    unsafe {
        // Get old value of this cell
        let val = sample(old_content_desc, sampler, x as i32, y as i32, width, height);
        // value is saved as direction (lower 3 bits) + age (upper 5 bits),
        // if the cell is dead, age is 0
        let dir = val & 7;
        let age = val >> 3;
        const MAX_AGE: f32 = 31.0;

        // Iterate over neighboring cells and count how many are alive.
        // Try to keep log of which direction neighbors are alive in.
        let mut new_dir_x = 0;
        let mut new_dir_y = 0;
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
                ) != 0
                {
                    sum += 1;
                    new_dir_x += i;
                    new_dir_y += j;
                }
            }
        }
        new_dir_x = new_dir_x.clamp(-1, 1);
        new_dir_y = new_dir_y.clamp(-1, 1);
        let new_dir = if age != 0 {
            // Keep dir if this cell is already alive
            dir
        } else {
            (new_dir_x + 1) as u32 + ((new_dir_y + 1) << 1) as u32
        };

        let new_age = if paused == 1 {
            age
        } else if age == 0 {
            if sum == 3 {
                // Becomes alive
                1
            } else {
                // Stays dead
                0
            }
        } else if sum == 2 || sum == 3 {
            // Stays alive
            // Increment age, cap at 31
            31.min(age + 1)
        } else {
            // Dies
            0
        };

        // Store the new state into the next image
        let mut new_val = new_age << 3;
        if new_age != 0 {
            new_val |= new_dir;
        }
        store(new_content_desc, x, y, new_val);

        // Scale up to screen size
        let x_screen = ((x as f32 / width as f32) * screen_width as f32) as u32;
        let next_x_screen = (((x + 1) as f32 / width as f32) * screen_width as f32) as u32;
        let y_screen = ((y as f32 / height as f32) * screen_height as f32) as u32;
        let next_y_screen = (((y + 1) as f32 / height as f32) * screen_height as f32) as u32;

        // We choose a color based on direction.
        // There is no grand underlying scheme, it should just look pretty.
        let red = 0.7 * (new_dir >> 1) as f32 / 3.0;
        let green = 0.7 * (new_dir & 3) as f32 / 3.0;

        let col = if new_age == 0 {
            // Dead is black
            RGBA([0.0, 0.0, 0.0, 1.0])
        } else if new_age <= 5 {
            // Steep linear gradient in the beginning
            RGBA([red, green, 1.0 - (new_age - 1) as f32 / 4.0 * 0.4, 1.0])
        } else {
            // Cubic for older cells
            let mut v = new_age as f32 / MAX_AGE - 1.0;
            v = v * v;
            v = v * v;
            let age_ratio = v;
            RGBA([red, green, age_ratio * 0.5 + 0.1, 1.0])
        };

        // Write screen image, this scales up as one cell in the field image usually corresponds to
        // multiple pixels on the screen
        for i in x_screen..next_x_screen {
            for j in y_screen..next_y_screen {
                image_store_color(col, 0xf, i, j, screen_desc, 0, 0);
            }
        }
    }
}

/// Kernel to initially clear the image.
///
/// Sets everything to zero.
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

    store(img, x, y, 0);
}

/// Kernel to set a specific cell in the image to alive or dead.
#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn set(img: ImageDesc, x: u32, y: u32, val: u32) {
    // Only the first thread does the store
    if workitem_id_x() != 0 || workitem_id_y() != 0 {
        return;
    }

    store(img, x, y, val << 3);
}
