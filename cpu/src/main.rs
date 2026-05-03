//! The CPU code that sets up a window, handles input and launches the GPU kernel for the
//! simulation.

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::PathBuf;

use clap::Parser;
use hip_runtime_sys::{
    hipDeviceProp_t, hipDeviceSynchronize, hipDriverGetVersion, hipError_t, hipFunction_t,
    hipGetDevice, hipGetDeviceCount, hipGetDeviceProperties, hipInit, hipModule_t,
    hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoadData, hipModuleUnload,
    hipRuntimeGetVersion, hipSetDevice,
};
use winit::event::{
    ButtonSource, DeviceId, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::keyboard::{Key, NamedKey};

mod vk;

#[derive(Parser)]
#[command()]
struct Cli {
    /// Path to compiled GPU kernel
    #[arg(
        short,
        long,
        default_value = "target/amdgcn-amd-amdhsa/release/gpu.elf"
    )]
    file: PathBuf,

    /// Index of the device to use
    #[arg(short, long, default_value_t = 0)]
    device_index: i32,
}

fn get_str(s: &[i8]) -> &str {
    let cs =
        std::ffi::CStr::from_bytes_until_nul(unsafe { std::mem::transmute::<&[i8], &[u8]>(s) })
            .unwrap();
    cs.to_str().unwrap()
}

struct App {
    /// The current index into the images.
    /// One image is used for reading the state, the other is written to.
    /// Flips between 0 and 1 every frame.
    image_index: u32,
    /// How long to sleep between frames.
    /// Allows to change the speed of the simulation.
    sleep_ms: u64,
    /// The compiled module with the GPU kernel
    module: hipModule_t,
    /// The kernel that runs a simulation step
    simulate: hipFunction_t,
    /// The kernel that sets a cell to a specified value
    set: hipFunction_t,
    /// Pressed buttons and associated value (set or unset tile).
    /// Used for drawing on the field.
    is_pressed: HashMap<Option<DeviceId>, bool>,
}

fn main() {
    // Parse arguments
    let args = Cli::parse();

    // Get some system information from HIP
    // Adjusted from https://github.com/cjordan/hip-sys/blob/5a55ab891dec0446a6b09152c385b1c8e4e6df45/examples/hip_info.rs
    // under MIT/Apache 2.0 by Dev Null
    let result = unsafe { hipInit(0) };
    assert_eq!(result, hipError_t::hipSuccess);

    let mut driver_version: i32 = 0;
    let result = unsafe { hipDriverGetVersion(&mut driver_version) };
    assert_eq!(result, hipError_t::hipSuccess);
    println!("Driver Version: {driver_version}");

    let mut runtime_version: i32 = 0;
    let result = unsafe { hipRuntimeGetVersion(&mut runtime_version) };
    assert_eq!(result, hipError_t::hipSuccess);
    println!("Runtime Version: {runtime_version}");

    // Get devices on the system and some of their information
    let mut device_count: i32 = 0;
    let result = unsafe { hipGetDeviceCount(&mut device_count) };
    assert_eq!(result, hipError_t::hipSuccess);
    println!("Device Count: {device_count}");

    for i in 0..device_count {
        // `arch` is the gfx version for which kernels need to be compiled
        let (name, arch, device_prop) = unsafe {
            let mut device_prop: hipDeviceProp_t = std::mem::zeroed();
            let result = hipGetDeviceProperties(&mut device_prop, i);
            assert_eq!(result, hipError_t::hipSuccess);
            (
                get_str(&device_prop.name).to_string(),
                get_str(&device_prop.gcnArchName).to_string(),
                device_prop,
            )
        };
        println!("Device {i}");
        println!("  {name} ({arch}) | multi {}", device_prop.isMultiGpuBoard);
        println!(
            "  mem    | VRAM: {}GiB, shared/block: {}KiB, ",
            device_prop.totalGlobalMem / (1024 * 1024 * 1024),
            device_prop.sharedMemPerBlock / 1024
        );
        println!(
            "  thread | max/block: {}, warpSize {}, {} processors with {} max threads, max [{} {} {}]",
            device_prop.maxThreadsPerBlock,
            device_prop.warpSize,
            device_prop.multiProcessorCount,
            device_prop.maxThreadsPerMultiProcessor,
            device_prop.maxThreadsDim[0],
            device_prop.maxThreadsDim[1],
            device_prop.maxThreadsDim[2]
        );
        println!(
            "  grid   | max [{} {} {}]",
            device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]
        );
    }

    let app = vk::App::new(
        // Default window dimensions
        1000,
        800,
        // Initialization when the window is available
        Box::new(move |win| {
            unsafe {
                // Set the default device, all device-specific functions will use this one
                println!("Set device {}", args.device_index);
                let result = hipSetDevice(args.device_index);
                assert_eq!(result, hipError_t::hipSuccess);
                let mut device = 0;
                let result = hipGetDevice(&mut device);
                assert_eq!(result, hipError_t::hipSuccess);

                // Load the executable that was compiled for the GPU
                println!("Load module from {}", args.file.display());
                let module_data = std::fs::read(args.file).unwrap();
                let mut module: hipModule_t = std::ptr::null_mut();
                let result = hipModuleLoadData(&mut module, module_data.as_ptr() as *const c_void);
                assert_eq!(result, hipError_t::hipSuccess);

                // Get kernel function from loaded module
                println!("Get function kernel");
                let mut simulate: hipFunction_t = std::ptr::null_mut();
                let result =
                    hipModuleGetFunction(&mut simulate, module, b"kernel\0".as_ptr() as *const _);
                assert_eq!(result, hipError_t::hipSuccess);

                /// Argument struct for the `clear` GPU function
                #[allow(dead_code)]
                struct ClearKernelArgs {
                    image_desc: [u8; 32],
                    width: u32,
                    height: u32,
                }

                // Get kernel to clear/initialize image to all 0
                println!("Get function clear");
                let mut clear: hipFunction_t = std::ptr::null_mut();
                let result =
                    hipModuleGetFunction(&mut clear, module, b"clear\0".as_ptr() as *const _);
                assert_eq!(result, hipError_t::hipSuccess);

                // Run clear kernel for the two field images
                let swapchain = win.swapchain.as_ref().unwrap();
                for image in &swapchain.content_image_descriptors {
                    let kernel_args = &mut ClearKernelArgs {
                        image_desc: *image,
                        width: swapchain.width,
                        height: swapchain.height,
                    };
                    let mut size = std::mem::size_of_val(kernel_args);

                    #[allow(clippy::manual_dangling_ptr)]
                    let mut config = [
                        0x1 as *mut c_void,                          // Next come arguments
                        kernel_args as *mut _ as *mut c_void,        // Pointer to arguments
                        0x2 as *mut c_void,                          // Next comes size
                        std::ptr::addr_of_mut!(size) as *mut c_void, // Pointer to size of arguments
                        0x3 as *mut c_void,                          // End
                    ];

                    const WG_SIZE: u32 = 16;
                    // Launch workgroups of size 16x16x1 to fill the image
                    let wg_x = swapchain.width.div_ceil(WG_SIZE);
                    let wg_y = swapchain.height.div_ceil(WG_SIZE);
                    let result = hipModuleLaunchKernel(
                        clear,
                        wg_x,                 // Workgroup count x
                        wg_y,                 // Workgroup count y
                        1,                    // Workgroup count z
                        WG_SIZE,              // Workgroup dim x
                        WG_SIZE,              // Workgroup dim y
                        1,                    // Workgroup dim z
                        0,                    // sharedMemBytes for extern shared variables
                        std::ptr::null_mut(), // stream
                        std::ptr::null_mut(), // params (unimplemented in hip)
                        config.as_mut_ptr(),  // arguments
                    );
                    assert_eq!(result, hipError_t::hipSuccess);

                    let result = hipDeviceSynchronize();
                    assert_eq!(result, hipError_t::hipSuccess);
                }
                println!("Images cleared");

                // Get kernel to set a cell
                let mut set: hipFunction_t = std::ptr::null_mut();
                let result = hipModuleGetFunction(&mut set, module, b"set\0".as_ptr() as *const _);
                assert_eq!(result, hipError_t::hipSuccess);

                Box::new(App {
                    image_index: 0,
                    sleep_ms: 100,
                    module,
                    simulate,
                    set,
                    is_pressed: Default::default(),
                })
            }
        }),
    );
    app.run().unwrap();
}

impl App {
    /// Set or clear a cell on the field.
    ///
    /// Coordinates given in screen space are transformed to cell coordinates first.
    /// `set` specifies if the cell should be marked alive or dead.
    fn set(&mut self, screen_x: f64, screen_y: f64, set: bool, win: &vk::Vk) {
        // Screen to field coordinates
        let swapchain = win.swapchain.as_ref().unwrap();
        let x = (screen_x as f32 / swapchain.surface_resolution.width as f32
            * swapchain.width as f32) as u32;
        let y = (screen_y as f32 / swapchain.surface_resolution.height as f32
            * swapchain.height as f32) as u32;

        unsafe {
            #[allow(dead_code)]
            struct KernelArgs {
                img: [u8; 32],
                x: u32,
                y: u32,
                val: u32,
            }

            let img = win.swapchain.as_ref().unwrap().content_image_descriptors
                [self.image_index as usize];
            let kernel_args = &mut KernelArgs {
                img,
                x,
                y,
                val: if set { 1 } else { 0 },
            };
            let mut size = std::mem::size_of_val(kernel_args);

            #[allow(clippy::manual_dangling_ptr)]
            let mut config = [
                0x1 as *mut c_void,                          // Next come arguments
                kernel_args as *mut _ as *mut c_void,        // Pointer to arguments
                0x2 as *mut c_void,                          // Next comes size
                std::ptr::addr_of_mut!(size) as *mut c_void, // Pointer to size of arguments
                0x3 as *mut c_void,                          // End
            ];

            let result = hipModuleLaunchKernel(
                self.set,
                1,                    // Workgroup count x
                1,                    // Workgroup count y
                1,                    // Workgroup count z
                1,                    // Workgroup dim x
                1,                    // Workgroup dim y
                1,                    // Workgroup dim z
                0,                    // sharedMemBytes for extern shared variables
                std::ptr::null_mut(), // stream
                std::ptr::null_mut(), // params (unimplemented in hip)
                config.as_mut_ptr(),  // arguments
            );
            assert_eq!(result, hipError_t::hipSuccess);

            let result = hipDeviceSynchronize();
            assert_eq!(result, hipError_t::hipSuccess);
        }
    }

    /// Speed up the simulation by reducing the sleep time between frames.
    fn make_faster(&mut self) {
        let mut new = (self.sleep_ms as f32 * 0.9) as u64;
        if new == self.sleep_ms && new > 1 {
            new -= 1;
        }
        if new == 0 {
            new = 1;
        }
        self.sleep_ms = new;
    }

    /// Slow down the simulation by increasing the sleep time between frames.
    fn make_slower(&mut self) {
        let mut new = (self.sleep_ms as f32 * 1.1) as u64;
        if new == self.sleep_ms {
            new += 1;
        }
        self.sleep_ms = new;
    }

    /// Zoom in or out the field.
    ///
    /// The field is always fully displayed in the window, so this changes the size of the field.
    fn zoom(&mut self, factor: f32, win: &mut vk::Vk) {
        let mut new = (win.tile_size as f32 * factor) as u32;
        if new == win.tile_size {
            // If the tile size is unchanged by the factor, increase/decrease by one
            if factor > 1.0 {
                new += 1;
            } else if new > 1 {
                new -= 1;
            }
        }
        if new == 0 {
            new = 1;
        }

        win.tile_size = new;
        win.ensure_swapchain(true);
    }
}

impl vk::MyApp for App {
    /// Simulate a single step and display the field
    fn on_render(&mut self, _: u32, win: &vk::Vk) {
        unsafe {
            std::thread::sleep(std::time::Duration::from_millis(self.sleep_ms));
            let swapchain = win.swapchain.as_ref().unwrap();
            // Get image descriptors
            let screen_desc = swapchain.screen_image_descriptor;
            let old_content_desc = swapchain.content_image_descriptors[self.image_index as usize];
            let new_content_desc =
                swapchain.content_image_descriptors[(self.image_index ^ 1) as usize];
            if !win.paused {
                self.image_index ^= 1;
            }
            let sampler = win.content_image_sampler;

            #[allow(dead_code)]
            struct KernelArgs {
                old_content_desc: [u8; 32],
                new_content_desc: [u8; 32],
                screen_desc: [u8; 32],
                sampler: [u8; 16],
                width: u32,
                height: u32,
                window_width: u32,
                window_height: u32,
            }

            let kernel_args = &mut KernelArgs {
                old_content_desc,
                new_content_desc,
                screen_desc,
                sampler,
                width: swapchain.width,
                height: swapchain.height,
                window_width: swapchain.surface_resolution.width,
                window_height: swapchain.surface_resolution.height,
            };
            let mut size = std::mem::size_of_val(kernel_args);

            #[allow(clippy::manual_dangling_ptr)]
            let mut config = [
                0x1 as *mut c_void,                          // Next come arguments
                kernel_args as *mut _ as *mut c_void,        // Pointer to arguments
                0x2 as *mut c_void,                          // Next comes size
                std::ptr::addr_of_mut!(size) as *mut c_void, // Pointer to size of arguments
                0x3 as *mut c_void,                          // End
            ];

            const WG_SIZE: u32 = 16;
            // Launch workgroups of size 16x16x1
            let wg_x = swapchain.width.div_ceil(WG_SIZE);
            let wg_y = swapchain.height.div_ceil(WG_SIZE);
            let result = hipModuleLaunchKernel(
                self.simulate,
                wg_x,                 // Workgroup count x
                wg_y,                 // Workgroup count y
                1,                    // Workgroup count z
                WG_SIZE,              // Workgroup dim x
                WG_SIZE,              // Workgroup dim y
                1,                    // Workgroup dim z
                0,                    // sharedMemBytes for extern shared variables
                std::ptr::null_mut(), // stream
                std::ptr::null_mut(), // params (unimplemented in hip)
                config.as_mut_ptr(),  // arguments
            );
            assert_eq!(result, hipError_t::hipSuccess);

            let result = hipDeviceSynchronize();
            assert_eq!(result, hipError_t::hipSuccess);
        }
    }

    /// Handle mouse and keyboard events
    fn on_event(&mut self, event: WindowEvent, win: &mut vk::Vk) {
        match event {
            WindowEvent::SurfaceResized(_) => win.ensure_swapchain(true),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Character(c),
                        ..
                    },
                ..
            } if c.as_str() == " " => win.paused = !win.paused,
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Character(c),
                        ..
                    },
                ..
            } if c.as_str() == "+" || c.as_str() == "." => self.make_faster(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Named(NamedKey::ArrowRight),
                        ..
                    },
                ..
            } => self.make_faster(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Character(c),
                        ..
                    },
                ..
            } if c.as_str() == "-" || c.as_str() == "," => self.make_slower(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Named(NamedKey::ArrowLeft),
                        ..
                    },
                ..
            } => self.make_slower(),
            WindowEvent::MouseWheel { delta, .. } => {
                let factor = match delta {
                    MouseScrollDelta::LineDelta(_, d) => 1.0 + d / 10.0,
                    MouseScrollDelta::PixelDelta(d) => 1.0 + d.y as f32 / 100.0,
                };

                self.zoom(factor, win);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Named(NamedKey::ArrowUp),
                        ..
                    },
                ..
            } => self.zoom(1.1, win),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Named(NamedKey::ArrowDown),
                        ..
                    },
                ..
            } => self.zoom(0.9, win),
            WindowEvent::PointerButton {
                state: ElementState::Pressed,
                button: ButtonSource::Mouse(MouseButton::Left),
                position,
                device_id,
                ..
            } => {
                self.set(position.x, position.y, true, win);
                self.is_pressed.insert(device_id, true);
            }
            WindowEvent::PointerButton {
                state: ElementState::Pressed,
                button: ButtonSource::Mouse(MouseButton::Right),
                position,
                device_id,
                ..
            } => {
                self.set(position.x, position.y, false, win);
                self.is_pressed.insert(device_id, false);
            }
            WindowEvent::PointerButton {
                state: ElementState::Released,
                button: ButtonSource::Mouse(MouseButton::Left),
                device_id,
                ..
            } => {
                self.is_pressed.remove(&device_id);
            }
            WindowEvent::PointerButton {
                state: ElementState::Released,
                button: ButtonSource::Mouse(MouseButton::Right),
                device_id,
                ..
            } => {
                self.is_pressed.remove(&device_id);
            }
            WindowEvent::PointerMoved {
                position,
                device_id,
                ..
            } => {
                // Also apply on drag
                if position.x >= 0.0
                    && position.y >= 0.0
                // TODO Max pos
                    && let Some(set) = self.is_pressed.get(&device_id)
                {
                    self.set(position.x, position.y, *set, win);
                }
            }
            _ => {}
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            println!("Free");
            let result = hipModuleUnload(self.module);
            assert_eq!(result, hipError_t::hipSuccess);
        }

        println!("Finished");
    }
}
