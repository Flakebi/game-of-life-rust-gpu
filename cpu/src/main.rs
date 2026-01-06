use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::path::PathBuf;

use clap::Parser;
use hip_runtime_sys::{
    hipDeviceProp_t, hipDeviceSynchronize, hipDeviceptr_t, hipDriverGetVersion, hipError_t,
    hipFree, hipFunction_t, hipGetDevice, hipGetDeviceCount, hipGetDeviceProperties, hipInit,
    hipMalloc, hipMemcpyHtoD, hipModule_t, hipModuleGetFunction, hipModuleLaunchKernel,
    hipModuleLoadData, hipModuleUnload, hipRuntimeGetVersion, hipSetDevice,
};
use winit::event::{ButtonSource, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::keyboard::Key;

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

    /// Name of the kernel
    #[arg(short, long, default_value = "kernel")]
    kernel: String,
}

fn get_str(s: &[i8]) -> &str {
    let cs =
        std::ffi::CStr::from_bytes_until_nul(unsafe { std::mem::transmute::<&[i8], &[u8]>(s) })
            .unwrap();
    cs.to_str().unwrap()
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

    let width: u32 = 100;
    let height: u32 = 100;
    let screen_width: u32 = 1920;
    let screen_height: u32 = 1000;
    let size = width * height;

    let app = vk::App::new(
        screen_width,
        screen_height,
        Box::new(move |win| {
            unsafe {
                println!("Set device {}", args.device_index);
                let result = hipSetDevice(args.device_index);
                assert_eq!(result, hipError_t::hipSuccess);
                let mut device = 0;
                let result = hipGetDevice(&mut device);
                assert_eq!(result, hipError_t::hipSuccess);

                // Allocate two buffers on the GPU for double buffering
                println!("Alloc memory");
                let mut a_device: hipDeviceptr_t = std::ptr::null_mut();
                let mut b_device: hipDeviceptr_t = std::ptr::null_mut();
                let result = hipMalloc(&mut a_device, size as usize);
                assert_eq!(result, hipError_t::hipSuccess);
                let result = hipMalloc(&mut b_device, size as usize);
                assert_eq!(result, hipError_t::hipSuccess);

                // Copy a to GPU buffers
                let mut a = vec![0u8; size as usize];
                println!(
                    "Copy memory from {:?} (cpu) to {:?} (gpu)",
                    a.as_ptr(),
                    a_device
                );
                let result = hipMemcpyHtoD(a_device, a.as_mut_ptr() as *mut c_void, size as usize);
                assert_eq!(result, hipError_t::hipSuccess);

                // Load the executable that was compiled for the GPU
                println!("Load module from {}", args.file.display());
                let module_data = std::fs::read(args.file).unwrap();
                let mut module: hipModule_t = std::ptr::null_mut();
                let result = hipModuleLoadData(&mut module, module_data.as_ptr() as *const c_void);
                assert_eq!(result, hipError_t::hipSuccess);

                // Get kernel function from loaded module
                println!("Get function {}", args.kernel);
                let mut function: hipFunction_t = std::ptr::null_mut();
                let kernel_name = CString::new(args.kernel.clone()).expect("Invalid kernel name");
                let result = hipModuleGetFunction(&mut function, module, kernel_name.as_ptr());
                assert_eq!(result, hipError_t::hipSuccess);

                #[allow(dead_code)]
                struct ClearKernelArgs {
                    image_desc: [u8; 32],
                    width: u32,
                    height: u32,
                }

                #[allow(dead_code)]
                struct SetKernelArgs {
                    image_desc: [u8; 32],
                    x: u32,
                    y: u32,
                    val: f32,
                }

                // Clear image
                println!("Get function clear");
                let mut clear: hipFunction_t = std::ptr::null_mut();
                let result =
                    hipModuleGetFunction(&mut clear, module, b"clear\0".as_ptr() as *const _);
                assert_eq!(result, hipError_t::hipSuccess);

                for image in &win.content_image_descriptors {
                    let kernel_args = &mut ClearKernelArgs {
                        image_desc: *image,
                        width,
                        height,
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
                    // Launch two workgroups (2x1x1), each of the size 16x16x1
                    let wg_x = width.div_ceil(WG_SIZE);
                    let wg_y = height.div_ceil(WG_SIZE);
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

                let mut set: hipFunction_t = std::ptr::null_mut();
                let result = hipModuleGetFunction(&mut set, module, b"set\0".as_ptr() as *const _);
                assert_eq!(result, hipError_t::hipSuccess);

                let mut check: hipFunction_t = std::ptr::null_mut();
                let result =
                    hipModuleGetFunction(&mut check, module, b"check\0".as_ptr() as *const _);
                assert_eq!(result, hipError_t::hipSuccess);

                let width = 100;
                let height = 100;
                Box::new(App {
                    i: 0,
                    paused: false,
                    width,
                    height,
                    screen_width,
                    screen_height,
                    module,
                    function,
                    set,
                    check,
                    is_pressed: Default::default(),
                })
            }
        }),
    );
    app.run().unwrap();
}

struct App {
    i: u32,
    paused: bool,
    width: u32,
    height: u32,
    screen_width: u32,
    screen_height: u32,
    module: hipModule_t,
    function: hipFunction_t,
    set: hipFunction_t,
    check: hipFunction_t,
    // Presset buttons and associated value (set or unset)
    is_pressed: HashMap<Option<DeviceId>, bool>,
}

impl App {
    // x and y are screen coordinates
    fn set(&mut self, screen_x: f64, screen_y: f64, set: bool, win: &vk::Vk) {
        // Screen to data coordinates
        let x = (screen_x as f32 / self.screen_width as f32 * self.width as f32) as u32;
        let y = (screen_y as f32 / self.screen_height as f32 * self.height as f32) as u32;
        println!("Set {screen_x},{screen_y} → {x},{y} = {set:?}");

        unsafe {
            #[allow(dead_code)]
            struct KernelArgs {
                img: [u8; 32],
                x: u32,
                y: u32,
                val: f32,
            }

            let i = if self.paused { self.i ^ 1 } else { self.i };
            let img = win.content_image_descriptors[i as usize];
            let kernel_args = &mut KernelArgs {
                img,
                x,
                y,
                val: if set { 1.0 } else { 0.0 },
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

            // println!("Launch {} {wg_x}x{wg_y}x1", args.kernel);
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

            // println!("Wait for finish");
            let result = hipDeviceSynchronize();
            assert_eq!(result, hipError_t::hipSuccess);
        }
        // self.check(x, y, set, win);
    }

    fn check(&mut self, x: u32, y: u32, set: bool, win: &vk::Vk) {
        unsafe {
            #[allow(dead_code)]
            struct KernelArgs {
                img: [u8; 32],
                sampler: [u8; 16],
                x: u32,
                y: u32,
                width: u32,
                height: u32,
                val: f32,
            }

            let img = win.content_image_descriptors[(self.i ^ 1) as usize];
            let sampler = win.content_image_sampler;
            let kernel_args = &mut KernelArgs {
                img,
                sampler,
                x,
                y,
                width: self.width,
                height: self.height,
                val: if set { 1.0 } else { 0.0 },
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

            // println!("Launch {} {wg_x}x{wg_y}x1", args.kernel);
            let result = hipModuleLaunchKernel(
                self.check,
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

            // println!("Wait for finish");
            let result = hipDeviceSynchronize();
            assert_eq!(result, hipError_t::hipSuccess);
        }
    }
}

impl vk::MyApp for App {
    fn on_render(&mut self, _: u32, win: &vk::Vk) {
        unsafe {
            std::thread::sleep(std::time::Duration::from_millis(100));
            if !self.paused {
                self.i ^= 1;
            }
            // TODO Write directly to screen image
            let screen_desc = win.screen_image_descriptor;
            let old_content_desc = win.content_image_descriptors[(self.i ^ 1) as usize];
            let new_content_desc = win.content_image_descriptors[self.i as usize];
            let sampler = win.content_image_sampler;

            #[allow(dead_code)]
            struct KernelArgs {
                old_content_desc: [u8; 32],
                new_content_desc: [u8; 32],
                screen_desc: [u8; 32],
                sampler: [u8; 16],
                width: u32,
                height: u32,
                screen_width: u32,
                screen_height: u32,
                paused: u32,
            }

            let kernel_args = &mut KernelArgs {
                old_content_desc,
                new_content_desc,
                screen_desc,
                sampler,
                width: self.width,
                height: self.height,
                screen_width: self.screen_width,
                screen_height: self.screen_height,
                paused: if self.paused { 1 } else { 0 },
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
            // Launch two workgroups (2x1x1), each of the size 16x16x1
            let wg_x = self.width.div_ceil(WG_SIZE);
            let wg_y = self.height.div_ceil(WG_SIZE);
            // println!("Launch {} {wg_x}x{wg_y}x1", args.kernel);
            let result = hipModuleLaunchKernel(
                self.function,
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

            // println!("Wait for finish");
            let result = hipDeviceSynchronize();
            assert_eq!(result, hipError_t::hipSuccess);
        }
    }

    fn on_event(&mut self, event: WindowEvent, win: &vk::Vk) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Character(c),
                        ..
                    },
                ..
            } if c.as_str() == " " => self.paused = !self.paused,
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
        /*println!("Copy memory back");
        let result = hipMemcpyDtoH(
            &mut output as *mut u32 as *mut c_void,
            output_device,
            mem::size_of_val(&output),
        );
        assert_eq!(result, hipError_t::hipSuccess);

        // Print result
        println!("Output: {}", output);*/

        unsafe {
            println!("Free");
            let result = hipModuleUnload(self.module);
            assert_eq!(result, hipError_t::hipSuccess);
            /*let result = hipFree(a_device);
            assert_eq!(result, hipError_t::hipSuccess);
            let result = hipFree(b_device);
            assert_eq!(result, hipError_t::hipSuccess);*/
        }

        // win.clear();
        println!("Finished");
    }
}
