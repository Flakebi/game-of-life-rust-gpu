use std::ffi::{CString, c_void};
use std::path::PathBuf;

use clap::Parser;
use hip_runtime_sys::{
    hipDeviceProp_t, hipDeviceSynchronize, hipDeviceptr_t, hipDriverGetVersion, hipError_t,
    hipFree, hipFunction_t, hipGetDevice, hipGetDeviceCount, hipGetDeviceProperties, hipInit,
    hipMalloc, hipMemcpyHtoD, hipModule_t, hipModuleGetFunction, hipModuleLaunchKernel,
    hipModuleLoadData, hipModuleUnload, hipRuntimeGetVersion, hipSetDevice,
};

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
    let size = width * height;

    vk::vk().unwrap();

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

        // Assemble arguments for the kernel.
        // Pass two pointers, input_device and output_device
        #[allow(dead_code)]
        struct KernelArgs {
            input: *mut c_void,
            output: *mut c_void,
            width: u32,
            height: u32,
        }
        let kernel_args: &mut KernelArgs = &mut KernelArgs {
            input: a_device,
            output: b_device,
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
        println!("Launch {} {wg_x}x{wg_y}x1", args.kernel);
        let result = hipModuleLaunchKernel(
            function,
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

        println!("Wait for finish");
        let result = hipDeviceSynchronize();
        assert_eq!(result, hipError_t::hipSuccess);

        /*println!("Copy memory back");
        let result = hipMemcpyDtoH(
            &mut output as *mut u32 as *mut c_void,
            output_device,
            mem::size_of_val(&output),
        );
        assert_eq!(result, hipError_t::hipSuccess);

        // Print result
        println!("Output: {}", output);*/

        println!("Free");
        let result = hipModuleUnload(module);
        assert_eq!(result, hipError_t::hipSuccess);
        let result = hipFree(a_device);
        assert_eq!(result, hipError_t::hipSuccess);
        let result = hipFree(b_device);
        assert_eq!(result, hipError_t::hipSuccess);

        println!("Finished");
    }
}
