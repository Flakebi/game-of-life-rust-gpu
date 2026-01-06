use std::default::Default;
use std::error::Error;
use std::ffi;
use std::{borrow::Cow, ops::Drop, os::raw::c_char};

use ash::vk;
use ash::{
    Device, Entry, Instance,
    ext::{debug_utils, descriptor_buffer},
    khr::{surface, swapchain},
};
use winit::application::ApplicationHandler;
use winit::window::{Window, WindowAttributes};
use winit::{
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
};

pub trait MyApp {
    fn on_render(&mut self, index: u32, win: &Vk);
    fn on_event(&mut self, event: WindowEvent, win: &mut Vk);
}

pub struct App {
    window_width: u32,
    window_height: u32,
    vk: Option<Vk>,
    create_app: Option<Box<dyn FnOnce(&Vk) -> Box<dyn MyApp>>>,
    app: Option<Box<dyn MyApp>>,
}

impl App {
    pub fn new(
        window_width: u32,
        window_height: u32,
        create_app: Box<dyn FnOnce(&Vk) -> Box<dyn MyApp>>,
    ) -> Self {
        Self {
            window_width,
            window_height,
            vk: None,
            create_app: Some(create_app),
            app: None,
        }
    }

    pub fn run(self) -> Result<(), Box<dyn Error>> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Wait);
        Ok(event_loop.run_app(self)?)
    }
}

impl ApplicationHandler for App {
    fn can_create_surfaces(&mut self, event_loop: &dyn winit::event_loop::ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("Game of Life")
                    .with_surface_size(winit::dpi::LogicalSize::new(
                        f64::from(self.window_width),
                        f64::from(self.window_height),
                    )),
            )
            .unwrap();
        let mut vk = Vk::new(window).unwrap();
        vk.ensure_swapchain(false);
        self.vk = Some(vk);
        self.app = Some(self.create_app.take().unwrap()(self.vk.as_ref().unwrap()));
    }

    fn window_event(
        &mut self,
        event_loop: &dyn winit::event_loop::ActiveEventLoop,
        _: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            e => {
                if let Some(app) = &mut self.app {
                    app.on_event(e, self.vk.as_mut().unwrap());
                    if self.vk.as_ref().unwrap().paused {
                        event_loop.set_control_flow(ControlFlow::Wait);
                    } else {
                        event_loop.set_control_flow(ControlFlow::Poll);
                    }
                }
            }
        }
    }

    fn about_to_wait(&mut self, _: &dyn winit::event_loop::ActiveEventLoop) {
        if let Some(vk) = &mut self.vk {
            vk.render(|i, vk| self.app.as_mut().unwrap().on_render(i, vk));
        }
    }
}

/// Helper function for submitting command buffers. Immediately waits for the fence before the command buffer
/// is executed. That way we can delay the waiting for the fences by 1 frame which is good for performance.
/// Make sure to create the fence in a signaled state on the first use.
#[allow(clippy::too_many_arguments)]
pub fn record_submit_commandbuffer<F: FnOnce(&Device, vk::CommandBuffer)>(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_reuse_fence: vk::Fence,
    submit_queue: vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    f: F,
) {
    unsafe {
        device
            .wait_for_fences(&[command_buffer_reuse_fence], true, u64::MAX)
            .expect("Wait for fence failed.");

        device
            .reset_fences(&[command_buffer_reuse_fence])
            .expect("Reset fences failed.");

        device
            .reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
            .expect("Reset command buffer failed.");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Begin commandbuffer");
        f(device, command_buffer);
        device
            .end_command_buffer(command_buffer)
            .expect("End commandbuffer");

        let command_buffers = vec![command_buffer];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);

        device
            .queue_submit(submit_queue, &[submit_info], command_buffer_reuse_fence)
            .expect("queue submit failed.");
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { *p_callback_data };
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        unsafe { ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy() }
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        unsafe { ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy() }
    };

    println!(
        "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
    );

    vk::FALSE
}

pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}

// Everything dependent on window/swapchain size
pub struct Swapchain {
    pub surface_resolution: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,

    pub screen_image: vk::Image,
    pub screen_image_view: vk::ImageView,
    pub screen_image_descriptor: [u8; 32],

    // Images containing game of life
    pub content_images: Vec<vk::Image>,
    pub content_image_views: Vec<vk::ImageView>,
    pub content_image_descriptors: Vec<[u8; 32]>,

    pub width: u32,
    pub height: u32,
}

#[allow(dead_code)]
pub struct Vk {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub surface_loader: surface::Instance,
    pub swapchain_loader: swapchain::Device,
    pub debug_utils_loader: debug_utils::Instance,
    pub window: Box<dyn Window>,
    pub debug_call_back: vk::DebugUtilsMessengerEXT,

    pub pdevice: vk::PhysicalDevice,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,
    pub present_queue: vk::Queue,
    pub present_mode: vk::PresentModeKHR,

    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,

    pub swapchain: Option<Swapchain>,

    pub content_image_sampler: [u8; 16],

    pub pool: vk::CommandPool,
    pub draw_command_buffer: vk::CommandBuffer,
    pub setup_command_buffer: vk::CommandBuffer,

    pub present_complete_semaphore: vk::Semaphore,
    pub rendering_complete_semaphore: vk::Semaphore,

    pub draw_commands_reuse_fence: vk::Fence,
    pub setup_commands_reuse_fence: vk::Fence,

    pub tile_size: u32,
    pub paused: bool,
}

impl Vk {
    pub fn new(window: Box<dyn Window>) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let entry = Entry::load().expect("Failed to load Vulkan");
            let app_name = ffi::CStr::from_bytes_with_nul_unchecked(b"GameOfLife\0");

            let layer_names = [ffi::CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )];
            let layers_names_raw: Vec<*const c_char> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let mut extension_names =
                ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())
                    .unwrap()
                    .to_vec();
            extension_names.push(debug_utils::NAME.as_ptr());

            let appinfo = vk::ApplicationInfo::default()
                .application_name(app_name)
                .application_version(0)
                .engine_name(app_name)
                .engine_version(0)
                .api_version(vk::make_api_version(0, 1, 4, 0));

            let create_flags = vk::InstanceCreateFlags::default();

            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&appinfo)
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&extension_names)
                .flags(create_flags);

            let instance: Instance = entry
                .create_instance(&create_info, None)
                .expect("Instance creation error");

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
            let debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();
            let surface = ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
            .unwrap();
            let pdevices = instance
                .enumerate_physical_devices()
                .expect("Physical device error");
            let surface_loader = surface::Instance::new(&entry, &instance);
            let (pdevice, queue_family_index) = pdevices
                .iter()
                .find_map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *pdevice,
                                            index as u32,
                                            surface,
                                        )
                                        .unwrap();
                            if supports_graphic_and_surface {
                                Some((*pdevice, index))
                            } else {
                                None
                            }
                        })
                })
                .expect("Couldn't find suitable device.");
            let queue_family_index = queue_family_index as u32;
            let device_extension_names_raw = [
                swapchain::NAME.as_ptr(),
                ash::khr::buffer_device_address::NAME.as_ptr(),
                ash::ext::descriptor_indexing::NAME.as_ptr(),
                ash::khr::synchronization2::NAME.as_ptr(),
                descriptor_buffer::NAME.as_ptr(),
            ];

            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                ..Default::default()
            };
            let priorities = [1.0];

            let queue_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);

            let mut device_descriptor_features =
                vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default().descriptor_buffer(true);

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&device_extension_names_raw)
                .enabled_features(&features)
                .push_next(&mut device_descriptor_features);

            let device: Device = instance
                .create_device(pdevice, &device_create_info, None)
                .unwrap();

            let present_queue = device.get_device_queue(queue_family_index, 0);

            let surface_format = surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap()[0];

            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(pdevice, surface)
                .unwrap();
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);
            let swapchain_loader = swapchain::Device::new(&instance, &device);

            let pool_create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            let pool = device.create_command_pool(&pool_create_info, None).unwrap();

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(2)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers = device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap();
            let setup_command_buffer = command_buffers[0];
            let draw_command_buffer = command_buffers[1];

            let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);
            let get_descriptor_device = descriptor_buffer::Device::new(&instance, &device);

            let content_image_sampler = {
                let sampler_info = vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .mip_lod_bias(0.0)
                    .min_lod(0.0)
                    .max_lod(0.0);
                let sampler = device.create_sampler(&sampler_info, None).unwrap();

                let info = vk::DescriptorGetInfoEXT::default()
                    .ty(vk::DescriptorType::SAMPLER)
                    .data(vk::DescriptorDataEXT {
                        p_sampler: &sampler as *const _,
                    });
                let mut data = [0u8; 16];
                get_descriptor_device.get_descriptor(&info, &mut data);
                data
            };

            let fence_create_info =
                vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

            let draw_commands_reuse_fence = device
                .create_fence(&fence_create_info, None)
                .expect("Create fence failed.");
            let setup_commands_reuse_fence = device
                .create_fence(&fence_create_info, None)
                .expect("Create fence failed.");

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            let rendering_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            Ok(Self {
                entry,
                instance,
                device,
                queue_family_index,
                pdevice,
                device_memory_properties,
                window,
                surface_loader,
                surface_format,
                present_queue,
                present_mode,
                swapchain_loader,
                swapchain: None,
                content_image_sampler,
                pool,
                draw_command_buffer,
                setup_command_buffer,
                present_complete_semaphore,
                rendering_complete_semaphore,
                draw_commands_reuse_fence,
                setup_commands_reuse_fence,
                surface,
                debug_call_back,
                debug_utils_loader,
                tile_size: 16,
                paused: false,
            })
        }
    }

    pub fn ensure_swapchain(&mut self, force_recreate: bool) {
        if self.swapchain.is_some() && !force_recreate {
            return;
        }

        unsafe {
            let surface_capabilities = self
                .surface_loader
                .get_physical_device_surface_capabilities(self.pdevice, self.surface)
                .unwrap();

            let mut desired_image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }
            let surface_resolution = match surface_capabilities.current_extent.width {
                u32::MAX => {
                    let size = self.window.surface_size();
                    vk::Extent2D {
                        width: size.width,
                        height: size.height,
                    }
                }
                _ => surface_capabilities.current_extent,
            };
            println!("Creating swapchain with size {surface_resolution:?}");

            let get_descriptor_device =
                descriptor_buffer::Device::new(&self.instance, &self.device);
            let create_img_internal = |width, height, format, usage| {
                let info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(vk::Extent3D::default().width(width).height(height).depth(1))
                    .mip_levels(1)
                    .array_layers(1)
                    .format(format)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .initial_layout(vk::ImageLayout::GENERAL)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .samples(vk::SampleCountFlags::TYPE_1);
                let image = self.device.create_image(&info, None).unwrap();

                let reqs = self.device.get_image_memory_requirements(image);
                let alloc_info = vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(
                        find_memorytype_index(
                            &reqs,
                            &self.device_memory_properties,
                            vk::MemoryPropertyFlags::empty(),
                        )
                        .expect("Failed to find memory type"),
                    );
                let memory = self.device.allocate_memory(&alloc_info, None).unwrap();
                self.device.bind_image_memory(image, memory, 0).unwrap();

                let create_view_info = vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);
                let view = self
                    .device
                    .create_image_view(&create_view_info, None)
                    .unwrap();

                let ii = vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL);
                let info = vk::DescriptorGetInfoEXT::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .data(vk::DescriptorDataEXT {
                        p_storage_image: &ii as *const _,
                    });
                let mut desc = [0u8; 32];
                get_descriptor_device.get_descriptor(&info, &mut desc);

                (image, view, desc)
            };

            let create_img = |width, height| {
                create_img_internal(
                    width,
                    height,
                    vk::Format::R8_UINT,
                    vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST,
                )
            };

            let width = surface_resolution.width / self.tile_size;
            let height = surface_resolution.height / self.tile_size;
            let a = create_img(width, height);
            let b = create_img(width, height);
            let (screen_image, screen_image_view, screen_image_descriptor) = create_img_internal(
                surface_resolution.width,
                surface_resolution.height,
                self.surface_format.format,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            );

            let content_images = vec![a.0, b.0];
            let content_image_views = vec![a.1, b.1];
            let content_image_descriptors = vec![a.2, b.2];

            if let Some(swapchain) = self.swapchain.take() {
                // Copy old to new content images
                record_submit_commandbuffer(
                    &self.device,
                    self.draw_command_buffer,
                    self.draw_commands_reuse_fence,
                    self.present_queue,
                    &[],
                    &[],
                    &[],
                    |device, draw_command_buffer| {
                        let sub_layers = vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1);
                        let regions = [vk::ImageCopy2::default()
                            .src_subresource(sub_layers)
                            .dst_subresource(sub_layers)
                            .extent(
                                vk::Extent3D::default()
                                    .width(width.min(swapchain.width))
                                    .height(height.min(swapchain.height))
                                    .depth(1),
                            )];

                        for i in 0..swapchain.content_images.len() {
                            let copy_info = vk::CopyImageInfo2::default()
                                .src_image(swapchain.content_images[i])
                                .src_image_layout(vk::ImageLayout::GENERAL)
                                .dst_image(content_images[i])
                                .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                                .regions(&regions);
                            device.cmd_copy_image2(draw_command_buffer, &copy_info);
                        }
                    },
                );

                self.device.device_wait_idle().unwrap();

                swapchain.clear(self);
            }

            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(desired_image_count)
                .image_color_space(self.surface_format.color_space)
                .image_format(self.surface_format.format)
                .image_extent(surface_resolution)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(self.present_mode)
                .clipped(true)
                .image_array_layers(1);

            let swapchain = self
                .swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();

            let present_images = self
                .swapchain_loader
                .get_swapchain_images(swapchain)
                .unwrap();
            let present_image_views: Vec<vk::ImageView> = present_images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo::default()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(image);
                    self.device
                        .create_image_view(&create_view_info, None)
                        .unwrap()
                })
                .collect();

            self.swapchain = Some(Swapchain {
                surface_resolution,
                swapchain,
                present_images,
                present_image_views,
                screen_image,
                screen_image_view,
                screen_image_descriptor,
                content_images,
                content_image_views,
                content_image_descriptors,

                width,
                height,
            })
        }
    }

    pub fn render<'a, F: FnMut(u32, &Self) + 'a>(&'a mut self, mut f: F) {
        unsafe {
            self.ensure_swapchain(false);
            let (mut present_index, suboptimal) = {
                let swapchain = self.swapchain.as_ref().unwrap();
                self.swapchain_loader
                    .acquire_next_image(
                        swapchain.swapchain,
                        std::u64::MAX,
                        self.present_complete_semaphore,
                        vk::Fence::null(),
                    )
                    .unwrap()
            };
            if suboptimal {
                println!("Is suboptimal");
                self.ensure_swapchain(true);
                let swapchain = self.swapchain.as_ref().unwrap();
                present_index = self
                    .swapchain_loader
                    .acquire_next_image(
                        swapchain.swapchain,
                        std::u64::MAX,
                        self.present_complete_semaphore,
                        vk::Fence::null(),
                    )
                    .unwrap()
                    .0;
            }
            f(present_index, self);
            let swapchain = self.swapchain.as_ref().unwrap();

            let present_image = swapchain.present_images[present_index as usize];
            record_submit_commandbuffer(
                &self.device,
                self.draw_command_buffer,
                self.draw_commands_reuse_fence,
                self.present_queue,
                &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                &[self.present_complete_semaphore],
                &[self.rendering_complete_semaphore],
                |device, draw_command_buffer| {
                    let sub_layers = vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1);
                    let regions = [vk::ImageCopy2::default()
                        .src_subresource(sub_layers)
                        .dst_subresource(sub_layers)
                        .extent(
                            vk::Extent3D::default()
                                .width(swapchain.surface_resolution.width)
                                .height(swapchain.surface_resolution.height)
                                .depth(1),
                        )];
                    let copy_info = vk::CopyImageInfo2::default()
                        .src_image(swapchain.screen_image)
                        .src_image_layout(vk::ImageLayout::GENERAL)
                        .dst_image(present_image)
                        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .regions(&regions);

                    let sub_range = vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1);
                    device.cmd_pipeline_barrier(
                        draw_command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::default(),
                        &[],
                        &[],
                        &[vk::ImageMemoryBarrier::default()
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(present_image)
                            .subresource_range(sub_range)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
                    );
                    device.cmd_copy_image2(draw_command_buffer, &copy_info);
                    device.cmd_pipeline_barrier(
                        draw_command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::default(),
                        &[],
                        &[],
                        &[vk::ImageMemoryBarrier::default()
                            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(present_image)
                            .subresource_range(sub_range)
                            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .dst_access_mask(vk::AccessFlags::empty())],
                    );
                    /*device.cmd_begin_render_pass(
                        draw_command_buffer,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    );
                    device.cmd_bind_pipeline(
                        draw_command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        graphic_pipeline,
                    );
                    device.cmd_set_viewport(draw_command_buffer, 0, &viewports);
                    device.cmd_set_scissor(draw_command_buffer, 0, &scissors);
                    device.cmd_bind_vertex_buffers(
                        draw_command_buffer,
                        0,
                        &[vertex_input_buffer],
                        &[0],
                    );
                    device.cmd_bind_index_buffer(
                        draw_command_buffer,
                        index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    device.cmd_draw_indexed(
                        draw_command_buffer,
                        index_buffer_data.len() as u32,
                        1,
                        0,
                        0,
                        1,
                    );
                    // Or draw without the index buffer
                    // device.cmd_draw(draw_command_buffer, 3, 1, 0, 0);
                    device.cmd_end_render_pass(draw_command_buffer);*/
                },
            );

            let wait_semaphors = [self.rendering_complete_semaphore];
            let swapchains = [swapchain.swapchain];
            let image_indices = [present_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&wait_semaphors) // &self.rendering_complete_semaphore)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
                .unwrap();
            self.device.device_wait_idle().unwrap();
        }
    }

    pub fn clear(&self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            /*self.device.free_memory(index_buffer_memory, None);
            self.device.destroy_buffer(index_buffer, None);
            self.device.free_memory(vertex_input_buffer_memory, None);
            self.device.destroy_buffer(vertex_input_buffer, None);
            for framebuffer in framebuffers {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            self.device.destroy_render_pass(renderpass, None);*/
        }
    }
}

impl Swapchain {
    fn clear(&self, vk: &Vk) {
        unsafe {
            for &image_view in self.present_image_views.iter() {
                vk.device.destroy_image_view(image_view, None);
            }
            vk.device.destroy_image_view(self.screen_image_view, None);
            vk.device.destroy_image(self.screen_image, None);
            for &image_view in self.content_image_views.iter() {
                vk.device.destroy_image_view(image_view, None);
            }
            for &image in self.content_images.iter() {
                vk.device.destroy_image(image, None);
            }
            vk.swapchain_loader.destroy_swapchain(self.swapchain, None);
        }
    }
}

impl Drop for Vk {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device
                .destroy_semaphore(self.present_complete_semaphore, None);
            self.device
                .destroy_semaphore(self.rendering_complete_semaphore, None);
            self.device
                .destroy_fence(self.draw_commands_reuse_fence, None);
            self.device
                .destroy_fence(self.setup_commands_reuse_fence, None);
            if let Some(swapchain) = self.swapchain.take() {
                swapchain.clear(self);
            }
            // self.device
            // .destroy_sampler(self.content_image_sampler, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_call_back, None);
            self.instance.destroy_instance(None);
        }
    }
}
