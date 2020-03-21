use vulkano::{
    app_info_from_cargo_toml,
    device::{Device, DeviceExtensions, Features},
    format::Format,
    image::ImageUsage,
    instance::{
        debug::{DebugCallback, MessageSeverity, MessageType},
        layers_list, Instance, InstanceExtensions, PhysicalDevice,
    },
    swapchain::{ColorSpace, CompositeAlpha, FullscreenExclusive, PresentMode, Swapchain},
    sync::SharingMode,
};
use vulkano_win::{required_extensions, VkSurfaceBuild};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    println!("=== Layers ===");
    layers_list().unwrap().for_each(|layer| {
        println!(
            "{} (version: {})",
            layer.name(),
            layer.implementation_version()
        );
        println!("Description: {}", layer.description());
    });

    // Create a Vulkan instance
    let app_info = app_info_from_cargo_toml!();
    let extensions = required_extensions();
    let instance = if cfg!(debug_assertions) {
        let extensions = InstanceExtensions {
            ext_debug_utils: true,
            ..extensions
        };

        // Load validation layers
        let validation_layers = [
            "VK_LAYER_LUNARG_standard_validation",
            "VK_LAYER_KHRONOS_validation",
        ];
        let supported_validation_layers: Vec<_> = layers_list()
            .unwrap()
            .filter(|layer| validation_layers.contains(&layer.name()))
            .collect();

        Instance::new(
            Some(&app_info),
            &extensions,
            supported_validation_layers.iter().map(|layer| layer.name()),
        )
    } else {
        Instance::new(Some(&app_info), &extensions, None)
    };
    let instance = instance.expect("Could not build a Vulkan instance");

    // Register a debugging callback
    if cfg!(debug_assertions) {
        let severity = MessageSeverity {
            error: true,
            warning: true,
            information: true,
            verbose: true,
        };
        let ty = MessageType::all();
        DebugCallback::new(&instance, severity, ty, |message| {
            let severity = if message.severity.error {
                "ERROR"
            } else if message.severity.warning {
                "Warning"
            } else if message.severity.information {
                "Info"
            } else {
                "Verbose"
            };
            let ty = if message.ty.general {
                "general"
            } else if message.ty.validation {
                "validation"
            } else {
                "performance"
            };
            eprintln!(
                "{} (type: {}) (layer: {}): {}",
                severity, ty, message.layer_prefix, message.description
            );
        })
        .ok();
    }

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    println!("=== Devices ===");
    PhysicalDevice::enumerate(&instance).for_each(|device| {
        println!("{} (index: {})", device.name(), device.index());
        println!(
            "API version: {}, Driver version: {}",
            device.api_version(),
            device.driver_version()
        );
        println!("Type: {:?}", device.ty());
    });

    let (device, mut queues) = PhysicalDevice::enumerate(&instance)
        .filter_map(|device| {
            let graphics_queue_family = device
                .queue_families()
                .find(|queue_family| queue_family.supports_graphics());
            let present_queue_family = device
                .queue_families()
                .find(|queue_family| surface.is_supported(*queue_family) == Ok(true));
            graphics_queue_family
                .and(present_queue_family)
                .and_then(|_| {
                    Some((
                        device,
                        vec![
                            graphics_queue_family.unwrap(),
                            present_queue_family.unwrap(),
                        ],
                    ))
                })
        })
        .filter_map(|(device, queue_families)| {
            let extensions = DeviceExtensions {
                khr_swapchain: true,
                ..DeviceExtensions::supported_by_device(device)
            };
            Device::new(
                device,
                &Features::none(),
                &extensions,
                queue_families
                    .iter()
                    .map(|queue_family| (*queue_family, 1.0)),
            )
            .ok()
        })
        .next()
        .expect("Could not find any GPU");
    let graphics_queue = queues
        .find(|queue| queue.family().supports_graphics())
        .unwrap();
    let present_queue = queues
        .find(|queue| surface.is_supported(queue.family()) == Ok(true))
        .unwrap();

    let (_swapchain, _spwapchain_image) = {
        let capabilities = surface.capabilities(device.physical_device()).unwrap();
        let num_images = 2; // Request double buffering
        let (format, color_space) = capabilities
            .supported_formats
            .iter()
            .find(|(format, color_space)| {
                *format == Format::B8G8R8A8Unorm && *color_space == ColorSpace::SrgbNonLinear
            })
            .expect("Not supported");
        let dimensions = if let Some(dimensions) = capabilities.current_extent {
            dimensions
        } else {
            let [w, h]: [u32; 2] = surface.window().inner_size().into();
            let [min_w, min_h] = capabilities.min_image_extent;
            let [max_w, max_h] = capabilities.max_image_extent;
            // clamp width and height
            [min_w.max(max_w.min(w)), min_h.max(max_h.min(h))]
        };
        let layers = 1; // Usually 1
        let image_usage = ImageUsage {
            color_attachment: true,
            ..ImageUsage::none()
        };
        let sharing = if graphics_queue.family() == present_queue.family() {
            SharingMode::from(&graphics_queue)
        } else {
            SharingMode::from(vec![&graphics_queue, &present_queue].as_slice())
        };
        let alpha = capabilities
            .supported_composite_alpha
            .iter()
            .find(|alpha| *alpha == CompositeAlpha::Opaque)
            .expect("Not supported");
        let present_mode = capabilities
            .present_modes
            .iter()
            .find(|mode| *mode == PresentMode::Fifo)
            .expect("Not supported");
        let clipped = true;
        Swapchain::new(
            device,
            surface,
            num_images,
            *format,
            dimensions,
            layers,
            image_usage,
            sharing,
            capabilities.current_transform,
            alpha,
            present_mode,
            FullscreenExclusive::Default,
            clipped,
            *color_space,
        )
        .unwrap()
    };

    event_loop.run(|event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            _ => {}
        };
    });
}
