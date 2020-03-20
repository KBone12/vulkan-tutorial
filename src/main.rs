use vulkano::{
    app_info_from_cargo_toml,
    device::{Device, DeviceExtensions, Features},
    instance::{
        debug::{DebugCallback, MessageSeverity, MessageType},
        layers_list, Instance, InstanceExtensions, PhysicalDevice,
    },
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
    let _surface = WindowBuilder::new()
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

    // Pick the logical device which supports graphics
    let (_device, _queues) = PhysicalDevice::enumerate(&instance)
        .filter_map(|device| {
            // Pick the physical device which supports graphics
            device
                .queue_families()
                .filter(|queue_family| queue_family.supports_graphics())
                .map(|queue_family| (device, queue_family))
                .next()
        })
        .filter_map(|(device, queue_family)| {
            Device::new(
                device,
                &Features::none(),
                &DeviceExtensions::supported_by_device(device),
                vec![(queue_family, 1.0)],
            )
            .ok()
        })
        .next()
        .expect("Could not find any GPU");

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
