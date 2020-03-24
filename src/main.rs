use std::{collections::HashSet, error::Error, sync::Arc};

use vulkano::{
    app_info_from_cargo_toml,
    command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState},
    descriptor::PipelineLayoutAbstract,
    device::{Device, DeviceCreationError, DeviceExtensions, Features, Queue},
    format::Format,
    framebuffer::{
        Framebuffer, FramebufferAbstract, FramebufferCreationError, RenderPassAbstract,
        RenderPassCreationError, Subpass,
    },
    image::{swapchain::SwapchainImage, ImageUsage},
    instance::{
        debug::{DebugCallback, DebugCallbackCreationError, MessageSeverity, MessageType},
        layers_list, Instance, InstanceCreationError, InstanceExtensions, PhysicalDevice,
    },
    pipeline::{
        vertex::{BufferlessDefinition, BufferlessVertices},
        viewport::Viewport,
        GraphicsPipeline,
    },
    single_pass_renderpass,
    swapchain::{
        acquire_next_image, AcquireError, CapabilitiesError, ColorSpace, CompositeAlpha,
        FullscreenExclusive, PresentMode, Surface, Swapchain, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture, SharingMode},
};
use vulkano_win::{required_extensions, VkSurfaceBuild};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn print_layers_list() {
    println!("=== Layers ===");
    layers_list().unwrap().for_each(|layer| {
        println!(
            "{} (version: {})",
            layer.name(),
            layer.implementation_version()
        );
        println!("Description: {}", layer.description());
    });
}

fn print_physical_devices(instance: &Arc<Instance>) {
    println!("=== Devices ===");
    PhysicalDevice::enumerate(instance).for_each(|device| {
        println!("{} (index: {})", device.name(), device.index());
        println!(
            "API version: {}, Driver version: {}",
            device.api_version(),
            device.driver_version()
        );
        println!("Type: {:?}", device.ty());
    });
}

fn create_vulkan_instance() -> Result<Arc<Instance>, InstanceCreationError> {
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
    instance
}

fn register_debug_callback(
    instance: &Arc<Instance>,
) -> Option<Result<DebugCallback, DebugCallbackCreationError>> {
    if !cfg!(debug_assertions) {
        return None;
    }

    let severity = MessageSeverity {
        error: true,
        warning: true,
        information: true,
        verbose: true,
    };
    let ty = MessageType::all();
    Some(DebugCallback::new(instance, severity, ty, |message| {
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
    }))
}

fn create_device_and_queues(
    instance: &Arc<Instance>,
    surface: &Arc<Surface<Window>>,
) -> Result<(Arc<Device>, Arc<Queue>, Arc<Queue>), DeviceCreationError> {
    let (device, queues) = PhysicalDevice::enumerate(instance)
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
                    // safe to unwrap
                    let graphics_queue_family = graphics_queue_family.unwrap();
                    let present_queue_family = present_queue_family.unwrap();

                    let mut queue_families_set = HashSet::new();
                    let unique_queue_families: Vec<_> =
                        vec![graphics_queue_family, present_queue_family]
                            .iter()
                            .filter(|queue_family| queue_families_set.insert(queue_family.id()))
                            .map(|queue_family| queue_family.to_owned())
                            .collect();
                    Some((device, unique_queue_families))
                })
        })
        .map(|(device, queue_families)| {
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
        })
        .filter(|device| device.is_ok())
        .next()
        .ok_or(DeviceCreationError::FeatureNotPresent)??; // If nothing found, return DeviceCreationError::FeatureNotPresent
    let queues: Vec<Arc<Queue>> = queues.collect();
    let graphics_queue = queues
        .iter()
        .find(|queue| queue.family().supports_graphics())
        .unwrap(); // Must safe
    let present_queue = queues
        .iter()
        .find(|queue| surface.is_supported(queue.family()) == Ok(true))
        .unwrap(); // Must safe
    Ok((device, graphics_queue.clone(), present_queue.clone()))
}

fn create_swapchain(
    surface: &Arc<Surface<Window>>,
    device: &Arc<Device>,
    graphics_queue: &Arc<Queue>,
    present_queue: &Arc<Queue>,
) -> Result<(Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>), SwapchainCreationError> {
    let capabilities = surface
        .capabilities(device.physical_device())
        .map_err(|e| match e {
            CapabilitiesError::OomError(e) => SwapchainCreationError::OomError(e),
            CapabilitiesError::SurfaceLost => SwapchainCreationError::SurfaceLost,
        })?;
    let num_images = capabilities
        .max_image_count
        .unwrap_or(capabilities.min_image_count + 1)
        .min(capabilities.min_image_count + 1);
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
        SharingMode::from(graphics_queue)
    } else {
        SharingMode::from(vec![graphics_queue, present_queue].as_slice())
    };
    let clipped = true;
    Swapchain::new(
        device.clone(),
        surface.clone(),
        num_images,
        Format::B8G8R8A8Unorm,
        dimensions,
        layers,
        image_usage,
        sharing,
        capabilities.current_transform,
        CompositeAlpha::Opaque,
        PresentMode::Fifo,
        FullscreenExclusive::Default,
        clipped,
        ColorSpace::SrgbNonLinear,
    )
}

fn create_render_pass(
    device: &Arc<Device>,
    color_format: Format,
) -> Result<Arc<dyn RenderPassAbstract + Send + Sync>, RenderPassCreationError> {
    Ok(Arc::new(single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: color_format,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )?))
}

fn create_graphics_pipeline(
    device: &Arc<Device>,
    dimensions: [f32; 2],
    render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>,
) -> Result<
    Arc<
        GraphicsPipeline<
            BufferlessDefinition,
            Box<dyn PipelineLayoutAbstract + Send + Sync>,
            Arc<dyn RenderPassAbstract + Send + Sync>,
        >,
    >,
    Box<dyn Error>,
> {
    mod vertex_shader {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shader/triangle.vert"
        }
    }
    mod fragment_shader {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shader/triangle.frag"
        }
    }
    let vertex_shader = vertex_shader::Shader::load(device.clone())?;
    let fragment_shader = fragment_shader::Shader::load(device.clone())?;
    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions,
        depth_range: 0.0..1.0,
    };
    Ok(Arc::new(
        GraphicsPipeline::start()
            .vertex_input(BufferlessDefinition)
            .vertex_shader(vertex_shader.main_entry_point(), ())
            .fragment_shader(fragment_shader.main_entry_point(), ())
            .triangle_list()
            .viewports(vec![viewport])
            .depth_clamp(false)
            .polygon_mode_fill()
            .line_width(1.0)
            .cull_mode_back()
            .front_face_clockwise()
            .blend_pass_through()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())?,
    ))
}

fn create_framebuffers(
    swapchain_images: &[Arc<SwapchainImage<Window>>],
    render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>,
) -> Result<Vec<Arc<dyn FramebufferAbstract + Send + Sync>>, FramebufferCreationError> {
    swapchain_images
        .iter()
        .map(|image| {
            Framebuffer::start(render_pass.clone())
                .add(image.clone())
                .unwrap()
                .build()
                .map(|framebuffer| {
                    let framebuffer: Arc<dyn FramebufferAbstract + Send + Sync> =
                        Arc::new(framebuffer);
                    framebuffer
                })
        })
        .collect()
}

fn create_command_buffers(
    framebuffers: &[Arc<dyn FramebufferAbstract + Send + Sync>],
    device: &Arc<Device>,
    graphics_queue: &Arc<Queue>,
    graphics_pipeline: &Arc<
        GraphicsPipeline<
            BufferlessDefinition,
            Box<dyn PipelineLayoutAbstract + Send + Sync>,
            Arc<dyn RenderPassAbstract + Send + Sync>,
        >,
    >,
) -> Result<Vec<Arc<AutoCommandBuffer>>, Box<dyn Error>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let vertices = BufferlessVertices {
                vertices: 3, // triangle
                instances: 1,
            };
            let command_buffer = AutoCommandBufferBuilder::primary_simultaneous_use(
                device.clone(),
                graphics_queue.family(),
            )?
            .begin_render_pass(
                framebuffer.clone(),
                false,
                vec![[0.0, 0.0, 0.0, 1.0].into()],
            )?
            .draw(
                graphics_pipeline.clone(),
                &DynamicState::none(),
                vertices,
                (),
                (),
            )?
            .end_render_pass()?
            .build()?;
            Ok(Arc::new(command_buffer))
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    print_layers_list();

    let instance = create_vulkan_instance()?;
    let _ = match register_debug_callback(&instance) {
        Some(result) => Some(result?),
        None => None, // None for release mode
    };

    print_physical_devices(&instance);

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let (device, graphics_queue, present_queue) = create_device_and_queues(&instance, &surface)?;
    let (mut swapchain, mut swapchain_images) =
        create_swapchain(&surface, &device, &graphics_queue, &present_queue)?;

    let mut render_pass = create_render_pass(&device, swapchain.format())?;
    let mut graphics_pipeline = create_graphics_pipeline(
        &device,
        [
            swapchain.dimensions()[0] as _,
            swapchain.dimensions()[1] as _,
        ],
        &render_pass,
    )?;

    let mut framebuffers: Vec<_> = create_framebuffers(&swapchain_images, &render_pass)?;
    let mut command_buffers: Vec<_> =
        create_command_buffers(&framebuffers, &device, &graphics_queue, &graphics_pipeline)?;

    let mut prev_future: Option<Box<dyn GpuFuture>> = None;
    let mut request_recreate_swapchain = false;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                request_recreate_swapchain = true;
            }
            Event::MainEventsCleared => {
                surface.window().request_redraw();
            }
            Event::RedrawRequested(_) => {
                if let Some(ref mut prev_future) = prev_future {
                    prev_future.cleanup_finished();
                }

                match acquire_next_image(swapchain.clone(), None) {
                    Ok((image_index, suboptimal, acquire_future)) => {
                        let command_buffer = command_buffers[image_index].clone();
                        let future: Box<dyn GpuFuture> =
                            if let Some(prev_future) = prev_future.take() {
                                Box::new(prev_future.join(acquire_future))
                            } else {
                                Box::new(acquire_future)
                            };
                        if let Ok(future) =
                            future.then_execute(graphics_queue.clone(), command_buffer)
                        {
                            let future = future
                                .then_swapchain_present(
                                    present_queue.clone(),
                                    swapchain.clone(),
                                    image_index,
                                )
                                .then_signal_fence_and_flush();
                            match future {
                                Ok(future) => {
                                    prev_future = Some(Box::new(future));
                                }
                                Err(FlushError::OutOfDate) => {
                                    request_recreate_swapchain = true;
                                    prev_future = Some(Box::new(sync::now(device.clone())));
                                }
                                Err(e) => {
                                    eprintln!("{}", e);
                                    prev_future = None;
                                }
                            }
                        }
                        if suboptimal {
                            request_recreate_swapchain = true;
                        }
                    }
                    Err(AcquireError::OutOfDate) => {
                        request_recreate_swapchain = true;
                        prev_future = Some(Box::new(sync::now(device.clone())));
                    }
                    Err(e) => {
                        eprintln!("{}", e);
                        prev_future = None;
                    }
                }

                if request_recreate_swapchain {
                    request_recreate_swapchain = false;
                    let result = swapchain
                        .recreate()
                        .map_err(|e| e.to_string())
                        .and_then(|(new_swapchain, new_swapchain_images)| {
                            swapchain = new_swapchain;
                            swapchain_images = new_swapchain_images;
                            create_render_pass(&device, swapchain.format())
                                .map_err(|e| e.to_string())
                        })
                        .and_then(|new_render_pass| {
                            render_pass = new_render_pass;
                            create_graphics_pipeline(
                                &device,
                                [
                                    swapchain.dimensions()[0] as _,
                                    swapchain.dimensions()[1] as _,
                                ],
                                &render_pass,
                            )
                            .map_err(|e| e.to_string())
                        })
                        .and_then(|new_graphics_pipeline| {
                            graphics_pipeline = new_graphics_pipeline;
                            create_framebuffers(&swapchain_images, &render_pass)
                                .map_err(|e| e.to_string())
                        })
                        .and_then(|new_framebuffers| {
                            framebuffers = new_framebuffers;
                            create_command_buffers(
                                &framebuffers,
                                &device,
                                &graphics_queue,
                                &graphics_pipeline,
                            )
                            .map_err(|e| e.to_string())
                        })
                        .and_then(|new_command_buffers| {
                            command_buffers = new_command_buffers;
                            Ok(())
                        });
                    if let Err(msg) = result {
                        eprintln!("{}", msg);
                    }
                }
            }
            _ => {}
        };
    });
}
