from scene import Scene
import taichi as ti


@ti.kernel
def write_circle(buffer: ti.template(), circle_radius: float):
    center = ti.Vector([buffer.shape[0] / 2.0, buffer.shape[1] / 2.0])
    sq_radius = circle_radius ** 2

    for u, v in buffer:
        pos = ti.Vector([u, v], dt=ti.float32) - center
        sq_norm = pos.dot(pos)
        if sq_norm > sq_radius:
            buffer[u, v] = 0.0
        else:
            buffer[u, v] = 1.0


@ti.kernel
def write_poses(pose: ti.template()):  # type: ignore
    pose[0, 0] = ti.Matrix(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )


@ti.kernel
def write_sphere(grid: ti.template(), grid_size: int):  # type: ignore
    span = ti.Vector([grid_size, grid_size, grid_size]) / 4
    sq_radius = span.norm_sqr()

    for x, y, z in grid:
        dist = ti.Vector([x, y, z]) - span
        if dist.norm_sqr() < sq_radius:
            grid[x, y, z].color = ti.Vector(
                [ti.random(), ti.random(), ti.random()]
            ).normalized()

            grid[x, y, z].opacity = ti.random()


def get_scene():
    # Build the Chinoxel scene
    # Settings here are completely arbitrary
    n_nodes = 20
    resolution = (800, 800)
    focal = 1 / resolution[0]
    scene = Scene(
        grid_size=(n_nodes, n_nodes, n_nodes),
        resolution=resolution,
        focal=focal,
        max_depth_ray=3,
        LR=0.1,
    )

    return scene, resolution


def demo_sphere_render():
    scene, resolution = get_scene()

    # Write a sphere in the scene, and check the rendering phase
    write_sphere(scene.grid, scene.grid_size[0])

    # Check the rendering
    gui = ti.GUI("Chinoxel", resolution[0], fast_gui=False)
    gui.fps_limit = 60

    while gui.running:
        scene.render()
        scene.tonemap()
        gui.set_image(scene.view_buffer)
        gui.show()


def demo_sphere_optimize(n_views: int = 10):
    scene, resolution = get_scene()

    # Generate the synthetic views
    views = []
    for _ in range(n_views):
        view = ti.field(dtype=ti.f32, shape=resolution)
        write_circle(view, 100.0)
        views.append(view)

    # Generate the synthetic camera poses
    poses = []
    for _ in range(n_views):
        # TODO: generate the virtual views,
        # turning around the sphere while looking at it
        pose = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(1, 1))
        write_poses(pose)
        poses.append(pose)

    # Optimize the scene on the {pose, views} set
    scene.optimize(poses, views, use_gui=True)


if __name__ == "__main__":
    ti.init(arch=ti.gpu)

    demo_sphere_render()
