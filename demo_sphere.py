from scene import Scene
import taichi as ti
import argparse


@ti.kernel
def write_circle(buffer: ti.template(), circle_radius: float):  # type: ignore
    """
    For optimization testing purposes,
    write a white circle in the frame buffer
    """
    center = ti.Vector([buffer.shape[0] / 2.0, buffer.shape[1] / 2.0])
    sq_radius = circle_radius**2

    for u, v in buffer:
        pos = ti.Vector([u, v], dt=ti.float32) - center
        sq_norm = pos.dot(pos)
        if sq_norm > sq_radius:
            buffer[u, v] = ti.Vector([0.0, 0.0, 0.0])
        else:
            buffer[u, v] = ti.Vector([1.0, 1.0, 0.0])


@ti.kernel
def write_sphere(grid: ti.template(), grid_size: int):  # type: ignore
    """
    For rendering testing purposes, write a sphere in the grid with random color
    and random opacity
    """
    span = ti.Vector([grid_size, grid_size, grid_size]) / 2
    sq_radius = (grid_size / 4) ** 2

    for x, y, z in grid:
        dist = ti.Vector([x, y, z]) - span
        if dist.norm_sqr() < sq_radius:
            grid[x, y, z].color = ti.Vector(
                [ti.random(), ti.random(), ti.random()]
            ).normalized()

            grid[x, y, z].opacity = ti.random()
        else:
            grid[x, y, z].opacity = 0.0
            grid[x, y, z].color = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def get_pose() -> ti.Struct:
    pose = ti.Struct(
        {
            "rotation": ti.Matrix(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "translation": ti.Vector([0.0, 0.0, 3.0]),
        }
    )

    return pose


def get_scene():
    # Build the Chinoxel scene
    # Settings here are completely arbitrary
    n_nodes = 10
    resolution = (480, 480)
    focal = 1.0 / resolution[0]
    scene = Scene(
        grid_size=(n_nodes, n_nodes, n_nodes),
        resolution=resolution,
        focal=focal,
        max_depth_ray=1,
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
        scene.orbital_inc_rotate()


def demo_sphere_optimize():
    scene, resolution = get_scene()

    # Check the rendering
    gui = ti.GUI("Chinoxel", resolution[0], fast_gui=False)
    gui.fps_limit = 60

    # Generate the synthetic view
    write_circle(scene.reference_buffer, 100.0)

    # Optimize the scene given the view
    i_step = 0
    max_steps = 5

    while gui.running:
        # learn for a while, then render
        if i_step < max_steps:
            scene.optimize()
            print(f"Optim: {i_step}/{max_steps} done")
        else:
            scene.render()
            scene.tonemap()

        i_step += 1
        gui.set_image(scene.view_buffer)
        gui.show()
        scene.orbital_inc_rotate()


if __name__ == "__main__":
    ti.init(arch=ti.gpu)

    parser = argparse.ArgumentParser("Chinoxel demo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--render", action="store_true")
    group.add_argument("--optimize", action="store_false")
    args = parser.parse_args()

    if args.render:
        demo_sphere_render()
    else:
        demo_sphere_optimize()
