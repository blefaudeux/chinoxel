from scene import Scene
import taichi as ti


@ti.kernel
def write_circle(buffer: ti.template(), circle_radius: float):
    center = ti.Vector([buffer.shape[0] / 2.0, buffer.shape[1] / 2.0])
    sq_radius = circle_radius**2

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


def demo_sphere(n_views: int = 10):
    # Build the Chinoxel scene
    n_nodes = 20
    resolution = (320, 320)
    focal = 1 / 320.0
    scene = Scene(
        grid_size=(n_nodes, n_nodes, n_nodes),
        resolution=resolution,
        focal=focal,
        max_depth_ray=1,
        LR=0.1,
    )

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
    demo_sphere()
