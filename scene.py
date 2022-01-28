import taichi as ti
from typing import Tuple, List

ti.init(arch=ti.gpu)

EPS = 1e-4
INF = 1e10


@ti.data_oriented
class Scene:
    def __init__(
        self,
        grid_size: Tuple[int, int, int],
        resolution: Tuple[int, int],
        fov: float,
        max_depth_ray: int,
        LR: float = 0.1,
    ):
        self.grid = ti.field(dtype=ti.f32, shape=grid_size, needs_grad=True)

        self.view_buffer = ti.field(dtype=ti.f32, shape=resolution)
        self.reference_buffer = None
        self.max_depth_ray = max_depth_ray
        self.fov = fov
        self.res = resolution
        self.aspect_ratio = resolution[0] / resolution[1]
        self.LR = LR
        self.loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.camera_pose = ti.Matrix(
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        )

    @ti.func
    def get_ray(self, u, v):
        # Classic pinhole model
        d = ti.Vector(
            [
                (
                    2 * self.fov * (u + ti.random()) / self.res[1]
                    - self.fov * self.aspect_ratio
                    - 1e-5
                ),
                2 * self.fov * (v + ti.random()) / self.res[1] - self.fov - 1e-5,
                -1.0,
                1.0,
            ]
        )
        d = d.normalized()

        # Matmul with the camera pose to move to the reference coordinate
        d_cam = self.camera_pose * d
        return ti.Vector([d_cam[0], d_cam[1], d_cam[2]])

    @ti.func
    def closest_node(self, position, ray):
        # Return the element of the grid which is the closest to this position
        # min_dist = inf

        # - Batch compute all the directions to the nodes
        # diff_field = field - position

        # # - closest one is the smallest positive
        # distances = ti.dot(diff_field, ray)

        # FIXME
        return self.grid[0, 0, 0]

    @ti.kernel
    def render(self):
        """
        Given a camera pose, generate the corresponding view
        """
        for u, v in self.view_buffer:
            # Compute the ray direction
            pos = self.camera_pose
            d = self.get_ray(u, v)

            #
            colour_acc = 1.0

            # Raymarch
            depth = 0

            while depth < self.max_depth_ray:
                node = self.closest_node(pos, d)
                colour_acc *= node  # FIXME
                depth += 1  # FIXME

            self.view_buffer[u, v] = colour_acc

    @ti.kernel
    def gradient_descent(self):
        for x, y, z in self.grid:
            self.grid[x, y, z] -= self.grid.grad[x, y, z] * self.LR

    @ti.kernel
    def compute_loss(self):
        """
        Given a reference view, create a loss by comparing it to the current view_buffer

        .. note: this is most probably not optimal in terms of speed, this could be 
            rewritten as a matmultiplications
        """
        for u, v in self.view_buffer:
            self.loss[None] += (
                self.view_buffer[u, v] - self.reference_buffer[u, v]
            ) ** 2

    @ti.kernel
    def random_grid(self):
        # FIXME: there's probably a primitive for that
        for x, y, z in self.grid:
            self.grid[x, y, z] = ti.random(ti.float32)

    def optimize(self, poses: List[ti.Matrix], views: List[ti.Vector]):
        """
        Given a set of views and corresponding poses, optimize the underlying scene
        """

        assert len(poses) == len(views), "You must provide one camera pose per view"

        # Initialize the grid with random data, the dummy way
        self.random_grid()

        for pose, view in zip(poses, views):
            # project the grid on this viewpoint
            self.view_buffer.fill(0.0)
            self.camera_pose = pose
            self.render()

            # loss is this vs. the reference at that point
            self.loss[None] = 0
            with ti.Tape(loss=self.loss):
                self.reference_buffer = view
                self.compute_loss()

            # update the field
            self.gradient_descent()

            # dummy, show the current grid
            print(self.grid)

            # TODO: sparsify ?
            # TODO: Adjust LR ?
