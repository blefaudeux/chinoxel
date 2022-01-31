import taichi as ti
from typing import Tuple, List

ti.init(arch=ti.cpu)

EPS = 1e-4
INF = 1e10

# Good resource:
# https://yuanming.taichi.graphics/publication/2020-taichi-tutorial/taichi-tutorial.pdf

_DEBUG = False


@ti.data_oriented
class Scene:
    def __init__(
        self,
        grid_size: Tuple[int, int, int],
        resolution: Tuple[int, int],
        focal: float,
        max_depth_ray: int,
        LR: float = 0.1,
    ):

        # For now just store a scalar value in the grid, as a follow up
        # we should store spherical harmonics
        self.grid = ti.field(dtype=ti.f32, shape=grid_size, needs_grad=True)
        self.grid_node_pos = ti.Vector.field(
            n=3, dtype=ti.f32, shape=grid_size, needs_grad=False
        )
        self.diff_field = ti.field(
            dtype=ti.f32, shape=self.grid_node_pos.shape, needs_grad=False
        )

        # Render view
        self.camera_pose = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(1, 1))
        self.view_buffer = ti.field(dtype=ti.f32, shape=resolution)
        self.reference_buffer = None

        self.max_depth_ray = max_depth_ray
        self.focal = focal
        self.res = resolution
        self.aspect_ratio = resolution[0] / resolution[1]

        # Optimization settings
        self.LR = LR
        self.loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        self.init()

    @ti.kernel
    def init(self):
        """Fill in the grid coordinates, becomes read-only after that.

        We arbitrarily constraint the grid to be smaller than [-1, 1]
        """
        center = ti.Vector(self.grid_node_pos.shape, ti.float32) / 2.0
        scale = (
            self.grid_node_pos.shape[0] / 2.0
        )  # FIXME: the grid could have different sizes over different dimensions

        for x, y, z in self.grid_node_pos:
            self.grid_node_pos[x, y, z] = (
                ti.Vector([x, y, z], dt=ti.float32) - center
            ) / scale

        # Init the camera pose matrix
        # NOTE: This will be overwritten when optimizing for {view/poses}
        self.camera_pose[0, 0] = ti.Matrix(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

    @ti.func
    def get_ray(self, u, v):
        """
        Given coordinates in the image plane,
        get the corresponding incoming ray.

        Coordinates are:
        - x : u
        - y : v
        - z : positive towards the sensor

        (x,y,z) is direct
        """

        # Classic pinhole model
        u_ = u - ti.static(self.res[0] / 2)
        v_ = v - ti.static(self.res[1] / 2)

        d = ti.Matrix(
            [
                -self.focal * u_,
                -self.focal * v_,
                -1.0,
            ]
        )
        d = d.normalized()

        # Matmul with the camera pose to move to the reference coordinate
        # d_cam = self.camera_pose[0, 0] @ d  # FIXME
        return d

    @ti.func
    def closest_node(self, pose, ray):
        """Return the element of the grid which is the closest to this position

        ..note: this could be computed in one go with linear algebra,
            probably sub optimal

        .. note: instead of computing all the distances,
            probably that some hierarchical partitioning would be faster
        """

        min_dist = INF
        x_min, y_min, z_min = -1, -1, -1

        # - Batch compute all the directions to the nodes
        X, Y, Z = self.grid_node_pos.shape

        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    los = self.grid_node_pos[x, y, z] - pose
                    proj = los.dot(ray)
                    diff = (los - proj * ray).norm()

                    # Now keep track of the node which is the closest
                    if diff < min_dist and proj > 0.0:
                        min_dist = diff
                        x_min, y_min, z_min = x, y, z

        return x_min, y_min, z_min, min_dist

    @ti.kernel
    def render(self):
        """
        Given a camera pose, generate the corresponding view
        """

        for u, v in self.view_buffer:
            # Compute the ray direction
            pos = ti.Vector(
                [
                    self.camera_pose[0, 0][0, 3],
                    self.camera_pose[0, 0][1, 3],
                    self.camera_pose[0, 0][2, 3],
                ]
            )
            d = self.get_ray(u, v)

            #
            colour_acc = 0.0

            # Raymarch
            # while depth < self.max_depth_ray:
            # Find the next "hit"
            x, y, z, dist = self.closest_node(pos, d)

            if dist < INF:
                # Simple accumulation over the scalar for now
                colour_acc += self.grid[x, y, z]  # FIXME

                # Update the position and keep going
                # FIXME: This is not quite correct,
                # the ray may go close but not to the node really
                pos += dist * d  # self.grid_node_pos[x, y, z]

            self.view_buffer[u, v] = colour_acc

    @ti.kernel
    def gradient_descent(self):
        """
        Given the computed gradient, adjust all the elements of the grid
        .. note: worth implementing some momentum ?
        """
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

    def optimize(self, poses: List[ti.Matrix], views: List[ti.Vector], use_gui: False):  # type: ignore
        """
        Given a set of views and corresponding poses, optimize the underlying scene
        """

        assert len(poses) == len(views), "You must provide one camera pose per view"

        if use_gui:
            gui = ti.GUI("Chinoxel", self.res, fast_gui=False)
            gui.fps_limit = 60
        else:
            gui = None

        # Put something in the grid, to make sure we get some gradient
        self.grid.fill(0.0)

        while not gui or gui.running:
            for pose, view in zip(poses, views):
                with ti.Tape(loss=self.loss):
                    # project the grid on this viewpoint
                    self.view_buffer.fill(0.0)
                    self.camera_pose[0, 0] = pose[0, 0]
                    self.render()

                    # loss is this vs. the reference at that point
                    self.reference_buffer = view
                    self.compute_loss()

                # update the field
                print("Loss: ", self.loss[None])
                self.gradient_descent()

                if _DEBUG:
                    self.render()
                    self.loss[None] = 0
                    self.compute_loss()
                    print("post loss: ", self.loss[None])

                # dummy, show the current grid
                if use_gui:
                    gui.set_image(self.view_buffer)
                    gui.show()

                print("Frame processed")

                # TODO: sparsify ?
                # TODO: Adjust LR ?
