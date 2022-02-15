from xml.etree.ElementInclude import include
import taichi as ti
from typing import Tuple, List
import math


EPS = 1e-4
INF = 1e10

# Good resource:
# https://yuanming.taichi.graphics/publication/2020-taichi-tutorial/taichi-tutorial.pdf

_DEBUG = True


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
        assert (
            grid_size[0] == grid_size[1] == grid_size[2]
        ), "We only support isotropic grids for now"

        # For now just store [RGBA] in the grid
        # TODO:
        # - follow up we should store spherical harmonics
        # - follow up store in a sparse structure
        # (see https://docs.taichi.graphics/lang/articles/advanced/sparse)

        self.grid = ti.Struct.field(
            {
                "color": ti.types.vector(3, ti.f32),
                "opacity": ti.f32,
                "pose": ti.types.vector(3, ti.f32),
            },
            shape=grid_size,
        )
        self.grid.color.fill(0.0)  # this is legit in taichi, and pretty wunderbar
        self.grid.opacity.fill(1.0)

        self.grid_size = grid_size

        # Render view
        self.camera_pose = ti.Struct(
            {
                "rotation": ti.Matrix.field(3, 3, ti.f32, shape=()),
                "translation": ti.Vector.field(3, ti.f32, shape=()),
            }
        )

        self.view_buffer = ti.Vector.field(n=3, dtype=ti.f32, shape=resolution)
        self.reference_buffer = ti.Vector.field(n=3, dtype=ti.f32, shape=resolution)

        self.max_depth_ray = max_depth_ray
        self.focal = focal
        self.res = resolution
        self.aspect_ratio = resolution[0] / resolution[1]

        # Optimization settings
        self.LR = LR
        self.loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        # Manual gradient tracing
        self.trace_rendering = False
        self.grid_grad = ti.Struct.field(
            {"color": ti.types.vector(3, ti.f32), "opacity": ti.f32,}, shape=grid_size,
        )

        self.euler = ti.Vector([0.0, 0.0, 0.0])
        self.init()

    @ti.kernel
    def init(self):
        """Fill in the grid coordinates, becomes read-only after that.

        We arbitrarily constraint the grid to be smaller than [-1, 1]
        """
        center = ti.Vector(self.grid_size, ti.float32) / 2.0
        scale = self.grid_size[0] / 2.0

        for x, y, z in self.grid:
            self.grid[x, y, z].pose = (
                ti.Vector([x, y, z], dt=ti.float32) - center
            ) / scale

        # Init the camera pose matrix
        self.camera_pose.rotation[None] = ti.Matrix(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],]
        )

        self.camera_pose.translation[None] = ti.Vector([0.0, 0.0, 3.0])

    @staticmethod
    @ti.func
    def get_matrix_from_euler(alpha, beta, gamma):
        """
        Euler angles are ill-defined, in that the effects depend on the rotation ordering.
        We stick to an arbitrary order here
        """
        rotation_alpha = ti.Matrix(
            [
                [ti.cos(alpha), ti.sin(alpha), 0.0,],
                [-ti.sin(alpha), ti.cos(alpha), 0.0,],
                [0.0, 0.0, 1.0,],
            ]
        )

        rotation_beta = ti.Matrix(
            [
                [1.0, 0.0, 0.0,],
                [0.0, ti.cos(beta), ti.sin(beta),],
                [0.0, -ti.sin(beta), ti.cos(beta),],
            ]
        )

        rotation_gamma = ti.Matrix(
            [
                [ti.cos(gamma), 0.0, ti.sin(gamma),],
                [0.0, 1.0, 0.0,],
                [-ti.sin(gamma), 0.0, ti.cos(gamma)],
            ]
        )

        return rotation_alpha @ rotation_beta @ rotation_gamma

    @ti.kernel
    def orbital_inc_rotate(self):
        inc = 1.0 / 180 * 3.1415  # 15 degrees ?

        rotation_increment = self.get_matrix_from_euler(inc, inc, 0.0)

        self.camera_pose.rotation[None] = (
            rotation_increment @ self.camera_pose.rotation[None]
        )

        self.camera_pose.translation[None] = rotation_increment @ self.camera_pose.translation[None] 


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

        d = ti.Matrix([-self.focal * u_, -self.focal * v_, -1.0])
        d = d.normalized()

        # Matmul with the camera pose to move to the reference coordinate
        d = self.camera_pose.rotation[None] @ d
        return d

    @ti.func
    def intersect(self, pose, ray):
        """Return the element of the grid which is the closest to the ray

        ..note: this could be computed in one go with linear algebra,
            probably sub optimal

        .. note: instead of computing all the distances,
            probably that some hierarchical partitioning would be faster
        """

        min_dist = INF
        x_min, y_min, z_min = -1, -1, -1

        # - Batch compute all the directions to the nodes
        diff_min = ti.Vector([0.0, 0.0, 0.0])

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):

                    # Compute the direct line of sight, then offset
                    line_of_sight = self.grid[x, y, z].pose - pose
                    proj = line_of_sight.dot(ray)

                    # Check that this point is not backwards
                    if proj > 0.0:
                        diff = proj * ray - line_of_sight
                        diff_norm = diff.norm()

                        # Now keep track of the node which is the closest
                        if diff_norm < min_dist:
                            min_dist = diff_norm
                            diff_min = diff
                            x_min, y_min, z_min = x, y, z

        return ti.Vector([x_min, y_min, z_min]), diff_min

    @ti.func
    def closest_node(self, pose, previous_node):
        """Return the element of the grid which is the next closest to the ray

        ..note: this could be computed in one go with linear algebra,
            probably sub optimal

        .. note: instead of computing all the distances,
            probably that some hierarchical partitioning would be faster
        """

        min_dist = INF

        X, Y, Z = self.grid_size

        x_min = max(previous_node[0] - 1, 0)
        x_max = min(previous_node[0] + 1, X)

        y_min = max(previous_node[1] - 1, 0)
        y_max = min(previous_node[1] + 1, Y)

        z_min = max(previous_node[2] - 1, 0)
        z_max = min(previous_node[2] + 1, Z)

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                for z in range(z_min, z_max):
                    dist = (self.grid[x, y, z].pose - pose).norm()

                    # Now keep track of the node which is the closest
                    if dist < min_dist:
                        min_dist = dist
                        x_min, y_min, z_min = x, y, z

        return min_dist < INF, ti.Vector([x_min, y_min, z_min], ti.i32)

    @ti.func
    def contrib(self, pos, x, y, z):
        dist = (pos - self.grid[x, y, z].pose).norm()

        return self.grid[x, y, z].opacity * dist

    @ti.func
    def w_contrib(self, norm, c, x, y, z):
        return ti.exp(-norm) * (1.0 - ti.exp(-c)) * self.grid[x, y, z].color

    @ti.func
    def interpolate(self, close, pos, carry):
        # Find the 8 closest nodes
        # we have the indices of the closest node, and the point where the
        # ray is right now
        diff = pos - self.grid[close[0], close[1], close[2]].pose

        dx = 1 if diff[0] > 0 else -1
        dy = 1 if diff[1] > 0 else -1
        dz = 1 if diff[2] > 0 else -1

        # 8 closest nodes are:
        # NOTE: This is a bit verbose, but completely unrolled and
        # only touches the right parts, should not be too bad
        acc = ti.Vector([0.0, 0.0, 0.0,])

        # TODO: Rewrite, this is horrible
        c0 = self.contrib(pos, close[0], close[1], close[2])
        acc += self.w_contrib(carry, c0, close[0], close[1], close[2])
        carry += c0

        # FIXME: The following breaks autodiff, not variable shadowing, not super clear why
        c1 = self.contrib(pos, close[0] + dx, close[1], close[2])
        acc += self.w_contrib(carry, c1, close[0] + dx, close[1], close[2])
        carry += c1

        c2 = self.contrib(pos, close[0] + dx, close[1] + dy, close[2])
        acc += self.w_contrib(carry, c2, close[0] + dx, close[1] + dy, close[2])
        carry += c2

        c3 = self.contrib(pos, close[0], close[1] + dy, close[2])
        acc += self.w_contrib(carry, c3, close[0], close[1] + dy, close[2])
        carry += c3

        c4 = self.contrib(pos, close[0], close[1], close[2] - dz)
        acc += self.w_contrib(carry, c4, close[0], close[1], close[2] - dz)
        carry += c4

        c5 = self.contrib(pos, close[0] + dx, close[1], close[2] - dz)
        acc += self.w_contrib(carry, c5, close[0] + dx, close[1], close[2] - dz)
        carry += c5

        c6 = self.contrib(pos, close[0] + dx, close[1] + dy, close[2] - dz)
        acc += self.w_contrib(carry, c6, close[0] + dx, close[1] + dy, close[2] - dz)
        carry += c6

        c7 = self.contrib(pos, close[0], close[1] + dy, close[2] - dz)
        acc += self.w_contrib(carry, c7, close[0], close[1] + dy, close[2] - dz)
        carry += c7

        return acc, carry

    @ti.kernel
    def tonemap(self):
        """
        Could be worth applying a better gamma curve
        """
        for i, j in self.view_buffer:
            self.view_buffer[i, j] = ti.sqrt(self.view_buffer[i, j])

    @ti.kernel
    def render(self):
        """
        Given a camera pose, generate the corresponding view
        """

        for u, v in self.view_buffer:
            cell_size = (self.grid[1, 0, 0].pose - self.grid[0, 0, 0].pose).norm()

            # Compute the initial ray direction
            pos = self.camera_pose.translation[None]
            ray = self.get_ray(u, v)
            ray_abs_max = ti.max(ray.max(), -ray.min())
            ray_step = ray / ray_abs_max * cell_size  # unitary on one direction

            # # Ray marching variables
            colour_acc = ti.Vector([0.0, 0.0, 0.0,])
            norm_acc = 0.0

            # First, intersection
            close, diff = self.intersect(pos, ray)
            pos = self.grid[close[0], close[1], close[2]].pose + diff
            hit = diff.norm() < cell_size

            # After that, marching cubes
            for steps in range(self.max_depth_ray):
                if not hit:
                    break

                # Fetch the colour in between the 8 closest nodes
                contrib, norm_acc = self.interpolate(close, pos, norm_acc)
                colour_acc += contrib

                # Move to the next cell
                pos = pos + ray_step  # Update the reference position

                # Update the closest point
                pos += cell_size * ray
                hit, close = self.closest_node(pos, close)

                steps += 1

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
    def reduce(self):
        """
        Given a reference view, create a loss by comparing it to the current view_buffer

        .. note: this is most probably not optimal in terms of speed, this could be
            rewritten as a matmultiplications
        """

        for u, v in self.view_buffer:
            self.view_buffer[u, v] -= self.reference_buffer[u, v]

    @ti.kernel
    def compute_loss(self):
        """
        Given a reference view, create a loss by comparing it to the current view_buffer

        .. note: this is most probably not optimal in terms of speed, this could be
            rewritten as a matmultiplications
        """
        for u, v in self.view_buffer:
            self.loss[None] += self.view_buffer[u, v].norm_sqr()

    def optimize(self, use_gui: False):  # type: ignore
        """
        Adapt the current scene to the current reference_buffer
        """

        if use_gui:
            gui = ti.GUI("Chinoxel", self.res, fast_gui=False)
            gui.fps_limit = 60
        else:
            gui = None

        # Put something in the grid, to make sure we get some gradient
        self.grid.fill(0.0)

        while not gui or gui.running:
            with ti.Tape(loss=self.loss):
                self.render()

                # loss is this vs. the reference at that point
                self.reduce()
                self.compute_loss()

            # update the field
            print("Loss: ", self.loss[None])
            # self.gradient_descent()

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
