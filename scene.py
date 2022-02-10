import taichi as ti
from typing import Tuple, List


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

        # For now just store [RGBA] in the grid
        # TODO: follow up we should store spherical harmonics

        self.grid = ti.Struct.field(
            {"color": ti.types.vector(3, ti.f32), "opacity": ti.f32,}, shape=grid_size,
        )
        self.grid.color.fill(0.0)  # this is legit in taichi, and pretty wunderbar
        self.grid.opacity.fill(1.0)

        self.grid_node_pos = ti.Vector.field(
            n=3, dtype=ti.f32, shape=grid_size, needs_grad=False
        )
        self.grid_size = grid_size

        # Render view
        self.camera_pose = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(1, 1))
        self.view_buffer = ti.Vector.field(n=3, dtype=ti.f32, shape=resolution)

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

        d = ti.Matrix([-self.focal * u_, -self.focal * v_, -1.0,])
        d = d.normalized()

        # Matmul with the camera pose to move to the reference coordinate
        # d_cam = self.camera_pose[0, 0] @ d  # FIXME
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
        X, Y, Z = self.grid_node_pos.shape
        diff_min = ti.Vector([0.0, 0.0, 0.0])

        for x in range(X):
            for y in range(Y):
                for z in range(Z):

                    # Compute the direct line of sight, then offset
                    los = self.grid_node_pos[x, y, z] - pose
                    proj = los.dot(ray)

                    # Check that this point is not backwards
                    if proj > 0.0:
                        diff = proj * ray - los
                        diff_norm = diff.norm()

                        # Now keep track of the node which is the closest
                        if diff_norm < min_dist:
                            min_dist = diff_norm
                            diff_min = diff
                            x_min, y_min, z_min = x, y, z

        return min_dist < INF, ti.Vector([x_min, y_min, z_min], ti.i32), diff_min

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
                    dist = (self.grid_node_pos[x, y, z] - pose).norm()

                    # Now keep track of the node which is the closest
                    if dist < min_dist:
                        min_dist = dist
                        x_min, y_min, z_min = x, y, z

        return min_dist < INF, ti.Vector([x_min, y_min, z_min], ti.i32)

    @ti.func
    def contrib(self, pos, x, y, z):
        dist = (pos - self.grid_node_pos[x, y, z]).norm()

        return self.grid[x, y, z].opacity * dist

    @ti.func
    def w_contrib(self, norm, c, x, y, z):
        return ti.exp(-norm) * (1.0 - ti.exp(-c)) * self.grid[x, y, z].color

    @ti.func
    def interpolate(self, close, pos, acc_norm):
        # Find the 8 closest nodes
        # we have the indices of the closest node, and the point where the
        # ray is right now
        diff = pos - self.grid_node_pos[close[0], close[1], close[2]]

        dx = 1 if diff[0] > 0 else -1
        dy = 1 if diff[1] > 0 else -1
        dz = 1 if diff[2] > 0 else -1

        # 8 closest nodes are:
        # NOTE: This is a bit verbose, but completely unrolled and
        # only touches the right parts, should not be too bad
        acc = ti.Vector([0.0, 0.0, 0.0,])

        # TODO: Rewrite, this is horrible
        x_, y_, z_ = close[0], close[1], close[2]

        c = self.contrib(pos, x_, y_, z_)
        acc_norm += c
        acc += self.w_contrib(acc_norm, c, x_, y_, z_)

        x_ += dx
        c = self.contrib(pos, x_, y_, z_)
        acc_norm += c
        acc += self.w_contrib(acc_norm, c, x_, y_, z_)

        y_ += dy
        c = self.contrib(pos, x_, y_, z_)
        acc_norm += c
        acc += self.w_contrib(acc_norm, c, x_, y_, z_)

        x_ -= dx
        c = self.contrib(pos, x_, y_, z_)
        acc_norm += c
        acc += self.w_contrib(acc_norm, c, x_, y_, z_)

        y_ -= dy
        z_ -= dz
        c = self.contrib(pos, x_, y_, z_)
        acc_norm += c
        acc += self.w_contrib(acc_norm, c, x_, y_, z_)

        x_ += dx
        c = self.contrib(pos, x_, y_, z_)
        acc_norm += c
        acc += self.w_contrib(acc_norm, c, x_, y_, z_)

        y_ += dy
        c = self.contrib(pos, x_, y_, z_)
        acc_norm += c
        acc += self.w_contrib(acc_norm, c, x_, y_, z_)

        x_ -= dx
        c = self.contrib(pos, x_, y_, z_)
        acc_norm += c
        acc += self.w_contrib(acc_norm, c, x_, y_, z_)

        return acc, acc_norm

    @ti.kernel
    def tonemap(self):
        """
        For now, just normalized the rendered view.
        Could be worth it applying a gamma curve for instance
        """
        for i, j in self.view_buffer:
            self.view_buffer[i, j] = ti.sqrt(self.view_buffer[i, j])

    @ti.kernel
    def render(self):
        """
        Given a camera pose, generate the corresponding view
        """

        # NOTE: assuming an isotropic grid
        cell_size = self.grid_node_pos[1, 0, 0] - self.grid_node_pos[1, 0, 0]

        for u, v in self.view_buffer:
            # Compute the initial ray direction
            pos = ti.Vector(
                [
                    self.camera_pose[0, 0][0, 3],
                    self.camera_pose[0, 0][1, 3],
                    self.camera_pose[0, 0][2, 3],
                ]
            )
            ray = self.get_ray(u, v)
            ray_abs_max = ti.max(ray.max(), -ray.min())
            ray_step = ray / ray_abs_max * cell_size  # unitary on one direction

            # Ray marching variables
            steps = 0
            colour_acc = ti.Vector([0.0, 0.0, 0.0,])
            norm_acc = 0.0

            # First, intersection
            hit, close, diff = self.intersect(pos, ray)

            pos = self.grid_node_pos[close[0], close[1], close[2]] + diff

            # After that, marching cubes
            while hit and steps < self.max_depth_ray:
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
