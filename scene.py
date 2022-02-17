import taichi as ti
from typing import Tuple

from geometry import get_matrix_from_euler

EPS = 1e-4
INF = 1e10

# Good resource:
# https://yuanming.taichi.graphics/publication/2020-taichi-tutorial/taichi-tutorial.pdf


# Remove taichi compiler intrinsics from the stack trace
_dev = True
if _dev:
    import sys

    sys.tracebacklimit = 0


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
        self.grid.opacity.fill(0.5)

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

        # Store the grid gradients.
        # This will be fused with the rendering computation if `trace_rendering` is set
        # Note that several kernels can concurrently increment the grid gradient,
        # so write accesses need to be atomic
        self.grid_grad = ti.Struct.field(
            {"color": ti.types.vector(3, ti.f32), "opacity": ti.f32, "valid": ti.i8},
            shape=grid_size,
        )

        # Allocate the stack which will host the backward tracing
        # Keep this sparse on purpose, at any point in time most of these
        # will be empty
        self.grad_stack = ti.root.pointer(ti.ij, resolution)
        grad_node = ti.Struct.field(
            {
                "color_grad": ti.f32,
                "opacity_grad": ti.f32,
                "pose": ti.types.vector(3, ti.i32),
            }
        )
        max_node_callstack = 8 * self.max_depth_ray
        self.node_grads = self.grad_stack.dense(ti.i, (max_node_callstack))
        self.node_grads.place(grad_node)

        self.reset_grads()
        self.init()

    @ti.kernel
    def init(self):
        """Fill in the grid coordinates, becomes read-only after that.

        We arbitrarily constraint the grid to be smaller than [-1, 1].
        The whole space is normalized by convention
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

    @ti.kernel
    def orbital_inc_rotate(self):
        """
        Rotate the grid by an angular increment.

        ... note: The rotation axis is arbitrary, could probably passed as a param eventually
        """
        inc = 1.0 / 180 * 3.1415  # 15 degrees ?

        rotation_increment = get_matrix_from_euler(inc, inc, 0.0)

        self.camera_pose.rotation[None] = (
            rotation_increment @ self.camera_pose.rotation[None]
        )

        self.camera_pose.translation[None] = (
            rotation_increment @ self.camera_pose.translation[None]
        )

    def reset_grads(self):
        self.grid_grad.color.fill(0.0)
        self.grid_grad.opacity.fill(0.0)
        self.grid_grad.valid.fill(0)
        self.loss[None] = 0.0

    @ti.kernel
    def tonemap(self):
        """
        Tonemap the render buffer, optional

        ... note: Could be worth applying a better gamma curve, maybe reset black level, ..
        """

        for i, j in self.view_buffer:
            self.view_buffer[i, j] = ti.sqrt(self.view_buffer[i, j])

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
        return d.normalized()

    @ti.func
    def intersect(self, pose, ray):
        """Return the element of the grid which is the closest to the ray

        ... note: this could be computed in one go with linear algebra,
            probably sub optimal

        ... note: instead of computing all the distances,
            probably that some hierarchical partitioning would be faster
        """

        cell_size = 0.5 * (self.grid[0, 0, 0].pose - self.grid[0, 0, 1].pose).norm_sqr()

        min_proj = INF
        i_min = ti.Vector([-1, -1, -1])
        diff_vec_min = ti.Vector([0.0, 0.0, 0.0])

        # - Batch compute all the directions to the nodes
        for x in range(self.grid_size[0]):  # this will be parallelized
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    if self.grid[x, y, z].opacity == 0.0:
                        continue

                    # Compute the direct line of sight, then offset
                    line_of_sight = self.grid[x, y, z].pose - pose
                    projection = line_of_sight.dot(ray)

                    # Check that this point is not backwards
                    if projection > 0.0:
                        # This the the distance to the node, but it cannot be a good metric
                        # for an intersection alone,
                        # since it could be at the back of the grid for all we know.

                        # So we use the distance to this shortest path to find the
                        # node that this ray first interacts with
                        diff_vec = projection * ray - line_of_sight
                        diff = diff_vec.norm_sqr()

                        # Now keep track of the node which is the closest
                        if diff < cell_size and projection < min_proj:
                            min_proj = projection
                            diff_vec_min = diff_vec
                            i_min = ti.Vector([x, y, z])

        return i_min, diff_vec_min

    @staticmethod
    @ti.func
    def get_bounds(start, ceil):
        return max(start - 1, 0), min(start + 1, ceil)

    @ti.func
    def closest_node(self, pose, previous_node):
        """Return the element of the grid which is the next closest to the ray

        ... note: this could be computed in one go with linear algebra,
            probably sub optimal

        ... note: instead of computing all the distances,
            probably that some hierarchical partitioning would be faster
        """

        min_dist = INF

        X, Y, Z = self.grid_size

        x_, y_, z_ = previous_node[0], previous_node[1], previous_node[2]
        x_min, x_max = self.get_bounds(x_, X)
        y_min, y_max = self.get_bounds(y_, Y)
        z_min, z_max = self.get_bounds(z_, Z)

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                for z in range(z_min, z_max):
                    dist = (self.grid[x, y, z].pose - pose).norm()

                    # Now keep track of the node which is the closest
                    if dist < min_dist:
                        min_dist = dist
                        x_, y_, z_ = x, y, z

        return min_dist < INF, ti.Vector([x_, y_, z_], ti.i32)

    @ti.func
    def d_contrib(self, pos, close):
        x, y, z = close
        dist = (pos - self.grid[x, y, z].pose).norm()
        return self.grid[x, y, z].opacity * dist, dist

    @ti.func
    def c_contrib(self, norm, dist_contribution, close):
        x, y, z = close
        return (
            ti.exp(-norm)
            * (1.0 - ti.exp(-dist_contribution))
            * self.grid[x, y, z].color
        )

    @ti.func
    def interpolate(self, close, pos, color_acc, norm_acc):
        distance_contrib, dist = self.d_contrib(pos, close)
        color_contrib = self.c_contrib(norm_acc, distance_contrib, close)

        color_acc += color_contrib * self.grid[close[0], close[1], close[2]].color
        norm_acc += distance_contrib

        return color_acc, norm_acc, dist, color_contrib

    @ti.kernel
    def render(self):
        """
        Given a camera pose and the scene knowledge baked in the grid,
        generate the corresponding view
        """

        cell_size = (self.grid[1, 0, 0].pose - self.grid[0, 0, 0].pose).norm()

        for u, v in self.view_buffer:
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

            # After that, something a bit like marching cubes
            hit = close[0] > 0
            for steps in range(self.max_depth_ray):
                if not hit:
                    break

                # Fetch the colour in between the 8 closest nodes
                # we have the indices of the closest node, and the point where the
                # ray is right now
                diff = pos - self.grid[close[0], close[1], close[2]].pose

                dx = 1 if diff[0] > 0 else -1
                dy = 1 if diff[1] > 0 else -1
                dz = 1 if diff[2] > 0 else -1

                colour_acc, norm_acc, dc, cc = self.interpolate(
                    close, pos, colour_acc, norm_acc
                )
                # if self.trace_rendering:
                #     node_grads[i_node].color_grad = cc
                #     node_grads[i_node].opacity_grad = dc
                #     node_grads[i_node].pose = close
                #     i_node += 1

                # TODO: store the grad stack directly in the corresponding SNodes
                # clean that up
                # TODO: the cube edges could be stored in a static pattern
                # this could be factorized
                close[0] += dx
                colour_acc, norm_acc, dc, cc = self.interpolate(
                    close, pos, colour_acc, norm_acc
                )

                close[1] += dy
                colour_acc, norm_acc, dc, cc = self.interpolate(
                    close, pos, colour_acc, norm_acc
                )

                close[0] -= dx
                colour_acc, norm_acc, dc, cc = self.interpolate(
                    close, pos, colour_acc, norm_acc
                )

                close[2] += dz
                colour_acc, norm_acc, dc, cc = self.interpolate(
                    close, pos, colour_acc, norm_acc
                )

                close[0] += dx
                colour_acc, norm_acc, dc, cc = self.interpolate(
                    close, pos, colour_acc, norm_acc
                )

                close[1] -= dy
                colour_acc, norm_acc, dc, cc = self.interpolate(
                    close, pos, colour_acc, norm_acc
                )

                close[0] -= dx
                colour_acc, norm_acc, dc, cc = self.interpolate(
                    close, pos, colour_acc, norm_acc
                )

                # Move to the next cell
                pos = pos + ray_step  # Update the reference position

                # Update the closest point
                pos += cell_size * ray
                hit, close = self.closest_node(pos, close)

                steps += 1

            self.view_buffer[u, v] = colour_acc

            # Fused BW pass
            if self.trace_rendering:
                # Per pixel RGB loss
                diff = self.reference_buffer[u, v] - self.view_buffer[u, v]

                # # Walk back the stack of contributions
                # for node in node_grads:
                #     (x, y, z) = node.pose
                #     cc = node.color_grad
                #     dc = node.opacity_grad

                #     # Warning: different threads can contribute to gradients
                #     # on the same node here, hence atomic adds are really required
                #     # TODO: compute the proper grads here
                #     # Formulas here are just placeholders
                #     ti.atomic_add(self.grid_grad[x, y, z].color, diff * cc)
                #     ti.atomic_add(self.grid_grad[x, y, z].opacity, diff.norm * dc)

    @ti.kernel
    def gradient_descent(self):
        """
        Given the computed gradient, adjust all the elements of the grid
        .. note: worth implementing some momentum ?
        """

        for x, y, z in self.grid_grad:
            if self.grid[x, y, z] > 0:
                self.grid[x, y, z].color -= self.LR * self.grid_grad[x, y, z].color
                self.grid[x, y, z].opacity -= self.LR * self.grid_grad[x, y, z].opacity

    def optimize(self, use_gui: False):  # type: ignore
        """
        Adapt the current scene to the current reference_buffer
        """

        if use_gui:
            gui = ti.GUI("Chinoxel", self.res, fast_gui=False)
            gui.fps_limit = 60
        else:
            gui = None

        # Put something in the grid, whatever
        # self.grid.fill(0.0)

        self.trace_rendering = True

        while not gui or gui.running:
            self.reset_grads()

            # Forward and backward passes are fused
            self.render()
            print("Loss: ", self.loss[None])

            # update the field
            self.gradient_descent()

            # dummy, show the current grid
            if use_gui:
                gui.set_image(self.view_buffer)
                gui.show()

            print("Frame processed")

        # Make sure that future render calls are not traced by default
        self.trace_rendering = False
