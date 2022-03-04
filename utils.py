import taichi as ti


@ti.func
def get_matrix_from_euler(alpha, beta, gamma):
    """
    Get a rotation matrix from the three euler angles

    ... note: Euler angles are ill-defined, in that the effects depend on the rotation ordering.
    We stick to an arbitrary order here
    """
    rotation_alpha = ti.Matrix(
        [
            [
                ti.cos(alpha),
                ti.sin(alpha),
                0.0,
            ],
            [
                -ti.sin(alpha),
                ti.cos(alpha),
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ]
    )

    rotation_beta = ti.Matrix(
        [
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                ti.cos(beta),
                ti.sin(beta),
            ],
            [
                0.0,
                -ti.sin(beta),
                ti.cos(beta),
            ],
        ]
    )

    rotation_gamma = ti.Matrix(
        [
            [
                ti.cos(gamma),
                0.0,
                ti.sin(gamma),
            ],
            [
                0.0,
                1.0,
                0.0,
            ],
            [-ti.sin(gamma), 0.0, ti.cos(gamma)],
        ]
    )

    return rotation_alpha @ rotation_beta @ rotation_gamma


@ti.func
def clamp(val) -> ti.f32:
    return ti.max(ti.min(val, 1.0), 0.0)
