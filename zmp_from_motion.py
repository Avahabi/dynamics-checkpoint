"""Compute how momentum rates shift the ZMP away from the CoM projection."""

from __future__ import annotations

import numpy as np


def calc_zmp(
    total_mass: float,
    gravity: float,
    com: np.ndarray,
    dP: np.ndarray,
    dL: np.ndarray,
    pz: float = 0.0,
) -> np.ndarray:
    vertical_reaction = total_mass * gravity + dP[2]
    if abs(vertical_reaction) < 1e-9:
        raise ValueError("The vertical reaction force is too small for a stable ZMP calculation.")

    px = (total_mass * gravity * com[0] + pz * dP[0] - dL[1]) / vertical_reaction
    py = (total_mass * gravity * com[1] + pz * dP[1] + dL[0]) / vertical_reaction
    return np.array([px, py, pz])


def main() -> None:
    mass = 32.0
    gravity = 9.81
    com = np.array([0.035, -0.015, 0.82])
    momentum_rate = np.array([14.4, -6.4, 1.6])

    no_angular_momentum_rate = np.array([0.0, 0.0, 0.0])
    arm_swing_angular_momentum_rate = np.array([1.20, -2.40, 0.15])

    com_projection = np.array([com[0], com[1], 0.0])
    zmp_without_arm_swing = calc_zmp(
        mass,
        gravity,
        com,
        momentum_rate,
        no_angular_momentum_rate,
    )
    zmp_with_arm_swing = calc_zmp(
        mass,
        gravity,
        com,
        momentum_rate,
        arm_swing_angular_momentum_rate,
    )

    print("CoM projection [m]:", np.round(com_projection, 4))
    print("momentum rate dP [N]:", np.round(momentum_rate, 4))
    print("angular momentum rate without arm swing [N m]:", np.round(no_angular_momentum_rate, 4))
    print("ZMP without arm swing [m]:", np.round(zmp_without_arm_swing, 4))
    print("angular momentum rate with arm swing [N m]:", np.round(arm_swing_angular_momentum_rate, 4))
    print("ZMP with arm swing [m]:", np.round(zmp_with_arm_swing, 4))
    print("ZMP shift caused by arm swing [m]:", np.round(zmp_with_arm_swing - zmp_without_arm_swing, 4))


if __name__ == "__main__":
    main()
