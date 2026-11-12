"""Estimate momentum and ZMP while the toy biped moves."""

from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import numpy as np

from inverse_kinematics import Pose, calc_jacobian, find_route, inverse_kinematics
from kinematics import forward_kinematics
from robot_model import BODY, LLEG_J2, LLEG_J3, LLEG_J4, LLEG_J5, RLEG_J2, RLEG_J3, RLEG_J4, RLEG_J5, setup_biped_robot
from visualization import draw_all_joints, set_biped_axes
from zmp_from_motion import calc_zmp


GRAVITY = 9.8
D_TIME = 0.01
END_TIME = 0.5


def lipm(t: float, origin: float, x0: float, tc: float) -> tuple[float, float]:
    """Return the body position and velocity used to drive the demo motion."""
    position = (x0 - origin) * np.cosh(t / tc) + origin
    velocity = (x0 - origin) / tc * np.sinh(t / tc)
    return float(position), float(velocity)


def link_com_world(link) -> np.ndarray:
    """Estimate a link CoM using the geometry available in the course model."""
    if link.vertex is not None:
        return link.p + link.R @ np.mean(link.vertex, axis=0)
    return link.p.copy()


def total_mass(links: list) -> float:
    """Return the total toy-biped mass."""
    return float(sum(link.mass for link in links[1:] if link is not None))


def calc_com(links: list) -> np.ndarray:
    """Calculate the whole-body CoM from the course link model."""
    mass_moment = np.zeros(3)
    for link in links[1:]:
        if link is not None:
            mass_moment += link.mass * link_com_world(link)
    return mass_moment / total_mass(links)


def link_com_positions(links: list) -> np.ndarray:
    """Return estimated world CoM positions for every link."""
    return np.array([link_com_world(link) for link in links[1:] if link is not None])


def link_masses(links: list) -> np.ndarray:
    """Return link masses for finite-difference momentum estimates."""
    return np.array([link.mass for link in links[1:] if link is not None], dtype=float)


def local_com(link) -> np.ndarray:
    """Return the local CoM convention used by the course model geometry."""
    if link.vertex is not None:
        return np.mean(link.vertex, axis=0)
    return np.zeros(3)


def local_inertia(link) -> np.ndarray:
    """Estimate box inertia for visible course links and zero otherwise."""
    if link.vertex is None:
        return np.zeros((3, 3))
    extents = np.ptp(link.vertex, axis=0)
    x, y, z = extents
    return link.mass * np.diag([y**2 + z**2, x**2 + z**2, x**2 + y**2]) / 12.0


def initialize_velocity_fields(links: list) -> None:
    """Add velocity fields used by the MATLAB-style momentum recursion."""
    for link in links[1:]:
        link.v = np.zeros(3)
        link.w = np.zeros(3)
        link.dq = 0.0


def forward_velocity(links: list, link_id: int = BODY) -> None:
    """Recursively update link linear and angular velocities."""
    if link_id == 0:
        return
    link = links[link_id]
    if link_id != BODY:
        parent = links[link.mother]
        link.v = parent.v + np.cross(parent.w, link.p - parent.p)
        link.w = parent.w + parent.R @ (link.a * link.dq)
    forward_velocity(links, link.sister)
    forward_velocity(links, link.child)


def set_target_link_velocity(links: list, to_link: int, target_p: np.ndarray, target_v: np.ndarray, target_w: np.ndarray) -> None:
    """Set route joint velocities so a target link has the requested velocity."""
    route = find_route(links, to_link)
    body = links[BODY]
    relative_v = target_v - body.v - np.cross(body.w, target_p - body.p)
    relative_w = target_w - body.w
    joint_velocity = np.linalg.lstsq(calc_jacobian(links, route), np.concatenate([relative_v, relative_w]), rcond=None)[0]
    for link_id, dq in zip(route, joint_velocity):
        links[link_id].dq = float(dq)


def calc_p_recursive(links: list, link_id: int = BODY) -> np.ndarray:
    """Calculate total linear momentum recursively."""
    if link_id == 0:
        return np.zeros(3)
    link = links[link_id]
    c1 = link.R @ local_com(link)
    momentum = link.mass * (link.v + np.cross(link.w, c1))
    return momentum + calc_p_recursive(links, link.sister) + calc_p_recursive(links, link.child)


def calc_l_recursive(links: list, link_id: int = BODY) -> np.ndarray:
    """Calculate total angular momentum recursively."""
    if link_id == 0:
        return np.zeros(3)
    link = links[link_id]
    c1 = link.R @ local_com(link)
    c = link.p + c1
    momentum = link.mass * (link.v + np.cross(link.w, c1))
    world_inertia = link.R @ local_inertia(link) @ link.R.T
    angular = np.cross(c, momentum) + world_inertia @ link.w
    return angular + calc_l_recursive(links, link.sister) + calc_l_recursive(links, link.child)


def solve_initial_stance(links: list) -> tuple[Pose, Pose]:
    """Place both feet with the course numerical-IK helper."""
    right_foot = Pose(np.array([0.0, -0.2, 0.1]), np.eye(3))
    left_foot = Pose(np.array([-0.04, 0.0, 0.05]), np.eye(3))

    inverse_kinematics(links, RLEG_J5, right_foot)
    inverse_kinematics(links, LLEG_J5, left_foot)
    forward_kinematics(links, BODY)
    return right_foot, left_foot


def set_matlab_initial_leg_posture(links: list) -> None:
    """Set the same bent-leg posture used before the MATLAB ZMP demo starts."""
    to_rad = np.pi / 180.0
    links[RLEG_J2].q = -10.0 * to_rad
    links[RLEG_J3].q = 20.0 * to_rad
    links[RLEG_J4].q = -10.0 * to_rad
    links[LLEG_J2].q = -10.0 * to_rad
    links[LLEG_J3].q = 20.0 * to_rad
    links[LLEG_J4].q = -10.0 * to_rad
    forward_kinematics(links, BODY)


def build_motion_states() -> list[dict]:
    """Generate the same type of ZMP-from-motion sequence as calculate_zmp.m."""
    links = setup_biped_robot()
    initialize_velocity_fields(links)
    set_matlab_initial_leg_posture(links)
    links[BODY].p = np.array([0.0, 0.0, 0.5])
    links[BODY].R = np.eye(3)
    links[BODY].v = np.zeros(3)
    links[BODY].w = np.zeros(3)
    forward_kinematics(links, BODY)

    _right_foot, left_foot = solve_initial_stance(links)
    set_target_link_velocity(links, RLEG_J5, np.array([0.0, -0.2, 0.1]), np.zeros(3), np.zeros(3))
    set_target_link_velocity(links, LLEG_J5, np.array([-0.04, 0.0, 0.05]), np.zeros(3), np.zeros(3))
    forward_velocity(links, BODY)

    com = calc_com(links)
    zc = com[2]
    tc = np.sqrt(zc / GRAVITY)
    cx0 = com[0]
    cy0 = com[1]
    previous_p = calc_p_recursive(links)
    previous_l = calc_l_recursive(links)
    mass = total_mass(links)

    states: list[dict] = []
    for time in np.arange(0.0, END_TIME + D_TIME / 2.0, D_TIME):
        px, vx = lipm(float(time), left_foot.p[0], cx0, tc)
        py, vy = lipm(float(time), left_foot.p[1], cy0, tc)

        links[BODY].p = np.array([px, py, 0.5])
        links[BODY].R = np.eye(3)
        links[BODY].v = np.array([vx, vy, 0.0])
        links[BODY].w = np.zeros(3)
        inverse_kinematics(links, LLEG_J5, left_foot)
        forward_kinematics(links, BODY)
        set_target_link_velocity(links, LLEG_J5, left_foot.p, np.zeros(3), np.zeros(3))
        forward_velocity(links, BODY)

        com = calc_com(links)
        momentum = calc_p_recursive(links)
        angular_momentum = calc_l_recursive(links)

        d_p = (momentum - previous_p) / D_TIME
        d_l = (angular_momentum - previous_l) / D_TIME
        previous_p = momentum
        previous_l = angular_momentum

        zmp = np.asarray(calc_zmp(mass, GRAVITY, com, d_p, d_l, 0.0), dtype=float).ravel()
        if zmp.size == 2:
            zmp = np.array([zmp[0], zmp[1], 0.0])

        states.append(
            {
                "time": float(time),
                "links": copy.deepcopy(links),
                "com": com,
                "zmp": zmp,
            }
        )

    return states


def draw_marker(ax, position: np.ndarray, color: str, label: str) -> None:
    """Draw a MATLAB-style cross marker for ZMP and CoM projection."""
    length = 0.2
    ax.plot([position[0] - length, position[0] + length], [position[1], position[1]], [position[2], position[2]], color=color)
    ax.plot([position[0], position[0]], [position[1] - length, position[1] + length], [position[2], position[2]], color=color)
    ax.plot([position[0], position[0]], [position[1], position[1]], [position[2], position[2] + length], color=color, label=label)


def draw_state(ax, state: dict) -> None:
    """Draw one course-style frame with biped, IZMP, and CoM projection."""
    ax.clear()
    draw_all_joints(ax, state["links"], BODY)
    draw_marker(ax, state["zmp"], "red", "IZMP")
    ax.scatter(*state["com"], color="purple", s=70, depthshade=False, label="CoM")
    draw_marker(ax, np.array([state["com"][0], state["com"][1], 0.0]), "blue", "CoM projection")
    set_biped_axes(ax)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0.0, 1.5)
    ax.set_box_aspect((1.0, 1.0, 1.5))
    ax.legend(loc="upper right")
    ax.text(0.5, -0.4, 1.4, f"time={state['time']:5.3f}")


def main() -> None:
    states = build_motion_states()

    figure = plt.figure(figsize=(7.0, 6.0))
    ax = figure.add_subplot(111, projection="3d")

    for state in states:
        draw_state(ax, state)
        plt.pause(0.01)

    plt.show()


if __name__ == "__main__":
    main()

# Verification answers:
# 1. During acceleration, momentum rates shift the ZMP away from the CoM projection.
# 2. Rates are finite differences of P and L between consecutive samples, so P and L are needed first.
# 3. The dL_y term (and the dP_z term in the denominator) moves p_x away from c_x.
