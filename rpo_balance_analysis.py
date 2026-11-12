"""Load RPO, apply a challenging stance, and compute gravity torques."""

from __future__ import annotations

import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def create_urdf_with_absolute_mesh_paths(urdf_path: Path) -> Path:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if not filename or filename.startswith("package://"):
            continue

        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = (urdf_path.parent / mesh_path).resolve()

        mesh.attrib["filename"] = mesh_path.as_posix()

    temp_file = tempfile.NamedTemporaryFile(mode="wb", suffix=".urdf", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    tree.write(temp_path, encoding="utf-8", xml_declaration=True)
    return temp_path


def set_joint_angle(model, q, joint_name: str, angle_rad: float) -> bool:
    joint_id = model.getJointId(joint_name)

    if joint_id >= model.njoints:
        print(f"Joint not found, skipped: {joint_name}")
        return False

    if model.nqs[joint_id] != 1:
        print(f"Joint is not 1-DoF, skipped: {joint_name}")
        return False

    q_index = model.idx_qs[joint_id]
    q[q_index] = angle_rad
    return True


def print_largest_joint_torques(model, gravity_vector, count: int = 12) -> None:
    print("\nLargest actuated gravity-compensation torques:")
    entries = []

    for joint_id, joint_name in enumerate(model.names):
        if joint_id == 0:
            continue

        if model.nvs[joint_id] != 1:
            continue

        v_index = model.idx_vs[joint_id]
        if v_index < 6:
            continue

        torque = float(gravity_vector[v_index])

        entries.append((abs(torque), torque, joint_name, v_index))

    entries.sort(reverse=True)

    for _, torque, joint_name, v_index in entries[:count]:
        print(f"  {joint_name:30s} index={v_index:2d} torque={torque: .4f} Nm")


def main() -> None:
    try:
        import pinocchio as pin
        from pinocchio.visualize import Panda3dVisualizer
    except ModuleNotFoundError as error:
        print("A required package is not installed.")
        print(f"Missing module: {error.name}")
        sys.exit(1)

    model_root = Path.home() / "dynamics" / "robot_models"
    robot_pkg = model_root / "rpo_description"
    urdf_path = robot_pkg / "urdf" / "rpo.urdf"

    if not urdf_path.exists():
        print("URDF file not found:")
        print(urdf_path)
        sys.exit(1)

    patched_urdf_path = create_urdf_with_absolute_mesh_paths(urdf_path)

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(patched_urdf_path),
        root_joint=pin.JointModelFreeFlyer(),
    )

    data = model.createData()

    q = pin.neutral(model)
    v = np.zeros(model.nv)

    q[2] = 0.75

    stance = {
        "left_thigh_yaw_joint": 0.00,
        "left_thigh_roll_joint": 0.00,
        "left_thigh_pitch_joint": 0.00,
        "left_knee_joint": 0.00,
        "left_ankle_pitch_joint": 0.00,
        "left_ankle_roll_joint": 0.00,
        "right_thigh_yaw_joint": 0.70,
        "right_thigh_roll_joint": -0.43,
        "right_thigh_pitch_joint": 0.35,
        "right_knee_joint": 0.66,
        "right_ankle_pitch_joint": 0.00,
        "right_ankle_roll_joint": 0.00,
        "torso_joint": 0.67,
        "left_arm_pitch_joint": 0.00,
        "left_arm_roll_joint": 1.05,
        "left_arm_yaw_joint": 0.04,
        "left_elbow_pitch_joint": 0.73,
        "left_elbow_yaw_joint": 0.00,
        "right_arm_pitch_joint": -0.27,
        "right_arm_roll_joint": -0.82,
        "right_arm_yaw_joint": -0.08,
        "right_elbow_pitch_joint": 0.68,
        "right_elbow_yaw_joint": -0.03,
    }

    applied = []
    skipped = []

    for joint_name, angle_rad in stance.items():
        if set_joint_angle(model, q, joint_name, angle_rad):
            applied.append(joint_name)
        else:
            skipped.append(joint_name)

    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)

    com = pin.centerOfMass(model, data, q)

    com_jacobian = pin.jacobianCenterOfMass(model, data, q)

    mass_matrix = pin.crba(model, data, q)

    mass_matrix = 0.5 * (mass_matrix + mass_matrix.T)

    gravity_vector = pin.computeGeneralizedGravity(model, data, q)

    print("RoboParty RPO challenging stance")
    print("URDF:", urdf_path)
    print(f"configuration dimension nq: {model.nq}")
    print(f"velocity dimension nv:      {model.nv}")
    print(f"number of joints:           {model.njoints}")
    print(f"number of visual objects:   {len(visual_model.geometryObjects)}")

    print(f"\nApplied joints: {len(applied)}/{len(stance)}")
    for name in applied:
        print(" ", name)

    if skipped:
        print("\nSkipped joints:")
        for name in skipped:
            print(" ", name)

    print("\nCenter of Mass [m]:")
    print(np.round(com, 4))

    print("\nCoM floor projection, used as static ZMP approximation [m]:")
    print(np.round(com[:2], 4))

    print("\nCenter of Mass Jacobian shape:")
    print(com_jacobian.shape)

    print("\nMass matrix shape:")
    print(mass_matrix.shape)

    print("\nFloating-base gravity wrench entries:")
    print(np.round(gravity_vector[:6], 4))

    print_largest_joint_torques(model, gravity_vector)

    viz = Panda3dVisualizer(model, collision_model, visual_model)
    viz.initViewer()
    viz.loadViewerModel("rpo_challenging_stance")
    viz.display(q)

    print("\nPanda3D is displaying the challenging RPO stance.")
    print("Close the Panda3D window, then press Enter here to finish.")
    input()


if __name__ == "__main__":
    main()

# Verification answers:
# 1. The right thigh roll, pitch and yaw joints (the support leg), then the arm roll joints.
# 2. Torques may exceed motor limits, the CoM/ZMP may leave the support foot, or the pose may only be reachable dynamically.
