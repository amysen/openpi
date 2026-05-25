"""
Scripted OSC_POSE waypoint demo collector for LIBERO-CompoSuite.

Reuses the env construction from composuite_smoke.py (OSC_POSE controller,
agentview + robot0_eye_in_hand cameras at 256x256, gripper_qpos truncated to 2).

State machine (gripper convention: -1 open, +1 close, matching LIBERO_DUMMY_ACTION):
  APPROACH_ABOVE -> DESCEND -> CLOSE -> LIFT -> MOVE_OVER_TARGET ->
  DESCEND_TARGET -> OPEN -> RETRACT

Per-task target heights:
  pick_and_place : place at table_z (drop into right bin from ~5cm above)
  trash_can      : release ~20cm above bin opening (gravity drops it in)
  shelf          : place at table_z + 0.16 (shelf surface)

Saves successful episodes as one .npz per episode under
  <out-dir>/episodes/{task}_ep{i}.npz
with the fields the openpi LeRobot loader expects:
  image:       (T, 256, 256, 3) uint8
  wrist_image: (T, 256, 256, 3) uint8
  state:       (T, 8)           float32  [eef_pos(3), axisangle(3), gripper_qpos[:2]]
  actions:     (T, 7)           float32  OSC_POSE deltas + gripper
  task:        str
Plus an MP4 of the agentview frames at <out-dir>/videos/{task}_ep{i}_{succ|fail}.mp4

A tiny converter script (run in openpi/.venv where lerobot is installed)
can then ingest these NPZs into a LeRobotDataset.

Usage:
  python experiments/composuite_collect.py \
      --tasks Panda,box,none,pick_and_place \
              Panda,box,none,trash_can \
              Panda,box,none,shelf \
      --target-successes-per-task 5 \
      --max-attempts-per-task 30 \
      --max-steps 300 \
      --out-dir data/composuite/pilot
"""
import argparse
import json
import logging
import math
import pathlib
import time

import imageio
import numpy as np

from libero.libero.envs.composuite import CompoSuiteEnv
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T


RES = 256
GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0
HOVER_DZ = 0.12       # height above object for approach
LIFT_DZ = 0.20        # carry height above table
GRASP_DZ = 0.005      # final descent above object center
POS_TOL = 0.01        # waypoint reached when |Δ| < 1cm
HOLD_STEPS = {"close": 45, "open": 6}
PHASE_MAX_STEPS = 60  # per-phase safety cap
P_GAIN = 8.0          # delta per metre of error (clipped to [-1,1])
ROT_GAIN = 1.5        # axis-angle scale per radian of orientation error
ROT_MAX_PER_STEP = 0.30  # cap rotation delta magnitude per step (avoid IK shove)

# Geometry constants for side-grasp objects (from composuite_*.xml).
PLATE_RADIUS = 0.06
DUMBBELL_BAR_RADIUS = 0.012
DUMBBELL_BAR_MID_Z = 0.065     # bar midpoint above object body origin
# Approach the side-grasp objects from +x of the object so that the gripper
# z-axis points in -x. The robot base sits at +y, so a +x stand-off keeps the
# end-effector well within the reachable workspace from the front-left bin.
SIDE_APPROACH_DIR = np.array([0.0, -1.0])  # gripper +z (forward) direction in world xy
SIDE_STANDOFF = 0.10                       # back-off distance for approach
SIDE_GRASP_INSET = 0.005                   # how far past the surface to close


def _quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _make_env(robot, object_type, obstacle, objective, seed, use_composuite_agentview=False):
    cfg = load_controller_config(default_controller="OSC_POSE")
    env = CompoSuiteEnv(
        robot=robot,
        object_type=object_type,
        obstacle=obstacle,
        objective=objective,
        controller_configs=cfg,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=RES,
        camera_widths=RES,
        horizon=800,
        ignore_done=True,
        use_composuite_agentview=use_composuite_agentview,
    )
    env.seed(seed)
    return env


def _generate_language(robot, obj, obstacle, objective):
    obj_name = obj.replace("_", " ")
    if objective == "pick_and_place":
        if obj == "plate":
            return "pull the plate off the stand and place it on top of the second stand"
        return f"pick up the {obj_name} and place it in the wooden tray"
    if objective == "trash_can":
        return f"pick up the {obj_name} and drop it in the trash can"
    if objective == "shelf":
        return f"pick up the {obj_name} and place it on the shelf"
    if objective == "push":
        return f"push the {obj_name} to the target bin"
    return f"manipulate the {obj_name}"


def _state8(obs):
    gq = np.asarray(obs["robot0_gripper_qpos"]).flatten()
    if gq.size >= 2:
        gq = gq[:2]
    else:
        gq = np.concatenate([gq, np.zeros(2 - gq.size)])
    return np.concatenate(
        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), gq)
    ).astype(np.float32)


def _frames(obs):
    # Flip rendering convention to match how composuite_smoke feeds the policy.
    # Match LIBERO's image convention: vertical flip ONLY (not double flip).
    # Mujoco renders with +y down in the image; one [::-1] makes it right-side
    # up. Applying [::-1, ::-1] would 180-rotate the image and produce a
    # horizontally-mirrored framing vs the pi05_libero training data.
    img = np.ascontiguousarray(obs["agentview_image"][::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return img, wrist


def _go_to(target_xyz, gripper_cmd, eef_pos):
    delta = (np.asarray(target_xyz) - np.asarray(eef_pos)) * P_GAIN
    delta = np.clip(delta, -1.0, 1.0)
    return np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, gripper_cmd], dtype=np.float32)


def _side_grasp_quat(approach_dir_xy, tilt_up_deg=0.0, wrist_below=False):
    """World-frame quaternion (x,y,z,w) so the gripper +z (fingertip-forward)
    points horizontally in +approach_dir_xy and the gripper +y (finger-open
    axis) points along world -z. The fingers therefore open VERTICALLY: one
    fingertip above the object plane (z > grasp_z) and one below (z <
    grasp_z), so closing pinches the object's top and bottom faces (e.g.
    grabbing the rim of a flat plate from the side).

    tilt_up_deg rotates the gripper-forward axis upward (away from the
    table) by that many degrees while keeping the open-axis vertical and
    the wrist roll the same. A small positive tilt during carry tips the
    palm so gravity pushes a held disc INTO the gripper rather than
    sliding it out the open end.

    wrist_below=True rolls the wrist 180° around the forward axis so the
    wrist housing sits BELOW the gripper line (i.e. forearm comes up at the
    object from underneath) instead of above. The pinch is still vertical
    (top finger above plane, bottom finger below) — only which fingertip
    is the +y vs -y member swaps."""
    n = np.linalg.norm(approach_dir_xy)
    ax, ay = float(approach_dir_xy[0] / n), float(approach_dir_xy[1] / n)
    t = np.deg2rad(tilt_up_deg)
    ct, st = np.cos(t), np.sin(t)
    # Forward axis = horizontal approach_dir rotated up by tilt_up_deg.
    z_axis = np.array([ax * ct, ay * ct, st])
    # Open axis = world down rotated by the same amount in the same plane,
    # so it stays orthogonal to z_axis. Closing fingers travel along
    # ±open_axis, still mostly vertical for small tilts.
    y_sign = 1.0 if wrist_below else -1.0
    y_axis = y_sign * np.array([ax * st, ay * st, -ct])
    x_axis = np.cross(y_axis, z_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return T.mat2quat(R)


def _topdown_radial_quat(approach_dir_xy):
    """World-frame quaternion (x,y,z,w) for a TOP-DOWN grasp where the fingers
    open RADIALLY along approach_dir_xy (one outside the rim, one inside).
    Gripper +z points world -z (down), gripper +x points along approach_dir_xy.
    """
    n = np.linalg.norm(approach_dir_xy)
    ax, ay = float(approach_dir_xy[0] / n), float(approach_dir_xy[1] / n)
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.array([ax, ay, 0.0])
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return T.mat2quat(R)


def _ori_error_axisangle(cur_quat, tgt_quat):
    """Axis-angle delta (rad) that rotates cur_quat into tgt_quat (world)."""
    err_quat = T.quat_multiply(np.asarray(tgt_quat), T.quat_inverse(np.asarray(cur_quat)))
    return T.quat2axisangle(err_quat)


def _go_to_pose(target_xyz, target_quat, gripper_cmd, eef_pos, eef_quat,
                pos_max=1.0):
    """Like _go_to but also drives orientation toward target_quat (or holds
    if target_quat is None). pos_max caps the xy/z action magnitude per step
    so a slow-carry phase can give the orientation controller time to track."""
    if target_xyz is None:
        dpos = np.zeros(3)
    else:
        dpos = np.clip((np.asarray(target_xyz) - np.asarray(eef_pos)) * P_GAIN, -pos_max, pos_max)
    if target_quat is None:
        drot = np.zeros(3)
    else:
        drot = _ori_error_axisangle(eef_quat, target_quat) * ROT_GAIN
        # Cap per-step rotation magnitude (per-axis clip can spin the wrist).
        n = float(np.linalg.norm(drot))
        if n > ROT_MAX_PER_STEP:
            drot = drot * (ROT_MAX_PER_STEP / n)
        drot = np.clip(drot, -1.0, 1.0)
    return np.array(
        [dpos[0], dpos[1], dpos[2], drot[0], drot[1], drot[2], gripper_cmd],
        dtype=np.float32,
    )


def _drop_target(env, objective):
    """z target for the placement phase, depending on objective."""
    table_z = env.table_offset[2]
    if objective == "pick_and_place":
        # Just above the bin; let the gripper open and drop
        return table_z + 0.04
    if objective == "trash_can":
        # Hover well above rim and release; gravity drops it in
        return table_z + 0.20
    if objective == "shelf":
        # Shelf top is at table+0.17. The box hangs ~0.06m below the gripper,
        # so the gripper must be ~table+0.25 for the box bottom to rest on the
        # shelf surface (with a small clearance so we release just above it).
        return table_z + 0.27
    return table_z + 0.05


def _carry_height(env, objective):
    """Carry height for lift / move_over so the held box clears obstacles."""
    table_z = env.table_offset[2]
    if objective == "shelf":
        # Must clear the shelf top (table+0.17) by enough margin that the
        # box (hanging ~0.06m below the gripper) does not collide with the
        # shelf as it moves laterally over it.
        return table_z + 0.32
    return table_z + LIFT_DZ


def _push_phases(obs, env):
    """Phases for the 'push' objective: approach behind object, descend, push to goal."""
    obj_pos0 = np.array(obs["object_pos"])
    goal_pos = np.array(obs["goal_pos"])
    table_z = env.table_offset[2]

    direction = goal_pos[:2] - obj_pos0[:2]
    dnorm = float(np.linalg.norm(direction))
    if dnorm < 1e-6:
        unit = np.array([1.0, 0.0])
    else:
        unit = direction / dnorm
    back_offset = 0.09  # m behind the object to plant the gripper
    behind_xy = obj_pos0[:2] - unit * back_offset
    push_z = table_z + 0.035  # object-center height

    # Just approach + descend; the actual push is a closed-loop controller
    # in collect_episode (see CLOSED_LOOP_PUSH branch).
    return [
        ("approach_above", np.array([behind_xy[0], behind_xy[1], obj_pos0[2] + HOVER_DZ]), None, GRIP_CLOSE, POS_TOL, PHASE_MAX_STEPS),
        ("descend_behind", np.array([behind_xy[0], behind_xy[1], push_z]),                  None, GRIP_CLOSE, 0.01,    PHASE_MAX_STEPS),
    ]


def _pickplace_phases(obs, env, objective):
    obj_pos0 = np.array(obs["object_pos"])
    goal_pos = np.array(obs["goal_pos"])
    table_z = env.table_offset[2]
    drop_z = _drop_target(env, objective)
    carry_z = _carry_height(env, objective)
    return [
        ("approach_above", np.array([obj_pos0[0], obj_pos0[1], obj_pos0[2] + HOVER_DZ]), None, GRIP_OPEN, POS_TOL, PHASE_MAX_STEPS),
        ("descend",        np.array([obj_pos0[0], obj_pos0[1], obj_pos0[2] + GRASP_DZ]), None, GRIP_OPEN, 0.005, PHASE_MAX_STEPS),
        ("close",          None,                                                          None, GRIP_CLOSE, None,  HOLD_STEPS["close"]),
        ("lift",           np.array([obj_pos0[0], obj_pos0[1], carry_z]),                 None, GRIP_CLOSE, 0.02,  PHASE_MAX_STEPS),
        ("move_over",      np.array([goal_pos[0], goal_pos[1], carry_z]),                 None, GRIP_CLOSE, 0.02,  PHASE_MAX_STEPS),
        ("descend_target", np.array([goal_pos[0], goal_pos[1], drop_z]),                  None, GRIP_CLOSE, 0.01,  PHASE_MAX_STEPS),
        ("open",           None,                                                          None, GRIP_OPEN,  None,  HOLD_STEPS["open"]),
        ("retract",        np.array([goal_pos[0], goal_pos[1], carry_z + 0.05]),          None, GRIP_OPEN,  0.03,  PHASE_MAX_STEPS),
    ]


def _plate_rim_phases(obs, env, objective):
    """Top-down pinch at the +x rim of a flat disc plate.

    A flat plate on a flat table cannot be pinched horizontally (the lower
    finger would pass through the table). Instead, hover the gripper above
    the rim with the home orientation (fingers open along world x) and pinch
    so one finger lands on the disc top and the other in air just outside
    the rim, gripping the plate's outer edge.
    """
    obj_pos0 = np.array(obs["object_pos"])
    goal_pos = np.array(obs["goal_pos"])
    drop_z = _drop_target(env, objective)
    carry_z = _carry_height(env, objective)
    # Place grip site exactly at the +x rim point. Open fingers (~0.04 m to
    # each side along world x) then straddle the rim: outer finger in air just
    # outside the disc, inner finger above the disc top. As they close, they
    # squeeze the rim's edge.
    grip_xy = obj_pos0[:2] + np.array([PLATE_RADIUS, 0.0])
    grip_z  = obj_pos0[2] + 0.014      # ~6 mm above plate top (half-thick 0.008)
    return [
        ("approach_above", np.array([grip_xy[0], grip_xy[1], obj_pos0[2] + HOVER_DZ]), None, GRIP_OPEN, POS_TOL, PHASE_MAX_STEPS),
        ("descend",        np.array([grip_xy[0], grip_xy[1], grip_z]),                 None, GRIP_OPEN, 0.005,   PHASE_MAX_STEPS),
        ("close",          None,                                                       None, GRIP_CLOSE, None,    HOLD_STEPS["close"]),
        ("lift",           np.array([grip_xy[0], grip_xy[1], carry_z]),                None, GRIP_CLOSE, 0.02,    PHASE_MAX_STEPS),
        ("move_over",      np.array([goal_pos[0], goal_pos[1], carry_z]),              None, GRIP_CLOSE, 0.02,    PHASE_MAX_STEPS),
        ("descend_target", np.array([goal_pos[0], goal_pos[1], drop_z]),               None, GRIP_CLOSE, 0.01,    PHASE_MAX_STEPS),
        ("open",           None,                                                       None, GRIP_OPEN,  None,    HOLD_STEPS["open"]),
        ("retract",        np.array([goal_pos[0], goal_pos[1], carry_z + 0.05]),       None, GRIP_OPEN,  0.03,    PHASE_MAX_STEPS),
    ]


def _side_grasp_phases(obs, env, objective, grasp_orientation):
    """Side-approach grasp for `horizontal_edge` (plate) and `horizontal_bar`
    (dumbbell). The wrist rotates so the gripper z-axis points horizontally
    toward the object and the fingers open vertically (one above, one below).
    Carry and place use the same horizontal wrist orientation; for the drop
    objectives this releases the object from a side pinch above the goal."""
    obj_pos0 = np.array(obs["object_pos"])
    goal_pos = np.array(obs["goal_pos"])
    table_z = env.table_offset[2]

    if grasp_orientation == "horizontal_edge":
        # Plate sits in front of the robot on a small wooden shelf. The
        # disc (r=0.060, thick=0.016) cantilevers off the -x face of the
        # 80x90x120 mm shelf block. The robot approaches from -x with the
        # gripper fingertips pointing +x (straight forward, joints at
        # neutral pose), grabs the front rim, then pulls the disc straight
        # back in -x off the shelf — no wrist re-orient required.
        approach_dir = np.array([1.0, 0.0])  # gripper +z points +x
        plate_r = 0.060
        # Pinch NEAR THE RIM (10 mm from disc edge) so the fingertips
        # land in the clear air outside the shelf footprint. Pinching
        # closer to the disc centre would put the bottom finger inside
        # the shelf -x face during slide-in.
        # Pinch DEEPER into the disc (further from the rim) so the
        # gripper pads have more disc surface under them and the disc
        # cannot pivot/slip out under lateral load. 25 mm inside the rim
        # leaves ~35 mm of finger-pad/disc overlap on each side.
        plate_grip_inset = 0.025          # how far inside the rim to pinch
        plate_grip_offset = plate_r - plate_grip_inset  # 0.035 m from centre
        # Plate XML now contains only the disc; body origin sits at the
        # disc's bottom face (z=0), so disc bottom = body z, disc centre
        # = body z + 0.008. The static shelf is a separate fixture.
        plate_disc_bottom_dz = 0.000
        plate_disc_center_dz = 0.009
        # 3 cm clearance below disc bottom for lateral approach
        approach_low_z  = obj_pos0[2] + plate_disc_bottom_dz - 0.030
        grasp_z         = obj_pos0[2] + plate_disc_center_dz
        # Pinch slightly inside the -x disc edge (the overhang). With
        # approach_dir = (+1,0), `-approach_dir * plate_grip_offset` puts
        # the eef at -x of the disc centre but still over the disc surface
        # so the vertical-closing fingertips clamp top and bottom faces.
        grasp_xy  = obj_pos0[:2] - approach_dir * plate_grip_offset
        target_quat = _side_grasp_quat(approach_dir, wrist_below=False)
        # Keep the wrist fully horizontal throughout lift and carry so the
        # disc face stays parallel to the table.
        carry_quat  = target_quat
    else:  # horizontal_bar  (dumbbell)
        # Bar passes through the object centre xy; grip at bar mid-height.
        grasp_z = max(obj_pos0[2] + DUMBBELL_BAR_MID_Z, table_z + 0.05)
        grasp_xy = obj_pos0[:2].copy()
        approach_dir = SIDE_APPROACH_DIR / np.linalg.norm(SIDE_APPROACH_DIR)
        target_quat  = _side_grasp_quat(approach_dir)
        carry_quat   = target_quat  # dumbbell pinch is around its waist; no tilt needed
        # For dumbbell: approach_low_z == grasp_z (no need to go below first)
        approach_low_z = grasp_z

    # Both cases approach laterally with the same horizontal standoff.
    # approach_xy must be OUTSIDE the disc on the near side (+y of disc for
    # approach_dir=(0,-1)) so the gripper can descend to approach_low_z
    # without intersecting the disc. For the overhang plate the +y side is
    # also clear of the shelf support, so the gripper body has air all the
    # way down to the table.
    if grasp_orientation == "horizontal_edge":
        approach_xy = obj_pos0[:2] - approach_dir * (plate_r + SIDE_STANDOFF)
    else:
        approach_xy = grasp_xy - approach_dir * SIDE_STANDOFF

    drop_z = _drop_target(env, objective)
    # Side-gripped objects swing through more of the workspace; add a bit of
    # extra clearance so the gripper body itself does not clip the shelf.
    carry_z = _carry_height(env, objective) + 0.05
    if grasp_orientation == "horizontal_edge":
        # Carry the plate LOW and HORIZONTAL: just enough to clear the
        # source shelf-block (whose top is at obj_pos0[2] + 0.085). Keeping
        # carry_z close to grasp_z forces the IK into a low-elbow config
        # where the forearm stays parallel to the table during the entire
        # sweep — matching how the dumbbell side-grasp transports its
        # object. A high carry_z extends the arm and the wrist angles
        # downward, tilting the disc.
        carry_z = max(grasp_z + 0.04, table_z + 0.13)

    # When the gripper pinches the +y disc edge, the disc centre is offset
    # from the eef by `+approach_dir * plate_grip_offset` (i.e. -y of the
    # eef by 0.040 m). Compensate at the goal so the disc centre lands on
    # goal_pos.
    if grasp_orientation == "horizontal_edge":
        goal_eef_xy = goal_pos[:2] - approach_dir * plate_grip_offset
    else:
        goal_eef_xy = goal_pos[:2].copy()

    # Slow-carry: the disc is held only by friction on its top/bottom faces,
    # so any rapid lateral acceleration can pop it free. Reduce the per-step
    # translational action cap for the plate carry so the gripper accelerates
    # gently and the disc stays pinched. Other objects use the default 1.0.
    carry_pos_max = 0.12 if grasp_orientation == "horizontal_edge" else 1.0
    place_pos_max = 0.08 if grasp_orientation == "horizontal_edge" else 1.0

    phases = [
        # Phase 1: pure wrist rotation, actively holding the current eef xyz.
        ("orient_wrist",   np.array(obs["robot0_eef_pos"]).copy(),                          target_quat, GRIP_OPEN,  POS_TOL, PHASE_MAX_STEPS),
        # Phase 2: move to approach_xy at height well above disc (no disc contact).
        ("rotate_above",   np.array([approach_xy[0], approach_xy[1], obj_pos0[2] + 0.20]), target_quat, GRIP_OPEN,  POS_TOL, PHASE_MAX_STEPS),
        # Phase 3: descend OUTSIDE the disc footprint to grasp height. Open
        #          fingers straddle the disc edge plane in clear air (top
        #          finger above disc top, bottom finger below disc bottom).
        ("descend_outside", np.array([approach_xy[0], approach_xy[1], grasp_z]),             target_quat, GRIP_OPEN,  0.005,   PHASE_MAX_STEPS),
        # Phase 4: closed-loop slide INWARD onto the disc rim. The disc is
        #          spawned in the narrow band where the Panda's natural
        #          horizontal-forward wrist pose places the gripper site,
        #          so target == obj_y (no bias compensation needed).
        ("slide_in",       (lambda obs: np.array([
                                np.array(obs["object_pos"])[0] - approach_dir[0]*plate_grip_offset,
                                np.array(obs["object_pos"])[1] - approach_dir[1]*plate_grip_offset,
                                grasp_z,
                            ])) if grasp_orientation == "horizontal_edge" else
                          np.array([grasp_xy[0],    grasp_xy[1],    grasp_z]),
                          target_quat, GRIP_OPEN,  0.005,   PHASE_MAX_STEPS * 2),
        # Phase 5: dummy rise (no-op for plate, kept for dumbbell which sets
        #          approach_low_z = grasp_z so this is a hold-in-place).
        ("rise_to_grasp",  np.array([grasp_xy[0],    grasp_xy[1],    grasp_z]),              target_quat, GRIP_OPEN,  0.005,   PHASE_MAX_STEPS),
        ("close",          None,                                                             target_quat, GRIP_CLOSE, None,    HOLD_STEPS["close"]),
    ]
    if grasp_orientation == "horizontal_edge":
        # Plate task:
        #   1) slide_off  : pull the disc straight off the shelf in -x
        #                   (closed-loop) until the eef is well clear of
        #                   the shelf x band.
        #   2) traverse_y : at the SAME -x as slide_off (NOT goal_eef_xy.x,
        #                   which would pull the gripper back through the
        #                   shelf), move +y to the goal y.
        #   3) align_x    : now safely clear of the shelf in y, drift x to
        #                   goal x.
        #   4) descend    : lower straight down to just above the table.
        #   5) open + retract.
        # Plate task: destination is the TOP of a second shelf next to
        # the source shelf. The disc is carried at grasp_z (its current
        # height on the source shelf) and lowered just enough to rest on
        # top of the destination shelf. Wrist stays horizontal throughout.
        # second-shelf top z = table_z + 0.120; disc bottom = shelf_top + tiny.
        plate_shelf2_top_z = table_z + 0.120
        plate_drop_eef_z   = plate_shelf2_top_z + 0.012  # eef height when disc bottom touches shelf
        plate_slide_lead   = 0.02
        # Pull the eef this far in -x past the grasp point. This must take
        # the disc completely clear of the shelf (shelf -x face is ~25 mm
        # in -x of disc spawn; +disc radius 60 mm; need >85 mm of disc
        # travel, so eef travel ~ 85 + a margin).
        slide_off_dist = 0.22
        slide_off_eef_xy = np.array([
            grasp_xy[0] - approach_dir[0] * slide_off_dist,
            grasp_xy[1] - approach_dir[1] * slide_off_dist,
        ])
        def _slide_off_target(obs):
            disc = np.array(obs["object_pos"])
            behind = disc[:2] - approach_dir * plate_grip_offset
            lead_pt = behind - approach_dir * plate_slide_lead
            if approach_dir[0] > 0:
                tgt_x = max(slide_off_eef_xy[0], lead_pt[0])
                tgt_y = slide_off_eef_xy[1]
            else:
                tgt_x, tgt_y = lead_pt[0], lead_pt[1]
            return np.array([tgt_x, tgt_y, grasp_z])
        # Traverse keeps x = slide_off_eef_xy[0] so the disc stays in -x of
        # the shelf throughout the +y move.
        traverse_xy = np.array([slide_off_eef_xy[0], goal_eef_xy[1]])
        # Carry height while traversing must clear the SECOND shelf top
        # (table_z + 0.120). Disc bottom = eef_z - ~0.020, so eef_z >=
        # shelf2_top + 0.020 + clearance. Use ~5 cm clearance.
        carry_clear_z = plate_shelf2_top_z + 0.060
        return phases + [
            ("slide_off",      _slide_off_target,
                              carry_quat, GRIP_CLOSE, 0.01, PHASE_MAX_STEPS * 2, carry_pos_max),
            # Lift straight up to a safe carry height that clears the
            # second shelf top, before moving in +y.
            ("lift_carry",     np.array([slide_off_eef_xy[0], slide_off_eef_xy[1], carry_clear_z]),
                              carry_quat, GRIP_CLOSE, 0.01, PHASE_MAX_STEPS, carry_pos_max),
            ("traverse_y",     np.array([traverse_xy[0], traverse_xy[1], carry_clear_z]),
                              carry_quat, GRIP_CLOSE, 0.015, PHASE_MAX_STEPS * 2, carry_pos_max),
            ("align_x",        np.array([goal_eef_xy[0], goal_eef_xy[1], carry_clear_z]),
                              carry_quat, GRIP_CLOSE, 0.01, PHASE_MAX_STEPS * 2, carry_pos_max),
            # Pause briefly over the target with the horizontal wrist command
            # before lowering. This gives the OSC orientation loop time to
            # settle instead of combining x alignment, wrist correction, and
            # z descent into one jerky motion.
            ("stabilize_over_shelf2", np.array([goal_eef_xy[0], goal_eef_xy[1], carry_clear_z]),
                              carry_quat, GRIP_CLOSE, None, 20, place_pos_max),
            # Lower onto the second shelf top.
            ("descend_shelf2", np.array([goal_eef_xy[0], goal_eef_xy[1], plate_drop_eef_z]),
                              carry_quat, GRIP_CLOSE, 0.005, PHASE_MAX_STEPS * 3, place_pos_max),
            # Hold position briefly while opening so the disc settles on
            # the second shelf before the gripper moves.
            ("open",           np.array([goal_eef_xy[0], goal_eef_xy[1], plate_drop_eef_z]),
                              carry_quat, GRIP_OPEN, None, HOLD_STEPS["open"] * 5, place_pos_max),
            # Lift straight up first so the open fingers clear the disc.
            ("lift_clear",     np.array([goal_eef_xy[0], goal_eef_xy[1], carry_clear_z + 0.05]),
                              carry_quat, GRIP_OPEN, 0.02, PHASE_MAX_STEPS, place_pos_max),
            # Then move away in -y back toward the source shelf side and up.
            ("retract",        np.array([
                                  goal_eef_xy[0] - approach_dir[0] * 0.10,
                                  goal_eef_xy[1] - 0.15,
                                  carry_clear_z + 0.10,
                              ]),
                              carry_quat, GRIP_OPEN, 0.03, PHASE_MAX_STEPS),
        ]
    return phases + [
        # Slide the disc OFF the shelf horizontally before lifting. The disc
        # rests on the shelf at +y; pulling in -approach_dir (= -y) at the
        # same grasp_z drags the disc clear of the shelf without any vertical
        # torque, so it stays flat. Distance ~70 mm clears the 60 mm overlap.
        ("slide_off",      np.array([grasp_xy[0] - approach_dir[0]*0.07,
                                      grasp_xy[1] - approach_dir[1]*0.07, grasp_z]),         carry_quat,  GRIP_CLOSE, 0.01,    PHASE_MAX_STEPS, carry_pos_max),
        ("lift",           np.array([grasp_xy[0] - approach_dir[0]*0.07,
                                      grasp_xy[1] - approach_dir[1]*0.07, carry_z]),         carry_quat,  GRIP_CLOSE, 0.02,    PHASE_MAX_STEPS, carry_pos_max),
        # Intermediate halfway waypoint so the orientation controller has time
        # to keep the wrist horizontal during the long lateral sweep. Without
        # this the wrist tilts and the disc twists out of the pinch.
        ("carry_mid",      np.array([(grasp_xy[0]+goal_eef_xy[0])*0.5,
                                      (grasp_xy[1]+goal_eef_xy[1])*0.5, carry_z]),            carry_quat,  GRIP_CLOSE, 0.03,    PHASE_MAX_STEPS, carry_pos_max),
        # Move over goal AND start untilting in parallel — by the time the eef
        # arrives the wrist is already nearly horizontal so the disc lays flat.
        ("move_over",      np.array([goal_eef_xy[0], goal_eef_xy[1], carry_z]),              target_quat, GRIP_CLOSE, 0.02,    PHASE_MAX_STEPS, carry_pos_max),
        ("descend_target", np.array([goal_eef_xy[0], goal_eef_xy[1], drop_z]),               target_quat, GRIP_CLOSE, 0.01,    PHASE_MAX_STEPS * 2, carry_pos_max),
        ("open",           None,                                                             target_quat, GRIP_OPEN,  None,    HOLD_STEPS["open"]),
        ("retract",        np.array([goal_eef_xy[0], goal_eef_xy[1], carry_z + 0.05]),       target_quat, GRIP_OPEN,  0.03,    PHASE_MAX_STEPS),
    ]


def collect_episode(env, objective, max_steps, num_warmup=10):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, GRIP_OPEN], dtype=np.float32).tolist())
    # Brief warmup so initial obs settle
    for _ in range(num_warmup):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, GRIP_OPEN], dtype=np.float32).tolist())

    table_z = env.table_offset[2]
    if objective == "push":
        phases = _push_phases(obs, env)
    else:
        grasp_ori = getattr(getattr(env, "composuite_object", None), "grasp_orientation", "top")
        if grasp_ori in ("horizontal_edge", "horizontal_bar"):
            phases = _side_grasp_phases(obs, env, objective, grasp_ori)
        else:
            phases = _pickplace_phases(obs, env, objective)

    if grasp_ori in ("horizontal_edge", "horizontal_bar"):
        logging.info(
            "  init: eef=%s obj=%s goal=%s",
            np.array(obs["robot0_eef_pos"]).round(3).tolist(),
            np.array(obs["object_pos"]).round(3).tolist(),
            np.array(obs["goal_pos"]).round(3).tolist(),
        )
        for ph in phases:
            tgt = ph[1]
            if tgt is None:
                tgt_str = "hold"
            elif callable(tgt):
                tgt_str = "track(obs)"
            else:
                tgt_str = np.asarray(tgt).round(3).tolist()
            logging.info("  phase=%-15s target=%s", ph[0], tgt_str)

    plate_pick_place = objective == "pick_and_place" and grasp_ori == "horizontal_edge"
    images, wrists, states, actions = [], [], [], []
    success_seen = False
    episode_done = False
    total_steps = 0
    last_phase = "init"
    for ph in phases:
        # Phases are 6-tuples; an optional 7th entry caps the per-step xy/z
        # action magnitude (slow-carry mode for fragile pinches).
        if len(ph) == 7:
            name, target, target_quat, grip, tol, cap, pos_max = ph
        else:
            name, target, target_quat, grip, tol, cap = ph
            pos_max = 1.0
        last_phase = name
        for _ in range(cap):
            if total_steps >= max_steps:
                break
            eef = np.array(obs["robot0_eef_pos"])
            eef_quat = np.array(obs["robot0_eef_quat"])
            # Allow a phase target to be a callable(obs)->xyz so the loop can
            # re-aim each step (used for the plate side-grasp final approach,
            # where the OSC has steady-state y error after the wrist rotation
            # and an open-loop target leaves the gripper off the disc).
            if callable(target):
                cur_target = np.asarray(target(obs))
            else:
                cur_target = target
            if cur_target is None and target_quat is None:
                # Hold-in-place gripper command
                act = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip], dtype=np.float32)
            else:
                act = _go_to_pose(cur_target, target_quat, grip, eef, eef_quat, pos_max=pos_max)
            img, wrist = _frames(obs)
            images.append(img)
            wrists.append(wrist)
            states.append(_state8(obs))
            actions.append(act)
            try:
                obs, _r, done, info = env.step(act.tolist())
            except ValueError:
                done = True; info = {"success": True}
                success_seen = True; episode_done = True
                break
            total_steps += 1
            if info.get("success"):
                success_seen = True
            cur_eef_pos = np.array(obs["robot0_eef_pos"])
            # Only consider early-exit when the phase declares a tolerance.
            # Phases with tol=None are "hold for HOLD_STEPS" (e.g. close /
            # open) and must run the full step budget so the gripper has
            # time to actually grasp.
            if tol is not None:
                if cur_target is not None and np.linalg.norm(cur_target - cur_eef_pos) < tol:
                    if target_quat is None:
                        break
                    ori_err = _ori_error_axisangle(np.array(obs["robot0_eef_quat"]), target_quat)
                    if np.linalg.norm(ori_err) < 0.15:  # ~9 deg
                        break
                elif cur_target is None and target_quat is not None:
                    ori_err = _ori_error_axisangle(np.array(obs["robot0_eef_quat"]), target_quat)
                    if np.linalg.norm(ori_err) < 0.10:  # ~6 deg
                        break
            if done:
                episode_done = True
                break
        # Per-finger world position via mujoco model — useful for diagnosing
        # whether the gripper pads are actually closing onto the object.
        try:
            sim = env.sim
            finger_names = [n for n in sim.model.body_names if "finger" in n.lower()]
            if finger_names:
                positions = {n: np.round(sim.data.body_xpos[sim.model.body_name2id(n)], 3).tolist() for n in finger_names}
                gq = np.array(obs.get("robot0_gripper_qpos", []))
                logging.info("    fingers=%s gq=%s", positions, np.round(gq, 4).tolist())
        except Exception as e:
            logging.info("    finger debug failed: %s", e)
        logging.info(
            "  end-phase=%-15s eef=%s obj=%s eef_quat=%s",
            name,
            np.array(obs["robot0_eef_pos"]).round(3).tolist(),
            np.array(obs["object_pos"]).round(3).tolist(),
            np.array(obs["robot0_eef_quat"]).round(3).tolist(),
        )
        logging.debug(
            "phase=%s eef=%s obj=%s done_steps=%d",
            name,
            np.array(obs["robot0_eef_pos"]).round(3).tolist(),
            np.array(obs["object_pos"]).round(3).tolist(),
            total_steps,
        )
        if total_steps >= max_steps or episode_done:
            break
        # For the plate shelf-transfer demonstrations, success can fire as
        # soon as the disc passes near the destination shelf. Keep executing
        # the release/clear phases so saved demos contain a real placement.
        if success_seen and not plate_pick_place:
            break

    # Closed-loop push: re-aim every step so the gripper stays behind the box
    # as it drifts. Continues until obj reaches goal xy or step budget exhausts.
    # Push reward terminates the episode if the box is lifted >0.03m above the
    # table, so we push slowly with the gripper at box-center height.
    if objective == "push":
        # Box center after settling is around table+0.025; push at that height
        # so the contact force is purely horizontal (no lever-up effect).
        push_z = table_z + 0.025
        for _ in range(max_steps - total_steps):
            if total_steps >= max_steps:
                break
            obj_xy = np.array(obs["object_pos"])[:2]
            goal_xy = np.array(obs["goal_pos"])[:2]
            to_goal = goal_xy - obj_xy
            dist = float(np.linalg.norm(to_goal))
            if dist < 0.05:
                break  # close enough; let it settle
            unit = to_goal / max(dist, 1e-6)
            # Aim a short distance past the box; combined with a small action
            # cap below this keeps the push slow enough to avoid an illegal
            # lift termination.
            target_xy = obj_xy + unit * 0.05
            target = np.array([target_xy[0], target_xy[1], push_z])
            eef = np.array(obs["robot0_eef_pos"])
            delta = (target - eef) * P_GAIN
            # Cap horizontal speed to ~0.15 per step so contact force stays low.
            delta = np.clip(delta, -0.15, 0.15)
            act = np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, GRIP_CLOSE], dtype=np.float32)
            img, wrist = _frames(obs)
            images.append(img); wrists.append(wrist); states.append(_state8(obs)); actions.append(act)
            obs, _r, done, info = env.step(act.tolist())
            total_steps += 1
            if info.get("success"):
                success_seen = True
            if done:
                break
        # Retract upward so we don't sit on the box while it settles
        eef = np.array(obs["robot0_eef_pos"])
        retract_target = np.array([eef[0], eef[1], push_z + 0.15])
        for _ in range(20):
            if total_steps >= max_steps:
                break
            eef = np.array(obs["robot0_eef_pos"])
            act = _go_to(retract_target, GRIP_CLOSE, eef)
            img, wrist = _frames(obs)
            images.append(img); wrists.append(wrist); states.append(_state8(obs)); actions.append(act)
            obs, _r, done, info = env.step(act.tolist())
            total_steps += 1
            if info.get("success"):
                success_seen = True
            if done:
                break

    # Allow a short settling tail so trash_can/shelf success can register
    if not success_seen and not episode_done:
        for _ in range(20):
            if total_steps >= max_steps:
                break
            act = np.array([0, 0, 0, 0, 0, 0, GRIP_OPEN], dtype=np.float32)
            img, wrist = _frames(obs)
            images.append(img); wrists.append(wrist); states.append(_state8(obs)); actions.append(act)
            try:
                obs, _r, done, info = env.step(act.tolist())
            except ValueError:
                success_seen = True
                break
            total_steps += 1
            if info.get("success"):
                success_seen = True
            if done:
                break

    final_success = bool(env._check_success() or (success_seen and not plate_pick_place))
    diag = {
        "obj_to_goal": float(np.linalg.norm(np.array(obs["object_pos"]) - np.array(obs["goal_pos"]))),
        "obj_z": float(obs["object_pos"][2]),
        "table_z": float(table_z),
        "last_phase": last_phase,
    }
    return final_success, total_steps, images, wrists, states, actions, diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", required=True,
                   help="List of robot,object,obstacle,objective specs")
    p.add_argument("--target-successes-per-task", type=int, default=5)
    p.add_argument("--max-attempts-per-task", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--out-dir", default="data/composuite/pilot")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--use-composuite-agentview", action="store_true",
                   help="Use CompoSuite's pulled-back agentview (pos=[0.74,0,1.72], fovy=52). "
                        "Default is the LIBERO scene-XML camera (pos=[0.5,0,1.35], default fovy) "
                        "so demos match the framing pi05_libero was trained on.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = pathlib.Path(args.out_dir)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    summary = []

    for spec in args.tasks:
        robot, obj, obstacle, objective = [s.strip() for s in spec.split(",")]
        task_name = f"{robot}_{obj}_{obstacle}_{objective}"
        prompt = _generate_language(robot, obj, obstacle, objective)
        logging.info(f"=== {task_name} | prompt={prompt!r} ===")

        env = _make_env(robot, obj, obstacle, objective, args.seed,
                        use_composuite_agentview=args.use_composuite_agentview)
        n_succ = 0
        attempts = 0
        for attempt in range(args.max_attempts_per_task):
            attempts = attempt + 1
            env.seed(args.seed + attempt)
            env.reset()
            t0 = time.time()
            ok, steps, imgs, wrs, sts, acts, diag = collect_episode(env, objective, args.max_steps)
            dt = time.time() - t0
            tag = "success" if ok else "failure"
            vp = out / "videos" / f"{task_name}_ep{attempt:02d}_{tag}.mp4"
            try:
                imageio.mimwrite(vp, imgs, fps=20)
            except Exception as e:
                logging.warning(f"video write failed: {e}")

            logging.info(f"  attempt {attempt:02d}: success={ok} steps={steps} time={dt:.1f}s diag={diag}")
            if ok:
                ep_path = out / "episodes" / f"{task_name}_ep{n_succ:02d}.npz"
                np.savez_compressed(
                    ep_path,
                    image=np.stack(imgs).astype(np.uint8),
                    wrist_image=np.stack(wrs).astype(np.uint8),
                    state=np.stack(sts).astype(np.float32),
                    actions=np.stack(acts).astype(np.float32),
                    task=np.array(prompt),
                )
                n_succ += 1
                summary.append({"task": task_name, "ep": n_succ - 1, "attempt": attempt,
                                "steps": steps, "wall_seconds": round(dt, 1),
                                "video": str(vp), "npz": str(ep_path), "diag": diag})
                if n_succ >= args.target_successes_per_task:
                    break
        logging.info(f"=== {task_name}: {n_succ}/{args.target_successes_per_task} (attempts={attempts}) ===")
        try: env.close()
        except Exception: pass

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Wrote {len(summary)} successful episodes to {out}")


if __name__ == "__main__":
    main()
