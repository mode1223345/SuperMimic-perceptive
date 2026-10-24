"""Check FK vs simulator, state round-trip, and reward on reference playback."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
add_launcher_args(parser)
args = parser.parse_args()

import torch

from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP
from supermimic.motion import MotionLib
from supermimic.robot import TRACKED_BODIES, default_joint_pos_list
from supermimic.rotations import quat_to_mat, rot_angle_sq

dev = "cuda:0"
kin = G1Kinematics(dev)
t0 = time.time()
motion = MotionLib.from_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lafan1_g1"), kin, dev)
print(f"motion: {len(motion.names)} clips, {motion.num_frames} frames ({motion.num_frames * 0.02 / 3600:.2f} h), bins {motion.num_bins}, load {time.time() - t0:.1f}s")
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, eval_mode=True)
    frames = motion.clip_start[torch.arange(args.num_envs, device=dev) % len(motion.names)] + 100
    env.reset(torch.arange(args.num_envs, device=dev), frames)
    env.scene.update(0.0)
    S = env.state()
    ref = motion.ref_state(frames)
    names = ["p", "R", "v", "w", "q", "qd"]
    for n, a, b in zip(names, S, ref):
        print(f"roundtrip {n}: max err {(a - b).abs().max().item():.2e}")
    # FK vs sim body frames
    d = env.robot.data
    ids = [env.robot.body_names.index(b) for b in TRACKED_BODIES]
    sp = d.body_link_pose_w.torch[:, ids, :3]
    sq = d.body_link_pose_w.torch[:, ids, 3:7]
    sR_xyzw = quat_to_mat(torch.cat((sq[..., 3:4], sq[..., :3]), -1))
    sR_wxyz = quat_to_mat(sq)
    bp, bR, bv, bw = kin.forward(*S)
    print("FK pos err", (bp - sp).norm(dim=-1).max().item())
    print("FK rot err xyzw", rot_angle_sq(bR, sR_xyzw).sqrt().max().item(), " wxyz", rot_angle_sq(bR, sR_wxyz).sqrt().max().item())
    sv = d.body_link_vel_w.torch[:, ids]
    print("FK linvel err", (bv - sv[..., :3]).norm(dim=-1).max().item(), " angvel err", (bw - sv[..., 3:]).norm(dim=-1).max().item())
    print("ref body vs FK(sim) pos err", (motion.body_p[frames] - bp).norm(dim=-1).max().item())
    # open-loop reference playback via joint targets
    rs = []
    for t in range(150):
        f1 = env.f + 1
        a = (motion.q[f1] - env.default_q) / env.alpha
        out = env.step(a)
        rs.append(out["r"].mean().item() / 0.1)
        if t % 25 == 0:
            print(t, "norm reward", rs[-1], "term frac", env.failed_eval.float().mean().item(), "e_body_pos", out["terms"]["e_body_pos"].mean().item())
