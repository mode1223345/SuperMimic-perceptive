"""Spawn G1s on Newton/MuJoCo-Warp, report ordering and stepping throughput."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--mj_contacts", type=int, default=1)
add_launcher_args(parser)
args = parser.parse_args()

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace

from supermimic.robot import G1_CFG


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/Robot")


physics = NewtonCfg(
    solver_cfg=MJWarpSolverCfg(
        njmax=200, nconmax=64, cone="pyramidal", integrator="implicitfast", use_mujoco_contacts=bool(args.mj_contacts)
    ),
    num_substeps=4,
)
sim_cfg = sim_utils.SimulationCfg(dt=0.02, device=args.device or "cuda:0", physics=physics)
with launch_simulation(sim_cfg, args):
    sim = SimulationContext(sim_cfg)
    scene = instantiate(SceneCfg(num_envs=args.num_envs, env_spacing=0.0, replicate_physics=True))
    sim.reset()
    robot = scene["robot"]
    print("JOINTS", robot.joint_names)
    print("BODIES", robot.body_names)
    print("default q", robot.data.default_joint_pos.torch[0])
    print("stiffness", robot.data.joint_stiffness.torch[0])
    print("armature", robot.data.joint_armature.torch[0])
    print("mass", robot.data.body_mass.torch[0].sum())
    target = robot.data.default_joint_pos.torch.clone()
    for it in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        n = 100
        for _ in range(n):
            robot.set_joint_position_target_index(target=target + 0.1 * torch.randn_like(target))
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(0.02)
            _ = robot.data.body_link_pose_w.torch
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"steps/s (control 50Hz) = {n * args.num_envs / dt:.0f}; root z mean {robot.data.root_link_pos_w.torch[:, 2].mean():.3f}")
