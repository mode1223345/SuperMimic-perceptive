"""Checkpoint evaluation (paper Appendix F.1) on all motions of the library.

Each motion plays once from its first frame with friction 1.0, no domain randomization, no
observation or action noise. A motion fails if the rollout triggers the training termination.
Normalized return is the undiscounted reward sum divided by its maximum attainable value.

Run as a watcher: ``python supermimic/evaluate.py --run_dir logs/<run> --watch``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
parser.add_argument("--safe_starts", action="store_true", help="evaluate suffixes from first obstacle-free reference frame; changes metric")
add_launcher_args(parser)
args = parser.parse_args()

import numpy as np
import torch

import supermimic.networks as networks
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MAX_STEP_REWARD, OBS_DIM, MDP
from supermimic.motion import MotionLib
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list
from supermimic.Heightmap import Heightmap, TerrainSpec
from supermimic.Checkpoint import check_checkpoint, map_metadata
from supermimic.Ae import encoder_from_checkpoint

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
kin = G1Kinematics(dev)
motion = MotionLib.from_dir(args.motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
C = len(motion.names)
# Construct the physical scene from saved training geometry, never a new profile.
if args.ckpt:
    first_checkpoint = torch.load(args.ckpt, map_location=dev)
    if "heightmap" not in first_checkpoint:
        raise ValueError("Checkpoint has no heightmap metadata; use the original evaluator for upstream checkpoints")
    map_info = first_checkpoint["heightmap"]
else:
    with open(os.path.join(args.run_dir, "Heightmap.json")) as handle:
        map_info = json.load(handle)
terrain_spec = TerrainSpec.from_dict(map_info["terrain"])
expected = map_metadata(terrain_spec.to_dict(), map_info["map_dim"], map_info.get('encoder'))
if expected != map_info:
    raise ValueError("Saved heightmap configuration/fingerprint is inconsistent")
heightmap = Heightmap(terrain_spec, dev)


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
    check_checkpoint(ck, map_info)
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    shared_encoder, _ = encoder_from_checkpoint(ck, dev)
    pi = Policy(OBS_DIM, 73 * K, hist=Hh, map_dim=map_info["map_dim"], map_encoder=shared_encoder).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    ids = torch.arange(C, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    starts = motion.clip_start.clone()
    if args.safe_starts:
        for c in range(C):
            candidates = env._safe_reset_frames[motion.clip_start[c]:motion.clip_end[c]].nonzero().flatten()
            if candidates.numel() == 0:
                raise ValueError(f"No safe evaluation start in {motion.names[c]}; use compatible motions")
            starts[c] += candidates[0]
    env.reset(ids, starts)
    env.refresh()
    clip_end = motion.clip_end.clone()
    ret = torch.zeros(C, device=dev)
    steps = torch.zeros(C, device=dev)
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    while not bool(env.done_eval.all()):
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K), heightmap(S[0], S[1]))
        alive = ~env.done_eval.clone()
        if record:
            from supermimic.rotations import mat_to_quat

            traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
            traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
        out = env.step(a)
        ret += torch.where(alive, out["r"], 0.0)
        steps += alive.float()
        env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
    evaluated_length = motion.clip_end - starts
    norm = ret / (evaluated_length.float() * MAX_STEP_REWARD)
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / (evaluated_length.sum() * MAX_STEP_REWARD)).item(),
        "terrain": terrain_spec.profile,
        "map_fingerprint": map_info["fingerprint"],
        "safe_starts": args.safe_starts,
        "start_frames": starts.cpu().tolist(),
        "n_failed": int(env.failed_eval.sum().item()),
        "failed": [motion.names[i] for i in env.failed_eval.nonzero().flatten().tolist()],
        "per_clip": {motion.names[i]: round(norm[i].item(), 4) for i in range(C)},
    }
    if record:
        np.savez_compressed(record, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy(),
                            terrain_boxes=np.asarray(terrain_spec.boxes, dtype=np.float32).reshape(-1, 5))  # fmt: skip
    return res


sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, C, motion, mdp, eval_mode=True, terrain_spec=terrain_spec)
    if args.ckpt:
        res = evaluate(env, first_checkpoint, args.record)
        print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
        sys.exit(0)
    protocol = "safe" if args.safe_starts else "full"
    result_dir = os.path.join(args.run_dir, f"Eval_{protocol}")
    os.makedirs(result_dir, exist_ok=True)
    out_csv = os.path.join(result_dir, "eval.csv")
    done = set()
    if os.path.exists(out_csv):
        done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:]}
    else:
        open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
    while True:
        ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
        todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
        todo = [c for c in todo if (int(os.path.basename(c)[5:11]) // 2000) % args.every == 0]
        for c in todo:
            it = int(os.path.basename(c)[5:11])
            ck = torch.load(c, map_location=dev)
            t0 = time.time()
            res = evaluate(env, ck)
            line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n"
            open(out_csv, "a").write(line)
            result_file = os.path.join(result_dir, os.path.basename(c).replace(".pt", ".json"))
            json.dump(res, open(result_file, "w"), indent=1)
            print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C} ({time.time() - t0:.0f}s)", flush=True)
            done.add(it)
        if not args.watch:
            break
        time.sleep(60)
