
import wandb
import os
import pickle
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from stable_baselines3 import PPO
from tools import initialize_env, sample_trajectory, sample_random_trajectory

""" 
This file is a helper for evaluation. 
It samples random lines and random trajectories to later use for computing # transitions and # regions over these lines and trajectories.
"""

def sample_random_lines(runs, num_samples):
    sampled_points = []

    for i in range(num_samples):
        idx = np.random.randint(low=0, high=len(runs)-1)
        run = runs[idx]

        env_name = run.config["env"]
        checkpoints_dir = run.config["save_dir"]
        run_folder_path = Path(checkpoints_dir) / f"car1d_{run.id}"

        # sample a trajectory from the final policy 
        best_policy_path = run_folder_path / 'best.zip'
        best_stats_path = run_folder_path / 'best_stats.pth'
        # load policy
        best_expert = PPO.load(best_policy_path)
        # initialize environment for input standardization
        best_env = initialize_env(env_name, best_stats_path)
        best_traj, _ = sample_trajectory(best_env, best_expert, is_det=True)
        best_states = best_traj['states']

        # randomly sample one state and add to set of randomly sampled points
        point_idx = np.random.randint(low=0, high=len(best_states)-1)
        sampled_points.append(best_states[point_idx])

    
    input_lines_dict = {"origin":[], "mean":[]}
    mean_sample = np.mean(sampled_points, axis=0)
    for point in sampled_points:
        point1, point2 = (np.zeros_like(point), point)
        input_lines_dict["origin"].append((point1, point2))
        point1, point2 = (mean_sample, point)
        input_lines_dict["mean"].append((point1, point2))
    
    return input_lines_dict


def load_random_trajectories(sweep_dir):
    random_trajectories_dir = sweep_dir / "random_trajectories"
    if not os.path.exists(random_trajectories_dir):
        raise ValueError(f"Directory {random_trajectories_dir} is Empty. Must sample and save random trajectories first.")
    
    num_random_trajectories = len(os.listdir(random_trajectories_dir))
    random_trajectories = []

    for i in range(num_random_trajectories):
        path = Path(random_trajectories_dir) / f"random_traj_{i}.traj"
        if os.path.exists(path):
            random_traj = pickle.load(open(path, "rb"))
            random_trajectories.append(random_traj)
    
    return random_trajectories


def load_random_lines(sweep_dir):
    random_lines_dir = sweep_dir / "random_lines"
    if not os.path.exists(random_lines_dir):
        raise ValueError(f"Directory {random_lines_dir} is Empty. Must sample and save random lines first.")

    random_lines_dict = {}

    for name in ["origin", "mean"]:
        path = Path(random_lines_dir) / f"{name}.lines"
        if os.path.exists(path):
            random_lines_dict[name] = pickle.load(open(path, "rb"))

    return random_lines_dict

    
def main(args):
    api = wandb.Api()
    sweep = api.sweep(f"{args.entity}/{args.project_name}/sweeps/{args.sweep_id}")
    runs = sweep.runs
    runs = sorted(runs, key=lambda x: x.id)
    env_name = runs[0].config["env"]
    
    input_lines_dict = sample_random_lines(runs=runs, num_samples=args.num_random_lines)
    for name in ["origin", "mean"]:
        file_path = Path(args.sweeps_dir) / args.sweep_id / "random_lines" / f"{name}.lines"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pickle.dump(input_lines_dict[name], open(file_path, "wb"))

    for i in range(args.num_random_trajectories):
        file_path = Path(args.sweeps_dir) / args.sweep_id / "random_trajectories" / f"random_traj_{i}.traj"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        random_traj, _ = sample_random_trajectory(env_name)
        pickle.dump(random_traj, open(file_path, "wb"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--entity", help="Weights & Biases entity", type=str)
    parser.add_argument("--project_name", help="Weights & Biases project name", required=True, type=str)
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--sweeps_dir", type=str, default="sweeps_data")
    parser.add_argument("--num_random_lines", type=int, default=100)
    parser.add_argument("--num_random_trajectories", type=int, default=10)
    args = parser.parse_args()
    main(args)