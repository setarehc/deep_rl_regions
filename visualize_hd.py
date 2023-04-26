import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from argparse import ArgumentParser
from pathlib import Path
from stable_baselines3.common.policies import ActorCriticPolicy

from tools import initialize_env, sample_trajectory, sample_random_trajectory, get_wandbs
from regions_counting_2d import get_sample_plane, count_regions_2d


def build_dataset(num_samples = 10, best_env=None, best_expert=None, env_name=None):
    x_data = []
    for i in range(num_samples):
        if best_env is not None and best_expert is not None:
            best_traj, _ = sample_trajectory(best_env, best_expert, is_det=False)
            x_data.append(best_traj['states'])
        else:
            if env_name is not None:
                raise ValueError("Environment which trajectories are sampled from must be indicated!")
            random_traj = sample_random_trajectory(env_name)
            x_data.append(random_traj['states'])
    x_data = np.reshape(np.stack(x_data), [-1, np.stack(x_data).shape[-1]])
    return x_data


def plot_divisions_hd(regions, save_path, Xs, Ys, save_name=None):
    """
    Used for plotting the state space divisions over a plane intersecting the state space of a high dimensional environment.
    """
    fig, ax = plt.subplots()
    for region in regions:
        vertices = region.vertices
        _ = ax.fill(vertices[:, 1], -vertices[:, 0], c=np.random.rand(3))
    plt.xticks([], [])
    plt.yticks([], [])
    for i in range(3):
        plt.plot(Xs[i], Ys[i], '.', color='black', markersize=4.0)
    ax.set_aspect("equal")
    save_path = Path(save_path) / "divisions_hd.pdf" if save_name is None else save_name
    fig.savefig(save_path, bbox_inches="tight")


def main(args):
    # Load policy
    expert = PPO.load(args.policy_path)
    policy: ActorCriticPolicy = expert.policy
    # Initialie environment
    env = initialize_env(args.env, args.stats_path)

    # Load a dataset from trajectories sampled from final fully trained policy
    # args.policy_path and args.stats_path must be in the format "run_epoch/latest.zip" and "run_epoch/latest_stats.pth"
    best_policy_path = Path(args.policy_path).parents[0].parents[0] / 'best.zip'
    best_stats_path = Path(args.stats_path).parents[0].parents[0] / 'best_stats.pth'
    best_expert = PPO.load(best_policy_path)
    best_policy: ActorCriticPolicy = best_expert.policy
    best_env = initialize_env(args.env, best_stats_path)
    data = build_dataset(10, best_env, best_expert)
    fn_weight, fn_bias, Xs, Ys = get_sample_plane(data)
    
    [the_weights, the_biases] = get_wandbs(policy)

    regions = count_regions_2d(the_weights, the_biases,
                               fn_weight, fn_bias, return_regions=True)

    print(f"{len(regions)} exist in the provided plane")

    # Plot linear regions
    plot_divisions_hd(regions, args.save_path, Xs, Ys)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env", help="Name of the environment as defined in __init__.py somewhere", type=str,
                        required=True)
    parser.add_argument("--policy_path", help="Path to policy zip file, if any. Otherwise compute null actions",
                        type=str, required=True)
    parser.add_argument("--stats_path", help="Path to policy normalization stats", type=str, required=True)
    parser.add_argument("--save_path", help="Path where the output plot will be saved", type=str, default=None)
    args = parser.parse_args()
    main(args)