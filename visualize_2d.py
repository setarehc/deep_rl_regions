
import matplotlib.pyplot as plt
from pathlib import Path
from colorhash import ColorHash
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from argparse import ArgumentParser
from tools import initialize_env, extract_embeddings, sample_trajectory
from plotting_tools import *


def sample_states(x_min=-50, x_max=150, x_dot_min=-20, x_dot_max=20, num_samples=300):
    """
    Performs grid sampling of states in the region indicated by the inputs.
    Change the inputs according to the environment specificatoins.
    """
    states = []
    for i in np.linspace(x_min, x_max, num_samples):
        for j in np.linspace(x_dot_min, x_dot_max, num_samples):
            states.append([i, j])
    states = np.stack(states)
    return states


def plot_divisions_2d(env, policy, states, save_path, scale=1.0, trajectory_states=None, visited_states=None, save_name=None, n_layer=None, normalize_obs=True):
    """
    Used for plotting the state space divisions of 2D environments.
    """
    # Extract activation patterns for sampled states from the input space
    embeddings_dict, num_neurons, _, _ = extract_embeddings(env, policy, states, n_layer=n_layer, normalize_obs=normalize_obs) 
    integer_embeddings = embeddings_dict["integer"]

    # Create color map corresponding to embeddings
    color_map = [ColorHash(i).hex for i in integer_embeddings]
    
    plt.figure(figsize=set_size("neurips22", fraction=scale))
    plt.scatter(states[:, 0], states[:, 1], s=1, c=color_map, rasterized=True)
    # Mark start and end points
    plt.plot(0.0, 0.0, 'o', color='blue', markersize=4*scale)
    plt.plot(100.0, 0.0, '*', color='red', markersize=4*scale)

    if trajectory_states is not None:
        # Mark start and end states of the trajectory
        plt.plot(trajectory_states[0,0], trajectory_states[0,1], 'o', color='blue', markersize=4*scale)
        plt.plot(trajectory_states[-1,0], trajectory_states[-1,1], 'o', color='black', markersize=4*scale)
        # Plot trajectory
        deltas = trajectory_states[1:] - trajectory_states[:-1]
        plt.quiver(np.asarray(trajectory_states[:-1, 0]), trajectory_states[:-1, 1], deltas[:, 0], deltas[:, 1], angles='xy', scale_units='xy',
                    scale=1., width=0.005, headaxislength=0)
        plt.title(f"State Space and A Trajectory Visualized with {num_neurons} Neurons")
        file_name = "divisionsPlusTrajectory.pdf"

    elif visited_states is not None:
        plt.scatter(visited_states[:, 0], visited_states[:, 1], c='black', alpha=0.085, cmap='virdis', s=0.1)
        plt.title(f"State Space and States Visited During Training Visualized with {num_neurons} Neurons")
        file_name = "divisionsPlusVisitedStates.pdf"
    
    else:
        plt.title(f"State Space Visualized with {num_neurons} Neurons")
        file_name = "divisions.pdf"
    
    plt.xlim(-50,150)
    plt.ylim(-20,20)

    save_path = Path(save_path) / file_name if save_name is None else save_name
    plt.savefig(save_path, bbox_inches="tight")


def main(args):
    # Grid sampling of input states
    states = sample_states()
    # Load policy
    expert = PPO.load(args.policy_path)
    policy: ActorCriticPolicy = expert.policy
    # Initialize environment
    env = initialize_env(args.env, args.stats_path)
    # Sample a deterministic trajectory from the policy
    trajectory, _ = sample_trajectory(env, expert)
    plot_divisions_2d(env, policy, states, save_path=args.save_path, trajectory_states=trajectory["states"])


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
