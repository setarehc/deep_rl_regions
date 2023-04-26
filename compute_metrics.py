import pickle
from pathlib import Path
from stable_baselines3 import PPO
from tools import *
from regions_counting_1d import *
from randomized_metric_helper import load_random_lines, load_random_trajectories


"""
This file contains all the functions required for creating the metrics reported in the paper. 
Note for creating normalized versions of metrics, some post processing is required. For instance, to create normalized densities,
one should load the metrics corresponding to number of transitions, trajectory length, and number of neurons. And then, divide 
number of transitions by the multiplication of trajectory length and number of neurons.
"""


def get_history(results_dict, run_folder_path, run_env_name, random_evaluation_dir, policy_type="deterministic", trajectory_type="fixed", 
                evaluate_random_trajectories=False, evaluate_random_lines=False, multi_processing=False):
    """ 
    Compute metrics and return a dictionary of computed metrics.
    """
    sorted_folder_paths = get_sorted_sub_folders(run_folder_path)

    # Compute mean number of regions and transitions for random trajectories
    if evaluate_random_trajectories:
        if random_evaluation_dir is None:
            raise ValueError("Must input directory of random trajectories and lines!")
        random_trajectories = load_random_trajectories(random_evaluation_dir)

    # Compute mean number of regions and transitions for random lines
    if evaluate_random_lines:
        if random_evaluation_dir is None:
            raise ValueError("Must input directory of random trajectories and lines!")
        random_lines_dict = load_random_lines(random_evaluation_dir)
    
    if trajectory_type == "fixed":
        path = Path(run_folder_path) / f"best_{policy_type}.traj"
        if os.path.exists(path):
            trajectory = pickle.load(open(path, "rb"))
        else:
            best_policy_path = Path(run_folder_path) / "best.zip"
            best_stats_path = Path(run_folder_path) / "best_stats.pth"
            best_expert = PPO.load(best_policy_path)
            best_env = initialize_env(run_env_name, best_stats_path)
            trajectory, _ = sample_trajectory(best_env, best_expert, is_det=(policy_type == "deterministic"))
            pickle.dump(trajectory, open(path, "wb"))
    else:
        trajectory = None

    for epoch_path in sorted_folder_paths:
        epoch = int(os.path.basename(epoch_path))

        policy_path = Path(epoch_path) / "latest.zip"
        stats_path = Path(epoch_path) / "latest_stats.pth"
        # load policy
        expert = PPO.load(policy_path)
        policy: ActorCriticPolicy = expert.policy
        # initialize environment for input standardization
        env = initialize_env(run_env_name, stats_path)

        # get trajectory from current snapshot of the policy if required
        if trajectory_type == "current":
            if epoch == 0:
                assert trajectory is None
            path = Path(epoch_path) / f"{policy_type}.traj"
            if os.path.exists(path):
                trajectory = pickle.load(open(path, "rb"))
            else:
                trajectory, _ = sample_trajectory(env, expert, is_det=(policy_type == "deterministic"))
                pickle.dump(trajectory, open(path, "wb"))
        
        # extract trajectory states
        states = trajectory["states"]

        if multi_processing:
            num_regions, num_transitions = get_visited_regions_analytic_parallel(env, policy, states)
        else:
            num_regions, num_transitions = get_visited_regions_analytic(env, policy, states)
                
        # log to wandb
        results_dict[epoch].update({f"num_regions_{policy_type}_{trajectory_type}": num_regions,
                                    f"num_transitions_{policy_type}_{trajectory_type}": num_transitions,
                                    f"{policy_type}_{trajectory_type}_trajectory_timesteps": len(states),
                                    f"{policy_type}_{trajectory_type}_trajectory_length": compute_trajectory_length(states),
                                    f"{policy_type}_{trajectory_type}_trajectory_length_normalized": compute_trajectory_length(states, env)})
        
        if evaluate_random_trajectories:
            num_regions_list = []
            num_transitions_list = []
            traj_len_list = []
            best_stats_path = Path(run_folder_path) / "best_stats.pth"
            best_env = initialize_env(run_env_name, best_stats_path)
            for random_traj in random_trajectories:
                states = random_traj["states"]
                if multi_processing:
                    num_regions, num_transitions = get_visited_regions_analytic_parallel(best_env, policy, states)
                else:
                    num_regions, num_transitions = get_visited_regions_analytic(best_env, policy, states)
                num_regions_list.append(num_regions)
                num_transitions_list.append(num_transitions)
                traj_len_list.append(compute_trajectory_length(states))
            
            mean_num_regions = np.mean(num_regions_list)
            std_num_regions = np.std(num_regions_list)
            mean_num_transitions = np.mean(num_transitions_list)
            std_num_transitions = np.std(num_transitions_list)
            
            mean_normalized_num_regions = np.mean(np.array(num_regions_list)/np.array(traj_len_list))
            std_normalized_num_regions = np.std(np.array(num_regions_list)/np.array(traj_len_list))
            mean_normalized_num_transitions = np.mean(np.array(num_transitions_list)/np.array(traj_len_list))
            std_normalized_num_transitions = np.std(np.array(num_transitions_list)/np.array(traj_len_list))
            
            mean_traj_length = np.mean(traj_len_list)
            std_traj_length = np.std(traj_len_list)
            
            results_dict[epoch].update({f"mean_num_regions_random_trajectories": mean_num_regions,
                                        f"std_num_regions_random_trajectories": std_num_regions,
                                        f"mean_num_transitions_random_trajectories": mean_num_transitions,
                                        f"std_num_transitions_random_trajectories": std_num_transitions,
                                        f"mean_normalized_num_regions_random_trajectories": mean_normalized_num_regions,
                                        f"std_normalized_num_regions_random_trajectories": std_normalized_num_regions,
                                        f"mean_normalized_num_transitions_random_trajectories": mean_normalized_num_transitions,
                                        f"std_normalized_num_transitions_random_trajectories": std_normalized_num_transitions,
                                        f"mean_trajectory_length_random_trajectories": mean_traj_length,
                                        f"std_trajectory_length_random_trajectories": std_traj_length})

        if evaluate_random_lines:
            num_regions_list = []
            num_transitions_list = []
            best_stats_path = Path(run_folder_path) / "best_stats.pth"
            best_env = initialize_env(run_env_name, best_stats_path)
            for name, lines in random_lines_dict.items():
                normalized_lines = [normalize_line_segment(best_env, p1 , p2) for (p1, p2) in lines]
                input_lines = [(p2-p1, p1) for (p1, p2) in normalized_lines]
                [weights, biases] = get_wandbs(policy)
                counts = [count_regions_1d(weights, biases, input_line_weight, input_line_bias, return_regions=False)
                         for (input_line_weight, input_line_bias) in input_lines]
                
                mean_num_regions = np.mean(counts)
                std_num_regions = np.std(counts)
                results_dict[epoch].update({f"mean_num_regions_random_lines_{name}": mean_num_regions,
                                            f"std_num_regions_random_lines_{name}": std_num_regions})
        gc.collect()
    # all metrics are added to results_dict


def get_summary(run_folder_path, run_env_name, run_policy_dims, policy_type="deterministic", is_PPO=True, multi_processing=False):
    # compute a summary including metrics and network info and return a json file or something
    summary_dict = {}

    # add network parameters to wandb
    in_dims = in_out_dims[run_env_name]["in"]
    out_dims = in_out_dims[run_env_name]["out"]
    config_dict = get_net_params(parse_str_arg(run_policy_dims), input_dims=in_dims, output_dims=out_dims)
    summary_dict.update({"depth": config_dict["depth"],
                         "width": config_dict["width"],
                         "n": config_dict["n"],
                         "n_params": config_dict["n_params"]})
    
    # add metrics computed on the best snapshot of the policy
    best_policy_path = Path(run_folder_path) / "best.zip"
    best_expert = PPO.load(best_policy_path)
    best_stats_path = Path(run_folder_path) / "best_stats.pth"
    best_env = initialize_env(run_env_name, best_stats_path)
    best_policy: ActorCriticPolicy = best_expert.policy
    
    path = Path(run_folder_path) / f"best_{policy_type}.traj"
    if os.path.exists(path):
        trajectory = pickle.load(open(path, "rb"))
    else:
        trajectory, _ = sample_trajectory(best_env, best_expert, is_det=(policy_type == "deterministic"))
        pickle.dump(trajectory, open(path, "wb"))
        
    states = trajectory["states"]
    length = compute_trajectory_length(states)
    
    if multi_processing:
        num_regions, num_transitions = get_visited_regions_analytic_parallel(best_env, best_policy, states)
    else:
        num_regions, num_transitions = get_visited_regions_analytic(best_env, best_policy, states)

    # log to wandb
    summary_dict.update({f"best_{policy_type}_num_regions": num_regions,
                         f"best_{policy_type}_num_transitions": num_transitions,
                         f"best_{policy_type}_trajectory_timesteps": len(states),
                         f"best_{policy_type}_trajectory_length": length,
                         f"best_normalized_{policy_type}_num_transitions": num_transitions / length,
                         f"normalize_obs_{policy_type}": best_env.norm_obs})
    
    # return computed summaries dict
    return summary_dict


def evaluate_analytic(env_name, policy_dims, run_folder_path, stochastic_policy=False,
                      random_evaluation_dir=None, evaluate_random_trajectories=False, 
                      evaluate_random_lines=False, multi_processing=False):
    """
    Computes all metrics and summaries for a (wandb) run with the input information
    """
    policy_types = ["deterministic", "stochastic"] if stochastic_policy else ["deterministic"]

    # compute metrics
    num_epochs = len(get_sorted_sub_folders(run_folder_path))
    final_metrics_dict = {epoch: {} for epoch in range(num_epochs)}
    for policy_type in policy_types:
        for trajectory_type in ["fixed", "current"]:
            get_history(final_metrics_dict, run_folder_path, env_name, random_evaluation_dir, policy_type, trajectory_type,
                        evaluate_random_trajectories, evaluate_random_lines, multi_processing)
    
    # compute summary
    final_summary_dict = {}
    for policy_type in policy_types:
        final_summary_dict.update(get_summary(run_folder_path, env_name, policy_dims, policy_type, multi_processing))
    
    # return metrics and summry
    return final_metrics_dict, final_summary_dict
