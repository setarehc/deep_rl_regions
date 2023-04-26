import torch
import gym
import numpy as np
import glob
import os
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv


### This part defines input-output dimensions for environments ###
in_out_dims = {
    "envs:Car1DEnv-v1": {"in": 2, "out": 1},
    "MountainCarContinuous-v0": {"in": 2, "out": 1},
    "HalfCheetah-v2": {"in": 17, "out": 6},
    "MountainCar-v0": {"in": 2, "out": 3},
    "Swimmer-v2": {"in": 8, "out": 2},
    "Walker2d-v2": {"in": 17, "out": 6},
    "Ant-v2": {"in": 111, "out": 8}
}
### End ###


class EnvFactory:
    """
    Factory pattern helps use parallel processing support more elegantly
    """
    def __init__(self, env_name):
        self.env_name = env_name

    def make_env(self):
        return gym.make(self.env_name)


def parse_str_arg(in_arg):
    if in_arg.startswith("c"):
        # in_arg format: c#,#,#
        in_arg = in_arg.lstrip("c")
        in_arg = in_arg.split(",")
        in_arg = list(map(lambda x: int(x), in_arg))
    return in_arg


def get_net_params(config_str, input_dims, output_dims):
    output = {}
    w = max(config_str)
    d = len(config_str)
    n = sum(config_str)
    weight_count = input_dims * config_str[0]
    for i in range(1, len(config_str)):
        weight_count += config_str[i-1] * config_str[i]
    weight_count += config_str[-1] * output_dims
    bias_count = n + output_dims
    output["depth"] = d
    output["width"] = w
    output["n"] = n
    output["n_params"] = weight_count + bias_count
    return output


def initialize_env(env_name, stats_path):
    factory = EnvFactory(env_name)
    env = DummyVecEnv([factory.make_env])
    env = VecNormalize.load(stats_path, env)
    return env


def extract_embeddings(env, policy, states, clip_actions=True, n_layer=None, normalize_obs=True):
    """
    Returns the activation pattern (binary and integer) of the policy network, for the given states from layer 1 to n_layer
    """
    env.clip_obs = np.inf
    env.training = False

    if normalize_obs:
        states_scaled = env.normalize_obs(states)
    else:
        states_scaled = states
        print("no normalization")
    states_tensor = torch.as_tensor(states_scaled).float().to(policy.device)
    
    policy_layers_content, actions, values = extract_features(states_tensor, policy, clip_actions, n_layer)

    binary_contents = []
    for content in policy_layers_content:
        binary_layer = content[1] > 0
        binary_contents.append(binary_layer.cpu().detach().numpy())
    
    binary_embeddings = np.concatenate(binary_contents, axis=1).astype(np.int)
    num_neurons = binary_embeddings.shape[1]#//2
    integer_embeddings = np.array([b.dot(1 << np.arange(b.size)[::-1]) for b in binary_embeddings])

    return {"integer": integer_embeddings, "binary": binary_embeddings}, num_neurons, actions, values


@torch.no_grad()
def extract_features(states, policy, clip_actions=True, n_layer=None):
    policy_net_depth = len(policy.mlp_extractor.policy_net)//2
    true_actions_tensor, true_values_tensor, log_prob = policy.forward(states, deterministic=True)
    features_tensor = policy.features_extractor.forward(states)
    shared_latents_tensor = policy.mlp_extractor.shared_net.forward(features_tensor)
    layers_content = []

    depth_lim = policy_net_depth if n_layer is None else n_layer

    for i in range(depth_lim):
        latents = policy.mlp_extractor.policy_net[i*2].forward(shared_latents_tensor if i == 0 else activations)
        activations = policy.mlp_extractor.policy_net[i*2+1].forward(latents)
        layers_content.append([latents, activations])
    
    actions_tensor = policy.action_net.forward(activations)
    values_tensor = policy.value_net.forward(policy.mlp_extractor.value_net.forward(shared_latents_tensor))
    
    assert actions_tensor.equal(true_actions_tensor)
    assert values_tensor.equal(true_values_tensor)

    # clip actions
    if clip_actions:
        actions_tensor = torch.clip(actions_tensor, min=-1, max=1)
    
    return layers_content, actions_tensor, values_tensor


def sample_trajectory(env, expert, is_det=True):
    """
    Returns a trajectory of states collected from the agent acting according to the policy in the environment.
    """
    trajectory = {}

    expert_state_dim = expert.observation_space.shape[0]
    policy: ActorCriticPolicy = expert.policy

    env.clip_obs = np.inf
    env.training = False

    obs = env.reset()
    total_reward = 0
    unnormalized_states = []
    actions = []
    unnormalized_rewards = [] 
    done = False

    num_states = 0
    
    unnormalized_states.append(env.unnormalize_obs(obs.reshape(-1)))

    while not done:
        if expert is None:
            action = env.action_space.sample()
            action = np.zeros_like(action)
        else:
            good_obs = obs[:, :expert_state_dim]
            action, _ = expert.predict(good_obs, deterministic=is_det)
        obs, reward, done, _ = env.step(action)
        num_states += 1
        unnormalized_state = env.unnormalize_obs(obs.reshape(-1))
        
        unnormalized_reward = env.unnormalize_reward(reward)
        total_reward += unnormalized_reward[0]
        
        if not done:
            unnormalized_states.append(unnormalized_state)
            actions.append(action.reshape(-1))
            unnormalized_rewards.append(unnormalized_reward)
        
    
    unnormalized_states = np.stack(unnormalized_states)
    actions = np.stack(actions)
    unnormalized_rewards = np.stack(unnormalized_rewards)
    

    trajectory = {"states": unnormalized_states, "actions": actions, "rewards": unnormalized_rewards}
    
    return trajectory, total_reward



def sample_random_trajectory(env_name, env_stats_path=None):
    """
    Samples a random trajectory by letting agent sampling random actions from the environment.
    
    Returned regions come from the regions visited during sampling the trajectory from the input policy.
    """
    trajectory = {}

    env = initialize_env(env_name, env_stats_path, clip_obs=np.inf)
    
    env.training = False

    obs = env.reset()
    total_reward = 0
    unnormalized_states = []
    actions = []
    unnormalized_rewards = []
    done = False

    num_states = 0
    unnormalized_states.append(env.unnormalize_obs(obs.reshape(-1)))
    
    while not done:
        action = env.action_space.sample()
        if env_name == "LunarLanderContinuous-v2" or "MountainCarContinuous-v0":
            obs, reward, done, _ = env.step([action])
        else:
            obs, reward, done, _ = env.step(action)
        num_states += 1
        unnormalized_state = env.unnormalize_obs(obs.reshape(-1))
        
        #reward = env.get_original_reward()
        unnormalized_reward = env.unnormalize_reward(reward)
        total_reward += unnormalized_reward[0]
        
        if not done:
            unnormalized_states.append(unnormalized_state)
            actions.append(action.reshape(-1))
            unnormalized_rewards.append(unnormalized_reward)
    
    unnormalized_states = np.stack(unnormalized_states)
    actions = np.stack(actions)
    unnormalized_rewards = np.stack(unnormalized_rewards)

    trajectory = {"states": unnormalized_states, "actions": actions, "rewards": unnormalized_rewards}
    print(f"Visited a total of {len(unnormalized_states)} states while sampling a single trajectory")
    
    return trajectory, total_reward


def get_wandbs(policy):
    weights = []
    biases = []
    # is a ppo policy
    policy_net_depth = len(policy.mlp_extractor.policy_net) // 2
    for i in range(policy_net_depth):
        weights.append(policy.mlp_extractor.policy_net[i*2].weight.cpu().detach().numpy())
        biases.append(policy.mlp_extractor.policy_net[i*2].bias.cpu().detach().numpy())
    return weights, biases


def get_sorted_sub_folders(folder_path):
    if len(set([os.path.isdir(os.path.join(folder_path, name)) for name in os.listdir(folder_path)])) > 1:
        dirs = glob.glob(f'{folder_path}/[0-9]*')
        dirs = filter(lambda x: os.path.basename(x).isdigit(), dirs)
        sorted_folder_paths = sorted(dirs, key=lambda x: int(os.path.split(x)[1]))
    else:
        sorted_folder_paths = [folder_path]
    return sorted_folder_paths


def get_net_params(config_str, input_dims, output_dims):
    output = {}
    w = max(config_str)
    d = len(config_str)
    n = sum(config_str)
    weight_count = input_dims * config_str[0]
    for i in range(1, len(config_str)):
        weight_count += config_str[i-1] * config_str[i]
    weight_count += config_str[-1] * output_dims
    bias_count = n + output_dims
    output["depth"] = d
    output["width"] = w
    output["n"] = n
    output["n_params"] = weight_count + bias_count
    return output


def sample_random_trajectory(env_name, env_stats_path=None):
    """
    Returns a random trajectory of states by letting the agent sample random actions from the environment.
    """
    trajectory = {}

    if env_stats_path is not None:
        env = initialize_env(env_name, env_stats_path)
        env.clip_obs = np.inf
    else:
        factory = EnvFactory(env_name)
        env = DummyVecEnv([factory.make_env])
        env = VecNormalize(env, clip_obs=np.inf)
    
    env.training = False

    obs = env.reset()
    total_reward = 0
    unnormalized_states = []
    actions = []
    unnormalized_rewards = [] 
    done = False

    num_states = 0
    
    unnormalized_states.append(env.unnormalize_obs(obs.reshape(-1)))

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        num_states += 1
        unnormalized_state = env.unnormalize_obs(obs.reshape(-1))
        
        unnormalized_reward = env.unnormalize_reward(reward)
        total_reward += unnormalized_reward[0]
        
        if not done:
            unnormalized_states.append(unnormalized_state)
            actions.append(action.reshape(-1))
            unnormalized_rewards.append(unnormalized_reward)
    
    unnormalized_states = np.stack(unnormalized_states)
    actions = np.stack(actions)
    unnormalized_rewards = np.stack(unnormalized_rewards)

    trajectory = {"states": unnormalized_states, "actions": actions, "rewards": unnormalized_rewards}
    
    return trajectory, total_reward