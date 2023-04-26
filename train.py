import numpy as np
import torch
import random
import wandb
import os
import sys
from pathlib import Path
from gym import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm
from argparse import ArgumentParser
from tools import parse_str_arg, EnvFactory


if "--unobserve" in sys.argv:
    os.environ["WANDB_MODE"] = "dryrun"
    

def set_seed(seed, env):
    if seed is not None:
        random.seed(seed+1)
        np.random.seed(seed+2)
        torch.manual_seed(seed+3)
        env.seed(seed)


def get_cumulative_rewards_from_vecenv_results(reward_rows, done_rows):
    """
    Rewards and dones are 2d numpy arrays.
    each row corresponds to a process running the environment.
    each column corresponds to a timestep.
    for each row, we accumulate up to the first case where done=True
    """
    cumulative_rewards = []
    for reward_row, done_row in zip(reward_rows, done_rows):
        cumulative_reward = 0
        for reward, done in zip(reward_row, done_row):
            cumulative_reward += reward
            if done:
                break
        cumulative_rewards.append(cumulative_reward)
    return np.array(cumulative_rewards)


class WAndBEvalCallback(BaseCallback):
    def __init__(self, render_env: Env, eval_every: int, envs: VecNormalize, verbose=0):
        self.render_env = render_env  # if render with rgb_array is implemented, use this to collect images
        self.eval_every = eval_every
        self.best_cumulative_rewards_mean = -np.inf
        self.envs = envs
        self.iteration = 0
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        metrics = {"n_calls": self.n_calls}

        run_dir = Path(args.save_dir) / f"{args.project_name}_{args.run_name}"
        os.makedirs(run_dir, exist_ok=True)

        # save snapshot of policy
        if self.iteration % args.save_every == 0:
            save_dir = os.path.join(run_dir, "{:d}".format(self.iteration))

            # save policy weights
            self.model.save(os.path.join(save_dir, "latest.zip".format(args.project_name, args.run_name)))

            # save stats for normalization
            stats_path = os.path.join(save_dir, "latest_stats.pth")
            self.envs.save(stats_path)

        if self.n_calls % self.eval_every == 0:
            obs_column = self.envs.reset()
            reward_columns = []
            done_columns = []
            actions = []

            self.envs.training = False
            
            for i in range(1000):
                action_column, states = self.model.predict(obs_column, deterministic=True)
                next_obs_column, old_reward_column, done_column, info = self.envs.step(action_column)
                for a in action_column:
                    actions.append(a)
                reward_column = self.envs.get_original_reward()
                reward_columns.append(reward_column)
                done_columns.append(done_column)
                obs_column = next_obs_column

            self.envs.training = True
            reward_rows = np.stack(reward_columns).transpose()
            done_rows = np.stack(done_columns).transpose()
            cumulative_rewards = get_cumulative_rewards_from_vecenv_results(reward_rows, done_rows)
            cumulative_rewards_mean = np.mean(cumulative_rewards)

            if cumulative_rewards_mean > self.best_cumulative_rewards_mean:
                self.best_cumulative_rewards_mean = cumulative_rewards_mean
                self.model.save(os.path.join(run_dir, "best.zip"))
                self.envs.save(os.path.join(run_dir, "best_stats.pth"))

            metrics.update({"cumulative_rewards_mean": cumulative_rewards_mean})

        self.iteration += 1
        wandb.log(metrics)


def main(args):
    """
        
    """
    if args.seed is not None:
        set_random_seed(args.seed)

    wandb.init(project=args.project_name, name=args.run_name, tags=["train"], entity=args.entity)

    if args.run_name is None:
        args.run_name = wandb.run.id
    wandb.config.update(args)
    args.policy_dims = parse_str_arg(args.policy_dims)
    args.value_dims = parse_str_arg(args.value_dims)

    n_envs = len(os.sched_getaffinity(0)) # number of cpus available to the current process
    factory = EnvFactory(args.env)

    render_env = factory.make_env() # used for rendering

    # Wrap the environment around parallel processing friendly wrapper, unless debug is on
    if args.debug:
        envs = DummyVecEnv([factory.make_env for _ in range(n_envs)])
    else:
        envs = SubprocVecEnv([factory.make_env for _ in range(n_envs)])

    if args.stats_path is None:
        envs = VecNormalize(envs, norm_obs=True, clip_obs=np.inf) # norm_obs=True before
    else:
        envs = VecNormalize.load(args.stats_path, envs)

    eval_callback = WAndBEvalCallback(render_env, args.eval_every, envs)
    
    if args.seed is not None:
        set_seed(args.seed, envs)

    print("Do random explorations to build running averages")
    envs.reset()
    for _ in tqdm(range(1000)):
        random_action = np.stack([envs.action_space.sample() for _ in range(n_envs)])
        envs.step(random_action)
    envs.training = False  # freeze the running averages
    
    # We use PPO by default, but it is easy to swap out for other algorithms
    if args.pretrained_path is not None:
        pretrained_path = args.pretrained_path
        learner = PPO.load(pretrained_path, envs, device=args.device)
        learner.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    else:
        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                             net_arch=[dict(vf=args.value_dims, pi=args.policy_dims)],
                             log_std_init=args.log_std_init,
                             squash_output=False)

        learner = PPO(MlpPolicy, envs, n_steps=args.n_steps, verbose=1, 
                      policy_kwargs=policy_kwargs, device=args.device, 
                      learning_rate=args.learning_rate, batch_size=args.batch_size)
    
        learner.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    render_env.close()
    envs.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--entity", help="Weights & Biases entity", type=str)
    parser.add_argument("--project_name", help="Weights & Biases project name", required=True, type=str)
    parser.add_argument("--run_name", help="Weights & Biases run name", type=str)
    parser.add_argument("--unobserve", help="Disable Weights & Biases", action="store_true")
    parser.add_argument("--env", help="Name of the environment as registered in __init__.py somewhere", required=True,
                        type=str)
    parser.add_argument("--n_steps", help="Number of timesteps in each rollouts when training with PPO", required=True,
                        type=int)
    parser.add_argument("--total_timesteps", help="Total timesteps to train with PPO", required=True,
                        type=int)
    parser.add_argument("--policy_dims", help="Hidden layers for policy network", type=str, required=True)
    parser.add_argument("--value_dims", help="Hidden layers for value predictor network", type=str, required=True)
    parser.add_argument("--eval_every", help="Evaluate current policy every eval_every episodes", required=True,
                        type=int)
    parser.add_argument("--pretrained_path", help="Path to the pretrained policy zip file, if any", type=str)
    parser.add_argument("--stats_path", help="Path to the pretrained policy normalizer stats file, if any", type=str)
    parser.add_argument("--log_std_init", help="Initial Gaussian policy exploration level", type=float, default=-2.0)
    parser.add_argument("--debug", help="Set true to disable parallel processing and run debugging programs",
                        action="store_true")
    parser.add_argument("--save_every", help="Save current policy every save_every iterations", required=True, type=int)
    parser.add_argument("--device", help="Device option for stable baselines algorithms", default="cpu")
    parser.add_argument("--seed", help="Random seed", type=int)
    parser.add_argument("--learning_rate", default="3e-4", type=float)
    parser.add_argument("--save_dir", default="checkpoints", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)

