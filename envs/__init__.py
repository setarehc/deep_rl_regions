import gym
import os


def register(id, **kvargs):
    if id in gym.envs.registration.registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, **kvargs)


# fixing package path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

register(
    id="Car1DEnv-v0",
    entry_point="envs.toy_envs:Car1DEnv0",
    max_episode_steps=1000,
)

register(
    id="Car1DEnv-v1",
    entry_point="envs.toy_envs:Car1DEnv1",
    max_episode_steps=1000,
)

register(
    id="Car1DEnv-v2",
    entry_point="envs.toy_envs:Car1DEnv2",
    max_episode_steps=1000,
)