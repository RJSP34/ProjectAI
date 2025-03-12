import gym
from gym.envs.registration import register

env_id = 'SpaceInvadersNoFrameskip-v4'

try:
    gym.spec(env_id)
    print(f"The '{env_id}' environment is already registered.")
except gym.error.UnregisteredEnv:
    register(
        id=env_id,
        entry_point='gym.envs.atari:AtariEnv',
        kwargs={'game': 'space_invaders', 'obs_type': 'image', 'frameskip': 1},
        max_episode_steps=10000,
        nondeterministic=False,
    )
    print(f"The '{env_id}' environment has been registered successfully.")

env = gym.make(env_id)