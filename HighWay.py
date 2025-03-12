import time

import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


def train_ppo_highway():
    env = gym.make('highway-fast-v0')
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), learning_rate=5e-4, )

    model.learn(int(2e4))
    model.save("models/ppo/highway")


def testing_ppo_highway():
    # Load the model
    env = gym.make("highway-fast-v0", render_mode="human")
    model = PPO.load("models/ppo/highway.zip")

    # Create a new instance of the environment
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Run the model on the environment for 1000 steps
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    # Close the environment
    env.close()


def record_ppo_highway():
    # Load the model
    env = gym.make("highway-fast-v0", render_mode="human")
    model = PPO.load("models/ppo/highway.zip")

    record_dir = "Videos/ppo/highway"
    env = Monitor(env, record_dir)

    # Create a new instance of the environment
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Run the model on the environment for 1000 steps
    obs = env.reset()
    for i in range(10000000000000000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.1)

    # Close the environment
    env.close()


def train_highway_merge_ppo():
    env = gym.make('merge-v0')
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), learning_rate=5e-4, )

    model.learn(int(2e4))
    model.save("models/ppo/highway_merge")


def testing_ppo_highway_merge():
    # Load the model
    env = gym.make("merge-v0", render_mode="human")
    model = PPO.load("models/ppo/highway_merge.zip")

    # Create a new instance of the environment
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Run the model on the environment for 1000 steps
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    # Close the environment
    env.close()


def train_highway_roundabout_ppo():
    env = gym.make('roundabout-v0')
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), learning_rate=5e-4, )

    model.learn(int(2e4))
    model.save("models/ppo/highway_roundabout")


def testing_ppo_highway_roundabout():
    # Load the model
    env = gym.make("roundabout-v0", render_mode="human")
    model = PPO.load("models/ppo/highway_roundabout.zip")

    # Create a new instance of the environment
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Run the model on the environment for 1000 steps
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    # Close the environment
    env.close()


def train_highway_a2c():
    env = gym.make("highway-fast-v0")
    model = A2C('MlpPolicy', env, verbose=1, n_steps=5000)
    model.learn(total_timesteps=100000)
    model.save("models/a2c/highway")


def predict_highway_a2c():
    env = gym.make("highway-v0", render_mode='human')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = A2C.load('models/a2c/highway')
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    env.close()


def train_highway_merge_a2c():
    env = gym.make("merge-v0")
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = A2C('MlpPolicy', env, verbose=1, n_steps=5000)
    model.learn(total_timesteps=100000)
    model.save("models/a2c/highway_merge")


def predict_highway_merge_a2c():
    env = gym.make("merge-v0", render_mode='human')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = A2C.load('models/a2c/highway_merge')
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    env.close()


def train_highway_roundabout_a2c():
    env = gym.make("roundabout-v0")
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = A2C("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), learning_rate=5e-4, )

    model.learn(int(2e5))
    model.save("models/a2c/highway_roundabout")


def predict_highway_roundabout_a2c():
    env = gym.make("roundabout-v0", render_mode='human')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = A2C.load('models/a2c/highway_roundabout')
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    env.close()


def train_highway_dqn():
    env = gym.make("highway-v0")
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1)
    model.learn(int(2e4))

    model.learn(total_timesteps=1000)
    model.save("models/dqn/highway")


def testing_dqn_highway():
    # Load the model
    env = gym.make("highway-v0", render_mode="human")
    model = DQN.load("models/dqn/highway.zip")

    # Create a new instance of the environment
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Run the model on the environment for 1000 steps
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    # Close the environment
    env.close()


def train_highway_merge_dqn():
    env = gym.make("merge-v0")
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("models/dqn/highway_merge")


def predict_highway_merge_dqn():
    env = gym.make("merge-v0", render_mode='human')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = DQN.load('models/dqn/highway_merge')
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    env.close()


def train_highway_roundabout_dqn():
    env = gym.make("roundabout-v0")
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = DQN("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), learning_rate=5e-4, )

    model.learn(int(2e4))
    model.save("models/dqn/highway_roundabout")


def predict_highway_roundabout_dqn():
    env = gym.make("roundabout-v0", render_mode='human')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = DQN.load('models/dqn/highway_roundabout')
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    env.close()


def evaluate_highway():
    # Load the trained model
    model = DQN.load("models/dqn/highway.zip")

    # Create the Acrobot environment
    env = gym.make("highway-v0")
    env = Monitor(env, filename="logs/dqn/highway_evaluation")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    # Print the performance metrics
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close the environment
    env.close()

    # Load the trained model
    model2 = A2C.load("models/a2c/highway.zip")

    # Create the Acrobot environment
    env2 = gym.make("highway-v0")
    env2 = Monitor(env2, filename="logs/a2c/highway_evaluation")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model2, env2, n_eval_episodes=10)

    # Print the performance metrics
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close the environment
    env2.close()

    # Load the trained model
    model3 = PPO.load("models/ppo/highway.zip")

    # Create the Acrobot environment
    env3 = gym.make("highway-v0")
    env3 = Monitor(env3, filename="logs/ppo/highway_evaluation")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model3, env3, n_eval_episodes=10)

    # Print the performance metrics
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close the environment
    env3.close()


def evaluate_merge():
    # Load the trained model
    #  model = DQN.load("models/dqn/highway_merge.zip")

    # Create the Acrobot environment
    # env = gym.make("merge-v0")
    # env = Monitor(env, filename="logs/dqn/highway_merge_evaluation")

    # Evaluate the model
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    # Print the performance metrics
    # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close the environment
    #  env.close()

    # Load the trained model
    model2 = A2C.load("models/a2c/highway_merge.zip")

    # Create the Acrobot environment
    env2 = gym.make("merge-v0")
    env2 = Monitor(env2, filename="logs/a2c/highway_merge_evaluation")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model2, env2, n_eval_episodes=10)

    # Print the performance metrics
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close the environment
    env2.close()

    # Load the trained model
    model3 = PPO.load("models/ppo/highway_merge.zip")

    # Create the Acrobot environment
    env3 = gym.make("merge-v0")
    env3 = Monitor(env3, filename="logs/ppo/highway_merge_evaluation")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model3, env3, n_eval_episodes=10)

    # Print the performance metrics
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close the environment
    env3.close()


if __name__ == "__main__":
    # train_ppo_highway()
    # testing_ppo_highway()
    # record_ppo_highway()
    # train_highway_a2c()
    # predict_highway_a2c()
    # train_highway_merge_ppo()
    # train_highway_merge_a2c()
    # train_highway_roundabout_ppo()
    # train_highway_roundabout_a2c()
    # predict_highway_roundabout_a2c()
    # testing_ppo_highway_roundabout()
    # train_highway_dqn()
    # testing_dqn_highway()
    # evaluate_highway()
    evaluate_merge()
