import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy


def train_acrobot_ppo():
    env = gym.make("Acrobot-v1")
    model = PPO('MlpPolicy', env, verbose=1, batch_size=5000)
    model.learn(total_timesteps=100000)
    model.save('models/ppo/acrobot_model_ppo')


def predict_acrobot_ppo():
    env = gym.make("Acrobot-v1", render_mode='human')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = PPO.load('models/ppo/acrobot_model_ppo.zip')
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    env.close()


def train_acrobot_a2c():
    env = gym.make("Acrobot-v1")
    model = A2C('MlpPolicy', env, verbose=1, n_steps=5000)
    model.learn(total_timesteps=100000)
    model.save('models/a2c/acrobot_model_a2c')


def predict_acrobot_a2c():
    env = gym.make("Acrobot-v1", render_mode='human')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # model = A2C.load('models/a2c/acrobot_model_a2c.zip')
    model = A2C.load('models/a2c/Acrobot-v1_1/Acrobot-v1.zip')

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    env.close()


def evaluate_acrobot():
    # Create the Acrobot environment
    env = gym.make("Acrobot-v1")
    env = Monitor(env, filename="logs/ppo/acrobot_evaluation")

    # Load the trained model
    model = PPO.load("models/ppo/acrobot_model_ppo.zip", env)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    # Print the performance metrics
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close the environment
    env.close()

    # Create the Acrobot environment
    env2 = gym.make("Acrobot-v1")
    env2 = Monitor(env2, filename="logs/a2c/acrobot_evaluation")

    # Load the trained model
    model2 = A2C.load("models/a2c/Acrobot-v1_1/Acrobot-v1.zip", env2)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model2, env2, n_eval_episodes=10)

    # Print the performance metrics
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close the environment
    env2.close()


if __name__ == "__main__":
    # train_acrobot_ppo()
    # predict_acrobot_ppo()
    # train_acrobot_a2c()
    # predict_acrobot_a2c()
    evaluate_acrobot()
