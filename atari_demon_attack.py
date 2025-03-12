import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


def train_atari_demon_attack_ppo():
    env = gym.make("ALE/DemonAttack-v5")
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO("MlpPolicy", env, verbose=1)

    model.learn(int(2e5))
    model.save("models/ppo/atari_demon_attack_model_ppo")


def predict_atari_demon_attack_ppo():
    # Load the model
    env = gym.make("ALE/DemonAttack-v5", render_mode="human")
    model = PPO.load("models/ppo/Atari_Demon_Attack")

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


def train_atari_demon_attack_a2c():
    env = gym.make("ALE/DemonAttack-v5")
    model = A2C('MlpPolicy', env, verbose=1, n_steps=5000)
    model.learn(total_timesteps=100000)
    model.save('models/a2c/Atari_Demon_model')


def predict_atari_demon_attack_a2c():
    env = gym.make("ALE/DemonAttack-v5", render_mode='human')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = A2C.load('models/a2c/Atari_Demon_model')
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

            # Close the environment
    env.close()


def evaluate_demonattack():
    # Load the trained model
    model = PPO.load("models/ppo/Atari_Demon_Attack.zip")

    # Create the Acrobot environment
    env = gym.make("ALE/DemonAttack-v5")
    env = Monitor(env, filename="logs/ppo/demonattack_evaluation")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    # Print the performance metrics
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close the environment
    env.close()

    # Load the trained model
    model2 = A2C.load("models/a2c/Atari_Demon_model.zip")


    # Create the Acrobot environment
    env2 = gym.make("ALE/DemonAttack-v5")
    env2 = Monitor(env2, filename="logs/a2c/demonattack_evaluation")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model2, env2, n_eval_episodes=10)

    # Print the performance metrics
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close the environment
    env2.close()


if __name__ == "__main__":
    # train_atari_demon_attack_ppo()
    #  predict_atari_demon_attack_ppo()
    # train_atari_demon_attack_a2c()
    # predict_atari_demon_attack_a2c()
    evaluate_demonattack()

