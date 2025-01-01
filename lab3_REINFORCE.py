import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""0. Для  обраного  у  лабораторній  роботі  2  середовища (Cart Pole) реалізувати наступний метод навчання з підкріпленням: REINFORCE"""

env = gym.make('CartPole-v1')
EPOCHS = 100
EPISODES = 150
GAMMA = 0.99
EPSILON = 0.9
LEARNING_RATE = 0.001

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# Функція для обчислення виграшу (returns)
def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy = PolicyNetwork(input_dim, output_dim)
optimizer = optim.RMSprop(policy.parameters(), lr=LEARNING_RATE)


"""1. Провести  навчання  агента"""

def train_agent(EPOCHS, EPISODES, GAMMA, EPSILON, LEARNING_RATE):
    # Для зберігання метрик
    avg_scores = []
    avg_Gt_v = []
    max_Gt_v = []

    # Тренування
    for epoch in range(EPOCHS):
        epoch_rewards = []
        epoch_Gt_v = []
        
        for episode in range(EPISODES):
            state, _ = env.reset()
            done = False
            episode_rewards = []
            log_probs = []
            
            while not done:
                state = torch.tensor(state, dtype=torch.float32)
                action_probs = policy(state.unsqueeze(0))
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()

                log_probs.append(action_dist.log_prob(action))

                next_state, reward, done, truncated, _ = env.step(action.item())
                if truncated or done:
                    break
                episode_rewards.append(reward)
                state = next_state

            returns = compute_returns(episode_rewards, GAMMA)
            returns = torch.tensor(returns, dtype=torch.float32)
            
            epoch_rewards.append(sum(episode_rewards))
            epoch_Gt_v.append(returns[0].item())
            
            log_probs_tensor = torch.stack(log_probs)
            loss = -torch.sum(log_probs_tensor * (returns - returns.mean()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_scores.append(np.mean(epoch_rewards))
        avg_Gt_v.append(np.mean(epoch_Gt_v))
        max_Gt_v.append(np.max(epoch_Gt_v))

        print(f"Epoch {epoch + 1}/{EPOCHS}: Avg Reward = {avg_scores[-1]:.2f}, Avg Gt-v = {avg_Gt_v[-1]:.2f}, Max Gt-v = {max_Gt_v[-1]:.4f}")
    
    return avg_scores, avg_Gt_v, max_Gt_v


avg_scores, avg_Gt_v, max_Gt_v = train_agent(EPOCHS, EPISODES, GAMMA, EPSILON, LEARNING_RATE)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(avg_scores)
plt.title('Average Reward per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Reward')

plt.subplot(1, 3, 2)
plt.plot(avg_Gt_v)
plt.title('Average Gt - v(s) per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Gt - v(s)')

plt.subplot(1, 3, 3)
plt.plot(max_Gt_v)
plt.title('Maximum Gt - v(s) per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Max Gt - v(s)')

plt.tight_layout()
plt.savefig("1-REINFORCE Graphs")
plt.show()


"""2. Для  методу  навчання  агента  провести  тестування, запустивши 100 епізодів. 
Виведіть на екран два графіки: винагорода та тривалість епізоду. """

def test_agent(test_episodes):
    # Для зберігання метрик
    all_rewards = []
    episode_durations = []

    # Тестування
    for test_episode in range(test_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        rewards_per_step = []

        while not done:
            state = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action_probs = policy(state.unsqueeze(0))
                action = torch.argmax(action_probs).item()
            
            next_state, reward, done, truncated, _ = env.step(action)
            if truncated:
                done = True

            episode_reward += reward
            rewards_per_step.append(reward)
            state = next_state
            step += 1

        all_rewards.append(episode_reward)
        episode_durations.append(step)

    return all_rewards, episode_durations


test_episodes = 100
all_rewards, episode_durations = test_agent(test_episodes)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(all_rewards)
plt.title('Rewards per Test Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(1, 2, 2)
plt.plot(episode_durations)
plt.title('Duration per Test Episode')
plt.xlabel('Episode')
plt.ylabel('Duration')

plt.tight_layout()
plt.savefig("2-REINFORCE Graphs")
plt.show()
