import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""0. Для  обраного  у  лабораторній  роботі  2  середовища (Cart Pole) реалізувати наступний метод навчання з підкріпленням: ACTOR-CRITIC"""

env = gym.make('CartPole-v1')
EPOCHS = 100
EPISODES = 150
GAMMA = 0.99
LEARNING_RATE = 0.001

# Мережа Actor
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 60),
            nn.ReLU(),
            nn.Linear(60, output_dim),
            nn.Softmax(dim=-1)
        )
 
    def forward(self, x):
        return self.fc(x)
 
# Мережа Critic
class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 60),
            nn.ReLU(),
            nn.Linear(60, 1)
        )
 
    def forward(self, x):
        return self.fc(x)


input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
actor = ActorNetwork(input_dim, output_dim)
critic = CriticNetwork(input_dim)
actor_optimizer = optim.RMSprop(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)


"""1. Провести  навчання  агента"""

def train_agent(actor, critic, actor_opt, critic_opt, EPOCHS, EPISODES, GAMMA):
    # Для зберігання метрик
    avg_TD_errors = [] # Avg Gt-v
    max_TD_errors = [] # Max Gt-v
    avg_scores = []
 
    # Тренування
    for epoch in range(EPOCHS):
        rewards_per_epoch = []
        td_errors_epoch = []
 
        for episode in range(EPISODES):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            state_encoded = torch.tensor(state, dtype=torch.float32)
            episode_reward = 0
            step_counter = 0
 
            while True:
                action_probs = actor(state_encoded)
                action = torch.multinomial(action_probs, 1).item()
 
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                next_state_encoded = torch.tensor(next_state, dtype=torch.float32)
 
                reward = torch.tensor(reward, dtype=torch.float32)
 
                td_target = reward + GAMMA * critic(next_state_encoded) * (1 - done) * (1 - truncated)
                td_error = td_target - critic(state_encoded)
 
                # Оновлення критика
                critic_loss = td_error.pow(2).mean()
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
 
                # Оновлення актора
                actor_loss = -torch.log(action_probs[action]) * td_error.detach()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
 
                episode_reward += reward.item()
                td_errors_epoch.append(td_error.item())
                step_counter += 1
 
                if done or step_counter >= 800:
                    rewards_per_epoch.append(episode_reward)
                    break
 
                state = next_state
                state_encoded = next_state_encoded
 
        avg_scores.append(np.mean(rewards_per_epoch))
        avg_TD_errors.append(np.mean(td_errors_epoch))
        max_TD_errors.append(np.max(td_errors_epoch))
 
        # Виведення інформації по кожній епосі
        print(f"Epoch {epoch + 1}/{EPOCHS}: Avg Reward = {avg_scores[-1]:.2f}, Avg Gt-v: {avg_TD_errors[-1]:.2f}, Max Gt-v: {max_TD_errors[-1]:.2f}")
 
    return avg_scores, avg_TD_errors, max_TD_errors
 

avg_scores, avg_Gt_v, max_Gt_v = train_agent(actor, critic, actor_optimizer, critic_optimizer, EPOCHS, EPISODES, GAMMA)

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
plt.title('Max Gt - v(s) per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Max Gt - v(s)')
 
plt.tight_layout()
plt.savefig("1-AC Graphs")
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
 
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action_probs = actor(state_tensor.unsqueeze(0))
                action = torch.argmax(action_probs).item()
 
            next_state, reward, done, truncated, _ = env.step(action)
            if truncated:
                done = True
 
            episode_reward += reward
            step += 1
            state = next_state
 
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
plt.savefig("2-AC Graphs")
plt.show()
