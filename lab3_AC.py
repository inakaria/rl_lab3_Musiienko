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


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 60),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(60, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(60, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        action_probs = self.actor(shared_out)
        state_value = self.critic(shared_out)
        return action_probs, state_value


input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
actor_critic = ActorCriticNetwork(input_dim, output_dim)
optimizer = optim.RMSprop(actor_critic.parameters(), lr=LEARNING_RATE)


"""1. Провести  навчання  агента"""

def train_agent(EPOCHS, EPISODES, GAMMA, LEARNING_RATE):
    # Для зберігання метрик
    avg_scores = []
    critic_losses = []
    actor_losses = []

    # Тренування
    for epoch in range(EPOCHS):
        epoch_rewards = []
        epoch_actor_losses = []
        epoch_critic_losses = []

        for episode in range(EPISODES):
            state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_probs, state_value = actor_critic(state_tensor)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()

                next_state, reward, done, truncated, _ = env.step(action.item())
                if truncated:
                    done = True

                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                _, next_state_value = actor_critic(next_state_tensor)

                target = reward + GAMMA * next_state_value.item() * (1 - int(done))
                td_error = target - state_value.item()

                # Оновлення критика
                critic_loss = td_error ** 2
                epoch_critic_losses.append(critic_loss)

                # Оновлення актора
                actor_loss = -action_dist.log_prob(action) * td_error
                epoch_actor_losses.append(actor_loss.item())

                loss = actor_loss + critic_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                state = next_state
                episode_reward += reward

            epoch_rewards.append(episode_reward)

        avg_scores.append(np.mean(epoch_rewards))
        critic_losses.append(np.mean(epoch_critic_losses))
        actor_losses.append(np.mean(epoch_actor_losses))

        print(f"Epoch {epoch + 1}/{EPOCHS}: Avg Reward = {avg_scores[-1]:.2f}, Critic Loss = {critic_losses[-1]:.4f}, Actor Loss = {actor_losses[-1]:.4f}")

    return avg_scores, critic_losses, actor_losses


avg_scores, critic_losses, actor_losses = train_agent(EPOCHS, EPISODES, GAMMA, LEARNING_RATE)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(avg_scores)
plt.title('Average Reward per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Reward')

plt.subplot(1, 3, 2)
plt.plot(critic_losses)
plt.title('Average Critic Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Critic Loss')

plt.subplot(1, 3, 3)
plt.plot(actor_losses)
plt.title('Average Actor Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Actor Loss')

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
        rewards_per_step = []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_probs, _ = actor_critic(state_tensor)
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
plt.savefig("2-AC Graphs")
plt.show()
