from environment import Environment
import random

env    = Environment(headless=True)
ACTION_COUNT = 5

episode_rewards = []

for episode in range(100):
    state      = env.reset()
    total_rew  = 0
    done       = False

    while not done:
        action              = random.randint(0, ACTION_COUNT - 1)
        state, reward, done, info = env.step(action)
        total_rew          += reward

    episode_rewards.append(total_rew)
    if (episode + 1) % 10 == 0:
        avg = sum(episode_rewards[-10:]) / 10
        print(f"Episode {episode+1:3d} | Son 10 ort. reward: {avg:.1f}")

print(f"\nBaseline özeti:")
print(f"  Ortalama reward : {sum(episode_rewards)/len(episode_rewards):.1f}")
print(f"  En iyi episode  : {max(episode_rewards):.1f}")
print(f"  En kötü episode : {min(episode_rewards):.1f}")