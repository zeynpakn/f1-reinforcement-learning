from environment import Environment
from q_learning_agent import QLearningAgent

env   = Environment(headless=True)
agent = QLearningAgent()

state = env.reset()

for _ in range(10):
    action                    = agent.choose_action(state)
    next_state, reward, done, info = env.step(action)
    agent.update(state, action, reward, next_state, done)
    state = next_state
    print(f"Aksiyon: {action} | Reward: {reward:6.1f} | Epsilon: {agent.epsilon:.3f} | Q-table boyutu: {len(agent.q_table)}")
    if done:
        state = env.reset()

agent.decay_epsilon()
print(f"\nEpsilon decay sonrası: {agent.epsilon:.4f}")
agent.save()