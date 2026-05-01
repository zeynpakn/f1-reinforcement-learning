from environment import Environment

env   = Environment(headless=True)
state = env.reset()

print("State boyutu:", len(state))
print("İlk state:  ", [f"{s:.2f}" for s in state])

state, reward, done, info = env.step(0)
print("Reward:", reward)
print("Done:  ", done)
print("Info:  ", info)