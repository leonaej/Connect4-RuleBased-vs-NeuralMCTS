import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from connect4_env.connect4_env import Connect4Env
from agents.random_agent import RandomAgent
import time

env=Connect4Env()
agent1=RandomAgent()
agent2=RandomAgent()

state=env.reset()

done=False


while not done:
    if env.current_player==1:
        action=agent1.select_action(env.get_valid_actions())
    else:
        action=agent2.select_action(env.get_valid_actions())    

    state, reward, done =env.step(action)

    print(state)
    time.sleep(0.3)

print("Game Over")

