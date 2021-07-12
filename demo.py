from turn_based_env import *

from copy import copy
import random

if __name__ == '__main__':
    env = TurnBasedRPGEnv(['hero'], ally_file="allies.csv", enemy_file="enemies.csv", action_file="actions.csv")
    curr_player = env.state.current_player()

    env.reset()
    is_terminal = False
    while not is_terminal:
        action_names = curr_player.get_legal_actions(env.entity_bank)
        actions = [env.entity_bank.create_action(a, curr_player) for a in action_names]
        action = random.choice(actions)
        targets = random.sample(env.state.enemies, action.n_targets) if action.n_targets > 0 else copy(env.state.enemies)
        *_, is_terminal = env.step(action, targets)
