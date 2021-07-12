from turn_based_env import *

from copy import copy
import random

if __name__ == '__main__':
    equipment = [
        ('big_sword', None, None),
        (None, None, None)
        ]
    env = TurnBasedRPGEnv(['hero', 'test_ally'], equipment, ally_file="allies.csv", enemy_file="enemies.csv", item_file="items.csv", action_file="actions.csv")
    env.reset()
    is_terminal = False
    while not is_terminal:
        curr_player = env.state.current_player()
        action_names = curr_player.get_legal_actions(env.entity_bank)
        actions = [env.entity_bank.create_action(a, curr_player) for a in action_names]
        action = random.choice(actions)
        targets = random.sample(env.state.enemies, action.n_targets) if action.n_targets > 0 else copy(env.state.enemies)
        *_, is_terminal = env.step(action, targets)
