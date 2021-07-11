from turn_based_env import *

import random

if __name__ == '__main__':
    env = TurnBasedRPGEnv(['hero'], ally_file="allies.csv", enemy_file="enemies.csv", action_file="actions.csv")
    curr_player = env.state.current_player()
    action_names = curr_player.get_legal_actions()
    actions = [env.entity_bank.create_action(a, curr_player) for a in action_names]

    env.reset()
    is_terminal = False
    while not is_terminal:
        action = random.choice(actions)
        *_, is_terminal = env.step(action, random.choices(env.state.enemies, k=action.n_targets))
