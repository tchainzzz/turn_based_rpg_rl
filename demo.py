from turn_based_env import *

if __name__ == '__main__':
    env = TurnBasedRPGEnv(['test_ally'], ally_file="allies.csv", enemy_file="enemies.csv")
    curr_player = env.state.current_player()
    action_names = curr_player.get_legal_actions()
    actions = [env.entity_bank.create_action(a, curr_player) for a in action_names]

    env.reset()
    is_terminal = False
    while not is_terminal:
        *_, is_terminal = env.step(actions[0], [env.state.enemies[0]])
