from copy import deepcopy
from dataclasses import dataclass
import random
from typing import List, Optional, Tuple

import pandas as pd

from agents import Agent, Ally, Enemy
from base_logger import logger
from statuses import StatusManager

MAX_PARTY_SIZE = 6
MAX_ENEMIES = 4

NEW_LEVEL_REWARD = 0
TIME_PENALTY_PER_ACTION = -1

GOLD_REWARD_WEIGHT = 1
LEVEL_REWARD_WEIGHT = 0
TIME_PENALTY_WEIGHT = 0


@dataclass
class Item:
    name: str
    type: str  # weapon, armor, item
    move: str
    hp_bonus: int = 0
    mp_bonus: int = 0
    attack_bonus: int = 0
    defense_bonus: int = 0


class State(object):
    def __init__(self, party: List[Ally], difficulty: int):
        self.party = party
        self.enemies = []
        self.turns_left = len(self.party)
        self.difficulty = difficulty
        self.gold = 0
        self.global_step = 0

    def current_player(self) -> Ally:
        for ally in self.party:
            if not ally.has_gone:
                return ally

    def set_enemies(self, enemies: List[Enemy]):
        self.enemies = enemies

    def format_battle_table(self):
        enemy_df = pd.DataFrame.from_dict([enemy.__dict__ for enemy in self.enemies])
        ally_df = pd.DataFrame.from_dict([ally.__dict__ for ally in self.party])
        return f"difficulty={self.difficulty}\n" + \
            f"gold={self.gold}\n" + \
            f"turns_left={self.turns_left}\n" + \
            f"global_step={self.global_step}\n" + \
            f"\nPARTY_STATUS\n{ally_df}\n" + \
            f"\nENEMY_STATUS\n{enemy_df}\n"

    def serialize(self):
        pass


@dataclass
class Action:
    name: str
    n_targets: int # number, or -1 for all
    target_type: str # ally or enemy

    hp_cost: int = 0
    mp_cost: int = 0
    attack_cost: int = 0
    defense_cost: int = 0
    status_self: str = None
    status_self_p: float = 1.

    hp_delta: int = 0
    mp_delta: int = 0
    attack_delta: int = 0
    defense_delta: int = 0
    status_target: str = None
    status_target_p: float = 1.
    effect_duration: int = 0


class EntityBank:

    def __init__(self, item_data, ally_data, enemy_data, action_data):
        self.item_data = item_data
        self.ally_data = ally_data
        self.enemy_data = enemy_data
        self.action_data = action_data

    @classmethod
    def from_file(cls, item_file=None, ally_file=None, enemy_file=None, action_file=None):
        item_data = None
        ally_data = None
        enemy_data = None
        action_data = None

        if item_file:
            item_data = pd.read_csv(item_file, index_col=0)
        if ally_file:
            ally_data = pd.read_csv(ally_file, index_col=0)
        if enemy_file:
            enemy_data = pd.read_csv(enemy_file, index_col=0)
        if action_file:
            action_data = pd.read_csv(action_file, index_col=0)
        return cls(item_data, ally_data, enemy_data, action_data)

    def get_item_id(self, name: str) -> int:
        return self.item_data.index.get_loc(name)

    def get_ally_id(self, name: str) -> int:
        return self.ally_data.index.get_loc(name)

    def get_enemy_id(self, name: str) -> int:
        return self.enemy_data.index.get_loc(name)

    def get_action_id(self, name: str) -> int:
        return self.action_data.index.get_loc(name)

    def create_item(self, name: str) -> Item:
        return Item(name=name, **self.item_data.loc[name].to_dict())

    def create_ally(self, name: str) -> Ally:
        return Ally(name=name, **self.ally_data.loc[name].to_dict())

    def create_enemy(self, name: str) -> Enemy:
        return Enemy(name=name, **self.enemy_data.loc[name].to_dict())

    def get_legal_enemies(self, difficulty: int) -> List[str]:
        difficulty_lb = (self.enemy_data['min_level'] <= difficulty)
        difficulty_ub = (self.enemy_data['max_level'] >= difficulty)
        return self.enemy_data[difficulty_lb & difficulty_ub].index.tolist()

    def create_action(self, name: str, agent: Optional[Agent] = None) -> Action:
        if name == 'basic':
            return Action(
                name=name,
                n_targets=1,
                target_type='ally' if isinstance(agent, Enemy) else 'enemy',
                hp_delta=-agent.attack,
            )
        else:
            return Action(name=name, **self.action_data.loc[name].to_dict())


class TurnBasedRPGEnv(object):
    def __init__(self, party: List[str],
                 item_file=None,
                 ally_file=None,
                 enemy_file=None,
                 action_file=None,
                 starting_difficulty=1,
                 dungeon_repeat_interval=100,
                 print_every=5):
        self.seed()
        self.entity_bank = EntityBank.from_file(
            item_file=item_file,
            ally_file=ally_file,
            enemy_file=enemy_file,
            action_file=action_file,
        )
        self.state = State([
            self.entity_bank.create_ally(ally_class) for ally_class in party],
            difficulty=starting_difficulty,
        )
        self.dungeon_repeat_interval = dungeon_repeat_interval
        self.starting_difficulty = starting_difficulty
        self.original_party = deepcopy(party)
        self.action_step = 0 # for logging only
        self.print_every = print_every
        self.new_level()

    def reset(self):
        self.seed()
        party = [self.entity_bank.create_ally(ally_class) for ally_class in self.original_party]
        self.state = State(party, difficulty=self.starting_difficulty)
        self.action_step = 0
        self.new_level()

    def seed(self, seed: Optional[int] = 42):
        random.seed(seed)

    def new_level(self):
        # determine the number of enemies needed
        n_enemies = 2
        enemies = []
        effective_difficulty = self.state.difficulty % self.dungeon_repeat_interval
        legal_enemy_list = self.entity_bank.get_legal_enemies(effective_difficulty)
        for _ in range(n_enemies):
            enemy_name = random.choice(legal_enemy_list)
            enemies.append(self.entity_bank.create_enemy(enemy_name))
        self.state.set_enemies(enemies)
        # generate base enemies
        # scale stats based on difficulty      

    def step(self, action: Action, targets: List[Agent], dry_run: Optional[bool] = False) -> Tuple[Tuple, int, bool]:
        reward = 0
        is_terminal = False
        if len(self.state.party) == 0:
            return self.state, reward, True
        copy_state = deepcopy(self.state)
        agent = self.state.current_player()
        target_indices = [self.state.enemies.index(target) for target in targets]
        logger.debug(f"PARTY {self.state.party.index(agent)} --({action.name})-> ENEMY {target_indices}")

        self.state, enemy_kill_reward = agent.execute(self.state, action, targets)
        reward += GOLD_REWARD_WEIGHT * enemy_kill_reward

        # if at this point, all of the party members have used up their turns, enemies move
        if self.state.turns_left == 0 and len(self.state.party):
            self.state = self.execute_enemy_turn(self.state)

        # if all enemies are defeated, start the next level
        if len(self.state.enemies) == 0:
            self.state.difficulty += 1
            logger.debug(f"GENERATE NEW LEVEL {self.state.difficulty} (+{NEW_LEVEL_REWARD})")
            reward += LEVEL_REWARD_WEIGHT * NEW_LEVEL_REWARD
            self.new_level()
        else:
            reward += TIME_PENALTY_WEIGHT * TIME_PENALTY_PER_ACTION

        # if everyone is dead, yeah, you lose
        is_terminal = (len(self.state.party) == 0)
        if dry_run:
            self.state = copy_state  # commit state only at end
        result_str = f"STEP {self.action_step}: reward={reward}, is_terminal={is_terminal}\nstate:\n{self.state.format_battle_table()}"
        if self.action_step % self.print_every == 0 or is_terminal:
            if is_terminal:
                logger.info(f"RUN ENDED (gold: {self.state.gold}, difficulty: {self.state.difficulty})")
            logger.info(result_str)
        else:
            logger.debug(result_str)
        self.action_step += 1
        return self.state, reward, is_terminal

    def execute_enemy_turn(self, state):
        for i, enemy in enumerate(state.enemies):
            actions = enemy.get_legal_actions()  # generic or special?
            enemy_action = self.entity_bank.create_action(random.choice(actions), enemy)
            if enemy_action.n_targets > 0:
                target = random.choices(state.party, k=enemy_action.n_targets)
            else:
                target = state.party
            target_indices = [state.party.index(t) for t in target]
            logger.debug(f"ENEMY {i} ({enemy.name}) --({enemy_action.name})-> PARTY {target_indices}")
            state, _ = enemy.execute(state, enemy_action, target)
            if len(state.party) == 0: break  # automatically end turn if all party members died
        for ally in state.party:
            ally.has_gone = False
        state.turns_left = len(state.party)
        return state
