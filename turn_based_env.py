from copy import copy, deepcopy
from dataclasses import dataclass
import random
from typing import List, Optional, Tuple

import pandas as pd

from agents import Agent, Ally, Enemy, FALLBACK_ATTACK_NAME
from base_logger import logger
from statuses import StatusManager

MAX_PARTY_SIZE = 6
MAX_ENEMIES = 4

NEW_LEVEL_REWARD = 0
TIME_PENALTY_PER_ACTION = -1

GOLD_REWARD_WEIGHT = 1
LEVEL_REWARD_WEIGHT = 0
TIME_PENALTY_WEIGHT = 0

N_ENEMIES_MAP = [1, 1, 1, 2, 1, 2, 2, 3, 4, 1]


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
    effect_scaling: str # proportional or absolute

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

    def pretty_repr(self):
        return "\n".join(["=".join([field_name, str(value)]) for field_name, value in self.__dict__.items()])


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

    def create_ally(self, name: str, position: int) -> Ally:
        return Ally(name=name, position=position, **self.ally_data.loc[name].to_dict())

    def create_enemy(self, name: str, position: int) -> Enemy:
        return Enemy(name=name, position=position, **self.enemy_data.loc[name].to_dict())

    def get_legal_enemies(self, difficulty: int) -> List[str]:
        difficulty_lb = (self.enemy_data['min_level'] <= difficulty)
        difficulty_ub = (self.enemy_data['max_level'] >= difficulty)
        return self.enemy_data[difficulty_lb & difficulty_ub].index.tolist()

    def create_action(self, name: str, agent: Optional[Agent] = None) -> Action:
        if name == FALLBACK_ATTACK_NAME:
            return Action(
                name=name,
                effect_scaling='absolute',
                n_targets=1,
                target_type='opponent',
                hp_delta=-agent.attack,
            )
        else:
            return Action(name=name, **self.action_data.loc[name].to_dict())


class TurnBasedRPGEnv(object):
    def __init__(self, party: List[str],
                 equipment,
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

        # change on reset
        party = [self.entity_bank.create_ally(ally_class, position=i) \
                    for i, ally_class \
                    in enumerate(party)]
        party = self.equip_party(party, equipment)
        self.state = State(party, difficulty=starting_difficulty)
        self.original_state = deepcopy(self.state)

        # never change
        self.dungeon_repeat_interval = dungeon_repeat_interval
        self.starting_difficulty = starting_difficulty
        self.action_step = 0 # for logging only
        self.print_every = print_every
        self.new_level()

    def equip_party(self, party, equipment):
        for ally, item_names in zip(party, equipment):
            for item in item_names:
                if item is not None:
                    item_obj = self.entity_bank.create_item(item)
                    logger.debug(f"PARTY {ally.id} EQUIPS {item_obj.name}")
                    ally.equip(item_obj)
        return party

    def reset(self):
        self.seed()
        self.state = self.original_state
        self.action_step = 0
        self.new_level()
        print(f"INITIAL STATE\n{self.state.format_battle_table()}")

    def seed(self, seed: Optional[int] = 42):
        random.seed(seed)

    def new_level(self):
        # determine the number of enemies needed
        n_enemies = N_ENEMIES_MAP[self.state.difficulty % len(N_ENEMIES_MAP)]
        enemies = []
        effective_difficulty = self.state.difficulty % self.dungeon_repeat_interval
        legal_enemy_list = self.entity_bank.get_legal_enemies(effective_difficulty)
        for _ in range(n_enemies):
            enemy_name = random.choice(legal_enemy_list)
            enemy = self.entity_bank.create_enemy(enemy_name, position=len(enemies))
            # TODO: scale stats based on difficulty      
            enemies.append(enemy)
        self.state.set_enemies(enemies)
        enemy_list = [e.id for e in enemies]
        for ally in self.state.party:
            ally.has_gone = False
        logger.info(f"NEW LEVEL {self.state.difficulty}: {enemy_list}")

    def step(self, action: Action, targets: List[Agent], dry_run: Optional[bool] = False) -> Tuple[Tuple, int, bool]:
        reward = 0
        is_terminal = False
        if len(self.state.party) == 0:
            return self.state, reward, True
        copy_state = deepcopy(self.state)
        agent = self.state.current_player()

        self.state, enemy_kill_reward = agent.execute(self.state, action, targets)
        reward += GOLD_REWARD_WEIGHT * enemy_kill_reward

        # if at this point, all of the party members have used up their turns, enemies move
        if self.state.turns_left == 0 and len(self.state.party):
            self.state = self.execute_enemy_turn(self.state)

        # if all enemies are defeated, start the next level
        if len(self.state.enemies) == 0:
            self.state.difficulty += 1
            logger.debug(f"LEVEL {self.state.difficulty} COMPLETE (+{NEW_LEVEL_REWARD})")
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
            actions = enemy.get_legal_actions(self.entity_bank)  # generic or special?
            enemy_action = self.entity_bank.create_action(random.choice(actions), enemy)
            target_list = state.party if enemy_action.target_type == 'opponent' else state.enemies
            if enemy_action.n_targets > 0:
                target = random.sample(target_list, enemy_action.n_targets)
            else:
                target = copy(target_list)
            target_ids = [t.id for t in target]
            logger.debug(f"ENEMY {i} ({enemy.name}) --({enemy_action.name})-> PARTY {target_ids}")
            state, _ = enemy.execute(state, enemy_action, target)
            if len(state.party) == 0: break  # automatically end turn if all party members died
        for ally in state.party:
            ally.has_gone = False
        state.turns_left = len(state.party)
        return state
