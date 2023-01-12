from typing import List
from luxai_s2.unit import Unit
from luxai_s2.config import EnvConfig
def is_day(config: EnvConfig, env_step):
    return env_step % config.CYCLE_LENGTH < config.DAY_LENGTH


def get_top_two_power_units(units: List[Unit]):
    most_power_unit: Unit = units[0]
    most_power = -1
    next_most_power_unit: Unit = units[1]
    next_most_power = -1
    for u in units:
        if (u.power > most_power):
            next_most_power_unit = most_power_unit
            most_power_unit = u
            most_power = u.power
        elif (u.power >= next_most_power): # >= check since we want to top 2 power units which can tie
            next_most_power_unit = u
            next_most_power = u.power

    return (most_power_unit, next_most_power_unit)