import sys
import enum
import numpy as np

from typing import Optional, NamedTuple
import os

def save_world_model(world_model: np.ndarray, fname: str = "logs/testmap.txt") -> None:
    """
    Saves the current world model into a text file so that it looks similar
    to the task description:

        0   : empty cell
        -1  : wall
        1   : start
        3   : not (yet) visible
        91  : oil
        92  : sand
        100 : goal

    The file will contain one row per line, with space separated integers.
    The directory 'logs' is created if it does not exist.
    """
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w", encoding="utf-8") as f:
        for row in world_model:
            f.write(" ".join(str(int(v)) for v in row))
            f.write("\n")


class CellType(enum.Enum):
    NOT_VISIBLE = 3
    WALL = -1

class Player(NamedTuple):
    x: int
    y: int
    vel_x: int
    vel_y: int

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vel_x, self.vel_y])

class Circuit(NamedTuple):
    track_shape: tuple[int, int]
    num_players: int
    visibility_radius: int

class State(NamedTuple):
    circuit: Circuit
    visible_track: np.ndarray
    players: list[Player]
    agent: Player

def read_initial_observation() -> Circuit:
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)

def read_observation(old_state: State) -> Optional[State]:
    line = input()
    if line == '~~~END~~~':
        return None
    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)
    players = []
    # this won't change
    circuit_data = old_state.circuit
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, input().split())
        # Calculating the velocity from the old state is left as an exercise to
        # the reader.
        players.append(Player(pposx, pposy, 0, 0))
    visible_track = np.full(circuit_data.track_shape, CellType.WALL.value)
    for i in range(2 * circuit_data.visibility_radius + 1):
        line = [int(a) for a in input().split()]
        x = posx - circuit_data.visibility_radius + i
        if x < 0 or x >= circuit_data.track_shape[0]:
            continue
        y_start = posy - circuit_data.visibility_radius
        y_end = y_start + 2 * circuit_data.visibility_radius + 1
        if y_start < 0:
            line = line[-y_start:]
            y_start = 0
        if y_end > circuit_data.track_shape[1]:
            line = line[:-(y_end - circuit_data.track_shape[1])]
            y_end = circuit_data.track_shape[1]
        visible_track[x, y_start:y_end] = line
    # We can't risk it. Put walls everywhere we can't see. Just to be safe.
    visible_track[visible_track == CellType.NOT_VISIBLE.value] = (
        CellType.WALL.value)
    return old_state._replace(
        visible_track=visible_track, players=players, agent=agent)



def calculate_move( state: State) -> tuple[int, int]:
    save_world_model(state.visible_track)

    
    
    return (1, 0)

def main():
    print('READY', flush=True)
    circuit = read_initial_observation()
    state: Optional[State] = State(circuit, None, [], None)  # type: ignore
    
    while True:
        assert state is not None
        state = read_observation(state)
        if state is None:
            return
        delta = calculate_move(state)
        print(f'{delta[0]} {delta[1]}', flush=True)

if __name__ == "__main__":
    main()
