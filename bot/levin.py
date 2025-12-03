import sys
import os
import enum
import numpy as np

from typing import Optional, NamedTuple


class CellType(enum.Enum):
    """
    Enum for the different cell types on the track.
    These values come from the task description / environment:
        0:  empty cell
       -1:  wall cell (everything outside the map is also wall)
        1:  start cell
        3:  cell is not visible (fog of war)
       91:  oil cell
       92:  sand cell
      100: goal cell
    """
    EMPTY = 0
    WALL = -1
    START = 1
    UNKNOWN = 2
    NOT_VISIBLE = 3
    OIL = 91
    SAND = 92
    GOAL = 100


class Player(NamedTuple):
    """
    Represents a player on the track.

    x, y          : current grid position
    vel_x, vel_y  : current velocity components
    """
    x: int
    y: int
    vel_x: int
    vel_y: int

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=int)

    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vel_x, self.vel_y], dtype=int)


class Circuit(NamedTuple):
    """
    Static information about the circuit:

    track_shape       : (height, width) of the whole map
    num_players       : number of players in this race
    visibility_radius : radius of the local square the agent can see
    """
    track_shape: tuple[int, int]
    num_players: int
    visibility_radius: int


class State(NamedTuple):
    """
    Full internal state of our agent.

    circuit       : static circuit info
    visible_track : "safe" map used for planning (unknown treated as wall)
    players       : list of all players' (last known) positions
    agent         : our own Player object
    world_model   : persistent world model built from all observations so far
                    (unknown cells are marked as CellType.NOT_VISIBLE)
    """
    circuit: Circuit
    visible_track: Optional[np.ndarray]
    players: list[Player]
    agent: Optional[Player]
    world_model: np.ndarray


def read_initial_observation() -> Circuit:
    """
    Reads the very first line from stdin that contains:
        H W num_players visibility_radius
    and returns the corresponding Circuit object.
    """
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)


def _update_world_model(
    world_model: np.ndarray,
    posx: int,
    posy: int,
    visibility_radius: int,
    track_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function that:
      * reads the local visibility square from stdin,
      * updates the persistent world_model with everything that is actually
        visible (cells with value != CellType.NOT_VISIBLE),
      * builds a "planning" map (visible_track) where every unknown cell
        is treated as a wall for safety.

    Returns (visible_track, updated_world_model).
    """
    height, width = track_shape
    # start with every unseen cell considered as a wall for planning
    visible_track = np.full(track_shape, CellType.WALL.value, dtype=int)

    for i in range(2 * visibility_radius + 1):
        # One line of the local map centred on the agent
        row_vals = [int(a) for a in input().split()]
        x = posx - visibility_radius + i
        if x < 0 or x >= height:
            # This row lies completely outside the global track
            continue

        y_start = posy - visibility_radius
        y_end = y_start + 2 * visibility_radius + 1

        # Cut the row so that it fits into [0, width)
        if y_start < 0:
            row_vals = row_vals[-y_start:]
            y_start = 0
        if y_end > width:
            row_vals = row_vals[:-(y_end - width)]
            y_end = width

        # Now row_vals has exactly (y_end - y_start) entries
        for offset, cell_val in enumerate(row_vals):
            y = y_start + offset

            # Update persistent world model if the cell is actually visible.
            # NOT_VISIBLE just means "not seen now", it should not overwrite
            # older knowledge.
            if cell_val != CellType.NOT_VISIBLE.value:
                world_model[x, y] = cell_val

            # For planning we only trust cells that we truly see; everything
            # else remains a WALL in visible_track.
            if cell_val != CellType.NOT_VISIBLE.value:
                visible_track[x, y] = cell_val

    return visible_track, world_model


def read_observation(old_state: State) -> Optional[State]:
    """
    Reads the observation for the current turn from stdin and returns
    a new State.

    The input format (per turn) is:
      * one line with:  posx posy velx vely  (our agent)
      * num_players lines with: pposx pposy   (other players)
      * (2 * visibility_radius + 1) lines with the local map

    If the special line "~~~END~~~" is received, the game ended and
    the function returns None.
    """
    line = input()
    if line == '~~~END~~~':
        return None

    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)
    players: list[Player] = []

    circuit_data = old_state.circuit

    # Read the positions of all players (velocity of the others is unknown
    # for us, so we keep vel=(0,0) as a placeholder).
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, input().split())
        players.append(Player(pposx, pposy, 0, 0))

    # Build new planning map and update persistent world model
    visible_track, world_model = _update_world_model(
        old_state.world_model.copy(),
        posx,
        posy,
        circuit_data.visibility_radius,
        circuit_data.track_shape,
    )

    # After updating the world model, immediately dump it to a file so we
    # can inspect what the agent currently knows about the world.
    save_world_model(world_model)

    return old_state._replace(
        visible_track=visible_track,
        players=players,
        agent=agent,
        world_model=world_model,
    )


def save_world_model(world_model: np.ndarray, fname: str = "logs/agentmap.txt") -> None:
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


def traversable(cell_value: int) -> bool:
    """
    Returns True if the given cell value is considered traversable
    for path-checking purposes.
    """
    return cell_value >= 0


def valid_line(state: State, pos1: np.ndarray, pos2: np.ndarray) -> bool:
    """
    Checks whether the straight line from pos1 to pos2 is free from walls
    on the current planning map (state.visible_track).

    This is a reimplementation of the judge's line-of-sight logic, using
    our own (possibly incomplete) knowledge of the world.
    """
    assert state.visible_track is not None, "visible_track not initialised yet."
    track = state.visible_track
    if (np.any(pos1 < 0) or np.any(pos2 < 0) or np.any(pos1 >= track.shape)
            or np.any(pos2 >= track.shape)):
        return False
    diff = pos2 - pos1
    # Go through the straight line connecting ``pos1`` and ``pos2``
    # cell-by-cell. Wall is blocking if either it is straight in the way or
    # there are two wall cells above/below each other and the line would go
    # "through" them.
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))  # direction: left or right
        for i in range(abs(diff[0]) + 1):
            x = int(pos1[0] + i * d)
            y = pos1[1] + i * slope * d
            y_ceil = int(np.ceil(y))
            y_floor = int(np.floor(y))
            if (not traversable(track[x, y_ceil])
                    and not traversable(track[x, y_floor])):
                return False
    # Do the same, but examine two-cell-wall configurations when they are
    # side-by-side (east-west).
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))  # direction: up or down
        for i in range(abs(diff[1]) + 1):
            x = pos1[0] + i * slope * d
            y = int(pos1[1] + i * d)
            x_ceil = int(np.ceil(x))
            x_floor = int(np.floor(x))
            if (not traversable(track[x_ceil, y])
                    and not traversable(track[x_floor, y])):
                return False
    return True


def calculate_move(rng: np.random.Generator, state: State) -> tuple[int, int]:
    """
    Main decision function of the agent.

    New behaviour:
      * the agent builds a persistent world model from all previous turns,
      * among all valid accelerations in {-1,0,1}^2 it chooses the one
        whose resulting position is closest to any known wall cell,
        effectively "following" the nearest wall.

    The function returns (dx, dy) which is the acceleration vector the
    environment expects.
    """
    assert state.agent is not None, "Agent state not initialised."
    self_pos = state.agent.pos
    self_vel = state.agent.vel

    # Precompute locations of all known wall cells in the world model
    wall_cells = np.argwhere(state.world_model == CellType.WALL.value)

    def distance_to_wall(pos: np.ndarray) -> float:
        """
        Returns the Euclidean distance from `pos` to the closest known wall.
        If we don't know any wall yet, returns a very large number so that
        every move is equally "bad" in this sense.
        """
        if wall_cells.size == 0:
            return 1e9
        # wall_cells has shape (N, 2); broadcasting pos over the first axis
        diffs = wall_cells - pos
        dists = np.linalg.norm(diffs, axis=1)
        return float(dists.min())

    def valid_move(next_pos: np.ndarray) -> bool:
        """
        A move is valid if the straight line is free and we don't collide with
        other players (based on their last known positions).
        """
        if not valid_line(state, self_pos, next_pos):
            return False
        # staying in place is always allowed from the collision point of view
        if np.all(next_pos == self_pos):
            return True
        # avoid stepping onto another player's cell
        for p in state.players:
            if np.all(next_pos == p.pos):
                return False
        return True

    # Desired centre if we keep the current velocity
    new_center = self_pos + self_vel

    best_moves: list[tuple[int, int]] = []
    best_dist = float("inf")

    # We consider every possible acceleration in {-1, 0, 1}^2
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            # Target position after applying this acceleration
            target = new_center + np.array([dx, dy], dtype=int)

            if not valid_move(target):
                continue

            dist = distance_to_wall(target)

            # We want to be as close as possible to some wall (but not inside it)
            if dist < best_dist - 1e-6:
                best_dist = dist
                best_moves = [(dx, dy)]
            elif abs(dist - best_dist) <= 1e-6:
                best_moves.append((dx, dy))

    if best_moves:
        # Break ties randomly so behaviour is not completely deterministic
        return tuple(rng.choice(best_moves))  # type: ignore[misc]

    # Fallback: if for some reason no move is valid, stay in place and complain
    print(
        "Not blind, just being brave! (No valid wall-following action found.)",
        file=sys.stderr,
    )
    return (0, 0)


def main():
    """
    Entry point of the agent program.

    It:
      * prints 'READY' so that the judge knows we have initialised,
      * reads the static circuit information,
      * initialises an empty world model,
      * then repeatedly:
          - reads observations,
          - updates the world model,
          - chooses an action using calculate_move,
          - prints the chosen (dx, dy) to stdout.
    """
    print("READY", flush=True)
    circuit = read_initial_observation()
    # Initialise our persistent world model with "not visible" everywhere
    world_model = np.full(
        circuit.track_shape, CellType.NOT_VISIBLE.value, dtype=int
    )
    state: Optional[State] = State(circuit, None, [], None, world_model)
    rng = np.random.default_rng(seed=1)

    while True:
        assert state is not None
        state = read_observation(state)
        if state is None:
            # Game over
            return
        delta = calculate_move(rng, state)
        print(f"{delta[0]} {delta[1]}", flush=True)


if __name__ == "__main__":
    main()
