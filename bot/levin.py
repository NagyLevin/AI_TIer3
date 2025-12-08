import sys
import os
import enum
import heapq
import numpy as np

from typing import Optional, NamedTuple


class CellType(enum.Enum):
    """
    Enum for the different cell types on the track.
    These values come from the task description / environment:
        0:  empty cell
       -1:  wall cell (everything outside the map is also wall)
        1:  start cell
        2:  unknown (currently unused, reserved)
        3:  cell is not visible (fog of war, used in world_model)
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
        """
        Returns the current position as a NumPy integer array [x, y].
        """
        return np.array([self.x, self.y], dtype=int)

    @property
    def vel(self) -> np.ndarray:
        """
        Returns the current velocity as a NumPy integer array [vel_x, vel_y].
        """
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
    visible_track : map used for planning (most up-to-date world_model)
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


# ---------------------------------------------------------------------- #
# Globals: last position/velocity – to penalise turning back
# ---------------------------------------------------------------------- #
LAST_POS: Optional[np.ndarray] = None
LAST_VEL: Optional[np.ndarray] = None


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
      * builds a "planning" map (visible_track) as a copy of the updated
        world_model.

    FONTOS: visible_track most már a world_model másolata, az ismeretlen
    cellákat nem tekintjük automatikusan falnak (csak WALL blokkol).
    """
    height, width = track_shape

    # Start from the previous world model for visible_track;
    # we will update both when we actually see cells.
    visible_track = world_model.copy()

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
            if cell_val != CellType.NOT_VISIBLE.value:
                world_model[x, y] = cell_val
                visible_track[x, y] = cell_val

    return visible_track, world_model


def read_observation(old_state: State) -> Optional[State]:
    """
    Reads the observation for the current turn from stdin and returns
    a new State.

    Input per turn:
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

    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, input().split())
        players.append(Player(pposx, pposy, 0, 0))

    visible_track, world_model = _update_world_model(
        old_state.world_model.copy(),
        posx,
        posy,
        circuit_data.visibility_radius,
        circuit_data.track_shape,
    )

    save_world_model(world_model)

    return old_state._replace(
        visible_track=visible_track,
        players=players,
        agent=agent,
        world_model=world_model,
    )


def save_world_model(world_model: np.ndarray, fname: str = "logs/agentmap.txt") -> None:
    """
    Saves the current world model into a text file (row of integers per line).
    """
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w", encoding="utf-8") as f:
        for row in world_model:
            f.write(" ".join(str(int(v)) for v in row))
            f.write("\n")


def traversable(cell_value: int) -> bool:
    """
    Returns True if the given cell value is considered traversable
    for path-checking purposes in the local planning map.

    Csak a WALL (és a pályán kívüli rész) blokkol, az ismeretlen / homok /
    olaj átjárható, csak nem kívánatos.
    """
    return cell_value != CellType.WALL.value


def valid_line(state: State, pos1: np.ndarray, pos2: np.ndarray) -> bool:
    """
    Checks whether the straight line from pos1 to pos2 is free from walls
    on the current planning map (state.visible_track).

    Uses traversable(), így a homok/olaj/unknown átjárható,
    de a fal blokkol.
    """
    assert state.visible_track is not None, "visible_track not initialised yet."
    track = state.visible_track
    if (np.any(pos1 < 0) or np.any(pos2 < 0) or
            np.any(pos1 >= track.shape) or np.any(pos2 >= track.shape)):
        return False
    diff = pos2 - pos1
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))  # direction in x
        for i in range(abs(diff[0]) + 1):
            x = int(pos1[0] + i * d)
            y = pos1[1] + i * slope * d
            y_ceil = int(np.ceil(y))
            y_floor = int(np.floor(y))
            if (not traversable(track[x, y_ceil])
                    and not traversable(track[x, y_floor])):
                return False
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))  # direction in y
        for i in range(abs(diff[1]) + 1):
            x = pos1[0] + i * slope * d
            y = int(pos1[1] + i * d)
            x_ceil = int(np.ceil(x))
            x_floor = int(np.floor(x))
            if (not traversable(track[x_ceil, y])
                    and not traversable(track[x_floor, y])):
                return False
    return True


# ---------------------------------------------------------------------- #
# A* PATH PLANNING HELPERS
# ---------------------------------------------------------------------- #

def build_traversable_mask(world_model: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask indicating which cells are traversable globally.

    Globális tervezésben:
      - WALL és NOT_VISIBLE = blokkolt,
      - többi (EMPTY/START/OIL/SAND/GOAL) = járható, de OIL/SAND drágább lesz.
    """
    traversable_mask = np.ones_like(world_model, dtype=bool)
    blocked = (
        (world_model == CellType.WALL.value) |
        (world_model == CellType.NOT_VISIBLE.value)
    )
    traversable_mask[blocked] = False
    return traversable_mask


def collect_goals(world_model: np.ndarray,
                  traversable_mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Goal cellák gyűjtése globális A*-hoz.

    1) Ha van GOAL, azokat használjuk.
    2) Különben frontier cellák (ismert, járható cella, ami UNKNOWN/NOT_VISIBLE
       szomszéd mellett van) – így az ismeretlen határára megyünk.
    """
    h, w = world_model.shape

    goal_positions = np.argwhere(world_model == CellType.GOAL.value)
    if goal_positions.size > 0:
        return [tuple(p) for p in goal_positions]

    frontier: list[tuple[int, int]] = []
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    for x in range(h):
        for y in range(w):
            if not traversable_mask[x, y]:
                continue
            cell = world_model[x, y]
            if cell in (CellType.EMPTY.value,
                        CellType.START.value,
                        CellType.OIL.value,
                        CellType.SAND.value):
                for dx, dy in directions:
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or ny < 0 or nx >= h or ny >= w:
                        continue
                    ncell = world_model[nx, ny]
                    if ncell in (CellType.NOT_VISIBLE.value,
                                 CellType.UNKNOWN.value):
                        frontier.append((x, y))
                        break

    return frontier


def astar_any_goal(
    traversable_mask: np.ndarray,
    world_model: np.ndarray,
    start: tuple[int, int],
    goals: list[tuple[int, int]],
) -> Optional[list[tuple[int, int]]]:
    """
    A* path planner on a grid with 8-connected neighbourhood.

    Új: terepköltségek
      - EMPTY/START/GOAL: 1x
      - OIL/SAND: 5x (nagy büntetés, de nem lehetetlen)
    """
    if not goals:
        return None

    h, w = traversable_mask.shape
    if (start[0] < 0 or start[1] < 0 or
            start[0] >= h or start[1] >= w):
        return None
    if not traversable_mask[start]:
        return None

    goal_set = set(goals)
    goals_arr = np.array(goals, dtype=float)

    def heuristic(cell: tuple[int, int]) -> float:
        cx, cy = cell
        diff = goals_arr - np.array([cx, cy], dtype=float)
        dists = np.sqrt((diff[:, 0] ** 2) + (diff[:, 1] ** 2))
        return float(dists.min())

    def terrain_factor(x: int, y: int) -> float:
        """
        Terepköltség szorzó a (x,y) cellába lépéshez.
        """
        v = world_model[x, y]
        if v in (CellType.OIL.value, CellType.SAND.value):
            return 5.0
        # minden más járható típus (EMPTY, START, GOAL, stb.)
        return 1.0

    neighbours = [
        (-1, 0, 1.0), (1, 0, 1.0),
        (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
        (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))
    ]

    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start), start))

    g_score: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        f, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        if current in goal_set:
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        for dx, dy, base_cost in neighbours:
            nx = cx + dx
            ny = cy + dy
            if nx < 0 or ny < 0 or nx >= h or ny >= w:
                continue
            if not traversable_mask[nx, ny]:
                continue

            move_cost = base_cost * terrain_factor(nx, ny)
            neighbour = (nx, ny)
            tentative_g = g_score[current] + move_cost

            old_g = g_score.get(neighbour, float("inf"))
            if tentative_g < old_g:
                g_score[neighbour] = tentative_g
                parent[neighbour] = current
                f_score = tentative_g + heuristic(neighbour)
                heapq.heappush(open_heap, (f_score, neighbour))

    return None


def plan_global_path(state: State) -> Optional[list[tuple[int, int]]]:
    """
    Builds a global A* path from the agent to:
      - the GOAL (if known), or
      - the nearest frontier cell (boundary of unknown).

    Plusz:
      - saját cellánkat kivesszük a goals-ból (ha van más is), hogy ne
        álljunk be sarokba célként.
      - OIL/SAND magas útköltségű, így az A* is kerüli.
    """
    assert state.agent is not None
    world_model = state.world_model.copy()
    traversable_mask = build_traversable_mask(world_model)

    h, w = world_model.shape
    for p in state.players:
        if 0 <= p.x < h and 0 <= p.y < w:
            traversable_mask[p.x, p.y] = False

    start = tuple(state.agent.pos.tolist())
    goals = collect_goals(world_model, traversable_mask)
    if not goals:
        return None

    if len(goals) > 1:
        goals = [g for g in goals if g != start]
        if not goals:
            return None

    return astar_any_goal(traversable_mask, world_model, start, goals)


# ---------------------------------------------------------------------- #
# MAIN DECISION FUNCTION
# ---------------------------------------------------------------------- #

def calculate_move(rng: np.random.Generator, state: State) -> tuple[int, int]:
    """
    Main decision function.

    - Globális A* a world_model alapján (cél: GOAL vagy frontier).
    - A path mentén kiválasztunk egy előre néző waypointot (lookahead).
    - Minden lehetséges accel (dx, dy) in {-1,0,1}^2 közül választunk:
        * new_vel = self_vel + accel
        * new_vel NEM lehet (0,0)  <-- sosem állunk meg
        * sebességkorlát (VEL_MAX),
        * valid_line + collision check,
        * extra büntetés, ha OIL/SAND mezőre lépünk,
        * kétlépcsős: ha van olyan lépés, ami nem megy visszafelé,
          csak ezek közül választunk.
    """
    assert state.agent is not None, "Agent state not initialised."
    assert state.visible_track is not None, "visible_track not initialised."

    global LAST_POS, LAST_VEL

    self_pos = state.agent.pos
    self_vel = state.agent.vel

    VEL_MAX = 4.0

    path = plan_global_path(state)

    # Choose lookahead waypoint
    if path is not None and len(path) >= 2:
        lookahead_index = min(len(path) - 1, 4)
        target_cell = np.array(path[lookahead_index], dtype=float)
    else:
        target_cell = None

    def valid_move(next_pos: np.ndarray, new_vel: np.ndarray) -> bool:
        """
        A move is valid if:
          * speed under VEL_MAX,
          * line-of-sight free of walls,
          * no collision with other players,
          * nem maradunk ugyanazon a cellán.
        """
        if np.linalg.norm(new_vel, ord=2) > VEL_MAX:
            return False

        if not valid_line(state, self_pos, next_pos):
            return False

        if np.all(next_pos == self_pos):
            return False

        for p in state.players:
            if np.all(next_pos == p.pos):
                return False
        return True

    # Forward direction
    if target_cell is not None:
        forward_vec = target_cell - self_pos.astype(float)
        if np.linalg.norm(forward_vec, ord=2) > 1e-6:
            forward_unit = forward_vec / np.linalg.norm(forward_vec, ord=2)
        else:
            forward_unit = np.zeros(2, dtype=float)
    else:
        if np.linalg.norm(self_vel, ord=2) > 1e-6:
            forward_unit = self_vel.astype(float) / np.linalg.norm(
                self_vel, ord=2
            )
        else:
            forward_unit = np.array([0.0, 1.0])

    # Összes jelölt lépést eltároljuk, aztán kétlépcsős szűrés
    candidates: list[dict] = []

    h, w = state.world_model.shape

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            accel = np.array([dx, dy], dtype=int)
            new_vel = self_vel + accel

            # Nem állunk meg: tiltjuk a zéró sebességet
            if np.all(new_vel == 0):
                continue

            next_pos = self_pos + new_vel

            if not valid_move(next_pos, new_vel):
                continue

            # Távolság a path-waypointhoz
            if target_cell is not None:
                dist_to_target = float(
                    np.linalg.norm(target_cell - next_pos.astype(float), ord=2)
                )
            else:
                dist_to_target = 0.0

            speed = float(np.linalg.norm(new_vel, ord=2))

            # Irány-illeszkedés
            if np.linalg.norm(new_vel, ord=2) > 1e-6:
                vel_unit = new_vel.astype(float) / np.linalg.norm(
                    new_vel, ord=2
                )
                forward_align = float(np.dot(vel_unit, forward_unit))
            else:
                forward_align = 0.0

            # Visszafordulás büntetése
            backward_penalty = 0.0
            if LAST_POS is not None:
                move_from_last = self_pos.astype(float) - LAST_POS.astype(float)
                if (np.linalg.norm(move_from_last, ord=2) > 1e-6 and
                        np.linalg.norm(new_vel, ord=2) > 1e-6):
                    dir_from_last = move_from_last / np.linalg.norm(
                        move_from_last, ord=2
                    )
                    vel_unit2 = new_vel.astype(float) / np.linalg.norm(
                        new_vel, ord=2
                    )
                    cos_angle = float(np.dot(vel_unit2, dir_from_last))
                    if cos_angle < 0:
                        backward_penalty = -cos_angle  # 0..2 kb

            # Terep büntetés (lokálisan is kerüljük az olajat/homokot)
            terrain_penalty = 0.0
            x, y = int(next_pos[0]), int(next_pos[1])
            if 0 <= x < h and 0 <= y < w:
                cell = state.world_model[x, y]
                if cell in (CellType.OIL.value, CellType.SAND.value):
                    terrain_penalty = 3.0  # elég nagy, de nem végtelen

            # Alap score – még NEM vesszük figyelembe, hogy hátrafelé van-e,
            # csak eltároljuk (később kétlépcsősen szűrünk).
            score = (
                dist_to_target
                + 0.2 * speed
                - 0.5 * forward_align
                + 0.8 * backward_penalty
                + terrain_penalty
            )

            candidates.append(
                {
                    "dx": dx,
                    "dy": dy,
                    "score": score,
                    "back": backward_penalty,
                }
            )

    if not candidates:
        print(
            "No valid move except maybe standing still – trapped?",
            file=sys.stderr,
        )
        return (0, 0)

    # Első lépés: próbáljunk olyan lépést választani, ami NEM megy hátra.
    non_back = [c for c in candidates if c["back"] < 1e-6]
    if non_back:
        pool = non_back
    else:
        # Ha MINDEN lépés hátra megy, akkor kénytelenek vagyunk valamelyiket
        # választani – zsákutcából vissza kell fordulni.
        pool = candidates

    # Második lépés: a kiválasztott pool-ból minimális score alapján döntünk.
    best_score = min(c["score"] for c in pool)
    best_moves = [
        (c["dx"], c["dy"])
        for c in pool
        if abs(c["score"] - best_score) <= 1e-6
    ]

    chosen = tuple(rng.choice(best_moves))  # type: ignore[misc]
    return chosen


def main():
    """
    Entry point of the agent program.
    """
    global LAST_POS, LAST_VEL

    print("READY", flush=True)
    circuit = read_initial_observation()
    world_model = np.full(
        circuit.track_shape, CellType.NOT_VISIBLE.value, dtype=int
    )
    state: Optional[State] = State(circuit, None, [], None, world_model)
    rng = np.random.default_rng(seed=1)

    prev_state: Optional[State] = None

    while True:
        assert state is not None
        prev_state = state
        state = read_observation(state)
        if state is None:
            return

        if prev_state is not None and prev_state.agent is not None:
            LAST_POS = prev_state.agent.pos.copy()
            LAST_VEL = prev_state.agent.vel.copy()

        delta = calculate_move(rng, state)
        print(f"{delta[0]} {delta[1]}", flush=True)


if __name__ == "__main__":
    main()
