import sys
import enum
import numpy as np

from typing import Optional, NamedTuple, List, Tuple

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
    visible_track: np.ndarray  # persistent world model (első körben None lesz)
    players: List[Player]
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
    players: List[Player] = []
    circuit_data = old_state.circuit

    # Perzisztens world modell: ha még nincs, akkor falakkal töltjük fel.
    if old_state.visible_track is None:
        visible_track = np.full(circuit_data.track_shape, CellType.WALL.value)
    else:
        visible_track = old_state.visible_track.copy()

    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, input().split())
        players.append(Player(pposx, pposy, 0, 0))

    # Csak az aktuálisan látható ablakot frissítjük.
    for i in range(2 * circuit_data.visibility_radius + 1):
        line_vals = [int(a) for a in input().split()]
        x = posx - circuit_data.visibility_radius + i
        if x < 0 or x >= circuit_data.track_shape[0]:
            continue
        y_start = posy - circuit_data.visibility_radius
        y_end = y_start + 2 * circuit_data.visibility_radius + 1
        if y_start < 0:
            line_vals = line_vals[-y_start:]
            y_start = 0
        if y_end > circuit_data.track_shape[1]:
            line_vals = line_vals[:-(y_end - circuit_data.track_shape[1])]
            y_end = circuit_data.track_shape[1]
        for offset, val in enumerate(line_vals):
            y = y_start + offset
            # NOT_VISIBLE-t konzervatívan falnak vesszük
            if val == CellType.NOT_VISIBLE.value:
                val = CellType.WALL.value
            visible_track[x, y] = val

    return old_state._replace(
        visible_track=visible_track, players=players, agent=agent)

def traversable(cell_value: int) -> bool:
    # skeleton: minden nem-negatív traversable, cél (100) is az
    return cell_value >= 0

def valid_line(state: State, pos1: np.ndarray, pos2: np.ndarray) -> bool:
    track = state.visible_track
    if (np.any(pos1 < 0) or np.any(pos2 < 0) or np.any(pos1 >= track.shape)
            or np.any(pos2 >= track.shape)):
        return False
    diff = pos2 - pos1
    # skeleton-féle line-of-sight falellenőrzés
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = np.sign(diff[0])  # direction: left or right
        for i in range(abs(diff[0]) + 1):
            x = pos1[0] + i*d
            y = pos1[1] + i*slope*d
            y_ceil = np.ceil(y).astype(int)
            y_floor = np.floor(y).astype(int)
            if (not traversable(track[x, y_ceil])
                    and not traversable(track[x, y_floor])):
                return False
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = np.sign(diff[1])  # direction: up or down
        for i in range(abs(diff[1]) + 1):
            x = pos1[0] + i*slope*d
            y = pos1[1] + i*d
            x_ceil = np.ceil(x).astype(int)
            x_floor = np.floor(x).astype(int)
            if (not traversable(track[x_ceil, y])
                    and not traversable(track[x_floor, y])):
                return False
    return True

# -------------------- A* pathfinding a 100-as célra -------------------------

def find_nearest_goal(track: np.ndarray, start_pos: np.ndarray) -> Optional[Tuple[int, int]]:
    """Megkeresi a legközelebbi 100-as célcellát a jelenleg ismert world modelben."""
    goal_cells = np.argwhere(track == 100)
    if goal_cells.size == 0:
        return None
    dists = np.abs(goal_cells[:, 0] - start_pos[0]) + np.abs(goal_cells[:, 1] - start_pos[1])
    idx = int(np.argmin(dists))
    gr, gc = int(goal_cells[idx, 0]), int(goal_cells[idx, 1])
    return gr, gc

def astar_grid(track: np.ndarray,
               start: Tuple[int, int],
               goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    A* a grid-en (4 irány), falakat kerüli, cél = 100.
    Nem veszi figyelembe a velocity-t, csak celláról cellára tervez.
    """
    H, W = track.shape
    sx, sy = start
    gx, gy = goal

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < H and 0 <= c < W

    def is_free(r: int, c: int) -> bool:
        return traversable(int(track[r, c]))

    def heuristic(r: int, c: int) -> float:
        return abs(r - gx) + abs(c - gy)

    start_node = (sx, sy)
    open_set: set[Tuple[int, int]] = {start_node}
    came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: dict[Tuple[int, int], float] = {start_node: 0.0}
    f_score: dict[Tuple[int, int], float] = {start_node: heuristic(sx, sy)}

    while open_set:
        current = min(open_set, key=lambda n: f_score.get(n, float('inf')))
        if current == (gx, gy):
            # reconstruct path
            path: List[Tuple[int, int]] = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        open_set.remove(current)
        cr, cc = current

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if not in_bounds(nr, nc):
                continue
            if not is_free(nr, nc):
                continue
            neighbour = (nr, nc)
            tentative_g = g_score[current] + 1.0
            if tentative_g >= g_score.get(neighbour, float('inf')):
                continue
            came_from[neighbour] = current
            g_score[neighbour] = tentative_g
            f_score[neighbour] = tentative_g + heuristic(nr, nc)
            open_set.add(neighbour)

    return None

def plan_next_waypoint(state: State) -> Optional[np.ndarray]:
    """
    A következő waypoint cella (a path második pontja) a legközelebbi 100-as cél felé.
    Ha nincs cél vagy út, None.
    """
    track = state.visible_track
    self_pos = state.agent.pos.astype(int)

    goal = find_nearest_goal(track, self_pos)
    if goal is None:
        return None

    path = astar_grid(track, (int(self_pos[0]), int(self_pos[1])), goal)
    if path is None or len(path) < 2:
        return None

    nr, nc = path[1]
    return np.array([nr, nc], dtype=int)

# -------------------- Mozgás (dvx, dvy) a path szerint + no-backtrack -------

def calculate_move(rng: np.random.Generator, state: State, previous_pos: Optional[np.ndarray]) -> tuple[int, int]:
    self_pos = state.agent.pos

    def valid_move(next_move: np.ndarray) -> bool:
        return (valid_line(state, self_pos, next_move) and
                (np.all(next_move == self_pos)
                 or not any(np.all(next_move == p.pos) for p in state.players)))

    def baseline_move() -> tuple[int, int]:
        # Eredeti skeleton controller – fallback, ha nincs cél/út,
        # de kiegészítve a "ne backtrackeljen" logikával.
        new_center = self_pos + state.agent.vel
        next_move = new_center
        if (np.any(next_move != self_pos) and valid_move(next_move)
                and rng.random() > 0.1):
            # ilyenkor a dv=(0,0), csak a jelenlegi velocity-t tartjuk
            return (0, 0)
        else:
            valid_moves_non_back: List[Tuple[int, int]] = []
            valid_moves_back: List[Tuple[int, int]] = []
            valid_stay: Optional[Tuple[int, int]] = None
            for i in range(-1, 2):
                for j in range(-1, 2):
                    delta = np.array([i, j])
                    next_move = new_center + delta
                    if not valid_move(next_move):
                        continue
                    if np.all(self_pos == next_move):
                        # itt csak állva maradunk, ez nem backtrack
                        valid_stay = (i, j)
                        continue
                    # Categorize as back / non-back relative to previous_pos
                    if previous_pos is not None and np.all(next_move == previous_pos):
                        valid_moves_back.append((i, j))
                    else:
                        valid_moves_non_back.append((i, j))

            if valid_moves_non_back:
                return tuple(rng.choice(valid_moves_non_back))
            elif valid_moves_back:
                # ide csak akkor jutunk, ha előre/szélre semmi nem volt
                return tuple(rng.choice(valid_moves_back))
            elif valid_stay is not None:
                return valid_stay
            else:
                print(
                    'Not blind, just being brave! (No valid action found.)',
                    file=sys.stderr)
                return (0, 0)

    # 1) próbálunk waypointot találni a 100-as cél felé
    waypoint = plan_next_waypoint(state)

    if waypoint is None:
        # nincs cél vagy út -> marad a baseline viselkedés
        return baseline_move()

    # 2) Válasszunk (dvx, dvy)-t, ami a waypoint felé mozgat
    #    úgy, hogy lehetőleg ne menjünk vissza az előző cellába.
    best_delta_non_back: Optional[Tuple[int, int]] = None
    best_score_non_back: float = float('inf')
    best_delta_back: Optional[Tuple[int, int]] = None
    best_score_back: float = float('inf')

    new_center = self_pos + state.agent.vel  # skeleton-féle "közép"

    for dvx in range(-1, 2):
        for dvy in range(-1, 2):
            delta = np.array([dvx, dvy])
            next_move = new_center + delta  # predicted next position

            if not valid_move(next_move):
                continue

            # score: távolság a waypointtól
            score = float(np.linalg.norm(next_move - waypoint))

            is_back = previous_pos is not None and np.all(next_move == previous_pos)

            if is_back:
                if score < best_score_back:
                    best_score_back = score
                    best_delta_back = (dvx, dvy)
            else:
                if score < best_score_non_back:
                    best_score_non_back = score
                    best_delta_non_back = (dvx, dvy)

    # Először nem-backtrack megoldást próbálunk választani
    if best_delta_non_back is not None:
        return best_delta_non_back
    # Ha nincs nem-backtrack lépés, de van backtrack, azt választjuk
    if best_delta_back is not None:
        return best_delta_back

    # ha semmi sem jó, inkább baseline
    return baseline_move()

def main():
    print('READY', flush=True)
    circuit = read_initial_observation()
    state: Optional[State] = State(circuit, None, [], None)  # type: ignore
    rng = np.random.default_rng(seed=1)
    while True:
        assert state is not None
        # elmentjük, hol voltunk az előző körben (arra nem akarunk visszamenni)
        old_agent = state.agent
        state = read_observation(state)
        if state is None:
            return
        previous_pos = None
        if old_agent is not None:
            previous_pos = old_agent.pos.copy()
        delta = calculate_move(rng, state, previous_pos)
        print(f'{delta[0]} {delta[1]}', flush=True)

if __name__ == "__main__":
    main()
