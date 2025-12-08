import sys
import enum
import numpy as np

from typing import Optional, NamedTuple, List, Tuple, Set

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

    # Perzisztens world modell:
    # ha még nincs, akkor NOT_VISIBLE (3) mindenhol; különben másolat a régiből.
    if old_state.visible_track is None:
        visible_track = np.full(circuit_data.track_shape,
                                CellType.NOT_VISIBLE.value, dtype=int)
    else:
        visible_track = old_state.visible_track.copy()

    # Játékosok pozíciói (velocity-t most nem számoljuk vissza)
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, input().split())
        players.append(Player(pposx, pposy, 0, 0))

    # Csak az aktuálisan látható ablakot frissítjük a world modellen.
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
            # Itt NEM konvertáljuk NOT_VISIBLE-t falnak, hanem meghagyjuk 3-nak,
            # hogy tudjuk, mi az, amit még nem láttunk.
            visible_track[x, y] = val

    return old_state._replace(
        visible_track=visible_track, players=players, agent=agent)

def traversable(cell_value: int) -> bool:
    # Csak a biztosan járható út (0) és a cél (100) traversable.
    # NOT_VISIBLE (3) most "ismeretlen", arra nem lépünk rá direkt.
    return cell_value == 0 or cell_value == 100

def valid_line(state: State, pos1: np.ndarray, pos2: np.ndarray) -> bool:
    track = state.visible_track
    if track is None:
        return False

    if (np.any(pos1 < 0) or np.any(pos2 < 0) or
        np.any(pos1 >= track.shape) or np.any(pos2 >= track.shape)):
        return False

    diff = pos2 - pos1

    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))  # direction: left or right
        for i in range(abs(diff[0]) + 1):
            x = pos1[0] + i*d
            y = pos1[1] + i*slope*d
            y_ceil = int(np.ceil(y))
            y_floor = int(np.floor(y))
            if (not traversable(int(track[x, y_ceil]))
                    and not traversable(int(track[x, y_floor]))):
                return False

    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))  # direction: up or down
        for i in range(abs(diff[1]) + 1):
            x = pos1[0] + i*slope*d
            y = pos1[1] + i*d
            x_ceil = int(np.ceil(x))
            x_floor = int(np.floor(x))
            if (not traversable(int(track[x_ceil, y]))
                    and not traversable(int(track[x_floor, y]))):
                return False
    return True

# -------------------- A* pathfinding a célra / frontierre -------------------

def find_nearest_goal(track: np.ndarray, start_pos: np.ndarray) -> Optional[Tuple[int, int]]:
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
    open_set: Set[Tuple[int, int]] = {start_node}
    came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: dict[Tuple[int, int], float] = {start_node: 0.0}
    f_score: dict[Tuple[int, int], float] = {start_node: heuristic(sx, sy)}

    while open_set:
        current = min(open_set, key=lambda n: f_score.get(n, float('inf')))
        if current == (gx, gy):
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

def has_unknown_neighbor(track: np.ndarray, r: int, c: int) -> bool:
    H, W = track.shape
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            if int(track[nr, nc]) == CellType.NOT_VISIBLE.value:
                return True
    return False

def compute_frontiers(track: np.ndarray,
                      visited: Optional[np.ndarray]) -> List[Tuple[int, int]]:
    """
    Frontier: olyan járható cella, aminek van NOT_VISIBLE szomszédja.
    Először azokat preferáljuk, ahol még nem jártunk (visited == False).
    """
    H, W = track.shape
    frontiers: List[Tuple[int, int]] = []

    # Először: nem visited frontier-ek
    for r in range(H):
        for c in range(W):
            if traversable(int(track[r, c])) and has_unknown_neighbor(track, r, c):
                if visited is None or not visited[r, c]:
                    frontiers.append((r, c))

    if frontiers:
        return frontiers

    # Ha már minden frontier visited, akkor engedjük a visited frontiereket is
    for r in range(H):
        for c in range(W):
            if traversable(int(track[r, c])) and has_unknown_neighbor(track, r, c):
                frontiers.append((r, c))

    return frontiers

def astar_to_frontier(track: np.ndarray,
                      start: Tuple[int, int],
                      frontiers: List[Tuple[int, int]],
                      visited: Optional[np.ndarray]) -> Optional[List[Tuple[int, int]]]:
    if not frontiers:
        return None

    H, W = track.shape
    sx, sy = start
    frontier_set = set(frontiers)

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < H and 0 <= c < W

    def is_free(r: int, c: int) -> bool:
        return traversable(int(track[r, c]))

    def heuristic(r: int, c: int) -> float:
        return min(abs(r - fr) + abs(c - fc) for fr, fc in frontiers)

    VISITED_PENALTY = 5.0

    start_node = (sx, sy)
    open_set: Set[Tuple[int, int]] = {start_node}
    came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: dict[Tuple[int, int], float] = {start_node: 0.0}
    f_score: dict[Tuple[int, int], float] = {start_node: heuristic(sx, sy)}

    while open_set:
        current = min(open_set, key=lambda n: f_score.get(n, float('inf')))
        if current in frontier_set:
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
            base_cost = 1.0
            if visited is not None and visited[nr, nc]:
                base_cost += VISITED_PENALTY  # büntetés visszamenésért

            tentative_g = g_score[current] + base_cost
            if tentative_g >= g_score.get(neighbour, float('inf')):
                continue
            came_from[neighbour] = current
            g_score[neighbour] = tentative_g
            f_score[neighbour] = tentative_g + heuristic(nr, nc)
            open_set.add(neighbour)

    return None

def plan_next_waypoint(track: np.ndarray,
                       agent_pos: np.ndarray,
                       visited: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Következő célcellát adja vissza:
    - ha látjuk a célt (100), arra tervezünk,
    - különben egy frontier cellára (exploráció).
    """
    pos = agent_pos.astype(int)

    # 1) Ha már látunk célt, menjünk oda
    goal = find_nearest_goal(track, pos)
    if goal is not None:
        path = astar_grid(track, (int(pos[0]), int(pos[1])), goal)
        if path is not None and len(path) >= 2:
            nr, nc = path[1]
            return np.array([nr, nc], dtype=int)
        return None

    # 2) Frontier-alapú exploráció
    frontiers = compute_frontiers(track, visited)
    if not frontiers:
        return None  # nincs frontier -> mindent feltérképeztünk, amit lehetett

    path = astar_to_frontier(track, (int(pos[0]), int(pos[1])), frontiers, visited)
    if path is None or len(path) < 2:
        return None

    nr, nc = path[1]
    return np.array([nr, nc], dtype=int)

# -------------------- Mozgás (dvx, dvy) + no-backtrack ----------------------

visited_track: Optional[np.ndarray] = None  # globális, hogy lássuk, hol jártunk

def calculate_move(
    rng: np.random.Generator,
    state: State,
    previous_pos: Optional[np.ndarray]
) -> tuple[int, int]:
    global visited_track
    self_pos = state.agent.pos

    def valid_move(next_move: np.ndarray) -> bool:
        return (valid_line(state, self_pos, next_move) and
                (np.all(next_move == self_pos)
                 or not any(np.all(next_move == p.pos) for p in state.players)))

    # Eredeti skeleton baseline, minimális no-backtrackkel
    def baseline_move() -> tuple[int, int]:
        new_center = self_pos + state.agent.vel
        next_move = new_center
        if (np.any(next_move != self_pos) and valid_move(next_move)
                and rng.random() > 0.1):
            # tartjuk a sebességet
            return (0, 0)
        else:
            valid_non_back: List[Tuple[int, int]] = []
            valid_back: List[Tuple[int, int]] = []
            valid_stay: Optional[Tuple[int, int]] = None

            last_step = None
            if previous_pos is not None:
                last_step = self_pos - previous_pos

            for dvx in range(-1, 2):
                for dvy in range(-1, 2):
                    delta = np.array([dvx, dvy])
                    next_move2 = new_center + delta
                    if not valid_move(next_move2):
                        continue
                    if np.all(next_move2 == self_pos):
                        valid_stay = (dvx, dvy)
                        continue

                    is_back = False
                    if last_step is not None and np.any(last_step != 0):
                        this_step = next_move2 - self_pos
                        if np.any(this_step != 0):
                            dot = float(np.dot(this_step, last_step))
                            if dot < 0:
                                is_back = True

                    if is_back:
                        valid_back.append((dvx, dvy))
                    else:
                        valid_non_back.append((dvx, dvy))

            if valid_non_back:
                return tuple(rng.choice(valid_non_back))
            elif valid_back:
                return tuple(rng.choice(valid_back))
            elif valid_stay is not None:
                return valid_stay
            else:
                print('No valid action found.', file=sys.stderr)
                return (0, 0)

    if state.visible_track is None:
        return baseline_move()

    # A*-os waypoint: vagy cél, vagy frontier
    waypoint = plan_next_waypoint(state.visible_track, self_pos, visited_track)
    if waypoint is None:
        return baseline_move()

    # Válasszunk dvx, dvy-t, ami közel visz a waypointhoz,
    # és lehetőleg nem fordul vissza.
    best_non_back: Optional[Tuple[int, int]] = None
    best_score_non_back = float('inf')
    best_back: Optional[Tuple[int, int]] = None
    best_score_back = float('inf')

    new_center = self_pos + state.agent.vel
    last_step = None
    if previous_pos is not None:
        last_step = self_pos - previous_pos

    for dvx in range(-1, 2):
        for dvy in range(-1, 2):
            delta = np.array([dvx, dvy])
            next_move = new_center + delta
            if not valid_move(next_move):
                continue
            this_step = next_move - self_pos
            score = float(np.linalg.norm(next_move - waypoint))

            is_back = False
            if last_step is not None and np.any(last_step != 0) and np.any(this_step != 0):
                dot = float(np.dot(this_step, last_step))
                if dot < 0:
                    is_back = True

            if is_back:
                if score < best_score_back:
                    best_score_back = score
                    best_back = (dvx, dvy)
            else:
                if score < best_score_non_back:
                    best_score_non_back = score
                    best_non_back = (dvx, dvy)

    if best_non_back is not None:
        return best_non_back
    if best_back is not None:
        return best_back

    return baseline_move()

# -------------------- main: judge-kommunikáció változatlan ------------------

def main():
    global visited_track

    print('READY', flush=True)
    circuit = read_initial_observation()
    state: Optional[State] = State(circuit, None, [], None)  # type: ignore
    rng = np.random.default_rng(seed=1)

    previous_pos: Optional[np.ndarray] = None

    while True:
        assert state is not None
        state = read_observation(state)
        if state is None:
            return

        # visited_track init
        if visited_track is None:
            visited_track = np.zeros(state.circuit.track_shape, dtype=bool)

        # aktuális pozíciót visited-nek jelöljük
        self_pos = state.agent.pos.astype(int)
        H, W = state.circuit.track_shape
        if 0 <= self_pos[0] < H and 0 <= self_pos[1] < W:
            visited_track[self_pos[0], self_pos[1]] = True

        delta = calculate_move(rng, state, previous_pos)
        print(f'{delta[0]} {delta[1]}', flush=True)

        previous_pos = self_pos

if __name__ == "__main__":
    main()
