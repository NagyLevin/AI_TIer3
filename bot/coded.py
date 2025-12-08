import sys
import enum
import numpy as np

from collections import deque
from typing import Optional, NamedTuple


class CellType(enum.IntEnum):
    """
    Cell types mirrored from the server.

    Values (must match grid_race_env.py on the server):

        GOAL        = 100
        START       = 1
        WALL        = -1
        UNKNOWN     = 2
        EMPTY       = 0
        NOT_VISIBLE = 3   # "fog of war" in observations
        OIL         = 91
        SAND        = 92
    """
    GOAL = 100
    START = 1
    WALL = -1
    UNKNOWN = 2
    EMPTY = 0
    NOT_VISIBLE = 3
    OIL = 91
    SAND = 92


# Ezek a "puha" mezők (olaj, homok) – kerülni próbáljuk őket.
SOFT_TERRAINS = {CellType.OIL.value, CellType.SAND.value}


class Player(NamedTuple):
    """
    Player description in the client state.

    x, y      : integer grid position (row, column)
    vel_x,y   : current velocity components
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
    Static circuit information sent at the very beginning.

    track_shape       : (H, W) size of the whole map
    num_players       : number of cars on the track
    visibility_radius : radius R of the (2R+1)x(2R+1) observation window
    """
    track_shape: tuple[int, int]
    num_players: int
    visibility_radius: int


class State(NamedTuple):
    """
    Complete client-side state.

    circuit        : static info about the circuit
    visible_track  : our *global* knowledge of the track (H x W)
                     - starts as all NOT_VISIBLE
                     - every turn we merge the newly seen patch into it
    players        : approximate positions of all players
    agent          : our own player
    """
    circuit: Circuit
    visible_track: Optional[np.ndarray]
    players: list[Player]
    agent: Optional[Player]


# Global visit counter to encourage full exploration
VISIT_COUNTS: Optional[np.ndarray] = None

# Soft speed limit for planning (environment allows any speed, but we
# keep it moderate so that we can still steer and stop safely).
SPEED_LIMIT = 4


def read_initial_observation() -> Circuit:
    """
    Read H, W, num_players, visibility_radius from stdin.

    This is called once at the very beginning.
    """
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)


def read_observation(old_state: State) -> Optional[State]:
    """
    Read one full observation from stdin and update our global track view.

    The judge sends:
      - one line with our position and velocity,
      - num_players lines with approximate player positions,
      - (2R+1) lines describing the (2R+1)x(2R+1) observation window.

    We merge the observed patch into a persistent HxW "visible_track"
    so that we slowly uncover the whole map.
    """
    line = input()
    if line == '~~~END~~~':
        # End of game
        return None

    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)

    players: list[Player] = []
    circuit_data = old_state.circuit

    # Read other players' approximate positions
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, input().split())
        players.append(Player(pposx, pposy, 0, 0))

    # Initialise or copy over our global track knowledge
    if old_state.visible_track is None:
        visible_track = np.full(
            circuit_data.track_shape,
            CellType.NOT_VISIBLE.value,
            dtype=int
        )
    else:
        visible_track = old_state.visible_track.copy()

    R = circuit_data.visibility_radius

    # Merge the (2R+1)x(2R+1) patch into visible_track.
    # We *only* overwrite with values that are not NOT_VISIBLE
    # to avoid erasing older information.
    for i in range(2 * R + 1):
        row_vals = [int(a) for a in input().split()]
        x = posx - R + i

        if x < 0 or x >= circuit_data.track_shape[0]:
            # Entire row is off-map
            continue

        y_start = posy - R
        y_end = posy + R + 1

        if y_start < 0:
            # Drop elements that would be left of the map
            row_vals = row_vals[-y_start:]
            y_start = 0
        if y_end > circuit_data.track_shape[1]:
            # Drop elements that would be right of the map
            row_vals = row_vals[:-(y_end - circuit_data.track_shape[1])]
            y_end = circuit_data.track_shape[1]

        for offset, y in enumerate(range(y_start, y_end)):
            val = row_vals[offset]
            if val == CellType.NOT_VISIBLE.value:
                # No new information here
                continue
            visible_track[x, y] = val

    return old_state._replace(
        visible_track=visible_track,
        players=players,
        agent=agent
    )


def traversable(cell_value: int) -> bool:
    """
    Decide if a numeric cell is traversable in the real physics.

    We consider cells traversable if:
      - they are NOT NOT_VISIBLE, and
      - their value is >= 0 (same rule as in the server code).

    This means OIL and SAND are traversable here; we will avoid them
    at the planning level, not by prohibiting them.
    """
    if cell_value == CellType.NOT_VISIBLE.value:
        return False
    return cell_value >= 0


def traversable_planning(cell_value: int, avoid_soft: bool) -> bool:
    """
    Traversability used in path-planning (BFS).

    If avoid_soft is True, OIL and SAND are treated as walls so that
    we only go there if there is literally no alternative path.
    """
    if cell_value == CellType.NOT_VISIBLE.value:
        return False
    if avoid_soft and cell_value in SOFT_TERRAINS:
        return False
    return cell_value >= 0


def is_line_clear(track: np.ndarray,
                  pos1: np.ndarray,
                  pos2: np.ndarray,
                  avoid_soft: bool = False) -> bool:
    """
    Check if the straight line between pos1 and pos2 stays on traversable
    cells (according to our *current* knowledge of the track).

    This mirrors the server's valid_line(...) logic and supports an
    'avoid_soft' flag for planning that can treat oil/sand as walls.
    """
    H, W = track.shape
    if (np.any(pos1 < 0) or np.any(pos2 < 0)
            or pos1[0] >= H or pos1[1] >= W
            or pos2[0] >= H or pos2[1] >= W):
        return False

    diff = pos2 - pos1

    # Scan in x direction (check vertical pairs)
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))
        for i in range(abs(int(diff[0])) + 1):
            x = int(pos1[0] + i * d)
            y = pos1[1] + i * slope * d
            y_floor = int(np.floor(y))
            y_ceil = int(np.ceil(y))
            if (0 <= x < H and 0 <= y_floor < W and 0 <= y_ceil < W):
                cell1 = track[x, y_floor]
                cell2 = track[x, y_ceil]
                if (not traversable_planning(cell1, avoid_soft)
                        and not traversable_planning(cell2, avoid_soft)):
                    return False
            else:
                return False

    # Scan in y direction (check horizontal pairs)
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))
        for i in range(abs(int(diff[1])) + 1):
            x = pos1[0] + i * slope * d
            y = int(pos1[1] + i * d)
            x_floor = int(np.floor(x))
            x_ceil = int(np.ceil(x))
            if (0 <= y < W and 0 <= x_floor < H and 0 <= x_ceil < H):
                cell1 = track[x_floor, y]
                cell2 = track[x_ceil, y]
                if (not traversable_planning(cell1, avoid_soft)
                        and not traversable_planning(cell2, avoid_soft)):
                    return False
            else:
                return False

    return True


def valid_line(state: State,
               pos1: np.ndarray,
               pos2: np.ndarray,
               avoid_soft: bool = False) -> bool:
    """
    Wrapper around is_line_clear that uses the state's global visible_track.
    """
    assert state.visible_track is not None
    return is_line_clear(state.visible_track, pos1, pos2, avoid_soft=avoid_soft)


def is_valid_move_to(state: State,
                     from_pos: np.ndarray,
                     next_pos: np.ndarray,
                     avoid_soft: bool = False) -> bool:
    """
    Check whether moving from from_pos to next_pos is valid:

      - the straight line must stay on traversable cells
        (and optionally must avoid oil / sand),
      - the target cell must not be occupied by another player
        (unless we are staying in place).
    """
    if not valid_line(state, from_pos, next_pos, avoid_soft=avoid_soft):
        return False

    if np.all(next_pos == from_pos):
        return True

    for p in state.players:
        if np.all(next_pos == p.pos):
            return False

    return True


def count_unknown_in_radius(track: np.ndarray,
                            center: np.ndarray,
                            radius: int) -> int:
    """
    Count how many NOT_VISIBLE cells are within a given Euclidean radius
    around 'center'.
    """
    H, W = track.shape
    x0, y0 = int(center[0]), int(center[1])
    unknown_val = CellType.NOT_VISIBLE.value
    r2 = radius * radius
    count = 0

    x_min = max(0, x0 - radius)
    x_max = min(H - 1, x0 + radius)
    y_min = max(0, y0 - radius)
    y_max = min(W - 1, y0 + radius)

    for x in range(x_min, x_max + 1):
        dx = x - x0
        for y in range(y_min, y_max + 1):
            dy = y - y0
            if dx * dx + dy * dy <= r2 and track[x, y] == unknown_val:
                count += 1

    return count


def find_path_to_nearest_target(track: np.ndarray,
                                start: np.ndarray,
                                target_cells: list[tuple[int, int]],
                                bias: Optional[np.ndarray] = None,
                                avoid_soft: bool = False
                                ) -> Optional[list[tuple[int, int]]]:
    """
    BFS path finder on the known track.

    - 'avoid_soft' jelzi, hogy olaj/homok mezőket falnak vegyük-e.
    - 'bias' opcionálisan a cél régió felé preferálja a lépéseket
      (pl. ha már látjuk a célt, de még nincs odáig út).
    """
    if not target_cells:
        return None

    H, W = track.shape
    start_cell = (int(start[0]), int(start[1]))
    targets = set((int(x), int(y)) for (x, y) in target_cells)

    if start_cell in targets:
        return [start_cell]

    q: deque[tuple[int, int]] = deque()
    q.append(start_cell)
    visited = {start_cell}
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    base_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        cx, cy = q.popleft()
        if (cx, cy) in targets:
            # Reconstruct path
            path = [(cx, cy)]
            while (cx, cy) != start_cell:
                cx, cy = parent[(cx, cy)]
                path.append((cx, cy))
            path.reverse()
            return path

        # bias: irány-prioritás a cél felé
        if bias is not None:
            gx, gy = int(bias[0]), int(bias[1])
            vecx, vecy = gx - cx, gy - cy

            def dir_score(d):
                dx, dy = d
                return dx * vecx + dy * vecy

            directions = sorted(base_directions, key=dir_score, reverse=True)
        else:
            directions = base_directions

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            ncell = (nx, ny)
            if ncell in visited:
                continue
            if not traversable_planning(track[nx, ny], avoid_soft):
                continue
            if not is_line_clear(track,
                                 np.array([cx, cy], dtype=int),
                                 np.array([nx, ny], dtype=int),
                                 avoid_soft=avoid_soft):
                continue
            visited.add(ncell)
            parent[ncell] = (cx, cy)
            q.append(ncell)

    return None


def find_path_to_nearest_goal(track: np.ndarray,
                              start: np.ndarray,
                              avoid_soft: bool = False
                              ) -> Optional[list[tuple[int, int]]]:
    """
    Wrapper, ami a legközelebbi 100-as GOAL cellához keres utat.
    """
    goal_positions = np.argwhere(track == CellType.GOAL.value)
    if goal_positions.size == 0:
        return None
    target_cells = [(int(x), int(y)) for x, y in goal_positions]
    return find_path_to_nearest_target(
        track, start, target_cells, bias=None, avoid_soft=avoid_soft
    )


def compute_frontier_cells(track: np.ndarray,
                           visibility_radius: int) -> list[tuple[int, int]]:
    """
    Frontier cellák (bejárható cellák, ahonnan még látszik ismeretlen rész).
    Ezeket kell végigjárni ahhoz, hogy a HxW bejárható részét feltérképezzük.
    """
    H, W = track.shape
    frontier: list[tuple[int, int]] = []

    for x in range(H):
        for y in range(W):
            if not traversable(track[x, y]):
                continue
            if count_unknown_in_radius(track,
                                       np.array([x, y], dtype=int),
                                       visibility_radius) > 0:
                frontier.append((x, y))

    return frontier


def choose_accel_toward_cell(state: State,
                             target_cell: np.ndarray,
                             visit_counts: Optional[np.ndarray] = None,
                             prefer_slow: bool = False,
                             avoid_soft: bool = False
                             ) -> Optional[tuple[int, int]]:
    """
    Path követése: válasszunk (ax, ay)-t a target_cell felé.

    - 9 gyorsulás (ax,ay in {-1,0,1}) közül választ.
    - new_vel = vel + delta, new_pos = pos + new_vel (gyorsulás összeadódik).
    - Csak olyan lépés érvényes, ami nem megy falba / más játékosba,
      és ha avoid_soft=True, akkor nem megy olaj/homok mezőkre sem.
    - distance + kis visited + soft penalty + (opcionális) sebesség büntetés
      alapján választjuk a legkisebb "költségű" gyorsulást.
    """
    assert state.agent is not None
    assert state.visible_track is not None

    pos = state.agent.pos
    vel = state.agent.vel
    track = state.visible_track

    best_delta: Optional[tuple[int, int]] = None
    best_score = float('inf')

    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            delta = np.array([ax, ay], dtype=int)
            new_vel = vel + delta
            speed = np.linalg.norm(new_vel, ord=2)

            # sebesség-korlát
            if speed > SPEED_LIMIT:
                continue

            new_pos = pos + new_vel

            if not is_valid_move_to(state, pos, new_pos, avoid_soft=avoid_soft):
                continue

            # targettől való távolság
            dist = np.linalg.norm(target_cell - new_pos)

            # látogatottság kis büntetése
            if visit_counts is not None:
                x, y = int(new_pos[0]), int(new_pos[1])
                if (0 <= x < visit_counts.shape[0]
                        and 0 <= y < visit_counts.shape[1]):
                    dist += 0.1 * visit_counts[x, y]

            # olaj/homok büntetés, ha épp nem kötelező elkerülni
            cell_val = track[int(new_pos[0]), int(new_pos[1])]
            if cell_val in SOFT_TERRAINS and not avoid_soft:
                dist += 5.0

            # lassabb mozgás preferálása
            if prefer_slow:
                dist += 0.05 * speed

            if dist < best_score:
                best_score = dist
                best_delta = (ax, ay)

    return best_delta


def choose_exploration_move(state: State,
                            visit_counts: np.ndarray,
                            rng: np.random.Generator,
                            goal_hint: Optional[np.ndarray] = None
                            ) -> tuple[int, int]:
    """
    Lokális explorációs lépés.

    - minden érvényes gyorsulásra becsüljük:
        * mennyi új cellát látunk (NOT_VISIBLE),
        * mennyire jártuk már azt a cellát,
        * mennyire messzire kerülünk a cél-régiótól (goal_hint),
        * olaj/homok mező-e (nagy büntetés).
    - a legnagyobb score-t adó lépést választjuk.

    Így az olaj/homok mezőket csak akkor választjuk, ha
    tényleg sok új területet nyitnak meg, és másik,
    olaj/homok-mentes lépés nem ad hasonlóan jó eredményt.
    """
    assert state.agent is not None
    assert state.visible_track is not None

    pos = state.agent.pos
    vel = state.agent.vel
    track = state.visible_track
    R = state.circuit.visibility_radius

    best_moves: list[tuple[int, int]] = []
    best_score = -float('inf')

    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            delta = np.array([ax, ay], dtype=int)
            new_vel = vel + delta
            speed = np.linalg.norm(new_vel, ord=2)

            # sebesség-korlát
            if speed > SPEED_LIMIT:
                continue

            new_pos = pos + new_vel

            # itt még engedjük az olaj/homokot, a pontozás bünteti
            if not is_valid_move_to(state, pos, new_pos, avoid_soft=False):
                continue

            unknown_count = count_unknown_in_radius(track, new_pos, R)

            x, y = int(new_pos[0]), int(new_pos[1])
            visited_penalty = 0
            if (0 <= x < visit_counts.shape[0]
                    and 0 <= y < visit_counts.shape[1]):
                visited_penalty = visit_counts[x, y]

            cell_val = track[x, y]
            soft_penalty = 0.0
            if cell_val in SOFT_TERRAINS:
                # nagyon nagy büntetés – csak akkor megy oda,
                # ha tényleg brutálisan sok új cellát fedne fel
                soft_penalty = 500.0

            # score: minél több új cella, minél kevesebb revisitelés,
            # minél kevesebb sebesség / soft penalty.
            score = unknown_count * 100.0 - visited_penalty - 0.5 * speed - soft_penalty

            # célhoz közeledés (ha tudjuk hol van)
            if goal_hint is not None:
                goal_dist = np.linalg.norm(goal_hint - new_pos)
                score -= 0.5 * goal_dist

            if score > best_score:
                best_score = score
                best_moves = [(ax, ay)]
            elif score == best_score:
                best_moves.append((ax, ay))

    if not best_moves:
        print(
            'Not blind, just being brave! (No valid exploration action found.)',
            file=sys.stderr
        )
        return (0, 0)

    chosen = rng.choice(best_moves)
    return int(chosen[0]), int(chosen[1])


def calculate_move(rng: np.random.Generator, state: State) -> tuple[int, int]:
    """
    High-level döntés:

    1) Globális visit heatmap frissítése.
    2) Cél (100-as cella) elérése:
       - először BFS olyan pályán, ahol olaj/homok = fal (avoid_soft=True),
       - ha így nincs út, újra BFS olaj/homok engedélyezésével.
    3) Ha nincs még cél-út:
       - frontier cellák (bejárható + körülötte unknown) keresése,
       - frontierre menő BFS ugyanúgy kétlépcsős (először soft nélkül),
       - ha már frontieren vagyunk / BFS failed, lokális exploráció
         (olaj/homok erősen büntetve).
    4) Ha frontier sincs (már mindent feltérképeztünk), próbálunk
       lassítani, biztonságosan megállni.
    """
    assert state.agent is not None
    assert state.visible_track is not None

    global VISIT_COUNTS

    track = state.visible_track
    pos = state.agent.pos

    # visit heatmap inicializálása
    if VISIT_COUNTS is None:
        VISIT_COUNTS = np.zeros(track.shape, dtype=int)

    # aktuális cella megjelölése
    VISIT_COUNTS[pos[0], pos[1]] += 1

    # célpozíciók (GOAL) felderítése
    goal_positions = np.argwhere(track == CellType.GOAL.value)
    goal_hint: Optional[np.ndarray] = None
    if goal_positions.size > 0:
        # a legközelebbi cél lesz a "goal_hint"
        dists = [np.linalg.norm(pos - g.astype(int)) for g in goal_positions]
        goal_hint = goal_positions[int(np.argmin(dists))].astype(int)

    # --- 1) Cél felé menés (ha van ismert cél) ---------------------------- #
    path_to_goal = find_path_to_nearest_goal(track, pos, avoid_soft=True)
    used_soft_for_goal = False

    if path_to_goal is None:
        # nincs soft-mentes út -> engedjük az olaj/homokot
        path_to_goal = find_path_to_nearest_goal(track, pos, avoid_soft=False)
        used_soft_for_goal = path_to_goal is not None

    if path_to_goal is not None:
        # már a célon állunk
        if len(path_to_goal) == 1:
            return (0, 0)

        next_waypoint = np.array(path_to_goal[1], dtype=int)
        accel = choose_accel_toward_cell(
            state,
            next_waypoint,
            visit_counts=VISIT_COUNTS,
            prefer_slow=True,
            avoid_soft=not used_soft_for_goal  # ha nem kellett soft, továbbra se menjünk rá
        )
        if accel is not None:
            return accel
        # ha valamiért nem talál gyorsulást a waypoint felé, esünk vissza explorációra

    # --- 2) Exploráció ---------------------------------------------------- #
    R = state.circuit.visibility_radius
    frontier_cells = compute_frontier_cells(track, R)

    if frontier_cells:
        # aktuális cella frontier-e?
        current_is_frontier = any(
            (pos[0] == fx and pos[1] == fy) for (fx, fy) in frontier_cells
        )

        if not current_is_frontier:
            # első kör: frontier elérése olaj/homok nélkül
            path_to_frontier = find_path_to_nearest_target(
                track, pos, frontier_cells, bias=goal_hint, avoid_soft=True
            )
            used_soft_for_frontier = False

            if path_to_frontier is None:
                # csak olaj/homokon/homokon át érhető el frontier
                path_to_frontier = find_path_to_nearest_target(
                    track, pos, frontier_cells, bias=goal_hint, avoid_soft=False
                )
                used_soft_for_frontier = path_to_frontier is not None

            if path_to_frontier is not None and len(path_to_frontier) > 1:
                next_waypoint = np.array(path_to_frontier[1], dtype=int)
                accel = choose_accel_toward_cell(
                    state,
                    next_waypoint,
                    visit_counts=VISIT_COUNTS,
                    prefer_slow=True,
                    avoid_soft=not used_soft_for_frontier
                )
                if accel is not None:
                    return accel
                # ha ez sem sikerül, lokális exploráció

        # már frontieren vagyunk / BFS nem ment -> lokális exploráció
        return choose_exploration_move(state, VISIT_COUNTS, rng, goal_hint)

    # --- 3) Mindent feltérképeztünk, cél sincs -> lassítsunk -------------- #
    vel = state.agent.vel
    if np.all(vel == 0):
        return (0, 0)

    desired_delta = -np.sign(vel).astype(int)
    candidate_pos = pos + vel + desired_delta

    if is_valid_move_to(state, pos, candidate_pos, avoid_soft=False):
        return int(desired_delta[0]), int(desired_delta[1])

    # ha a sima lassítás sem fér bele, használjuk a lokális heurisztikát
    return choose_exploration_move(state, VISIT_COUNTS, rng, goal_hint)


def main():
    """
    Main loop of the bot:

      - Print 'READY' so that the judge knows we are alive.
      - Read initial circuit information.
      - Then repeatedly:
          * read an observation,
          * compute our acceleration,
          * print it to stdout.
    """
    print('READY', flush=True)
    circuit = read_initial_observation()
    state: Optional[State] = State(circuit, None, [], None)  # type: ignore
    rng = np.random.default_rng(seed=1)

    while True:
        assert state is not None
        state = read_observation(state)
        if state is None:
            return
        delta = calculate_move(rng, state)
        print(f'{delta[0]} {delta[1]}', flush=True)


if __name__ == "__main__":
    main()
