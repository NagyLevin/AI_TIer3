import sys
import enum  # Enum support
import numpy as np  # Matek + tömbök
import heapq  # Prio queue A*-hoz
from collections import deque  # Gyors queue BFS-hez
from typing import Optional, NamedTuple, List, Tuple


class CellType(enum.Enum):
    """
    Map tile kódok
    """
    WALL = -1           # Fal
    EMPTY = 0           # Aszfalt, mehet
    START = 1           # Start
    UNKNOWN = -2        # Fog of war (felfedezni!)
    NOT_VISIBLE = 3     # cant see this
    OIL = 91            # oil, csúszik
    SAND = 92           # sand, lassú
    GOAL = 100          # goal, ide kell eljutni


# Gyors lookup set a veszélyekre, hatha boviteni kell meg massal is
HAZARDS = {CellType.OIL.value, CellType.SAND.value}


class Player(NamedTuple):
    """
    Player struct
    """
    x: int      # pos X
    y: int      # pos Y
    vel_x: int  # Sebesség X
    vel_y: int  # Sebesség Y

    # Helper: pos -> numpy vector
    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    # Helper: vel -> numpy vector
    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vel_x, self.vel_y])


class Circuit(NamedTuple):
    """
    Static pálya adatok
    """
    track_shape: tuple[int, int]  # H x W méret
    num_players: int              # Hányan vagyunk
    visibility_radius: int        # Látótáv


class State(NamedTuple):
    """
    Aktuális snapshot
    """
    circuit: Circuit              # Pálya infók
    visible_track: np.ndarray     # Mit látok most
    players: list[Player]         # Hol vannak a többiek
    agent: Player                 # Hol vagyok én


class GlobalMap:
    """
    Egesz térkép
    """

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape  # Méret mentés
        # Init: minden cella UNKNOWN (-2)
        self.grid = np.full(shape, CellType.UNKNOWN.value, dtype=int)

    def update(self, visible_track: np.ndarray):
        """
        Térkép frissítése
        """
        # Mask: ami NEM latszik
        mask = (visible_track != CellType.NOT_VISIBLE.value)
        # Csak a látható részeket írjuk felül
        self.grid[mask] = visible_track[mask]


def find_nearest_unknown(
    start: tuple[int, int],
    gmap: GlobalMap,
    avoid_hazards: bool = True
) -> Optional[tuple[int, int]]:
    """
    BFS -> legközelebbi UNKNOWN keresése
    """
    queue = deque([start])  # Queue init starttal
    visited = {start}       # Visited set init

    # Quick check: ha már ismeretlenben állunk
    if gmap.grid[start] == CellType.UNKNOWN.value:
        return start  # Kész is vagyunk

    H, W = gmap.shape  # Méretek kiszedése

    # 8 irány definíció
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while queue:  # Amíg van elem
        cx, cy = queue.popleft()  # Pop current
        for dx, dy in directions:  # Szomszédok
            nx, ny = cx + dx, cy + dy  # Új koord

            # Map bounds check
            if not (0 <= nx < H and 0 <= ny < W):
                continue

            # Visited check
            if (nx, ny) in visited:
                continue

            val = gmap.grid[nx, ny]  # Cella típusa

            # Wall check
            if val == CellType.WALL.value:
                continue

            # Hazard check (ha be van kapcsolva)
            if avoid_hazards and val in HAZARDS:
                continue

            # Target found: ismeretlen
            if val == CellType.UNKNOWN.value:
                return (nx, ny)  # Visszaadjuk

            # Next step
            visited.add((nx, ny))   # <<< fix: különben BFS szétcsúszik
            queue.append((nx, ny))  # Push queue

    return None  # Nincs találat


# ────────────────────────────────────────────────────────────────────────────────
# PATHFINDING (A*)
# ────────────────────────────────────────────────────────────────────────────────

def run_astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    gmap: GlobalMap,
    avoid_hazards: bool
) -> List[tuple[int, int]]:
    """
    Standard A*
    """
    H, W = gmap.shape  # Méretek
    (sx, sy) = start   # Start tuple
    (gx, gy) = goal    # Goal tuple

    # Validálások (start/cél pályán belül?)
    if not (0 <= sx < H and 0 <= sy < W): return []
    if not (0 <= gx < H and 0 <= gy < W): return []
    # Cél nem lehet falban
    if gmap.grid[gx, gy] == CellType.WALL.value: return []

    # Priority Queue init: (cost, pos)
    pq: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
    # Path rekonstrukcióhoz
    came_from: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    # G-score (eddigi költség)
    cost_so_far: dict[Tuple[int, int], float] = {start: 0.0}

    # Irányok
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while pq:  # Amíg van node
        _, current = heapq.heappop(pq)  # Get best
        if current == goal:  # Célba értünk
            break

        cx, cy = current  # Current coords

        for dx, dy in directions:
            if dx == 0 and dy == 0: continue

            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < H and 0 <= ny < W): continue

            val = gmap.grid[nx, ny]

            if val == CellType.WALL.value: continue
            if avoid_hazards and val in HAZARDS: continue

            # Cost calc (hazard = 20, sima = 1)
            step_cost = 20.0 if val in HAZARDS else 1.0
            new_cost = cost_so_far[current] + step_cost

            if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                cost_so_far[(nx, ny)] = new_cost
                priority = new_cost + abs(gx - nx) + abs(gy - ny)
                heapq.heappush(pq, (priority, (nx, ny)))
                came_from[(nx, ny)] = current

    if goal not in came_from:
        return []

    path: List[Tuple[int, int]] = []
    curr: Optional[Tuple[int, int]] = goal
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path


# ────────────────────────────────────────────────────────────────────────────────
# extra stuff: "physics A*" so we can actually go fast on straights
# (nem csak cella->cella, hanem (pos+vel) móka is)
# ────────────────────────────────────────────────────────────────────────────────

def _walk_grid_cells(x0: int, y0: int, x1: int, y1: int):
    # super basic line walk (bresenham-ish), good enough to avoid wall clipping
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield (x, y)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _segment_ok(
    a: tuple[int, int],
    b: tuple[int, int],
    gmap: GlobalMap,
    avoid_hazards: bool
) -> bool:
    # checks the whole segment, not just the end cell (important when speed > 1)
    H, W = gmap.shape
    first = True
    for cx, cy in _walk_grid_cells(a[0], a[1], b[0], b[1]):
        if first:
            first = False
            continue
        if not (0 <= cx < H and 0 <= cy < W):
            return False
        v = gmap.grid[cx, cy]
        if v == CellType.WALL.value:
            return False
        if avoid_hazards and v in HAZARDS:
            return False
    return True


def _h_guess(
    pos: tuple[int, int],
    goal: tuple[int, int],
    speed_cap: int
) -> float:
    # lazy heuristic: euclid / speed_cap (bigger speed means fewer turns)
    dx = goal[0] - pos[0]
    dy = goal[1] - pos[1]
    d = float((dx * dx + dy * dy) ** 0.5)
    return d / max(1.0, float(speed_cap))


def _first_accel_from_kin_astar(
    start_pos: tuple[int, int],
    start_vel: tuple[int, int],
    goal: tuple[int, int],
    gmap: GlobalMap,
    avoid_hazards: bool,
    speed_cap: int,
    blocked: set[tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    """
    physics-aware A*:
    state = (x, y, vx, vy)
    try ax,ay in [-1..1], do v'=v+a, p'=p+v'
    return the first (ax,ay) on the best path
    """
    H, W = gmap.shape
    sx, sy = start_pos
    svx, svy = start_vel
    gx, gy = goal

    if not (0 <= sx < H and 0 <= sy < W):
        return None
    if not (0 <= gx < H and 0 <= gy < W):
        return None
    if gmap.grid[gx, gy] == CellType.WALL.value:
        return None
    if (gx, gy) in blocked:
        return None

    svx = int(np.clip(svx, -speed_cap, speed_cap))
    svy = int(np.clip(svy, -speed_cap, speed_cap))

    start = (sx, sy, svx, svy)

    pq: List[Tuple[float, float, Tuple[int, int, int, int]]] = []
    heapq.heappush(pq, (_h_guess((sx, sy), goal, speed_cap), 0.0, start))

    # parent map but we also store which accel got us there
    parent: dict[Tuple[int, int, int, int], Tuple[Tuple[int, int, int, int], Tuple[int, int]]] = {}
    best_g: dict[Tuple[int, int, int, int], float] = {start: 0.0}

    acc_opts = [(ax, ay) for ax in (-1, 0, 1) for ay in (-1, 0, 1)]

    while pq:
        _, g_here, st = heapq.heappop(pq)
        x, y, vx, vy = st

        if (x, y) == goal:
            # unwind until first move
            cur = st
            while cur in parent:
                prev, act = parent[cur]
                if prev == start:
                    return act
                cur = prev
            return (0, 0)

        for ax, ay in acc_opts:
            nvx = vx + ax
            nvy = vy + ay

            if abs(nvx) > speed_cap or abs(nvy) > speed_cap:
                continue

            nx = x + nvx
            ny = y + nvy

            if not (0 <= nx < H and 0 <= ny < W):
                continue

            if (nx, ny) in blocked:
                continue

            tile = gmap.grid[nx, ny]
            if tile == CellType.WALL.value:
                continue
            if avoid_hazards and tile in HAZARDS:
                continue

            if not _segment_ok((x, y), (nx, ny), gmap, avoid_hazards):
                continue

            # cost: mostly 1 per turn, but punish hazards a bit so it prefers clean
            step = 1.0
            if tile == CellType.OIL.value:
                step += 50.0
            if tile == CellType.SAND.value:
                step += 30.0
            if tile == CellType.UNKNOWN.value:
                step += 1.0

            ng = g_here + step
            nxt = (nx, ny, nvx, nvy)

            if nxt not in best_g or ng < best_g[nxt]:
                best_g[nxt] = ng
                f = ng + _h_guess((nx, ny), goal, speed_cap)
                heapq.heappush(pq, (f, ng, nxt))
                parent[nxt] = (st, (ax, ay))

    return None


def _pick_far_target(
    path: List[Tuple[int, int]],
    state: State,
    lookahead: int
) -> Optional[Tuple[int, int]]:
    # goal: instead of aiming at next cell, aim a bit ahead -> then physics A* can speed up
    if len(path) < 2:
        return None

    my_pos = (state.agent.x, state.agent.y)
    blocked = {(p.x, p.y) for p in state.players}
    if my_pos in blocked:
        blocked.remove(my_pos)

    idx = min(len(path) - 1, max(1, lookahead))

    for i in range(idx, 0, -1):
        c = path[i]
        if c not in blocked:
            return c

    return choose_next_free_cell_on_path(path, state)


def _try_fast_step(
    state: State,
    gmap: GlobalMap,
    target: tuple[int, int],
    avoid_hazards: bool
) -> Optional[Tuple[int, int]]:
    # próbálunk nagyobb cap-ekkel először, ha nem megy akkor visszaesünk
    R = state.circuit.visibility_radius
    max_cap = max(1, min(R // 2, 3))

    my_pos = (state.agent.x, state.agent.y)
    my_vel = (state.agent.vel_x, state.agent.vel_y)

    blocked = {(p.x, p.y) for p in state.players}
    if my_pos in blocked:
        blocked.remove(my_pos)

    for cap in range(max_cap, 0, -1):
        act = _first_accel_from_kin_astar(
            my_pos,
            my_vel,
            target,
            gmap,
            avoid_hazards=avoid_hazards,
            speed_cap=cap,
            blocked=blocked
        )
        if act is not None:
            return act

    return None


def read_initial_observation() -> Circuit:
    """
    Kommunkication with the judege
    """
    line = sys.stdin.readline()  # Read line
    if not line: return None     # EOF check
    # Parse ints
    H, W, num_players, visibility_radius = map(int, line.split())
    # Return struct
    return Circuit((H, W), num_players, visibility_radius)


def read_observation(old_state: State) -> Optional[State]:
    """
    Körönkénti update olvasás
    """
    line = sys.stdin.readline()  # Read me
    # EOF / End check
    if not line or line.strip() == '~~~END~~~':
        return None

    # Parse my agent data
    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)

    players: List[Player] = []  # Enemy list init
    circuit_data = old_state.circuit  # Ref to static data

    # Read other players
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, sys.stdin.readline().split())  # Read coords
        players.append(Player(pposx, pposy, 0, 0))  # Add to list

    # Lokális map init (mind NOT_VISIBLE)
    visible_track = np.full(
        circuit_data.track_shape,
        CellType.NOT_VISIBLE.value,
        dtype=int
    )

    R = circuit_data.visibility_radius  # Radius
    H, W = circuit_data.track_shape     # Map dims

    # Read vision grid (2R+1 lines)
    for i in range(2 * R + 1):
        raw_line = sys.stdin.readline()  # Read row
        if not raw_line: break  # Safety break
        vals = [int(a) for a in raw_line.split()]  # Parse row

        # Calc global X
        x = posx - R + i
        if x < 0 or x >= H: continue  # Skip if out of bounds

        # Calc global Y range
        y_start = posy - R
        y_end = y_start + 2 * R + 1

        # Clipping logic (ha kilóg a mapről)
        line_slice_start = 0
        line_slice_end = len(vals)
        target_y_start = max(0, y_start)
        target_y_end = min(W, y_end)

        # Balra kilógás fix
        if y_start < 0:
            line_slice_start = -y_start
        # Jobbra kilógás fix
        if y_end > W:
            line_slice_end -= (y_end - W)

        # Copy data to local map
        if line_slice_start < line_slice_end:
            visible_track[x, target_y_start:target_y_end] = vals[
                line_slice_start:line_slice_end
            ]

    # Return updated state
    return old_state._replace(
        visible_track=visible_track, players=players, agent=agent
    )


def choose_next_free_cell_on_path(
    path: List[Tuple[int, int]],
    state: State
) -> Optional[Tuple[int, int]]:
    """
    Útvonalon első üres hely keresése
    """
    if len(path) < 2: return None  # Túl rövid út legalabb a sart pozició és a cel ahova menni akarunk benne kell hogy legyen

    my_pos = (state.agent.x, state.agent.y)  # Saját pos
    # Többiek pozíciói (set)
    occupied = {(p.x, p.y) for p in state.players}

    # Magamat kiveszem
    if my_pos in occupied:
        occupied.remove(my_pos)

    # Path check
    for cell in path[1:]:
        if cell not in occupied:  # Ha üres
            return cell  # Found it

    return None  # Mindenki útban van


def calculate_move_logic(state: State, gmap: GlobalMap) -> Tuple[int, int]:
    """

    fö mozgas logikam

    """
    my_pos = (state.agent.x, state.agent.y)  # My Coords

    # 1. Map update
    gmap.update(state.visible_track)

    # 2. Célkeresés
    goals = np.argwhere(gmap.grid == CellType.GOAL.value)

    # Ha látunk célt
    if len(goals) > 0:
        # Foglalt check
        occupied = {(p.x, p.y) for p in state.players}
        if my_pos in occupied: occupied.remove(my_pos)

        #pot celok
        candidates: List[Tuple[bool, int, Tuple[int, int]]] = []
        for gr, gc in goals:
            cell = (int(gr), int(gc))  # Coords
            dist = abs(gr - my_pos[0]) + abs(gc - my_pos[1])  # Manhattan dist
            is_occ = cell in occupied  # Occupied flag
            candidates.append((is_occ, dist, cell))  # Add to list

        # Sort: 1. Szabad?, 2. Távolság
        candidates.sort(key=lambda t: (t[0], t[1]))
        goal_tuple = candidates[0][2]  # Best goal

        # A* (óvatosan)
        path = run_astar(my_pos, goal_tuple, gmap, avoid_hazards=True)
        # Fallback: mehet homokon is ha el lenne zarva a cel homokkal vagy olajal
        if len(path) < 2:
            path = run_astar(my_pos, goal_tuple, gmap, avoid_hazards=False)

        # Ha van út
        if path:
            # ok itt jön a "menjünk gyorsan ha lehet" rész
            lookahead = max(2, min(len(path) - 1, state.circuit.visibility_radius))
            far_cell = _pick_far_target(path, state, lookahead)

            if far_cell is not None:
                act = _try_fast_step(state, gmap, far_cell, avoid_hazards=True)
                if act is None:
                    act = _try_fast_step(state, gmap, far_cell, avoid_hazards=False)
                if act is not None:
                    return act

            # Következő üres mező
            next_cell = choose_next_free_cell_on_path(path, state)
            if next_cell is not None:
                # PD controller
                return control_to_cell(state, next_cell)

    # 3. Exploration if nincs cél
    # Legközelebbi unknown keresése kerüljök ki a hazardokat
    target = find_nearest_unknown(my_pos, gmap, avoid_hazards=True)
    allow_hazards_for_explore = False

    # Ha nincs tiszta, mehet hazard
    if target is None:
        target = find_nearest_unknown(my_pos, gmap, avoid_hazards=False)
        allow_hazards_for_explore = True  # Flag set

    # Ha van target
    if target is not None:
        # A* tervezés
        path = run_astar(
            my_pos,
            target,
            gmap,
            avoid_hazards=not allow_hazards_for_explore
        )
        # Fallback check
        if len(path) < 2 and allow_hazards_for_explore:
            path = run_astar(my_pos, target, gmap, avoid_hazards=False)

        # Execute path
        if path:
            # ugyanaz a trükk explorationnél is: előre célzunk, hátha egyenes
            lookahead = max(2, min(len(path) - 1, state.circuit.visibility_radius))
            far_cell = _pick_far_target(path, state, lookahead)

            if far_cell is not None:
                act = _try_fast_step(
                    state, gmap, far_cell, avoid_hazards=not allow_hazards_for_explore
                )
                if act is None and allow_hazards_for_explore:
                    act = _try_fast_step(state, gmap, far_cell, avoid_hazards=False)
                if act is not None:
                    return act

            next_cell = choose_next_free_cell_on_path(path, state)
            if next_cell is not None:
                return control_to_cell(state, next_cell)

    # 4. Panic / Idle: Fék
    vel = state.agent.vel  # Current vel
    # Ellenkező irányú gyorsítás (clamp)
    ax = int(np.clip(-vel[0], -1, 1))
    ay = int(np.clip(-vel[1], -1, 1))
    return (ax, ay)  # Fék parancs


def control_to_cell(state: State, next_cell: tuple[int, int]) -> Tuple[int, int]:
    """
    Brute force számítás, megnézzük a 9 opciót, és azt választjuk, ahol nem crashelünk, minden rendben van
    segít simán eljutni A-ból B-be anélkül, hogy túlszaladnál vagy lengedeznél
    """
    desired_pos = np.array(next_cell, dtype=float)  # Target pos
    current_pos = state.agent.pos.astype(float)     # My pos
    current_vel = state.agent.vel.astype(float)     # My vel
    other_positions = [p.pos for p in state.players] # Other players

    best_score = float('inf')  # Init worst score
    best_acc = (0, 0)          # Init acc

    # Brute-force: mind a 9 opció
    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            # Calc new physics
            new_vel = current_vel + np.array([ax, ay], dtype=float)
            new_pos = current_pos + new_vel
            new_pos_int = new_pos.astype(int)

            # Collision check
            if any(np.array_equal(new_pos_int, op) for op in other_positions):
                continue  # Skip

            # Score számítás
            dist = float(np.linalg.norm(desired_pos - new_pos))  # Distance
            speed = float(np.linalg.norm(new_vel))               # Speed

            # Target speed logic (lassíts ha közel)
            desired_speed = min(4.0, dist)
            speed_penalty = abs(speed - desired_speed)

            # Energy saving penalty
            acc_mag = abs(ax) + abs(ay)

            # Weighted score
            score = dist + 0.5 * speed_penalty + 0.1 * acc_mag

            # Min search
            if score < best_score:
                best_score = score
                best_acc = (ax, ay)

    # Ha minden crash-elne -> Vészfék
    if best_score == float('inf'):
        vel = state.agent.vel
        ax = int(np.clip(-vel[0], -1, 1))
        ay = int(np.clip(-vel[1], -1, 1))
        return (ax, ay)

    return best_acc  # Return best move


def main():
    """
    Entry point
    """
    print('READY', flush=True)  # Handshake
    circuit = read_initial_observation()  # Static map with hxw
    if circuit is None: return  # Error exit

    gmap = GlobalMap(circuit.track_shape)  # Map init
    state: Optional[State] = State(circuit, None, [], None) # empty stat, so that while can start

    while True:  # Game loop
        state = read_observation(state)  # Read round
        if state is None: break  # End check

        # Calc logic
        dx, dy = calculate_move_logic(state, gmap)

        # Judgenak küldöm a lépésemet
        print(f'{dx} {dy}', flush=True)


if __name__ == "__main__":
    main()  # Run
