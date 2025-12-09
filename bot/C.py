import sys
import enum
import numpy as np
import heapq
from collections import deque
from typing import Optional, NamedTuple, List, Tuple

# ────────────────────────────────────────────────────────────────────────────────
# Típusok & Konstansok
# ────────────────────────────────────────────────────────────────────────────────

class CellType(enum.Enum):
    """
    Pálya elemek kódjai
    """
    WALL = -1           # Fal, blokkol
    EMPTY = 0           # Üres, járható
    START = 1           # Start
    UNKNOWN = -2        # Ismeretlen (fog-of-war)
    NOT_VISIBLE = 3     # Épp nem látszik
    OIL = 91            # Olaj (csúszik)
    SAND = 92           # Homok (lassít)
    GOAL = 100          # Cél

# Veszélyes mezők halmaza (gyors check)
HAZARDS = {CellType.OIL.value, CellType.SAND.value}


class Player(NamedTuple):
    """
    Player struct (pos + vel)
    """
    x: int
    y: int
    vel_x: int
    vel_y: int

    # Helper: pos -> numpy array
    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    # Helper: vel -> numpy array
    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vel_x, self.vel_y])


class Circuit(NamedTuple):
    """
    Static pálya infók (csak 1x jön)
    """
    track_shape: tuple[int, int]  # Méret
    num_players: int              # Player count
    visibility_radius: int        # Látótáv


class State(NamedTuple):
    """
    Aktuális game state (körönként frissül)
    """
    circuit: Circuit
    visible_track: np.ndarray     # Lokális view
    players: list[Player]         # Többiek
    agent: Player                 # Én


# ────────────────────────────────────────────────────────────────────────────────
# TÉRKÉP / MAP
# ────────────────────────────────────────────────────────────────────────────────

class GlobalMap:
    """
    Full map tárolása. Elején csupa UNKNOWN.
    """

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        # Init: minden -2 (unknown)
        self.grid = np.full(shape, CellType.UNKNOWN.value, dtype=int)

    def update(self, visible_track: np.ndarray):
        """
        Map update a látható résszel
        """
        # Maszk: ami nem NOT_VISIBLE (3), azt beírjuk
        mask = (visible_track != CellType.NOT_VISIBLE.value)
        self.grid[mask] = visible_track[mask]


# ────────────────────────────────────────────────────────────────────────────────
# FELDERÍTÉS (BFS)
# ────────────────────────────────────────────────────────────────────────────────

def find_nearest_unknown(
    start: tuple[int, int],
    gmap: GlobalMap,
    avoid_hazards: bool = True
) -> Optional[tuple[int, int]]:
    """
    BFS -> legközelebbi UNKNOWN mező keresése.
    avoid_hazards: kerülje-e az olajat/homokot
    """
    queue = deque([start])
    visited = {start}

    # Edge case: ha pont ismeretlenben állunk
    if gmap.grid[start] == CellType.UNKNOWN.value:
        return start

    H, W = gmap.shape

    # 8 irány
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            
            # Map bounds check
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            
            # Már volt?
            if (nx, ny) in visited:
                continue

            val = gmap.grid[nx, ny]

            # Falba ne menjünk
            if val == CellType.WALL.value:
                continue

            # Hazard check
            if avoid_hazards and val in HAZARDS:
                continue

            # Találat: ismeretlen mező
            if val == CellType.UNKNOWN.value:
                return (nx, ny)

            # További keresés
            visited.add((nx, ny))
            queue.append((nx, ny))

    return None  # Nincs elérhető ismeretlen


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
    Standard A* implementáció.
    Costs: Sima=1, Hazard=20
    """
    H, W = gmap.shape
    (sx, sy) = start
    (gx, gy) = goal

    # Validálások (start/cél pályán belül?)
    if not (0 <= sx < H and 0 <= sy < W): return []
    if not (0 <= gx < H and 0 <= gy < W): return []
    # Cél nem lehet falban
    if gmap.grid[gx, gy] == CellType.WALL.value: return []

    # Priority Queue: (cost, pos)
    pq: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
    came_from: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    cost_so_far: dict[Tuple[int, int], float] = {start: 0.0}

    # 8 irány
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while pq:
        _, current = heapq.heappop(pq)
        if current == goal:
            break # Megérkeztünk

        cx, cy = current

        for dx, dy in directions:
            if dx == 0 and dy == 0: continue
            
            nx, ny = cx + dx, cy + dy
            # Bounds check
            if not (0 <= nx < H and 0 <= ny < W): continue

            val = gmap.grid[nx, ny]

            if val == CellType.WALL.value: continue
            # Hazard szűrés ha kell
            if avoid_hazards and val in HAZARDS: continue

            # Költség számolás (hazard drága)
            step_cost = 20.0 if val in HAZARDS else 1.0

            new_cost = cost_so_far[current] + step_cost

            # Ha jobb utat találtunk vagy új a node
            if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                cost_so_far[(nx, ny)] = new_cost
                # Heurisztika: Manhattan dist
                priority = new_cost + abs(gx - nx) + abs(gy - ny)
                heapq.heappush(pq, (priority, (nx, ny)))
                came_from[(nx, ny)] = current

    if goal not in came_from:
        return [] # Nincs út

    # Útvonal rekonstrukció visszafelé
    path: List[Tuple[int, int]] = []
    curr: Optional[Tuple[int, int]] = goal
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path


# ────────────────────────────────────────────────────────────────────────────────
# IO / KOMMUNIKÁCIÓ
# ────────────────────────────────────────────────────────────────────────────────

def read_initial_observation() -> Circuit:
    """
    Init sor olvasása (méretek, player count)
    """
    line = sys.stdin.readline()
    if not line:
        return None # EOF / Error
    H, W, num_players, visibility_radius = map(int, line.split())
    return Circuit((H, W), num_players, visibility_radius)


def read_observation(old_state: State) -> Optional[State]:
    """
    Körönkénti state olvasás.
    Látómező összeállítása slice-okból.
    """
    line = sys.stdin.readline()
    # Check EOF or END signal
    if not line or line.strip() == '~~~END~~~':
        return None

    # Saját adatok
    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)

    players: List[Player] = []
    circuit_data = old_state.circuit

    # Többiek pozíciója
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, sys.stdin.readline().split())
        players.append(Player(pposx, pposy, 0, 0))

    # Lokális térkép init (NOT_VISIBLE)
    visible_track = np.full(
        circuit_data.track_shape,
        CellType.NOT_VISIBLE.value,
        dtype=int
    )

    R = circuit_data.visibility_radius
    H, W = circuit_data.track_shape

    # Látott sorok parse-olása (2R+1 db)
    for i in range(2 * R + 1):
        raw_line = sys.stdin.readline()
        if not raw_line: break
        vals = [int(a) for a in raw_line.split()]

        # Globális X számolás
        x = posx - R + i
        if x < 0 or x >= H: continue # Pályán kívül

        # Globális Y tartomány
        y_start = posy - R
        y_end = y_start + 2 * R + 1

        # Clipping logika (ha kilóg a látómező)
        line_slice_start = 0
        line_slice_end = len(vals)
        target_y_start = max(0, y_start)
        target_y_end = min(W, y_end)

        # Balra kilóg
        if y_start < 0:
            line_slice_start = -y_start
        # Jobbra kilóg
        if y_end > W:
            line_slice_end -= (y_end - W)

        # Copy data
        if line_slice_start < line_slice_end:
            visible_track[x, target_y_start:target_y_end] = vals[
                line_slice_start:line_slice_end
            ]

    return old_state._replace(
        visible_track=visible_track, players=players, agent=agent
    )


# ────────────────────────────────────────────────────────────────────────────────
# LOGIKA UTÓMUNKA
# ────────────────────────────────────────────────────────────────────────────────

def choose_next_free_cell_on_path(
    path: List[Tuple[int, int]],
    state: State
) -> Optional[Tuple[int, int]]:
    """
    Útvonalon első SZABAD hely keresése.
    Hogy ne menjünk neki másnak.
    """
    if len(path) < 2:
        return None

    my_pos = (state.agent.x, state.agent.y)
    # Többiek pozíciói set-ben
    occupied = {(p.x, p.y) for p in state.players}
    
    # Magunkat kivesszük
    if my_pos in occupied:
        occupied.remove(my_pos)

    # Path bejárása, check occupancy
    for cell in path[1:]:
        if cell not in occupied:
            return cell

    return None # Minden foglalt


# ────────────────────────────────────────────────────────────────────────────────
# MAIN LOGIKA
# ────────────────────────────────────────────────────────────────────────────────

def calculate_move_logic(state: State, gmap: GlobalMap) -> Tuple[int, int]:
    """
    Döntéshozatal (Brain).
    """
    my_pos = (state.agent.x, state.agent.y)

    # 1. Global map update
    gmap.update(state.visible_track)

    # 2. Célkeresés (GOAL)
    goals = np.argwhere(gmap.grid == CellType.GOAL.value)
    
    # Van cél?
    if len(goals) > 0:
        # Foglalt helyek szűrése
        occupied = {(p.x, p.y) for p in state.players}
        if my_pos in occupied: occupied.remove(my_pos)

        # Cél jelöltek listázása
        candidates: List[Tuple[bool, int, Tuple[int, int]]] = []
        for gr, gc in goals:
            cell = (int(gr), int(gc))
            dist = abs(gr - my_pos[0]) + abs(gc - my_pos[1])
            is_occ = cell in occupied
            candidates.append((is_occ, dist, cell))

        # Sort: 1. Szabad?, 2. Távolság
        candidates.sort(key=lambda t: (t[0], t[1]))
        goal_tuple = candidates[0][2]

        # A* (hazard kerüléssel)
        path = run_astar(my_pos, goal_tuple, gmap, avoid_hazards=True)
        # Fallback: hazard-on keresztül
        if len(path) < 2:
            path = run_astar(my_pos, goal_tuple, gmap, avoid_hazards=False)

        if path:
            next_cell = choose_next_free_cell_on_path(path, state)
            if next_cell is not None:
                # PD controller hívás
                return _pd_control_to_cell(state, next_cell)

    # 3. Exploration (nincs látható cél)
    # Legközelebbi UNKNOWN keresése
    target = find_nearest_unknown(my_pos, gmap, avoid_hazards=True)
    allow_hazards_for_explore = False

    # Ha nincs tiszta út, mehet hazard-on át is
    if target is None:
        target = find_nearest_unknown(my_pos, gmap, avoid_hazards=False)
        allow_hazards_for_explore = True

    if target is not None:
        # Útvonalterv a felfedezetlenhez
        path = run_astar(
            my_pos,
            target,
            gmap,
            avoid_hazards=not allow_hazards_for_explore
        )
        # Fallback check
        if len(path) < 2 and allow_hazards_for_explore:
            path = run_astar(my_pos, target, gmap, avoid_hazards=False)

        if path:
            next_cell = choose_next_free_cell_on_path(path, state)
            if next_cell is not None:
                return _pd_control_to_cell(state, next_cell)

    # 4. Panic mode: fék
    vel = state.agent.vel
    # Ellenkező irányú gyorsítás (clampelt)
    ax = int(np.clip(-vel[0], -1, 1))
    ay = int(np.clip(-vel[1], -1, 1))
    return (ax, ay)


def _pd_control_to_cell(state: State, next_cell: tuple[int, int]) -> Tuple[int, int]:
    """
    PD vezérlő. Kiszámolja a legjobb (ax, ay)-t.
    """
    desired_pos = np.array(next_cell, dtype=float)
    current_pos = state.agent.pos.astype(float)
    current_vel = state.agent.vel.astype(float)
    other_positions = [p.pos for p in state.players]

    best_score = float('inf')
    best_acc = (0, 0)

    # Brute-force: mind a 9 opció check (-1, 0, 1)
    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            new_vel = current_vel + np.array([ax, ay], dtype=float)
            new_pos = current_pos + new_vel
            new_pos_int = new_pos.astype(int)

            # Collision check
            if any(np.array_equal(new_pos_int, op) for op in other_positions):
                continue

            # Pontozás (táv + sebesség)
            dist = float(np.linalg.norm(desired_pos - new_pos))
            speed = float(np.linalg.norm(new_vel))
            
            # Target speed logic
            desired_speed = min(4.0, dist)
            speed_penalty = abs(speed - desired_speed)
            
            # Kis büntetés a nagy gyorsításért (energia)
            acc_mag = abs(ax) + abs(ay)
            
            score = dist + 0.5 * speed_penalty + 0.1 * acc_mag

            if score < best_score:
                best_score = score
                best_acc = (ax, ay)

    # Ha minden rossz (collision), vészfék
    if best_score == float('inf'):
        vel = state.agent.vel
        ax = int(np.clip(-vel[0], -1, 1))
        ay = int(np.clip(-vel[1], -1, 1))
        return (ax, ay)

    return best_acc


# ────────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────────

def main():
    """
    Entry point
    """
    print('READY', flush=True) # Handshake
    circuit = read_initial_observation() # Map infók
    if circuit is None:
        return

    gmap = GlobalMap(circuit.track_shape)
    state: Optional[State] = State(circuit, None, [], None) # type: ignore

    while True: # Game loop
        state = read_observation(state) # Új adatok
        if state is None: # Vége
            break
        
        # Logic hívás
        dx, dy = calculate_move_logic(state, gmap)
        
        # Output
        print(f'{dx} {dy}', flush=True)


if __name__ == "__main__":
    main()