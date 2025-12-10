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
            visited.add((nx, ny))  # Mark visited
            queue.append((nx, ny)) # Push queue

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
        _, current = heapq.heappop(pq)  # Get best _, azt jelenti hogy többet nem kell igy szoktak jelölni
        if current == goal:  # Célba értünk
            break

        cx, cy = current  # Current coords

        for dx, dy in directions:  # dx,dy delta azaz változás
            if dx == 0 and dy == 0: continue  # Skip self
            
            nx, ny = cx + dx, cy + dy  # nx,ny Next coords
            # Bounds check palya hatarokon ne menjünk ki
            if not (0 <= nx < H and 0 <= ny < W): continue

            val = gmap.grid[nx, ny]  # Cella típusa

            if val == CellType.WALL.value: continue  # Skip wall
            # Hazard skip ha kell
            if avoid_hazards and val in HAZARDS: continue

            # Cost calc (hazard = 20, sima = 1)
            step_cost = 20.0 if val in HAZARDS else 1.0

            new_cost = cost_so_far[current] + step_cost  # Új G-score költség a starttól eddig a mezöig ha jobb akkor ez lehet az uj node lasd lenntebb
            

            # Ha jobb utat találtunk vagy új a node
            if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                cost_so_far[(nx, ny)] = new_cost  # Update cost
                # Heurisztika (Manhattan)
                priority = new_cost + abs(gx - nx) + abs(gy - ny)
                heapq.heappush(pq, (priority, (nx, ny)))  # Push PQ
                came_from[(nx, ny)] = current  # Save parent

    if goal not in came_from:  # Ha nem találtuk meg
        return []

    # Path rekonstrukció visszafelé a celtol, ha megvan, eger sajt pelda vissza a celtol jobb otlet
    path: List[Tuple[int, int]] = []
    curr: Optional[Tuple[int, int]] = goal
    while curr is not None:  # Backtrack startig
        path.append(curr)
        curr = came_from[curr]
    path.reverse()  # Fordítás
    return path


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
        candidates: List[Tuple[bool, int, Tuple[int, int]]] = [] #(True,  2,  (10, 12)),  # A közeli, de foglalt
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