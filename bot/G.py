import sys
import enum
import numpy as np
import heapq
from collections import deque
from typing import Optional, NamedTuple, List, Tuple, Set

# --- JÁTÉK KONSTANSOK ---
class CellType(enum.Enum):
    WALL = -1
    EMPTY = 0
    START = 1
    UNKNOWN = -2
    NOT_VISIBLE = 3
    OIL = 91
    SAND = 92
    GOAL = 100

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

# --- NAVIGÁCIÓS LOGIKA ---

class GlobalMap:
    def __init__(self, shape):
        self.shape = shape
        self.grid = np.full(shape, CellType.UNKNOWN.value, dtype=int)

    def update(self, visible_track: np.ndarray):
        mask = (visible_track != CellType.NOT_VISIBLE.value)
        self.grid[mask] = visible_track[mask]

    def get_cost(self, x, y):
        val = self.grid[x, y]
        if val == CellType.WALL.value:
            return float('inf')
        if val == CellType.UNKNOWN.value:
            return 1
        if val == CellType.SAND.value or val == CellType.OIL.value:
            return 20 
        return 1

    def is_traversable(self, x, y):
        if not (0 <= x < self.shape[0] and 0 <= y < self.shape[1]):
            return False
        return self.grid[x, y] != CellType.WALL.value

# MÓDOSÍTVA: Most már megkapja az occupied (foglalt) mezők halmazát is
def find_nearest_unknown(start: tuple[int, int], gmap: GlobalMap, occupied: Set[tuple[int, int]]) -> Optional[tuple[int, int]]:
    """BFS a legközelebbi SZABAD ismeretlen mezőre."""
    queue = deque([start])
    visited = {start}
    
    # Ha épp ismeretlenben állunk, és nem vagyunk blokkolva, maradhatunk
    if gmap.grid[start] == CellType.UNKNOWN.value and start not in occupied:
        return start

    while queue:
        cx, cy = queue.popleft()
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            
            if 0 <= nx < gmap.shape[0] and 0 <= ny < gmap.shape[1]:
                if gmap.grid[nx, ny] == CellType.WALL.value:
                    continue
                
                # ÚJ: Ellenőrzés - ha ismeretlen, DE foglalt, akkor nem jó cél!
                if gmap.grid[nx, ny] == CellType.UNKNOWN.value:
                    if (nx, ny) not in occupied:
                        return (nx, ny)
                    # Ha foglalt, tovább keresünk (mintha fal lenne, vagy csak simán nem cél)
                
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return None

# MÓDOSÍTVA: A* is figyelembe veszi a többi játékost akadályként
def run_astar(start: tuple[int, int], goal: tuple[int, int], gmap: GlobalMap, occupied: Set[tuple[int, int]]):
    pq = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while pq:
        _, current = heapq.heappop(pq)
        
        if current == goal:
            break
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0: continue
                
                nx, ny = current[0] + dx, current[1] + dy
                
                if not gmap.is_traversable(nx, ny):
                    continue
                
                # ÚJ: Dinamikus akadály elkerülés
                # Ha a mezőn áll valaki (és nem az a célunk), akkor kerüljük el
                # Nagyon nagy költséget adunk neki, így csak végszükség esetén megy arra
                tile_cost = gmap.get_cost(nx, ny)
                if (nx, ny) in occupied and (nx, ny) != goal:
                    tile_cost += 1000 # "Lágy" fal: átmehetünk, ha nincs más, de inkább kerüljük

                new_cost = cost_so_far[current] + tile_cost
                
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost + (abs(goal[0] - nx) + abs(goal[1] - ny))
                    heapq.heappush(pq, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current
                    
    if goal not in came_from:
        return []
        
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path

# --- KOMMUNIKÁCIÓ ÉS MAIN ---

def read_initial_observation() -> Circuit:
    line = sys.stdin.readline()
    if not line: return None
    H, W, num_players, visibility_radius = map(int, line.split())
    return Circuit((H, W), num_players, visibility_radius)

def read_observation(old_state: State) -> Optional[State]:
    line = sys.stdin.readline()
    if not line or line.strip() == '~~~END~~~':
        return None
    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)
    
    players = []
    circuit_data = old_state.circuit
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, sys.stdin.readline().split())
        players.append(Player(pposx, pposy, 0, 0))
    
    visible_track = np.full(circuit_data.track_shape, CellType.NOT_VISIBLE.value)
    for i in range(2 * circuit_data.visibility_radius + 1):
        raw = sys.stdin.readline()
        if not raw: break
        vals = [int(a) for a in raw.split()]
        
        x = posx - circuit_data.visibility_radius + i
        if x < 0 or x >= circuit_data.track_shape[0]: continue
        y_start = posy - circuit_data.visibility_radius
        y_end = y_start + 2 * circuit_data.visibility_radius + 1
        
        sl_start, sl_end = 0, len(vals)
        tgt_y_start, tgt_y_end = max(0, y_start), min(circuit_data.track_shape[1], y_end)
        
        if y_start < 0: sl_start = -y_start
        if y_end > circuit_data.track_shape[1]: sl_end -= (y_end - circuit_data.track_shape[1])
            
        if sl_start < sl_end:
            visible_track[x, tgt_y_start:tgt_y_end] = vals[sl_start:sl_end]

    return old_state._replace(visible_track=visible_track, players=players, agent=agent)

def get_alternative_goal(target: Tuple[int, int], gmap: GlobalMap, occupied: Set[Tuple[int, int]]) -> Tuple[int, int]:
    """Ha a cél foglalt, keresünk egy szomszédos szabad mezőt."""
    if target not in occupied:
        return target
    
    # Körbenézünk a cél körül
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx, ny = target[0] + dx, target[1] + dy
            if gmap.is_traversable(nx, ny) and (nx, ny) not in occupied:
                return (nx, ny)
    return target # Ha minden kötél szakad, marad az eredeti

def calculate_move_logic(state: State, gmap: GlobalMap) -> Tuple[int, int]:
    my_pos = (state.agent.x, state.agent.y)
    
    # ÚJ: Összeszedjük, hol vannak a többiek
    occupied_cells = set()
    for p in state.players:
        # A players listában mi NEM vagyunk benne elvileg (GridRace függő), 
        # de ha benne lennénk, magunkat ki kell szűrni.
        # A judge.py alapján a players listában mindenki benne lehet, VAGY a circuit.players-ből jön.
        # Biztonság kedvéért: ha a koordináta nem egyezik a miénkkel, akkor foglalt.
        if (p.x, p.y) != my_pos:
            occupied_cells.add((p.x, p.y))

    gmap.update(state.visible_track)
    
    # 1. Célkeresés (figyelembe véve a foglalt mezőket)
    target = find_nearest_unknown(my_pos, gmap, occupied_cells)
    
    if target is None:
        # Ha nincs ismeretlen, keressük a GOAL-t
        goals = np.argwhere(gmap.grid == CellType.GOAL.value)
        if len(goals) > 0:
            # Legközelebbi cél
            dists = np.sum(np.abs(goals - np.array(my_pos)), axis=1)
            raw_target = tuple(goals[np.argmin(dists)])
            
            # ÚJ: Ha a konkrét CÉLMEZŐN áll valaki, keressünk mellé alternatívát
            target = get_alternative_goal(raw_target, gmap, occupied_cells)
        else:
            return (0, 0)

    # 2. Útvonaltervezés (occupied átadva az A*-nak)
    path = run_astar(my_pos, target, gmap, occupied_cells)
    
    if len(path) < 2:
        return (0, 0)
    
    next_cell = path[1]
    
    # 3. Fizika
    desired_pos = np.array(next_cell)
    current_pos = state.agent.pos
    current_vel = state.agent.vel
    
    needed_vel = desired_pos - current_pos
    acceleration = needed_vel - current_vel
    
    ax = int(np.clip(acceleration[0], -1, 1))
    ay = int(np.clip(acceleration[1], -1, 1))
    
    return (ax, ay)

def main():
    print('READY', flush=True)
    circuit = read_initial_observation()
    if circuit is None: return
    
    gmap = GlobalMap(circuit.track_shape)
    state: Optional[State] = State(circuit, None, [], None) # type: ignore
    
    while True:
        state = read_observation(state)
        if state is None: break
        
        dx, dy = calculate_move_logic(state, gmap)
        print(f'{dx} {dy}', flush=True)

if __name__ == "__main__":
    main()