import sys
import enum
import numpy as np
import heapq
from collections import deque
from typing import Optional, NamedTuple, List, Tuple

# --- JÁTÉK KONSTANSOK (a grid_race_env.py alapján) ---
class CellType(enum.Enum):
    WALL = -1
    EMPTY = 0
    START = 1
    UNKNOWN = -2      # Belső jelölés a még nem látott területre
    NOT_VISIBLE = 3   # A bemenetben kapott "nem látható" érték
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

# --- NAVIGÁCIÓS LOGIKA (AGY) ---

class GlobalMap:
    def __init__(self, shape):
        self.shape = shape
        # Kezdetben mindent UNKNOWN-ra állítunk
        self.grid = np.full(shape, CellType.UNKNOWN.value, dtype=int)

    def update(self, visible_track: np.ndarray):
        # A visible_track mérete megegyezik a pályáéval, de tele van 3-asokkal (NOT_VISIBLE),
        # kivéve ott, amit éppen látunk.
        # Frissítjük a gridet ott, ahol a bejövő adat NEM 'NOT_VISIBLE'
        mask = (visible_track != CellType.NOT_VISIBLE.value)
        self.grid[mask] = visible_track[mask]

    def get_cost(self, x, y):
        """Költség a mozgáshoz. Fal végtelen, Homok/Olaj drága."""
        val = self.grid[x, y]
        if val == CellType.WALL.value:
            return float('inf')
        if val == CellType.UNKNOWN.value:
            return 1 # Az ismeretlenbe bátran bemegyünk
        if val == CellType.SAND.value or val == CellType.OIL.value:
            return 20 # Büntetés, de átjárható (kerüljük, ha lehet)
        return 1 # Sima út (EMPTY, START, GOAL)

    def is_traversable(self, x, y):
        if not (0 <= x < self.shape[0] and 0 <= y < self.shape[1]):
            return False
        return self.grid[x, y] != CellType.WALL.value

def find_nearest_unknown(start: tuple[int, int], gmap: GlobalMap) -> Optional[tuple[int, int]]:
    """BFS algoritmus a legközelebbi felfedezetlen (UNKNOWN) pont megtalálására."""
    queue = deque([start])
    visited = {start}
    
    # Ha mi magunk ismeretlenben állnánk (startnál fura lenne), vagy a cél ismeretlen
    if gmap.grid[start] == CellType.UNKNOWN.value:
        return start

    while queue:
        cx, cy = queue.popleft()
        
        # 4 irányú szomszédság
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            
            if 0 <= nx < gmap.shape[0] and 0 <= ny < gmap.shape[1]:
                # Ha fal, nem megyünk arra
                if gmap.grid[nx, ny] == CellType.WALL.value:
                    continue
                
                # HA megtaláltuk az ismeretlent, EZ a cél!
                if gmap.grid[nx, ny] == CellType.UNKNOWN.value:
                    return (nx, ny)
                
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return None

def run_astar(start: tuple[int, int], goal: tuple[int, int], gmap: GlobalMap):
    """A* útvonalkeresés a starttól a goalig, figyelembe véve a terepviszonyokat."""
    # (költség, x, y)
    pq = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while pq:
        _, current = heapq.heappop(pq)
        
        if current == goal:
            break
        
        # 8 irányba léphetünk (átlósan is) a fizika miatt célszerű
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0: continue
                
                nx, ny = current[0] + dx, current[1] + dy
                
                if not gmap.is_traversable(nx, ny):
                    continue
                
                new_cost = cost_so_far[current] + gmap.get_cost(nx, ny)
                
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    # Heurisztika: Manhattan távolság
                    priority = new_cost + (abs(goal[0] - nx) + abs(goal[1] - ny))
                    heapq.heappush(pq, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current
                    
    # Útvonal rekonstrukció
    if goal not in came_from:
        return []
        
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path # [Start, lépés1, lépés2, ... Goal]

# --- KOMMUNIKÁCIÓ ÉS ÁLLAPOTKEZELÉS (A "Lieutenant") ---

def read_initial_observation() -> Circuit:
    # H W num_players visibility_radius
    line = sys.stdin.readline()
    if not line:
        return None
    parts = list(map(int, line.split()))
    H, W, num_players, visibility_radius = parts
    return Circuit((H, W), num_players, visibility_radius)

def read_observation(old_state: State) -> Optional[State]:
    line = sys.stdin.readline()
    if not line or line.strip() == '~~~END~~~':
        return None
    
    # Saját adataink
    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)
    
    players = []
    circuit_data = old_state.circuit
    
    # Többi játékos
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, sys.stdin.readline().split())
        players.append(Player(pposx, pposy, 0, 0))
    
    # Pálya látott része
    # Kezdetben csupa NOT_VISIBLE (3)
    visible_track = np.full(circuit_data.track_shape, CellType.NOT_VISIBLE.value)
    
    for i in range(2 * circuit_data.visibility_radius + 1):
        raw_line = sys.stdin.readline()
        if not raw_line: break
        vals = [int(a) for a in raw_line.split()]
        
        # A grid_race logikája szerint a koordináták számítása:
        x = posx - circuit_data.visibility_radius + i
        if x < 0 or x >= circuit_data.track_shape[0]:
            continue
            
        y_start = posy - circuit_data.visibility_radius
        y_end = y_start + 2 * circuit_data.visibility_radius + 1
        
        # Vágás, ha kilóg a pályáról vízszintesen
        line_slice_start = 0
        line_slice_end = len(vals)
        
        target_y_start = max(0, y_start)
        target_y_end = min(circuit_data.track_shape[1], y_end)
        
        if y_start < 0:
            line_slice_start = -y_start
        if y_end > circuit_data.track_shape[1]:
            line_slice_end -= (y_end - circuit_data.track_shape[1])
            
        if line_slice_start < line_slice_end:
            visible_track[x, target_y_start:target_y_end] = vals[line_slice_start:line_slice_end]

    return old_state._replace(
        visible_track=visible_track, players=players, agent=agent)

def calculate_move_logic(state: State, gmap: GlobalMap) -> Tuple[int, int]:
    """Itt történik a döntéshozatal."""
    my_pos = (state.agent.x, state.agent.y)
    
    # 1. Frissítjük a világtérképet
    gmap.update(state.visible_track)
    
    # 2. Megkeressük a legközelebbi ismeretlen mezőt (Subgoal)
    target = find_nearest_unknown(my_pos, gmap)
    
    if target is None:
        # Ha nincs ismeretlen, keressük meg a CÉLT (GOAL = 100)
        # Ez egy egyszerű szkennelés a térképen
        goals = np.argwhere(gmap.grid == CellType.GOAL.value)
        if len(goals) > 0:
            # Legközelebbi célpont
            dists = np.sum(np.abs(goals - np.array(my_pos)), axis=1)
            target = tuple(goals[np.argmin(dists)])
        else:
            # Ha nincs cél és minden felderítve -> véletlen
            return (0, 0)

    # 3. Útvonaltervezés A*-gal a targethez
    path = run_astar(my_pos, target, gmap)
    
    if len(path) < 2:
        # Már ott vagyunk, vagy nem találtunk utat -> lassítsunk
        return (0, 0)
    
    # A path[0] a mostani pozíció, path[1] a következő lépés
    next_cell = path[1]
    
    # 4. Fizikai vezérlés (PD controller)
    # Hová akarunk menni?
    desired_pos = np.array(next_cell)
    current_pos = state.agent.pos
    current_vel = state.agent.vel
    
    # Milyen sebesség kell, hogy odaérjünk? (cél - jelenlegi hely)
    needed_vel = desired_pos - current_pos
    
    # Mekkora gyorsítás kell ehhez? (kell - van)
    acceleration = needed_vel - current_vel
    
    # A gyorsítás korlátozása -1, 0, 1 közé
    ax = int(np.clip(acceleration[0], -1, 1))
    ay = int(np.clip(acceleration[1], -1, 1))
    
    return (ax, ay)

def main():
    # Kézfogás a szerverrel (ha szükséges lenne), de a judge.py stdin-t figyel
    print('READY', flush=True)
    
    circuit = read_initial_observation()
    if circuit is None:
        return
        
    # Inicializáljuk a globális térképet
    gmap = GlobalMap(circuit.track_shape)
    
    # Kezdő állapot
    state: Optional[State] = State(circuit, None, [], None) # type: ignore
    
    while True:
        # Olvasás
        state = read_observation(state)
        if state is None:
            break
            
        # Gondolkodás
        dx, dy = calculate_move_logic(state, gmap)
        
        # Válasz küldése
        print(f'{dx} {dy}', flush=True)

if __name__ == "__main__":
    main()