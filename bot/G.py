import sys
import enum
import numpy as np
import heapq
import os
import datetime
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

# --- TÉRKÉP ---
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

# --- ALGORITMUSOK ---

def find_nearest_reachable_target(start: tuple[int, int], gmap: GlobalMap, occupied: Set[tuple[int, int]], ignore_players: bool = False) -> Tuple[Optional[tuple[int, int]], str]:
    """
    Flood Fill keresés (BFS).
    JAVÍTVA: Most már 8 irányba keres (átlósan is)!
    """
    mode_str = "LAZA" if ignore_players else "SZIGORÚ"
    
    queue = deque([start])
    visited = {start}
    
    val_start = gmap.grid[start]
    if val_start == CellType.UNKNOWN.value and (ignore_players or start not in occupied):
        return start, 'UNKNOWN'
    if val_start == CellType.GOAL.value:
        return start, 'GOAL'

    nodes = 0
    while queue:
        cx, cy = queue.popleft()
        nodes += 1
        
        # JAVÍTÁS ITT: 8 irányú szomszédság vizsgálata
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0: continue
                
                nx, ny = cx + dx, cy + dy
                
                # Pályán belül?
                if 0 <= nx < gmap.shape[0] and 0 <= ny < gmap.shape[1]:
                    
                    # 1. FAL
                    if gmap.grid[nx, ny] == CellType.WALL.value:
                        continue
                    
                    if (nx, ny) in visited:
                        continue
                    
                    # 2. JÁTÉKOSOK
                    if not ignore_players and (nx, ny) in occupied:
                        continue

                    # 3. TALÁLAT
                    cell_val = gmap.grid[nx, ny]
                    
                    if cell_val == CellType.UNKNOWN.value:
                        
                        return (nx, ny), 'UNKNOWN'
                    
                    if cell_val == CellType.GOAL.value:
                       
                        return (nx, ny), 'GOAL'

                    visited.add((nx, ny))
                    queue.append((nx, ny))
    
    
    return None, 'NONE'

def run_astar(start: tuple[int, int], goal: tuple[int, int], gmap: GlobalMap, occupied: Set[tuple[int, int]]):
    # Az A* már eddig is 8 irányú volt, ez jó.
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
                
                tile_cost = gmap.get_cost(nx, ny)
                if (nx, ny) in occupied and (nx, ny) != goal:
                    tile_cost += 500 

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

def get_random_valid_move(state: State, gmap: GlobalMap) -> Tuple[int, int]:
    possible_moves = []
    current_vel = state.agent.vel
    current_pos = state.agent.pos
    
    for ax in range(-1, 2):
        for ay in range(-1, 2):
            next_pos_x = current_pos[0] + current_vel[0] + ax
            next_pos_y = current_pos[1] + current_vel[1] + ay
            
            if gmap.is_traversable(int(next_pos_x), int(next_pos_y)):
                if ax != 0 or ay != 0:
                     possible_moves.append((ax, ay))
                else:
                     possible_moves.append((ax, ay))

    if possible_moves:
        idx = np.random.randint(0, len(possible_moves))
        return possible_moves[idx]
    return (0, 0)

# --- FŐ LOGIKA ---

def calculate_move_logic(state: State, gmap: GlobalMap) -> Tuple[int, int]:
    my_pos = (state.agent.x, state.agent.y)
    
    
    
    gmap.update(state.visible_track)
    
    occupied_cells = set()
    for p in state.players:
        if (p.x, p.y) != my_pos:
            occupied_cells.add((p.x, p.y))

    # 1. 8-irányú Flood Fill (Szigorú)
    target, t_type = find_nearest_reachable_target(my_pos, gmap, occupied_cells, ignore_players=False)
    
    # 2. 8-irányú Flood Fill (Laza)
    if target is None:
        
        target, t_type = find_nearest_reachable_target(my_pos, gmap, occupied_cells, ignore_players=True)

    path = []
    if target is not None:
        path = run_astar(my_pos, target, gmap, occupied_cells)
    
    if path and len(path) >= 2:
        next_cell = path[1]
        
        
        desired_pos = np.array(next_cell)
        current_pos = state.agent.pos
        current_vel = state.agent.vel
        needed_vel = desired_pos - current_pos
        acceleration = needed_vel - current_vel
        
        ax = int(np.clip(acceleration[0], -1, 1))
        ay = int(np.clip(acceleration[1], -1, 1))
        return (ax, ay)
    else:
        
        return get_random_valid_move(state, gmap) # panik mód

# --- INPUT/OUTPUT ---

def read_initial_observation() -> Circuit:
    line = sys.stdin.readline()
    if not line: return None
    try:
        parts = list(map(int, line.split()))
        if len(parts) == 4:
            H, W, num_players, visibility_radius = parts
            return Circuit((H, W), num_players, visibility_radius)
    except ValueError:
        return None
    return None

def read_observation(old_state: State) -> Optional[State]:
    line = sys.stdin.readline()
    if not line or line.strip() == '~~~END~~~':
        return None
    
    try:
        posx, posy, velx, vely = map(int, line.split())
        agent = Player(posx, posy, velx, vely)
        
        players = []
        circuit_data = old_state.circuit
        for _ in range(circuit_data.num_players):
            line_p = sys.stdin.readline()
            if not line_p: break
            pposx, pposy = map(int, line_p.split())
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
    except ValueError:
        return None

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