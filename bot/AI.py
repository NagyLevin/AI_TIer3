import sys
import enum
import math
import os
from collections import defaultdict, deque
from typing import Optional, NamedTuple, Tuple, List, Dict, Set

import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Basic types (judge-compatible)
# ────────────────────────────────────────────────────────────────────────────────

class CellType(enum.Enum):
    GOAL = 100
    START = 1
    WALL = -1
    UNKNOWN = 2
    EMPTY = 0
    NOT_VISIBLE = 3

class Player(NamedTuple):
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
    track_shape: tuple[int, int]
    num_players: int
    visibility_radius: int

class State(NamedTuple):
    circuit: Circuit
    visible_track: Optional[np.ndarray]    # safety map: NOT_VISIBLE -> WALL
    visible_raw: Optional[np.ndarray]      # raw window with NOT_VISIBLE intact
    players: List[Player]
    agent: Optional[Player]

# ────────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────────

def read_initial_observation() -> Circuit:
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)

def read_observation(old_state: State) -> Optional[State]:
    line = input()
    if line == '~~~END~~~':
        return None

    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)
    circuit = old_state.circuit

    players: List[Player] = []
    for _ in range(circuit.num_players):
        pposx, pposy = map(int, input().split())
        players.append(Player(pposx, pposy, 0, 0))

    H, W = circuit.track_shape
    R = circuit.visibility_radius
    visible_raw = np.full((H, W), CellType.NOT_VISIBLE.value, dtype=int)
    visible_track = np.full((H, W), CellType.WALL.value, dtype=int)

    for i in range(2 * R + 1):
        row_vals = [int(a) for a in input().split()]
        x = posx - R + i
        if 0 <= x < H:
            y_start = posy - R
            y_end   = posy + R + 1
            loc = row_vals
            ys = y_start
            if y_start < 0:
                loc = loc[-y_start:]
                ys = 0
            if y_end > W:
                loc = loc[:-(y_end - W)]
            ye = ys + len(loc)
            if ys < ye:
                visible_raw[x, ys:ye] = loc
                safety = [CellType.WALL.value if v == CellType.NOT_VISIBLE.value else v for v in loc]
                visible_track[x, ys:ye] = safety

    return old_state._replace(visible_track=visible_track,
                              visible_raw=visible_raw,
                              players=players, agent=agent)

# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────

DIRS_4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIRS_8 = [
    (-1, 0), (0, 1), (1, 0), (0, -1),
    (-1, -1), (-1, 1), (1, 1), (1, -1)
]

def tri(n: int) -> int:
    return n*(n+1)//2

def brakingOk(vx: int, vy: int, rSafe: int) -> bool:
    return (tri(abs(vx)) <= rSafe) and (tri(abs(vy)) <= rSafe)

def validLineLocal(state: State, p1: np.ndarray, p2: np.ndarray) -> bool:
    track = state.visible_track
    if track is None: return False
    H, W = track.shape
    if (np.any(p1 < 0) or np.any(p2 < 0) or
        p1[0] >= H or p1[1] >= W or p2[0] >= H or p2[1] >= W):
        return False
    diff = p2 - p1
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))
        for i in range(abs(diff[0]) + 1):
            x = int(p1[0] + i*d)
            y = p1[1] + i*slope*d
            yC = int(np.ceil(y)); yF = int(np.floor(y))
            if (track[x, yC] < 0 and track[x, yF] < 0): return False
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))
        for i in range(abs(diff[1]) + 1):
            x = p1[0] + i*slope*d
            y = int(p1[1] + i*d)
            xC = int(np.ceil(x)); xF = int(np.floor(x))
            if (track[xC, y] < 0 and track[xF, y] < 0): return False
    return True

def find_reachable_zero(state: State, world: 'WorldModel', start_pos: np.ndarray) -> bool:
    if state.agent is None: return False
    
    q = deque([(int(start_pos[0]), int(start_pos[1]))])
    visited = set([(int(start_pos[0]), int(start_pos[1]))])
    
    H, W = world.shape
    R = state.circuit.visibility_radius
    ax, ay = int(state.agent.x), int(state.agent.y)
    
    min_x, max_x = max(0, ax - R), min(H, ax + R + 1)
    min_y, max_y = max(0, ay - R), min(W, ay + R + 1)

    while q:
        x, y = q.popleft()
        if (world.known_map[x, y] == CellType.EMPTY.value and
            world.visited_count[x, y] == 0):
            return True

        for dx, dy in DIRS_8:
            nx, ny = x + dx, y + dy
            if (min_x <= nx < max_x and min_y <= ny < max_y and
                (nx, ny) not in visited):
                if world.traversable(nx, ny):
                    visited.add((nx, ny))
                    q.append((nx, ny))
    return False

# ────────────────────────────────────────────────────────────────────────────────
# World model + hazard tracking
# ────────────────────────────────────────────────────────────────────────────────

def is_traversable_val(v: int) -> bool:
    # UNKNOWN-ból nem tervezünk utat, csak amit már láttunk (>=0, nem UNKNOWN)
    return (v >= 0) and (v != CellType.UNKNOWN.value)

# minden nem klasszikus pozitív tile hazard/new
def is_hazard_val(v: int) -> bool:
    if v < 0:
        return False
    if v in (
        CellType.EMPTY.value,
        CellType.START.value,
        CellType.GOAL.value,
        CellType.UNKNOWN.value,
        CellType.NOT_VISIBLE.value
    ):
        return False
    return True

class WorldModel:
    def __init__(self, shape: tuple[int, int]) -> None:
        H, W = shape
        self.shape = shape
        self.known_map = np.full((H, W), CellType.UNKNOWN.value, dtype=int)
        self.visited_count = np.zeros((H, W), dtype=int)
        self.last_pos: Optional[Tuple[int,int]] = None

        # hazard típus-nyilvántartás
        self.hazard_val = np.full((H, W), -1, dtype=int)  # -1 = nem hazard
        self.hazard_char_map: Dict[int, str] = {}
        self.hazard_char_pool = list("BCDFHJKLMPQRTUVWXYZ")

        self.turn = 0
        self.dump_file = "logs/map_dump.txt"
        self._dump_initialized = False

    def updateWithObservation(self, st: State) -> None:
        if st.visible_raw is None: return
        raw = st.visible_raw
        seen = (raw != CellType.NOT_VISIBLE.value)
        self.known_map[seen] = raw[seen]

        hazard_mask = np.vectorize(is_hazard_val)(self.known_map)
        self.hazard_val[:, :] = -1
        self.hazard_val[hazard_mask] = self.known_map[hazard_mask]

    def traversable(self, x: int, y: int) -> bool:
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W): return False
        return is_traversable_val(self.known_map[x, y])

    def is_hazard(self, x: int, y: int) -> bool:
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W):
            return False
        return int(self.hazard_val[x, y]) != -1

    def get_hazard_value(self, x: int, y: int) -> Optional[int]:
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W):
            return None
        v = int(self.hazard_val[x, y])
        return v if v != -1 else None

    def get_hazard_char(self, tile_value: int) -> str:
        ch = self.hazard_char_map.get(tile_value)
        if ch is not None:
            return ch
        if self.hazard_char_pool:
            ch = self.hazard_char_pool.pop(0)
        else:
            ch = 'X'
        self.hazard_char_map[tile_value] = ch
        return ch

# ────────────────────────────────────────────────────────────────────────────────
# Explorer policy: GOAL-prioritás + frontier-alapú feltérképezés
# ────────────────────────────────────────────────────────────────────────────────

def left_of(d: Tuple[int,int]) -> Tuple[int,int]:
    dx, dy = d
    return (-dy, dx)

def right_of(d: Tuple[int,int]) -> Tuple[int,int]:
    dx, dy = d
    return (dy, -dx)

def back_of(d: Tuple[int,int]) -> Tuple[int,int]:
    dx, dy = d
    return (-dx, -dy)

class LeftWallPolicy:
    """
    GOAL-prioritásos, frontier-alapú explorer (a név csak kompatibilitásból maradt).

    Prioritási sorrend:
    0) Ha van ismert, elérhető GOAL (100-as cella), arra megyünk (legrövidebb út BFS-sel).
    1) Ha nincs GOAL, akkor BFS-sel frontier-t keresünk (ismeretlen szomszéddal rendelkező cella).
    2) Ha frontier sincs, akkor lokálisan a legkevésbé látogatott, nem-hazard szomszéd felé megyünk.
    """
    def __init__(self, world: WorldModel) -> None:
        self.world = world
        self.heading: Tuple[int,int] = (0, 1)    # default EAST

    def _is_wall_local(self, state: State, x: int, y: int) -> bool:
        if state.visible_raw is None: return False
        H, W = state.visible_raw.shape
        if not (0 <= x < H and 0 <= y < W): return False
        return state.visible_raw[x, y] == CellType.WALL.value

    def _is_free_local(self, state: State, x: int, y: int) -> bool:
        if state.visible_track is None: return False
        H, W = state.visible_track.shape
        if not (0 <= x < H and 0 <= y < W): return False
        return state.visible_track[x, y] >= 0

    def _is_hazard(self, x: int, y: int) -> bool:
        return self.world.is_hazard(x, y)

    def _get_visit_count(self, x: int, y: int) -> float:
        H, W = self.world.shape
        if not (0 <= x < H and 0 <= y < W):
            return float('inf')
        return float(self.world.visited_count[x, y])

    def _ensure_heading(self, state: State) -> None:
        if state.agent is None: return
        vx, vy = int(state.agent.vel_x), int(state.agent.vel_y)
        if vx == 0 and vy == 0:
            return
        if abs(vx) >= abs(vy):
            self.heading = (int(np.sign(vx)), 0)
        else:
            self.heading = (0, int(np.sign(vy)))

    # 0) GOAL-BFS
    def _find_goal_step(self, start: Tuple[int,int]) -> Optional[Tuple[Tuple[int,int], str]]:
        """
        BFS az ismert pályán, hogy elérjük a legközelebbi GOAL (100) cellát.
        Ha találunk utat, az első rácslépést (start -> next_step) adjuk vissza.
        """
        H, W = self.world.shape
        sx, sy = start

        q = deque()
        q.append((sx, sy))
        prev: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {(sx, sy): None}

        goal: Optional[Tuple[int,int]] = None

        while q:
            x, y = q.popleft()
            if self.world.known_map[x, y] == CellType.GOAL.value:
                goal = (x, y)
                break
            for dx, dy in DIRS_4:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    if (nx, ny) not in prev and self.world.traversable(nx, ny):
                        prev[(nx, ny)] = (x, y)
                        q.append((nx, ny))

        if goal is None:
            return None  # még nem látjuk / nem érjük el a célt

        # Útvonal visszafejtése: goal -> ... -> start
        path: List[Tuple[int,int]] = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()  # [start, ..., goal]

        if len(path) == 1:
            # már a célon állunk
            return (path[0], "on_goal")

        next_step = path[1]
        return next_step, "go_goal"

    # 1) FRONTIER-LOGIKA
    def _is_frontier_cell(self, x: int, y: int) -> bool:
        """
        Frontier: bejárható, ismert cella, amelynek legalább egy UNKNOWN szomszédja van.
        """
        H, W = self.world.shape
        if not self.world.traversable(x, y):
            return False
        if self.world.known_map[x, y] == CellType.UNKNOWN.value:
            return False
        for dx, dy in DIRS_4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                if self.world.known_map[nx, ny] == CellType.UNKNOWN.value:
                    return True
        return False

    def _find_frontier_step(self, start: Tuple[int,int]) -> Optional[Tuple[Tuple[int,int], str]]:
        """
        BFS az ismert pályán, hogy elérjünk egy frontier cellához.
        Ha találunk, az első rácslépést adjuk vissza.
        """
        H, W = self.world.shape
        sx, sy = start

        q = deque()
        q.append((sx, sy))
        prev: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {(sx, sy): None}

        frontier: Optional[Tuple[int,int]] = None

        while q:
            x, y = q.popleft()
            if (x, y) != (sx, sy) and self._is_frontier_cell(x, y):
                frontier = (x, y)
                break
            for dx, dy in DIRS_4:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    if (nx, ny) not in prev and self.world.traversable(nx, ny):
                        prev[(nx, ny)] = (x, y)
                        q.append((nx, ny))

        if frontier is None:
            return None  # nincs elérhető frontier

        # Útvonal visszafejtése: frontier -> ... -> start
        path: List[Tuple[int,int]] = []
        cur = frontier
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()  # [start, ..., frontier]

        if len(path) >= 2:
            next_step = path[1]
            return next_step, "explore_frontier"
        else:
            # start maga frontier, válassz egy szomszédot, ami UNKNOWN-ok felé néz
            best = None
            for dx, dy in DIRS_8:
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < H and 0 <= ny < W and self.world.traversable(nx, ny):
                    unknown_neighbors = 0
                    for ddx, ddy in DIRS_4:
                        ex, ey = nx + ddx, ny + ddy
                        if 0 <= ex < H and 0 <= ey < W and \
                           self.world.known_map[ex, ey] == CellType.UNKNOWN.value:
                            unknown_neighbors += 1
                    vc = self._get_visit_count(nx, ny)
                    score = (-unknown_neighbors, vc)  # több UNKNOWN, kevesebb visit a jobb
                    if best is None or score < best[0]:
                        best = (score, (nx, ny))
            if best is None:
                return None
            return best[1], "explore_frontier_local"

    def next_grid_target(self, state: State) -> Tuple[Tuple[int,int], str]:
        """
        0) Ha tudunk úton menni a GOAL-ra, menjünk oda.
        1) Ha nem, akkor keressük meg a legközelebbi frontiert.
        2) Ha frontier sincs, lokálisan a legkevésbé látogatott, nem-hazard szomszéd.
        """
        assert state.agent is not None
        ax, ay = int(state.agent.x), int(state.agent.y)
        self._ensure_heading(state)

        # 0) GOAL-prioritás
        goal_res = self._find_goal_step((ax, ay))
        if goal_res is not None:
            return goal_res  # (target_cell, "go_goal" / "on_goal")

        # 1) FRONTIER keresés
        frontier_res = self._find_frontier_step((ax, ay))
        if frontier_res is not None:
            return frontier_res  # (target_cell, "explore_frontier" / "explore_frontier_local")

        # 2) Nincs frontier: lokális "legkevésbé látogatott" szomszéd (8 irány)
        H, W = self.world.shape
        candidates: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []
        for dx, dy in DIRS_8:
            nx, ny = ax + dx, ay + dy
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            if not self._is_free_local(state, nx, ny):
                continue
            haz = 1 if self._is_hazard(nx, ny) else 0
            vc = self._get_visit_count(nx, ny)
            dist = math.hypot(dx, dy)
            # Rendezés: hazard (0=safe, 1=hazard), visit_count, distance
            candidates.append(((haz, vc, dist), (nx, ny)))

        if candidates:
            candidates.sort()
            best_target = candidates[0][1]
            return best_target, "local_explore"

        # Semmi sem elérhető, maradjunk
        return (ax, ay), "stuck"

# ────────────────────────────────────────────────────────────────────────────────
# Low-level driver (hazard-aware scoring)
# ────────────────────────────────────────────────────────────────────────────────

def choose_accel_toward_cell(state: State,
                             world: WorldModel,
                             policy: LeftWallPolicy,
                             target_cell: Tuple[int,int],
                             mode: str) -> Tuple[int,int]:
    assert state.agent is not None and state.visible_track is not None
    rSafe = max(0, state.circuit.visibility_radius - 1)

    p = state.agent.pos
    v = state.agent.vel
    vx, vy = int(v[0]), int(v[1])

    ax_agent, ay_agent = int(state.agent.x), int(state.agent.y)
    
    has_adjacent_zero_4dir = False
    for dx, dy in DIRS_4: 
        nx, ny = ax_agent + dx, ay_agent + dy
        if (world.traversable(nx, ny) and
            world.known_map[nx, ny] == CellType.EMPTY.value and
            world.visited_count[nx, ny] == 0):
            has_adjacent_zero_4dir = True
            break
            
    has_adjacent_zero_8dir = False
    if has_adjacent_zero_4dir:
         has_adjacent_zero_8dir = True
    else:
        for dx, dy in DIRS_8: 
            nx, ny = ax_agent + dx, ay_agent + dy
            if (world.traversable(nx, ny) and
                world.known_map[nx, ny] == CellType.EMPTY.value and
                world.visited_count[nx, ny] == 0):
                has_adjacent_zero_8dir = True
                break
            
    visible_zero_reachable = False
    if not has_adjacent_zero_8dir:
        visible_zero_reachable = find_reachable_zero(state, world, state.agent.pos)

    force_slow_down = (not has_adjacent_zero_4dir and has_adjacent_zero_8dir) or \
                      (not has_adjacent_zero_8dir and visible_zero_reachable)
    
    # FULL STOP override – ha kell lassítani, először próbáljunk fékezni
    if force_slow_down and (vx != 0 or vy != 0):
        possible_brakes = [
            (-int(np.sign(vx)), -int(np.sign(vy))),
            (0, 0)
        ]
        
        for ax, ay in possible_brakes:
            nvx, nvy = vx + ax, vy + ay
            next_pos = p + v + np.array([ax, ay], dtype=int)
            
            if brakingOk(nvx, nvy, rSafe) and \
               validLineLocal(state, p, next_pos) and \
               not any(np.all(next_pos == q.pos) for q in state.players):
                return ax, ay
        
    max_safe = max(1.0, math.sqrt(2 * max(0, rSafe)))
    
    if force_slow_down:
        target_speed = 1.0
    else:
        tx, ty = target_cell
        target_is_hazard = world.is_hazard(tx, ty) if (0 <= tx < world.shape[0] and 0 <= ty < world.shape[1]) else False

        if (0 <= tx < world.shape[0] and 0 <= ty < world.shape[1] and
            world.visited_count[tx, ty] == 0 and not target_is_hazard):
            target_speed = max_safe
        elif mode.startswith("search_") or mode == "corridor_visited" or mode.startswith("fallback"):
            target_speed = max(1.0, 0.5 * max_safe)
        elif target_is_hazard:
            target_speed = max(1.0, 0.5 * max_safe)
        else:
            target_speed = max(1.5, 0.7 * max_safe)

    to_cell = np.array([target_cell[0], target_cell[1]], dtype=float) - p.astype(float)
    n_to = float(np.linalg.norm(to_cell)) or 1.0
    desired_dir = to_cell / n_to

    best = None

    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            nvx, nvy = vx + ax, vy + ay
            if not brakingOk(nvx, nvy, rSafe):
                continue
            next_pos = p + v + np.array([ax, ay], dtype=int)
            nx, ny = int(next_pos[0]), int(next_pos[1])

            if not validLineLocal(state, p, next_pos):
                continue
            if any(np.all(next_pos == q.pos) for q in state.players):
                continue

            dist_cell = float(np.linalg.norm(next_pos.astype(float) - np.array(target_cell, dtype=float)))
            speed_next = float(math.hypot(nvx, nvy))
            speed_pen = abs(speed_next - target_speed)

            heading_pen = 0.0
            if speed_next > 0.0:
                vel_dir = np.array([nvx, nvy], dtype=float) / max(speed_next, 1e-9)
                heading_pen = (1.0 - float(np.dot(vel_dir, desired_dir))) * 0.8

            visit_pen = 0.0
            hazard_pen = 0.0
            if 0 <= nx < world.shape[0] and 0 <= ny < world.shape[1]:
                visit_pen = 100.0 * float(world.visited_count[nx, ny])
                if world.is_hazard(nx, ny):
                    hazard_pen = 500.0

            score = (2.0 * dist_cell) + (0.9 * speed_pen) + heading_pen + visit_pen + hazard_pen
            cand = (score, (ax, ay))
            if (best is None) or (cand[0] < best[0]):
                best = cand

    if best is not None:
        return best[1]

    # végső fallback: fékezés / coast, ha még lehet
    for ax, ay in ((-np.sign(vx), -np.sign(vy)), (0, 0)):
        nvx, nvy = vx + ax, vy + ay
        nxt = p + v + np.array([ax, ay], dtype=int)
        if brakingOk(nvx, nvy, rSafe) and validLineLocal(state, p, nxt):
            if not any(np.all(nxt == q.pos) for q in state.players):
                return int(ax), int(ay)

    return (0, 0)

# ────────────────────────────────────────────────────────────────────────────────
# ASCII dump
# ────────────────────────────────────────────────────────────────────────────────

def dump_ascii(world: WorldModel, policy: LeftWallPolicy, state: State, mode: str) -> None:
    if state.agent is None:
        return
    H, W = world.shape
    km = world.known_map
    vis = world.visited_count

    grid = [['?' for _ in range(W)] for _ in range(H)]
    for x in range(H):
        for y in range(W):
            v = km[x, y]
            if v == CellType.WALL.value:
                grid[x][y] = '#'
            elif v == CellType.GOAL.value:
                grid[x][y] = 'G'
            elif v == CellType.START.value:
                grid[x][y] = 'S'
            elif v == CellType.EMPTY.value:
                grid[x][y] = '.'
            elif v == CellType.UNKNOWN.value:
                grid[x][y] = '?'
            elif is_hazard_val(v):
                grid[x][y] = world.get_hazard_char(int(v))

    # visit-count overlay (nem hazardokra)
    for x in range(H):
        for y in range(W):
            vis_val = vis[x, y]
            if vis_val > 0 and not world.is_hazard(x, y) and grid[x][y] not in ('#', 'G', 'S'):
                if vis_val < 10:
                    grid[x][y] = str(int(vis_val))
                elif vis_val < 36:
                    grid[x][y] = chr(ord('a') + int(vis_val) - 10)
                else:
                    grid[x][y] = '+'

    for p in state.players:
        if 0 <= p.x < H and 0 <= p.y < W:
            grid[p.x][p.y] = 'O'

    ax, ay = int(state.agent.x), int(state.agent.y)
    if 0 <= ax < H and 0 <= ay < W:
        grid[ax][ay] = 'A'

    hdr = []
    hdr.append(
        f"TURN {world.turn}  pos=({ax},{ay}) vel=({int(state.agent.vel_x)},{int(state.agent.vel_y)}) "
        f"mode={mode} heading={policy.heading}"
    )
    hdr.append("LEGEND: #=WALL  ?=UNKNOWN  .=EMPTY  G=GOAL  S=START  A=agent  O=other  [1-9,a-z,+]=visit count")
    if world.hazard_char_map:
        hdr.append("HAZARD TYPES:")
        for tile_val, ch in sorted(world.hazard_char_map.items(), key=lambda t: t[0]):
            hdr.append(f"  {ch} = tile {tile_val}")

    lines = ["\n".join(hdr)]
    for x in range(H):
        lines.append("".join(grid[x]))
    lines.append("")

    log_dir = os.path.dirname(world.dump_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    mode_flag = "a"
    if not world._dump_initialized:
        mode_flag = "w"
        world._dump_initialized = True
    with open(world.dump_file, mode_flag, encoding="utf-8") as f:
        f.write("\n".join(lines))

# ────────────────────────────────────────────────────────────────────────────────
# Decision loop
# ────────────────────────────────────────────────────────────────────────────────

def calculateMove(world: WorldModel, policy: LeftWallPolicy, state: State) -> Tuple[int,int]:
    assert state.agent is not None and state.visible_raw is not None
    world.updateWithObservation(state)

    ax, ay = int(state.agent.x), int(state.agent.y)
    world.visited_count[ax, ay] += 1

    target_cell, mode = policy.next_grid_target(state)
    ax_cmd, ay_cmd = choose_accel_toward_cell(state, world, policy, target_cell, mode)

    world.last_pos = (ax, ay)
    dump_ascii(world, policy, state, mode)
    world.turn += 1

    return ax_cmd, ay_cmd

# ────────────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    print("READY", flush=True)
    circuit = read_initial_observation()
    state: Optional[State] = State(circuit, None, None, [], None)

    world = WorldModel(circuit.track_shape)
    policy = LeftWallPolicy(world)

    while True:
        assert state is not None
        state = read_observation(state)
        if state is None:
            return

        ax, ay = calculateMove(world, policy, state)

        # clamp to judge-legal range
        ax = -1 if ax < -1 else (1 if ax > 1 else int(ax))
        ay = -1 if ay < -1 else (1 if ay > 1 else int(ay))
        print(f"{ax} {ay}", flush=True)

if __name__ == "__main__":
    main()
