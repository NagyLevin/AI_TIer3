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
    OIL = 91      # oil
    SAND = 92     # sand

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

# Hazard tile sets
SAND_TILES = {CellType.SAND.value}  # 92
OIL_TILES  = {CellType.OIL.value}   # 91

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

def is_sand_val(v: int) -> bool:
    return v in SAND_TILES

def is_oil_val(v: int) -> bool:
    return v in OIL_TILES

def is_traversable_val(v: int) -> bool:
    # UNKNOWN-ra nem tervezünk utat; minden más >=0 mehet (hazard is),
    # de a BFS-ben opcionálisan kizárjuk a hazardot.
    return (v >= 0) and (v != CellType.UNKNOWN.value)

def is_hazard_val(v: int) -> bool:
    if is_sand_val(v) or is_oil_val(v):
        return True
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

        # hazard type registry
        self.hazard_val = np.full((H, W), -1, dtype=int)  # -1 = not hazard
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
        return is_traversable_val(int(self.known_map[x, y]))

    def is_hazard(self, x: int, y: int) -> bool:
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W):
            return False
        return int(self.hazard_val[x, y]) != -1

    def is_sand(self, x: int, y: int) -> bool:
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W):
            return False
        return is_sand_val(int(self.known_map[x, y]))

    def is_oil(self, x: int, y: int) -> bool:
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W):
            return False
        return is_oil_val(int(self.known_map[x, y]))

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
# Explorer policy: GOAL-first + frontier exploration
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
    GOAL-prioritised, frontier-based explorer (name kept for compatibility).

    Priority:
      0) If GOAL (100) is known and reachable, go there.
         - First try a non-hazard path.
         - If impossible, allow sand/oil (hazard) tiles.
      1) Else, go to a frontier cell (known, with UNKNOWN neighbour).
      2) Else, local least-visited neighbour.
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

    # ── GOAL BFS (two phases: non-hazard, then hazard-allowed) ────────────────

    def _bfs_to_goal(self, start: Tuple[int,int], allow_hazard: bool) -> Optional[Tuple[Tuple[int,int], str]]:
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
                if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in prev:
                    tile_val = int(self.world.known_map[nx, ny])
                    if not is_traversable_val(tile_val):
                        continue
                    if (not allow_hazard) and is_hazard_val(tile_val):
                        continue
                    prev[(nx, ny)] = (x, y)
                    q.append((nx, ny))

        if goal is None:
            return None

        # reconstruct path
        path: List[Tuple[int,int]] = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()

        if len(path) == 1:
            return (path[0], "on_goal")

        # estimate distance to first turn along the path
        turn_dist = 999
        if len(path) >= 3:
            dir0 = (path[1][0] - path[0][0], path[1][1] - path[0][1])
            turn_dist = len(path)
            for i in range(2, len(path)):
                diri = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
                if diri != dir0:
                    turn_dist = i - 1
                    break

        next_step = path[1]
        mode = "go_goal"
        if allow_hazard and any(is_hazard_val(int(self.world.known_map[x, y])) for (x, y) in path[1:]):
            mode = "go_goal_through_hazard"

        if turn_dist <= 3:
            mode += "_turn_soon"
        elif turn_dist <= 6:
            mode += "_turn_med"

        return next_step, mode

    def _find_goal_step(self, start: Tuple[int,int]) -> Optional[Tuple[Tuple[int,int], str]]:
        res = self._bfs_to_goal(start, allow_hazard=False)
        if res is not None:
            return res
        return self._bfs_to_goal(start, allow_hazard=True)

    # ── FRONTIER BFS (same two-phase logic) ───────────────────────────────────

    def _is_frontier_cell(self, x: int, y: int) -> bool:
        """
        Frontier: traversable, known cell with at least one UNKNOWN neighbour.
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

    def _bfs_to_frontier(self, start: Tuple[int,int], allow_hazard: bool) -> Optional[Tuple[Tuple[int,int], str]]:
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
                if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in prev:
                    tile_val = int(self.world.known_map[nx, ny])
                    if not is_traversable_val(tile_val):
                        continue
                    if (not allow_hazard) and is_hazard_val(tile_val):
                        continue
                    prev[(nx, ny)] = (x, y)
                    q.append((nx, ny))

        if frontier is None:
            return None

        path: List[Tuple[int,int]] = []
        cur = frontier
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()

        if len(path) >= 2:
            next_step = path[1]

            # distance to first turn
            turn_dist = 999
            if len(path) >= 3:
                dir0 = (path[1][0] - path[0][0], path[1][1] - path[0][1])
                turn_dist = len(path)
                for i in range(2, len(path)):
                    diri = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
                    if diri != dir0:
                        turn_dist = i - 1
                        break

            mode = "explore_frontier"
            if allow_hazard and any(is_hazard_val(int(self.world.known_map[x, y])) for (x, y) in path[1:]):
                mode = "explore_frontier_through_hazard"

            if turn_dist <= 3:
                mode += "_turn_soon"
            elif turn_dist <= 6:
                mode += "_turn_med"

            return next_step, mode

        # start itself is frontier – local fallback
        sx, sy = start
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
                score = (-unknown_neighbors, vc)
                if best is None or score < best[0]:
                    best = (score, (nx, ny))
        if best is None:
            return None
        return best[1], "explore_frontier_local"

    def _find_frontier_step(self, start: Tuple[int,int]) -> Optional[Tuple[Tuple[int,int], str]]:
        res = self._bfs_to_frontier(start, allow_hazard=False)
        if res is not None:
            return res
        return self._bfs_to_frontier(start, allow_hazard=True)

    def next_grid_target(self, state: State) -> Tuple[Tuple[int,int], str]:
        """
        0) If we can path to GOAL, go there (first without hazards, then with).
        1) Else go to a frontier cell.
        2) Else local least-visited neighbour.
        """
        assert state.agent is not None
        ax, ay = int(state.agent.x), int(state.agent.y)
        self._ensure_heading(state)

        # GOAL first
        goal_res = self._find_goal_step((ax, ay))
        if goal_res is not None:
            return goal_res

        # then frontier
        frontier_res = self._find_frontier_step((ax, ay))
        if frontier_res is not None:
            return frontier_res

        # local fallback
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
            candidates.append(((haz, vc, dist), (nx, ny)))

        if candidates:
            candidates.sort()
            best_target = candidates[0][1]
            return best_target, "local_explore"

        return (ax, ay), "stuck"

# ────────────────────────────────────────────────────────────────────────────────
# Low-level driver (turn-aware slowing + hazard-aware scoring)
# ────────────────────────────────────────────────────────────────────────────────

def choose_accel_toward_cell(state: State,
                             world: WorldModel,
                             policy: LeftWallPolicy,
                             target_cell: Tuple[int,int],
                             mode: str) -> Tuple[int,int]:
    """
    Low-level controller that is *explicitly* turn-aware.
    - If the high-level mode string contains "turn_soon" / "turn_med",
      we start braking early so we don't fly past openings and ping-pong.
    """
    assert state.agent is not None and state.visible_track is not None
    rSafe = max(0, state.circuit.visibility_radius - 1)

    p = state.agent.pos
    v = state.agent.vel
    vx, vy = int(v[0]), int(v[1])

    ax_agent, ay_agent = int(state.agent.x), int(state.agent.y)
    tx, ty = target_cell

    # current speed magnitude
    speed_now = float(math.hypot(vx, vy))

    # ── 1. classic "0-cell" slowdown context ──────────────────────────────────
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

    force_slow_down = ((not has_adjacent_zero_4dir) and has_adjacent_zero_8dir) or \
                      ((not has_adjacent_zero_8dir) and visible_zero_reachable)

    # ── 2. NEW: turn-aware braking ────────────────────────────────────────────
    turn_soon = ("turn_soon" in mode)
    turn_med  = ("turn_med" in mode)

    force_brake_for_turn = False
    if turn_soon and speed_now > 1.0:
        force_brake_for_turn = True
    elif turn_med and speed_now > 2.5:
        force_brake_for_turn = True

    if (force_slow_down or force_brake_for_turn) and (vx != 0 or vy != 0):
        possible_brakes = [
            (-int(np.sign(vx)), -int(np.sign(vy))),  # full brake
            (0, 0),                                  # coast
        ]
        for ax, ay in possible_brakes:
            nvx, nvy = vx + ax, vy + ay
            next_pos = p + v + np.array([ax, ay], dtype=int)
            if brakingOk(nvx, nvy, rSafe) and \
               validLineLocal(state, p, next_pos) and \
               not any(np.all(next_pos == q.pos) for q in state.players):
                return ax, ay

    # ── 3. target speed with global cap ───────────────────────────────────────
    max_safe = max(1.0, math.sqrt(2 * max(0, rSafe)))
    max_safe = min(max_safe, 2.0)  # HARD CAP: about 2 tiles / turn

    if force_slow_down:
        target_speed = 1.0
    else:
        target_is_hazard = (0 <= tx < world.shape[0] and
                            0 <= ty < world.shape[1] and
                            world.is_hazard(tx, ty))

        if (0 <= tx < world.shape[0] and 0 <= ty < world.shape[1] and
            world.visited_count[tx, ty] == 0 and not target_is_hazard):
            target_speed = max_safe
        elif mode.startswith("explore_frontier") or mode == "local_explore":
            target_speed = max(1.0, 0.8 * max_safe)
        elif target_is_hazard:
            target_speed = max(1.0, 0.5 * max_safe)
        elif mode.startswith("go_goal"):
            target_speed = max(1.5, 0.9 * max_safe)
        else:
            target_speed = max(1.5, 0.7 * max_safe)

    if turn_soon:
        target_speed = min(target_speed, 1.0)
    elif turn_med:
        target_speed = min(target_speed, 1.5)

    # surface-based adjustment: sand / oil
    if 0 <= tx < world.shape[0] and 0 <= ty < world.shape[1]:
        if world.is_sand(tx, ty) or world.is_sand(ax_agent, ay_agent):
            target_speed = min(target_speed, 1.0)
        if world.is_oil(tx, ty) or world.is_oil(ax_agent, ay_agent):
            target_speed = min(target_speed, 0.6)

    # ── 4. desired direction ──────────────────────────────────────────────────
    to_cell = np.array([target_cell[0], target_cell[1]], dtype=float) - p.astype(float)
    n_to = float(np.linalg.norm(to_cell)) or 1.0
    desired_dir = to_cell / n_to

    if speed_now > 0.0:
        vel_dir = np.array([vx, vy], dtype=float) / max(speed_now, 1e-9)
        turn_cos = float(np.dot(vel_dir, desired_dir))
        if turn_cos < 0.5:       # > ~60° turn
            target_speed = min(target_speed, 1.0)
        elif turn_cos < 0.8:     # ~36–60°
            target_speed = min(target_speed, 1.5)

    # ── 5. evaluate candidate accelerations ───────────────────────────────────
    best = None

    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            nvx, nvy = vx + ax, vy + ay
            speed_next = float(math.hypot(nvx, nvy))

            # forbid too high speeds near turns
            if turn_soon and speed_next > 1.5:
                continue
            if turn_med and speed_next > 2.5:
                continue

            if not brakingOk(nvx, nvy, rSafe):
                continue
            next_pos = p + v + np.array([ax, ay], dtype=int)
            nx, ny = int(next_pos[0]), int(next_pos[1])

            if not validLineLocal(state, p, next_pos):
                continue
            if any(np.all(next_pos == q.pos) for q in state.players):
                continue

            dist_cell = float(np.linalg.norm(
                next_pos.astype(float) - np.array(target_cell, dtype=float)
            ))
            speed_pen = abs(speed_next - target_speed)

            heading_pen = 0.0
            if speed_next > 0.0:
                vel_dir = np.array([nvx, nvy], dtype=float) / max(speed_next, 1e-9)
                heading_pen = (1.0 - float(np.dot(vel_dir, desired_dir))) * 0.8

            visit_pen = 0.0
            hazard_pen = 0.0
            if 0 <= nx < world.shape[0] and 0 <= ny < world.shape[1]:
                vc = float(world.visited_count[nx, ny])
                if mode.startswith("go_goal"):
                    visit_pen = 10.0 * vc
                elif mode.startswith("explore_frontier"):
                    visit_pen = 50.0 * vc
                else:
                    visit_pen = 100.0 * vc

                if world.is_oil(nx, ny):
                    hazard_pen = 500.0
                elif world.is_sand(nx, ny):
                    hazard_pen = 300.0
                elif world.is_hazard(nx, ny):
                    hazard_pen = 400.0

                if mode.startswith("go_goal"):
                    hazard_pen *= 0.2
                elif mode.startswith("explore_frontier"):
                    hazard_pen *= 0.5

            score = (2.0 * dist_cell) + (0.9 * speed_pen) + heading_pen + visit_pen + hazard_pen
            cand = (score, (ax, ay))
            if (best is None) or (cand[0] < best[0]):
                best = cand

    if best is not None:
        return best[1]

    # ── 6. final fallback: brake / coast ──────────────────────────────────────
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
            elif is_hazard_val(int(v)):
                grid[x][y] = world.get_hazard_char(int(v))

    # visit-count overlay (non-hazards)
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
