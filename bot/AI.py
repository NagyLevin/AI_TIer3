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
    """Cell types as defined in the ICPC-style race problem."""
    GOAL = 100
    START = 1
    WALL = -1
    UNKNOWN = 2
    EMPTY = 0
    NOT_VISIBLE = 3
    OIL = 91      # oil (hazard)
    SAND = 92     # sand (hazard)


class Player(NamedTuple):
    """
    Player state as read from input.
    We only know other players' positions; their velocities are set to 0.
    """
    x: int
    y: int
    vel_x: int
    vel_y: int

    @property
    def pos(self) -> np.ndarray:
        """Return position as numpy array [row, col]."""
        return np.array([self.x, self.y], dtype=int)

    @property
    def vel(self) -> np.ndarray:
        """Return velocity as numpy array [vx, vy]."""
        return np.array([self.vel_x, self.vel_y], dtype=int)


class Circuit(NamedTuple):
    """Static circuit data from the first line of input."""
    track_shape: tuple[int, int]
    num_players: int
    visibility_radius: int


class State(NamedTuple):
    """
    Per-turn state:
      - visible_track: NOT_VISIBLE already mapped to WALL (safety map),
      - visible_raw:   NOT_VISIBLE kept as 3, so we know what we haven't seen yet.
    """
    circuit: Circuit
    visible_track: Optional[np.ndarray]
    visible_raw: Optional[np.ndarray]
    players: List[Player]
    agent: Optional[Player]


# Hazard tile ID sets (easy to change if spec changes)
SAND_TILES = {CellType.SAND.value}  # {92}
OIL_TILES  = {CellType.OIL.value}   # {91}

# ────────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────────

def read_initial_observation() -> Circuit:
    """
    Read the first line:  H W num_players visibility_radius
    and construct a Circuit object.
    """
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)


def read_observation(old_state: State) -> Optional[State]:
    """
    Read one turn's observation:
      - agent's position and velocity,
      - other players' positions,
      - the (2R+1)x(2R+1) local window around the agent.

    Returns updated State or None if the game ended (~~~END~~~).
    """
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

    # Insert the (2R+1)x(2R+1) window into global arrays
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
                # For safety, NOT_VISIBLE is treated as WALL in visible_track
                safety = [
                    CellType.WALL.value if v == CellType.NOT_VISIBLE.value else v
                    for v in loc
                ]
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
    """
    Triangular number: minimal stopping distance for speed n
    with 1-cell deceleration each turn.
    """
    return n * (n + 1) // 2


def brakingOk(vx: int, vy: int, rSafe: int) -> bool:
    """
    Global braking constraint on both axes:
    we must be able to stop within rSafe cells along x and y.
    """
    return (tri(abs(vx)) <= rSafe) and (tri(abs(vy)) <= rSafe)


def validLineLocal(state: State, p1: np.ndarray, p2: np.ndarray) -> bool:
    """
    Line-of-sight collision check between p1 and p2 using visible_track.
    Treats NOT_VISIBLE as WALL (we already mapped it in visible_track).
    """
    track = state.visible_track
    if track is None:
        return False
    H, W = track.shape
    if (np.any(p1 < 0) or np.any(p2 < 0) or
        p1[0] >= H or p1[1] >= W or p2[0] >= H or p2[1] >= W):
        return False

    diff = p2 - p1

    # Vertical-ish traversal
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))
        for i in range(abs(diff[0]) + 1):
            x = int(p1[0] + i * d)
            y = p1[1] + i * slope * d
            yC = int(np.ceil(y))
            yF = int(np.floor(y))
            if track[x, yC] < 0 and track[x, yF] < 0:
                return False

    # Horizontal-ish traversal
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))
        for i in range(abs(diff[1]) + 1):
            x = p1[0] + i * slope * d
            y = int(p1[1] + i * d)
            xC = int(np.ceil(x))
            xF = int(np.floor(x))
            if track[xC, y] < 0 and track[xF, y] < 0:
                return False

    return True


def find_reachable_zero(state: State,
                        world: 'WorldModel',
                        start_pos: np.ndarray) -> bool:
    """
    Local BFS inside the visible window to see if there is a reachable EMPTY
    cell (0) with visited_count == 0 from start_pos.

    Used to detect "near unexplored area" and slow down for finer exploration.
    """
    if state.agent is None:
        return False

    q = deque([(int(start_pos[0]), int(start_pos[1]))])
    visited = {(int(start_pos[0]), int(start_pos[1]))}

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
    """Return True if v is a sand tile value."""
    return v in SAND_TILES


def is_oil_val(v: int) -> bool:
    """Return True if v is an oil tile value."""
    return v in OIL_TILES


def is_traversable_val(v: int) -> bool:
    """
    For BFS: UNKNOWN (2) is not traversable; everything else >= 0 is,
    including hazards (we filter them separately if needed).
    """
    return (v >= 0) and (v != CellType.UNKNOWN.value)


def is_hazard_val(v: int) -> bool:
    """
    Returns True for any "weird" positive tile that is not EMPTY/START/GOAL.
    By spec: sand=92, oil=91, plus any other special positive tiles.
    """
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
    """
    Global world model.

    - known_map: global HxW map (UNKNOWN at start, updated from visible_raw).
    - visited_count[x,y]: how many times we *stood* on cell (x,y).
    - hazard_val: cached hazard tile values for ASCII display.
    - last_pos: previous agent position (for anti ping-pong).
    """
    def __init__(self, shape: tuple[int, int]) -> None:
        H, W = shape
        self.shape = shape
        self.known_map = np.full((H, W), CellType.UNKNOWN.value, dtype=int)
        self.visited_count = np.zeros((H, W), dtype=int)
        self.last_pos: Optional[Tuple[int, int]] = None

        # hazard tracking
        self.hazard_val = np.full((H, W), -1, dtype=int)  # -1 = not hazard
        self.hazard_char_map: Dict[int, str] = {}
        # Letters reserved for hazard drawing in ASCII dump
        self.hazard_char_pool = list("BCDFHJKLMPQRTUVWXYZ")

        self.turn = 0
        self.dump_file = "logs/map_dump.txt"
        self._dump_initialized = False

    def updateWithObservation(self, st: State) -> None:
        """
        Merge the latest visible_raw into known_map and recompute hazard_val.
        """
        if st.visible_raw is None:
            return
        raw = st.visible_raw
        seen = (raw != CellType.NOT_VISIBLE.value)
        self.known_map[seen] = raw[seen]

        hazard_mask = np.vectorize(is_hazard_val)(self.known_map)
        self.hazard_val[:, :] = -1
        self.hazard_val[hazard_mask] = self.known_map[hazard_mask]

    def traversable(self, x: int, y: int) -> bool:
        """Check if (x,y) is in-bounds and traversable by BFS rules."""
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W):
            return False
        return is_traversable_val(int(self.known_map[x, y]))

    def is_hazard(self, x: int, y: int) -> bool:
        """Return True if (x,y) is any hazard tile."""
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W):
            return False
        return int(self.hazard_val[x, y]) != -1

    def is_sand(self, x: int, y: int) -> bool:
        """Return True if (x,y) is sand."""
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W):
            return False
        return is_sand_val(int(self.known_map[x, y]))

    def is_oil(self, x: int, y: int) -> bool:
        """Return True if (x,y) is oil."""
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W):
            return False
        return is_oil_val(int(self.known_map[x, y]))

    def get_hazard_value(self, x: int, y: int) -> Optional[int]:
        """Return the hazard tile value at (x,y) or None if not hazard."""
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W):
            return None
        v = int(self.hazard_val[x, y])
        return v if v != -1 else None

    def get_hazard_char(self, tile_value: int) -> str:
        """
        Map a hazard tile value to an ASCII character for dump.
        Re-uses characters when we run out.
        """
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
# Explorer policy: GOAL-first + frontier-based exploration
# ────────────────────────────────────────────────────────────────────────────────

def left_of(d: Tuple[int, int]) -> Tuple[int, int]:
    dx, dy = d
    return (-dy, dx)

def right_of(d: Tuple[int, int]) -> Tuple[int, int]:
    dx, dy = d
    return (dy, -dx)

def back_of(d: Tuple[int, int]) -> Tuple[int, int]:
    dx, dy = d
    return (-dx, -dy)


class LeftWallPolicy:
    """
    Global explorer with GOAL-first BFS and frontier-based exploration.

    Priority:
      0) If we know a path to GOAL, follow it:
         - first try a path that avoids hazards,
         - if none exists, allow hazards (sand/oil) along the path.
      1) If no GOAL path, BFS to frontier cells (known, traversable cells
         with at least one UNKNOWN 4-neighbour), same hazard rules as above.
      2) If even frontier is unreachable, do a local fallback:
         pick least-visited, non-hazard neighbour (8-dir) if possible.
    """
    def __init__(self, world: WorldModel) -> None:
        self.world = world
        self.heading: Tuple[int, int] = (0, 1)  # approximate heading (debug only)

    def _ensure_heading_from_velocity(self, state: State) -> None:
        """Update stored heading from agent's velocity if non-zero."""
        if state.agent is None:
            return
        vx, vy = int(state.agent.vel_x), int(state.agent.vel_y)
        if vx == 0 and vy == 0:
            return
        if abs(vx) >= abs(vy):
            self.heading = (int(np.sign(vx)), 0)
        else:
            self.heading = (0, int(np.sign(vy)))

    def _bfs_to_condition(
        self,
        start: Tuple[int, int],
        cond,
        allow_hazard: bool,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Generic BFS on world.known_map from 'start' to the first cell
        satisfying cond(x, y, world).

        If allow_hazard is False, BFS avoids hazard tiles.
        If allow_hazard is True, BFS may cross sand/oil/etc.
        """
        H, W = self.world.shape
        sx, sy = start
        if not self.world.traversable(sx, sy):
            return None

        q = deque([start])
        prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while q:
            x, y = q.popleft()
            if cond(x, y, self.world):
                # reconstruct path
                path: List[Tuple[int, int]] = []
                cur: Optional[Tuple[int, int]] = (x, y)
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path

            for dx, dy in DIRS_4:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < H and 0 <= ny < W):
                    continue
                if (nx, ny) in prev:
                    continue
                tile_val = int(self.world.known_map[nx, ny])
                if not is_traversable_val(tile_val):
                    continue
                if (not allow_hazard) and is_hazard_val(tile_val):
                    continue
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))

        return None

    @staticmethod
    def _is_goal(x: int, y: int, world: WorldModel) -> bool:
        """BFS condition: this cell is a GOAL."""
        return world.known_map[x, y] == CellType.GOAL.value

    @staticmethod
    def _is_frontier(x: int, y: int, world: WorldModel) -> bool:
        """
        Frontier: known traversable cell that has at least one UNKNOWN
        neighbour in 4-connectivity.
        """
        H, W = world.shape
        if not world.traversable(x, y):
            return False
        if world.known_map[x, y] == CellType.UNKNOWN.value:
            return False
        for dx, dy in DIRS_4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                if world.known_map[nx, ny] == CellType.UNKNOWN.value:
                    return True
        return False

    def _path_to_goal(
        self, start: Tuple[int, int]
    ) -> Optional[Tuple[List[Tuple[int, int]], str]]:
        """
        BFS path to GOAL:
          - first avoid hazards if possible,
          - if that fails, allow hazards.

        Also annotate the mode string with:
          *_turn_soon  if first turn along the BFS path is very near,
          *_turn_med   if it is moderately near.
        """
        # First try non-hazard path
        path = self._bfs_to_condition(start, self._is_goal, allow_hazard=False)
        mode = "go_goal"
        if path is None:
            # Then try path allowing hazards
            path = self._bfs_to_condition(start, self._is_goal, allow_hazard=True)
            if path is None:
                return None
            mode = "go_goal_through_hazard"

        # Compute distance to first direction change along path
        turn_dist = 999
        if len(path) >= 3:
            dir0 = (path[1][0] - path[0][0], path[1][1] - path[0][1])
            turn_dist = len(path)
            for i in range(2, len(path)):
                d = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
                if d != dir0:
                    turn_dist = i - 1
                    break

        if turn_dist <= 3:
            mode += "_turn_soon"
        elif turn_dist <= 6:
            mode += "_turn_med"

        return path, mode

    def _path_to_frontier(
        self, start: Tuple[int, int]
    ) -> Optional[Tuple[List[Tuple[int, int]], str]]:
        """
        BFS path to frontier cell:
          - first avoid hazards,
          - then allow hazards if needed.
        Also encodes *_turn_soon / *_turn_med in the mode string.

        Ha az útvonal csak az aktuális cellát tartalmazza (path hossza = 1),
        akkor ez azt jelenti, hogy **magán a frontier cellán állunk**. Ilyenkor
        választunk egy szomszéd cellát, amely az ismeretlen irány felé visz,
        és hozzáadjuk a path-hoz, hogy tényleges lépés legyen a target.
        """
        path = self._bfs_to_condition(
            start, self._is_frontier, allow_hazard=False
        )
        mode = "explore_frontier"
        allow_hazard_used = False
        if path is None:
            path = self._bfs_to_condition(
                start, self._is_frontier, allow_hazard=True
            )
            if path is None:
                return None
            mode = "explore_frontier_through_hazard"
            allow_hazard_used = True

        H, W = self.world.shape

        # Ha a path csak a start cellát tartalmazza (már frontier-en állunk),
        # akkor válasszunk egy szomszédot, amely UNKNOWN felé mutat.
        if len(path) == 1:
            sx, sy = path[0]
            best = None
            for dx, dy in DIRS_8:
                nx, ny = sx + dx, sy + dy
                if not (0 <= nx < H and 0 <= ny < W):
                    continue
                tile_val = int(self.world.known_map[nx, ny])
                if not is_traversable_val(tile_val):
                    continue
                if (not allow_hazard_used) and is_hazard_val(tile_val):
                    continue

                # hány UNKNOWN szomszédja van ennek a jelöltnek?
                unknown_neighbors = 0
                for ddx, ddy in DIRS_4:
                    ex, ey = nx + ddx, ny + ddy
                    if 0 <= ex < H and 0 <= ey < W:
                        if self.world.known_map[ex, ey] == CellType.UNKNOWN.value:
                            unknown_neighbors += 1

                vc = float(self.world.visited_count[nx, ny])
                score = (-unknown_neighbors, vc)  # több UNKNOWN, kevesebb visit a jobb
                if best is None or score < best[0]:
                    best = (score, (nx, ny))

            if best is not None:
                path.append(best[1])

        # Ezután normálisan felcímkézzük a kanyar távolságot.
        turn_dist = 999
        if len(path) >= 3:
            dir0 = (path[1][0] - path[0][0], path[1][1] - path[0][1])
            turn_dist = len(path)
            for i in range(2, len(path)):
                d = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
                if d != dir0:
                    turn_dist = i - 1
                    break

        if turn_dist <= 3:
            mode += "_turn_soon"
        elif turn_dist <= 6:
            mode += "_turn_med"

        return path, mode

    def _choose_target_from_path(
        self,
        path: List[Tuple[int, int]],
        mode: str,
        state: State
    ) -> Tuple[Tuple[int, int], str]:
        """
        Given a BFS path [start, ..., goal/frontier] and a base mode string,
        pick a target cell along the path depending on the current speed.

        Lassú mozgás → path[1] (következő cella),
        gyorsabb mozgás → kicsit előrébb nézünk (path[2] / path[3]),
        hogy előre vegyük fel a kanyarodás irányát.
        """
        assert state.agent is not None
        if len(path) == 1:
            return path[0], mode

        vx, vy = int(state.agent.vel_x), int(state.agent.vel_y)
        speed_now = float(math.hypot(vx, vy))

        # Egyszerű lookahead: minél gyorsabbak vagyunk, annál előrébb nézünk az úton
        if speed_now <= 0.5:
            lookahead = 1
        elif speed_now <= 1.5:
            lookahead = 2
        else:
            lookahead = 3

        idx = min(lookahead, len(path) - 1)
        return path[idx], mode

    def next_grid_target(self, state: State) -> Tuple[Tuple[int, int], str]:
        """
        Decide the next *grid cell* to aim for, plus a mode string.

        Returns:
          (target_cell, mode)
        where:
          - target_cell is an (x, y) tuple,
          - mode encodes what we're doing (go_goal..., explore_frontier..., local_explore, etc.)
            and may include suffixes _turn_soon / _turn_med.
        """
        assert state.agent is not None
        ax, ay = int(state.agent.x), int(state.agent.y)
        self._ensure_heading_from_velocity(state)

        start = (ax, ay)

        # 0) GOAL-first, if any path exists
        goal_res = self._path_to_goal(start)
        if goal_res is not None:
            path, mode = goal_res
            target, mode = self._choose_target_from_path(path, mode, state)
            return target, mode

        # 1) Frontier-based exploration
        frontier_res = self._path_to_frontier(start)
        if frontier_res is not None:
            path, mode = frontier_res
            target, mode = self._choose_target_from_path(path, mode, state)
            return target, mode

        # 2) Local fallback: least visited neighbour (8-dir), prefer non-hazard
        H, W = self.world.shape
        candidates: List[Tuple[Tuple[float, float, float], Tuple[int, int]]] = []
        for dx, dy in DIRS_8:
            nx, ny = ax + dx, ay + dy
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            if not self.world.traversable(nx, ny):
                continue
            haz = 1 if self.world.is_hazard(nx, ny) else 0
            vc = float(self.world.visited_count[nx, ny])
            dist = math.hypot(dx, dy)
            candidates.append(((haz, vc, dist), (nx, ny)))

        if candidates:
            candidates.sort()
            best_target = candidates[0][1]
            return best_target, "local_explore"

        # Totally stuck: stay in place
        return (ax, ay), "stuck"

# ────────────────────────────────────────────────────────────────────────────────
# Low-level driver (corner-aware, hazard-aware, anti ping-pong)
# ────────────────────────────────────────────────────────────────────────────────

def choose_accel_toward_cell(state: State,
                             world: WorldModel,
                             policy: LeftWallPolicy,
                             target_cell: Tuple[int, int],
                             mode: str) -> Tuple[int, int]:
    """
    Corner-aware low-level controller.

    Fő feladatai:
      - Válassza ki az (ax, ay) gyorsulást {-1, 0, 1}^2-ből.
      - Tartsa be a fékezési feltételeket (globális és irányfüggő).
      - Kerülje a falakat (validLineLocal) és az ütközést más játékosokkal.
      - Mozogjon a target_cell felé, ésszerű célsebességgel (target_speed).
      - Lehetőleg kerülje a hazard (homok/olaj) mezőket, kivéve ha a high-level
        mód kifejezetten engedi (go_goal_through_hazard, explore_frontier_through_hazard).
      - Csökkentse a ping-pong jelenséget:
          * erős büntetés az azonnali visszalépésre (world.last_pos),
          * kanyar előtt és rossz irányú sebességnél agresszíven fékezzen,
          * kemény sebességkorlátot használjon kanyar-zónában.
      - Ha közel a target_cell, próbáljon pont a cellára ráállni kis sebességgel
        (snap), hogy “bekanyarodjon” a nyílásba és ne szaladjon túl.
    """
    assert state.agent is not None and state.visible_track is not None

    H, W = world.shape
    rSafe = max(0, state.circuit.visibility_radius - 1)

    p = state.agent.pos.astype(int)
    v = state.agent.vel.astype(int)
    vx, vy = int(v[0]), int(v[1])

    ax_agent, ay_agent = int(state.agent.x), int(state.agent.y)
    tx, ty = target_cell

    # ───────────────────────────────────────────────────────────────────
    # Ha a high-level ugyanarra a cellára céloz, ahol állunk (nem GOAL),
    # akkor válasszunk inkább egy szomszéd cellát targetnek, hogy ne
    # ragadjunk ott helyben.
    # ───────────────────────────────────────────────────────────────────
    if (tx, ty) == (ax_agent, ay_agent):
        cur_val = int(world.known_map[ax_agent, ay_agent])
        if cur_val != CellType.GOAL.value:
            allow_hazard_here = ("through_hazard" in mode)
            best = None
            for dx, dy in DIRS_8:
                nx, ny = ax_agent + dx, ay_agent + dy
                if not (0 <= nx < H and 0 <= ny < W):
                    continue
                tile_val = int(world.known_map[nx, ny])
                if not is_traversable_val(tile_val):
                    continue
                if (not allow_hazard_here) and is_hazard_val(tile_val):
                    continue

                # hány UNKNOWN szomszédja van ennek a jelöltnek?
                unknown_neighbors = 0
                for ddx, ddy in DIRS_4:
                    ex, ey = nx + ddx, ny + ddy
                    if 0 <= ex < H and 0 <= ey < W:
                        if world.known_map[ex, ey] == CellType.UNKNOWN.value:
                            unknown_neighbors += 1

                vc = float(world.visited_count[nx, ny])
                haz = 1 if world.is_hazard(nx, ny) else 0
                # több UNKNOWN, kevesebb visit, nem-hazard -> jobb
                score = (-unknown_neighbors, haz, vc)
                if best is None or score < best[0]:
                    best = (score, (nx, ny))

            if best is not None:
                tx, ty = best[1]
                target_cell = (tx, ty)

    track = state.visible_track
    R = state.circuit.visibility_radius

    # ───────────────────────────────────────────────────────────────────
    # Directional braking: mennyi szabad cella van falig x/y irányban?
    # ───────────────────────────────────────────────────────────────────
    def free_steps_dir(dx: int, dy: int) -> int:
        """
        Count how many traversable cells in visible_track we can go in
        direction (dx, dy) from the current position before hitting a wall
        or leaving the visible area.
        """
        steps = 0
        x, y = int(p[0]), int(p[1])
        while steps < R:
            x += dx
            y += dy
            if x < 0 or x >= H or y < 0 or y >= W:
                break
            if track[x, y] < 0:  # treat negative as wall
                break
            steps += 1
        return steps

    free_x_pos = free_steps_dir(1, 0)   # down  (vx > 0)
    free_x_neg = free_steps_dir(-1, 0)  # up    (vx < 0)
    free_y_pos = free_steps_dir(0, 1)   # right (vy > 0)
    free_y_neg = free_steps_dir(0, -1)  # left  (vy < 0)

    def directional_braking_ok(nvx: int, nvy: int) -> bool:
        """
        Per-axis stopping safety check based on free space until wall.
        """
        if nvx > 0 and tri(nvx) > free_x_pos:
            return False
        if nvx < 0 and tri(-nvx) > free_x_neg:
            return False
        if nvy > 0 and tri(nvy) > free_y_pos:
            return False
        if nvy < 0 and tri(-nvy) > free_y_neg:
            return False
        return True

    def directional_ok(nvx: int, nvy: int) -> bool:
        """
        Relaxed version of directional braking:

        - Ha a következő sebesség komponensei abszolút értékben <= 1,
          akkor mindig engedett (feltéve, hogy a globális brakingOk és a
          falellenőrzés rendben van). Ez biztosítja, hogy álló helyzetből
          mindig tudjunk **legalább 1 cellát lépni** bármely szabad irányba.
        - Nagyobb sebességeknél a szigorú directional_braking_ok lép életbe.
        """
        if abs(nvx) <= 1 and abs(nvy) <= 1:
            return True
        return directional_braking_ok(nvx, nvy)

    # ───────────────────────────────────────────────────────────────────
    # Geometria a célcellához képest
    # ───────────────────────────────────────────────────────────────────
    speed_now = float(math.hypot(vx, vy))

    dx_t = tx - ax_agent
    dy_t = ty - ay_agent
    dist_grid = max(abs(dx_t), abs(dy_t))

    # Desired direction (unit vector) from current position to target cell
    to_cell = np.array([tx, ty], dtype=float) - p.astype(float)
    n_to = float(np.linalg.norm(to_cell)) or 1.0
    desired_dir = to_cell / n_to

    if speed_now > 0.0:
        vel_dir = np.array([vx, vy], dtype=float) / max(speed_now, 1e-9)
        cos_vel_desired = float(np.dot(vel_dir, desired_dir))
    else:
        vel_dir = np.zeros(2, dtype=float)
        cos_vel_desired = 1.0  # álló helyzetben nincs irányhiba

    # ───────────────────────────────────────────────────────────────────
    # 0-cell (EMPTY & még nem járt) közeli lassítás
    # ───────────────────────────────────────────────────────────────────
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

    force_slow_down = (
        ((not has_adjacent_zero_4dir) and has_adjacent_zero_8dir) or
        ((not has_adjacent_zero_8dir) and visible_zero_reachable)
    )

    # ───────────────────────────────────────────────────────────────────
    # Turn-aware braking: turn_soon / turn_med flag a mode-ból
    # ───────────────────────────────────────────────────────────────────
    turn_soon = ("turn_soon" in mode)
    turn_med = ("turn_med" in mode)

    force_brake_for_turn = False
    # Kanyar miatt általános lassítás
    if turn_soon and speed_now > 0.5:
        force_brake_for_turn = True
    elif turn_med and speed_now > 2.0:
        force_brake_for_turn = True

    # Ha nagyon közel a cél, de túl gyorsak vagyunk, fékezzünk
    if dist_grid <= 2 and speed_now > 2.0:
        force_brake_for_turn = True

    # Ha a sebesség rossz irányba mutat (hátra vagy erősen oldalra), erős fék
    if speed_now > 0.0:
        if cos_vel_desired < 0.0:  # gyakorlatilag hátrafelé megyünk a célhoz képest
            force_brake_for_turn = True
        elif turn_soon and cos_vel_desired < 0.5:
            force_brake_for_turn = True

    # Emergency braking pass: először próbáljunk kifejezetten fékezni,
    # mielőtt bonyolultabb pontozásba kezdünk.
    if (force_slow_down or force_brake_for_turn) and (vx != 0 or vy != 0):
        brake_candidates = [
            (-int(np.sign(vx)), -int(np.sign(vy))),
            (-int(np.sign(vx)), 0),
            (0, -int(np.sign(vy))),
            (0, 0),
        ]
        for ax, ay in brake_candidates:
            nvx, nvy = vx + ax, vy + ay
            next_pos = p + v + np.array([ax, ay], dtype=int)
            if not brakingOk(nvx, nvy, rSafe):
                continue
            if not directional_ok(nvx, nvy):
                continue
            if not validLineLocal(state, p, next_pos):
                continue
            if any(np.all(next_pos == q.pos) for q in state.players):
                continue
            return ax, ay

    # ───────────────────────────────────────────────────────────────────
    # Célsebesség (target_speed) meghatározása
    # ───────────────────────────────────────────────────────────────────
    max_safe = max(1.0, math.sqrt(2 * max(0, rSafe)))
    max_safe = min(max_safe, 2.0)  # kemény felső korlát, hogy ne őrüljön meg

    if force_slow_down:
        target_speed = 1.0
    else:
        in_bounds_target = (0 <= tx < H and 0 <= ty < W)
        target_is_hazard = (
            in_bounds_target and world.is_hazard(tx, ty)
        )

        if (in_bounds_target and
            world.visited_count[tx, ty] == 0 and not target_is_hazard):
            # új, nem-hazard cella -> mehet gyorsabban
            target_speed = max_safe
        elif mode.startswith("explore_frontier") or mode.startswith("local_explore"):
            target_speed = max(1.0, 0.8 * max_safe)
        elif target_is_hazard:
            target_speed = max(1.0, 0.5 * max_safe)
        elif mode.startswith("go_goal"):
            target_speed = max(1.5, 0.9 * max_safe)
        else:
            target_speed = max(1.5, 0.7 * max_safe)

    # extra lassítás kanyar távolság alapján
    if turn_soon:
        target_speed = min(target_speed, 1.0)
    elif turn_med:
        target_speed = min(target_speed, 1.5)

    # felület alapú módosítások (homok/olaj)
    if 0 <= tx < H and 0 <= ty < W:
        if world.is_sand(tx, ty) or world.is_sand(ax_agent, ay_agent):
            target_speed = min(target_speed, 1.0)
        if world.is_oil(tx, ty) or world.is_oil(ax_agent, ay_agent):
            target_speed = min(target_speed, 0.6)

    # további clamp, ha nagyon rossz az irány
    if speed_now > 0.0:
        if cos_vel_desired < 0.3:
            target_speed = min(target_speed, 1.0)
        elif cos_vel_desired < 0.7:
            target_speed = min(target_speed, 1.5)

    # Globális kemény sebességkorlát is
    hard_speed_cap = max_safe + 0.5

    # Corner-zóna: közel a célcellához vagy egy közelgő kanyarhoz
    corner_zone = (dist_grid <= 4) or turn_soon or turn_med

    # ───────────────────────────────────────────────────────────────────
    # Near-target snap: ha közel vagyunk, próbáljunk pont a célcellára állni
    # (CSAK ha a célcellánk NEM a jelenlegi cella!)
    # ───────────────────────────────────────────────────────────────────
    def generate_candidates():
        for ax in (-1, 0, 1):
            for ay in (-1, 0, 1):
                yield ax, ay

    if 0 < dist_grid <= 3:
        snap_best = None
        for ax, ay in generate_candidates():
            nvx, nvy = vx + ax, vy + ay
            if not brakingOk(nvx, nvy, rSafe):
                continue
            if not directional_ok(nvx, nvy):
                continue
            speed_next = float(math.hypot(nvx, nvy))
            if speed_next > hard_speed_cap:
                continue
            if corner_zone:
                if dist_grid <= 2 and speed_next > 1.0:
                    continue
                elif dist_grid <= 3 and speed_next > 1.5:
                    continue

            next_pos = p + v + np.array([ax, ay], dtype=int)
            nx, ny = int(next_pos[0]), int(next_pos[1])
            if nx == tx and ny == ty:
                if not validLineLocal(state, p, next_pos):
                    continue
                if any(np.all(next_pos == q.pos) for q in state.players):
                    continue
                score = speed_next
                if snap_best is None or score < snap_best[0]:
                    snap_best = (score, (ax, ay))
        if snap_best is not None:
            return snap_best[1]

    # ───────────────────────────────────────────────────────────────────
    # Teljes pontozásos keresés, erős backtracking büntetéssel
    # ───────────────────────────────────────────────────────────────────
    best = None
    last_pos = world.last_pos

    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            nvx, nvy = vx + ax, vy + ay
            if not brakingOk(nvx, nvy, rSafe):
                continue
            if not directional_ok(nvx, nvy):
                continue

            speed_next = float(math.hypot(nvx, nvy))
            if speed_next > hard_speed_cap:
                continue

            if corner_zone:
                if dist_grid <= 2 and speed_next > 1.0:
                    continue
                elif dist_grid <= 3 and speed_next > 1.5:
                    continue
                elif speed_next > 2.0:
                    continue

            next_pos = p + v + np.array([ax, ay], dtype=int)
            nx, ny = int(next_pos[0]), int(next_pos[1])

            if not (0 <= nx < H and 0 <= ny < W):
                continue
            if not validLineLocal(state, p, next_pos):
                continue
            if any(np.all(next_pos == q.pos) for q in state.players):
                continue

            if last_pos is not None and (nx, ny) == last_pos:
                continue

            dist_cell = float(
                np.linalg.norm(
                    next_pos.astype(float) - np.array(target_cell, dtype=float)
                )
            )

            speed_pen = abs(speed_next - target_speed)

            if speed_next > 0.0:
                vel_dir_next = np.array([nvx, nvy], dtype=float) / max(speed_next, 1e-9)
                cos_next = float(np.dot(vel_dir_next, desired_dir))
            else:
                cos_next = 1.0

            heading_weight = 1.0
            if turn_soon:
                heading_weight = 3.0
            elif turn_med:
                heading_weight = 1.8

            heading_pen = (1.0 - cos_next) * heading_weight

            if cos_next < 0.0 and speed_next > 0.0:
                continue

            vc = float(world.visited_count[nx, ny])
            if mode.startswith("go_goal"):
                visit_pen = 10.0 * vc
            elif mode.startswith("explore_frontier"):
                visit_pen = 50.0 * vc
            else:
                visit_pen = 100.0 * vc

            hazard_pen = 0.0
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

            score = (
                1.5 * dist_cell
                + 2.5 * speed_pen
                + heading_pen
                + visit_pen
                + hazard_pen
            )
            if best is None or score < best[0]:
                best = (score, (ax, ay))

    if best is not None:
        return best[1]

    # ───────────────────────────────────────────────────────────────────
    # Végső fallback:
    #  - ha mozgunk, először fékezés, aztán esetleg 0 0
    #  - ha állunk, próbáljunk valamilyen nem-null gyorsulást találni,
    #    csak végső esetben maradjunk 0 0-n.
    # ───────────────────────────────────────────────────────────────────
    # Mozgás közben: fékezős irány, majd (0,0) ha nagyon muszáj
    if vx != 0 or vy != 0:
        for ax, ay in ((-int(np.sign(vx)), -int(np.sign(vy))), (0, 0)):
            nvx, nvy = vx + ax, vy + ay
            nxt = p + v + np.array([ax, ay], dtype=int)
            nx, ny = int(nxt[0]), int(nxt[1])
            if not brakingOk(nvx, nvy, rSafe):
                continue
            if not directional_ok(nvx, nvy):
                continue
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            if not validLineLocal(state, p, nxt):
                continue
            if any(np.all(nxt == q.pos) for q in state.players):
                continue
            # ne lépjünk vissza last_pos-ra, ha nem muszáj
            if world.last_pos is not None and (nx, ny) == world.last_pos:
                continue
            return int(ax), int(ay)
        return (0, 0)

    # Álló helyzetből: próbáljunk valamerre elmozdulni
    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            if ax == 0 and ay == 0:
                continue
            nvx, nvy = vx + ax, vy + ay
            nxt = p + v + np.array([ax, ay], dtype=int)
            nx, ny = int(nxt[0]), int(nxt[1])
            if not brakingOk(nvx, nvy, rSafe):
                continue
            if not directional_ok(nvx, nvy):
                continue
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            if not validLineLocal(state, p, nxt):
                continue
            if any(np.all(nxt == q.pos) for q in state.players):
                continue
            if world.last_pos is not None and (nx, ny) == world.last_pos:
                continue
            return int(ax), int(ay)

    # Ha semmi nem megy, maradjunk
    return (0, 0)

# ────────────────────────────────────────────────────────────────────────────────
# ASCII dump
# ────────────────────────────────────────────────────────────────────────────────

def dump_ascii(world: WorldModel, policy: LeftWallPolicy, state: State, mode: str) -> None:
    """
    Kirajzolja az aktuális world.known_map-et logs/map_dump.txt fájlba ASCII-ben.
    """
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

    hdr: List[str] = []
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
        os.makedirs(log_dir, exist_ok=True)

    mode_flag = "a"
    if not world._dump_initialized:
        mode_flag = "w"
        world._dump_initialized = True
    with open(world.dump_file, mode_flag, encoding="utf-8") as f:
        f.write("\n".join(lines))

# ────────────────────────────────────────────────────────────────────────────────
# Decision loop
# ────────────────────────────────────────────────────────────────────────────────

def calculateMove(world: WorldModel, policy: LeftWallPolicy, state: State) -> Tuple[int, int]:
    """
    Egy lépés döntése:
      - frissíti a world.known_map-et az új megfigyeléssel,
      - növeli a visited_count-ot az aktuális cellán,
      - kéri a high-level policy-től a következő target_cell-et és mode-ot,
      - low-level controllerrel kiválasztja az (ax, ay) gyorsulást.
    """
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
    """
    Judge entry point.
    """
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

        ax = -1 if ax < -1 else (1 if ax > 1 else int(ax))
        ay = -1 if ay < -1 else (1 if ay > 1 else int(ay))
        print(f"{ax} {ay}", flush=True)


if __name__ == "__main__":
    main()
