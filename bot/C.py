import sys
import enum
import numpy as np
import heapq
from collections import deque
from typing import Optional, NamedTuple, List, Tuple


# ────────────────────────────────────────────────────────────────────────────────
# Cell type definitions (must match the environment)
# ────────────────────────────────────────────────────────────────────────────────

class CellType(enum.Enum):
    """
    CellType values:
      - WALL: wall, not passable
      - EMPTY: empty road
      - START: starting cell
      - UNKNOWN: internal "never seen" marker (fog-of-war)
      - NOT_VISIBLE: input value for "currently not visible"
      - OIL, SAND: hazardous cells (we avoid them if possible)
      - GOAL: finish cell
    """
    WALL = -1
    EMPTY = 0
    START = 1
    UNKNOWN = -2          # belső fog-of-war, nem a pálya 2-es kódja
    NOT_VISIBLE = 3
    OIL = 91
    SAND = 92
    GOAL = 100


# Hazard cells that we try to avoid
HAZARDS = {CellType.OIL.value, CellType.SAND.value}


class Player(NamedTuple):
    """
    Single player state: position + velocity.
    """
    x: int
    y: int
    vel_x: int
    vel_y: int

    @property
    def pos(self) -> np.ndarray:
        """Return position as [row, col] numpy array."""
        return np.array([self.x, self.y])

    @property
    def vel(self) -> np.ndarray:
        """Return velocity as [vx, vy] numpy array."""
        return np.array([self.vel_x, self.vel_y])


class Circuit(NamedTuple):
    """
    Static circuit information: size, player count, visibility radius.
    """
    track_shape: tuple[int, int]
    num_players: int
    visibility_radius: int


class State(NamedTuple):
    """
    One full observation from judge + our own agent.
    """
    circuit: Circuit
    visible_track: np.ndarray
    players: list[Player]
    agent: Player


# ────────────────────────────────────────────────────────────────────────────────
# Global world model
# ────────────────────────────────────────────────────────────────────────────────

class GlobalMap:
    """
    Global HxW map that stores everything we have ever seen.

    self.grid:
      - initially: UNKNOWN (-2) everywhere,
      - on each step: wherever visible_track != NOT_VISIBLE, we copy
        that value into self.grid.
    """

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        self.grid = np.full(shape, CellType.UNKNOWN.value, dtype=int)

    def update(self, visible_track: np.ndarray):
        """
        Update global grid with the latest visible_track:
          - only overwrite cells where visible_track != NOT_VISIBLE.
        """
        mask = (visible_track != CellType.NOT_VISIBLE.value)
        self.grid[mask] = visible_track[mask]


# ────────────────────────────────────────────────────────────────────────────────
# Target selection: nearest UNKNOWN and GOAL
# ────────────────────────────────────────────────────────────────────────────────

def find_nearest_unknown(
    start: tuple[int, int],
    gmap: GlobalMap,
    avoid_hazards: bool = True
) -> Optional[tuple[int, int]]:
    """
    8-directional BFS to find the nearest completely UNKNOWN (-2) cell.

    Parameters:
      - start: starting (x, y) cell
      - gmap: global map
      - avoid_hazards: if True, OIL/SAND cells are treated as blocked;
                       if False, BFS is allowed to walk through them.

    Returns:
      - coordinates of the nearest UNKNOWN cell, or
      - None if no UNKNOWN cell is reachable.
    """
    queue = deque([start])
    visited = {start}

    if gmap.grid[start] == CellType.UNKNOWN.value:
        return start

    H, W = gmap.shape

    # 8 directions (including diagonals)
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            if (nx, ny) in visited:
                continue

            val = gmap.grid[nx, ny]

            # Solid wall
            if val == CellType.WALL.value:
                continue

            # Optional hazard avoidance
            if avoid_hazards and val in HAZARDS:
                continue

            # Found an unknown cell -> this is our discovery target
            if val == CellType.UNKNOWN.value:
                return (nx, ny)

            visited.add((nx, ny))
            queue.append((nx, ny))

    return None


# ────────────────────────────────────────────────────────────────────────────────
# Path planning with A*
# ────────────────────────────────────────────────────────────────────────────────

def run_astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    gmap: GlobalMap,
    avoid_hazards: bool
) -> List[tuple[int, int]]:
    """
    A* pathfinding from start to goal on the global map.

    Parameters:
      - start, goal: cell coordinates (x, y)
      - gmap: global map
      - avoid_hazards: if True, we treat OIL/SAND as blocked (like walls);
                       if False, we allow them but at high cost.

    Cost model:
      - WALL: impassable
      - UNKNOWN and normal cells (EMPTY, START, GOAL, etc): cost = 1
      - OIL/SAND (if allowed): cost = 20

    Returns:
      - path: [start, step1, ..., goal] if reachable,
      - empty list [] if no path is found.
    """
    H, W = gmap.shape
    (sx, sy) = start
    (gx, gy) = goal

    if not (0 <= sx < H and 0 <= sy < W):
        return []
    if not (0 <= gx < H and 0 <= gy < W):
        return []
    if gmap.grid[gx, gy] == CellType.WALL.value:
        return []

    pq: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
    came_from: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    cost_so_far: dict[Tuple[int, int], float] = {start: 0.0}

    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while pq:
        _, current = heapq.heappop(pq)
        if current == goal:
            break

        cx, cy = current

        for dx, dy in directions:
            if dx == 0 and dy == 0:
                continue
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < H and 0 <= ny < W):
                continue

            val = gmap.grid[nx, ny]

            if val == CellType.WALL.value:
                continue
            if avoid_hazards and val in HAZARDS:
                continue

            if val in HAZARDS:
                step_cost = 20.0
            else:
                step_cost = 1.0

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
# I/O with judge
# ────────────────────────────────────────────────────────────────────────────────

def read_initial_observation() -> Circuit:
    """
    Read the first line:
      H W num_players visibility_radius
    and return a Circuit object.
    """
    line = sys.stdin.readline()
    if not line:
        return None  # type: ignore
    H, W, num_players, visibility_radius = map(int, line.split())
    return Circuit((H, W), num_players, visibility_radius)


def read_observation(old_state: State) -> Optional[State]:
    """
    Each turn:
      - read our own position + velocity,
      - read all other players' positions,
      - read the (2R+1)x(2R+1) local window,
    and build a full HxW visible_track with NOT_VISIBLE everywhere else.
    """
    line = sys.stdin.readline()
    if not line or line.strip() == '~~~END~~~':
        return None

    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)

    players: List[Player] = []
    circuit_data = old_state.circuit

    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, sys.stdin.readline().split())
        players.append(Player(pposx, pposy, 0, 0))

    visible_track = np.full(
        circuit_data.track_shape,
        CellType.NOT_VISIBLE.value,
        dtype=int
    )

    R = circuit_data.visibility_radius
    H, W = circuit_data.track_shape

    for i in range(2 * R + 1):
        raw_line = sys.stdin.readline()
        if not raw_line:
            break
        vals = [int(a) for a in raw_line.split()]

        x = posx - R + i
        if x < 0 or x >= H:
            continue

        y_start = posy - R
        y_end = y_start + 2 * R + 1

        line_slice_start = 0
        line_slice_end = len(vals)

        target_y_start = max(0, y_start)
        target_y_end = min(W, y_end)

        if y_start < 0:
            line_slice_start = -y_start
        if y_end > W:
            line_slice_end -= (y_end - W)

        if line_slice_start < line_slice_end:
            visible_track[x, target_y_start:target_y_end] = vals[
                line_slice_start:line_slice_end
            ]

    return old_state._replace(
        visible_track=visible_track, players=players, agent=agent
    )


# ────────────────────────────────────────────────────────────────────────────────
# Path post-processing: skip occupied cells
# ────────────────────────────────────────────────────────────────────────────────

def choose_next_free_cell_on_path(
    path: List[Tuple[int, int]],
    state: State
) -> Optional[Tuple[int, int]]:
    """
    Given a path [start, step1, step2, ...], choose the next waypoint that is
    NOT currently occupied by another player.

    We skip cells that are currently taken by other players, because if we
    target them directly, the controller will try to stand exactly on a
    blocked cell and end up just braking in front of it.

    Returns:
      - the first free cell from path[1:], or
      - None if every cell on the path (except start) is currently occupied.
    """
    if len(path) < 2:
        return None

    my_pos = (state.agent.x, state.agent.y)
    occupied = {(p.x, p.y) for p in state.players}
    # do not treat our own cell as occupied
    if my_pos in occupied:
        occupied.remove(my_pos)

    for cell in path[1:]:
        if cell not in occupied:
            return cell

    return None


# ────────────────────────────────────────────────────────────────────────────────
# Decision logic: target selection + acceleration
# ────────────────────────────────────────────────────────────────────────────────

def calculate_move_logic(state: State, gmap: GlobalMap) -> Tuple[int, int]:
    """
    Main decision function per turn.

    Steps:
      1) Update global map from current visible_track.
      2) If any GOAL cell is visible:
           - choose the closest UN-OCCUPIED GOAL (if any),
             otherwise the closest GOAL (all are occupied),
           - plan path with A* (first without hazards, then with hazards),
           - select the first *unoccupied* cell along that path,
           - steer towards that cell.
      3) If no GOAL is visible:
           - find nearest UNKNOWN cell with 8-directional BFS:
               * first without hazards,
               * if none reachable, then allowing hazards.
           - plan an A* path there (same hazard logic),
           - again select first unoccupied cell on that path.
      4) If no valid path / waypoint is found:
           - gently brake (try to slow down / stop).
    """
    my_pos = (state.agent.x, state.agent.y)

    # 1) Update global map with fresh visible info
    gmap.update(state.visible_track)

    # 2) Try to go directly for GOAL if we see it
    goals = np.argwhere(gmap.grid == CellType.GOAL.value)
    if len(goals) > 0:
        # occupied positions (other players)
        occupied = {(p.x, p.y) for p in state.players}
        if my_pos in occupied:
            occupied.remove(my_pos)

        # Build candidate list: (is_occupied, distance, (gx, gy))
        candidates: List[Tuple[bool, int, Tuple[int, int]]] = []
        for gr, gc in goals:
            cell = (int(gr), int(gc))
            dist = abs(gr - my_pos[0]) + abs(gc - my_pos[1])
            is_occ = cell in occupied
            candidates.append((is_occ, dist, cell))

        # Sort: free goals first (is_occupied=False), then nearest by distance
        candidates.sort(key=lambda t: (t[0], t[1]))
        goal_tuple = candidates[0][2]

        # A* to the chosen goal
        path = run_astar(my_pos, goal_tuple, gmap, avoid_hazards=True)
        if len(path) < 2:
            path = run_astar(my_pos, goal_tuple, gmap, avoid_hazards=False)

        if path:
            next_cell = choose_next_free_cell_on_path(path, state)
            if next_cell is not None:
                return _pd_control_to_cell(state, next_cell)
        # if no free cell along goal path, fall back to exploration / braking

    # 3) Exploration: nearest UNKNOWN
    target = find_nearest_unknown(my_pos, gmap, avoid_hazards=True)
    allow_hazards_for_explore = False

    if target is None:
        target = find_nearest_unknown(my_pos, gmap, avoid_hazards=False)
        allow_hazards_for_explore = True

    if target is not None:
        path = run_astar(
            my_pos,
            target,
            gmap,
            avoid_hazards=not allow_hazards_for_explore
        )
        if len(path) < 2 and allow_hazards_for_explore:
            path = run_astar(my_pos, target, gmap, avoid_hazards=False)

        if path:
            next_cell = choose_next_free_cell_on_path(path, state)
            if next_cell is not None:
                return _pd_control_to_cell(state, next_cell)

    # 4) No target / path -> brake towards zero velocity
    vel = state.agent.vel
    ax = int(np.clip(-vel[0], -1, 1))
    ay = int(np.clip(-vel[1], -1, 1))
    return (ax, ay)


def _pd_control_to_cell(state: State, next_cell: tuple[int, int]) -> Tuple[int, int]:
    """
    Local PD-like controller that chooses (ax, ay) towards a target cell, while
    avoiding stepping exactly onto other players.

    It:
      - iterates over all accelerations (ax, ay) in {-1, 0, 1}²,
      - computes new_vel = vel + a and new_pos = pos + new_vel,
      - discards candidates where new_pos coincides with any other player,
      - scores remaining moves by distance to target + speed penalty + small
        acceleration penalty,
      - picks the move with the smallest score,
      - if *all* candidates collide, it brakes instead.
    """
    desired_pos = np.array(next_cell, dtype=float)
    current_pos = state.agent.pos.astype(float)
    current_vel = state.agent.vel.astype(float)
    other_positions = [p.pos for p in state.players]

    best_score = float('inf')
    best_acc = (0, 0)

    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            new_vel = current_vel + np.array([ax, ay], dtype=float)
            new_pos = current_pos + new_vel
            new_pos_int = new_pos.astype(int)

            # Collision avoidance with other players
            if any(np.array_equal(new_pos_int, op) for op in other_positions):
                continue

            dist = float(np.linalg.norm(desired_pos - new_pos))
            speed = float(np.linalg.norm(new_vel))
            desired_speed = min(4.0, dist)
            speed_penalty = abs(speed - desired_speed)
            acc_mag = abs(ax) + abs(ay)
            score = dist + 0.5 * speed_penalty + 0.1 * acc_mag

            if score < best_score:
                best_score = score
                best_acc = (ax, ay)

    if best_score == float('inf'):
        # If every move would collide, brake instead
        vel = state.agent.vel
        ax = int(np.clip(-vel[0], -1, 1))
        ay = int(np.clip(-vel[1], -1, 1))
        return (ax, ay)

    return best_acc


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    """
    Entry point:
      - prints READY,
      - reads initial circuit info,
      - builds GlobalMap,
      - then in a loop:
          * read_observation,
          * calculate_move_logic,
          * print (ax, ay).
    """
    print('READY', flush=True)
    circuit = read_initial_observation()
    if circuit is None:
        return

    gmap = GlobalMap(circuit.track_shape)
    state: Optional[State] = State(circuit, None, [], None)  # type: ignore

    while True:
        state = read_observation(state)
        if state is None:
            break
        dx, dy = calculate_move_logic(state, gmap)
        print(f'{dx} {dy}', flush=True)


if __name__ == "__main__":
    main()
