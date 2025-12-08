import sys
import enum
import numpy as np

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, NamedTuple, List, Tuple, Dict, Set


# ────────────────────────────────────────────────────────────────────────────────
# Cell types – must match the server's grid_race_env.CellType values
# GOAL / START / WALL / EMPTY / NOT_VISIBLE / OIL / SAND
# (OIL=91, SAND=92 according to the task description.)
# ────────────────────────────────────────────────────────────────────────────────

class CellType(enum.Enum):
    GOAL = 100
    START = 1
    WALL = -1
    UNKNOWN = 2         # our global "never seen" marker
    EMPTY = 0
    NOT_VISIBLE = 3     # local "currently not visible" marker from the judge
    OIL = 91
    SAND = 92


class Player(NamedTuple):
    """
    Represents a player (including the agent):
    - x, y: position on the grid
    - vel_x, vel_y: current velocity components
    """
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
    """
    Basic information about the circuit:
    - track_shape: (H, W) size of the full map
    - num_players: number of players in the race
    - visibility_radius: R, the judge sends a (2R+1)x(2R+1) window around the agent
    """
    track_shape: tuple[int, int]
    num_players: int
    visibility_radius: int


@dataclass
class BotState:
    """
    Internal state of our bot, kept across turns.

    - circuit: static track info
    - visible_track: global map built from all past observations
    - players: last known positions of other players
    - agent: our own Player (position + velocity)
    - visit_counts: how many times we have visited each cell
    - visited: set of cells (r, c) where the agent has already been
    - dfs_stack: depth-first exploration stack of target cells to visit
    - step: global time step counter
    - target_cell: current DFS target (we try to go there before picking a new one)
    """
    circuit: Circuit
    visible_track: np.ndarray
    players: List[Player]
    agent: Player
    visit_counts: np.ndarray
    visited: Set[Tuple[int, int]] = field(default_factory=set)
    dfs_stack: List[Tuple[int, int]] = field(default_factory=list)
    step: int = 0
    target_cell: Optional[Tuple[int, int]] = None


# ────────────────────────────────────────────────────────────────────────────────
# Input reading
# ────────────────────────────────────────────────────────────────────────────────

def read_initial_observation() -> Circuit:
    """
    Reads the very first line from stdin:
    H W num_players visibility_radius
    and returns a Circuit object describing the track.
    """
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)


def read_observation(circuit: Circuit,
                     old_state: Optional[BotState]) -> Optional[BotState]:
    """
    Reads one full observation step from stdin and updates / creates the BotState.

    Protocol per turn:
    - 1 line: posx posy velx vely  (our agent)
    - num_players lines: pposx pposy  (other players)
    - (2R+1) lines: local track window centered at agent (may contain NOT_VISIBLE)

    We:
    - create the BotState on the first call (UNKNOWN everywhere initially),
    - then on each call integrate the local window into the global visible_track,
      BUT we never overwrite with NOT_VISIBLE (3), so we keep old knowledge,
    - update visit_counts, visited set, and step counter.
    """
    line = input().strip()
    if line == '~~~END~~~':
        return None

    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)

    if old_state is None:
        # First turn: initialize global map and visit counters.
        H, W = circuit.track_shape
        visible_track = np.full((H, W), CellType.UNKNOWN.value, dtype=int)
        visit_counts = np.zeros((H, W), dtype=np.int32)
        state = BotState(
            circuit=circuit,
            visible_track=visible_track,
            players=[],
            agent=agent,
            visit_counts=visit_counts,
        )
    else:
        # Reuse existing state; only replace agent and players, and update the map.
        state = old_state
        state.agent = agent

    # Read other players' positions.
    players: List[Player] = []
    for _ in range(circuit.num_players):
        pposx, pposy = map(int, input().split())
        # We don't know their velocities; we just store (0, 0) for others.
        players.append(Player(pposx, pposy, 0, 0))
    state.players = players

    # Integrate the local track window into the global visible_track.
    R = circuit.visibility_radius
    H, W = circuit.track_shape

    for i in range(2 * R + 1):
        row_values = [int(a) for a in input().split()]
        x = posx - R + i
        if x < 0 or x >= H:
            # Outside global map vertically.
            continue

        y_start = posy - R
        y_end = y_start + 2 * R + 1

        # Clip horizontally to [0, W).
        if y_start < 0:
            row_values = row_values[-y_start:]
            y_start = 0
        if y_end > W:
            row_values = row_values[:-(y_end - W)]
            y_end = W

        row_arr = np.array(row_values, dtype=int)

        # We DO NOT overwrite cells with NOT_VISIBLE (3), so we keep old knowledge.
        mask_visible = row_arr != CellType.NOT_VISIBLE.value
        if np.any(mask_visible):
            segment = state.visible_track[x, y_start:y_end]
            segment[mask_visible] = row_arr[mask_visible]
            state.visible_track[x, y_start:y_end] = segment

    # Update visitation info for the agent's current cell.
    state.visit_counts[posx, posy] += 1
    state.visited.add((posx, posy))
    state.step += 1

    return state


# ────────────────────────────────────────────────────────────────────────────────
# Helper functions: traversability, neighbors, path planning
# ────────────────────────────────────────────────────────────────────────────────

def traversable(cell_value: int) -> bool:
    """
    Returns True if a cell value is considered traversable for path planning.

    - WALL (-1)       → not traversable
    - UNKNOWN (2)     → treated as not traversable (we do NOT drive blindly)
    - all other >= 0  → traversable (EMPTY, START, GOAL, OIL, SAND)
    """
    if cell_value == CellType.WALL.value:
        return False
    if cell_value == CellType.UNKNOWN.value:
        return False
    return cell_value >= 0


def neighbors4(pos: Tuple[int, int],
               shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Returns 4-connected neighbors (up, down, left, right) within the given grid.
    """
    (x, y) = pos
    H, W = shape
    out: List[Tuple[int, int]] = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < H and 0 <= ny < W:
            out.append((nx, ny))
    return out


def bfs_to_goal(track: np.ndarray,
                start: Tuple[int, int],
                goals: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    """
    Simple BFS to the nearest GOAL cell.

    - track: global map (int matrix)
    - start: starting cell (r, c)
    - goals: set of goal cells

    Returns:
    - path: list of cells from start to a goal (inclusive),
    - None: if no goal is reachable.
    """
    H, W = track.shape
    q: deque[Tuple[int, int]] = deque()
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited: Set[Tuple[int, int]] = set()

    q.append(start)
    visited.add(start)

    while q:
        cur = q.popleft()
        if cur in goals:
            # Reconstruct path.
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return path

        for nb in neighbors4(cur, (H, W)):
            if nb in visited:
                continue
            if not traversable(track[nb]):
                continue
            visited.add(nb)
            parent[nb] = cur
            q.append(nb)

    return None


def bfs_to_target(track: np.ndarray,
                  start: Tuple[int, int],
                  target: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    BFS shortest path from start to a single target cell.

    Used for DFS-style coverage: we pick a 'target' cell from our dfs_stack
    and then use BFS to get a path to it on the known traversable cells.
    """
    H, W = track.shape

    if not traversable(track[target]):
        return None

    q: deque[Tuple[int, int]] = deque()
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited: Set[Tuple[int, int]] = set()

    q.append(start)
    visited.add(start)

    while q:
        cur = q.popleft()
        if cur == target:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return path

        for nb in neighbors4(cur, (H, W)):
            if nb in visited:
                continue
            if not traversable(track[nb]):
                continue
            visited.add(nb)
            parent[nb] = cur
            q.append(nb)

    return None


def valid_line(state: BotState,
               pos1: np.ndarray,
               pos2: np.ndarray) -> bool:
    """
    Checks if the straight line from pos1 to pos2 is fully traversable in our
    current global map (visible_track).

    This mimics the judge's line-of-motion check:
    - we sample the line cell-by-cell along x and y,
    - if at any step we see that BOTH candidate cells are non-traversable,
      then the move is invalid.
    """
    track = state.visible_track
    if (np.any(pos1 < 0) or np.any(pos2 < 0)
            or np.any(pos1 >= track.shape)
            or np.any(pos2 >= track.shape)):
        return False

    diff = pos2 - pos1

    # Vertical sweep (consider walls stacked N-S).
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))  # left or right
        for i in range(abs(int(diff[0])) + 1):
            x = int(pos1[0] + i * d)
            y = pos1[1] + i * slope * d
            y_ceil = int(np.ceil(y))
            y_floor = int(np.floor(y))
            if (not traversable(track[x, y_ceil])
                    and not traversable(track[x, y_floor])):
                return False

    # Horizontal sweep (consider walls stacked E-W).
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))  # up or down
        for i in range(abs(int(diff[1])) + 1):
            x = pos1[0] + i * slope * d
            y = int(pos1[1] + i * d)
            x_ceil = int(np.ceil(x))
            x_floor = int(np.floor(x))
            if (not traversable(track[x_ceil, y])
                    and not traversable(track[x_floor, y])):
                return False

    return True


def cell_fully_explored(cell: Tuple[int, int],
                        track: np.ndarray,
                        visited: Set[Tuple[int, int]]) -> bool:
    """
    Returns True if all traversable neighbors of 'cell' have already been visited.
    Used to clean up DFS targets that are 'done'.
    """
    H, W = track.shape
    for nb in neighbors4(cell, (H, W)):
        if not traversable(track[nb]):
            continue
        if nb not in visited:
            return False
    return True


def update_dfs_stack(state: BotState) -> None:
    """
    Updates the DFS exploration stack:

    - from the agent's current cell, we add all traversable, not-yet-visited
      neighbors that are not already in the stack;
    - then we prune the top of the stack if that cell is already fully explored
      (all traversable neighbors have been visited).
    """
    track = state.visible_track
    H, W = track.shape
    cur_cell = (int(state.agent.x), int(state.agent.y))

    # 1) Add new neighbors of current cell to the DFS stack (depth-first style).
    for nb in neighbors4(cur_cell, (H, W)):
        if not traversable(track[nb]):
            continue
        if nb in state.visited:
            continue
        if nb in state.dfs_stack:
            continue
        state.dfs_stack.append(nb)

    # 2) Remove cells from top of stack that are fully explored.
    while state.dfs_stack:
        top = state.dfs_stack[-1]
        # If we haven't visited this cell yet, we certainly still want to go there.
        if top not in state.visited:
            break
        # If visited and all traversable neighbors are already visited, pop it.
        if cell_fully_explored(top, track, state.visited):
            state.dfs_stack.pop()
        else:
            break


# ────────────────────────────────────────────────────────────────────────────────
# Decision function: calculate_move
# ────────────────────────────────────────────────────────────────────────────────

def calculate_move(rng: np.random.Generator, state: BotState) -> Tuple[int, int]:
    """
    Chooses the acceleration (dx, dy), where dx, dy ∈ {-1, 0, 1}.

    Strategy:
    1. Use the global memory map (visible_track) built from all past turns.
    2. If any GOAL cell is known, we BFS directly to the closest GOAL.
    3. Otherwise perform DFS-style coverage:
       - maintain a stack of target cells (dfs_stack),
       - from the current cell we add new neighbors (not yet visited),
       - we keep a persistent target_cell (top of dfs_stack) and move toward it
         with BFS, avoiding random backtracking.
    4. For the chosen BFS path, we select the acceleration that:
       - keeps the move valid (valid_line, no collisions),
       - moves closer to the next cell in the path,
       - avoids oil/sand if possible,
       - penalizes high speed and revisiting heavily visited cells,
         so we only go back to known areas if we must.
    """
    track = state.visible_track
    H, W = track.shape

    self_pos = state.agent.pos.astype(int)
    self_vel = state.agent.vel.astype(int)
    start = (int(self_pos[0]), int(self_pos[1]))

    # If we happened to reach our previous DFS target, forget it
    # so we can pick a new one if needed.
    if state.target_cell is not None and start == state.target_cell:
        state.target_cell = None

    # Update DFS stack (discover new neighbors, prune fully explored ones).
    update_dfs_stack(state)

    # 1) If we know any GOAL cells, try to go there first.
    goal_indices = np.argwhere(track == CellType.GOAL.value)
    goals: Set[Tuple[int, int]] = set(
        (int(g[0]), int(g[1])) for g in goal_indices
    )

    path_to_follow: Optional[List[Tuple[int, int]]] = None

    if goals:
        path_to_follow = bfs_to_goal(track, start, goals)

    # 2) If no GOAL path, use DFS-style coverage.
    if path_to_follow is None:
        # Validate existing target_cell, if any.
        if state.target_cell is not None:
            path = bfs_to_target(track, start, state.target_cell)
            if path is None:
                # Target became unreachable (or map updated); discard it.
                state.target_cell = None
            else:
                path_to_follow = path

        # If we still don't have a path, pick a new target from the DFS stack.
        if path_to_follow is None:
            while state.dfs_stack:
                candidate = state.dfs_stack[-1]
                path = bfs_to_target(track, start, candidate)
                if path is None:
                    # Unreachable candidate → drop and try next.
                    state.dfs_stack.pop()
                    continue
                state.target_cell = candidate
                path_to_follow = path
                break

    # 3) If we have no path at all (no GOAL and DFS stack empty),
    #    try to stand still or fallback to any safe move.
    if path_to_follow is None or len(path_to_follow) == 0:
        # Try to simply keep velocity or stand still, if valid.
        stay_delta = np.array([0, 0], dtype=int)
        stay_vel = self_vel + stay_delta
        stay_pos = self_pos + stay_vel
        if valid_line(state, self_pos, stay_pos):
            return (0, 0)

        # Fallback: pick any valid safe move (very rare situation).
        valid_moves: List[Tuple[int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                delta = np.array([dx, dy], dtype=int)
                new_vel = self_vel + delta
                new_pos = self_pos + new_vel
                if not (0 <= new_pos[0] < H and 0 <= new_pos[1] < W):
                    continue
                if not valid_line(state, self_pos, new_pos):
                    continue
                if any(np.all(new_pos == p.pos) for p in state.players):
                    continue
                valid_moves.append((dx, dy))
        if valid_moves:
            return tuple(rng.choice(valid_moves))
        return (0, 0)

    # 4) We have a BFS path. Use its next cell as local movement target.
    if len(path_to_follow) >= 2:
        target_cell = np.array(path_to_follow[1], dtype=int)
    else:
        target_cell = np.array(path_to_follow[0], dtype=int)

    # ── Choose acceleration (dx, dy) toward target_cell ────────────────────────
    best_action: Optional[Tuple[int, int]] = None
    best_score: float = float('inf')

    MAX_SPEED_L1 = 4  # simple speed cap (|vx| + |vy|)

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            delta = np.array([dx, dy], dtype=int)
            new_vel = self_vel + delta
            speed_l1 = int(abs(new_vel[0]) + abs(new_vel[1]))
            if speed_l1 > MAX_SPEED_L1:
                continue

            new_pos = self_pos + new_vel

            # Stay inside map bounds.
            if not (0 <= new_pos[0] < H and 0 <= new_pos[1] < W):
                continue

            # Check line-of-motion validity (no passing through walls/unknown).
            if not valid_line(state, self_pos, new_pos):
                continue

            # Don't move onto another player's position.
            if any(np.all(new_pos == p.pos) for p in state.players):
                continue

            # Distance to the local target along the grid.
            diff = new_pos - target_cell
            dist2 = float(diff[0] * diff[0] + diff[1] * diff[1])

            # Penalty for bad terrain (we prefer EMPTY over SAND/OIL).
            cell_val = int(track[int(new_pos[0]), int(new_pos[1])])
            cell_penalty = 0.0
            if cell_val == CellType.SAND.value:
                cell_penalty += 3.0
            elif cell_val == CellType.OIL.value:
                cell_penalty += 5.0

            # Penalty for high speed: we don't want uncontrollable velocity.
            speed_penalty = 0.5 * speed_l1

            # Penalty for revisiting heavily visited cells:
            # this encourages wide coverage and avoids backtracking unless necessary.
            revisit_count = int(state.visit_counts[int(new_pos[0]), int(new_pos[1])])
            revisit_penalty = 2.0 * revisit_count

            score = dist2 + cell_penalty + speed_penalty + revisit_penalty

            if score < best_score:
                best_score = score
                best_action = (dx, dy)

    # If we found a good action, use it.
    if best_action is not None:
        return best_action

    # Fallback: try to stay if nothing else worked.
    if valid_line(state, self_pos, self_pos + self_vel):
        return (0, 0)

    # Final fallback: any valid move.
    valid_moves: List[Tuple[int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            delta = np.array([dx, dy], dtype=int)
            new_vel = self_vel + delta
            new_pos = self_pos + new_vel
            if not (0 <= new_pos[0] < H and 0 <= new_pos[1] < W):
                continue
            if not valid_line(state, self_pos, new_pos):
                continue
            if any(np.all(new_pos == p.pos) for p in state.players):
                continue
            valid_moves.append((dx, dy))
    if valid_moves:
        return tuple(rng.choice(valid_moves))
    return (0, 0)


# ────────────────────────────────────────────────────────────────────────────────
# main – communication loop with the judge
# ────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Main entry point of the bot:

    - prints "READY" (the judge waits for this),
    - reads the initial circuit description,
    - then in each turn:
        * reads the observation,
        * updates the BotState (global map + DFS structures),
        * calls calculate_move to get (dx, dy),
        * prints the chosen acceleration.
    """
    print('READY', flush=True)
    circuit = read_initial_observation()

    state: Optional[BotState] = None
    rng = np.random.default_rng(seed=1)

    while True:
        state = read_observation(circuit, state)
        if state is None:
            return
        delta = calculate_move(rng, state)
        print(f'{delta[0]} {delta[1]}', flush=True)


if __name__ == "__main__":
    main()
