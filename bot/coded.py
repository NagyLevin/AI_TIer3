import sys
import enum
import numpy as np

from collections import deque
from typing import Optional, NamedTuple


class CellType(enum.Enum):
    """
    Cell types that we care about in the client.

    We only explicitly use:
      - WALL: impassable cell on the track
      - NOT_VISIBLE: cells that the judge tells us we can't see right now

    Every other integer value in the map (0, 1, 2, 91, 92, 100, ...)
    is treated as a normal traversable track cell.
    """
    NOT_VISIBLE = 3
    WALL = -1


class Player(NamedTuple):
    """
    Player description in the client state.

    x, y      : integer grid position (row, column)
    vel_x,y   : current velocity components
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
    Static circuit information sent at the very beginning.

    track_shape       : (H, W) size of the whole map
    num_players       : number of cars on the track
    visibility_radius : radius R of the (2R+1)x(2R+1) observation window
    """
    track_shape: tuple[int, int]
    num_players: int
    visibility_radius: int


class State(NamedTuple):
    """
    Complete client-side state.

    circuit        : static info about the circuit
    visible_track  : our *global* knowledge of the track (H x W)
                     - starts as all NOT_VISIBLE
                     - every turn we merge the newly seen patch into it
    players        : approximate positions of all players
    agent          : our own player
    """
    circuit: Circuit
    visible_track: Optional[np.ndarray]
    players: list[Player]
    agent: Optional[Player]


# Global visit counter to encourage full exploration
VISIT_COUNTS: Optional[np.ndarray] = None

# Soft speed limit for planning (environment allows any speed, but we
# keep it moderate so that we can still steer and stop safely).
SPEED_LIMIT = 4


def read_initial_observation() -> Circuit:
    """
    Read H, W, num_players, visibility_radius from stdin.

    This is called once at the very beginning.
    """
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)


def read_observation(old_state: State) -> Optional[State]:
    """
    Read one full observation from stdin and update our global track view.

    The judge sends:
      - one line with our position and velocity,
      - num_players lines with approximate player positions,
      - (2R+1) lines describing the (2R+1)x(2R+1) observation window.

    We merge the observed patch into a persistent HxW "visible_track"
    so that we slowly uncover the whole map.
    """
    line = input()
    if line == '~~~END~~~':
        # End of game
        return None

    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)

    players: list[Player] = []
    circuit_data = old_state.circuit

    # Read other players' approximate positions
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, input().split())
        players.append(Player(pposx, pposy, 0, 0))

    # Initialise or copy over our global track knowledge
    if old_state.visible_track is None:
        visible_track = np.full(
            circuit_data.track_shape,
            CellType.NOT_VISIBLE.value,
            dtype=int
        )
    else:
        visible_track = old_state.visible_track.copy()

    R = circuit_data.visibility_radius

    # Merge the (2R+1)x(2R+1) patch into visible_track.
    # We *only* overwrite with values that are not NOT_VISIBLE
    # to avoid erasing older information.
    for i in range(2 * R + 1):
        row_vals = [int(a) for a in input().split()]
        x = posx - R + i

        if x < 0 or x >= circuit_data.track_shape[0]:
            # Entire row is off-map
            continue

        y_start = posy - R
        y_end = posy + R + 1

        if y_start < 0:
            # Drop elements that would be left of the map
            row_vals = row_vals[-y_start:]
            y_start = 0
        if y_end > circuit_data.track_shape[1]:
            # Drop elements that would be right of the map
            row_vals = row_vals[:-(y_end - circuit_data.track_shape[1])]
            y_end = circuit_data.track_shape[1]

        for offset, y in enumerate(range(y_start, y_end)):
            val = row_vals[offset]
            if val == CellType.NOT_VISIBLE.value:
                # No new information here
                continue
            visible_track[x, y] = val

    return old_state._replace(
        visible_track=visible_track,
        players=players,
        agent=agent
    )


def traversable(cell_value: int) -> bool:
    """
    Decide if a numeric cell is traversable.

    We consider cells traversable if:
      - they are NOT NOT_VISIBLE, and
      - their value is >= 0 (same rule as in the server code).
    """
    if cell_value == CellType.NOT_VISIBLE.value:
        return False
    return cell_value >= 0


def is_line_clear(track: np.ndarray,
                  pos1: np.ndarray,
                  pos2: np.ndarray) -> bool:
    """
    Check if the straight line between pos1 and pos2 stays on traversable
    cells (according to our *current* knowledge of the track).

    This mirrors the server's valid_line(...) logic:
      - we walk along the line in x and in y,
      - the line is blocked if we find two adjacent non-traversable
        cells above/below or left/right of the line.

    We also reject lines that go out of bounds.
    """
    H, W = track.shape
    if (np.any(pos1 < 0) or np.any(pos2 < 0)
            or pos1[0] >= H or pos1[1] >= W
            or pos2[0] >= H or pos2[1] >= W):
        return False

    diff = pos2 - pos1

    # Scan in x direction (check vertical pairs)
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))
        for i in range(abs(int(diff[0])) + 1):
            x = int(pos1[0] + i * d)
            y = pos1[1] + i * slope * d
            y_floor = int(np.floor(y))
            y_ceil = int(np.ceil(y))
            if (0 <= x < H and 0 <= y_floor < W and 0 <= y_ceil < W):
                if (not traversable(track[x, y_floor])
                        and not traversable(track[x, y_ceil])):
                    return False
            else:
                return False

    # Scan in y direction (check horizontal pairs)
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))
        for i in range(abs(int(diff[1])) + 1):
            x = pos1[0] + i * slope * d
            y = int(pos1[1] + i * d)
            x_floor = int(np.floor(x))
            x_ceil = int(np.ceil(x))
            if (0 <= y < W and 0 <= x_floor < H and 0 <= x_ceil < H):
                if (not traversable(track[x_floor, y])
                        and not traversable(track[x_ceil, y])):
                    return False
            else:
                return False

    return True


def valid_line(state: State, pos1: np.ndarray, pos2: np.ndarray) -> bool:
    """
    Wrapper around is_line_clear that uses the state's global visible_track.
    """
    assert state.visible_track is not None
    return is_line_clear(state.visible_track, pos1, pos2)


def is_valid_move_to(state: State,
                     from_pos: np.ndarray,
                     next_pos: np.ndarray) -> bool:
    """
    Check whether moving from from_pos to next_pos is valid:

      - the straight line must stay on traversable cells
      - the target cell must not be occupied by another player
        (unless we are staying in place).
    """
    if not valid_line(state, from_pos, next_pos):
        return False

    if np.all(next_pos == from_pos):
        return True

    for p in state.players:
        if np.all(next_pos == p.pos):
            return False

    return True


def count_unknown_in_radius(track: np.ndarray,
                            center: np.ndarray,
                            radius: int) -> int:
    """
    Count how many NOT_VISIBLE cells are within a given Euclidean radius
    around 'center'.
    """
    H, W = track.shape
    x0, y0 = int(center[0]), int(center[1])
    unknown_val = CellType.NOT_VISIBLE.value
    r2 = radius * radius
    count = 0

    x_min = max(0, x0 - radius)
    x_max = min(H - 1, x0 + radius)
    y_min = max(0, y0 - radius)
    y_max = min(W - 1, y0 + radius)

    for x in range(x_min, x_max + 1):
        dx = x - x0
        for y in range(y_min, y_max + 1):
            dy = y - y0
            if dx * dx + dy * dy <= r2 and track[x, y] == unknown_val:
                count += 1

    return count


def find_path_to_nearest_target(track: np.ndarray,
                                start: np.ndarray,
                                target_cells: list[tuple[int, int]],
                                bias: Optional[np.ndarray] = None
                                ) -> Optional[list[tuple[int, int]]]:
    """
    Generic BFS path finder on the currently known track.

    Parameters
    ----------
    track : np.ndarray
        Our HxW map of integers.
    start : np.ndarray
        Starting cell coordinates [x, y].
    target_cells : list[(int, int)]
        List of target cells to which we would like to find a path.
    bias : np.ndarray or None
        Optional "direction preference". If provided, the BFS will expand
        neighbours in the direction of 'bias' first, so we tend to find
        targets on that side of the map sooner (but still with a valid BFS
        shortest path for the chosen target).

    Returns
    -------
    path : list[(int, int)] or None
        A path (in number of steps shortest to the chosen target) from
        start to the first reached target cell, including both start and
        target. Uses 4-neighbour moves and also checks the line-of-sight
        constraint between neighbouring cells, so the path is compatible
        with the physics.
    """
    if not target_cells:
        return None

    H, W = track.shape
    start_cell = (int(start[0]), int(start[1]))
    targets = set((int(x), int(y)) for (x, y) in target_cells)

    if start_cell in targets:
        return [start_cell]

    q: deque[tuple[int, int]] = deque()
    q.append(start_cell)
    visited = {start_cell}
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    base_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        cx, cy = q.popleft()
        if (cx, cy) in targets:
            # Reconstruct path
            path = [(cx, cy)]
            while (cx, cy) != start_cell:
                cx, cy = parent[(cx, cy)]
                path.append((cx, cy))
            path.reverse()
            return path

        # Possibly reorder directions based on bias (goal hint).
        if bias is not None:
            gx, gy = int(bias[0]), int(bias[1])
            vecx, vecy = gx - cx, gy - cy

            def dir_score(d):
                dx, dy = d
                # Prefer directions with larger dot product with vector to bias
                return dx * vecx + dy * vecy

            directions = sorted(base_directions, key=dir_score, reverse=True)
        else:
            directions = base_directions

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            ncell = (nx, ny)
            if ncell in visited:
                continue
            if not traversable(track[nx, ny]):
                continue
            if not is_line_clear(track,
                                 np.array([cx, cy], dtype=int),
                                 np.array([nx, ny], dtype=int)):
                continue
            visited.add(ncell)
            parent[ncell] = (cx, cy)
            q.append(ncell)

    return None


def find_path_to_nearest_goal(track: np.ndarray,
                              start: np.ndarray
                              ) -> Optional[list[tuple[int, int]]]:
    """
    Thin wrapper around find_path_to_nearest_target for GOAL cells.

    GOAL cells are those with value 100 in the track.
    """
    goal_positions = np.argwhere(track == 100)
    if goal_positions.size == 0:
        return None
    target_cells = [(int(x), int(y)) for x, y in goal_positions]
    # For the actual goal path we do NOT bias BFS, we want the
    # shortest path on the known map.
    return find_path_to_nearest_target(track, start, target_cells, bias=None)


def compute_frontier_cells(track: np.ndarray,
                           visibility_radius: int) -> list[tuple[int, int]]:
    """
    Compute the list of "frontier" cells for exploration.

    A frontier cell is:
      - traversable, and
      - has at least one NOT_VISIBLE cell in its visibility radius.

    Visiting all frontier cells (that are reachable) is enough to uncover
    all reachable parts of the map.
    """
    H, W = track.shape
    frontier: list[tuple[int, int]] = []

    for x in range(H):
        for y in range(W):
            if not traversable(track[x, y]):
                continue
            if count_unknown_in_radius(track,
                                       np.array([x, y], dtype=int),
                                       visibility_radius) > 0:
                frontier.append((x, y))

    return frontier


def choose_accel_toward_cell(state: State,
                             target_cell: np.ndarray,
                             visit_counts: Optional[np.ndarray] = None,
                             prefer_slow: bool = False
                             ) -> Optional[tuple[int, int]]:
    """
    Choose an acceleration (ax, ay) in {-1, 0, 1}^2 that moves us towards
    'target_cell' while respecting acceleration physics and walls.

    - We simulate all 9 possible accelerations.
    - For each, we compute the resulting velocity and next position:
        new_vel  = vel + delta
        new_pos  = pos + new_vel
      (tehát a két egymás utáni 1 1 gyorsulásból tényleg 2 2 sebesség lesz)
    - Only moves that keep the line clear and avoid other players are valid.
    - We impose a soft SPEED_LIMIT on |new_vel|.
    - The chosen action minimises distance to the target cell, with an
      optional bias towards slower speeds (prefer_slow=True).
    """
    assert state.agent is not None
    assert state.visible_track is not None

    pos = state.agent.pos
    vel = state.agent.vel

    best_delta: Optional[tuple[int, int]] = None
    best_score = float('inf')

    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            delta = np.array([ax, ay], dtype=int)
            new_vel = vel + delta
            speed = np.linalg.norm(new_vel, ord=2)

            # Enforce soft speed limit
            if speed > SPEED_LIMIT:
                continue

            new_pos = pos + new_vel

            if not is_valid_move_to(state, pos, new_pos):
                continue

            # Base score: distance from new_pos to target_cell
            dist = np.linalg.norm(target_cell - new_pos)

            # Add small penalty for revisiting very often
            if visit_counts is not None:
                x, y = int(new_pos[0]), int(new_pos[1])
                if (0 <= x < visit_counts.shape[0]
                        and 0 <= y < visit_counts.shape[1]):
                    dist += 0.1 * visit_counts[x, y]

            # Optionally prefer slower motion when close to the target
            if prefer_slow:
                dist += 0.05 * speed

            if dist < best_score:
                best_score = dist
                best_delta = (ax, ay)

    return best_delta


def choose_exploration_move(state: State,
                            visit_counts: np.ndarray,
                            rng: np.random.Generator,
                            goal_hint: Optional[np.ndarray] = None
                            ) -> tuple[int, int]:
    """
    Low-level "local" exploration step.

    For each valid acceleration delta in {-1,0,1}^2 we:
      - simulate the resulting new position,
      - count how many NOT_VISIBLE cells would be inside our visibility
        radius from that position,
      - subtract a small penalty for visiting cells many times,
      - enforce the SPEED_LIMIT,
      - if a goal hint is known, we add a small preference to moves
        that keep us closer to the goal region.

    Among all valid accelerations we choose the one with highest score.
    """
    assert state.agent is not None
    assert state.visible_track is not None

    pos = state.agent.pos
    vel = state.agent.vel
    track = state.visible_track
    R = state.circuit.visibility_radius

    best_moves: list[tuple[int, int]] = []
    best_score = -float('inf')

    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            delta = np.array([ax, ay], dtype=int)
            new_vel = vel + delta
            speed = np.linalg.norm(new_vel, ord=2)

            # Keep speed under control
            if speed > SPEED_LIMIT:
                continue

            new_pos = pos + new_vel

            if not is_valid_move_to(state, pos, new_pos):
                continue

            unknown_count = count_unknown_in_radius(track, new_pos, R)

            x, y = int(new_pos[0]), int(new_pos[1])
            visited_penalty = 0
            if (0 <= x < visit_counts.shape[0]
                    and 0 <= y < visit_counts.shape[1]):
                visited_penalty = visit_counts[x, y]

            # Big weight on revealing new cells, small penalty on
            # revisiting the same place over and over.
            score = unknown_count * 100.0 - visited_penalty - 0.5 * speed

            # If we already know (or suspect) where the goal is, prefer moves
            # that keep us closer to the goal region.
            if goal_hint is not None:
                goal_dist = np.linalg.norm(goal_hint - new_pos)
                score -= 0.5 * goal_dist

            if score > best_score:
                best_score = score
                best_moves = [(ax, ay)]
            elif score == best_score:
                best_moves.append((ax, ay))

    if not best_moves:
        # No valid move (very rare) -> stay in place
        print(
            'Not blind, just being brave! (No valid exploration action found.)',
            file=sys.stderr
        )
        return (0, 0)

    chosen = rng.choice(best_moves)
    return int(chosen[0]), int(chosen[1])


def calculate_move(rng: np.random.Generator, state: State) -> tuple[int, int]:
    """
    High-level decision function for our AI.

    Logic per turn:
      1. Maintain a global visit heatmap.
      2. Try to find a *valid path* on the known map to a GOAL cell (value 100)
         using BFS and the same line-of-sight constraint as the server.
         If such a path exists, we follow it with a path-following controller.
      3. If there is no known valid path to any goal:
         a) If we have seen the goal somewhere, bias exploration and frontier
            choice towards that side of the map (frontiers "closer" to the
            goal are preferred), so we don't waste time exploring the opposite
            side first.
         b) Build the list of frontier cells (traversable cells from which
            we can still see some NOT_VISIBLE tiles).
         c) If frontiers exist, BFS to the nearest (biased) one and move along
            that path. When we are already on a frontier, use a local
            exploration heuristic to open up as much new territory as
            possible.
         d) If there are no frontier cells either, the reachable part of
            the map is fully explored. In that case, we simply try to slow
            down and stay safe.
    """
    assert state.agent is not None
    assert state.visible_track is not None

    global VISIT_COUNTS

    track = state.visible_track
    pos = state.agent.pos

    # Initialise visit heatmap the first time we are called
    if VISIT_COUNTS is None:
        VISIT_COUNTS = np.zeros(track.shape, dtype=int)

    # Mark current cell as visited
    VISIT_COUNTS[pos[0], pos[1]] += 1

    # Discover all known goal cells (100)
    goal_positions = np.argwhere(track == 100)
    goal_hint: Optional[np.ndarray] = None
    if goal_positions.size > 0:
        # Use the closest known goal as direction hint
        dists = [np.linalg.norm(pos - g.astype(int)) for g in goal_positions]
        goal_hint = goal_positions[int(np.argmin(dists))].astype(int)

    # --- 1) Try to go to a known, reachable GOAL --------------------------- #
    path_to_goal = find_path_to_nearest_goal(track, pos)

    if path_to_goal is not None:
        # If path length is 1, we are already on the goal cell.
        if len(path_to_goal) == 1:
            return (0, 0)

        # Take the second cell on the path as a waypoint.
        next_waypoint = np.array(path_to_goal[1], dtype=int)
        accel = choose_accel_toward_cell(
            state,
            next_waypoint,
            visit_counts=VISIT_COUNTS,
            prefer_slow=True
        )
        if accel is not None:
            return accel
        # If there is no valid acceleration towards the waypoint,
        # we fall back to exploration for this turn.

    # --- 2) Exploration mode ----------------------------------------------- #
    R = state.circuit.visibility_radius
    frontier_cells = compute_frontier_cells(track, R)

    if frontier_cells:
        # First, see if we are already on a frontier cell.
        current_is_frontier = any(
            (pos[0] == fx and pos[1] == fy) for (fx, fy) in frontier_cells
        )

        if not current_is_frontier:
            # We are not on a frontier yet -> go to the nearest frontier cell.
            # If we have a goal_hint (we have seen the goal but not yet have
            # a valid path), BFS will be biased towards frontiers that are
            # on the same side of the map as the goal.
            path_to_frontier = find_path_to_nearest_target(
                track, pos, frontier_cells, bias=goal_hint
            )
            if path_to_frontier is not None and len(path_to_frontier) > 1:
                next_waypoint = np.array(path_to_frontier[1], dtype=int)
                accel = choose_accel_toward_cell(
                    state,
                    next_waypoint,
                    visit_counts=VISIT_COUNTS,
                    prefer_slow=True
                )
                if accel is not None:
                    return accel
                # If we cannot find a valid acceleration towards the
                # waypoint, we fall back to local exploration.

        # Either we are already on a frontier, or pathfinding failed.
        # Local exploration is optionally biased towards the goal.
        return choose_exploration_move(state, VISIT_COUNTS, rng, goal_hint)

    # --- 3) Fully explored & no reachable goal ----------------------------- #
    # No unknown cells in the neighbourhood of any traversable tile:
    # the reachable part of the map is fully explored and we do not
    # see any goal. In this case we simply try to slow down safely.
    vel = state.agent.vel
    if np.all(vel == 0):
        return (0, 0)

    # Try to decelerate (acceleration opposite to velocity, clipped to [-1,1])
    desired_delta = -np.sign(vel).astype(int)
    candidate_pos = pos + vel + desired_delta

    if is_valid_move_to(state, pos, candidate_pos):
        return int(desired_delta[0]), int(desired_delta[1])

    # If decelerating directly is not possible, fall back to local exploration
    # which still respects walls and SPEED_LIMIT.
    return choose_exploration_move(state, VISIT_COUNTS, rng, goal_hint)


def main():
    """
    Main loop of the bot:

      - Print 'READY' so that the judge knows we are alive.
      - Read initial circuit information.
      - Then repeatedly:
          * read an observation,
          * compute our acceleration,
          * print it to stdout.
    """
    print('READY', flush=True)
    circuit = read_initial_observation()
    state: Optional[State] = State(circuit, None, [], None)  # type: ignore
    rng = np.random.default_rng(seed=1)

    while True:
        assert state is not None
        state = read_observation(state)
        if state is None:
            return
        delta = calculate_move(rng, state)
        print(f'{delta[0]} {delta[1]}', flush=True)


if __name__ == "__main__":
    main()
