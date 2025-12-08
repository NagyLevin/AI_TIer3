import sys
import enum
import math
import os
from collections import deque
from typing import Optional, NamedTuple, Tuple, List, Dict

import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Basic types (judge-compatible)
# ────────────────────────────────────────────────────────────────────────────────

class CellType(enum.Enum):
    """
    CellType enum, a judge / grid_race_env kódjával kompatibilis értékekkel.

      -1 : WALL
       0 : EMPTY
       1 : START
       2 : UNKNOWN (belső jelölés, saját modellnek)
       3 : NOT_VISIBLE (fog of war)
      91 : OIL  (csúszós mező)
      92 : SAND (homok, lassít)
     100 : GOAL
    """
    GOAL = 100
    START = 1
    WALL = -1
    UNKNOWN = 2
    EMPTY = 0
    NOT_VISIBLE = 3
    OIL = 91
    SAND = 92


class Player(NamedTuple):
    """
    Játékos állapot:
      x, y   : rács pozíció
      vel_x, vel_y : sebesség komponensek
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
        """
        A játékos sebességét adja vissza np.array([vel_x, vel_y]) formában.
        """
        return np.array([self.vel_x, self.vel_y], dtype=int)


class Circuit(NamedTuple):
    """
    Statikus pálya-információ:
      track_shape       : (H, W)
      num_players       : játékosok száma
      visibility_radius : lokális látótávolság
    """
    track_shape: tuple[int, int]
    num_players: int
    visibility_radius: int


class State(NamedTuple):
    """
    Agent állapota:
      circuit       : statikus pálya-információ
      visible_track : "safe" map, NOT_VISIBLE -> WALL
      visible_raw   : nyers látott map, NOT_VISIBLE benne marad 3-ként
      players       : játékosok (legutóbbi ismert pozícióval)
      agent         : saját Player
    """
    circuit: Circuit
    visible_track: Optional[np.ndarray]
    visible_raw: Optional[np.ndarray]
    players: List[Player]
    agent: Optional[Player]

# ────────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────────

def read_initial_observation() -> Circuit:
    """
    Első sor beolvasása:
      H W num_players visibility_radius
    és Circuit objektummá alakítása.
    """
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)


def read_observation(old_state: State) -> Optional[State]:
    """
    Egy kör megfigyelésének beolvasása.

    Formátum:
      - sor:  posx posy velx vely    (agent)
      - num_players sor: pposx pposy (minden játékos)
      - 2R+1 sor: lokális térkép

    Ha '~~~END~~~', akkor vége a játéknak -> None.
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

    # visible_raw: a teljes pályaméretre kiterített nyers map (3-asokkal)
    visible_raw = np.full((H, W), CellType.NOT_VISIBLE.value, dtype=int)
    # visible_track: konzervatív "biztonsági" map: NOT_VISIBLE -> WALL
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
                # NOT_VISIBLE -> WALL a tervezési mapben
                safety = [
                    CellType.WALL.value if v == CellType.NOT_VISIBLE.value else v
                    for v in loc
                ]
                visible_track[x, ys:ye] = safety

    return old_state._replace(
        visible_track=visible_track,
        visible_raw=visible_raw,
        players=players,
        agent=agent,
    )

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
    Háromszögszám: 1 + 2 + ... + n
    A féktávolság becslésére használjuk.
    """
    return n * (n + 1) // 2


def brakingOk(vx: int, vy: int, rSafe: int) -> bool:
    """
    Ellenőrzi, hogy a jelenlegi sebességet le lehet-e fékezni
    a látóhatáron belül (külön x és y irányban).
    """
    return (tri(abs(vx)) <= rSafe) and (tri(abs(vy)) <= rSafe)


def validLineLocal(state: State, p1: np.ndarray, p2: np.ndarray) -> bool:
    """
    Judge-féle line-of-sight ellenőrzés saját visible_track alapján.
    Ha két cella közötti egyenes út végig nem fal, akkor True.
    """
    track = state.visible_track
    if track is None:
        return False
    H, W = track.shape
    if (np.any(p1 < 0) or np.any(p2 < 0) or
        p1[0] >= H or p1[1] >= W or p2[0] >= H or p2[1] >= W):
        return False
    diff = p2 - p1
    # függőleges komponens vizsgálata
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))
        for i in range(abs(diff[0]) + 1):
            x = int(p1[0] + i * d)
            y = p1[1] + i * slope * d
            yC = int(np.ceil(y))
            yF = int(np.floor(y))
            if (track[x, yC] < 0 and track[x, yF] < 0):
                return False
    # vízszintes komponens vizsgálata
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))
        for i in range(abs(diff[1]) + 1):
            x = p1[0] + i * slope * d
            y = int(p1[1] + i * d)
            xC = int(np.ceil(x))
            xF = int(np.floor(x))
            if (track[xC, y] < 0 and track[xF, y] < 0):
                return False
    return True


def find_reachable_zero(state: State, world: 'WorldModel', start_pos: np.ndarray) -> bool:
    """
    BFS-sel megnézi, hogy a jelenlegi látókörben van-e olyan
    elérhető EMPTY cella, amit még nem látogattunk (visited_count == 0).
    """
    if state.agent is None:
        return False

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
    """
    Igaz, ha a cella bejárható:
      - nem fal
      - nem UNKNOWN (belső jelölés)
    """
    return (v >= 0) and (v != CellType.UNKNOWN.value)


def is_hazard_val(v: int) -> bool:
    """
    Igaz, ha a cella veszélyes (hazard):
      - jelenleg: minden NEM negatív, ami NEM
        EMPTY, START, GOAL, UNKNOWN, NOT_VISIBLE.
      -> így OIL (91) és SAND (92) is hazard.
    """
    if v < 0:
        return False
    if v in (
        CellType.EMPTY.value,
        CellType.START.value,
        CellType.GOAL.value,
        CellType.UNKNOWN.value,
        CellType.NOT_VISIBLE.value,
    ):
        return False
    return True


class WorldModel:
    """
    Perzisztens világtérkép:
      - known_map      : eddig látott pályaértékek (eredeti kód: -1,0,1,91,92,100,3,...)
      - visited_count  : hányszor jártunk az adott cellán
      - hazard_val     : ha hazard, akkor a cella értéke (pl. 91, 92), különben -1
      - hazard_char_map: hazard tile value -> betű, ASCII dumphoz
    """
    def __init__(self, shape: tuple[int, int]) -> None:
        H, W = shape
        self.shape = shape
        self.known_map = np.full((H, W), CellType.UNKNOWN.value, dtype=int)
        self.visited_count = np.zeros((H, W), dtype=int)
        self.last_pos: Optional[Tuple[int, int]] = None

        # hazard típus-nyilvántartás (tile-érték szerint)
        self.hazard_val = np.full((H, W), -1, dtype=int)  # -1 = nem hazard
        self.hazard_char_map: Dict[int, str] = {}
        # H és O kimarad (H=Sand, O=Oil az ASCII dumpban)
        self.hazard_char_pool = list("BCDFJKLMPQRTUVWXYZ")

        self.turn = 0
        self.dump_file = "logs/map_dump.txt"
        self._dump_initialized = False

    def updateWithObservation(self, st: State) -> None:
        """
        Beemeli az új megfigyelést:
          - visible_raw alapján frissíti a known_map-et
          - újraszámolja, hol vannak hazard cellák
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
        """
        Igaz, ha (x,y) pálya koordináta bejárható (nem fal, nem UNKNOWN).
        """
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W):
            return False
        return is_traversable_val(self.known_map[x, y])

    def is_hazard(self, x: int, y: int) -> bool:
        """
        Igaz, ha (x,y) hazard mező (pl. 91 = oil, 92 = sand).
        """
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W):
            return False
        return int(self.hazard_val[x, y]) != -1

    def get_hazard_value(self, x: int, y: int) -> Optional[int]:
        """
        Visszaadja a cella hazard értékét (pl. 91, 92), ha van, különben None.
        """
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W):
            return None
        v = int(self.hazard_val[x, y])
        return v if v != -1 else None

    def get_hazard_char(self, tile_value: int) -> str:
        """
        A megadott hazard tile értékhez (pl. 91, 92, vagy bármi más)
        visszaad egy ASCII betűt a map_dump vizualizációhoz.

        Oil (91) és Sand (92) speciális betűt kap a dump_ascii-ben (O/H),
        ezért itt inkább más hazardokra használjuk a pool-t.
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
# Exploration policy (globális BFS, minimális backtrack)
# ────────────────────────────────────────────────────────────────────────────────

def left_of(d: Tuple[int,int]) -> Tuple[int,int]:
    """
    Balra forgatott irány (dx,dy) -> (-dy, dx)
    (csak a heading vizualizációhoz tartjuk meg)
    """
    dx, dy = d
    return (-dy, dx)

def right_of(d: Tuple[int,int]) -> Tuple[int,int]:
    """
    Jobbra forgatott irány (dx,dy) -> (dy, -dx)
    (csak a heading vizualizációhoz tartjuk meg)
    """
    dx, dy = d
    return (dy, -dx)

def back_of(d: Tuple[int,int]) -> Tuple[int,int]:
    """
    Hátrafelé mutató irány (dx,dy) -> (-dx, -dy)
    """
    dx, dy = d
    return (-dx, -dy)


class LeftWallPolicy:
    """
    ÚJ POLITIKA: globális, BFS-alapú exploration, minimális backtrack.

    Logika:
      1) Ha már látjuk a GOAL cellát, arra tervezünk (BFS a pályán).
      2) Ha nincs ismert GOAL:
         - keressük a legközelebbi olyan EMPTY cellát, amit még nem
           látogattunk (visited_count == 0) -> erre megyünk.
      3) Ha már minden EMPTY cellát meglátogattuk:
         - keressük a legközelebbi "frontier" cellát:
             EMPTY vagy START, amelynek van UNKNOWN/NOT_VISIBLE szomszédja.
      4) Ha ilyen sincs, akkor már mindent feltérképeztünk -> maradunk.

    A heading mezőt csak a debug / ASCII log vizualizációjához használjuk.
    """
    def __init__(self, world: WorldModel) -> None:
        self.world = world
        self.heading: Tuple[int,int] = (0, 1)  # kezdeti "előre" irány

    def _ensure_heading_from_velocity(self, state: State) -> None:
        """
        Ha a sebesség nem 0, akkor abból következtetünk a heading irányára.
        """
        if state.agent is None:
            return
        vx, vy = int(state.agent.vel_x), int(state.agent.vel_y)
        if vx == 0 and vy == 0:
            return
        if abs(vx) >= abs(vy):
            self.heading = (int(np.sign(vx)), 0)
        else:
            self.heading = (0, int(np.sign(vy)))

    def _bfs_find(self,
                  start: Tuple[int,int],
                  cond) -> Optional[Tuple[int,int]]:
        """
        Általános BFS-kereső:
          - start cellából indul,
          - world.traversable grid-en mozog (4-irány),
          - cond(x, y, world) feltételre hoz vissza az első cellát
            (legkisebb BFS távolság).
        """
        H, W = self.world.shape
        sx, sy = start
        if not self.world.traversable(sx, sy):
            return None

        q = deque([start])
        seen = {start}

        while q:
            x, y = q.popleft()
            if cond(x, y, self.world):
                return (x, y)
            for dx, dy in DIRS_4:
                nx, ny = x + dx, y + dy
                if (0 <= nx < H and 0 <= ny < W and
                    (nx, ny) not in seen and
                    self.world.traversable(nx, ny)):
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return None

    def next_grid_target(self, state: State) -> Tuple[Tuple[int,int], str]:
        """
        Eldönti, hogy a rácson melyik cellára szeretnénk "tartani" ebben a körben.

        Visszaad:
          - cél cella (rácskoordináta)
          - mód string debug / log célra
        """
        assert state.agent is not None
        ax, ay = int(state.agent.x), int(state.agent.y)

        # heading csak vizualizációhoz kell
        self._ensure_heading_from_velocity(state)

        # 1) Ha ismerünk GOAL-t, arra megyünk
        def is_goal(x: int, y: int, world: WorldModel) -> bool:
            return world.known_map[x, y] == CellType.GOAL.value

        goal_cell = self._bfs_find((ax, ay), is_goal)
        if goal_cell is not None:
            gx, gy = goal_cell
            dx, dy = gx - ax, gy - ay
            if dx != 0 or dy != 0:
                if abs(dx) >= abs(dy):
                    self.heading = (int(np.sign(dx)), 0)
                else:
                    self.heading = (0, int(np.sign(dy)))
            return goal_cell, "go_goal"

        # 2) Legközelebbi EMPTY, amit még nem látogattunk
        def is_unvisited_empty(x: int, y: int, world: WorldModel) -> bool:
            return (world.known_map[x, y] == CellType.EMPTY.value and
                    world.visited_count[x, y] == 0)

        unv_cell = self._bfs_find((ax, ay), is_unvisited_empty)
        if unv_cell is not None:
            ux, uy = unv_cell
            dx, dy = ux - ax, uy - ay
            if dx != 0 or dy != 0:
                if abs(dx) >= abs(dy):
                    self.heading = (int(np.sign(dx)), 0)
                else:
                    self.heading = (0, int(np.sign(dy)))
            return unv_cell, "explore_unvisited"

        # 3) Ha minden EMPTY visited, keressünk frontier cellát
        H, W = self.world.shape
        def is_frontier(x: int, y: int, world: WorldModel) -> bool:
            v = world.known_map[x, y]
            if v not in (CellType.EMPTY.value, CellType.START.value):
                return False
            for dx, dy in DIRS_4:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    nv = world.known_map[nx, ny]
                    if nv in (CellType.UNKNOWN.value, CellType.NOT_VISIBLE.value):
                        return True
            return False

        frontier = self._bfs_find((ax, ay), is_frontier)
        if frontier is not None:
            fx, fy = frontier
            dx, dy = fx - ax, fy - ay
            if dx != 0 or dy != 0:
                if abs(dx) >= abs(dy):
                    self.heading = (int(np.sign(dx)), 0)
                else:
                    self.heading = (0, int(np.sign(dy)))
            return frontier, "explore_frontier"

        # 4) Minden feltérképezve => maradunk, nincs konkrét új cél
        return (ax, ay), "fully_explored"

# ────────────────────────────────────────────────────────────────────────────────
# Low-level driver (hazard-aware scoring)
# ────────────────────────────────────────────────────────────────────────────────

def choose_accel_toward_cell(state: State,
                             world: WorldModel,
                             policy: LeftWallPolicy,
                             target_cell: Tuple[int,int],
                             mode: str) -> Tuple[int,int]:
    """
    Low-level döntés:
      - adott rácscella (target_cell) felé próbál gyorsítani,
      - figyelembe veszi:
          * féktávolságot (brakingOk),
          * falakat (validLineLocal),
          * ütközést más játékosokkal,
          * minél kevesebbet járjunk már ismert cellákra (visited_count),
          * erősen bünteti a hazard (oil/sand) cellákat.
    """
    assert state.agent is not None and state.visible_track is not None
    rSafe = max(0, state.circuit.visibility_radius - 1)

    p = state.agent.pos
    v = state.agent.vel
    vx, vy = int(v[0]), int(v[1])

    ax_agent, ay_agent = int(state.agent.x), int(state.agent.y)

    # Megnézzük, mennyire van közel "új" cella (EMPTY + visited_count==0),
    # hogy ha közel van, lassítsunk, ne száguldjunk bele.
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

    # Ha közel új cella felé közeledünk, előnyben részesítjük a lassítást,
    # hogy ne rohanjunk túl rajta.
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

    # maximális "kényelmes" sebesség (a látóhatár függvényében)
    max_safe = max(1.0, math.sqrt(2 * max(0, rSafe)))

    # Célzott sebesség a mód alapján
    if force_slow_down:
        target_speed = 1.0
    else:
        tx, ty = target_cell
        target_is_hazard = world.is_hazard(tx, ty) if (0 <= tx < world.shape[0] and 0 <= ty < world.shape[1]) else False

        if (0 <= tx < world.shape[0] and 0 <= ty < world.shape[1] and
            world.known_map[tx, ty] == CellType.EMPTY.value and
            world.visited_count[tx, ty] == 0 and not target_is_hazard):
            # Teljesen új cella: mehetünk bátrabban
            target_speed = max_safe
        elif mode.startswith("search_") or mode.startswith("fallback") or mode == "corridor_visited":
            target_speed = max(1.0, 0.5 * max_safe)
        elif target_is_hazard:
            target_speed = max(1.0, 0.5 * max_safe)
        else:
            target_speed = max(1.5, 0.7 * max_safe)

    # Cél irány vektor (rácskoordinátákon)
    to_cell = np.array([target_cell[0], target_cell[1]], dtype=float) - p.astype(float)
    n_to = float(np.linalg.norm(to_cell)) or 1.0
    desired_dir = to_cell / n_to

    best = None
    H, W = world.shape

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
            if 0 <= nx < H and 0 <= ny < W:
                visit_pen = 100.0 * float(world.visited_count[nx, ny])
                if world.is_hazard(nx, ny):
                    hazard_pen = 500.0

            score = (2.0 * dist_cell) + (0.9 * speed_pen) + heading_pen + visit_pen + hazard_pen
            cand = (score, (ax, ay))
            if (best is None) or (cand[0] < best[0]):
                best = cand

    if best is not None:
        return best[1]

    # Ha semmi sem megy, próbáljunk meg lassítani / megállni biztonságosan
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
    """
    ASCII pályalog kiírása.

    Speciális jelölések:
      - '#' : fal
      - '.' : üres
      - 'G' : cél
      - 'S' : start
      - '?' : ismeretlen
      - 'O' : Oil (tile 91)
      - 'H' : Sand (tile 92)
      - 'A' : agent
      - 'O' (játékos) : más játékos pozíciója a rácson
      - [1-9,a-z,+] : látogatásszám (nem hazard cellákon)
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
            elif int(v) == CellType.OIL.value:
                grid[x][y] = 'O'   # Oil (91)
            elif int(v) == CellType.SAND.value:
                grid[x][y] = 'H'   # Sand (92)
            elif is_hazard_val(v):
                grid[x][y] = world.get_hazard_char(int(v))

    # visit count overlay (nem hazard, nem fal, nem G/S)
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

    # többi játékos
    for p in state.players:
        if 0 <= p.x < H and 0 <= p.y < W:
            grid[p.x][p.y] = 'O'

    # saját pozíció
    ax, ay = int(state.agent.x), int(state.agent.y)
    if 0 <= ax < H and 0 <= ay < W:
        grid[ax][ay] = 'A'

    hdr = []
    hdr.append(
        f"TURN {world.turn}  pos=({ax},{ay}) vel=({int(state.agent.vel_x)},{int(state.agent.vel_y)}) "
        f"mode={mode} heading={policy.heading}"
    )
    hdr.append("LEGEND: #=WALL  ?=UNKNOWN  .=EMPTY  G=GOAL  S=START  A=agent  O=Oil  H=Sand  [1-9,a-z,+]=visit count")
    if world.hazard_char_map:
        hdr.append("OTHER HAZARD TYPES:")
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
    """
    Fő döntési függvény egy körre:
      - frissíti a világmodellt az aktuális megfigyeléssel,
      - növeli az aktuális cella visited_count-ját,
      - globális BFS-sel választ egy cél cellát (exploration logika),
      - low-level szinten kiválasztja az ideális (ax, ay) gyorsítást,
      - kiírja az ASCII-dumpot a logs/map_dump.txt-be.
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
    Program belépési pontja:
      - READY kiírás,
      - Circuit beolvasása,
      - WorldModel + Policy inicializálása,
      - amíg nem jön ~~~END~~~:
          * read_observation
          * calculateMove
          * (ax, ay) kiírása
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

        # biztos, ami biztos: -1..1 közé szorítjuk az accel komponenseket
        ax = -1 if ax < -1 else (1 if ax > 1 else int(ax))
        ay = -1 if ay < -1 else (1 if ay > 1 else int(ay))
        print(f"{ax} {ay}", flush=True)


if __name__ == "__main__":
    main()
