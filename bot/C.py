import sys
import enum
import numpy as np
import heapq
from collections import deque
from typing import Optional, NamedTuple, List, Tuple


# ────────────────────────────────────────────────────────────────────────────────
# JÁTÉK KONSTANSOK (grid_race_env alapján)
# ────────────────────────────────────────────────────────────────────────────────

class CellType(enum.Enum):
    """
    CellType értékek:
      - WALL: fal, teljesen átjárhatatlan
      - EMPTY: üres/útfelület
      - START: rajt
      - UNKNOWN: belső jelölés a TELJESEN FELDERÍTETLEN (fog-of-war) cellákra
      - NOT_VISIBLE: bemenetbeli "nem látjuk most" érték
      - OIL, SAND: veszélyes mezők
      - GOAL: cél
    """
    WALL = -1
    EMPTY = 0
    START = 1
    UNKNOWN = -2       # belső fog-of-war jelölés (nem azonos a pálya 2-esével!)
    NOT_VISIBLE = 3
    OIL = 91
    SAND = 92
    GOAL = 100


HAZARDS = {CellType.OIL.value, CellType.SAND.value}


class Player(NamedTuple):
    """
    Egy játékos állapota: pozíció + sebesség.
    """
    x: int
    y: int
    vel_x: int
    vel_y: int

    @property
    def pos(self) -> np.ndarray:
        """Visszaadja a pozíciót [row, col] numpy vektorként."""
        return np.array([self.x, self.y])

    @property
    def vel(self) -> np.ndarray:
        """Visszaadja a sebességet [vx, vy] numpy vektorként."""
        return np.array([self.vel_x, self.vel_y])


class Circuit(NamedTuple):
    """
    Pálya metaadatai: méret, játékosok száma, látótáv.
    """
    track_shape: tuple[int, int]
    num_players: int
    visibility_radius: int


class State(NamedTuple):
    """
    Aktuális állapot:
      - circuit: pálya adatok
      - visible_track: a judge által adott aktuális HxW rács (NOT_VISIBLE = 3)
      - players: többi játékos
      - agent: saját játékosunk
    """
    circuit: Circuit
    visible_track: np.ndarray
    players: list[Player]
    agent: Player


# ────────────────────────────────────────────────────────────────────────────────
# GLOBÁLIS TÉRKÉP
# ────────────────────────────────────────────────────────────────────────────────

class GlobalMap:
    """
    Egy HxW-s saját világmodell, amelybe beírjuk az eddig látott pályarészleteket.

    self.grid értékei:
      - kezdetben: UNKNOWN (-2) mindenhol (fog-of-war)
      - frissítéskor: ahol az új visible_track NEM NOT_VISIBLE (3), oda bemásoljuk
        a kapott cellaértéket (WALL, EMPTY, START, GOAL, OIL, SAND, stb.).
    """

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        self.grid = np.full(shape, CellType.UNKNOWN.value, dtype=int)

    def update(self, visible_track: np.ndarray):
        """
        Frissíti a globális térképet az aktuális visible_track alapján.
        Csak azokat a cellákat írjuk felül, ahol a visible_track != NOT_VISIBLE.
        """
        mask = (visible_track != CellType.NOT_VISIBLE.value)
        self.grid[mask] = visible_track[mask]


# ────────────────────────────────────────────────────────────────────────────────
# CÉLVÁLASZTÁS: ISMERETLEN ÉS CÉL KERESÉSE
# ────────────────────────────────────────────────────────────────────────────────

def find_nearest_unknown(
    start: tuple[int, int],
    gmap: GlobalMap,
    avoid_hazards: bool = True
) -> Optional[tuple[int, int]]:
    """
    BFS a legközelebbi TELJESEN FELDERÍTETLEN (UNKNOWN = -2) cellára.

    Paraméterek:
      - start: induló pozíció (saját cellánk)
      - gmap: globális térkép
      - avoid_hazards: ha True, akkor OIL/SAND cellákon át SEM megy a BFS,
                       mintha fal lenne.
                       Ha False, akkor a BFS ezeken is átmehet.

    Visszaad:
      - (x, y) koordináta a legközelebbi UNKNOWN cellára,
      - None, ha nem található elérhető UNKNOWN.
    """
    queue = deque([start])
    visited = {start}

    # Ha véletlenül magunk is UNKNOWN-on állnánk (elméletben ritka), az is jó cél.
    if gmap.grid[start] == CellType.UNKNOWN.value:
        return start

    H, W = gmap.shape

    while queue:
        cx, cy = queue.popleft()

        # 4-irányú BFS
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            if (nx, ny) in visited:
                continue

            val = gmap.grid[nx, ny]

            # Fal: ide se lépünk
            if val == CellType.WALL.value:
                continue

            # Ha hazardot kerüljük, akkor OIL/SAND is blokk
            if avoid_hazards and val in HAZARDS:
                continue

            # Ha UNKNOWN (fog-of-war) cellát találunk → ez a célunk
            if val == CellType.UNKNOWN.value:
                return (nx, ny)

            visited.add((nx, ny))
            queue.append((nx, ny))

    return None


# ────────────────────────────────────────────────────────────────────────────────
# ÚTVONALKERESÉS: A* HAZARD-KERÜLÉSSEL
# ────────────────────────────────────────────────────────────────────────────────

def run_astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    gmap: GlobalMap,
    avoid_hazards: bool
) -> List[tuple[int, int]]:
    """
    A* útvonal keresése start → goal között a globális térképen.

    Paraméterek:
      - start, goal: (x, y) cella koordináták
      - gmap: globális térkép
      - avoid_hazards: ha True, akkor az A* sem engedi, hogy az útvonal
        OIL/SAND cellán menjen át (mintha fal lenne).
        Ha False, akkor OIL/SAND átjárható, de magas költségű.

    Költség:
      - WALL: átjárhatatlan
      - UNKNOWN(-2) és normál (EMPTY/START/GOAL/2-es cella): költség = 1
      - OIL/SAND, ha engedélyezett: költség = 20

    Visszaad:
      - path: [start, lépés1, ..., goal] cella-koordináták listája,
      - üres lista, ha nem talál útvonalat.
    """
    H, W = gmap.shape
    (sx, sy) = start
    (gx, gy) = goal

    # Gyors sanity check: ha start vagy goal kívül esik, vagy fal, nincs út
    if not (0 <= sx < H and 0 <= sy < W):
        return []
    if not (0 <= gx < H and 0 <= gy < W):
        return []
    if gmap.grid[gx, gy] == CellType.WALL.value:
        return []

    # (költség, (x, y))
    pq: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
    came_from: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    cost_so_far: dict[Tuple[int, int], float] = {start: 0.0}

    while pq:
        _, current = heapq.heappop(pq)

        if current == goal:
            break

        cx, cy = current

        # 8-irányú lépés (átlósat is engedünk)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < H and 0 <= ny < W):
                    continue

                val = gmap.grid[nx, ny]

                # Fal soha
                if val == CellType.WALL.value:
                    continue

                # Hazard kerülése, ha kérjük
                if avoid_hazards and val in HAZARDS:
                    continue

                # Lépés költség meghatározása
                if val in HAZARDS:
                    step_cost = 20.0  # drága, de engedélyezett
                else:
                    step_cost = 1.0   # normál/UNKNOWN/GOAL/START/EMPTY/2-es

                new_cost = cost_so_far[current] + step_cost

                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    # Heurisztika: Manhattan-táv
                    priority = new_cost + abs(gx - nx) + abs(gy - ny)
                    heapq.heappush(pq, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current

    if goal not in came_from:
        return []

    # Útvonal visszafejtése
    path: List[Tuple[int, int]] = []
    curr: Optional[Tuple[int, int]] = goal
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path


# ────────────────────────────────────────────────────────────────────────────────
# BEMENET OLVASÁSA (JUDGE KOMMUNIKÁCIÓ)
# ────────────────────────────────────────────────────────────────────────────────

def read_initial_observation() -> Circuit:
    """
    Beolvassa az első sort:
      H W num_players visibility_radius
    és létrehoz egy Circuit objektumot.
    """
    line = sys.stdin.readline()
    if not line:
        return None  # type: ignore
    H, W, num_players, visibility_radius = map(int, line.split())
    return Circuit((H, W), num_players, visibility_radius)


def read_observation(old_state: State) -> Optional[State]:
    """
    Minden körben:
      - beolvassa a saját pozíciónkat + sebességünket,
      - a többi játékos pozícióját,
      - majd a (2R+1)x(2R+1) lokális rácsot,
    és ebből egy HxW-s visible_track-et állít elő, ahol alapból NOT_VISIBLE (3),
    csak a most látott rész kerül beírva.
    """
    line = sys.stdin.readline()
    if not line or line.strip() == '~~~END~~~':
        return None

    # Saját adataink
    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)

    players: List[Player] = []
    circuit_data = old_state.circuit

    # Többi játékos pozíciója
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, sys.stdin.readline().split())
        players.append(Player(pposx, pposy, 0, 0))

    # Pálya látott része (láthatósági kör)
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

        # Vágás vízszintesen, ha kilóg
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
# DÖNTÉSHOZATAL: CÉL VÁLASZTÁSA ÉS GYORSULÁS SZÁMÍTÁSA
# ────────────────────────────────────────────────────────────────────────────────

def calculate_move_logic(state: State, gmap: GlobalMap) -> Tuple[int, int]:
    """
    Fő döntési logika minden körben.

    Lépések:
      1) Frissítjük a globális térképet a most látott visible_track alapján.
      2) Ha látunk GOAL mezőt:
           - kiválasztjuk a legközelebbit (Manhattan-táv),
           - A*-gal utat tervezünk hozzá (először hazard nélkül,
             ha nem megy, akkor hazarddal),
           - ha van legalább 1 lépés, ütközés-kerülő lokális vezérlővel oda gyorsítunk.
      3) Ha még nem látunk GOAL-t:
           - legközelebbi UNKNOWN(-2) cellát keressük BFS-sel úgy,
             hogy először hazardmentesen járjuk be a teret,
             ha nincs ilyen, akkor engedjük a hazardokat.
           - A*-gal ugyanígy (hazard nélkül, majd szükség esetén hazarddal)
             utat tervezünk a kiválasztott ismeretlen felé.
      4) Ha sem cél, sem elérhető ismeretlen nincs → próbáljunk lassítani / megállni.
    """
    my_pos = (state.agent.x, state.agent.y)

    # 1) Globális térkép frissítése
    gmap.update(state.visible_track)

    # 2) Először nézzük meg, látunk-e GOAL-t
    goals = np.argwhere(gmap.grid == CellType.GOAL.value)
    if len(goals) > 0:
        # Legközelebbi GOAL kiválasztása Manhattan-távolság alapján
        dists = np.sum(np.abs(goals - np.array(my_pos)), axis=1)
        goal_tuple = tuple(goals[np.argmin(dists)])  # type: ignore

        # A* hazard nélkül
        path = run_astar(my_pos, goal_tuple, gmap, avoid_hazards=True)
        if len(path) < 2:
            # Ha így nem megy, próbáljuk hazarddal (utolsó esély)
            path = run_astar(my_pos, goal_tuple, gmap, avoid_hazards=False)

        if len(path) >= 2:
            next_cell = path[1]
            return _pd_control_to_cell(state, next_cell)

        # Ha valamiért a cél se elérhető, esünk vissza a felfedezés logikára

    # 3) Felfedezés: legközelebbi UNKNOWN keresése
    target = find_nearest_unknown(my_pos, gmap, avoid_hazards=True)
    allow_hazards_for_explore = False

    if target is None:
        # Nincs hazardmentesen elérhető ismeretlen → engedjük a hazardot
        target = find_nearest_unknown(my_pos, gmap, avoid_hazards=False)
        allow_hazards_for_explore = True

    if target is not None:
        # Megpróbálunk hazard nélkül eljutni hozzá (ha lehet)
        path = run_astar(my_pos, target, gmap, avoid_hazards=not allow_hazards_for_explore)
        if len(path) < 2 and allow_hazards_for_explore:
            # Ha csak hazarddal elérhető, próbáljuk úgy is
            path = run_astar(my_pos, target, gmap, avoid_hazards=False)

        if len(path) >= 2:
            next_cell = path[1]
            return _pd_control_to_cell(state, next_cell)

    # 4) Ha nincs cél és ismeretlen sincs → fékezzünk / maradjunk
    vel = state.agent.vel
    ax = int(np.clip(-vel[0], -1, 1))
    ay = int(np.clip(-vel[1], -1, 1))
    return (ax, ay)


def _pd_control_to_cell(state: State, next_cell: tuple[int, int]) -> Tuple[int, int]:
    """
    Lokális vezérlő, ami figyelembe veszi a TÖBBI JÁTÉKOST is.

    Lépések:
      - végigpróbál minden (ax, ay) ∈ {-1,0,1}×{-1,0,1} gyorsulást,
      - kiszámolja az új sebességet és pozíciót (new_vel, new_pos),
      - ha new_pos bármelyik másik játékos pozíciója → ELDOBJA (ütközés miatt),
      - a maradék jelöltek közül egy egyszerű költség alapján választ:
          * minél közelebb legyen a next_cell-hez,
          * ésszerű sebesség (ne legyen túl gyors),
          * kisebb gyorsítás előnyben.
      - ha MINDEN lehetséges lépés ütközne, inkább fékez (próbál meg lassulni).
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

            # Ütközés ellenőrzése: ne lépjünk másik játékos cellájára
            if any(np.array_equal(new_pos_int, op) for op in other_positions):
                continue

            # Távolság a kívánt cellától
            dist = float(np.linalg.norm(desired_pos - new_pos))

            # Sebesség büntetés: szeretnénk nagyjából "dist" nagyságú sebességet
            speed = float(np.linalg.norm(new_vel))
            desired_speed = min(4.0, dist)
            speed_penalty = abs(speed - desired_speed)

            # Kis büntetés a gyorsulás abszolút nagyságára
            acc_mag = abs(ax) + abs(ay)

            score = dist + 0.5 * speed_penalty + 0.1 * acc_mag

            if score < best_score:
                best_score = score
                best_acc = (ax, ay)

    if best_score == float('inf'):
        # Minden lehetséges lépés ütközne → inkább fékezzünk
        vel = state.agent.vel
        ax = int(np.clip(-vel[0], -1, 1))
        ay = int(np.clip(-vel[1], -1, 1))
        return (ax, ay)

    return best_acc


# ────────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────────

def main():
    """
    Fő belépési pont:
      - kiírja a READY-t,
      - beolvassa a pálya méretét (HxW, num_players, visibility_radius),
      - létrehoz egy GlobalMap-et,
      - ciklusban:
          * read_observation -> új State
          * calculate_move_logic -> (dx, dy) gyorsulás
          * kiírja a gyorsulást.
    """
    print('READY', flush=True)

    circuit = read_initial_observation()
    if circuit is None:
        return

    # Globális térkép inicializálása
    gmap = GlobalMap(circuit.track_shape)

    # Kezdő state (visible_track és agent majd read_observation-ben áll be)
    state: Optional[State] = State(circuit, None, [], None)  # type: ignore

    while True:
        state = read_observation(state)
        if state is None:
            break

        dx, dy = calculate_move_logic(state, gmap)
        print(f'{dx} {dy}', flush=True)


if __name__ == "__main__":
    main()
