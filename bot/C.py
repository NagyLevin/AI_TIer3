import sys
import enum
import numpy as np
import heapq
from collections import deque
from typing import Optional, NamedTuple, List, Tuple

# ────────────────────────────────────────────────────────────────────────────────
# Típusdefiníciók és konstansok
# ────────────────────────────────────────────────────────────────────────────────

class CellType(enum.Enum):
    """
    A pálya elemei és a hozzájuk tartozó kódok.
    """
    WALL = -1           # Fal, ezen nem megyünk át
    EMPTY = 0           # Sima aszfalt
    START = 1           # Startvonal
    UNKNOWN = -2        # Belső 'köd', amit még nem láttunk (fog-of-war)
    NOT_VISIBLE = 3     # A szerver szerint ez most takarásban van
    OIL = 91            # Olajfolt, csúszik!
    SAND = 92           # Homokágy, nagyon lassít
    GOAL = 100          # A cél, ide akarunk eljutni

# Ezeket a veszélyes dolgokat próbáljuk elkerülni, ha lehet
HAZARDS = {CellType.OIL.value, CellType.SAND.value}


class Player(NamedTuple):
    """
    Egy játékos adatai: pozíció és sebesség.
    """
    x: int
    y: int
    vel_x: int
    vel_y: int

    # Kis segéd, hogy numpy-val könnyebb legyen számolni
    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vel_x, self.vel_y])


class Circuit(NamedTuple):
    """
    A pálya fix adatai, ezt csak egyszer kapjuk meg az elején.
    """
    track_shape: tuple[int, int]
    num_players: int
    visibility_radius: int


class State(NamedTuple):
    """
    Az aktuális kör állapota: mit látunk, hol vannak a többiek.
    """
    circuit: Circuit
    visible_track: np.ndarray   # A lokális látómezőnk
    players: list[Player]       # Mindenki más
    agent: Player               # A saját botunk


# ────────────────────────────────────────────────────────────────────────────────
# TÉRKÉP KEZELÉS (Világmodell)
# ────────────────────────────────────────────────────────────────────────────────

class GlobalMap:
    """
    Itt tároljuk a teljes pályát. Kezdetben tiszta homály (UNKNOWN),
    aztán ahogy haladunk, folyamatosan töltjük fel adatokkal.
    """

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        # Kezdetben minden ismeretlen (-2)
        self.grid = np.full(shape, CellType.UNKNOWN.value, dtype=int)

    def update(self, visible_track: np.ndarray):
        """
        Frissítjük a térképet azzal, amit éppen látunk.
        Csak azt írjuk felül, ami ténylegesen látszik (nem NOT_VISIBLE).
        """
        mask = (visible_track != CellType.NOT_VISIBLE.value)
        self.grid[mask] = visible_track[mask]


# ────────────────────────────────────────────────────────────────────────────────
# CÉLKERESÉS (Felderítés)
# ────────────────────────────────────────────────────────────────────────────────

def find_nearest_unknown(
    start: tuple[int, int],
    gmap: GlobalMap,
    avoid_hazards: bool = True
) -> Optional[tuple[int, int]]:
    """
    Sima szélességi keresés (BFS), hogy találjunk egy közeli ismeretlen mezőt.
    Ez lesz a cél, ha éppen nem látjuk a befutót.
    
    avoid_hazards: Ha igaz, kikerüljük az olajat/homokot is.
    """
    queue = deque([start])
    visited = {start}

    # Ha már eleve ismeretlenben állunk (fura lenne, de hátha)
    if gmap.grid[start] == CellType.UNKNOWN.value:
        return start

    H, W = gmap.shape

    # 8 irányba nézelődünk
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            
            # Pályán kívülre nem megyünk
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            if (nx, ny) in visited:
                continue

            val = gmap.grid[nx, ny]

            # Falnak nem megyünk
            if val == CellType.WALL.value:
                continue

            # Ha óvatosak vagyunk, a veszélyeket is kerüljük
            if avoid_hazards and val in HAZARDS:
                continue

            # Bingó! Találtunk egy ismeretlen foltot
            if val == CellType.UNKNOWN.value:
                return (nx, ny)

            visited.add((nx, ny))
            queue.append((nx, ny))

    return None


# ────────────────────────────────────────────────────────────────────────────────
# ÚTVONALTERVEZÉS (A*)
# ────────────────────────────────────────────────────────────────────────────────

def run_astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    gmap: GlobalMap,
    avoid_hazards: bool
) -> List[tuple[int, int]]:
    """
    A jó öreg A* algoritmus, hogy eljussunk A-ból B-be a lehető legolcsóbban.
    
    Költségek:
      - Sima út: 1
      - Veszély (ha engedjük): 20 (tehát inkább kerülünk, ha van más út)
    """
    H, W = gmap.shape
    (sx, sy) = start
    (gx, gy) = goal

    # Alapvető validációk
    if not (0 <= sx < H and 0 <= sy < W): return []
    if not (0 <= gx < H and 0 <= gy < W): return []
    if gmap.grid[gx, gy] == CellType.WALL.value: return []

    # Priority queue: (költség, pozíció)
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
            if dx == 0 and dy == 0: continue
            
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < H and 0 <= ny < W): continue

            val = gmap.grid[nx, ny]

            if val == CellType.WALL.value: continue
            if avoid_hazards and val in HAZARDS: continue

            # Költség számolás
            step_cost = 20.0 if val in HAZARDS else 1.0

            new_cost = cost_so_far[current] + step_cost

            if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                cost_so_far[(nx, ny)] = new_cost
                # Heurisztika: Manhattan távolság a célig
                priority = new_cost + abs(gx - nx) + abs(gy - ny)
                heapq.heappush(pq, (priority, (nx, ny)))
                came_from[(nx, ny)] = current

    if goal not in came_from:
        return []

    # Visszafejtjük az utat
    path: List[Tuple[int, int]] = []
    curr: Optional[Tuple[int, int]] = goal
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path


# ────────────────────────────────────────────────────────────────────────────────
# KOMMUNIKÁCIÓ A SZERVERREL
# ────────────────────────────────────────────────────────────────────────────────

def read_initial_observation() -> Circuit:
    """
    A legelső sor beolvasása: pálya mérete, játékosok száma, látótávolság.
    """
    line = sys.stdin.readline()
    if not line:
        return None  # type: ignore
    H, W, num_players, visibility_radius = map(int, line.split())
    return Circuit((H, W), num_players, visibility_radius)


def read_observation(old_state: State) -> Optional[State]:
    """
    Minden kör elején beolvassuk az aktuális helyzetet.
    Itt rakjuk össze a 'visible_track'-et a kapott szeletekből.
    """
    line = sys.stdin.readline()
    if not line or line.strip() == '~~~END~~~':
        return None

    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)

    players: List[Player] = []
    circuit_data = old_state.circuit

    # Többi játékos pozíciója
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, sys.stdin.readline().split())
        players.append(Player(pposx, pposy, 0, 0))

    # A látott területet először 'NOT_VISIBLE'-re állítjuk
    visible_track = np.full(
        circuit_data.track_shape,
        CellType.NOT_VISIBLE.value,
        dtype=int
    )

    R = circuit_data.visibility_radius
    H, W = circuit_data.track_shape

    # Beolvassuk a sorokat és beillesztjük a mátrixba
    for i in range(2 * R + 1):
        raw_line = sys.stdin.readline()
        if not raw_line: break
        vals = [int(a) for a in raw_line.split()]

        x = posx - R + i
        if x < 0 or x >= H: continue

        y_start = posy - R
        y_end = y_start + 2 * R + 1

        # Kicsit matekozni kell a szeleteléssel, ha lelógunk a pályáról
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
# ÚTVONAL UTÓFELDOLGOZÁS
# ────────────────────────────────────────────────────────────────────────────────

def choose_next_free_cell_on_path(
    path: List[Tuple[int, int]],
    state: State
) -> Optional[Tuple[int, int]]:
    """
    Végigmegyünk a tervezett útvonalon és kiválasztjuk az első olyan pontot,
    ahol épp NEM áll senki.
    
    Ez azért kell, mert ha pont a célmezőn áll valaki, és mi oda akarunk menni,
    akkor belécsúszhatunk vagy feleslegesen fékezünk.
    """
    if len(path) < 2:
        return None

    my_pos = (state.agent.x, state.agent.y)
    occupied = {(p.x, p.y) for p in state.players}
    
    # Saját magunk nem számítunk akadálynak
    if my_pos in occupied:
        occupied.remove(my_pos)

    # Keressünk egy szabad helyet az úton
    for cell in path[1:]:
        if cell not in occupied:
            return cell

    return None


# ────────────────────────────────────────────────────────────────────────────────
# FŐ DÖNTÉSHOZATAL
# ────────────────────────────────────────────────────────────────────────────────

def calculate_move_logic(state: State, gmap: GlobalMap) -> Tuple[int, int]:
    """
    Az agy. Itt dől el, mit csinálunk.
    
    Stratégia:
      1. Térkép frissítés.
      2. Látjuk a célt? -> Tervezzünk utat oda (ha kell, kerüljük a többieket).
      3. Nem látjuk? -> Keressük a legközelebbi ismeretlent (felderítés).
      4. Ha semmi nem jön össze -> Fékezzünk le.
    """
    my_pos = (state.agent.x, state.agent.y)

    # 1. Frissítjük a világtérképet
    gmap.update(state.visible_track)

    # 2. Célkeresés (GOAL)
    goals = np.argwhere(gmap.grid == CellType.GOAL.value)
    if len(goals) > 0:
        # Megnézzük kik vannak útban
        occupied = {(p.x, p.y) for p in state.players}
        if my_pos in occupied: occupied.remove(my_pos)

        # Összegyűjtjük a lehetséges célpontokat
        candidates: List[Tuple[bool, int, Tuple[int, int]]] = []
        for gr, gc in goals:
            cell = (int(gr), int(gc))
            dist = abs(gr - my_pos[0]) + abs(gc - my_pos[1])
            is_occ = cell in occupied
            candidates.append((is_occ, dist, cell))

        # Rendezés: először a szabadok, aztán a közelebbi
        candidates.sort(key=lambda t: (t[0], t[1]))
        goal_tuple = candidates[0][2]

        # Először próbáljuk meg "tisztán", veszélyek nélkül
        path = run_astar(my_pos, goal_tuple, gmap, avoid_hazards=True)
        # Ha nem megy, akkor bevállaljuk a homokot is
        if len(path) < 2:
            path = run_astar(my_pos, goal_tuple, gmap, avoid_hazards=False)

        if path:
            next_cell = choose_next_free_cell_on_path(path, state)
            if next_cell is not None:
                return _pd_control_to_cell(state, next_cell)

    # 3. Felderítés (EXPLORE)
    target = find_nearest_unknown(my_pos, gmap, avoid_hazards=True)
    allow_hazards_for_explore = False

    # Ha nem találunk tiszta utat az ismeretlenbe, akkor kockáztatunk
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

    # 4. Pánik / Pihenő mód: fékezünk
    vel = state.agent.vel
    ax = int(np.clip(-vel[0], -1, 1))
    ay = int(np.clip(-vel[1], -1, 1))
    return (ax, ay)


def _pd_control_to_cell(state: State, next_cell: tuple[int, int]) -> Tuple[int, int]:
    """
    Egy "okosabb" mozgásvezérlő. Nem csak vakon gyorsítunk a cél felé.
    
    Végignézzük mind a 9 lehetséges gyorsítást (-1, 0, 1 kombók), és pontozzuk őket:
      - Mennyire visz közel a célhoz?
      - Mennyire tartja a sebességet?
      - Nekiütközünk-e valakinek?
      
    A legjobbat választjuk.
    """
    desired_pos = np.array(next_cell, dtype=float)
    current_pos = state.agent.pos.astype(float)
    current_vel = state.agent.vel.astype(float)
    other_positions = [p.pos for p in state.players]

    best_score = float('inf')
    best_acc = (0, 0)

    # Brute-force végignézzük a gyorsításokat
    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            new_vel = current_vel + np.array([ax, ay], dtype=float)
            new_pos = current_pos + new_vel
            new_pos_int = new_pos.astype(int)

            # Ütközésvizsgálat: ha rálépnénk valakire, az felejtős
            if any(np.array_equal(new_pos_int, op) for op in other_positions):
                continue

            dist = float(np.linalg.norm(desired_pos - new_pos))
            speed = float(np.linalg.norm(new_vel))
            
            # Szeretnénk haladni, de nem túl gyorsan (hogy tudjunk kanyarodni)
            desired_speed = min(4.0, dist)
            speed_penalty = abs(speed - desired_speed)
            
            # Energiatakarékosság: a nagy gyorsítás picit büntetve van
            acc_mag = abs(ax) + abs(ay)
            
            # A pontszám (kisebb a jobb)
            score = dist + 0.5 * speed_penalty + 0.1 * acc_mag

            if score < best_score:
                best_score = score
                best_acc = (ax, ay)

    if best_score == float('inf'):
        # Ha minden lépés ütközéshez vezet, akkor vészfék
        vel = state.agent.vel
        ax = int(np.clip(-vel[0], -1, 1))
        ay = int(np.clip(-vel[1], -1, 1))
        return (ax, ay)

    return best_acc


# ────────────────────────────────────────────────────────────────────────────────
# MAIN PROGRAM
# ────────────────────────────────────────────────────────────────────────────────

def main():
    """
    A fő belépési pont.
      1. Kézfogás (READY)
      2. Pálya infók beolvasása
      3. Végtelen ciklus: olvasás -> gondolkodás -> cselekvés
    """
    print('READY', flush=True)
    circuit = read_initial_observation()
    if circuit is None:
        return

    gmap = GlobalMap(circuit.track_shape)
    state: Optional[State] = State(circuit, None, [], None) # type: ignore

    while True:
        state = read_observation(state)
        if state is None:
            break
        
        # Itt történik a csoda
        dx, dy = calculate_move_logic(state, gmap)
        
        # Elküldjük a lépést
        print(f'{dx} {dy}', flush=True)


if __name__ == "__main__":
    main()