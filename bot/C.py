import sys
import enum
from collections import deque
from typing import Optional, NamedTuple, List, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────────────────────────
# Cell típusok (egyezzenek a judge / grid_race_env értékeivel)
# ────────────────────────────────────────────────────────────────────────────────

class CellType(enum.IntEnum):
    """
    CellType értékek: ezeknek egyezniük kell a grid_race_env.CellType-tal.
    GOAL, START, WALL, UNKNOWN, EMPTY, NOT_VISIBLE, OIL, SAND.
    """
    GOAL = 100
    START = 1
    WALL = -1
    UNKNOWN = 2
    EMPTY = 0
    NOT_VISIBLE = 3
    OIL = 91
    SAND = 92


HAZARDS = {CellType.OIL.value, CellType.SAND.value}


# ────────────────────────────────────────────────────────────────────────────────
# Alap típusok
# ────────────────────────────────────────────────────────────────────────────────

class Player(NamedTuple):
    """
    Egy játékos állapota (pozíció + sebesség).
    """
    x: int
    y: int
    vel_x: int
    vel_y: int

    @property
    def pos(self) -> np.ndarray:
        """Visszaadja a pozíciót [row, col] numpy vektorként."""
        return np.array([self.x, self.y], dtype=int)

    @property
    def vel(self) -> np.ndarray:
        """Visszaadja a sebességet [vx, vy] numpy vektorként."""
        return np.array([self.vel_x, self.vel_y], dtype=int)


class Circuit(NamedTuple):
    """
    A pályáról érkező állandó adatok: méret, játékosok száma, látótáv.
    """
    track_shape: tuple[int, int]
    num_players: int
    visibility_radius: int


class State(NamedTuple):
    """
    A bot által tárolt állapot:
      - circuit: pálya metaadatok
      - visible_track: SAJÁT világmodellünk (teljes HxW rács, int cellaértékek)
      - players: többi játékos pozíciója
      - agent: saját játékosunk állapota
    """
    circuit: Circuit
    visible_track: Optional[np.ndarray]
    players: List[Player]
    agent: Optional[Player]


# ────────────────────────────────────────────────────────────────────────────────
# Bemenet olvasása
# ────────────────────────────────────────────────────────────────────────────────

def read_initial_observation() -> Circuit:
    """
    Beolvassa a kezdeti adatokat:
      H, W, num_players, visibility_radius
    Ezekből létrehoz egy Circuit objektumot.
    """
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)


def read_observation(old_state: State) -> Optional[State]:
    """
    Minden kör elején beolvassa:
      - a saját pozíciónkat + sebességünket
      - a többi játékos pozícióját
      - a lokális (2R+1)x(2R+1) rácsot

    A lokális rácsból frissíti a SAJÁT világmodellünket (teljes HxW rácsot),
    ahol:
      - amit látunk, azt felülírjuk
      - a NOT_VISIBLE cellákat NEM írjuk rá, így ott megmarad a korábbi érték
        (tipikusan UNKNOWN, amíg egyszer meg nem látjuk).
    """
    line = input().strip()
    if line == '~~~END~~~':
        return None

    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)

    circuit_data = old_state.circuit

    # Játékoslista beolvasása (csak pozíciókat ismerjük, sebességet 0-ra tesszük)
    players: List[Player] = []
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, input().split())
        players.append(Player(pposx, pposy, 0, 0))

    H, W = circuit_data.track_shape
    R = circuit_data.visibility_radius

    # Világmodell inicializálás: első körben minden UNKNOWN
    if old_state.visible_track is None:
        world = np.full((H, W), CellType.UNKNOWN.value, dtype=int)
    else:
        world = old_state.visible_track.copy()

    # Lokális térkép beolvasása és beillesztése a világmodellbe
    for i in range(2 * R + 1):
        row_vals = [int(a) for a in input().split()]
        gx = posx - R + i  # globális x (sor)
        if gx < 0 or gx >= H:
            continue

        y_start = posy - R
        y_end = y_start + 2 * R + 1

        local_line = row_vals

        # Levágás bal oldalon
        if y_start < 0:
            local_line = local_line[-y_start:]
            y_start = 0

        # Levágás jobb oldalon
        if y_end > W:
            local_line = local_line[:-(y_end - W)]
            y_end = W

        # local_line hossza most = y_end - y_start
        for dy, val in enumerate(local_line):
            gy = y_start + dy
            # NOT_VISIBLE: nem írunk rá, hogy megmaradjon a korábbi tudás
            if val == CellType.NOT_VISIBLE.value:
                continue
            world[gx, gy] = val

    return State(circuit_data, world, players, agent)


# ────────────────────────────────────────────────────────────────────────────────
# Geometria: vonalellenőrzés (ütközés falakkal)
# ────────────────────────────────────────────────────────────────────────────────

def traversable(cell_value: int) -> bool:
    """
    Annak eldöntése, hogy egy cella BELÁTHATÓAN járható-e a fal-ellenőrzéshez.
    Itt csak a WALL blokkol, minden nem negatív érték átjárható (út, start,
    cél, olaj, homok, stb.).
    """
    return cell_value >= 0


def valid_line(state: State, pos1: np.ndarray, pos2: np.ndarray) -> bool:
    """
    Megnézi, hogy a pos1 és pos2 közötti egyenes út közben ütköznénk-e falba.
    Ugyanaz az elv, mint a judge kódjában:
      - végigmegy a rácson a két pont közötti szakasz mentén
      - ha két egymás feletti / melletti cella mindkettő fal, az blokkol.
    """
    track = state.visible_track
    assert track is not None, "visible_track must be initialised before using valid_line."

    # Pályán belül vagyunk?
    if (np.any(pos1 < 0) or np.any(pos2 < 0) or
            np.any(pos1 >= track.shape) or np.any(pos2 >= track.shape)):
        return False

    diff = pos2 - pos1

    # Függőleges irányú lépés ellenőrzése (north-south)
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))  # +1 vagy -1
        for i in range(abs(diff[0]) + 1):
            x = int(pos1[0] + i * d)
            y = pos1[1] + i * slope * d
            y_ceil = int(np.ceil(y))
            y_floor = int(np.floor(y))
            if (not traversable(track[x, y_ceil])
                    and not traversable(track[x, y_floor])):
                return False

    # Vízszintes irányú lépés ellenőrzése (east-west)
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))  # +1 vagy -1
        for i in range(abs(diff[1]) + 1):
            x = pos1[0] + i * slope * d
            y = int(pos1[1] + i * d)
            x_ceil = int(np.ceil(x))
            x_floor = int(np.floor(x))
            if (not traversable(track[x_ceil, y])
                    and not traversable(track[x_floor, y])):
                return False

    return True


# ────────────────────────────────────────────────────────────────────────────────
# Világmodell alapú útkeresés (frontier + "A*" helyett egyszerű BFS)
# ────────────────────────────────────────────────────────────────────────────────

def is_passable_bfs(val: int, allow_hazards: bool) -> bool:
    """
    Eldönti, hogy egy cellán átléphetünk-e az útkeresés során.

    Paraméterek:
      - val: cella értéke
      - allow_hazards: ha False, akkor olaj/homok cellákat falnak tekintjük

    UNKNOWN és WALL sosem átjárható, mert oda még nem akarunk konkrétan lépni.
    """
    if val == CellType.WALL.value:
        return False
    if val == CellType.UNKNOWN.value:
        return False
    if not allow_hazards and val in HAZARDS:
        return False
    return True


def compute_frontiers(track: np.ndarray) -> List[Tuple[int, int]]:
    """
    Frontier cellák kiválasztása:
      - olyan ismert, nem-fal, nem-hazard cellák, amelyek legalább egy
        UNKNOWN szomszéddal rendelkeznek (4-irányú szomszédság).
    Ezek felé érdemes menni felfedezéskor.
    """
    H, W = track.shape
    frontiers: List[Tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            val = track[r, c]
            if val == CellType.WALL.value:
                continue
            if val == CellType.UNKNOWN.value:
                continue
            if val == CellType.NOT_VISIBLE.value:
                continue
            # A frontiereket csak biztonságos cellákból választjuk (nem hazard)
            if val in HAZARDS:
                continue

            # Van-e UNKNOWN szomszéd?
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if track[nr, nc] == CellType.UNKNOWN.value:
                        frontiers.append((r, c))
                        break
    return frontiers


def bfs_shortest_path(
    track: np.ndarray,
    start: Tuple[int, int],
    targets: List[Tuple[int, int]],
    allow_hazards: bool
) -> Optional[List[Tuple[int, int]]]:
    """
    Egyszerű BFS (egyenlő élköltségű) útkeresés a legközelebbi célcelláig.

    Paraméterek:
      - track: világmodell rács (int cellaértékek)
      - start: induló pozíció (r, c)
      - targets: cél cellák listája
      - allow_hazards: engedjük-e, hogy az út olaj/homok cellákon vezessen

    Visszatér:
      - path: [start, ..., target] cellák listája, ha van elérhető cél
      - None: ha nincs elérhető cél
    """
    H, W = track.shape
    sr, sc = start
    if not (0 <= sr < H and 0 <= sc < W):
        return None

    # Ha a start cella sem passzol, nincs értelme keresni
    if not is_passable_bfs(track[sr, sc], allow_hazards):
        return None

    target_set = set(targets)
    if (sr, sc) in target_set:
        return [start]

    visited = [[False] * W for _ in range(H)]
    parent: dict[Tuple[int, int], Tuple[int, int]] = {}

    q = deque()
    q.append((sr, sc))
    visited[sr][sc] = True

    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if visited[nr][nc]:
                continue
            if not is_passable_bfs(track[nr, nc], allow_hazards):
                continue

            visited[nr][nc] = True
            parent[(nr, nc)] = (r, c)

            if (nr, nc) in target_set:
                # Út rekonstruálása
                path = [(nr, nc)]
                cur = (nr, nc)
                while cur != (sr, sc):
                    cur = parent[cur]
                    path.append(cur)
                path.append((sr, sc))
                path.reverse()
                return path

            q.append((nr, nc))

    return None


def plan_path(state: State) -> Tuple[Optional[List[Tuple[int, int]]], bool]:
    """
    Magas szintű útvonaltervezés:

      1) Ha már látunk GOAL cellá(ka)t:
         - először megpróbál hazard-mentes utat találni (allow_hazards=False),
         - ha nincs ilyen, engedélyezi a hazardot (allow_hazards=True).

      2) Ha még nincs ismert GOAL:
         - frontier cellákhoz tervez utat (olyan ismert cellák, amelyek
           UNKNOWN szomszédot határolnak),
         - először hazard nélkül, ha az nem megy, akkor hazarddal.

    Visszatér:
      - path: [start, ..., target] cellák listája, vagy None ha nincs
      - allow_hazards: bool, hogy az ehhez a path-hoz engedélyeztük-e
        a hazard cellák használatát (erre a lokális vezérlésnek szüksége van).
    """
    assert state.visible_track is not None
    track = state.visible_track
    pos = state.agent.pos  # type: ignore
    start = (int(pos[0]), int(pos[1]))

    # 1) Ha van cél (GOAL) a már látott pályán, arra megyünk
    goal_cells = np.argwhere(track == CellType.GOAL.value)
    if goal_cells.size > 0:
        goals = [(int(r), int(c)) for r, c in goal_cells]
        # Először hazard nélkül
        path = bfs_shortest_path(track, start, goals, allow_hazards=False)
        if path is not None:
            return path, False
        # Ha csak hazardon át lehet odaérni, akkor azt is engedjük
        path = bfs_shortest_path(track, start, goals, allow_hazards=True)
        if path is not None:
            return path, True

    # 2) Felfedezés: frontier cellák keresése
    frontiers = compute_frontiers(track)
    if not frontiers:
        # Nincs frontier: vagy mindent látunk, vagy a maradék ismeretlen
        # régió elérhetetlen – ilyenkor nincs jó cél, path=None
        return None, False

    # Először hazard nélkül
    path = bfs_shortest_path(track, start, frontiers, allow_hazards=False)
    if path is not None:
        return path, False

    # Ha minden frontier útvonalát hazard zárja el, engedjük a hazardot
    path = bfs_shortest_path(track, start, frontiers, allow_hazards=True)
    if path is not None:
        return path, True

    # Semmi sem érhető el
    return None, False


# ────────────────────────────────────────────────────────────────────────────────
# Lokális vezérlés: gyorsulás választása egy célcella felé
# ────────────────────────────────────────────────────────────────────────────────

def choose_accel_toward_target(
    state: State,
    target_cell: np.ndarray,
    allow_hazards: bool
) -> Tuple[int, int]:
    """
    Lokális vezérlő:
      - összes lehetséges gyorsulást (ax, ay) ∈ {-1,0,1}×{-1,0,1} végignézi
      - kiszámolja az ebből adódó új sebességet és pozíciót
      - ellenőrzi:
          * pályán belül marad-e,
          * a lépés hossza (pos → new_pos) nem nagyobb, mint a látótáv,
          * nem megy falnak (valid_line),
          * nem lép másik játékos cellájára,
          * ha allow_hazards=False, nem lép hazard cellára
      - egy egyszerű költségfüggvény alapján kiválasztja a legjobb lépést:
          * minél közelebb legyünk a target_cell-hez
          * a sebesség se legyen se túl nagy, se túl kicsi
          * lehetőleg kerüljük a hazardot, még ha allow_hazards=True is
    """
    assert state.visible_track is not None
    track = state.visible_track
    pos = state.agent.pos.astype(int)   # type: ignore
    vel = state.agent.vel.astype(int)   # type: ignore

    vis_r = state.circuit.visibility_radius
    H, W = track.shape

    best_acc = (0, 0)
    best_score = float('inf')

    target_cell = target_cell.astype(float)

    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            new_vel = vel + np.array([ax, ay], dtype=int)
            new_pos = pos + new_vel

            nr, nc = int(new_pos[0]), int(new_pos[1])

            # Pályán belül?
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue

            # Ne lépjünk nagyobbat, mint a látótáv, hogy a vonalban lévő
            # összes cellát biztosan láttuk már.
            step_vec = new_pos - pos
            step_dist = float(np.linalg.norm(step_vec, ord=2))
            if step_dist > vis_r:
                continue

            # Fal-ellenőrzés
            if not valid_line(state, pos, new_pos):
                continue

            # Ne ütközzünk másik játékossal
            collision = any(np.array_equal(new_pos, p.pos) for p in state.players)
            if collision:
                continue

            cell_val = track[nr, nc]
            hazard_here = cell_val in HAZARDS

            # Ha hazard használata tiltott, akkor ide nem lépünk
            if hazard_here and not allow_hazards:
                continue

            # Távolság a célcellától a lépés után
            dist_after = float(np.linalg.norm(target_cell - new_pos, ord=2))

            # Sebesség büntetés: szeretnénk nem túlságosan gyorsan menni,
            # különösen a cél közelében.
            speed = float(np.linalg.norm(new_vel, ord=2))
            desired_speed = min(4.0, dist_after)
            speed_penalty = abs(speed - desired_speed)

            # Hazard büntetés akkor is, ha épp engedtük (csak ha muszáj, menjünk rá)
            hazard_penalty = 0.0
            if hazard_here:
                hazard_penalty = 5.0

            # Gyorsulás nagyságának kis büntetése, hogy ne “rángassuk” a mozgást
            acc_mag = abs(ax) + abs(ay)
            acc_penalty = 0.1 * acc_mag

            score = dist_after + speed_penalty + hazard_penalty + acc_penalty

            # Tie-break: kisebb gyorsulás előnyben
            if score < best_score - 1e-6 or (
                abs(score - best_score) <= 1e-6 and acc_mag < abs(best_acc[0]) + abs(best_acc[1])
            ):
                best_score = score
                best_acc = (ax, ay)

    # Ha valamiért semmi sem volt érvényes, maradunk (0,0)
    if best_score == float('inf'):
        return (0, 0)

    return best_acc


# ────────────────────────────────────────────────────────────────────────────────
# Fő döntési logika: calculate_move
# ────────────────────────────────────────────────────────────────────────────────

def calculate_move(rng: np.random.Generator, state: State) -> tuple[int, int]:
    """
    A bot fő döntési függvénye:
      1) Meghívja a plan_path-et, hogy kapjon egy útvonalat (BFS) és egy
         allow_hazards flag-et.
      2) Ha van path, a path[1]-et tekinti következő "subgoal" cellának.
      3) A choose_accel_toward_target segítségével kiválaszt egy (ax, ay)
         gyorsulást, ami ebbe az irányba visz, miközben falakat/hazardot
         kerül.
    """
    if state.visible_track is None or state.agent is None:
        # Biztonsági fallback: amíg nincs rendes állapot, maradjunk egy helyben.
        return (0, 0)

    path, allow_hazards = plan_path(state)

    # Ha nincs se cél, se frontier, próbáljunk inkább megállni/nyugton maradni
    if path is None or len(path) < 2:
        # Egyszerű "fékezés": a sebesség ellentétével gyorsítunk
        vel = state.agent.vel
        ax = -np.sign(vel[0])
        ay = -np.sign(vel[1])
        return (int(ax), int(ay))

    # path[0] = aktuális cella, path[1] = következő rácspont, ami felé megyünk
    next_cell = np.array(path[1], dtype=int)
    ax, ay = choose_accel_toward_target(state, next_cell, allow_hazards)
    return (int(ax), int(ay))


# ────────────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    """
    Fő belépési pont:
      - kiírja a READY jelet
      - beolvassa az initial observation-t
      - inicializál egy State-et (világmodell None-ként indul)
      - minden körben:
          * read_observation -> frissített State
          * calculate_move -> (ax, ay)
          * kiírja a (ax, ay)-t
    """
    print('READY', flush=True)
    circuit = read_initial_observation()
    state: Optional[State] = State(circuit, None, [], None)
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
