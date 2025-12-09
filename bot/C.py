import sys  
import enum  
import numpy as np  
import heapq  
from collections import deque  
from typing import Optional, NamedTuple, List, Tuple  


# Típusdefiníciók és konstansok


class CellType(enum.Enum):
    """
    A pálya elemei és a hozzájuk tartozó kódok.
    """
    WALL = -1           # Fal típus kódja, ezen nem lehet átmenni
    EMPTY = 0           # Üres út kódja, ide léphetünk
    START = 1           # Start pozíció kódja
    UNKNOWN = -2        # Belső jelölés: olyan terület, amit még sosem láttunk
    NOT_VISIBLE = 3     # Szerver jelölés: jelenleg a ködben van (nem látjuk)
    OIL = 91            # Olajfolt kódja (csúszós akadály)
    SAND = 92           # Homok kódja (lassító akadály)
    GOAL = 100          # Célvonal kódja

# Létrehozunk egy halmazt a veszélyes mezőkből a gyors ellenőrzéshez
HAZARDS = {CellType.OIL.value, CellType.SAND.value}


class Player(NamedTuple):
    """
    Egy játékos adatai: pozíció és sebesség tárolására szolgáló struktúra.
    """
    x: int      # Játékos X koordinátája (sor)
    y: int      # Játékos Y koordinátája (oszlop)
    vel_x: int  # Sebességvektor X irányú komponense
    vel_y: int  # Sebességvektor Y irányú komponense

    # Segédfüggvény (property), hogy tömbként is lekérhessük a pozíciót
    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])  # Visszaadja a [sor, oszlop] vektort

    # Segédfüggvény a sebességvektor lekéréséhez
    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vel_x, self.vel_y]) # Visszaadja a [vx, vy] vektort


class Circuit(NamedTuple):
    """
    A pálya fix adatai, ezt csak egyszer kapjuk meg az elején.
    """
    track_shape: tuple[int, int]  # A pálya mérete (magasság, szélesség)
    num_players: int              # Játékosok száma
    visibility_radius: int        # Milyen messzire látunk (sugár)


class State(NamedTuple):
    """
    Az aktuális kör állapota: mit látunk, hol vannak a többiek.
    """
    circuit: Circuit              # Hivatkozás a pálya adataira
    visible_track: np.ndarray     # A pályának az a része, amit most látunk
    players: list[Player]         # Lista az összes játékos adatával
    agent: Player                 # Külön tároljuk a saját botunk adatait


# ────────────────────────────────────────────────────────────────────────────────
# TÉRKÉP KEZELÉS (Világmodell)
# ────────────────────────────────────────────────────────────────────────────────

class GlobalMap:
    """
    Itt tároljuk a teljes pályát. Kezdetben tiszta homály (UNKNOWN),
    aztán ahogy haladunk, folyamatosan töltjük fel adatokkal.
    """

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape  # Eltároljuk a pálya méretét
        # Létrehozunk egy akkora mátrixot, mint a pálya, csupa UNKNOWN (-2) értékkel
        self.grid = np.full(shape, CellType.UNKNOWN.value, dtype=int)

    def update(self, visible_track: np.ndarray):
        """
        Frissítjük a térképet azzal, amit éppen látunk.
        Csak azt írjuk felül, ami ténylegesen látszik (nem NOT_VISIBLE).
        """
        # Készítünk egy logikai maszkot: True ott, ahol NEM 'nem látható' a mező
        mask = (visible_track != CellType.NOT_VISIBLE.value)
        # A globális térkép megfelelő helyeire bemásoljuk a most látott értékeket
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
    queue = deque([start])  # Létrehozzuk a sort a BFS-hez, a kezdőponttal
    visited = {start}       # Nyilvántartjuk a már meglátogatott mezőket

    # Ha véletlenül pont ismeretlen mezőn állnánk (elvileg lehetetlen, de biztos ami biztos)
    if gmap.grid[start] == CellType.UNKNOWN.value:
        return start

    H, W = gmap.shape  # Pálya dimenziói

    # A 8 lehetséges szomszédos irány koordináta-eltolásai (átlósan is)
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while queue:  # Amíg van elem a sorban
        cx, cy = queue.popleft()  # Kivesszük a sor elejét (jelenlegi pozíció)
        for dx, dy in directions: # Végigmegyünk minden irányon
            nx, ny = cx + dx, cy + dy  # Kiszámoljuk az új koordinátát
            
            # Ellenőrizzük, hogy a pályán belül vagyunk-e
            if not (0 <= nx < H and 0 <= ny < W):
                continue # Ha nem, ugrunk a következőre
            
            # Ha már láttuk ezt a mezőt a keresés során, kihagyjuk
            if (nx, ny) in visited:
                continue

            val = gmap.grid[nx, ny]  # Lekérjük a mező típusát a térképről

            # Ha fal, akkor nem mehetünk arra
            if val == CellType.WALL.value:
                continue

            # Ha kerülni kell a veszélyt, és ez veszélyes mező, kihagyjuk
            if avoid_hazards and val in HAZARDS:
                continue

            # Ha ismeretlen mezőt találtunk, megvan a cél!
            if val == CellType.UNKNOWN.value:
                return (nx, ny) # Visszaadjuk a koordinátát

            # Ha átjárható, de nem cél, akkor berakjuk a sorba további keresésre
            visited.add((nx, ny))
            queue.append((nx, ny))

    return None  # Ha elfogyott a sor és nem találtunk semmit


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
    H, W = gmap.shape  # Pálya méretei
    (sx, sy) = start   # Start koordináták kibontása
    (gx, gy) = goal    # Cél koordináták kibontása

    # Ellenőrzés: Start a pályán van-e
    if not (0 <= sx < H and 0 <= sy < W): return []
    # Ellenőrzés: Cél a pályán van-e
    if not (0 <= gx < H and 0 <= gy < W): return []
    # Ellenőrzés: Ha a cél falban van, nem tudunk odamenni
    if gmap.grid[gx, gy] == CellType.WALL.value: return []

    # Prioritási sor inicializálása: (költség, koordináta)
    pq: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
    # Honnan jöttünk? Ez kell majd az útvonal rekonstrukcióhoz
    came_from: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    # Eddigi költség a starthoz képest
    cost_so_far: dict[Tuple[int, int], float] = {start: 0.0}

    # 8 irány definíciója
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while pq:  # Amíg van vizsgálandó csomópont
        _, current = heapq.heappop(pq)  # Kivesszük a legkisebb költségűt
        if current == goal:  # Ha elértük a célt
            break            # Vége a keresésnek

        cx, cy = current  # Jelenlegi koordináták

        for dx, dy in directions:  # Szomszédok vizsgálata
            if dx == 0 and dy == 0: continue # Helyben maradás nem érdekel
            
            nx, ny = cx + dx, cy + dy  # Szomszéd koordinátája
            # Pályahatár ellenőrzés
            if not (0 <= nx < H and 0 <= ny < W): continue

            val = gmap.grid[nx, ny]  # Mező típusa

            if val == CellType.WALL.value: continue # Falon nem megyünk át
            # Ha kerülni kell a veszélyt, akkor itt is tiltott
            if avoid_hazards and val in HAZARDS: continue

            # Költség meghatározása: Veszélyes mező drága (20), sima olcsó (1)
            step_cost = 20.0 if val in HAZARDS else 1.0

            # Új költség = eddigi költség + lépés költsége
            new_cost = cost_so_far[current] + step_cost

            # Ha még nem voltunk itt, vagy olcsóbb útvonalat találtunk
            if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                cost_so_far[(nx, ny)] = new_cost # Frissítjük a költséget
                # Heurisztika: Manhattan távolság a célig (hátralévő becsült út)
                priority = new_cost + abs(gx - nx) + abs(gy - ny)
                # Berakjuk a prioritási sorba
                heapq.heappush(pq, (priority, (nx, ny)))
                # Eltároljuk, hogy ide a 'current' mezőről léptünk
                came_from[(nx, ny)] = current

    # Ha a cél nincs a 'came_from'-ban, akkor nem találtunk utat
    if goal not in came_from:
        return []

    # Útvonal visszafejtése a céltól a startig
    path: List[Tuple[int, int]] = []
    curr: Optional[Tuple[int, int]] = goal
    while curr is not None:  # Amíg vissza nem érünk a starthoz (ahol None a szülő)
        path.append(curr)    # Hozzáadjuk az útvonalhoz
        curr = came_from[curr] # Lépünk visszafelé
    path.reverse()  # Megfordítjuk a listát, hogy start->cél sorrend legyen
    return path


# ────────────────────────────────────────────────────────────────────────────────
# KOMMUNIKÁCIÓ A SZERVERREL
# ────────────────────────────────────────────────────────────────────────────────

def read_initial_observation() -> Circuit:
    """
    A legelső sor beolvasása: pálya mérete, játékosok száma, látótávolság.
    """
    line = sys.stdin.readline()  # Beolvassuk az első sort
    if not line:
        return None  # Ha üres, kilépünk (hiba vagy vége)
    # Feldaraboljuk a sort és egésszé (int) konvertáljuk
    H, W, num_players, visibility_radius = map(int, line.split())
    # Visszaadjuk a Circuit objektumot
    return Circuit((H, W), num_players, visibility_radius)


def read_observation(old_state: State) -> Optional[State]:
    """
    Minden kör elején beolvassuk az aktuális helyzetet.
    Itt rakjuk össze a 'visible_track'-et a kapott szeletekből.
    """
    line = sys.stdin.readline()  # Beolvassuk a saját botunk adatait
    # Ha nincs sor, vagy a szerver végét jelzett
    if not line or line.strip() == '~~~END~~~':
        return None

    # Feldolgozzuk a koordinátákat és sebességet
    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)  # Létrehozzuk a saját játékos objektumot

    players: List[Player] = []  # Lista a többi játékosnak
    circuit_data = old_state.circuit  # A pálya fix adatait az előző állapotból vesszük

    # Beolvassuk a többi játékos pozícióját
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, sys.stdin.readline().split()) # Beolvasás
        players.append(Player(pposx, pposy, 0, 0)) # Hozzáadás a listához (sebességet nem kapunk róluk)

    # Létrehozunk egy üres térképet, csupa 'NEM LÁTHATÓ' jelöléssel
    visible_track = np.full(
        circuit_data.track_shape,
        CellType.NOT_VISIBLE.value,
        dtype=int
    )

    R = circuit_data.visibility_radius  # Látótávolság rövidítése
    H, W = circuit_data.track_shape     # Pálya méretek

    # Beolvassuk a látott sorokat (2*R + 1 sornyit kapunk)
    for i in range(2 * R + 1):
        raw_line = sys.stdin.readline() # Egy sor beolvasása
        if not raw_line: break          # Ha hiba van, kilépünk
        vals = [int(a) for a in raw_line.split()] # Számokká alakítjuk

        # Kiszámoljuk, melyik globális sornak felel meg ez a sor
        x = posx - R + i
        # Ha ez a sor kilógna a pályáról, átugorjuk
        if x < 0 or x >= H: continue

        # Kiszámoljuk az oszlop-intervallumot (globális Y koordináták)
        y_start = posy - R
        y_end = y_start + 2 * R + 1

        # Előkészítjük a szeletelést (clipping), ha a látómező lelógna oldalra
        line_slice_start = 0
        line_slice_end = len(vals)
        target_y_start = max(0, y_start)   # Nem lehet kisebb mint 0
        target_y_end = min(W, y_end)       # Nem lehet nagyobb mint a pálya szélessége

        # Ha balra kilógunk, eltoljuk a kezdőindexet
        if y_start < 0:
            line_slice_start = -y_start
        # Ha jobbra kilógunk, csökkentjük a végindexet
        if y_end > W:
            line_slice_end -= (y_end - W)

        # Ha van érvényes adat, bemásoljuk a visible_track tömbbe
        if line_slice_start < line_slice_end:
            visible_track[x, target_y_start:target_y_end] = vals[
                line_slice_start:line_slice_end
            ]

    # Visszaadjuk az új State objektumot a friss adatokkal
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
    """
    # Ha az útvonal túl rövid (nincs legalább 1 lépés), akkor nincs mit tenni
    if len(path) < 2:
        return None

    my_pos = (state.agent.x, state.agent.y)  # Saját pozíció
    # Halmazba gyűjtjük az összes játékos pozícióját
    occupied = {(p.x, p.y) for p in state.players}
    
    # Saját magunk nem számítunk akadálynak, kivesszük a halmazból
    if my_pos in occupied:
        occupied.remove(my_pos)

    # Végigmegyünk az útvonalon az első lépéstől kezdve (a 0. a start)
    for cell in path[1:]:
        # Ha a mező nincs a foglaltak között
        if cell not in occupied:
            return cell  # Ez lesz a célmezőnk

    return None  # Ha minden mező foglalt az úton


# ────────────────────────────────────────────────────────────────────────────────
# FŐ DÖNTÉSHOZATAL
# ────────────────────────────────────────────────────────────────────────────────

def calculate_move_logic(state: State, gmap: GlobalMap) -> Tuple[int, int]:
    """
    Az agy. Itt dől el, mit csinálunk.
    """
    my_pos = (state.agent.x, state.agent.y) # Saját pozíció

    # 1. Frissítjük a világtérképet a most látott adatokkal
    gmap.update(state.visible_track)

    # 2. Megkeressük az összes CÉL (GOAL) mezőt a térképen
    goals = np.argwhere(gmap.grid == CellType.GOAL.value)
    
    # Ha látunk célt (van elem a goals listában)
    if len(goals) > 0:
        # Összeszedjük a foglalt mezőket (mások állnak ott)
        occupied = {(p.x, p.y) for p in state.players}
        if my_pos in occupied: occupied.remove(my_pos) # Magunkat kivesszük

        # Lehetséges célpontok kiértékelése
        candidates: List[Tuple[bool, int, Tuple[int, int]]] = []
        for gr, gc in goals:
            cell = (int(gr), int(gc)) # Cél koordináta
            dist = abs(gr - my_pos[0]) + abs(gc - my_pos[1]) # Távolság tőlünk
            is_occ = cell in occupied # Foglalt-e?
            # Hozzáadjuk a listához: (foglalt?, távolság, koordináta)
            candidates.append((is_occ, dist, cell))

        # Rendezés: elsődleges szempont, hogy NEM foglalt, másodlagos a távolság
        candidates.sort(key=lambda t: (t[0], t[1]))
        goal_tuple = candidates[0][2] # Kiválasztjuk a legjobbat

        # Tervezünk A* utat a célhoz, veszélyek elkerülésével
        path = run_astar(my_pos, goal_tuple, gmap, avoid_hazards=True)
        # Ha nincs biztonságos út, megpróbáljuk veszélyeken (homok/olaj) át is
        if len(path) < 2:
            path = run_astar(my_pos, goal_tuple, gmap, avoid_hazards=False)

        # Ha van érvényes útvonal
        if path:
            # Kiválasztjuk a következő szabad mezőt az úton
            next_cell = choose_next_free_cell_on_path(path, state)
            if next_cell is not None:
                # Kiszámoljuk a gyorsítást a vezérlővel
                return _pd_control_to_cell(state, next_cell)

    # 3. Ha nincs látható cél, jön a FELDERÍTÉS (Explore)
    # Keresünk ismeretlen területet, kerülve a veszélyeket
    target = find_nearest_unknown(my_pos, gmap, avoid_hazards=True)
    allow_hazards_for_explore = False

    # Ha így nem találunk, akkor megengedjük a veszélyes utakat is
    if target is None:
        target = find_nearest_unknown(my_pos, gmap, avoid_hazards=False)
        allow_hazards_for_explore = True

    # Ha találtunk felderítési célt
    if target is not None:
        # Tervezünk oda egy utat
        path = run_astar(
            my_pos,
            target,
            gmap,
            # Ha muszáj, átmegyünk a veszélyen
            avoid_hazards=not allow_hazards_for_explore
        )
        # Ha nincs biztonságos út, de a cél csak veszélyesen érhető el
        if len(path) < 2 and allow_hazards_for_explore:
            path = run_astar(my_pos, target, gmap, avoid_hazards=False)

        # Ha van út, megyünk
        if path:
            next_cell = choose_next_free_cell_on_path(path, state)
            if next_cell is not None:
                return _pd_control_to_cell(state, next_cell)

    # 4. Pánik / Pihenő mód: Ha nincs semmi teendő, fékezünk
    vel = state.agent.vel # Jelenlegi sebesség
    # Ellentétes irányú gyorsítást számolunk a megálláshoz (clip -1 és 1 közé)
    ax = int(np.clip(-vel[0], -1, 1))
    ay = int(np.clip(-vel[1], -1, 1))
    return (ax, ay) # Visszaadjuk a fék parancsot


def _pd_control_to_cell(state: State, next_cell: tuple[int, int]) -> Tuple[int, int]:
    """
    Egy "okosabb" mozgásvezérlő. Kiszámolja a legjobb gyorsítást (ax, ay).
    """
    # A célmező koordinátája lebegőpontosként
    desired_pos = np.array(next_cell, dtype=float)
    # Jelenlegi pozíció és sebesség
    current_pos = state.agent.pos.astype(float)
    current_vel = state.agent.vel.astype(float)
    # Többi játékos pozícióinak listája
    other_positions = [p.pos for p in state.players]

    best_score = float('inf') # Legjobb pontszám (kezdetben végtelen, mert minimalizálunk)
    best_acc = (0, 0)         # Legjobb gyorsítás tárolója

    # Brute-force: végignézzük mind a 9 lehetséges gyorsítást (-1, 0, 1)
    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            # Kiszámoljuk az új sebességet
            new_vel = current_vel + np.array([ax, ay], dtype=float)
            # Kiszámoljuk, hova érkeznénk ezzel a sebességgel
            new_pos = current_pos + new_vel
            new_pos_int = new_pos.astype(int) # Egész számra kerekítjük az ellenőrzéshez

            # Ütközésvizsgálat: ha pont rálépnénk valakire, az felejtős
            if any(np.array_equal(new_pos_int, op) for op in other_positions):
                continue

            # Távolság a célmezőtől
            dist = float(np.linalg.norm(desired_pos - new_pos))
            # Új sebesség nagysága
            speed = float(np.linalg.norm(new_vel))
            
            # Kívánt sebesség: ha messze vagyunk, lehet gyors, ha közel, lassítunk
            desired_speed = min(4.0, dist)
            # Büntetés, ha a sebességünk eltér az ideálistól
            speed_penalty = abs(speed - desired_speed)
            
            # Energiatakarékosság: a nagy gyorsítás picit büntetve van (hogy ne rángasson)
            acc_mag = abs(ax) + abs(ay)
            
            # A pontszám kalkulációja (súlyozott összeg)
            score = dist + 0.5 * speed_penalty + 0.1 * acc_mag

            # Ha ez jobb (kisebb) pontszám, mint eddig bármi, elmentjük
            if score < best_score:
                best_score = score
                best_acc = (ax, ay)

    # Ha minden lépés ütközéshez vezet (score maradt végtelen)
    if best_score == float('inf'):
        # Akkor vészfékezünk
        vel = state.agent.vel
        ax = int(np.clip(-vel[0], -1, 1))
        ay = int(np.clip(-vel[1], -1, 1))
        return (ax, ay)

    return best_acc # Visszaadjuk a legjobb gyorsítást


# ────────────────────────────────────────────────────────────────────────────────
# MAIN PROGRAM
# ────────────────────────────────────────────────────────────────────────────────

def main():
    """
    A fő belépési pont.
    """
    print('READY', flush=True) # Jelezzük a szervernek, hogy készen állunk
    circuit = read_initial_observation() # Beolvassuk a pálya adatait
    if circuit is None: # Ha hiba volt, kilépünk
        return

    # Létrehozzuk a memóriában a térképet
    gmap = GlobalMap(circuit.track_shape)
    # Inicializáljuk az állapotot (egyelőre üres)
    state: Optional[State] = State(circuit, None, [], None) # type: ignore

    while True: # Végtelen ciklus a játék körökhöz
        state = read_observation(state) # Beolvassuk az új kör adatait
        if state is None: # Ha nincs adat (vége a meccsnek), kilépünk
            break
        
        # Itt történik a döntéshozatal
        dx, dy = calculate_move_logic(state, gmap)
        
        # Elküldjük a lépést (gyorsulás X, gyorsulás Y) a standard kimenetre
        print(f'{dx} {dy}', flush=True)


if __name__ == "__main__":
    main() # Futtatjuk a main függvényt