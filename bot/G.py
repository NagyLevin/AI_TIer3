import sys      
import enum    
import numpy as np 
import heapq    
import os       
import datetime 
from collections import deque 
from typing import Optional, NamedTuple, List, Tuple, Set 

# --- JÁTÉK KONSTANSOK ---
# Ezek a számok jelölik, mi van a térkép egy adott kockáján.
class CellType(enum.Enum):
    WALL = -1       # Fal (nem lehet rá lépni)
    EMPTY = 0       # Üres út (aszfalt)
    START = 1       # Rajtvonal
    UNKNOWN = -2    # SAJÁT JELÖLÉS: Még nem láttuk ezt a részt (felfedezésre vár)
    NOT_VISIBLE = 3 # A szerver küldi: "ez a rész most nem látható"
    OIL = 91        # Olajfolt (megcsúszol rajta)
    SAND = 92       # Homok (lelassít)
    GOAL = 100      # Célvonal

# Egy játékos (vagy a saját botunk) adatait tároló struktúra
class Player(NamedTuple):
    x: int          # X koordináta
    y: int          # Y koordináta
    vel_x: int      # Sebesség X irányban
    vel_y: int      # Sebesség Y irányban

    # Segédfüggvény: visszaadja a pozíciót numpy tömbként (könnyebb számolni vele)
    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    # Segédfüggvény: visszaadja a sebességet numpy tömbként
    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vel_x, self.vel_y])

# A pálya alapvető adatait tároló struktúra
class Circuit(NamedTuple):
    track_shape: tuple[int, int] # A pálya mérete (Magasság, Szélesség)
    num_players: int             # Játékosok száma
    visibility_radius: int       # Milyen messzire látunk

# A játék pillanatnyi állapota (minden körben frissül)
class State(NamedTuple):
    circuit: Circuit             # A pálya adatai
    visible_track: np.ndarray    # Az a kis térképrészlet, amit éppen látunk
    players: list[Player]        # A többi játékos listája
    agent: Player                # A mi saját botunk adatai

# --- TÉRKÉP (MEMÓRIA) ---
# Ez az osztály felelős azért, hogy megjegyezzük a pályát.
class GlobalMap:
    def __init__(self, shape):
        self.shape = shape  # Eltároljuk a pálya méretét (H, W)
        # Létrehozunk egy akkora rácsot, mint a pálya, és feltöltjük UNKNOWN (-2) értékkel.
        # Kezdetben a bot számára a világ sötét.
        self.grid = np.full(shape, CellType.UNKNOWN.value, dtype=int)

    def update(self, visible_track: np.ndarray):
        # Ez a függvény frissíti a térképet az új látvánnyal.
        # Megkeressük azokat a pontokat, amik NEM "NOT_VISIBLE" (3-as) típusúak.
        mask = (visible_track != CellType.NOT_VISIBLE.value)
        # Csak a látható részeket másoljuk be a memóriánkba, felülírva a régit.
        self.grid[mask] = visible_track[mask]

    def get_cost(self, x, y):
        # Ez a függvény mondja meg az A*-nak, mennyire "drága" rálépni egy mezőre.
        val = self.grid[x, y]
        if val == CellType.WALL.value:
            return float('inf') # Falra lépni lehetetlen (végtelen költség)
        if val == CellType.UNKNOWN.value:
            return 1  # Az ismeretlent szeretjük (olcsó), oda akarunk menni felfedezni!
        if val == CellType.SAND.value or val == CellType.OIL.value:
            return 20 # A homok és olaj drága (lassít/csúszik), kerüljük el, ha lehet.
        return 1      # Sima út (EMPTY, START, GOAL) olcsó, ide szívesen megyünk.

    def is_traversable(self, x, y):
        # Ellenőrzi, hogy a koordináta érvényes-e.
        # 1. Benne van-e a pálya határaiban (0 és H/W között)?
        if not (0 <= x < self.shape[0] and 0 <= y < self.shape[1]):
            return False
        # 2. Nem fal-e az adott mező?
        return self.grid[x, y] != CellType.WALL.value

# --- ALGORITMUSOK ---

def find_nearest_reachable_target(start: tuple[int, int], gmap: GlobalMap, occupied: Set[tuple[int, int]], ignore_players: bool = False) -> Tuple[Optional[tuple[int, int]], str]:
    """
    Flood Fill keresés (BFS - Szélességi keresés).
    Ez a függvény keresi meg a legközelebbi célt (UNKNOWN vagy GOAL).
    Olyan, mintha vizet öntenénk a start mezőre, ami szétfolyik minden irányba.
    """
    # A mód string csak debugolásra lenne jó, itt nem használjuk, de jelzi a logikát.
    mode_str = "LAZA" if ignore_players else "SZIGORÚ"
    
    queue = deque([start]) # Létrehozzuk a sort, betesszük a kezdőpozíciót.
    visited = {start}      # Nyilvántartjuk, hol jártunk már, hogy ne körözzünk.
    
    # Ellenőrizzük, hogy ahol állunk, az már cél-e?
    val_start = gmap.grid[start]
    # Ha ismeretlenben állunk (és nem foglalt vagy ignoráljuk a foglaltságot), akkor megtaláltuk.
    if val_start == CellType.UNKNOWN.value and (ignore_players or start not in occupied):
        return start, 'UNKNOWN'
    # Ha a célban állunk, nyertünk.
    if val_start == CellType.GOAL.value:
        return start, 'GOAL'

    nodes = 0
    while queue:            # Amíg van vizsgálandó mező a sorban...
        cx, cy = queue.popleft() # Kivesszük a következőt.
        nodes += 1
        
        # JAVÍTÁS ITT: 8 irányú szomszédság vizsgálata (átlósan is!)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0: continue # Önmagunkat nem vizsgáljuk
                
                nx, ny = cx + dx, cy + dy # A szomszéd koordinátája
                
                # Pályán belül vagyunk?
                if 0 <= nx < gmap.shape[0] and 0 <= ny < gmap.shape[1]:
                    
                    # 1. FAL ELLENŐRZÉS: Ha fal, nem megyünk arra.
                    if gmap.grid[nx, ny] == CellType.WALL.value:
                        continue
                    
                    # Ha már láttuk ezt a mezőt ebben a keresésben, kihagyjuk.
                    if (nx, ny) in visited:
                        continue
                    
                    # 2. JÁTÉKOSOK: Ha nem "ignore_players" módban vagyunk,
                    # akkor a többi játékos pozíciója falnak számít (elkerüljük őket).
                    if not ignore_players and (nx, ny) in occupied:
                        continue

                    # 3. TALÁLAT ELLENŐRZÉS
                    cell_val = gmap.grid[nx, ny]
                    
                    # Ha ISMERETLENT találtunk, visszaadjuk a koordinátát.
                    if cell_val == CellType.UNKNOWN.value:
                        return (nx, ny), 'UNKNOWN'
                    
                    # Ha CÉLT találtunk, visszaadjuk.
                    if cell_val == CellType.GOAL.value:
                        return (nx, ny), 'GOAL'

                    # Ha nem találtunk semmit, de az út járható, betesszük a sorba,
                    # hogy innen folytassuk a keresést a következő körben.
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    
    # Ha kiürült a sor és nem találtunk semmit:
    return None, 'NONE'

def run_astar(start: tuple[int, int], goal: tuple[int, int], gmap: GlobalMap, occupied: Set[tuple[int, int]]):
    """
    A* (A-csillag) Útvonaltervező.
    Ha a Flood Fill megmondta, HOVA menjünk (goal), ez megmondja, HOGYAN (path).
    """
    # Prioritási sor: (költség, koordináta). Mindig a legolcsóbb utat vesszük előre.
    pq = [(0, start)]
    came_from = {start: None} # Ez tárolja, honnan léptünk ide (az útvonal visszafejtéséhez).
    cost_so_far = {start: 0}  # Mennyibe került eljutni ide a starttól.
    
    while pq:
        _, current = heapq.heappop(pq) # Kivesszük a legkisebb költségű elemet.
        
        if current == goal: # Ha elértük a célt, kész vagyunk.
            break
        
        # Szomszédok vizsgálata (8 irány)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0: continue
                
                nx, ny = current[0] + dx, current[1] + dy
                
                # Ha nem járható (fal vagy pályán kívül), kihagyjuk.
                if not gmap.is_traversable(nx, ny):
                    continue
                
                # Lekérjük a mező költségét (pl. homok=20, üres=1)
                tile_cost = gmap.get_cost(nx, ny)
                
                # Ha foglalt a mező (másik játékos), büntetést adunk hozzá (+500).
                # Így az A* megpróbálja elkerülni, de ha nincs más út, átmegy rajta.
                if (nx, ny) in occupied and (nx, ny) != goal:
                    tile_cost += 500 

                new_cost = cost_so_far[current] + tile_cost
                
                # Ha ez egy új út, vagy olcsóbb, mint amit eddig ismertünk:
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    # Heurisztika: Manhattan távolság a célig. Ez húzza a keresést a cél felé.
                    priority = new_cost + (abs(goal[0] - nx) + abs(goal[1] - ny))
                    heapq.heappush(pq, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current # Elmentjük, hogy innen jöttünk.
                    
    # Ha nem találtunk utat a célhoz:
    if goal not in came_from:
        return []
        
    # Útvonal rekonstruálása visszafelé (Céltól a Startig)
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    path.reverse() # Megfordítjuk, hogy Start -> Cél legyen.
    return path

def get_random_valid_move(state: State, gmap: GlobalMap) -> Tuple[int, int]:
    """
    Pánik mód: Ha beszorultunk vagy nincs cél, próbálunk véletlenszerűen lépni,
    hogy ne álljunk egy helyben.
    """
    possible_moves = []
    current_vel = state.agent.vel
    current_pos = state.agent.pos
    
    # Kipróbáljuk mind a 9 lehetséges gyorsítást (-1, 0, 1)
    for ax in range(-1, 2):
        for ay in range(-1, 2):
            # Kiszámoljuk, hova érkeznénk ezzel a gyorsítással
            next_pos_x = current_pos[0] + current_vel[0] + ax
            next_pos_y = current_pos[1] + current_vel[1] + ay
            
            # Ha az eredmény nem fal, akkor ez egy lehetséges lépés
            if gmap.is_traversable(int(next_pos_x), int(next_pos_y)):
                if ax != 0 or ay != 0:
                     possible_moves.append((ax, ay)) # Preferáljuk a mozgást
                else:
                     possible_moves.append((ax, ay))

    # Ha van lehetséges lépés, választunk egyet véletlenszerűen
    if possible_moves:
        idx = np.random.randint(0, len(possible_moves))
        return possible_moves[idx]
    return (0, 0) # Ha teljesen be vagyunk falazva, marad a (0,0)

# --- FŐ LOGIKA ---

def calculate_move_logic(state: State, gmap: GlobalMap) -> Tuple[int, int]:
    """
    Ez a "főagy", ami minden körben lefut és eldönti, mit lépjünk.
    """
    my_pos = (state.agent.x, state.agent.y)
    
    # 1. Térkép frissítése: Amit most látunk, beírjuk a memóriába.
    gmap.update(state.visible_track)
    
    # Összegyűjtjük a többi játékos pozícióját, hogy elkerülhessük őket.
    occupied_cells = set()
    for p in state.players:
        if (p.x, p.y) != my_pos:
            occupied_cells.add((p.x, p.y))

    # 2. Célkeresés
    # Először "Szigorú" módban próbálkozunk: a többi játékos falnak számít.
    target, t_type = find_nearest_reachable_target(my_pos, gmap, occupied_cells, ignore_players=False)
    
    # Ha így nem találunk célt (pl. valaki elállja az ajtót),
    # átváltunk "Laza" módra: átnézünk a játékosokon.
    if target is None:
        target, t_type = find_nearest_reachable_target(my_pos, gmap, occupied_cells, ignore_players=True)

    # 3. Útvonaltervezés
    path = []
    if target is not None:
        # Ha van cél, megtervezzük oda az utat az A*-gal.
        path = run_astar(my_pos, target, gmap, occupied_cells)
    
    # 4. Fizikai számítás (Gyorsítás/Kormányzás)
    if path and len(path) >= 2:
        # A path[0] a jelenlegi helyünk, a path[1] a következő mező, ahova lépni kell.
        next_cell = path[1]
        
        # PD szabályzó (Proportional-Derivative controller) logikája:
        desired_pos = np.array(next_cell) # Ahova menni akarunk
        current_pos = state.agent.pos     # Ahol vagyunk
        current_vel = state.agent.vel     # Amilyen gyorsan megyünk most
        
        # Ahhoz, hogy a következő körben ott legyünk, ennyivel kéne mennünk:
        needed_vel = desired_pos - current_pos
        # A szükséges gyorsítás = (Kívánt sebesség) - (Jelenlegi sebesség)
        acceleration = needed_vel - current_vel
        
        # A gyorsítást korlátozzuk -1 és 1 közé (ez a játék szabálya).
        ax = int(np.clip(acceleration[0], -1, 1))
        ay = int(np.clip(acceleration[1], -1, 1))
        return (ax, ay)
    else:
        # Ha nincs cél vagy nincs út, Pánik módba kapcsolunk, hogy ne fagyjunk le.
        return get_random_valid_move(state, gmap) 

# --- INPUT/OUTPUT (Kommunikáció a bíróval) ---

def read_initial_observation() -> Circuit:
    """A játék elején egyszer fut le, beolvassa a pálya méreteit."""
    line = sys.stdin.readline()
    if not line: return None
    try:
        parts = list(map(int, line.split()))
        if len(parts) == 4:
            H, W, num_players, visibility_radius = parts
            return Circuit((H, W), num_players, visibility_radius)
    except ValueError:
        return None
    return None

def read_observation(old_state: State) -> Optional[State]:
    """Minden kör elején lefut, beolvassa az aktuális állapotot."""
    line = sys.stdin.readline()
    if not line or line.strip() == '~~~END~~~': # Ha a bíró azt mondja vége
        return None
    
    try:
        # Beolvassuk a saját adatainkat
        posx, posy, velx, vely = map(int, line.split())
        agent = Player(posx, posy, velx, vely)
        
        # Beolvassuk a többi játékost
        players = []
        circuit_data = old_state.circuit
        for _ in range(circuit_data.num_players):
            line_p = sys.stdin.readline()
            if not line_p: break
            pposx, pposy = map(int, line_p.split())
            players.append(Player(pposx, pposy, 0, 0))
        
        # Beolvassuk a látható pályarészletet
        visible_track = np.full(circuit_data.track_shape, CellType.NOT_VISIBLE.value)
        for i in range(2 * circuit_data.visibility_radius + 1):
            raw = sys.stdin.readline()
            if not raw: break
            vals = [int(a) for a in raw.split()]
            
            # Koordináta transzformáció (lokálisból globálisba)
            x = posx - circuit_data.visibility_radius + i
            if x < 0 or x >= circuit_data.track_shape[0]: continue
            
            # Vágás, ha a látómező kilógna a pályáról
            y_start = posy - circuit_data.visibility_radius
            y_end = y_start + 2 * circuit_data.visibility_radius + 1
            
            sl_start, sl_end = 0, len(vals)
            tgt_y_start, tgt_y_end = max(0, y_start), min(circuit_data.track_shape[1], y_end)
            
            if y_start < 0: sl_start = -y_start
            if y_end > circuit_data.track_shape[1]: sl_end -= (y_end - circuit_data.track_shape[1])
                
            if sl_start < sl_end:
                visible_track[x, tgt_y_start:tgt_y_end] = vals[sl_start:sl_end]

        return old_state._replace(visible_track=visible_track, players=players, agent=agent)
    except ValueError:
        return None

def main():
    # Kézfogás a szerverrel: jelezzük, hogy készen állunk.
    print('READY', flush=True)
    
    # Pálya adatainak beolvasása
    circuit = read_initial_observation()
    if circuit is None: return
        
    # Létrehozzuk a globális térképet (memóriát)
    gmap = GlobalMap(circuit.track_shape)
    # Kezdőállapot létrehozása
    state: Optional[State] = State(circuit, None, [], None) # type: ignore
    
    # Végtelen ciklus: amíg tart a verseny
    while True:
        # Beolvassuk az új állapotot
        state = read_observation(state)
        if state is None: break # Ha vége a játéknak, kilépünk
        
        # Kiszámoljuk a lépést (dx, dy gyorsulás)
        dx, dy = calculate_move_logic(state, gmap)
        
        # Elküldjük a választ a bírónak
        print(f'{dx} {dy}', flush=True)

if __name__ == "__main__":
    main()