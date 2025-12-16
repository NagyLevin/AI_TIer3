# Autonomous Inteligent Agent Navigation Project by Szabó Levente

## Describe the problem to be solved.

The objective of this project is to develop an autonomous agent capable of navigating a grid-based environment. The core challenge is that the agent follows a discrete physics model with inertia: the current position is determined by the velocity vector, and the agent can only control acceleration:

($\Delta v \in \{-1, 0, 1\}$ in $x$ and $y$ directions)

The complexity of the problem evolved through three distinct tiers:

### Tier 1 (Full Information): The agent receives the entire map at the start. The goal is to plan an optimal path from Start to Goal, accounting for the agent's inertia (it cannot stop instantly).

### Tier 2 (Partial Observability): A "Fog of War" mechanic is introduced. The agent only sees a limited radius around itself. The problem shifts from pure pathfinding to Exploration. The agent must find the goal with limoted visibility.

### Tier 3 (Environmental Hazards): The map includes special terrain types that alter physics (based on the provided specification):

- Oil: Friction is lost. The acceleration is chosen uniformly randomly from the set $\{-1, 0, 1\}^2$, meaning the agent loses control over its movement vector.
- Sand: High friction. The acceleration is forced to a value that decelerates the agent (chosen from the subset of accelerations that result in the position closest to the current one).

## Are you building on prior work? If yes, cite all sources properly.

Yes. The solution builds on standard search and pathfinding algorithms, plus general ideas from robotics navigation.

### Core Algorithms

- **A\* Search Algorithm** – used for global pathfinding on the grid.
  - Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*.

- **Breadth-First Search (BFS)** – used for exploration: the agent searches for the nearest `UNKNOWN` cell in the global map and then plans a path toward it.

### Implementation Notes

- Exploration is implemented as a **BFS-based search for the nearest unknown cell** on the internally built map, which is conceptually similar to moving towards the frontier between known and unknown areas, but is **not a full frontier-based exploration implementation**.

- Local control is implemented as a **brute-force search over all 9 possible accelerations** `(-1, 0, 1)^2`, selecting the one that minimizes a heuristic cost (distance to target cell, speed error, and acceleration magnitude). This plays a similar role to a local path-tracking controller, but is **not a direct implementation of pure pursuit**.

### Example Resources / Inspirations

During development, I looked at various example implementations and discussions, such as:

- YouTube – maze solving / pathfinding strategies:  
  - https://www.youtube.com/watch?v=gaGaPpnexxA
  - https://www.youtube.com/watch?v=YhC8i2rVLuA
- StackOverflow – labyrinth solver in Python:  
  - https://stackoverflow.com/questions/66805673/labyrinth-solver-on-python-language
- GitHub – A\* maze solver in Python:  
  - https://github.com/aleksibovellan/python-a-star-maze-solver

These were used as examples and inspiration for understanding BFS and A\*-based pathfinding; the final integration with inertia and the global/local control structure is my own implementation.

### State-Space Representation

#### Environment: Occupancy Grid (+ Cell Types)

I represented the world as a **2D occupancy grid** (NumPy array) of integer tile codes.

- In the early Tier 1 version this is simply:
  - `circuit.track: np.ndarray` – the full static map, where each cell encodes wall / empty / goal 
- In the later Tier 2 / Tier 3 versions this became:
  - `GlobalMap.grid` or `WorldModel.known_map` – a global map of size `(H, W)` which is gradually filled from observations.
  - Special cell codes are used:
    - `WALL`, `EMPTY`, `START`, `GOAL`
    - `UNKNOWN` (never seen yet)
    - `NOT_VISIBLE` (currently not in the vision cone)
    - And in Tier 3: `OIL`, `SAND` as hazard cell types.

This is still a classic **occupancy grid**, just with richer semantics. All higher-level logic (A\*, BFS, frontier detection, hazard-aware costs) uses this grid.

**Why this is a good fit:**

**Why this is a good fit**

The crucial property of this task is that the environment is a **fixed, finite grid** of size `H × W`, and the **goal is guaranteed to be somewhere inside this bounding box**.  
Because of this, it is natural to represent the world as a global occupancy grid (`GlobalMap.grid` / `WorldModel.known_map`) that has exactly these dimensions.

With this representation:

For me the heureka moment while solving this task was that this is a different situation from classic “lost robot” problems on an unbounded plane, where the robot and the goal could be anywhere and you would need belief-state search or probabilistic exploration.  
Here the task is more constrained: *the goal cannot escape the `H × W` box*(the agent gets the size from the judge every time before any other steps). Eventually visits all reachable cells (BFS over `GlobalMap.grid` / `WorldModel.known_map`), 
- Therefore the search space is **discrete and finite**: there are only `H × W` cells to discover.
- If the agent systematically expand cells with BFS and always chase remaining `UNKNOWN` areas, the search is **complete**:  
  as long as the goal is reachable, exploring the whole grid will eventually **guarantee** that the agent finds it.
- Under fog-of-war, the agent simply **fills the grid over time** from local observations. The underlying state space does not change – we are just gradually turning `UNKNOWN` cells into concrete types (EMPTY, WALL, GOAL, OIL, SAND).



---

#### Agent State: Position + Velocity

The agent itself is represented as a small structured state:

- In all versions inherited from the example code:
  - `Player(x, y, vel_x, vel_y)`


#### Planning State Across Versions

On top of the raw grid + player state, each tier adds its own **planning state space**:

##### Tier 1 – Full Map, Position-Based Planning

In the very first Tier 1 implementation:

- The environment state is just `Circuit(track, num_players)`.
- The **planning state** for A\* is **2D positions**:
  - `astar(track, players, start_xy, goals)` works in plain `(x, y)` space.
- Inertia is handled “after” planning:
  - The path is a sequence of cells.
  - `calculate_move` / `calculate_move_logic` or `control_to_cell` then computes an acceleration `(ax, ay)` that follows that path as closely as possible, given current `(vx, vy)`.

So the planner sees a **2D grid**, while the controller sees the full **4D physics state** `(x, y, vx, vy)`.

#### Tier 2 – Partial Observability with a Global World Model

In Tier 2 the main change is **partial observability** (fog of war).  
The environment is still a fixed `H × W` grid, but the agent only sees a local window each turn.

I therefore introduced a persistent **world model**:

- **Global map of the environment**  
  - `WorldModel.known_map: np.ndarray` is a `H × W` occupancy grid.
  - Cell types (via `CellType`) include:
    - `WALL`, `EMPTY`, `START`, `GOAL`
    - `UNKNOWN` – never observed so far
    - `NOT_VISIBLE` – used only in the local view, not stored permanently
  - Each turn, `WorldModel.updateWithObservation(state)` merges `state.visible_raw` into `known_map`, replacing `UNKNOWN` with the newly seen tiles.

- **Agent state**  
  - The agent is still a `Player(x, y, vel_x, vel_y)`, i.e. state = `(x, y, vx, vy)`.
  - This reflects the discrete inertial dynamics: `v_next = v + Δv`, `p_next = p + v_next`.

On top of `known_map` I added extra *state layers* to guide exploration and avoid oscillations:

- `visited_count[x, y]` – how often each cell was visited,
- `edge_visits[(x, y, nx, ny)]` – how often each directed edge was used,
- `backtrail` – a short history of recent positions,
- `tried_exits`, `branch_stack` – storing which directions have been tried at junctions,
- `commit_target` – a temporary subgoal (frontier-like cell or known goal) that the agent “locks onto” for several steps.

Planning in Tier 2 uses the same grid representation but in two different state spaces:

1. **CoarsePlanner2D** – 2D A* on positions `(x, y)` over `known_map`  
   - Gives a coarse path on the grid.
   - Uses different costs for `EMPTY`, `UNKNOWN`, and `GOAL`, plus backtrail penalties.

2. **AStarPlanner** – 4D A* on `(x, y, vx, vy)`  
   - State = `(x, y, vx, vy)`, actions = accelerations `(ax, ay) ∈ {-1, 0, 1}²`.
   - Uses `validLineOnMap` to ensure not cut through walls.
   - Includes penalties for braking, sharp turns, backtracking and repeated edges.

The high-level **driver** (`pure_pursuit_move` + `calculateMove`) then:

- Chooses or updates a `commit_target`,
- Plans a coarse path to it in `(x, y)`,
- Picks the furthest *visible* waypoint on that path,
- Computes a target speed based on curvature and distance,
- And finally evaluates all 9 accelerations to pick the best one in the current `(x, y, vx, vy)` state.
---

#### Tier 3 – Same Grid Representation, Extended with Hazard Tiles

Tier 3 keeps the same fundamental **grid + inertial agent** representation, but extends the map with **hazard tiles**.

The core representation is again a **2D grid** of size `H × W`:

- `GlobalMap.grid: np.ndarray` is an occupancy grid over the whole track.
- Cell types (via `CellType`) now include:
  - `WALL`, `EMPTY`, `START`, `GOAL`
  - `UNKNOWN` for unexplored cells (`-2` in the code)
  - `NOT_VISIBLE` for cells outside the current vision
  - plus **hazards**:
    - `OIL = 91`
    - `SAND = 92`
- Each turn, `gmap.update(visible_track)` merges the newly observed window into `GlobalMap.grid` in exactly the same spirit as Tier 2: keep a **global map** and fill in more and more cells over time.

The **agent state** is still `(x, y, vx, vy)` via the `Player` tuple, identical to Tier 2.

Exploration and planning in Tier 3 reuse the same pattern as in Tier 2, but in a lighter form:

- **Exploration: BFS on the global grid**  
  - `find_nearest_unknown(start_xy, gmap, avoid_hazards)` runs a BFS on `GlobalMap.grid` to the closest `UNKNOWN` cell.
  - This leverages the fact that the goal is guaranteed to lie within the finite `H × W` grid: if we keep targeting unknown cells, BFS-based exploration will eventually cover the whole reachable map and thus *guarantee* that we find the goal.

- **Path planning: A* on the global grid**  
  - `run_astar(start_xy, target_xy, gmap, avoid_hazards)` performs 2D A* in position space.
  - I can:
    - either completely **avoid** hazard cells when possible (`avoid_hazards=True`),
    - or allow them with a higher step cost (hazards cost 20 vs normal cells cost 1).

### Would any other representation be viable?

Yes, there are several alternative representations possible in theory, but for this task they are less practical than the occupancy grid I chose to use in the end.

1. **Continuous Geometric Map (polygons / exact geometry)**  
   In this approach, obstacles and free space would be represented by continuous shapes (polygons, line segments, etc.).  
   For our problem this would be overkill and computationally expensive, and harder to program.
   

2. **Graph of Waypoints / Roadmap**  
   Another option would be to compress the grid into a sparse graph:
   - Nodes = key points (intersections, corners, waypoints)  
   - Edges = traversable corridors between them  
   This can work well on a **fully known, static** map, because the graph is small and path planning becomes fast.  
   However, in Tier 2 and Tier 3 the map is **partially observable**:
   - New cells are discovered every turn as the agent moves.
   - Obstacles and free space appear gradually instead of being known upfront.  
   Maintaining and constantly updating a sparse graph in such a setting (adding/removing nodes and edges, rewiring when an unexplored region opens up) would add a lot of bookkeeping and complexity.  
   For this project, keeping a simple **grid map** and running BFS/A* directly on it is both simpler and more robust.

3. **Visit-count / “heat map” overlay on top of the grid**  
   A third idea (which I also experimented with conceptually) is to store, for each cell, how many times it has been visited:
   - Each cell holds an integer counter.
   - Unvisited cells start at `0`, each visit increases the counter by `+1`.
   - Walls (or permanently forbidden cells) can be marked with a special value (e.g. a very large cost or a separate flag).
   A simple policy could then be: **prefer moves toward cells with smaller visit counts**, so the agent is biased towards unexplored or rarely visited areas.  
   This is essentially a scalar field defined on top of the same `H × W` grid.

   Used **alone** as “always go to the smallest counter”, this representation would have drawbacks:
   - It can easily get stuck in local minima or oscillate between a few cells.
   - It ignores the actual **goal** once it becomes visible (the agent keeps chasing “unvisited” instead of “goal”).  
   
Overall, other representations are possible, but they either:
- add significant complexity (polygons, dynamic waypoint graphs), or  
- reduce to “extra layers” on the same grid (visit-count / heat map).  

### Are there uncertainties in the problem?

Yes, the level of uncertainty varies by Tier:

- **Tier 1 – Deterministic Environment**  
  The environment is deterministic. The full map is known and the physics are fully predictable.
  The only kind of uncertainty i can think of in this version are the other agents that can move unexpectedly.

- **Tier 2 – Spatial Uncertainty**  
  The map layout (walls, goal location) is unknown until observed due to the fog-of-war mechanic. The agent gradually discovers the environment as it moves.

- **Tier 3 – Physical Uncertainty**

  - **Stochastic Physics (Oil)**  
    Movement on oil is probabilistic; the agent cannot deterministically predict its next velocity vector because the acceleration is chosen randomly on oil tiles.

  - **Adversarial / Unknown Agents**  
    The movement of opponent agents is unknown. For planning within a single frame the agent treats them as static obstacles at their current positions, but their future positions are uncertain.

### What form of input and output is expected? Are these realistic for a real-world, physical application? If not, is the transformation feasible?

**Input:**  
The agent receives a stream of text containing its kinematic state $(x, y, v_x, v_y)$ and a local grid view of the surroundings.

**Realism:**  
This is highly realistic at an abstract level. Autonomous mobile robots like vacuum robots, self-driving cars often use LIDAR or cameras to build a local Occupancy Grid, which is exactly the data structure that the `read_observation` function processes.

However, real robotic systems have additional complications that are not modeled here, such as:

- sensor noise and partial occlusions,  
- wheel slip and friction effects,  
- accumulated odometry drift and calibration errors,  
- actuation limits, delays and controller imperfections.

So this project is a good **high-level abstraction** of the perception–planning interface (grid map + kinematic state), but it is not a complete drop-in model for a physical robot without further layers of estimation and control.

**Output:**  
The agent outputs an acceleration vector $\Delta v \in \{-1, 0, 1\}^2$.

**Realism:**  
This corresponds to a simplified "Drive-by-Wire" digital control interface. Real actuators (throttle, steering, motor torque) are continuous, but high-level planners are often implemented as discrete updates to a reference velocity or steering command. A low-level controller would then translate these discrete accelerations into continuous motor commands while handling friction, wheelbase geometry, and other physical constraints.

**Feasibility:**  
If the input were raw sensor data like point clouds, a SLAM (Simultaneous Localization and Mapping) layer could be inserted to transform that data into the grid format used by our solver. This kind of perception-to-grid mapping is standard in robotics. Additional localization and control layers would be required to handle real-world effects such as slippage, drift and hardware imperfections, but the transformation to the state representation used in this project is feasible and well-studied.

### What algorithms did you consider, and which ones did you include in your submission? How do they compare in terms of performance?

#### Tier 1 (Considered & Included): Standard A*

For Tier 1 I needed a global planner on a fully known, static grid.

- **Also considered:**  
  We briefly considered simpler alternatives such as:
  - plain BFS / Dijkstra (unweighted / uniform step cost),  
  - greedy best-first search (heuristic only, no g-cost).  

  However, A* is the de facto standard for grid-based pathfinding because it combines:
  - optimality (with an admissible heuristic),
  - very good performance in practice,
  - and simple implementation.  

  For this type of problem shortest path on a known grid, A* is just the best solution, so I deliberately stayed with A* instead of reinventing a weaker or slower variant.

#### Tier 2 (Considered): Kinodynamic A* (4D)

In Tier 2 I experimented with planning directly in the dynamic state space, including inertia:

- **Considered (prototype only):** Kinodynamic A* in 4D  
  I tried searching the state space $(x, y, v_x, v_y)$:
  - Nodes include both position and velocity.
  - Edges correspond to applying one of the 9 accelerations $(\Delta v_x, \Delta v_y) \in \{-1, 0, 1\}^2$, followed by the discrete physics update.

#### Tier 3 (Included): Hybrid Architecture

For the final solution I choose a **hybrid architecture** that separates **Global Planning** from **Local Control**. This keeps the global planner in 2D (fast) while still handling inertia at the local level.

- **BFS (Breadth-First Search) – Exploration**  
  Used when the goal is not yet visible. BFS runs on the global grid and finds the nearest `UNKNOWN` tile.  
  This systematically explores the bounded `H × W` world and guarantees that the agent will eventually discover the goal if it is reachable.

- **Weighted A\* – Global Planning**  
  Used for 2D path planning on the grid once a target (goal or exploration target) is known.  
  The planner assigns:
  - low cost to normal traversable cells,
  - high cost to hazard tiles (Sand, Oil)  
  so it naturally prefers safe routes and only uses hazardous tiles when unavoidable.

- **Local Controller (Brute Force over 9 Moves)**  
  Replaces the slow 4D kinodynamic A* as the main dynamic planner.  
  For each step it:
  - takes the next cell on the global A* path,
  - simulates all 9 possible accelerations for one time step,
  - filters out unsafe moves (collisions, clearly bad transitions),
  - and chooses the move that minimizes the distance to the next path node while keeping a reasonable speed and smooth acceleration.

**Performance:**  
This hybrid approach is:

- **significantly faster** (practically orders of magnitude) than the full 4D A* search, because:
  - global planning happens in 2D,
  - the expensive dynamic reasoning is limited to a 9-move local search each step;
- robust in the presence of hazards (Sand / Oil) due to the weighted costs in A* and the local safety checks in the controller.

In summary, I considered:
- **A\*** (kept, and used heavily),
- and finally converged on a **BFS + Weighted A\* + local brute-force controller** hybrid, which gives a good balance between optimality, robustness and runtime.

### Did you implement any AI algorithm by yourself? If yes, what did you learn during the implementation work?

Yes. All core algorithms A*, BFS, and the World Model were implemented manually in Python without using external pathfinding libraries.

**Key learnings:**

- **Heuristic sensitivity (Tier 1):**  
  In Tier 1 I experimented with different heuristics for A* (Manhattan vs. Euclidean distance) and saw directly how the choice of heuristic affects both the number of expanded nodes and overall runtime on a grid. Even small changes in the heuristic can make A* noticeably faster or slower.

- **State space explosion (Tier 2):**  
  Implementing a 4D planner over $(x, y, v_x, v_y)$ made it very clear how quickly the search space blows up when velocity is added to the graph. The branching factor and extra dimensions led to timeouts and memory issues. This experience motivated the Tier 3 architectural decision to decouple **Planning** (2D grid A*) from **Control** (local physics via a small 9-action search).

- **Exploration logic and visited sets (Tier 2 / Tier 3):**  
  Implementing the BFS-based exploration towards `UNKNOWN` cells (frontier-like behaviour) showed how important it is to manage `visited` sets correctly. Without strict bookkeeping, the agent can easily end up in infinite loops or repeatedly traverse the same cul-de-sac in an unknown map. Careful handling of visited nodes, frontier cells, and backtracking was crucial for stable exploration.

### Have you used heuristics or heuristic functions? If yes, what, why, how?

Yes – in all three tiers I rely quite heavily on heuristics.  
Below is a compact, “report-style” summary, in the same spirit as your example.

#### Tier 1 - Global Pathfinding (A*)
- **Tier 1:** Euclidean A* + simple straight-line speed heuristic.  
**What**  
Standard A* on the grid, with an **Euclidean distance** heuristic:

- `h(n) = ||n - goal||₂ = sqrt((x_goal - x_n)² + (y_goal - y_n)²)`  
- Movement is 8-directional (including diagonals), so this matches the move model.

**Why**  
We need a **global route** from the agent to a goal cell on a fully known track.  
With 8-direction moves, Euclidean distance is a natural lower bound on the remaining path length → good guidance and still reasonably admissible.

**How**  
In `astar(...)` we keep `g_score` (cost-so-far) and `f_score = g + h`.  
Each expansion picks the node with minimal `f_score`, where `h` is the Euclidean distance to the **closest goal**.  
This makes A* expand nodes that are both cheap so far and geometrically close to a goal.


#### Tier 2 – World Model

- **Tier 2:** rich world model with:  
  - cell costs, frontier & info-gain heuristics,  
  - braking-distance safety,  
  - 4D A* with turn/backtrack/visit penalties,  
  - pure-pursuit-style controller with a multi-term score.

**What**  

- `known_map` with explicit `UNKNOWN` cells.  
- Cell costs in coarse planner:  
  - empty / goal ≈ cheap,  
  - unknown = more expensive but allowed,  
  - walls = blocked.  
- Extra penalties for:  
  - recently visited cells (backtrail),  
  - repeatedly used edges,  
  - often visited nodes.  
- Frontier detection:  
  - frontier = traversable cell having at least one `UNKNOWN` neighbour,  
  - `_info_gain` counts unknown cells in a 3×3 window.

**Why**  
Avoid endless loops and back-and-forth movement.  
Prefer exploring **new** areas when no goal is visible.  
Prefer frontier cells that reveal **more** unknown tiles.

**How**  
Coarse 2D A* uses these cell costs and a Euclidean heuristic.  
The subgoal policy chooses:

- known GOAL (nearest),  
- otherwise, frontier cells reachable from current position,  
- ranked by distance, turn angle and information gain.


#### Tier 3 – Global A* 
- **Tier 3:** BFS + weighted A* for global decisions

**What**  

- `find_nearest_unknown`: BFS to the nearest `UNKNOWN` tile (exploration target).  
- `run_astar`:  
  - Manhattan heuristic `|gx - x| + |gy - y|`,  
  - step cost:  
    - 1 on normal cells,  
    - 20 on hazards (Oil/Sand).

**Why**  
BFS provides a simple way to find the closest unexplored region.  
Weighted A* strongly prefers safe routes but can still use hazards when unavoidable.

**How**  

- If a GOAL is visible → A* to GOAL.  
- Otherwise → BFS to nearest unknown → A* to that.  
- If “safe A*” fails, retry A* with hazards allowed.

### Evaluate the performance of the proposed solution, and provide a description of its limitations, and how it could be improved

#### Performance

The Tier 3 solution is **robust and computationally efficient**:

- **Weighted A\*** on the global grid:
  - strongly prefers "asphalt",
  - treats Sand and Oil as high-cost cells, so it **naturally avoids** them unless there is no alternative,
  - this indirectly reduces situations where the physics engine forces random sliding or heavy deceleration, but at the same time if the only way to the goal is blocked by oil or sand than it still choses that way.
- **Separation of concerns**:
  - **Global Planning** (A\* / BFS) runs in pure 2D, so it stays very fast,
  - **Local Control** only brute-forces 9 acceleration options per step, which is effectively O(1) per tick.

---

#### Limitations

1. **Short-sighted physics (myopic Local Controller)**  
   - The Local Controller only simulates **one step ahead** in terms of physics.  
   - It does not reason explicitly about multi-step braking; it just matches speed to the *current* distance to the next waypoint.  
   - On a long straight followed by a sharp corner, it can accelerate too much and:
     - realise too late that a wall is approaching,
     - fail to brake early enough,
     - and end up crashing because it didn’t plan braking distance 2–3 steps in advance.

2. **Reactive agent handling (no trajectory prediction)**  
   - Other agents are treated as **static obstacles** at their current positions.  
   - The agent does not model where enemies will be at `t+1, t+2, ...`, so:
     - it may plan a safe-looking path that becomes blocked by a moving agents a turn later,
     - there is no anticipation this spot will be occupied, better avoid it now.

3. **Conservative acceleration in the final solution**  
   - In the final Tier 3 bot I deliberately **did not push for the most aggressive acceleration profile**.  
   - Instead of tuning for maximum lap speed, I chose a **more robust, conservative controller**:
     - prioritising stability, safe braking, and not getting stuck,
     - even if that means losing some raw speed on “ideal” tracks.
   - The reason is practical: the more complex Tier 2 bot with fancy 4D planning and richer heuristics **did not perform reliably on the hidden grading tracks**, so for the final submission I preferred:
     - fewer hard-to-tune parameters,
     - simpler behaviour that generalises better,
     - and a lower risk of catastrophic failure on unseen maps.

---

#### Possible improvements

1. **Lookahead control (multi-step search in acceleration space)**  
   - Extend the Local Controller from 1-step to a **3–4 step lookahead**:
     - using a small-depth DFS/beam search, or
     - a shallow minimax-style search over `(ax, ay)` sequences.
   - This would allow explicit modelling of **braking distance** and “can I still stop before that wall?” decisions, rather than relying on a one-step heuristic.

2. **Trajectory prediction for other agents**  
   - Add a simple **linear prediction** for opponents:
     - assume they keep their current velocity,
     - predict their positions at `t+1` (or `t+1..t+2`),
     - mark those predicted cells as temporarily forbidden or high cost in the planner.
   - This would turn agent handling from purely reactive (“oops, they’re in front of me now”) into **anticipatory avoidance** (choosing a lane or timing that avoids collisions altogether).

3. **Smarter speed policy (tighter risk–speed tradeoff)**  
   - With lookahead braking logic in place, the controller could safely use **more aggressive acceleration** on straights:
     - higher target speeds when braking distance is sufficient,
     - automatic early braking when approaching tight corners or unknown areas.
   - This would partly recover the raw speed of the Tier 2 ideas, but with **explicit safety checks** instead of purely heuristic trust.
4. **Physics-aware A* (state = position + velocity)**  
   - I also tried an alternative planner that searches in **kinematic state space** instead of just grid cells:
     - node = `(x, y, vx, vy)`,
     - action = `(ax, ay)` where `ax, ay ∈ {-1, 0, 1}`,
     - transition = `v' = v + a`, `p' = p + v'`.
   - Why its a good solution?
     - it enables **fast straight-line driving** (speed > 1) without the one-cell at a time logic that i currenty use
     - it can plan acceleration and braking
   - Key safety detail when speed > 1:
     - each move checks between old and new position (Bresenham-ish walk),
     - prevents me from “wall clipping” / skipping through obstacles just because the endpoint is free.
   - Practical add-ons that made it usable:
     - **lookahead target selection** (aim a few cells ahead on the A* path so speed-up is actually worth it),
     - **speed caps** (try higher caps first, fall back to lower caps if it becomes unsafe/blocked),
     - treat other players as **blocked goal/blocked landing cells** for the kinematic planner.

   - Status / why it didn’t make it into the final submission:
     - This method **works** and is probably **stronger / faster** than what I submitted.
     - However, I opted for a more **stable and well-tested** version for the project delivery.
     - The submitted solution is likely **safer** less moving parts, fewer edge cases 
       even if it leaves some speed on the table.
   - Conclusion:
     - Kept it as a strong candidate for future improvement: promising performance upside,
       but needs more testing / tuning to be used on all maps.
---

In summary: the final solution trades some potential speed for **robustness and predictability**, which was intentional given the hidden test tracks. With multi-step lookahead and basic opponent prediction, it could be pushed towards a faster, more “risky” exploring style without losing that robustness.
