"""
Informed and Uninformed Search on a Maze
BFS, DFS, UCS, A* and GBFS

Conventions (used for every algorithm, and stated in the report):
  * Moves are tried in the order Up, Down, Left, Right.
  * Entering an open cell costs 1; entering a mud cell (M) costs 5.
  * The goal node counts as an expanded node.
  * Max frontier = the largest number of entries held in the queue/stack
    (DFS keeps duplicate entries, so its stack can be larger).
  * Heuristic = Manhattan distance to the goal.

Run:  python search_maze.py
Only the standard library is needed (matplotlib is optional, for one graph).
"""
import heapq
import random
import time
from collections import deque

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]      # Up, Down, Left, Right
MUD_COST = 5

# ----------------------------------------------------------------------
# 1. Maze
# ----------------------------------------------------------------------
# S = start, G = goal, # = wall, M = mud (cost 5), . = open (cost 1)
TEST_MAZE = [
    "S..#",
    "#M..",
    "....",
    "#.MG",
]


class Maze:
    def __init__(self, rows):
        self.rows = rows
        self.R, self.C = len(rows), len(rows[0])
        self.start = self.goal = None
        for r in range(self.R):
            for c in range(self.C):
                if rows[r][c] == "S":
                    self.start = (r, c)
                elif rows[r][c] == "G":
                    self.goal = (r, c)

    def cost(self, cell):
        return MUD_COST if self.rows[cell[0]][cell[1]] == "M" else 1

    def neighbours(self, cell):
        for dr, dc in MOVES:
            r, c = cell[0] + dr, cell[1] + dc
            if 0 <= r < self.R and 0 <= c < self.C and self.rows[r][c] != "#":
                yield (r, c)

    def manhattan(self, cell):
        return abs(cell[0] - self.goal[0]) + abs(cell[1] - self.goal[1])

    def path_cost(self, path):
        return sum(self.cost(p) for p in path[1:])


def random_maze(n, first_seed=1):
    """n x n maze: 20% walls, 10% mud. Seeds are tried (first_seed, +1, ...)
    until the goal can be reached, so the same maze is always produced."""
    seed = first_seed
    while True:
        rng = random.Random(seed)
        rows = []
        for r in range(n):
            row = ""
            for c in range(n):
                if (r, c) == (0, 0):
                    row += "S"
                elif (r, c) == (n - 1, n - 1):
                    row += "G"
                else:
                    x = rng.random()
                    row += "#" if x < 0.20 else ("M" if x < 0.30 else ".")
            rows.append(row)
        maze = Maze(rows)
        if bfs(maze) is not None:
            return maze, seed
        seed += 1


# ----------------------------------------------------------------------
# 2. Search algorithms
#    Each returns (path, expansion_order, max_frontier) or None.
# ----------------------------------------------------------------------
def rebuild(parent, node):
    path = [node]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


def bfs(maze):
    """FIFO queue: expands the oldest node first."""
    frontier = deque([maze.start])
    parent = {maze.start: None}
    order, max_frontier = [], 1
    while frontier:
        node = frontier.popleft()
        order.append(node)
        if node == maze.goal:
            return rebuild(parent, node), order, max_frontier
        for nxt in maze.neighbours(node):
            if nxt not in parent:
                parent[nxt] = node
                frontier.append(nxt)
        max_frontier = max(max_frontier, len(frontier))
    return None


def dfs(maze):
    """LIFO stack: expands the newest node first.
    Neighbours are pushed in reverse so that Up is popped first."""
    frontier = [maze.start]
    parent = {maze.start: None}
    expanded, order, max_frontier = set(), [], 1
    while frontier:
        node = frontier.pop()
        if node in expanded:
            continue
        expanded.add(node)
        order.append(node)
        if node == maze.goal:
            return rebuild(parent, node), order, max_frontier
        for nxt in reversed(list(maze.neighbours(node))):
            if nxt not in expanded:
                parent[nxt] = node
                frontier.append(nxt)
        max_frontier = max(max_frontier, len(frontier))
    return None


def best_first(maze, priority):
    """Priority queue. priority(g, node) decides which node is expanded next.
    Ties go to the entry that was added first."""
    counter = 0
    frontier = [(priority(0, maze.start), 0, maze.start)]
    parent = {maze.start: None}
    best_g = {maze.start: 0}
    expanded, order, max_frontier = set(), [], 1
    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node in expanded:
            continue
        expanded.add(node)
        order.append(node)
        if node == maze.goal:
            return rebuild(parent, node), order, max_frontier
        g = best_g[node]
        for nxt in maze.neighbours(node):
            new_g = g + maze.cost(nxt)
            if nxt not in best_g or new_g < best_g[nxt]:
                best_g[nxt] = new_g
                parent[nxt] = node
                counter += 1
                heapq.heappush(frontier, (priority(new_g, nxt), counter, nxt))
        max_frontier = max(max_frontier, len(frontier))
    return None


def ucs(maze):
    return best_first(maze, lambda g, n: g)                      # f = g


def astar(maze, h=None):
    h = h or maze.manhattan
    return best_first(maze, lambda g, n: g + h(n))               # f = g + h


def gbfs(maze, h=None):
    h = h or maze.manhattan
    return best_first(maze, lambda g, n: h(n))                   # f = h


# ----------------------------------------------------------------------
# 3. Measuring
# ----------------------------------------------------------------------
def time_ms(fn, budget=0.05, batches=5):
    """Average time of one run in ms (best of several batches)."""
    best = float("inf")
    for _ in range(batches):
        t0, runs = time.perf_counter(), 0
        while time.perf_counter() - t0 < budget:
            fn()
            runs += 1
        best = min(best, (time.perf_counter() - t0) / runs * 1000)
    return best


def run(maze, fn, with_time=True):
    path, order, max_frontier = fn(maze)
    return {
        "nodes": len(order),
        "frontier": max_frontier,
        "steps": len(path) - 1,
        "cost": maze.path_cost(path),
        "ms": time_ms(lambda: fn(maze)) if with_time else None,
        "path": path,
        "order": order,
    }


# ----------------------------------------------------------------------
# 4. Independent checks (separate code, does not use the algorithms above)
# ----------------------------------------------------------------------
def optimal_cost(maze):
    """Plain Dijkstra, written separately, to confirm the optimal cost."""
    INF = float("inf")
    dist = [[INF] * maze.C for _ in range(maze.R)]
    dist[0][0] = 0
    heap = [(0, 0, 0)]
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r][c]:
            continue
        for nr, nc in maze.neighbours((r, c)):
            nd = d + maze.cost((nr, nc))
            if nd < dist[nr][nc]:
                dist[nr][nc] = nd
                heapq.heappush(heap, (nd, nr, nc))
    return dist[maze.goal[0]][maze.goal[1]]


def brute_force_cost(maze):
    """Tries every simple path (only practical for the small 4x4 maze)."""
    best = [float("inf")]

    def go(cell, seen, cost):
        if cost >= best[0]:
            return
        if cell == maze.goal:
            best[0] = cost
            return
        for nxt in maze.neighbours(cell):
            if nxt not in seen:
                go(nxt, seen | {nxt}, cost + maze.cost(nxt))

    go(maze.start, {maze.start}, 0)
    return best[0]


def valid_path(maze, path):
    if path[0] != maze.start or path[-1] != maze.goal:
        return False
    for a, b in zip(path, path[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
        if maze.rows[b[0]][b[1]] == "#":
            return False
    return True


def check(label, condition):
    print(f"  [{'OK' if condition else 'FAIL'}] {label}")
    return condition


# ----------------------------------------------------------------------
# 5. Printing helpers for the 4x4 maze (letters for the hand-drawn figures)
# ----------------------------------------------------------------------
def letter_names(maze):
    """S and G keep their names; other open cells get A, B, C... (G skipped)."""
    letters = iter("ABCDEFHIJKLMNOPQRTUVWXYZ")
    names = {}
    for r in range(maze.R):
        for c in range(maze.C):
            ch = maze.rows[r][c]
            if ch == "#":
                continue
            names[(r, c)] = ch if ch in "SG" else next(letters)
    return names


def print_small_maze_results(maze):
    names = letter_names(maze)
    print("Maze (letters):")
    for r in range(maze.R):
        print("  " + " ".join(f"{names.get((r, c), '#'):>2}" for c in range(maze.C)))
    print("Mud cells:", ", ".join(names[(r, c)] for r in range(maze.R)
                                  for c in range(maze.C) if maze.rows[r][c] == "M"))
    print()
    optimum = optimal_cost(maze)
    algos = [("BFS", bfs), ("DFS", dfs), ("UCS", ucs), ("A*", astar), ("GBFS", gbfs)]
    results = {}
    for name, fn in algos:
        res = run(maze, fn)
        results[name] = res
        print(f"--- {name} ---")
        print("  Path          :", " -> ".join(names[p] for p in res["path"]))
        print("  Expansion order:", ", ".join(f"{i}:{names[p]}" for i, p in
                                              enumerate(res["order"], 1)))
        number = {p: i for i, p in enumerate(res["order"], 1)}
        print("  Expansion grid :")
        for r in range(maze.R):
            cells = []
            for c in range(maze.C):
                if (r, c) not in names:
                    cells.append("   #   ")
                else:
                    cells.append(f"{names[(r, c)]}={number.get((r, c), '-')}".center(7))
            print("    " + "".join(cells))
        print(f"  Nodes expanded : {res['nodes']}")
        print(f"  Max frontier   : {res['frontier']}")
        print(f"  Path steps/cost: {res['steps']} / {res['cost']}")
        print(f"  Time           : {res['ms']:.3f} ms")
        print(f"  Optimal?       : {'Yes' if res['cost'] == optimum else 'No'}\n")
    return results, optimum


def print_comparison(results, optimum):
    print("Comparison table (4x4 maze)")
    print(f"{'Algorithm':<10}{'Nodes':>7}{'MaxFrontier':>13}{'Time(ms)':>10}"
          f"{'Cost':>6}  Optimal?")
    for name, r in results.items():
        print(f"{name:<10}{r['nodes']:>7}{r['frontier']:>13}{r['ms']:>10.3f}"
              f"{r['cost']:>6}  {'Yes' if r['cost'] == optimum else 'No'}")
    print()


# ----------------------------------------------------------------------
# 6. Experiments
# ----------------------------------------------------------------------
def size_experiment(sizes=(10, 25, 50)):
    algos = [("BFS", bfs), ("DFS", dfs), ("UCS", ucs), ("A*", astar), ("GBFS", gbfs)]
    table = {name: {} for name, _ in algos}
    optimum = {}
    for n in sizes:
        maze, seed = random_maze(n)
        optimum[n] = optimal_cost(maze)
        print(f"{n}x{n} maze (seed {seed}): {sum(r.count('#') for r in maze.rows)} walls, "
              f"{sum(r.count('M') for r in maze.rows)} mud cells, optimal cost {optimum[n]}")
        for name, fn in algos:
            table[name][n] = run(maze, fn)
    print("\nNodes expanded (path cost)")
    print(f"{'Algorithm':<10}" + "".join(f"{str(n) + 'x' + str(n):>16}" for n in sizes))
    for name, _ in algos:
        print(f"{name:<10}" + "".join(
            f"{str(table[name][n]['nodes']) + ' (' + str(table[name][n]['cost']) + ')':>16}"
            for n in sizes))
    print("\nTime (ms)")
    print(f"{'Algorithm':<10}" + "".join(f"{str(n) + 'x' + str(n):>16}" for n in sizes))
    for name, _ in algos:
        print(f"{name:<10}" + "".join(f"{table[name][n]['ms']:>16.3f}" for n in sizes))
    print("\nMax frontier size")
    print(f"{'Algorithm':<10}" + "".join(f"{str(n) + 'x' + str(n):>16}" for n in sizes))
    for name, _ in algos:
        print(f"{name:<10}" + "".join(f"{table[name][n]['frontier']:>16}" for n in sizes))
    print()
    return table, optimum


def heuristic_experiment(sizes=(10, 25, 50)):
    print("Heuristic experiment: nodes expanded / path cost  (time in ms for A*)")
    for n in sizes:
        maze, _ = random_maze(n)
        best = optimal_cost(maze)
        heuristics = [
            ("h (Manhattan)", maze.manhattan),
            ("2h", lambda c, m=maze: 2 * m.manhattan(c)),
            ("3h", lambda c, m=maze: 3 * m.manhattan(c)),
            ("5h", lambda c, m=maze: 5 * m.manhattan(c)),
            ("3*drow + dcol", lambda c, m=maze: 3 * abs(c[0] - m.goal[0])
                                                  + abs(c[1] - m.goal[1])),
        ]
        print(f"\n{n}x{n} maze, optimal cost = {best}")
        print(f"{'Heuristic':<16}{'A* nodes/cost':>16}{'A* ms':>9}{'Optimal?':>10}"
              f"{'GBFS nodes/cost':>18}")
        for label, h in heuristics:
            a = run(maze, lambda m, h=h: astar(m, h))
            g = run(maze, lambda m, h=h: gbfs(m, h), with_time=False)
            print(f"{label:<16}{str(a['nodes']) + ' / ' + str(a['cost']):>16}"
                  f"{a['ms']:>9.3f}{('Yes' if a['cost'] == best else 'No'):>10}"
                  f"{str(g['nodes']) + ' / ' + str(g['cost']):>18}")
    print()


# ----------------------------------------------------------------------
# 7. Verification (uses the independent checks from section 4)
# ----------------------------------------------------------------------
def verification():
    print("Verification checks")
    algos = [("BFS", bfs), ("DFS", dfs), ("UCS", ucs), ("A*", astar), ("GBFS", gbfs)]
    mazes = [("4x4", Maze(TEST_MAZE))]
    for n in (10, 25, 50):
        mazes.append((f"{n}x{n}", random_maze(n)[0]))
    ok = True
    for label, maze in mazes:
        best = optimal_cost(maze)
        res = {name: run(maze, fn, with_time=False) for name, fn in algos}
        print(f" {label}: independent optimal cost = {best}")
        ok &= check("all five paths are valid (start to goal, no walls)",
                    all(valid_path(maze, r["path"]) for r in res.values()))
        ok &= check("UCS cost equals the independent optimal cost",
                    res["UCS"]["cost"] == best)
        ok &= check("A* cost equals the optimal cost (admissible heuristic)",
                    res["A*"]["cost"] == best)
        ok &= check("no algorithm beats the optimal cost",
                    all(r["cost"] >= best for r in res.values()))
        ok &= check("BFS has the fewest steps",
                    all(res["BFS"]["steps"] <= r["steps"] for r in res.values()))
        if label == "4x4":          # brute force is only practical on the small maze
            ok &= check("brute force over all paths gives the same optimal cost",
                        brute_force_cost(maze) == best)
    print("\nAll checks passed.\n" if ok else "\nSOME CHECKS FAILED.\n")


# ----------------------------------------------------------------------
# 8. Optional graph
# ----------------------------------------------------------------------
def plot_nodes(table, sizes=(10, 25, 50)):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed, graph skipped)")
        return
    # Three separate graphs: nodes, execution time and memory (max frontier).
    # Nodes and frontier are counts (same every run); time (ms) varies a little per run.
    graphs = [("nodes", "Nodes expanded", "Nodes expanded vs maze size",
               "nodes_vs_size.png"),
              ("ms", "Execution time (ms)", "Execution time vs maze size",
               "time_vs_size.png"),
              ("frontier", "Max queue/stack size", "Memory (max queue/stack size) vs maze size",
               "memory_vs_size.png")]
    for key, ylabel, title, filename in graphs:
        plt.figure(figsize=(6, 4))
        for name, data in table.items():
            plt.plot(sizes, [data[n][key] for n in sizes], marker="o", label=name)
        plt.xlabel("Maze size (n x n)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(sizes)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Graph saved as {filename}")


if __name__ == "__main__":
    small = Maze(TEST_MAZE)
    print("=" * 60, "\nPART 1: 4x4 test maze\n" + "=" * 60)
    results, optimum = print_small_maze_results(small)
    print_comparison(results, optimum)
    print("=" * 60, "\nPART 2: effect of problem size\n" + "=" * 60)
    table, _ = size_experiment()
    print("=" * 60, "\nPART 3: effect of changing the heuristic\n" + "=" * 60)
    heuristic_experiment()
    print("=" * 60, "\nPART 4: verification\n" + "=" * 60)
    verification()
    plot_nodes(table)
