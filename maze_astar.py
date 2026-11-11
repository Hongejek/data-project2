"""
题目一：迷宫寻路（A* 算法）
================================================
网格约定：
- 'S' : 起点
- 'E' : 终点
- '.' : 可通过路径
- 'X' : 不可通过的障碍物

规则：
- 只能上下左右移动
- 每移动一步代价为 1
- 启发函数使用曼哈顿距离

输入：start=(row, col), end=(row, col), grid=List[List[str]]
输出：从起点到终点的最短路径；无解返回 []
"""

from typing import Dict, List, Optional, Tuple
import heapq
import math

Coord = Tuple[int, int]
DIRECTIONS: Tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def heuristic(a: Coord, b: Coord) -> int:
    """曼哈顿距离。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(grid: List[List[str]], node: Coord) -> List[Coord]:
    """返回上下左右四个方向中可通过的邻居。"""
    if not grid or not grid[0]:
        return []

    rows, cols = len(grid), len(grid[0])
    r, c = node
    neighbors: List[Coord] = []

    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "X":
            neighbors.append((nr, nc))

    return neighbors


def validate_input(start: Coord, end: Coord, grid: List[List[str]]) -> bool:
    """检查地图非空、为矩形，且起点和终点在范围内且不是障碍物。"""
    if not grid or not grid[0]:
        return False

    cols = len(grid[0])
    if any(len(row) != cols for row in grid):
        return False

    rows = len(grid)
    for r, c in (start, end):
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        if grid[r][c] == "X":
            return False

    return True


def reconstruct_path(came_from: Dict[Coord, Coord], end: Coord) -> List[Coord]:
    """从终点沿 came_from 回溯路径；起点没有前驱，回溯到它自然停止。"""
    path = [end]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def a_star(
    start: Coord,
    end: Coord,
    grid: List[List[str]],
    stats: Optional[Dict[str, int]] = None,
    use_heuristic: bool = True,
) -> List[Coord]:
    """返回从 start 到 end 的最短路径；无解返回 []。

    Args:
        start: 起点 (row, col)
        end:   终点 (row, col)
        grid:  迷宫地图
        stats: 可选统计字典：expanded / generated / max_open
        use_heuristic: False 时 h=0，退化为 Dijkstra，用作对照
    """
    if stats is not None:
        stats.clear()
        stats.update({"expanded": 0, "generated": 0, "max_open": 0})

    if not validate_input(start, end, grid):
        return []

    if start == end:
        return [start]

    # 优先队列元素：(f, g, 坐标)
    start_h = heuristic(start, end) if use_heuristic else 0
    open_heap: List[Tuple[int, int, Coord]] = [(start_h, 0, start)]
    g_score: Dict[Coord, int] = {start: 0}
    came_from: Dict[Coord, Coord] = {}

    while open_heap:
        if stats is not None:
            stats["max_open"] = max(stats["max_open"], len(open_heap))

        _, g, current = heapq.heappop(open_heap)

        # 同一节点可能被多次压入，跳过旧记录
        if g != g_score.get(current):
            continue

        if stats is not None:
            stats["expanded"] += 1

        if current == end:
            return reconstruct_path(came_from, end)

        for neighbor in get_neighbors(grid, current):
            tentative_g = g + 1
            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                h = heuristic(neighbor, end) if use_heuristic else 0
                f = tentative_g + h
                heapq.heappush(open_heap, (f, tentative_g, neighbor))
                if stats is not None:
                    stats["generated"] += 1

    return []


def _run_test_case() -> None:
    """运行题目给定用例，并检查路径合法性。"""
    start: Coord = (0, 0)
    end: Coord = (3, 4)
    maze: List[List[str]] = [
        ["S", ".", ".", "X", "."],
        [".", "X", ".", ".", "."],
        [".", "X", ".", "X", "."],
        [".", ".", ".", "X", "E"],
        ["X", "X", ".", ".", "."],
    ]

    stats: Dict[str, int] = {}
    path = a_star(start, end, maze, stats)

    assert path, "给定用例应该存在路径"
    assert path[0] == start and path[-1] == end
    assert len(path) - 1 == 7, f"最短步数应为 7，实际 {len(path) - 1}"

    # 检查路径连续且不经过 X
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "路径不连续"
        assert maze[r2][c2] != "X", "路径经过障碍物"

    # 无解用例
    blocked = [
        ["S", "X", "E"],
        ["X", "X", "X"],
        ["X", "X", "X"],
    ]
    assert a_star((0, 0), (0, 2), blocked) == []

    print("题目一测试通过")
    print("最短路径:", path)
    print("最短步数:", len(path) - 1)
    print("A* 统计:", stats)


if __name__ == "__main__":
    _run_test_case()
