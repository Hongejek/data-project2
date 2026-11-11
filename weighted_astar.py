"""
题目二：加权地形（A* 算法）
================================================
地图中的数字表示“进入该格”的代价：
- 'S' : 起点，代价按 1 处理
- 'E' : 终点，进入代价为 0
- 'X' : 障碍物，不可通行
- 数字 : 该地形进入代价，例如 1 平原、3 沼泽、8/9 高代价地形

输出从起点到终点的最小总代价；无解返回 -1。

启发函数说明
------------
因为 E 的进入代价是 0，普通曼哈顿距离可能高估 1。
例如 S-1-E：曼哈顿距离为 2，但实际代价只有 1。
因此这里使用可采纳的启发：

    h(n) = max(0, Manhattan(n, E) - 1) * min_cost

其中 min_cost 是可通行格子的最小非负代价。若地图中存在 0 代价格，
min_cost = 0，此时 h = 0，算法退化为 Dijkstra，仍然正确。
"""

from typing import Any, Dict, List, Optional, Tuple
import heapq
import math

Coord = Tuple[int, int]

# 上下左右
DIRECTIONS: Tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


# ============================================================
# 1. 基础工具
# ============================================================

def manhattan(a: Coord, b: Coord) -> int:
    """返回两点之间的曼哈顿距离。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def cell_cost(cell: Any) -> Optional[int]:
    """返回进入该格子的代价；障碍物返回 None。

    同时兼容字符串地图（['S', '1', 'E']）和数字地图（['S', 1, 'E']）。
    """
    if isinstance(cell, str):
        text = cell.strip()
        if text == "X":
            return None
        if text == "E":
            return 0
        if text in ("S", "."):
            return 1
        if text.isdigit():
            return int(text)
        raise ValueError(f"无法识别的格子内容: {cell!r}")

    if isinstance(cell, bool):
        raise TypeError(f"无法识别的格子内容: {cell!r}")

    if isinstance(cell, int):
        if cell < 0:
            raise ValueError(f"地形代价不能为负数: {cell!r}")
        return cell

    raise TypeError(f"无法识别的格子内容: {cell!r}")


def _grid_shape(grid: List[List[Any]]) -> Optional[Tuple[int, int]]:
    """检查地图是否为非空矩形，返回 (rows, cols)；不合法返回 None。"""
    if not grid or not grid[0]:
        return None
    cols = len(grid[0])
    if any(len(row) != cols for row in grid):
        return None
    return len(grid), cols


def _min_passable_cost(grid: List[List[Any]], end: Coord) -> int:
    """可通行格子的最小代价，用于构造启发函数。

    终点 E 的代价固定为 0，不参与最小值的计算；如果存在其他 0 代价格，
    最小值就是 0，启发函数自动退化为 0，保证可采纳性。
    """
    costs: List[int] = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if (r, c) == end:
                continue
            cost = cell_cost(cell)
            if cost is not None:
                costs.append(cost)
    return min(costs) if costs else 1


def heuristic_weighted(pos: Coord, end: Coord, min_cost: int) -> int:
    """加权地形下的可采纳启发函数。

    任何一条从 pos 到 end 的路径，至少有 Manhattan(pos, end) 步。
    最后一步进入 E 的代价为 0，其余每一步的代价至少为 min_cost，
    所以真实剩余代价 >= max(0, Manhattan - 1) * min_cost。
    """
    if min_cost < 0:
        raise ValueError("min_cost 不能为负数")
    return max(0, manhattan(pos, end) - 1) * min_cost


def _reconstruct_path(
    came_from: Dict[Coord, Coord], start: Coord, end: Coord
) -> List[Coord]:
    """从 came_from 回溯出 start -> end 的路径。"""
    path = [end]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


# ============================================================
# 2. A* 核心
# ============================================================

def _search(
    start: Coord,
    end: Coord,
    grid: List[List[Any]],
    stats: Optional[Dict[str, int]] = None,
    use_heuristic: bool = True,
) -> Tuple[int, List[Coord]]:
    """A* 主循环，返回 (最小代价, 路径)；无解返回 (-1, [])。"""
    if stats is not None:
        stats.clear()
        stats.update({"expanded": 0, "generated": 0, "max_open": 0})

    shape = _grid_shape(grid)
    if shape is None:
        return -1, []

    rows, cols = shape
    sr, sc = start
    er, ec = end

    if not (0 <= sr < rows and 0 <= sc < cols):
        return -1, []
    if not (0 <= er < rows and 0 <= ec < cols):
        return -1, []

    # 起点或终点本身是障碍物
    if cell_cost(grid[sr][sc]) is None or cell_cost(grid[er][ec]) is None:
        return -1, []

    if start == end:
        return 0, [start]

    min_cost = _min_passable_cost(grid, end)

    # 优先队列元素：(f, g, 坐标)
    start_h = heuristic_weighted(start, end, min_cost) if use_heuristic else 0
    open_heap: List[Tuple[int, int, Coord]] = [(start_h, 0, start)]
    g_score: Dict[Coord, int] = {start: 0}
    came_from: Dict[Coord, Coord] = {}

    while open_heap:
        if stats is not None:
            stats["max_open"] = max(stats["max_open"], len(open_heap))

        _, g, current = heapq.heappop(open_heap)

        # 同一个节点可能被多次压入，跳过旧记录
        if g != g_score.get(current):
            continue

        if stats is not None:
            stats["expanded"] += 1

        if current == end:
            return g, _reconstruct_path(came_from, start, end)

        r, c = current
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            step_cost = cell_cost(grid[nr][nc])
            if step_cost is None:
                continue

            neighbor = (nr, nc)
            tentative_g = g + step_cost
            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                h = (
                    heuristic_weighted(neighbor, end, min_cost)
                    if use_heuristic
                    else 0
                )
                f = tentative_g + h
                heapq.heappush(open_heap, (f, tentative_g, neighbor))
                if stats is not None:
                    stats["generated"] += 1

    return -1, []


def a_star_weighted(
    start: Coord,
    end: Coord,
    grid: List[List[Any]],
    stats: Optional[Dict[str, int]] = None,
    use_heuristic: bool = True,
) -> int:
    """返回从 start 到 end 的最小总代价；无解返回 -1。

    Args:
        start: 起点坐标 (row, col)
        end:   终点坐标 (row, col)
        grid:  二维地图，元素可以是 'S'/'E'/'X'/数字/'.'
        stats: 可选字典，用于收集 expanded/generated/max_open 统计信息
        use_heuristic: False 时 h=0，退化为 Dijkstra，用作对照

    Returns:
        最小总代价，或 -1。
    """
    cost, _ = _search(start, end, grid, stats, use_heuristic)
    return cost


def a_star_weighted_path(
    start: Coord,
    end: Coord,
    grid: List[List[Any]],
    stats: Optional[Dict[str, int]] = None,
    use_heuristic: bool = True,
) -> Tuple[int, List[Coord]]:
    """返回 (最小总代价, 路径)；无解返回 (-1, [])。

    这个版本主要用于实验报告里展示路径和逐步代价。
    """
    return _search(start, end, grid, stats, use_heuristic)


# ============================================================
# 3. 测试
# ============================================================

def _run_tests() -> None:
    """运行题目给定用例和几个边界用例。"""
    grid = [
        ["S", "1", "1", "8", "1"],
        ["1", "8", "3", "1", "1"],
        ["1", "8", "1", "8", "1"],
        ["1", "1", "1", "8", "1"],
        ["8", "8", "1", "1", "E"],
    ]

    # 题目给定用例：预期 7
    assert a_star_weighted((0, 0), (4, 4), grid) == 7

    cost, path = a_star_weighted_path((0, 0), (4, 4), grid)
    assert cost == 7
    assert path[0] == (0, 0) and path[-1] == (4, 4)

    # E 的进入代价为 0：S-1-E 的实际代价是 1，而不是曼哈顿距离 2
    assert a_star_weighted((0, 0), (0, 2), [["S", "1", "E"]]) == 1

    # 无解
    blocked = [
        ["S", ".", "X", "E"],
        [".", "X", "X", "X"],
        [".", ".", ".", "."],
    ]
    assert a_star_weighted((0, 0), (0, 3), blocked) == -1

    # 起点即终点
    assert a_star_weighted((0, 0), (0, 0), [["S"]]) == 0

    # 数字与字符串混用
    int_grid = [
        ["S", 1, 1],
        [1, 9, 1],
        [1, 1, "E"],
    ]
    assert a_star_weighted((0, 0), (2, 2), int_grid) == 3

    # 统计信息
    stats: Dict[str, int] = {}
    assert a_star_weighted((0, 0), (4, 4), grid, stats) == 7
    assert stats["expanded"] > 0 and stats["generated"] > 0

    print("题目二测试通过")
    print("给定用例最小代价:", cost)
    print("给定用例路径:", path)
    print("A* 统计:", stats)


if __name__ == "__main__":
    _run_tests()
