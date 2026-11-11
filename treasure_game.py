"""
题目四：竞争环境下的宝藏分配（综合题）
================================================
A（MAX）与 B（MIN）轮流争夺四个宝藏：
- 行动顺序：A -> B -> A -> B，共四回合
- 每回合选择一个尚未获得的宝藏，沿最小代价路径移动到该位置
- 经过其他宝藏不会自动收集；已获得宝藏的位置仍可通行
- 另一名玩家不阻挡寻路

收益：
    Gain(p, Ti) = Vi - d(p, Ti)
    U = Score_A - Score_B

A 选择使 U 最大的策略，B 选择使 U 最小的策略。
搜索到深度 H 或宝藏取完时，以当前 U 作为评价值。

本文件完成：
1. 用 A* 计算 A、B、T1~T4 两两之间的最小移动代价，得到 6×6 矩阵
2. 建立博弈状态，分别用 Minimax 和 Alpha-Beta 求 H=2、H=4 时 A 的最优首步
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

from weighted_astar import a_star_weighted_path


# ============================================================
# 1. 地图与关键点
# ============================================================

POINT_NAMES: Tuple[str, ...] = ("A", "B", "T1", "T2", "T3", "T4")

POINT_POS: Dict[str, Tuple[int, int]] = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0),
}

TREASURE_VALUES: Dict[str, int] = {
    "T1": 9,
    "T2": 6,
    "T3": 7,
    "T4": 7,
}

TREASURES: Tuple[str, ...] = ("T1", "T2", "T3", "T4")
POINT_INDEX: Dict[str, int] = {name: i for i, name in enumerate(POINT_NAMES)}


def normalize_grid(grid: List[List[Any]]) -> List[List[Any]]:
    """把 A/B/T 等关键点转成代价 1，数字转成 int，X 保持不变。

    weighted_astar 只认识 S/E/X/数字/'.'，所以先做一次归一化，
    这样题目四的地图可以直接复用题目二的 A*。
    """
    normalized: List[List[Any]] = []
    for row in grid:
        new_row: List[Any] = []
        for cell in row:
            if isinstance(cell, str):
                text = cell.strip()
                if text == "X":
                    new_row.append("X")
                elif text.isdigit():
                    new_row.append(int(text))
                elif text in POINT_POS or text in ("S", "."):
                    new_row.append(1)
                elif text == "E":
                    new_row.append(0)
                else:
                    raise ValueError(f"无法识别的格子内容: {cell!r}")
            elif isinstance(cell, int) and not isinstance(cell, bool):
                if cell < 0:
                    raise ValueError(f"地形代价不能为负数: {cell!r}")
                new_row.append(cell)
            else:
                raise TypeError(f"无法识别的格子内容: {cell!r}")
        normalized.append(new_row)
    return normalized


def build_distance_matrix(grid: List[List[Any]]) -> List[List[int]]:
    """用 A* 计算 6 个关键点两两之间的最小移动代价。

    返回 6×6 矩阵，行列顺序为 A, B, T1, T2, T3, T4。
    不可达的格子填 -1。
    """
    normalized = normalize_grid(grid)
    n = len(POINT_NAMES)
    matrix = [[0] * n for _ in range(n)]

    for i, source in enumerate(POINT_NAMES):
        for j, target in enumerate(POINT_NAMES):
            if i == j:
                continue
            cost, _ = a_star_weighted_path(
                POINT_POS[source], POINT_POS[target], normalized
            )
            matrix[i][j] = cost

    return matrix


def format_distance_matrix(matrix: List[List[int]], width: int = 4) -> str:
    """把 6×6 距离矩阵格式化成便于打印/写报告的表格。"""
    header = " " * width + "".join(f"{name:>{width}}" for name in POINT_NAMES)
    lines = [header]
    for name, row in zip(POINT_NAMES, matrix):
        lines.append(f"{name:>{width}}" + "".join(f"{value:>{width}}" for value in row))
    return "\n".join(lines)


# ============================================================
# 2. 博弈状态与搜索
# ============================================================

@dataclass(frozen=True)
class GameState:
    """一个不可变的博弈状态。"""

    pos_a: str                 # A 当前所在关键点
    pos_b: str                 # B 当前所在关键点
    remaining: Tuple[str, ...] # 尚未获得的宝藏
    score_a: int               # A 的累计收益
    score_b: int               # B 的累计收益
    turn: str                  # "A" 或 "B"


def _new_stats(stats: Dict[str, int]) -> None:
    stats.clear()
    stats.update(
        {
            "visited": 0,
            "leaf_evals": 0,
            "prunes": 0,
            "pruned_children": 0,
        }
    )


def _search(
    distance_matrix: List[List[int]],
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    use_alpha_beta: bool,
    stats: Dict[str, int],
) -> Tuple[int, Optional[str]]:
    """递归搜索，返回 (当前状态的 U 值, 当前玩家的最优目标)。"""
    stats["visited"] += 1

    # 到达搜索深度，或者宝藏已经取完
    if depth == 0 or not state.remaining:
        stats["leaf_evals"] += 1
        return state.score_a - state.score_b, None

    maximizing = state.turn == "A"
    best_value = -math.inf if maximizing else math.inf
    best_target: Optional[str] = None

    for index, target in enumerate(state.remaining):
        current_pos = state.pos_a if maximizing else state.pos_b
        distance = distance_matrix[POINT_INDEX[current_pos]][POINT_INDEX[target]]
        if distance < 0:
            continue

        gain = TREASURE_VALUES[target] - distance
        new_remaining = state.remaining[:index] + state.remaining[index + 1 :]

        if maximizing:
            new_state = GameState(
                pos_a=target,
                pos_b=state.pos_b,
                remaining=new_remaining,
                score_a=state.score_a + gain,
                score_b=state.score_b,
                turn="B",
            )
        else:
            new_state = GameState(
                pos_a=state.pos_a,
                pos_b=target,
                remaining=new_remaining,
                score_a=state.score_a,
                score_b=state.score_b + gain,
                turn="A",
            )

        child_value, _ = _search(
            distance_matrix,
            new_state,
            depth - 1,
            alpha,
            beta,
            use_alpha_beta,
            stats,
        )

        if maximizing:
            if child_value > best_value:
                best_value = child_value
                best_target = target
            if use_alpha_beta:
                alpha = max(alpha, best_value)
        else:
            if child_value < best_value:
                best_value = child_value
                best_target = target
            if use_alpha_beta:
                beta = min(beta, best_value)

        if use_alpha_beta and alpha >= beta:
            remaining_count = len(state.remaining) - index - 1
            if remaining_count > 0:
                stats["prunes"] += 1
                stats["pruned_children"] += remaining_count
            break

    # 所有动作都不可达：按当前局面评估，避免返回 inf
    if best_target is None:
        stats["leaf_evals"] += 1
        return state.score_a - state.score_b, None

    return best_value, best_target


def solve_treasure(
    distance_matrix: List[List[int]],
    depth: int,
    use_alpha_beta: bool = True,
    stats: Optional[Dict[str, int]] = None,
    treasure_order: Tuple[str, ...] = TREASURES,
) -> Dict[str, Any]:
    """在给定距离矩阵上求解 A 的最优首步。

    Args:
        distance_matrix: 6×6 距离矩阵
        depth:          搜索深度 H（2 或 4）
        use_alpha_beta: True 用 Alpha-Beta，False 用 Minimax 基线
        stats:          可选统计字典
        treasure_order: 宝藏枚举顺序，默认 T1~T4；可用于测试访问顺序影响

    Returns:
        {"best_target": "T2", "best_value": -1} 形式的字典
    """
    if stats is None:
        stats = {}
    _new_stats(stats)

    initial = GameState(
        pos_a="A",
        pos_b="B",
        remaining=treasure_order,
        score_a=0,
        score_b=0,
        turn="A",
    )

    value, target = _search(
        distance_matrix,
        initial,
        depth,
        -math.inf,
        math.inf,
        use_alpha_beta,
        stats,
    )
    return {"best_target": target, "best_value": value}


def solve_from_grid(
    grid: List[List[Any]],
    depth: int,
    use_alpha_beta: bool = True,
    stats: Optional[Dict[str, int]] = None,
    treasure_order: Tuple[str, ...] = TREASURES,
) -> Dict[str, Any]:
    """便捷入口：先在 grid 上算距离矩阵，再求解。"""
    return solve_treasure(
        build_distance_matrix(grid),
        depth,
        use_alpha_beta,
        stats,
        treasure_order,
    )


# ============================================================
# 3. 测试
# ============================================================

def _run_tests() -> None:
    grid = [
        ["A", "1", "1", "X", "2", "1", "T1"],
        ["1", "X", "6", "X", "1", "X", "1"],
        ["1", "1", "T2", "1", "1", "1", "1"],
        ["X", "1", "X", "8", "X", "1", "X"],
        ["1", "1", "1", "1", "T3", "1", "1"],
        ["1", "X", "9", "X", "1", "X", "1"],
        ["T4", "1", "1", "X", "1", "1", "B"],
    ]

    matrix = build_distance_matrix(grid)
    print("6×6 距离矩阵:")
    print(format_distance_matrix(matrix))

    # 关键点代价都为 1，矩阵应当对称
    for i in range(len(POINT_NAMES)):
        assert matrix[i][i] == 0
        for j in range(len(POINT_NAMES)):
            assert matrix[i][j] == matrix[j][i], (i, j, matrix[i][j], matrix[j][i])

    expected = {
        2: {"best_target": "T2", "best_value": -1},
        4: {"best_target": "T3", "best_value": 3},
    }

    for depth in (2, 4):
        minimax_stats: Dict[str, int] = {}
        alpha_stats: Dict[str, int] = {}

        minimax_result = solve_treasure(matrix, depth, False, minimax_stats)
        alpha_result = solve_treasure(matrix, depth, True, alpha_stats)

        print(f"H={depth} Minimax   :", minimax_result, minimax_stats)
        print(f"H={depth} Alpha-Beta:", alpha_result, alpha_stats)

        assert minimax_result == expected[depth], minimax_result
        assert alpha_result == expected[depth], alpha_result
        assert alpha_stats["visited"] <= minimax_stats["visited"]

    print("题目四测试通过")


if __name__ == "__main__":
    _run_tests()
