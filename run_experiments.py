"""
实验二数据生成脚本
================================================
运行：

    python run_experiments.py

脚本会完成三类工作：
1. 跑通四道题的给定用例，并做一致性断言；
2. 生成报告需要的对照实验数据（节点数、时间、剪枝数等）；
3. 在 results/ 下输出 JSON、CSV 和控制台 Markdown 表格。

对照实验包括：
- 题目一：曼哈顿启发 vs h=0（Dijkstra）
- 题目二：安全启发 vs h=0
- 题目三：Minimax vs Alpha-Beta，不同子节点顺序
- 题目四：H=2/H=4，不同宝藏访问顺序
"""

from __future__ import annotations

import csv
import json
import platform
import random
import timeit
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import alpha_beta
import maze_astar
import treasure_game
import weighted_astar


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
TABLE_LINES: List[str] = []


# ============================================================
# 1. 通用工具
# ============================================================

def measure_ms(
    func: Callable[[], Any],
    number: int = 200,
    repeat: int = 5,
) -> float:
    """返回函数单次运行的平均耗时（毫秒）。

    先预热一次，再重复 repeat 组、每组 number 次，取最好的一组，
    以减少操作系统调度带来的噪声。节点数等确定性指标仍以 stats 为准。
    """
    func()  # warmup
    timer = timeit.Timer(func)
    times = timer.repeat(repeat=repeat, number=number)
    return min(times) / number * 1000.0


def print_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """打印 Markdown 表格，并收集到 results/experiment_tables.md。"""
    lines = [
        f"\n## {title}",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    text = "\n".join(lines)
    print(text)
    TABLE_LINES.append(text)


def write_csv(filename: str, headers: Sequence[str], rows: Sequence[Dict[str, Any]]) -> Path:
    """把字典列表写成 CSV（utf-8-sig 方便 Excel 直接打开）。"""
    path = RESULTS_DIR / filename
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(header, "") for header in headers])
    return path


def write_json(filename: str, payload: Dict[str, Any]) -> Path:
    path = RESULTS_DIR / filename
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


# ============================================================
# 2. 实验地图
# ============================================================

GIVEN_MAZE: List[List[str]] = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]


def make_wall_maze(size: int = 15) -> List[List[str]]:
    """生成一张带竖墙的迷宫，墙中间留一个缺口。"""
    grid = [["."] * size for _ in range(size)]
    middle = size // 2
    for r in range(size):
        if r != middle:
            grid[r][middle] = "X"
    grid[0][0] = "S"
    grid[size - 1][size - 1] = "E"
    return grid


GIVEN_WEIGHTED: List[List[Any]] = [
    ["S", "1", "1", "8", "1"],
    ["1", "8", "3", "1", "1"],
    ["1", "8", "1", "8", "1"],
    ["1", "1", "1", "8", "1"],
    ["8", "8", "1", "1", "E"],
]


def make_weighted_map(size: int = 12, seed: int = 2026) -> List[List[Any]]:
    """生成一张加权地图，并保留一条从左上到右下的确定通路。

    通路沿第一行再沿最后一列，保证一定可达；其余格子随机放障碍或高代价格。
    """
    rng = random.Random(seed)
    grid: List[List[Any]] = [
        [rng.choice([1, 1, 1, 3, 8]) for _ in range(size)]
        for _ in range(size)
    ]

    # 保留第一行和最后一列作为确定通路
    for c in range(size):
        grid[0][c] = 1
    for r in range(size):
        grid[r][size - 1] = 1

    # 内部随机放障碍
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if rng.random() < 0.22:
                grid[r][c] = "X"

    grid[0][0] = "S"
    grid[size - 1][size - 1] = "E"
    return grid


GIVEN_TREASURE_GRID: List[List[Any]] = [
    ["A", "1", "1", "X", "2", "1", "T1"],
    ["1", "X", "6", "X", "1", "X", "1"],
    ["1", "1", "T2", "1", "1", "1", "1"],
    ["X", "1", "X", "8", "X", "1", "X"],
    ["1", "1", "1", "1", "T3", "1", "1"],
    ["1", "X", "9", "X", "1", "X", "1"],
    ["T4", "1", "1", "X", "1", "1", "B"],
]


# ============================================================
# 3. 题目一：迷宫寻路
# ============================================================

def experiment_maze() -> List[Dict[str, Any]]:
    cases = [
        ("题目给定地图 5×5", GIVEN_MAZE, (0, 0), (3, 4)),
        ("带墙地图 15×15", make_wall_maze(), (0, 0), (14, 14)),
    ]
    rows: List[Dict[str, Any]] = []

    for name, grid, start, end in cases:
        path_with_h = maze_astar.a_star(start, end, grid, use_heuristic=True)
        path_without_h = maze_astar.a_star(start, end, grid, use_heuristic=False)

        assert len(path_with_h) == len(path_without_h), f"{name} 两种启发得到不同路径长度"

        for label, use_heuristic in (("曼哈顿", True), ("h=0 (Dijkstra)", False)):
            stats: Dict[str, int] = {}
            path = maze_astar.a_star(
                start,
                end,
                grid,
                stats=stats,
                use_heuristic=use_heuristic,
            )
            elapsed = measure_ms(
                lambda h=use_heuristic: maze_astar.a_star(
                    start, end, grid, use_heuristic=h
                ),
                number=200,
            )
            rows.append(
                {
                    "map": name,
                    "heuristic": label,
                    "path_len": len(path) - 1 if path else -1,
                    "expanded": stats["expanded"],
                    "generated": stats["generated"],
                    "max_open": stats["max_open"],
                    "time_ms": round(elapsed, 4),
                }
            )

    return rows


# ============================================================
# 4. 题目二：加权地形
# ============================================================

def experiment_weighted() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cases = [
        ("题目给定地图 5×5", GIVEN_WEIGHTED, (0, 0), (4, 4)),
        ("随机加权地图 12×12", make_weighted_map(), (0, 0), (11, 11)),
    ]
    rows: List[Dict[str, Any]] = []
    breakdown: List[Dict[str, Any]] = []

    for name, grid, start, end in cases:
        cost_with_h, path_with_h = weighted_astar.a_star_weighted_path(
            start, end, grid, use_heuristic=True
        )
        cost_without_h, path_without_h = weighted_astar.a_star_weighted_path(
            start, end, grid, use_heuristic=False
        )
        assert cost_with_h == cost_without_h, f"{name} 两种启发得到不同最小代价"

        for label, use_heuristic in (("安全启发", True), ("h=0 (Dijkstra)", False)):
            stats: Dict[str, int] = {}
            cost, path = weighted_astar.a_star_weighted_path(
                start,
                end,
                grid,
                stats=stats,
                use_heuristic=use_heuristic,
            )
            elapsed = measure_ms(
                lambda h=use_heuristic: weighted_astar.a_star_weighted(
                    start, end, grid, use_heuristic=h
                ),
                number=200,
            )
            rows.append(
                {
                    "map": name,
                    "heuristic": label,
                    "cost": cost,
                    "path_len": len(path) - 1 if path else -1,
                    "expanded": stats["expanded"],
                    "generated": stats["generated"],
                    "max_open": stats["max_open"],
                    "time_ms": round(elapsed, 4),
                }
            )

        # 给定用例额外输出路径的逐步代价，报告里可以直接用表
        if name.startswith("题目给定"):
            cumulative = 0
            for step, (r, c) in enumerate(path_with_h):
                enter_cost = 0 if step == 0 else weighted_astar.cell_cost(grid[r][c])
                if enter_cost is None:
                    raise AssertionError("最优路径经过了障碍物")
                cumulative += enter_cost
                breakdown.append(
                    {
                        "step": step,
                        "position": f"({r}, {c})",
                        "cell": grid[r][c],
                        "enter_cost": enter_cost,
                        "cumulative_cost": cumulative,
                    }
                )

    return rows, breakdown


# ============================================================
# 5. 题目三：Minimax / Alpha-Beta
# ============================================================

def build_wide_tree(reverse_children: bool = False) -> alpha_beta.TreeNode:
    """构造一棵剪枝效果明显的宽树：MAX -> 4×MIN -> 4×MAX -> 2 叶。

    第一棵 MIN 子树的值较高，使根节点的 alpha 很快升高；
    后三棵 MIN 子树在第一个 MAX 子节点后就能触发剪枝。
    """
    groups = [
        [[100, 99], [98, 97], [96, 95], [94, 93]],
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        [[9, 10], [11, 12], [13, 14], [15, 16]],
        [[17, 18], [19, 20], [21, 22], [23, 24]],
    ]

    min_nodes: List[alpha_beta.TreeNode] = []
    for group in groups:
        max_nodes = [
            alpha_beta.TreeNode(
                children=[alpha_beta.TreeNode(value=value) for value in pair]
            )
            for pair in group
        ]
        if reverse_children:
            max_nodes.reverse()
        min_nodes.append(alpha_beta.TreeNode(children=max_nodes))

    if reverse_children:
        min_nodes.reverse()
    return alpha_beta.TreeNode(children=min_nodes)


def experiment_alpha_beta() -> List[Dict[str, Any]]:
    cases = [
        ("题目示例树", lambda reverse: alpha_beta.build_example_tree(reverse_children=reverse), 5),
        ("宽树 4×4×2", lambda reverse: build_wide_tree(reverse_children=reverse), 94),
    ]
    rows: List[Dict[str, Any]] = []

    for name, builder, expected_value in cases:
        for reverse in (False, True):
            root = builder(reverse)
            order_label = "从右到左" if reverse else "从左到右"
            values: Dict[str, int] = {}

            for algorithm, use_alpha_beta in (("Minimax", False), ("Alpha-Beta", True)):
                stats: Dict[str, int] = {}
                if use_alpha_beta:
                    value = alpha_beta.alpha_beta(root, stats)
                    elapsed = measure_ms(
                        lambda r=root: alpha_beta.alpha_beta(r),
                        number=1000,
                    )
                else:
                    value = alpha_beta.minimax(root, stats)
                    elapsed = measure_ms(
                        lambda r=root: alpha_beta.minimax(r),
                        number=1000,
                    )

                values[algorithm] = value
                rows.append(
                    {
                        "tree": name,
                        "order": order_label,
                        "algorithm": algorithm,
                        "best_value": value,
                        "visited": stats["visited"],
                        "leaf_evals": stats["leaf_evals"],
                        "prunes": stats.get("prunes", 0),
                        "pruned_children": stats.get("pruned_children", 0),
                        "time_ms": round(elapsed, 4),
                    }
                )

            assert values["Minimax"] == values["Alpha-Beta"], f"{name} 两种算法结果不一致"
            assert values["Minimax"] == expected_value, f"{name} 预期值错误"

    return rows


# ============================================================
# 6. 题目四：宝藏分配
# ============================================================

def experiment_treasure() -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    matrix = treasure_game.build_distance_matrix(GIVEN_TREASURE_GRID)

    orders = [
        ("T1→T2→T3→T4", treasure_game.TREASURES),
        ("T4→T3→T2→T1", tuple(reversed(treasure_game.TREASURES))),
        ("价值优先 T1→T3→T4→T2", ("T1", "T3", "T4", "T2")),
    ]
    expected = {
        2: {"best_target": "T2", "best_value": -1},
        4: {"best_target": "T3", "best_value": 3},
    }
    rows: List[Dict[str, Any]] = []

    for order_name, order in orders:
        for depth in (2, 4):
            values: Dict[str, int] = {}

            for algorithm, use_alpha_beta in (("Minimax", False), ("Alpha-Beta", True)):
                stats: Dict[str, int] = {}
                result = treasure_game.solve_treasure(
                    matrix,
                    depth,
                    use_alpha_beta=use_alpha_beta,
                    stats=stats,
                    treasure_order=order,
                )
                elapsed = measure_ms(
                    lambda d=depth, ab=use_alpha_beta, o=order: treasure_game.solve_treasure(
                        matrix,
                        d,
                        use_alpha_beta=ab,
                        treasure_order=o,
                    ),
                    number=300,
                )

                values[algorithm] = result["best_value"]
                rows.append(
                    {
                        "order": order_name,
                        "depth": depth,
                        "algorithm": algorithm,
                        "best_target": result["best_target"],
                        "best_value": result["best_value"],
                        "visited": stats["visited"],
                        "leaf_evals": stats["leaf_evals"],
                        "prunes": stats.get("prunes", 0),
                        "pruned_children": stats.get("pruned_children", 0),
                        "time_ms": round(elapsed, 4),
                    }
                )

                if order_name == "T1→T2→T3→T4":
                    assert result == expected[depth], f"H={depth} 给定预期不一致: {result}"

            assert values["Minimax"] == values["Alpha-Beta"], (
                f"{order_name}, H={depth} 两种算法结果不一致"
            )

    return matrix, rows


# ============================================================
# 7. 汇总与输出
# ============================================================

def summarize_maze(rows: Sequence[Dict[str, Any]]) -> List[List[Any]]:
    summary: List[List[Any]] = []
    for map_name in dict.fromkeys(row["map"] for row in rows):
        base = next(
            row
            for row in rows
            if row["map"] == map_name and row["heuristic"] == "h=0 (Dijkstra)"
        )
        with_h = next(
            row
            for row in rows
            if row["map"] == map_name and row["heuristic"] == "曼哈顿"
        )
        reduction = (base["expanded"] - with_h["expanded"]) / base["expanded"] * 100
        summary.append(
            [
                map_name,
                base["expanded"],
                with_h["expanded"],
                f"{reduction:.1f}%",
                fmt(base["time_ms"]),
                fmt(with_h["time_ms"]),
            ]
        )
    return summary


def summarize_weighted(rows: Sequence[Dict[str, Any]]) -> List[List[Any]]:
    summary: List[List[Any]] = []
    for map_name in dict.fromkeys(row["map"] for row in rows):
        base = next(
            row
            for row in rows
            if row["map"] == map_name and row["heuristic"] == "h=0 (Dijkstra)"
        )
        with_h = next(
            row
            for row in rows
            if row["map"] == map_name and row["heuristic"] == "安全启发"
        )
        reduction = (base["expanded"] - with_h["expanded"]) / base["expanded"] * 100
        summary.append(
            [
                map_name,
                base["expanded"],
                with_h["expanded"],
                f"{reduction:.1f}%",
                fmt(base["time_ms"]),
                fmt(with_h["time_ms"]),
            ]
        )
    return summary


def summarize_alpha_beta(rows: Sequence[Dict[str, Any]]) -> List[List[Any]]:
    summary: List[List[Any]] = []
    seen = set()
    for row in rows:
        key = (row["tree"], row["order"])
        if key in seen:
            continue
        seen.add(key)
        minimax_row = next(
            item
            for item in rows
            if item["tree"] == key[0]
            and item["order"] == key[1]
            and item["algorithm"] == "Minimax"
        )
        alpha_row = next(
            item
            for item in rows
            if item["tree"] == key[0]
            and item["order"] == key[1]
            and item["algorithm"] == "Alpha-Beta"
        )
        reduction = (
            (minimax_row["visited"] - alpha_row["visited"])
            / minimax_row["visited"]
            * 100
        )
        summary.append(
            [
                key[0],
                key[1],
                minimax_row["visited"],
                alpha_row["visited"],
                f"{reduction:.1f}%",
                alpha_row["prunes"],
                alpha_row["pruned_children"],
            ]
        )
    return summary


def summarize_treasure(rows: Sequence[Dict[str, Any]]) -> List[List[Any]]:
    summary: List[List[Any]] = []
    seen = set()
    for row in rows:
        key = (row["order"], row["depth"])
        if key in seen:
            continue
        seen.add(key)
        minimax_row = next(
            item
            for item in rows
            if item["order"] == key[0]
            and item["depth"] == key[1]
            and item["algorithm"] == "Minimax"
        )
        alpha_row = next(
            item
            for item in rows
            if item["order"] == key[0]
            and item["depth"] == key[1]
            and item["algorithm"] == "Alpha-Beta"
        )
        reduction = (
            (minimax_row["visited"] - alpha_row["visited"])
            / minimax_row["visited"]
            * 100
        )
        summary.append(
            [
                key[0],
                key[1],
                f"{minimax_row['best_target']} / {minimax_row['best_value']}",
                minimax_row["visited"],
                alpha_row["visited"],
                f"{reduction:.1f}%",
                alpha_row["prunes"],
            ]
        )
    return summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_LINES.clear()

    print("实验二数据生成")
    print("Python:", platform.python_version())

    maze_rows = experiment_maze()
    weighted_rows, weighted_breakdown = experiment_weighted()
    alpha_rows = experiment_alpha_beta()
    distance_matrix, treasure_rows = experiment_treasure()

    # 控制台表格
    print_table(
        "题目一：曼哈顿启发 vs h=0",
        ["地图", "启发", "路径长度", "expanded", "generated", "max_open", "time_ms"],
        [
            [
                row["map"],
                row["heuristic"],
                row["path_len"],
                row["expanded"],
                row["generated"],
                row["max_open"],
                fmt(row["time_ms"]),
            ]
            for row in maze_rows
        ],
    )
    print_table(
        "题目一：节点数对比摘要",
        ["地图", "h=0 expanded", "曼哈顿 expanded", "节点减少", "h=0 time_ms", "曼哈顿 time_ms"],
        summarize_maze(maze_rows),
    )

    print_table(
        "题目二：安全启发 vs h=0",
        ["地图", "启发", "最小代价", "路径长度", "expanded", "generated", "max_open", "time_ms"],
        [
            [
                row["map"],
                row["heuristic"],
                row["cost"],
                row["path_len"],
                row["expanded"],
                row["generated"],
                row["max_open"],
                fmt(row["time_ms"]),
            ]
            for row in weighted_rows
        ],
    )
    print_table(
        "题目二：节点数对比摘要",
        ["地图", "h=0 expanded", "安全启发 expanded", "节点减少", "h=0 time_ms", "安全启发 time_ms"],
        summarize_weighted(weighted_rows),
    )
    print_table(
        "题目二：给定路径的逐步代价",
        ["step", "position", "cell", "enter_cost", "cumulative_cost"],
        [
            [
                row["step"],
                row["position"],
                row["cell"],
                row["enter_cost"],
                row["cumulative_cost"],
            ]
            for row in weighted_breakdown
        ],
    )

    print_table(
        "题目三：Minimax vs Alpha-Beta",
        ["树", "顺序", "算法", "best_value", "visited", "leaf_evals", "prunes", "pruned_children", "time_ms"],
        [
            [
                row["tree"],
                row["order"],
                row["algorithm"],
                row["best_value"],
                row["visited"],
                row["leaf_evals"],
                row["prunes"],
                row["pruned_children"],
                fmt(row["time_ms"]),
            ]
            for row in alpha_rows
        ],
    )
    print_table(
        "题目三：剪枝效果摘要",
        ["树", "顺序", "Minimax visited", "Alpha-Beta visited", "节点减少", "prunes", "pruned_children"],
        summarize_alpha_beta(alpha_rows),
    )

    matrix_lines = [
        "\n## 题目四：6×6 距离矩阵",
        "| point | " + " | ".join(treasure_game.POINT_NAMES) + " |",
        "| --- | " + " | ".join(["---"] * len(treasure_game.POINT_NAMES)) + " |",
    ]
    for name, row in zip(treasure_game.POINT_NAMES, distance_matrix):
        matrix_lines.append("| " + name + " | " + " | ".join(str(value) for value in row) + " |")
    matrix_text = "\n".join(matrix_lines)
    print(matrix_text)
    TABLE_LINES.append(matrix_text)

    print_table(
        "题目四：不同深度与访问顺序",
        ["顺序", "H", "算法", "best_target", "best_value", "visited", "leaf_evals", "prunes", "pruned_children", "time_ms"],
        [
            [
                row["order"],
                row["depth"],
                row["algorithm"],
                row["best_target"],
                row["best_value"],
                row["visited"],
                row["leaf_evals"],
                row["prunes"],
                row["pruned_children"],
                fmt(row["time_ms"]),
            ]
            for row in treasure_rows
        ],
    )
    print_table(
        "题目四：剪枝效果摘要",
        ["顺序", "H", "Minimax 结果", "Minimax visited", "Alpha-Beta visited", "节点减少", "prunes"],
        summarize_treasure(treasure_rows),
    )

    # 写文件
    csv_paths = [
        write_csv(
            "maze_results.csv",
            ["map", "heuristic", "path_len", "expanded", "generated", "max_open", "time_ms"],
            maze_rows,
        ),
        write_csv(
            "weighted_results.csv",
            ["map", "heuristic", "cost", "path_len", "expanded", "generated", "max_open", "time_ms"],
            weighted_rows,
        ),
        write_csv(
            "weighted_path_breakdown.csv",
            ["step", "position", "cell", "enter_cost", "cumulative_cost"],
            weighted_breakdown,
        ),
        write_csv(
            "alpha_beta_results.csv",
            ["tree", "order", "algorithm", "best_value", "visited", "leaf_evals", "prunes", "pruned_children", "time_ms"],
            alpha_rows,
        ),
        write_csv(
            "treasure_results.csv",
            ["order", "depth", "algorithm", "best_target", "best_value", "visited", "leaf_evals", "prunes", "pruned_children", "time_ms"],
            treasure_rows,
        ),
    ]

    # 距离矩阵单独写 CSV
    matrix_path = RESULTS_DIR / "distance_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point", *treasure_game.POINT_NAMES])
        for name, row in zip(treasure_game.POINT_NAMES, distance_matrix):
            writer.writerow([name, *row])

    payload = {
        "meta": {
            "python": platform.python_version(),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "time_note": "time_ms 为多次运行的最小平均值，节点数等确定性指标以 stats 为准",
        },
        "maze": maze_rows,
        "weighted": weighted_rows,
        "weighted_path_breakdown": weighted_breakdown,
        "alpha_beta": alpha_rows,
        "treasure": {
            "points": list(treasure_game.POINT_NAMES),
            "treasure_values": treasure_game.TREASURE_VALUES,
            "distance_matrix": distance_matrix,
            "results": treasure_rows,
        },
    }
    json_path = write_json("experiment_results.json", payload)

    tables_path = RESULTS_DIR / "experiment_tables.md"
    tables_path.write_text(
        "# 实验二数据表\n" + "\n".join(TABLE_LINES),
        encoding="utf-8",
    )

    print("\n结果文件：")
    for path in [*csv_paths, matrix_path, json_path, tables_path]:
        print("-", path)


if __name__ == "__main__":
    main()
