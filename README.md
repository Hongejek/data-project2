# 实验二：搜索算法

本目录包含实验二四道题的 Python 实现，均只依赖 Python 标准库。

## 文件说明

| 文件 | 对应题目 | 内容 |
|---|---|---|
| `maze_astar.py` | 题目一：迷宫寻路 | 均匀代价 A*，返回最短路径坐标列表 |
| `weighted_astar.py` | 题目二：加权地形 | 带进入代价的 A*，返回最小总代价 |
| `alpha_beta.py` | 题目三：博弈树 | Minimax 与 Alpha-Beta，返回 MAX 最优得分 |
| `treasure_game.py` | 题目四：宝藏分配 | 6×6 距离矩阵 + Minimax/Alpha-Beta 博弈搜索 |
| `run_experiments.py` | 全部题目 | 批量运行对照实验，生成 `results/` 下的 JSON/CSV |

## 运行方式

在实验二目录下执行：

```bash
python maze_astar.py
python weighted_astar.py
python alpha_beta.py
python treasure_game.py
python run_experiments.py
```

每个文件末尾都带有题目给定用例的自检，运行后应看到“测试通过”。

`run_experiments.py` 运行对照实验，并在控制台打印 Markdown 表格。
结果文件保存在 `results/`：

- `experiment_results.json`：全部实验数据
- `maze_results.csv`：题目一 h=0 vs 曼哈顿
- `weighted_results.csv`：题目二 h=0 vs 安全启发
- `weighted_path_breakdown.csv`：题目二给定路径的逐步代价
- `alpha_beta_results.csv`：题目三 Minimax vs Alpha-Beta
- `treasure_results.csv`：题目四不同深度/访问顺序
- `distance_matrix.csv`：题目四 6×6 距离矩阵
- `experiment_tables.md`：控制台 Markdown 表格汇总

## 给定用例的预期结果

- 题目一：最短路径共 7 步，例如  
  `[(0,0), (0,1), (0,2), (1,2), (1,3), (1,4), (2,4), (3,4)]`
- 题目二：最小总代价 `7`
- 题目三：MAX 最优得分 `5`
- 题目四：
  - `H = 2`：`{"best_target": "T2", "best_value": -1}`
  - `H = 4`：`{"best_target": "T3", "best_value": 3}`

## 实现要点

### 题目一

- 状态就是格子坐标，`g` 为步数，`h` 为曼哈顿距离。
- 用优先队列取 `f = g + h` 最小的节点；`came_from` 回溯路径。
- 无解返回空列表。

### 题目二

- 进入格子的代价等于该格地形值；`S` 按 1 处理，`E` 的进入代价为 0，`X` 不可通行。
- 因为 `E` 的进入代价为 0，普通曼哈顿距离可能高估 1。例如 `S-1-E` 的曼哈顿距离是 2，但实际代价是 1。
- 因此使用可采纳启发：

  ```text
  h(n) = max(0, Manhattan(n, E) - 1) * min_cost
  ```

  其中 `min_cost` 是可通行格子的最小代价；若存在 0 代价格，`h = 0`，算法退化为 Dijkstra，仍然正确。
- 返回最小总代价；无解返回 `-1`。

### 题目三

- `TreeNode` 的叶子节点有 `value`，内部节点有 `children`。
- 根节点按 MAX 层处理，递归时 MAX/MIN 交替。
- `minimax` 是不剪枝的基线；`alpha_beta` 在 MAX 层更新 `alpha`、MIN 层更新 `beta`，当 `alpha >= beta` 时剪枝。
- 两个函数都支持传入 `stats` 字典，记录访问节点数、叶子评估数和剪枝数。

### 题目四

- 先把 `A`、`B`、`T1`~`T4` 归一化为代价 1，复用题目二的 A* 计算 6×6 距离矩阵。
- 博弈状态包含：A 位置、B 位置、剩余宝藏、A 累计收益、B 累计收益、当前玩家。
- 收益 `Gain(p, Ti) = Vi - d(p, Ti)`，评价函数 `U = Score_A - Score_B`。
- A 最大化 U，B 最小化 U；搜索到深度 H 或宝藏取完时返回当前 U。
- 距离矩阵只算一次，博弈树中直接查表，避免重复跑 A*。
