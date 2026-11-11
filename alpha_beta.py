"""
题目三：极大极小博弈树的最佳得分（Minimax + Alpha-Beta）
================================================
树节点结构：
- 叶子节点有 value
- 非叶子节点有 children
- 根节点是 MAX 玩家
- 子节点访问顺序为从左到右

本文件同时实现：
1. minimax(root)   —— 不剪枝的基线
2. alpha_beta(root) —— Alpha-Beta 剪枝
3. build_example_tree() —— 题目给定的树

统计信息通过可选的 stats 字典返回，便于实验报告比较节点数和剪枝次数。
"""

from typing import Dict, List, Optional
import math


class TreeNode:
    """博弈树节点。

    Args:
        value:    叶子节点的得分；非叶子节点可为 None
        children: 子节点列表；叶子节点为空列表
    """

    def __init__(
        self,
        value: Optional[int] = None,
        children: Optional[List["TreeNode"]] = None,
    ) -> None:
        self.value = value
        self.children = children if children is not None else []

    def __repr__(self) -> str:
        if not self.children:
            return f"TreeNode(value={self.value})"
        return f"TreeNode(children={len(self.children)})"


def _leaf_value(node: TreeNode) -> int:
    """读取叶子节点的值，并做一次防御性检查。"""
    if node.value is None:
        raise ValueError("叶子节点缺少 value")
    return node.value


# ============================================================
# 1. Minimax（基线，不剪枝）
# ============================================================

def minimax(
    root: TreeNode,
    stats: Optional[Dict[str, int]] = None,
) -> int:
    """返回根节点（MAX）的最优得分。

    Args:
        root:  树的根节点，按 MAX 层处理
        stats: 可选字典，收集 visited / leaf_evals
    """
    if stats is not None:
        stats.clear()
        stats.update({"visited": 0, "leaf_evals": 0})
    return _minimax(root, True, stats)


def _minimax(
    node: TreeNode,
    is_max: bool,
    stats: Optional[Dict[str, int]],
) -> int:
    if stats is not None:
        stats["visited"] += 1

    # 叶子节点
    if not node.children:
        if stats is not None:
            stats["leaf_evals"] += 1
        return _leaf_value(node)

    if is_max:
        best = -math.inf
        for child in node.children:
            value = _minimax(child, False, stats)
            if value > best:
                best = value
        return best

    best = math.inf
    for child in node.children:
        value = _minimax(child, True, stats)
        if value < best:
            best = value
    return best


# ============================================================
# 2. Alpha-Beta 剪枝
# ============================================================

def alpha_beta(
    root: TreeNode,
    stats: Optional[Dict[str, int]] = None,
) -> int:
    """返回根节点（MAX）的最优得分（Alpha-Beta 剪枝版）。

    Args:
        root:  树的根节点，按 MAX 层处理
        stats: 可选字典，收集 visited / leaf_evals / prunes / pruned_children
    """
    if stats is not None:
        stats.clear()
        stats.update(
            {
                "visited": 0,
                "leaf_evals": 0,
                "prunes": 0,
                "pruned_children": 0,
            }
        )
    return _alpha_beta(root, -math.inf, math.inf, True, stats)


def _alpha_beta(
    node: TreeNode,
    alpha: float,
    beta: float,
    is_max: bool,
    stats: Optional[Dict[str, int]],
) -> int:
    if stats is not None:
        stats["visited"] += 1

    # 叶子节点
    if not node.children:
        if stats is not None:
            stats["leaf_evals"] += 1
        return _leaf_value(node)

    if is_max:
        value = -math.inf
        for index, child in enumerate(node.children):
            value = max(value, _alpha_beta(child, alpha, beta, False, stats))
            alpha = max(alpha, value)

            # MAX 层已经能保证 >= alpha，MIN 层不会让它发生，剪枝
            if alpha >= beta:
                remaining = len(node.children) - index - 1
                if stats is not None and remaining > 0:
                    stats["prunes"] += 1
                    stats["pruned_children"] += remaining
                break
        return value

    value = math.inf
    for index, child in enumerate(node.children):
        value = min(value, _alpha_beta(child, alpha, beta, True, stats))
        beta = min(beta, value)

        # MIN 层已经能保证 <= beta，MAX 层不会让它发生，剪枝
        if alpha >= beta:
            remaining = len(node.children) - index - 1
            if stats is not None and remaining > 0:
                stats["prunes"] += 1
                stats["pruned_children"] += remaining
            break
    return value


# ============================================================
# 3. 题目给定的树
# ============================================================

def _ordered(items: List[TreeNode], reverse_children: bool) -> List[TreeNode]:
    """按需返回正序或反序的子节点列表。"""
    return list(reversed(items)) if reverse_children else list(items)


def build_example_tree(reverse_children: bool = False) -> TreeNode:
    r"""构造题目中的示例树。

    结构：
              MAX
            /     \
          MIN     MIN
         /  \     /  \
       MAX    MAX   MAX
       / \    / \   / \
      3  5   6  9   1  2

    """
    # 叶子节点
    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)

    # 中间层 MAX
    max1 = TreeNode(children=_ordered([leaf1, leaf2], reverse_children))
    max2 = TreeNode(children=_ordered([leaf3, leaf4], reverse_children))
    max3 = TreeNode(children=_ordered([leaf5, leaf6], reverse_children))

    # 中间层 MIN
    min1 = TreeNode(children=_ordered([max1, max2], reverse_children))
    min2 = TreeNode(children=_ordered([max2, max3], reverse_children))

    # 根节点 MAX
    return TreeNode(children=_ordered([min1, min2], reverse_children))


# ============================================================
# 4. 测试
# ============================================================

def _run_tests() -> None:
    root = build_example_tree()

    # 题目预期输出：5
    assert minimax(root) == 5
    assert alpha_beta(root) == 5

    min_stats: Dict[str, int] = {}
    ab_stats: Dict[str, int] = {}
    assert minimax(root, min_stats) == 5
    assert alpha_beta(root, ab_stats) == 5

    # Alpha-Beta 只应减少搜索量，不应改变结果
    assert ab_stats["visited"] <= min_stats["visited"]

    # 改变子节点访问顺序，结果仍应为 5
    reversed_root = build_example_tree(reverse_children=True)
    reversed_stats: Dict[str, int] = {}
    assert alpha_beta(reversed_root, reversed_stats) == 5

    print("题目三测试通过")
    print("Minimax 最佳得分:", minimax(root))
    print("Minimax 统计:", min_stats)
    print("Alpha-Beta 最佳得分:", alpha_beta(root))
    print("Alpha-Beta 统计:", ab_stats)
    print("反转子节点顺序后的 Alpha-Beta:", 5, reversed_stats)


if __name__ == "__main__":
    _run_tests()
