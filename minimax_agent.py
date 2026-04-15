"""
第三小问（选做）：Minimax 智能体

实现 Minimax + Alpha-Beta 剪枝算法，与 MCTS 对比效果。
可选实现，用于对比不同搜索算法的差异。

参考：《深度学习与围棋》第 3 章
"""

from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move

__all__ = ["MinimaxAgent"]



class MinimaxAgent:
    """
    Minimax 智能体（带 Alpha-Beta 剪枝）。

    属性：
        max_depth: 搜索最大深度
        evaluator: 局面评估函数
    """

    def __init__(self, max_depth=3, evaluator=None):
        self.max_depth = max_depth
        # 默认评估函数（TODO：学生可替换为神经网络）
        self.evaluator = evaluator or self._default_evaluator
        self.cache = GameResultCache()

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        # TODO: 实现 Minimax 搜索，调用 minimax 或 alphabeta
        best_move = None

        legal_moves = [
            move for move in self._get_ordered_moves(game_state)
            if move.is_play
        ]

        if not legal_moves:
            return Move.pass_turn()
        
        if game_state.next_player == Player.black:
            best_value = -float("inf")
            for move in legal_moves:
                next_state = game_state.apply_move(move)
                value = self.alphabeta(
                    next_state,
                    self.max_depth - 1,
                    -float("inf"),
                    float("inf"),
                    False,
                )
                if value > best_value:
                    best_value = value
                    best_move = move
        else:
            best_value = float("inf")
            for move in legal_moves:
                next_state = game_state.apply_move(move)
                value = self.alphabeta(
                    next_state,
                    self.max_depth - 1,
                    -float("inf"),
                    float("inf"),
                    True,
                )
                if value < best_value:
                    best_value = value
                    best_move = move

        return best_move if best_move is not None else Move.pass_turn()

    def minimax(self, game_state, depth, maximizing_player):
        """
        基础 Minimax 算法。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            maximizing_player: 是否在当前层最大化（True=我方）

        Returns:
            该局面的评估值
        """
        # TODO: 实现 Minimax
        # 提示：
        # 1. 终局或 depth=0 时返回评估值
        if depth == 0 or game_state.is_over():
            return self.evaluator(game_state)

        legal_moves = [
            move for move in game_state.legal_moves()
            if move.is_play
        ]
        if not legal_moves:
            return self.evaluator(game_state)

        # 2. 如果是最大化方：取所有子节点最大值
        if maximizing_player:
            best_value = -float("inf")
            for move in legal_moves:
                next_state = game_state.apply_move(move)
                value = self.minimax(next_state, depth - 1, False)
                best_value = max(best_value, value)
            return best_value

        # 3. 如果是最小化方：取所有子节点最小值
        best_value = float("inf")
        for move in legal_moves:
            next_state = game_state.apply_move(move)
            value = self.minimax(next_state, depth - 1, True)
            best_value = min(best_value, value)
        return best_value

    def alphabeta(self, game_state, depth, alpha, beta, maximizing_player):
        """
        Alpha-Beta 剪枝优化版 Minimax。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            alpha: 当前最大下界
            beta: 当前最小上界
            maximizing_player: 是否在当前层最大化

        Returns:
            该局面的评估值
        """
        # TODO: 实现 Alpha-Beta 剪枝
        # 提示：在 minimax 基础上添加剪枝逻辑
        # - 最大化方：如果 value >= beta 则剪枝
        # - 最小化方：如果 value <= alpha 则剪枝
        if depth == 0 or game_state.is_over():
            return self.evaluator(game_state)
        state_key = (game_state.board.zobrist_hash(), game_state.next_player)
        cached = self.cache.get(state_key)
        if cached is not None and cached['depth'] >= depth:
            return cached['value']

        legal_moves = [
            move for move in game_state.legal_moves()
            if move.is_play
        ]
        if not legal_moves:
            return self.evaluator(game_state)

        if maximizing_player:
            value = -float("inf")
            for move in legal_moves:
                next_state = game_state.apply_move(move)
                value = max(
                    value,
                    self.alphabeta(next_state, depth - 1, alpha, beta, False)
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            self.cache.put(state_key, depth, value, 'exact')
            return value

        value = float("inf")
        for move in legal_moves:
            next_state = game_state.apply_move(move)
            value = min(
                value,
                self.alphabeta(next_state, depth - 1, alpha, beta, True)
            )
            beta = min(beta, value)
            if beta <= alpha:
                break
        self.cache.put(state_key, depth, value, 'exact')
        return value        

    def _default_evaluator(self, game_state):
        """
        默认局面评估函数（简单版本）。

        学生作业：替换为更复杂的评估函数，如：
            - 气数统计
            - 眼位识别
            - 神经网络评估

        Args:
            game_state: 游戏状态

        Returns:
            评估值（正数对我方有利）
        """
        # TODO: 实现简单的启发式评估
        if game_state.is_over():
            winner = game_state.winner()
            if winner == Player.black:
                return 1000
            elif winner == Player.white:
                return -1000
            return 0

        black_stones = 0
        white_stones = 0
        black_liberties = 0
        white_liberties = 0
        board = game_state.board
        visited_strings = set()

        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                point = Point(row, col)
                color = board.get(point)

                if color == Player.black:
                    black_stones += 1
                    string = board.get_go_string(point)
                    if id(string) not in visited_strings:
                        black_liberties += string.num_liberties
                        visited_strings.add(id(string))

                elif color == Player.white:
                    white_stones += 1
                    string = board.get_go_string(point)
                    if id(string) not in visited_strings:
                        white_liberties += string.num_liberties
                        visited_strings.add(id(string))

        score = (black_stones - white_stones) + 0.1 * (black_liberties - white_liberties)

        return score

    def _get_ordered_moves(self, game_state):
        """
        获取排序后的候选棋步（用于优化剪枝效率）。

        好的排序能让 Alpha-Beta 剪掉更多分支。

        Args:
            game_state: 游戏状态

        Returns:
            按启发式排序的棋步列表
        """
        # TODO: 实现棋步排序
        # 提示：优先检查吃子、提子、连络等好棋
        moves = [
            move for move in game_state.legal_moves()
            if move.is_play
        ]

        center_row = (game_state.board.num_rows + 1) / 2
        center_col = (game_state.board.num_cols + 1) / 2

        moves.sort(
            key=lambda move: abs(move.point.row - center_row) + abs(move.point.col - center_col)
        )
        return moves


class GameResultCache:
    """
    局面缓存（Transposition Table）。

    用 Zobrist 哈希缓存已评估的局面，避免重复计算。
    """

    def __init__(self):
        self.cache = {}

    def get(self, zobrist_hash):
        """获取缓存的评估值。"""
        return self.cache.get(zobrist_hash)

    def put(self, zobrist_hash, depth, value, flag='exact'):
        """
        缓存评估结果。

        Args:
            zobrist_hash: 局面哈希
            depth: 搜索深度
            value: 评估值
            flag: 'exact'/'lower'/'upper'（精确值/下界/上界）
        """
        # TODO: 实现缓存逻辑（考虑深度优先替换策略）
        old_entry = self.cache.get(zobrist_hash)
        if old_entry is None or depth >= old_entry['depth']:
            self.cache[zobrist_hash] = {
                'depth': depth,
                'value': value,
                'flag': flag,
            }
