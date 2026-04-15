
# 1. 依赖导入
# argparse：处理命令行参数
# threading：让 AI 在后台线程中思考，避免界面卡死
# tkinter：构建图形界面
# messagebox：弹出确认框/警告框

import argparse
import threading
import tkinter as tk
from tkinter import messagebox

# 围棋核心逻辑：游戏状态、棋步、玩家、坐标点、计分函数
from dlgo import GameState, Move, Player, Point, compute_game_result

# 2. 工具函数层
def player_name(player):
    return "黑棋" if player == Player.black else "白棋"


def create_ai(ai_type, mcts_rounds, minimax_depth):

    """
    根据用户选择的 AI 类型，创建对应的智能体对象。
    参数：
        ai_type: "random" / "mcts" / "minimax"
        mcts_rounds: MCTS 模拟轮数
        minimax_depth: Minimax 搜索深度
    """
     
    if ai_type == "random":
        from agents.random_agent import RandomAgent

        return RandomAgent()
    if ai_type == "mcts":
        from agents.mcts_agent import MCTSAgent

        return MCTSAgent(num_rounds=mcts_rounds)
    if ai_type == "minimax":
        from agents.minimax_agent import MinimaxAgent

        return MinimaxAgent(max_depth=minimax_depth)
    raise ValueError(f"未知 AI 类型: {ai_type}")


# 3. 整个图形界面的核心。负责：
# 1）保存当前对局状态
# 2）管理窗口和控件
# 3）处理人类与 AI 的交互

class GoGUI:
    def __init__(
        self,
        board_size=5,
        human_color="black",
        ai_type="mcts",
        mcts_rounds=1000,
        minimax_depth=3,
    ):
        """
        初始化界面对象和游戏状态。

        这里主要完成三件事：
        1. 保存棋盘大小、执子方、AI 参数等配置
        2. 初始化围棋对局状态和历史记录
        3. 创建 Tk 窗口并完成首次绘制
        """
        self.board_size = board_size
        self.human_player = Player.black if human_color == "black" else Player.white
        self.ai_player = self.human_player.other
        self.ai_type = ai_type
        self.mcts_rounds = mcts_rounds
        self.minimax_depth = minimax_depth
        self.ai = create_ai(self.ai_type, self.mcts_rounds, self.minimax_depth)

        self.game_state = GameState.new_game(board_size)
        self.history = [self.game_state]
        self.ai_thinking = False

        self.margin = 30
        self.cell = 60 if board_size <= 9 else 40
        self.board_px = self.margin * 2 + self.cell * (self.board_size - 1)

        self.root = tk.Tk()
        self.root.title("围棋 AI 人机对弈")
        self.root.resizable(False, False)

        self.status_var = tk.StringVar()
        self.ai_mode_var = tk.StringVar(value=self.ai_type)

        self._build_ui()
        self._redraw_all()
        self._maybe_ai_turn()

    # 4. 界面搭建层
    # 负责创建窗口中的各种控件：
    # 顶部模式选择、中间棋盘画布、底部状态栏

    def _build_ui(self):
        top = tk.Frame(self.root, padx=10, pady=8)
        top.pack(fill=tk.X)

        tk.Label(top, text="AI 模式:").pack(side=tk.LEFT, padx=(10, 4))
        self.mode_menu = tk.OptionMenu(top, self.ai_mode_var, "random", "mcts", "minimax", command=self._on_mode_changed)
        self.mode_menu.config(width=10)
        self.mode_menu.pack(side=tk.LEFT, padx=(0, 10))

        mode_text = self._mode_info_text()
        self.mode_label = tk.Label(top, text=mode_text, anchor="w")
        self.mode_label.pack(side=tk.LEFT, padx=10)

        # 操作按钮：停一手、认输、悔棋、新游戏
        btn_frame = tk.Frame(top)
        btn_frame.pack(fill=tk.X, padx=10)
        self.pass_btn = tk.Button(btn_frame, text="停一手", width=8, command=self._on_pass)
        self.pass_btn.pack(side=tk.LEFT, padx=4)
        self.resign_btn = tk.Button(btn_frame, text="认输", width=8, command=self._on_resign)
        self.resign_btn.pack(side=tk.LEFT, padx=4)
        self.undo_btn = tk.Button(btn_frame, text="悔棋", width=8, command=self._on_undo)
        self.undo_btn.pack(side=tk.LEFT, padx=4)
        self.new_btn = tk.Button(btn_frame, text="新游戏", width=8, command=self._on_new_game)
        self.new_btn.pack(side=tk.LEFT, padx=4)

        self.canvas = tk.Canvas(
            self.root,
            width=self.board_px,
            height=self.board_px,
            bg="#deb56f",
            highlightthickness=0,
        )
        self.canvas.pack(padx=10, pady=6)
        self.canvas.bind("<Button-1>", self._on_board_click)

        bottom = tk.Frame(self.root, padx=10, pady=8)
        bottom.pack(fill=tk.X)
        tk.Label(bottom, textvariable=self.status_var, anchor="w").pack(fill=tk.X)

    # 5. 坐标转换层
     # 负责界面坐标 <-> 棋盘坐标的转换。

    def _coord_to_point(self, x, y):
        col = round((x - self.margin) / self.cell) + 1
        row = round((y - self.margin) / self.cell) + 1
        if not (1 <= row <= self.board_size and 1 <= col <= self.board_size):
            return None

        px = self.margin + (col - 1) * self.cell
        py = self.margin + (row - 1) * self.cell
        if abs(px - x) > self.cell * 0.45 or abs(py - y) > self.cell * 0.45:
            return None
        return Point(row, col)

    def _point_to_pixel(self, point):
        x = self.margin + (point.col - 1) * self.cell
        y = self.margin + (point.row - 1) * self.cell
        return x, y

    # 6. 绘制显示层
    # 负责把当前 game_state 渲染到界面上：
    # 棋盘线、棋子、状态文字、按钮状态等

    def _draw_grid(self):
        m = self.margin
        n = self.board_size
        c = self.cell
        for i in range(n):
            x = m + i * c
            self.canvas.create_line(x, m, x, m + (n - 1) * c, width=2)
            y = m + i * c
            self.canvas.create_line(m, y, m + (n - 1) * c, y, width=2)

    def _draw_stones(self):
        r = self.cell * 0.38
        for row in range(1, self.board_size + 1):
            for col in range(1, self.board_size + 1):
                point = Point(row, col)
                stone = self.game_state.board.get(point)
                if stone is None:
                    continue
                x, y = self._point_to_pixel(point)
                color = "#111111" if stone == Player.black else "#f4f4f4"
                outline = "#000000" if stone == Player.black else "#808080"
                self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=outline, width=2)

    def _stone_counts(self):
        black = 0
        white = 0
        for row in range(1, self.board_size + 1):
            for col in range(1, self.board_size + 1):
                stone = self.game_state.board.get(Point(row, col))
                if stone == Player.black:
                    black += 1
                elif stone == Player.white:
                    white += 1
        return black, white

    def _status_text(self):
        black, white = self._stone_counts()
        if self.game_state.is_over():
            winner = self.game_state.winner()
            if self.game_state.last_move is not None and self.game_state.last_move.is_resign:
                return f"对局结束：{player_name(winner)}获胜（对手认输） | 棋盘子数 黑:{black} 白:{white}"

            result = compute_game_result(self.game_state)
            white_total = result.w + result.komi
            margin = result.winning_margin
            if winner is None:
                base = "对局结束：平局"
            elif winner == Player.black:
                base = f"对局结束：黑棋获胜(B+{margin:.1f})"
            else:
                base = f"对局结束：白棋获胜(W+{margin:.1f})"
            return (
                f"{base} | 计分 黑:{result.b:.1f} 白:{result.w:.1f}+贴目{result.komi:.1f}={white_total:.1f}"
                f" | 棋盘子数 黑:{black} 白:{white}"
            )

        turn = player_name(self.game_state.next_player)
        if self.ai_thinking:
            return f"轮到 {turn}，AI 正在思考... | 棋盘子数 黑:{black} 白:{white}"
        return f"轮到 {turn} | 棋盘子数 黑:{black} 白:{white}"

    def _mode_info_text(self):
        if self.ai_type == "mcts":
            return f"人类(黑棋) vs MCTS AI(白棋)"
        if self.ai_type == "minimax":
            return f"人类(黑棋) vs Minimax AI(白棋)"
        return f"人类(黑棋) vs 随机 AI(白棋)"

    def _redraw_all(self):
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_stones()
        self.mode_label.config(text=self._mode_info_text())
        self.status_var.set(self._status_text())
        self._refresh_buttons()

    def _refresh_buttons(self):
        game_over = self.game_state.is_over()
        human_turn = self.game_state.next_player == self.human_player
        playable = (not game_over) and (not self.ai_thinking) and human_turn

        state = tk.NORMAL if playable else tk.DISABLED
        self.pass_btn.config(state=state)
        self.resign_btn.config(state=state)

        can_undo = (not self.ai_thinking) and len(self.history) > 1
        self.undo_btn.config(state=tk.NORMAL if can_undo else tk.DISABLED)
        self.new_btn.config(state=tk.NORMAL)
        
        mode_state = tk.NORMAL if not self.ai_thinking else tk.DISABLED
        self.mode_menu.config(state=mode_state)

    # 7. 人类交互层
    # 处理用户的各种输入：
    # 鼠标落子、停一手、认输、切换模式、新游戏、悔棋

    def _try_apply_human_move(self, move):
        if not self.game_state.is_valid_move(move):
            self.status_var.set("非法落子，请选择其他位置。")
            return
        self.game_state = self.game_state.apply_move(move)
        self.history.append(self.game_state)
        self._redraw_all()
        self._maybe_ai_turn()

    def _on_board_click(self, event):
        if self.ai_thinking or self.game_state.is_over():
            return
        if self.game_state.next_player != self.human_player:
            return

        point = self._coord_to_point(event.x, event.y)
        if point is None:
            return
        self._try_apply_human_move(Move.play(point))

    def _on_pass(self):
        self._try_apply_human_move(Move.pass_turn())

    def _on_resign(self):
        if messagebox.askyesno("确认", "确认认输吗？"):
            self._try_apply_human_move(Move.resign())

    def _on_mode_changed(self, new_mode):
        """AI 模式下拉菜单改变时的回调。"""
        if self.ai_thinking:
            return
        self.ai_type = new_mode
        self.ai = create_ai(self.ai_type, self.mcts_rounds, self.minimax_depth)
        self._on_new_game()

    def _on_new_game(self):
        self.game_state = GameState.new_game(self.board_size)
        self.history = [self.game_state]
        self.ai_thinking = False
        self._redraw_all()
        self._maybe_ai_turn()

    def _on_undo(self):
        if self.ai_thinking or len(self.history) <= 1:
            return

        self.history.pop()
        if len(self.history) > 1 and self.history[-1].next_player == self.ai_player:
            self.history.pop()

        self.game_state = self.history[-1]
        self._redraw_all()

    # 8. AI 控制层
    # 负责判断是否该轮到 AI、后台计算 AI 落子、
    # 并在计算结束后安全地更新界面。

    def _maybe_ai_turn(self):
        if self.game_state.is_over() or self.game_state.next_player != self.ai_player:
            self._redraw_all()
            return

        self.ai_thinking = True
        self._redraw_all()

        worker = threading.Thread(target=self._compute_ai_move, daemon=True)
        worker.start()

    def _compute_ai_move(self):
        error = None
        move = None
        try:
            move = self.ai.select_move(self.game_state)
        except Exception as exc:
            error = str(exc)
            move = Move.pass_turn()

        self.root.after(0, lambda: self._finish_ai_move(move, error))

    def _finish_ai_move(self, move, error):
        self.ai_thinking = False
        if error:
            messagebox.showwarning("AI 出错", f"AI 走子失败，已自动停一手。\n{error}")

        if not self.game_state.is_valid_move(move):
            move = Move.pass_turn()

        self.game_state = self.game_state.apply_move(move)
        self.history.append(self.game_state)
        self._redraw_all()

    # 9. 运行层
    # 启动整个 Tkinter 主事件循环

    def run(self):
        self.root.mainloop()

# 10. 程序入口层

def main():
    parser = argparse.ArgumentParser(description="围棋图形界面人机对弈")
    parser.add_argument("--size", type=int, default=5, help="棋盘大小")
    parser.add_argument("--human", choices=["black", "white"], default="black", help="玩家执子")
    parser.add_argument("--ai", choices=["random", "mcts", "minimax"], default="mcts", help="AI 类型")
    parser.add_argument("--mcts-rounds", type=int, default=60, help="MCTS 模拟轮数")
    parser.add_argument("--minimax-depth", type=int, default=3, help="Minimax 搜索深度")
    args = parser.parse_args()

    app = GoGUI(
        board_size=args.size,
        human_color=args.human,
        ai_type=args.ai,
        mcts_rounds=args.mcts_rounds,
        minimax_depth=args.minimax_depth,
    )
    app.run()


if __name__ == "__main__":
    main()
