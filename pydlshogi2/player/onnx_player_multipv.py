import time
import math
import onnxruntime
import numpy as np

from pydlshogi2.player.onnx_player import OnnxPlayer
from cshogi.dlshogi import make_input_features, make_move_label, FEATURES1_NUM, FEATURES2_NUM
from cshogi import move_to_usi

DEFAULT_MULTI_PV = 8

class OnnxPlayerMultiPV(OnnxPlayer):
    # USIエンジンの名前
    name = 'python-dlshogi-onnx-multipv'
    # デフォルトモデル
    DEFAULT_MODELFILE = 'model/model-0000167.onnx'

    # Number of PVs
    multipv = DEFAULT_MULTI_PV

    bestmoves = []
    bestvalues = []
    ponder_moves = []
    pvs = []
    cps = []


    # 最善手取得とinfoの表示
    def get_bestmove_and_print_pv(self):

        # 探索にかかった時間を求める
        finish_time = time.time() - self.begin_time

        self.bestmoves.clear()
        self.bestvalues.clear()
        self.ponder_moves.clear()
        self.pvs.clear()
        self.cps.clear()

        # 訪問回数最大の手を選択する
        current_node = self.tree.current_head
        sorted_indices = np.argsort(-current_node.child_move_count)

        print(len(current_node.child_move_count))
        print((current_node.child_move_count))

        for i in range(self.multipv):
            if i >= len(current_node.child_move_count):
                break

            selected_index = sorted_indices[i]

            if current_node.child_move_count[selected_index] <= 0:
                break

            # 選択した着手の勝率の算出
            bestvalue = current_node.child_sum_value[selected_index] / current_node.child_move_count[selected_index]
            self.bestvalues.append(bestvalue)

            bestmove = current_node.child_move[selected_index]
            self.bestmoves.append(bestmove)

            # 勝率を評価値に変換
            if bestvalue == 1.0:
                cp = 30000
            elif bestvalue == 0.0:
                cp = -30000
            else:
                print(i, bestmove, bestvalue)
                cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)
            self.cps.append(cp)

            # PV
            pv = move_to_usi(bestmove)
            ponder_move = None
            pv_node = current_node
            while pv_node.child_node:
                pv_node = pv_node.child_node[selected_index]
                if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                    break
                selected_index = np.argmax(pv_node.child_move_count)
                pv += ' ' + move_to_usi(pv_node.child_move[selected_index])
                if ponder_move is None:
                    ponder_move = pv_node.child_move[selected_index]
            self.pvs.append(pv)
            self.ponder_moves.append(ponder_move)

            print('info multipv {} nps {} time {} nodes {} score cp {} pv {}'.format(
                i + 1,
                int(self.playout_count / finish_time) if finish_time > 0 else 0,
                int(finish_time * 1000),
                current_node.move_count,
                cp, pv), flush=True)

        return self.bestmoves[0], self.bestvalues[0], self.ponder_moves[0]
    
    def usi(self):
        super().usi()
        print('option name MultiPV type spin default ' + str(DEFAULT_MULTI_PV) + ' min 1 max 30')

    def setoption(self, args):
        if args[1].lower() == 'multipv':
            self.multipv = int(args[3])
        else:
            super().setoption(args)

    def go(self, check_mate=False):
        # 探索開始時刻の記録
        self.begin_time = time.time()

        # 投了チェック
        if self.root_board.is_game_over():
            return 'resign', None

        # 入玉宣言勝ちチェック
        if self.root_board.is_nyugyoku():
            return 'win', None

        current_node = self.tree.current_head

        if check_mate:
            # 詰みの場合
            if current_node.value == VALUE_WIN:
                matemove = self.root_board.mate_move(3)
                if matemove != 0:
                    print('info score mate 3 pv {}'.format(move_to_usi(matemove)), flush=True)
                    return move_to_usi(matemove), None
            if not self.root_board.is_check():
                matemove = self.root_board.mate_move_in_1ply()
                if matemove:
                    print('info score mate 1 pv {}'.format(move_to_usi(matemove)), flush=True)
                    return move_to_usi(matemove), None

        # プレイアウト数をクリア
        self.playout_count = 0

        # ルートノードが未展開の場合、展開する
        if current_node.child_move is None:
            current_node.expand_node(self.root_board)

        # 候補手が1つの場合は、その手を返す
        if self.halt is None and len(current_node.child_move) == 1:
            if current_node.child_move_count[0] > 0:
                bestmove, bestvalue, ponder_move = self.get_bestmove_and_print_pv()
                return move_to_usi(bestmove), move_to_usi(ponder_move) if ponder_move else None
            else:
                return move_to_usi(current_node.child_move[0]), None

        # ルートノードが未評価の場合、評価する
        if current_node.policy is None:
            self.current_batch_index = 0
            self.queue_node(self.root_board, current_node)
            self.eval_node()

        # 探索
        self.search()

        # 最善手の取得とPVの表示
        bestmove, bestvalue, ponder_move = self.get_bestmove_and_print_pv()

        # for debug
        if self.debug:
            for i in range(len(current_node.child_move)):
                print('{:3}:{:5} move_count:{:4} nn_rate:{:.5f} win_rate:{:.5f}'.format(
                    i, move_to_usi(current_node.child_move[i]),
                    current_node.child_move_count[i],
                    current_node.policy[i],
                    current_node.child_sum_value[i] / current_node.child_move_count[i] if current_node.child_move_count[i] > 0 else 0))

        # 閾値未満の場合投了
        if bestvalue < self.resign_threshold:
            return 'resign', None

        return move_to_usi(bestmove), move_to_usi(ponder_move) if ponder_move else None



if __name__ == '__main__':
    player = OnnxPlayerMultiPV()
    player.run()