import math, random
import numpy as np
from typing import Any
from ultralytics import YOLO

from pathlib import Path


# ボードの (x,y) 座標と半径Rを保存するインスタンスを生成するクラス
class Board_Info :
    def __init__(self , arg_x : int , arg_y : int , arg_radius : int) :
        self.x = arg_x
        self.y = arg_y
        self.radius = arg_radius

    def __str__(self) -> str:
        return "x:{}, y:{}, radius:{}".format(self.x, self.y, self.radius)
        
# ボール中心の (x,y) 座標を保存するインスタンスを生成するクラス
class Ball_Info :
    def __init__(self , arg_x : int , arg_y : int, color : list) :
        self.x = arg_x
        self.y = arg_y
        self.color = color

    def __str__(self) -> str:
        return "x:{}, y:{}".format(self.x, self.y)

# ボールの (r,θ) 極座標を保存するインスタンスを生成するクラス
class Polar_Coordinate :
    def __init__(self , arg_r : float , arg_θ : float) :
        self.r = arg_r

        # ball_radianが0~360の範囲に収まるようにする
        if arg_θ < 0 or arg_θ > 360:
            arg_θ %= 360
        self.θ = arg_θ

    def __str__(self) -> str:
        return "r:{}, θ:{}".format(self.r, self.θ)


class DartboardEvaluator2:
    def __init__(self, yolo_model_path):
        
        # YOLOv8のコンソール出力を停止
        from logging import getLogger
        logger = getLogger('ultralytics')
        logger.disabled = True

        # YOLO
        self.yolo_model_path    = yolo_model_path
        self.yolo_model         = YOLO(str(self.yolo_model_path))
        self.yolo_model_classes = self.yolo_model.names
        self.result = None

        # YOLO検出の設定
        self.detect_confidence_threshold = 0.5
        self.detect_IoU = 0.001
        self.detect_max_instance = 300
        self.detect_source_augment = True
        
        # ラベルの情報
        self.board_label_key = 2
        self.ball_label_key  = 1

        # ボードの情報
        self.board_radius_divisions:int = 3     # ボードの円の数
        self.board_circle_divisions:int = 12    # ボードの円の分割数
        self.board_each_circle_radius:list[int] = [0, 0.214, 0.650, 1.0]   # ボードの各円の半径割合

        # ボードの点数表
        score_table1 = [200]
        score_table2 = [160, 100, 150, 90, 140, 80, 190, 130, 180, 120, 170, 110]
        score_table3 = [40, 10, 30, 10, 20, 10, 70, 10, 60, 10, 50, 10]
        score_table  = [score_table1, score_table2, score_table3]
        # score_tableの中の配列の長さをcircle_divisionsまで拡張する
        self.board_score_table:list = [(table*(math.ceil(self.board_circle_divisions/len(table))))[:12] for table in score_table]


    def __call__(self, input_img:np.ndarray) -> int:
        red_score_list = []
        blue_score_list = []

        self.detect(input_img)

        if self.result is None:
            raise Exception("YOLOv8の検出結果がありません。")

        board  = self.get_board()
        if board is None:
            return None
        
        balls = self.get_balls()

        for ball in balls:
            polar_coordinate = self.ball_coordinate_detect(board, ball)
            
            if ball.color[0] > ball.color[2]:
                red_score_list += [self.score_calculate(polar_coordinate)]
            else:
                blue_score_list += [self.score_calculate(polar_coordinate)]

        return (red_score_list, blue_score_list)

    def detect(self, input_img:np.ndarray) -> Any:
        self.result = self.yolo_model(
                source    = input_img,
                conf      = self.detect_confidence_threshold,
                iou       = self.detect_IoU,
                max_det   = self.detect_max_instance,
                augment   = self.detect_source_augment,
                save      = False,
                classes   = None, # [1, 2, 3],
            )
        return self.result

    def get_board(self):
        boxes = self.result[0].boxes.data.tolist()

        try:
            board_bbox = [i[:4] for i in boxes if i[5] == self.board_label_key][0]
        except:
            return None
    
        board_center_x = (board_bbox[0] + board_bbox[2]) / 2
        board_center_y = (board_bbox[1] + board_bbox[3]) / 2
        board_radius = board_bbox[2] - board_center_x
        
        board = Board_Info(board_center_x , board_center_y , board_radius)

        return board

    def get_balls(self):
        balls = []
        boxes = self.result[0].boxes.data.tolist()
        
        try:
            ball_bboxes = [i[:4] for i in boxes if i[5] == self.ball_label_key]
        except:
            return None
        
        for ball_bbox in ball_bboxes:
            ball_center_x = (ball_bbox[0] + ball_bbox[2]) / 2
            ball_center_y = (ball_bbox[1] + ball_bbox[3]) / 2
            
            ball_color = self.get_ball_color(ball_bbox)

            ball = Ball_Info(ball_center_x , ball_center_y, color=ball_color)
            balls.append(ball)

        return balls

    def get_visualized_img(self):
        if self.result is None:
            raise Exception("YOLOv8の検出結果がありません。")
        
        return self.result[0].plot(conf=False, labels=False)
        
    def ball_coordinate_detect(self, board:Board_Info, ball:Ball_Info):
    
        # ボード中心とボール中心のX座標の差分を計算
        difference_x_coordinate =abs(board.x - ball.x)
        # ボード中心とボール中心のY座標の差分を計算
        difference_y_coordinate = abs(board.y - ball.y)
        # 上記2つの長さから、三平方の定理を用いてrを計算
        r = math.sqrt(difference_x_coordinate**2 + difference_y_coordinate**2)
        # 上記で求めたrを、ボードの半径を用いて正規化
        normalized_r = r / board.radius
    
        # 三角関数cosの値を求める (x -> 底辺 / r -> 斜辺)
        cos_value = difference_x_coordinate / r 
        # 上記で求めた値から、θを逆算(ラジアン出力)
        θ = math.acos(cos_value)
        #　上記で求めたθを、°に変換
        θ_degree = math.degrees(θ)

        # ボールが第2象限にある時
        if   ((board.x > ball.x) and (board.y > ball.y)) :
            θ_degree = 180 - θ_degree
        # ボールが第3象限にある時
        elif ((board.x > ball.x) and (board.y < ball.y)) :
            θ_degree += 180 
        # ボールが第4象限にある時
        elif ((board.x < ball.x) and (board.y < ball.y)) : 
            θ_degree = 360 - θ_degree
        
        polar_coordinate = Polar_Coordinate(normalized_r , θ_degree) 

        return polar_coordinate

    def score_calculate(self, polar_coordinate:Polar_Coordinate) -> int:
        """
        ball_distance: ボードの中心とダーツの距離 (0.0 ~ 1.0)
        ball_radian:   ボードの中心座標からダーツまでの角度[rad]  (0 ~ 360)
        """

        score = 0
        # ダーツがボードの外にある場合
        if polar_coordinate.r > 1.0:
            return score
        
        base_angle = float(360/self.board_circle_divisions) 
        
        for j in range(self.board_radius_divisions) :
            if (self.board_each_circle_radius[j] <= polar_coordinate.r < self.board_each_circle_radius[j+1]) : # ダーツの位置を、rから判断
                # print("darts_distanceは{}番目の円の範囲にありました。" .format(j+1))  
            
                for k in range(self.board_circle_divisions) : 
                    if(base_angle * k <= polar_coordinate.θ < base_angle * (k+1)) :  # ダーツの位置を、θから判断
                        # print("θは{}°以内の位置にありました。" .format(base_angle * (k+1)))
                        return self.board_score_table[j][k]          

        return score

    def get_ball_color(self, ball_bbox):
        org_img = self.result[0].orig_img
        
        bbox_coord = list(map(int, ball_bbox[0:4]))
        
        
        thr = 0.1
        ball_img = org_img[int(bbox_coord[1]*(thr+1)):int(bbox_coord[3]*thr), int(bbox_coord[0]*(thr+1)):int(bbox_coord[2]*thr)]
    
        mean_color_rgb = ball_img.mean(axis=(0, 1))
        
        return mean_color_rgb.tolist()
        
        
    
