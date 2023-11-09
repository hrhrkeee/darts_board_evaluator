import cv2
import IPython
from PIL import Image
from io import BytesIO

import utils
from dartboard_evaluator import DartboardEvaluator


darts_eval = DartboardEvaluator(yolo_model_path = "./models/20230727-1623_yolov8s_sentan-B_v4/weights/best.pt")

cap = cv2.VideoCapture(0)
assert cap.isOpened(), 'Could not open video device'

try:
    while(True):
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (1280, 720))

            socre = darts_eval(frame)
            frame = darts_eval.get_visualized_img()
            frame = cv2.putText(frame, str(socre), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
            
            count = 0
            for i, box in enumerate(darts_eval.result[0].boxes.data.tolist()):
                bbox_coord = list(map(int, box[0:4]))
                conf = box[4]
                label = int(box[5])

                if label == 1:
                    count += 1
                    label = f"ball{count} "
                elif label == 2:
                    label = "board"

                text = f"{label} : {bbox_coord[:2]}, {bbox_coord[2:4]} ({conf:.2f})"
                frame = cv2.putText(frame, text, (10, 120+(20*i)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            frame = frame[:, :, ::-1]

            # GUIに表示
            # 全画面で表示
            cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # 画像を表示
            cv2.imshow("Camera", frame)

            # qキーが押されたら途中終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    cap.release()
    print('Stream stopped')

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()
