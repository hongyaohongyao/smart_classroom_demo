import time

import cv2
import torch

from models.fatigue_monitor import SleepDetector
from models.pose_estimator import AlphaPoseEstimator
from models.yolo_detector import YoloV5Detector
from utils.vis import draw_keypoints136

yolov5_weight = './weights/yolov5s.torchscript.pt'
alphapose_weight = './weights/halpe136_mobile.torchscript.pth'

box_color = (0, 255, 0)

torch._C._jit_set_profiling_mode(False)
torch.jit.optimized_execution(False)

if __name__ == '__main__':
    detector = YoloV5Detector(weights=yolov5_weight, device='cuda')
    pose = AlphaPoseEstimator(weights=alphapose_weight, device='cuda',
                              face_aligner_weights='weights/mobilenet56_se_external_model_best.torchscript.pth')

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
    i = 0
    while ret:
        last_time = time.time()
        pred = detector.detect(frame)
        preds_kps, preds_scores = pose.estimate(frame, pred)
        for det in pred:
            show_text = "person: %.2f" % det[4]
            cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), box_color, 2)
            cv2.putText(frame, show_text,
                        (det[0], det[1]),
                        cv2.FONT_HERSHEY_COMPLEX,
                        float((det[2] - det[0]) / 200),
                        box_color)
        if preds_kps.shape[0] > 0:
            draw_keypoints136(frame, preds_kps, preds_scores)
            if SleepDetector.detector_v2(preds_kps[:, 26:94].numpy())[0] == 1:
                print('======================================')
        # if preds_scores.shape[0] > 0:
        #     face_preds_score = preds_scores[0, 26:94]
        #     print(torch.mean(face_preds_score[27:48]),
        #           torch.mean(face_preds_score[48:68]))

        current_time = time.time()
        fps = 1 / (current_time - last_time)

        cv2.putText(frame, "FPS: %.2f" % fps, (0, 52), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imshow("yolov5", frame)
        # 下一帧
        ret, frame = cap.read()
        if cv2.waitKey(30) and 0xFF == 'q':
            break
        # Releasing all the resources
    cap.release()
    cv2.destroyAllWindows()
