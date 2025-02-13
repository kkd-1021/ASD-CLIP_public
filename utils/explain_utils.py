



import os
import cv2

from ultralytics import YOLO
model_pose = YOLO('yolov8n-pose.pt')
# 读取视频

img_folder_path="path_to_dict/result_explain"








for file_name in os.listdir(img_folder_path):
    # 检查文件名是否以"bk"开头，并且是文件而不是目录
    if file_name.startswith('bk') and os.path.isfile(os.path.join(img_folder_path, file_name)):
        file_path = os.path.join(img_folder_path, file_name)

        img = cv2.imread(file_path)
        yolo_results = model_pose.track(img, persist=True)
        annotated_frame = yolo_results[0].plot()

        cv2.imwrite("path_to_dict/result_explain/yolo_"+file_name+".jpg", annotated_frame)




