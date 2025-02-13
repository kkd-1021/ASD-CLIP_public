import os
import cv2

from ultralytics import YOLO
model_pose = YOLO('yolov8n-pose.pt')

def check_two_person(frame, frame_idx):
    yolo_results = model_pose.track(frame, persist=False)
    annotated_frame = yolo_results[0].plot()

    #cv2.imwrite("sample_view/frame" + str(frame_idx) + ".jpg", annotated_frame)
    if len(yolo_results[0].boxes)<2:
        return False
    else:
        return True

# 读取视频
video_fold_path = "org_video_path"
output_fold_path = "processed_video_path"
file_list = os.listdir(video_fold_path)

print(file_list)

for video in file_list:
    index = file_list.index(video)
    video_name = os.path.basename(video)

    input = os.path.join(video_fold_path, video_name)
    cap = cv2.VideoCapture(input)
    print(video_name)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # 获取原视频帧率
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    # 计算采样间隔
    sampling_interval = int(original_fps / 3) if original_fps > 3 else 1

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频帧宽度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频帧高度
    output_path = os.path.join(output_fold_path, video_name)  # 新视频文件路径
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 3, (frame_width, frame_height))

    frame_count = 0
    valid_frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # 只处理采样间隔帧中的一帧
        if frame_count % sampling_interval != 0:
            continue

        print(frame_count)
        print(video_name)
        print(str(index)+'/'+str(len(file_list)))

        is_two_person = check_two_person(frame, frame_count)
        if is_two_person:
            valid_frame_count += 1
            out.write(frame)

    if valid_frame_count < 180:
        out.release()
        os.remove(output_path)
        print(output_path)
        print("drop")
    else:
        out.release()

    cap.release()




