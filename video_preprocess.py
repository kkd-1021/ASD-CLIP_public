import os
import cv2
from ultralytics import YOLO


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'yolov8n-pose.pt')
model_pose = YOLO(model_path)


def check_two_person(frame):

    results = model_pose.track(frame, persist=False)
    return len(results[0].boxes) >= 2


def process_video(input_path, output_folder):

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"can't open video: {input_path}")
        return

    # 视频参数
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    segment_duration = 180  # 3分钟（秒）
    target_fps = 3
    frames_per_segment = segment_duration * target_fps  # 540帧/段
    sampling_interval = max(1, int(original_fps // target_fps))


    base_name = os.path.splitext(os.path.basename(input_path))[0]

    segment_count = 0
    current_segment_frames = []
    count = 0
    for global_frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break



        if global_frame_idx % sampling_interval != 0:
            continue

        count += 1
        if check_two_person(frame):
            current_segment_frames.append(frame)

            if count >= frames_per_segment:
                output_path = os.path.join(
                    output_folder,
                    f"{base_name}_{segment_count}.mp4"
                )


                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    target_fps,
                    (frame_width, frame_height)
                )
                for f in current_segment_frames[:frames_per_segment]:
                    writer.write(f)
                writer.release()

                print(f"saved {output_path}")
                current_segment_frames = []
                segment_count += 1
                count=0

    cap.release()


# 主处理流程
input_folder = "org_video_path"
output_folder = "processed_video_path"

for filename in os.listdir(input_folder):
    if filename.endswith((".mp4", ".MOV",".MP4")):
        input_path = os.path.join(input_folder, filename)
        print(f"start process{filename}")

        process_video(input_path, output_folder)


output_file_path = "clinical_data/total_video_list.txt"
# 创建目录
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
with open(output_file_path, 'w') as f:
    for filename in os.listdir(output_folder):
        if filename.endswith((".mp4", ".MOV", ".MP4")):
            f.write(filename + '\n')