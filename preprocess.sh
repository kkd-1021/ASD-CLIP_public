mkdir -p processed_video_path
mkdir -p video_labels
python video_preprocess.py
python generate_label/label_1Fold.py