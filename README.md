📌 Object Detection & Tracking Prototype
🔎 Project Overview

This project implements a basic object detection and tracking system using a pretrained YOLOv8 model. The system detects objects in video frames, assigns tracking IDs across frames using centroid-based tracking logic, and generates summary statistics such as object counts and tracking duration.
The prototype runs locally and produces an annotated output video along with a CSV summary file.

⚙️ Technologies Used
Python
YOLOv8 (Ultralytics) – Object detection
OpenCV – Video processing
NumPy & Pandas – Data handling and statistics

🚀 Features
Object detection from video frames
Tracking objects across frames
Unique tracking ID assignment
Output video with bounding boxes and IDs
CSV summary with object statistics

▶️ How to Run
Install dependencies:
pip install -r requirements.txt

Place input video:
input_video.mp4

Run:
python main.py

📂 Output Generated
output/tracked_output.mp4 → Annotated output video
output/object_summary.csv → Summary statistics

⚠️ Limitations
Simple centroid tracker may assign new IDs if objects move abruptly or detection fluctuates.
No model fine-tuning performed (pretrained weights used).

📈 Future Improvements
Use advanced tracking algorithms (DeepSORT, ByteTrack)
Fine-tune detection model for specific datasets
Real-time analytics dashboard integration
