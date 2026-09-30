import cv2
import os

# ==============================
# VIDEO PATH
# ==============================
video_path = r"C:\AI Projects\HelmetViolationSystem\test_videos\trafficvideo.mp4"

# ==============================
# OUTPUT FOLDER
# ==============================
output_dir = r"C:\AI Projects\HelmetViolationSystem\video_frames_100"

os.makedirs(output_dir, exist_ok=True)

# ==============================
# OPEN VIDEO
# ==============================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Total frames: {total_frames}")
print(f"FPS: {fps:.2f}")

# ==============================
# NUMBER OF FRAMES TO EXTRACT
# ==============================
num_frames = 100

# Calculate evenly spaced frame numbers
frame_numbers = [
    int(i * (total_frames - 1) / (num_frames - 1))
    for i in range(num_frames)
]

# ==============================
# EXTRACT FRAMES
# ==============================
saved = 0

for i, frame_number in enumerate(frame_numbers, start=1):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if not success:
        print(f"Could not read frame {frame_number}")
        continue

    filename = os.path.join(
        output_dir,
        f"traffic_frame_{i:03d}.jpg"
    )

    cv2.imwrite(filename, frame)

    saved += 1

    print(
        f"Saved {saved}/{num_frames} "
        f"(video frame {frame_number})"
    )

cap.release()

print("\n==============================")
print("FRAME EXTRACTION COMPLETE")
print("==============================")
print(f"Frames saved: {saved}")
print(f"Location: {output_dir}")