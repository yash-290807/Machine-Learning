import cv2
import os

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = r"C:\AI Projects\HelmetViolationSystem\test_videos\trafficvideo.mp4"

OUTPUT_DIR = r"C:\AI Projects\HelmetViolationSystem\datasets\cctv_training\images"

# Number of frames to extract
NUM_FRAMES = 300


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# OPEN VIDEO
# ============================================================

print("=" * 60)
print("CCTV TRAINING FRAME EXTRACTION")
print("=" * 60)

print("\nOpening video...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    print(VIDEO_PATH)
    exit()


# ============================================================
# VIDEO INFORMATION
# ============================================================

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration = total_frames / fps if fps > 0 else 0

print("\nVideo information:")
print("Width        :", width)
print("Height       :", height)
print("FPS          :", fps)
print("Total frames :", total_frames)
print("Duration     :", round(duration, 2), "seconds")

print("\nExtracting", NUM_FRAMES, "frames...")


# ============================================================
# CALCULATE FRAME POSITIONS
# ============================================================

frame_numbers = [
    int(i * (total_frames - 1) / (NUM_FRAMES - 1))
    for i in range(NUM_FRAMES)
]


# ============================================================
# EXTRACT FRAMES
# ============================================================

saved = 0

for i, frame_number in enumerate(frame_numbers):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if not ret:
        print("WARNING: Could not read frame", frame_number)
        continue

    filename = os.path.join(
        OUTPUT_DIR,
        f"cctv_{saved:04d}.jpg"
    )

    cv2.imwrite(filename, frame)

    saved += 1

    if saved % 25 == 0 or saved == NUM_FRAMES:
        percentage = (saved / NUM_FRAMES) * 100
        print(
            f"Saved: {saved}/{NUM_FRAMES} "
            f"({percentage:.1f}%)"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("FRAME EXTRACTION COMPLETE")
print("=" * 60)

print("\nFrames saved:", saved)

print("\nLocation:")
print(OUTPUT_DIR)

print("\nNext step:")
print("Label these CCTV frames as helmet / no_helmet.")