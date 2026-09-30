from ultralytics import YOLO
import cv2
import os


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"C:\AI Projects\HelmetViolationSystem\runs\helmet_yolo11n-3\weights\best.pt"

VIDEO_PATH = r"C:\AI Projects\HelmetViolationSystem\test_videos\trafficvideo.mp4"

OUTPUT_PATH = r"C:\AI Projects\HelmetViolationSystem\outputs\helmet_detection.mp4"

CONFIDENCE = 0.25

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found!")
    print(MODEL_PATH)
    exit()

if not os.path.exists(VIDEO_PATH):
    print("ERROR: Video not found!")
    print(VIDEO_PATH)
    exit()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("=" * 60)
print("HELMET VIOLATION DETECTION")
print("=" * 60)

print()
print("Loading trained YOLO11 model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")
print("Classes:", model.names)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print()
print("Video information:")
print("Width       :", width)
print("Height      :", height)
print("FPS         :", fps)
print("Total frames:", total_frames)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("ERROR: Could not create output video.")
    cap.release()
    exit()


# ============================================================
# PROCESS VIDEO
# ============================================================

print()
print("Starting detection...")
print("The processed video will be saved automatically.")
print()

frame_count = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    # --------------------------------------------------------
    # DRAW DETECTION BOXES
    # --------------------------------------------------------

    annotated_frame = results[0].plot()

    # --------------------------------------------------------
    # SAVE FRAME TO OUTPUT VIDEO
    # --------------------------------------------------------

    out.write(annotated_frame)

    # --------------------------------------------------------
    # SHOW PROGRESS EVERY 100 FRAMES
    # --------------------------------------------------------

    if frame_count % 100 == 0:

        percentage = (frame_count / total_frames) * 100

        print(
            f"Processed: {frame_count}/{total_frames} "
            f"({percentage:.1f}%)"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()


# ============================================================
# DETECTION COMPLETE
# ============================================================

print()
print("=" * 60)
print("DETECTION COMPLETE")
print("=" * 60)

print()
print("Frames processed:", frame_count)

print()
print("Output video:")
print(OUTPUT_PATH)

print()
print("You can now open the output video and check the detections.")