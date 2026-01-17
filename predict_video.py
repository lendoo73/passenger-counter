import cv2
from ultralytics import YOLO
import yt_dlp
from collections import deque, Counter

# -----------------------------
# Step 1: Download YouTube video
# -----------------------------
url = "https://www.youtube.com/watch?v=3Ek0tg0bRIM"
video_path = "video.mp4"

ydl_opts = {"format": "best", "outtmpl": video_path}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

# -----------------------------
# Step 2: Load YOLO model
# -----------------------------
model = YOLO("runs/detect/train4/weights/best.pt")

# -----------------------------
# Step 3: Open video
# -----------------------------
cap = cv2.VideoCapture(video_path)

# Optional: save processed video
save_output = True
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# -----------------------------
# Step 4: Rolling window for passenger counts
# -----------------------------
window_size = 144                  # last X frames
count_window = deque(maxlen=window_size)

def majority_count(window):
    if not window:
        return 0
    c = Counter(window)
    return c.most_common(1)[0][0]

# -----------------------------
# Step 5: Process frames
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame)
    r = results[0]

    # Count passengers in this frame
    num_persons = len(r.boxes)
    count_window.append(num_persons)

    # Rolling majority
    stable_count = majority_count(count_window)

    # Draw YOLO boxes
    frame = r.plot()

    # Overlay the stable passenger count
    cv2.putText(frame, f"Passengers: {stable_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Show live video
    cv2.imshow("YOLO Passenger Detection", frame)

    # Write to output file
    if save_output:
        out.write(frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
