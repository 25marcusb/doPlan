import tkinter as tk
import csv
import cv2
from PIL import Image, ImageTk
import os
import random

# ---------------- SETTINGS ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(SCRIPT_DIR, "settings.txt")

if not os.path.exists(SETTINGS_FILE):
    raise FileNotFoundError(f"{SETTINGS_FILE} not found!")

# Default values
video_dir = ""
base_name = ""
min_segment_length = 150
output_folder = os.path.join(SCRIPT_DIR, "outputs")  # default if not in settings

# Load settings
with open(SETTINGS_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "video_dir":
            video_dir = value
        elif key == "base_name":
            base_name = value
        elif key == "min_segment_length":
            min_segment_length = int(value)
        elif key == "output_folder":
            output_folder = value

if not video_dir or not base_name:
    raise ValueError("video_dir and base_name must be set in settings.txt")

# ---------------- VIDEO SETUP ----------------
camera_ids = ["L0", "L1", "L2", "F0", "B0", "R0", "R1", "R2"]
caps = {}
for cam in camera_ids:
    path = os.path.join(video_dir, f"{base_name}{cam}.mp4")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    caps[cam] = cv2.VideoCapture(path)

paused = False
frame_images = {}

ref_cap = caps["F0"]
total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames < min_segment_length:
    raise RuntimeError(f"Video too short for minimum segment length of {min_segment_length} frames.")

# ---------------- OUTPUT ----------------
os.makedirs(output_folder, exist_ok=True)
csv_path = os.path.join(output_folder, base_name + ".csv")

# ---------------- TKINTER ----------------
root = tk.Tk()
root.title("Multi-View Random Segment Labeler")
root.geometry("1150x950")

# ---------------- VARIABLES ----------------
frame_label_var = tk.StringVar(value="Frame: 0")
username_var = tk.StringVar(value="")
segment_start = tk.IntVar(value=0)
segment_end = tk.IntVar(value=0)
jump_var = tk.IntVar(value=100)

# ---------------- FUNCTIONS ----------------
def current_frame():
    return int(caps["F0"].get(cv2.CAP_PROP_POS_FRAMES))

def redraw_current_frames():
    for cam, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        size = (640, 360) if cam == "F0" else (320, 180)
        frame = cv2.resize(frame, size)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        frame_images[cam] = img
        video_labels[cam].config(image=img)
        video_labels[cam].image = img

def update_frames():
    if not paused:
        redraw_current_frames()
        if current_frame() >= segment_end.get():
            toggle_pause()
    frame_label_var.set(f"Frame: {current_frame()} (Segment: {segment_start.get()}-{segment_end.get()})")
    root.after(30, update_frames)

def toggle_pause():
    global paused
    paused = not paused
    pause_button.config(text="Play" if paused else "Pause")
    if paused and current_frame() > segment_end.get():
        set_segment_start_frame()

def jump_frames(offset):
    target = current_frame() + offset
    if offset != 0:
        target -= 1
    target = max(segment_start.get(), min(segment_end.get(), target))
    for cap in caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    redraw_current_frames()

def set_segment_start_frame():
    for cap in caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, segment_start.get())
    redraw_current_frames()

def random_segment():
    start = random.randint(0, total_frames - min_segment_length)
    end = random.randint(start + min_segment_length, total_frames - 1)
    segment_start.set(start)
    segment_end.set(end)
    set_segment_start_frame()
    global paused
    paused = False
    pause_button.config(text="Pause")

def replay_segment():
    set_segment_start_frame()
    global paused
    paused = False
    pause_button.config(text="Pause")

def save_segment_label():
    label = entry.get()
    commentary = commentary_entry.get()
    username = username_var.get()
    start = segment_start.get()
    end = segment_end.get()
    file_exists = os.path.exists(csv_path)
    file_empty = (not file_exists) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if file_empty:
            writer.writerow(["start_frame", "end_frame", "label", "commentary", "username"])
        writer.writerow([start, end, label, commentary, username])
    entry.delete(0, tk.END)
    commentary_entry.delete(0, tk.END)

def quit_program():
    for cap in caps.values():
        cap.release()
    root.destroy()

# ---------------- GUI LAYOUT ----------------
main_frame = tk.Frame(root)
main_frame.pack(padx=10, pady=10)

# Username input
username_frame = tk.Frame(main_frame)
username_frame.pack(pady=(0, 6))
tk.Label(username_frame, text="Username:").pack(side=tk.LEFT)
tk.Entry(username_frame, textvariable=username_var, width=20).pack(side=tk.LEFT)

# Video frame
video_frame = tk.Frame(main_frame)
video_frame.pack()
video_labels = {}
for i, cam in enumerate(["L0", "L1", "L2"]):
    lbl = tk.Label(video_frame)
    lbl.grid(row=i, column=0, padx=5, pady=5)
    video_labels[cam] = lbl
video_labels["F0"] = tk.Label(video_frame)
video_labels["F0"].grid(row=0, column=1, rowspan=2, padx=5, pady=5)
video_labels["B0"] = tk.Label(video_frame)
video_labels["B0"].grid(row=2, column=1, padx=5, pady=5)
for i, cam in enumerate(["R0", "R1", "R2"]):
    lbl = tk.Label(video_frame)
    lbl.grid(row=i, column=2, padx=5, pady=5)
    video_labels[cam] = lbl

# Annotation entries
entry = tk.Entry(main_frame, width=80)
entry.pack(pady=(10, 4))
commentary_entry = tk.Entry(main_frame, width=80)
commentary_entry.pack(pady=(0, 10))

# Frame display
tk.Label(main_frame, textvariable=frame_label_var, font=("Helvetica", 12)).pack(pady=4)

# Controls
controls = tk.Frame(main_frame)
controls.pack(pady=6)
tk.Label(controls, text="Jump").pack(side=tk.LEFT)
tk.Entry(controls, textvariable=jump_var, width=6).pack(side=tk.LEFT, padx=3)
tk.Button(controls, text="<<", command=lambda: jump_frames(-jump_var.get())).pack(side=tk.LEFT, padx=4)
tk.Button(controls, text=">>", command=lambda: jump_frames(jump_var.get())).pack(side=tk.LEFT, padx=4)
pause_button = tk.Button(controls, text="Pause", command=toggle_pause)
pause_button.pack(side=tk.LEFT, padx=5)
tk.Button(controls, text="Replay Segment", command=replay_segment).pack(side=tk.LEFT, padx=5)
tk.Button(controls, text="Next Segment", command=random_segment).pack(side=tk.LEFT, padx=5)
tk.Button(controls, text="Save Segment", command=save_segment_label).pack(side=tk.LEFT, padx=5)
tk.Button(controls, text="Quit", command=quit_program).pack(side=tk.LEFT, padx=5)

# ---------------- START ----------------
random_segment()
update_frames()
root.mainloop()
