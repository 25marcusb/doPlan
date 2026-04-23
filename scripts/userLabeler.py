# to be run by the labeler, draws settings from settings.txt

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

video_dir = ""
base_name = ""
min_segment_length = 150
output_folder = os.path.join(SCRIPT_DIR, "outputs")
debug = 0

commentary_options = [
    "Non-Referential",
    "Static Referential",
    "Dynamic Referential",
    "Both",
]

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
        elif key == "debug":
            debug = int(value)
        elif key == "commentary_options":
            commentary_options = [v.strip() for v in value.split(",") if v.strip()]

if not video_dir:
    raise ValueError("video_dir must be set in settings.txt")

if debug not in (0, 1):
    raise ValueError("debug must be 0 or 1 in settings.txt")

if not os.path.isdir(video_dir):
    raise FileNotFoundError(f"Video directory not found: {video_dir}")

# ---------------- VIDEO SETUP ----------------
camera_ids = ["L0", "L1", "L2", "F0", "B0", "R0", "R1", "R2", "gps_map"]
caps = {}

paused = False
frame_images = {}
frame_buffers = {}

BASE_SMALL_W = 320
BASE_SMALL_H = 180
BASE_LARGE_W = 640
BASE_LARGE_H = 360

current_folder_name = ""
current_video_folder = ""
total_frames = 0

# ---------------- OUTPUT ----------------
os.makedirs(output_folder, exist_ok=True)
csv_path = os.path.join(output_folder, "labels.csv")

# ---------------- TKINTER ----------------
root = tk.Tk()
root.title("Multi-View Random Segment Labeler")

MAIN_WINDOW_W = 1150
MAIN_WINDOW_H = 950
MAP_WINDOW_W = 600
MAP_WINDOW_H = 600
WINDOW_GAP = 16
WINDOW_TOP = 40

# ---------------- VARIABLES ----------------
frame_label_var = tk.StringVar(value="Frame: 0")
time_label_var = tk.StringVar(value="Elapsed: 00:00.0")
username_var = tk.StringVar(value="")
basename_var = tk.StringVar(value="File: ")

segment_start = tk.IntVar(value=0)
segment_end = tk.IntVar(value=0)
jump_var = tk.IntVar(value=100)

commentary_var = tk.StringVar(value=commentary_options[0])

# ---------------- FUNCTIONS ----------------
def get_valid_video_folders():
    valid_folders = []
    required_files = {f"{cam}.mp4" for cam in camera_ids}

    for root_dir, _, files in os.walk(video_dir):
        file_set = set(files)
        if required_files.issubset(file_set):
            valid_folders.append(root_dir)

    if not valid_folders:
        raise RuntimeError(
            f"No valid video folders found in {video_dir}. "
            f"Each folder must contain: {', '.join(sorted(required_files))}"
        )

    return valid_folders

def release_caps():
    global caps
    for cap in caps.values():
        cap.release()
    caps = {}

def release_partial_caps(partial_caps):
    for cap in partial_caps.values():
        cap.release()

def load_video_folder(folder_path):
    global caps, current_folder_name, current_video_folder, total_frames, frame_buffers

    release_caps()
    frame_buffers = {}

    new_caps = {}
    for cam in camera_ids:
        path = os.path.join(folder_path, f"{cam}.mp4")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        new_caps[cam] = cap

    ref_cap = new_caps["F0"]
    new_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if new_total_frames < min_segment_length:
        release_partial_caps(new_caps)
        raise RuntimeError(
            f"Video folder too short for minimum segment length of "
            f"{min_segment_length} frames: {folder_path}"
        )

    caps = new_caps
    current_video_folder = folder_path
    current_folder_name = os.path.basename(folder_path)
    total_frames = new_total_frames
    basename_var.set(f"File: {current_folder_name}")

def choose_random_folder():
    valid_folders = get_valid_video_folders()
    return random.choice(valid_folders)

def current_frame():
    if "F0" not in caps:
        return 0
    return int(caps["F0"].get(cv2.CAP_PROP_POS_FRAMES))

def update_elapsed_time():
    frame = current_frame()
    start = segment_start.get()

    elapsed_frames = max(0, frame - start)
    elapsed_seconds = elapsed_frames * 0.1

    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    tenths = int((elapsed_seconds - int(elapsed_seconds)) * 10)

    time_label_var.set(f"Elapsed: {minutes:02d}:{seconds:02d}.{tenths}")

def get_video_scale():
    main_w = max(1, main_frame.winfo_width())
    main_h = max(1, main_frame.winfo_height())

    reserve_h = (
        username_frame.winfo_reqheight()
        + entry.winfo_reqheight()
        + commentary_menu.winfo_reqheight()
        + time_status_label.winfo_reqheight()
        + controls.winfo_reqheight()
        + 90
    )

    if debug == 1:
        reserve_h += basename_label.winfo_reqheight()
        reserve_h += frame_status_label.winfo_reqheight()

    avail_w = max(200, main_w - 30)
    avail_h = max(120, main_h - reserve_h)

    base_w = (BASE_SMALL_W * 2) + BASE_LARGE_W + 30
    base_h = (BASE_SMALL_H * 3) + 30

    scale = min(avail_w / base_w, avail_h / base_h)
    return max(0.2, scale)

def get_target_size(cam):
    if cam == "gps_map":
        if "gps_map" in video_labels and video_labels["gps_map"].winfo_exists():
            return 600, 600
        return BASE_SMALL_W, BASE_SMALL_H

    scale = get_video_scale()
    if cam == "F0":
        return int(BASE_LARGE_W * scale), int(BASE_LARGE_H * scale)
    return int(BASE_SMALL_W * scale), int(BASE_SMALL_H * scale)

def redraw_current_frames(read_new=True):
    for cam, cap in caps.items():
        if read_new:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_buffers[cam] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_rgb = frame_buffers.get(cam)
        if frame_rgb is None:
            continue

        label_widget = video_labels.get(cam)
        if label_widget is None or not label_widget.winfo_exists():
            continue

        target_w, target_h = get_target_size(cam)
        frame = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        frame_images[cam] = img
        label_widget.config(image=img)
        label_widget.image = img

def update_frames():
    if caps and not paused:
        redraw_current_frames(read_new=True)
        if current_frame() >= segment_end.get():
            toggle_pause()

    if debug == 1:
        frame_label_var.set(
            f"Frame: {current_frame()} "
            f"(Segment: {segment_start.get()}-{segment_end.get()})"
        )

    update_elapsed_time()
    root.after(90, update_frames)

def toggle_pause():
    global paused
    paused = not paused
    pause_button.config(text="Play" if paused else "Pause")
    if paused and current_frame() > segment_end.get():
        set_segment_start_frame()

def jump_frames(offset):
    if not caps:
        return

    target = current_frame() + offset
    if offset != 0:
        target -= 1
    target = max(segment_start.get(), min(segment_end.get(), target))
    for cap in caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    redraw_current_frames()

def set_segment_start_frame():
    if not caps:
        return

    for cap in caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, segment_start.get())
    redraw_current_frames()

def random_segment():
    global paused

    while True:
        folder_path = choose_random_folder()
        try:
            load_video_folder(folder_path)
            break
        except RuntimeError:
            continue

    start = random.randint(0, total_frames - min_segment_length)
    end = random.randint(start + min_segment_length, total_frames - 1)
    segment_start.set(start)
    segment_end.set(end)
    set_segment_start_frame()

    paused = False
    pause_button.config(text="Pause")

def replay_segment():
    global paused
    set_segment_start_frame()
    paused = False
    pause_button.config(text="Pause")

def write_label_row(label_text):
    commentary = commentary_var.get()
    username = username_var.get().strip()
    start = segment_start.get()
    end = segment_end.get()

    file_exists = os.path.exists(csv_path)
    file_empty = (not file_exists) or (os.path.getsize(csv_path) == 0)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if file_empty:
            writer.writerow(
                ["folder", "start_frame", "end_frame", "label", "commentary", "username"]
            )
        writer.writerow([current_folder_name, start, end, label_text, commentary, username])

def save_segment_label():
    label = entry.get()
    write_label_row(label)

    entry.delete(0, tk.END)
    commentary_var.set(commentary_options[0])

def save_and_next():
    label = entry.get()
    write_label_row(label)

    entry.delete(0, tk.END)
    commentary_var.set(commentary_options[0])

    random_segment()

def nothing_to_annotate():
    write_label_row("")

    entry.delete(0, tk.END)
    commentary_var.set(commentary_options[0])

    random_segment()

def quit_program():
    release_caps()
    root.destroy()

def on_window_resize(_event):
    if caps:
        redraw_current_frames(read_new=False)

def on_map_window_resize(_event):
    if caps:
        redraw_current_frames(read_new=False)

# ---------------- GUI LAYOUT ----------------
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

basename_label = tk.Label(
    main_frame,
    textvariable=basename_var,
    font=("Helvetica", 14, "bold")
)
if debug == 1:
    basename_label.pack(pady=(0, 6))

username_frame = tk.Frame(main_frame)
username_frame.pack(pady=(0, 6))
tk.Label(username_frame, text="Username:").pack(side=tk.LEFT)
tk.Entry(username_frame, textvariable=username_var, width=20).pack(side=tk.LEFT)

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

entry = tk.Entry(main_frame, width=80)
entry.pack(pady=(10, 4))

commentary_menu = tk.OptionMenu(
    main_frame,
    commentary_var,
    *commentary_options
)
commentary_menu.pack(pady=(0, 10))

frame_status_label = tk.Label(
    main_frame,
    textvariable=frame_label_var,
    font=("Helvetica", 12)
)
if debug == 1:
    frame_status_label.pack(pady=4)

time_status_label = tk.Label(
    main_frame,
    textvariable=time_label_var,
    font=("Helvetica", 12)
)
time_status_label.pack(pady=2)

controls = tk.Frame(main_frame)
controls.pack(pady=6)

tk.Label(controls, text="Jump").pack(side=tk.LEFT)
tk.Entry(controls, textvariable=jump_var, width=6).pack(side=tk.LEFT, padx=3)

tk.Button(
    controls, text="<<",
    command=lambda: jump_frames(-jump_var.get())
).pack(side=tk.LEFT, padx=4)

tk.Button(
    controls, text=">>",
    command=lambda: jump_frames(jump_var.get())
).pack(side=tk.LEFT, padx=4)

pause_button = tk.Button(controls, text="Pause", command=toggle_pause)
pause_button.pack(side=tk.LEFT, padx=5)

tk.Button(
    controls, text="Replay Segment",
    command=replay_segment
).pack(side=tk.LEFT, padx=5)

tk.Button(
    controls, text="Next Segment",
    command=random_segment
).pack(side=tk.LEFT, padx=5)

tk.Button(
    controls, text="Save Segment",
    command=save_segment_label
).pack(side=tk.LEFT, padx=5)

tk.Button(
    controls, text="Save and Next",
    command=save_and_next
).pack(side=tk.LEFT, padx=5)

tk.Button(
    controls, text="Nothing to Annotate",
    command=nothing_to_annotate
).pack(side=tk.LEFT, padx=5)

tk.Button(
    controls, text="Quit",
    command=quit_program
).pack(side=tk.LEFT, padx=5)

map_window = tk.Toplevel(root)
map_window.title("Map View")
map_window.geometry(f"{MAP_WINDOW_W}x{MAP_WINDOW_H}")

map_frame = tk.Frame(map_window)
map_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

video_labels["gps_map"] = tk.Label(map_frame)
video_labels["gps_map"].pack(fill=tk.BOTH, expand=True)

def position_windows_side_by_side():
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    max_h = max(MAIN_WINDOW_H, MAP_WINDOW_H)
    top_y = max(0, min(WINDOW_TOP, screen_h - max_h))

    total_w = MAIN_WINDOW_W + WINDOW_GAP + MAP_WINDOW_W
    main_x = max(0, (screen_w - total_w) // 2)
    map_x = main_x + MAIN_WINDOW_W + WINDOW_GAP

    if map_x + MAP_WINDOW_W > screen_w:
        map_x = max(0, screen_w - MAP_WINDOW_W)
    if main_x + MAIN_WINDOW_W > screen_w:
        main_x = 0

    root.geometry(f"{MAIN_WINDOW_W}x{MAIN_WINDOW_H}+{main_x}+{top_y}")
    map_window.geometry(f"{MAP_WINDOW_W}x{MAP_WINDOW_H}+{map_x}+{top_y}")

root.bind("<Configure>", on_window_resize)
map_window.bind("<Configure>", on_map_window_resize)
root.after(0, position_windows_side_by_side)

# ---------------- START ----------------
random_segment()
update_frames()
root.mainloop()