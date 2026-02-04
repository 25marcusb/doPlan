import tkinter as tk
from tkinter import ttk
import csv
import cv2
from PIL import Image, ImageTk
import os

# ---------------- VIDEO SETUP ----------------
video_dir = "/media/cvrr/0A6AF7D76AF7BE0F/CompetitionData/dataset/videos"
base_name = "2021.05.12.22.00.38_veh-35_01008_01518"

camera_ids = ["L0", "L1", "L2", "F0", "B0", "R0", "R1", "R2"]

caps = {}
for cam in camera_ids:
    path = os.path.join(video_dir, f"{base_name}{cam}.mp4")
    caps[cam] = cv2.VideoCapture(path)

ref_cap = caps["F0"]
total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames <= 0:
    total_frames = 1

paused = False
frame_images = {}
updating_scale = False

# ---------------- CSV SETUP ----------------
output_folder = "/home/cvrr/Desktop/VLM Competition/doPlan/outputs"
os.makedirs(output_folder, exist_ok=True)

csv_path = os.path.join(output_folder, base_name + ".csv")

# ---------------- FUNCTIONS ----------------
def update_progress_ui():
    global updating_scale
    current = int(caps["F0"].get(cv2.CAP_PROP_POS_FRAMES))

    updating_scale = True
    scrub_scale.set(current)
    updating_scale = False

    frame_counter_label.config(text=f"{current} / {total_frames - 1}")

    redraw_anchor_markers()

def redraw_current_frames():
    global frame_images

    for cam, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if cam == "F0":
            frame = cv2.resize(frame, (640, 360))
        else:
            frame = cv2.resize(frame, (320, 180))

        img = ImageTk.PhotoImage(Image.fromarray(frame))
        frame_images[cam] = img
        video_labels[cam].config(image=img)
        video_labels[cam].image = img
    
    update_progress_ui()


def update_frames():
    global frame_images

    if not paused:
        for cam, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if cam == "F0":
                frame = cv2.resize(frame, (640, 360))
            else:
                frame = cv2.resize(frame, (320, 180))

            img = ImageTk.PhotoImage(Image.fromarray(frame))
            frame_images[cam] = img
            video_labels[cam].config(image=img)
            video_labels[cam].image = img
    
    update_progress_ui()
    root.after(30, update_frames)


def save_text():
    label = entry.get().strip()
    commentary = commentary_entry.get().strip()

    s = clamp_frame(start_var.get())
    e = clamp_frame(end_var.get())

    if e < s:
        s, e = e, s # start must be less than end

    file_exists = os.path.exists(csv_path)
    file_empty = (not file_exists) or (os.path.getsize(csv_path) == 0)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if file_empty:
            writer.writerow(["start_frame", "end_frame", "label", "commentary"])
        writer.writerow([s, e, label, commentary])

    next_start = default_next_start()
    start_var.set(next_start)
    end_var.set(next_start)

    entry.delete(0, tk.END)
    commentary_entry.delete(0, tk.END)


def toggle_pause():
    global paused
    paused = not paused
    pause_button.config(text="Play" if paused else "Pause")

    if paused:
        # Automatically set End anchor to where user paused
        end_var.set(clamp_frame(current_frame()))


def jump_frames(offset):
    ref_cap = caps["F0"]
    current = int(ref_cap.get(cv2.CAP_PROP_POS_FRAMES))
    target = max(0, current + offset)

    for cap in caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)

    redraw_current_frames()

def quit_program():
    for cap in caps.values():
        cap.release()
    root.destroy()

def current_frame():
    return int(caps["F0"].get(cv2.CAP_PROP_POS_FRAMES))


def clamp_frame(x):
    return max(0, min(total_frames - 1, int(x)))


def set_start_to_current():
    start_var.set(clamp_frame(current_frame()))


def set_end_to_current():
    end_var.set(clamp_frame(current_frame()))


def default_next_start():
    """
    Start anchor placed at end of previous annotation + 1 (or 0 if none).
    Not enforced; user can adjust.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return 0

    # Try to read last line to get last end
    try:
        with open(csv_path, "r", newline="") as f:
            rows = [r for r in csv.reader(f) if r]
        if len(rows) == 0:
            return 0
        last = rows[-1]
        # If there's a header, skip it
        if last[0] == "start_frame":
            return 0
        last_end = int(last[1])
        return clamp_frame(last_end + 1)
    except Exception:
        return 0

# ---------------- TKINTER GUI ----------------
root = tk.Tk()
root.title("Multi-View Video Labeler")
root.geometry("1100x820")

start_var = tk.IntVar(master=root, value=0)
end_var   = tk.IntVar(master=root, value=total_frames - 1)

main_frame = tk.Frame(root)
main_frame.pack(padx=10, pady=10)

video_frame = tk.Frame(main_frame)
video_frame.pack()

video_labels = {}

# Left column (L)
for i, cam in enumerate(["L0", "L1", "L2"]):
    lbl = tk.Label(video_frame)
    lbl.grid(row=i, column=0, padx=5, pady=5)
    video_labels[cam] = lbl

# Middle column
video_labels["F0"] = tk.Label(video_frame)
video_labels["F0"].grid(row=0, column=1, rowspan=2, padx=5, pady=5)

video_labels["B0"] = tk.Label(video_frame)
video_labels["B0"].grid(row=2, column=1, padx=5, pady=5)

# Right column (R)
for i, cam in enumerate(["R0", "R1", "R2"]):
    lbl = tk.Label(video_frame)
    lbl.grid(row=i, column=2, padx=5, pady=5)
    video_labels[cam] = lbl

# Progress bar
progress_frame = tk.Frame(main_frame)
progress_frame.pack(fill="x", pady=(8, 4))

slider_container = tk.Frame(progress_frame)
slider_container.pack(side=tk.LEFT, fill="x", expand=True)

frame_counter_label = tk.Label(progress_frame, text="0 / 0")
frame_counter_label.pack(side=tk.RIGHT, padx=(10, 0))

marker_canvas = tk.Canvas(slider_container, height=12, highlightthickness=0)
marker_canvas.pack(fill="x", side=tk.TOP)

def frame_to_x(frame):
    w = marker_canvas.winfo_width()
    if total_frames <= 1 or w <= 1:
        return 0
    margin = 6
    usable = max(1, w - 2 * margin)
    return int(margin + (clamp_frame(frame) / (total_frames - 1)) * usable)

def redraw_anchor_markers():
    marker_canvas.delete("all")

    if marker_canvas.winfo_width() <= 1:
        root.after(50, redraw_anchor_markers)
        return

    xs = frame_to_x(start_var.get())
    xe = frame_to_x(end_var.get())

    size = 4
    y = 6

    marker_canvas.create_rectangle(
        xs - size, y - size, xs + size, y + size,
        outline="green", fill="green"
    )
    marker_canvas.create_rectangle(
        xe - size, y - size, xe + size, y + size,
        outline="red", fill="red"
    )

marker_canvas.bind("<Configure>", lambda e: redraw_anchor_markers())

def on_scrub(value):
    global updating_scale
    if updating_scale:
        return

    target = int(float(value))
    for cap in caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    redraw_current_frames()

scrub_scale = tk.Scale(
    slider_container,
    from_=0,
    to=total_frames - 1,
    orient="horizontal",
    showvalue=0,
    resolution=1,
    command=on_scrub,
    sliderlength=12,
    bd=0,
    highlightthickness=0
)
scrub_scale.pack(fill="x", expand=True)

start_var.trace_add("write", lambda *args: redraw_anchor_markers())
end_var.trace_add("write", lambda *args: redraw_anchor_markers())

# Range Anchors UI
range_frame = tk.Frame(main_frame)
range_frame.pack(fill="x", pady=(4, 6))

range_inner = tk.Frame(range_frame)
range_inner.pack()

tk.Label(range_inner, text="Start").pack(side=tk.LEFT)
start_spin = tk.Spinbox(
    range_inner, from_=0, to=total_frames - 1,
    textvariable=start_var, width=8
)
start_spin.pack(side=tk.LEFT, padx=(4, 12))

tk.Label(range_inner, text="End").pack(side=tk.LEFT)
end_spin = tk.Spinbox(
    range_inner, from_=0, to=total_frames - 1,
    textvariable=end_var, width=8
)
end_spin.pack(side=tk.LEFT, padx=(4, 12))

tk.Button(range_inner, text="Set Start = current", command=set_start_to_current)\
    .pack(side=tk.LEFT, padx=5)
tk.Button(range_inner, text="Set End = current", command=set_end_to_current)\
    .pack(side=tk.LEFT, padx=5)

# Entry boxes
entry = tk.Entry(main_frame, width=80)
entry.pack(pady=(10, 4))
entry.insert(0, "")

commentary_entry = tk.Entry(main_frame, width=80)
commentary_entry.pack(pady=(0, 10))
commentary_entry.insert(0, "")

# Buttons
button_frame = tk.Frame(main_frame)
button_frame.pack(pady=5)

save_button = tk.Button(button_frame, text="Save", command=save_text)
save_button.pack(side=tk.LEFT, padx=5)

pause_button = tk.Button(button_frame, text="Pause", command=toggle_pause)
pause_button.pack(side=tk.LEFT, padx=5)

back_button = tk.Button(
    button_frame, text="<< 100", command=lambda: jump_frames(-100)
)
back_button.pack(side=tk.LEFT, padx=5)

forward_button = tk.Button(
    button_frame, text="100 >>", command=lambda: jump_frames(100)
)
forward_button.pack(side=tk.LEFT, padx=5)

quit_button = tk.Button(button_frame, text="Quit", command=quit_program)
quit_button.pack(side=tk.LEFT, padx=5)

# Start
start_var.set(default_next_start())
end_var.set(total_frames - 1)
update_frames()
root.mainloop()
