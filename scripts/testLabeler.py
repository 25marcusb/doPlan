#used for testing the labeler or advance through videos

import tkinter as tk
import csv
import cv2
from PIL import Image, ImageTk
import os

# ---------------- VIDEO SETUP ----------------
video_dir = "/media/cvrr/0A6AF7D76AF7BE0F/CompetitionData/dataset/videos"
base_name = "2021.05.12.22.28.35_veh-35_00620_01164"

camera_ids = ["L0", "L1", "L2", "F0", "B0", "R0", "R1", "R2"]

caps = {}
for cam in camera_ids:
    path = os.path.join(video_dir, f"{base_name}{cam}.mp4")
    caps[cam] = cv2.VideoCapture(path)

paused = False
frame_images = {}

# ---------------- CSV SETUP ----------------
output_folder = "/home/cvrr/Desktop/VLM Competition/doPlan/outputs"
os.makedirs(output_folder, exist_ok=True)
csv_path = os.path.join(output_folder, base_name + ".csv")

# ---------------- TKINTER ----------------
root = tk.Tk()
root.title("Multi-View Video Labeler")
root.geometry("1150x900")

# Frame control vars (MUST be after root)
start_var = tk.IntVar(value=0)
end_var = tk.IntVar(value=0)
jump_var = tk.IntVar(value=100)
frame_label_var = tk.StringVar(value="Frame: 0")

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

    frame_label_var.set(f"Frame: {current_frame()}")
    root.after(30, update_frames)


def toggle_pause():
    global paused
    paused = not paused
    pause_button.config(text="Play" if paused else "Pause")


def jump_frames(offset):
    target = current_frame() + offset

    if offset > 0:
        target -= 1  # adjust for read() advancing the frame
    elif offset < 0:
        target -= 1  # adjust for read() advancing the frame backward

    target = max(0, target)

    for cap in caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)

    redraw_current_frames()




def set_start():
    start_var.set(current_frame())


def set_end():
    end_var.set(current_frame())


def save_range_label():
    label = entry.get()
    commentary = commentary_entry.get()

    start = start_var.get()
    end = end_var.get()

    if start > end:
        start, end = end, start

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([start, end, label, commentary])

    entry.delete(0, tk.END)
    commentary_entry.delete(0, tk.END)


def quit_program():
    for cap in caps.values():
        cap.release()
    root.destroy()

# ---------------- GUI LAYOUT ----------------
main_frame = tk.Frame(root)
main_frame.pack(padx=10, pady=10)

video_frame = tk.Frame(main_frame)
video_frame.pack()

video_labels = {}

# Left cameras
for i, cam in enumerate(["L0", "L1", "L2"]):
    lbl = tk.Label(video_frame)
    lbl.grid(row=i, column=0, padx=5, pady=5)
    video_labels[cam] = lbl

# Center
video_labels["F0"] = tk.Label(video_frame)
video_labels["F0"].grid(row=0, column=1, rowspan=2, padx=5, pady=5)

video_labels["B0"] = tk.Label(video_frame)
video_labels["B0"].grid(row=2, column=1, padx=5, pady=5)

# Right cameras
for i, cam in enumerate(["R0", "R1", "R2"]):
    lbl = tk.Label(video_frame)
    lbl.grid(row=i, column=2, padx=5, pady=5)
    video_labels[cam] = lbl

# ---------------- LABEL INPUT ----------------
entry = tk.Entry(main_frame, width=80)
entry.pack(pady=(10, 4))

commentary_entry = tk.Entry(main_frame, width=80)
commentary_entry.pack(pady=(0, 10))

# ---------------- FRAME RANGE CONTROLS ----------------
range_frame = tk.Frame(main_frame)
range_frame.pack(pady=6)

tk.Label(range_frame, text="Start").grid(row=0, column=0)
tk.Entry(range_frame, textvariable=start_var, width=8).grid(row=0, column=1)
tk.Button(range_frame, text="Set", command=set_start).grid(row=0, column=2)

tk.Label(range_frame, text="End").grid(row=0, column=3)
tk.Entry(range_frame, textvariable=end_var, width=8).grid(row=0, column=4)
tk.Button(range_frame, text="Set", command=set_end).grid(row=0, column=5)

# ---------------- FRAME DISPLAY ----------------
tk.Label(main_frame, textvariable=frame_label_var, font=("Helvetica", 12)).pack(pady=4)

# ---------------- CONTROLS ----------------
controls = tk.Frame(main_frame)
controls.pack(pady=6)

tk.Label(controls, text="Jump").pack(side=tk.LEFT)
tk.Entry(controls, textvariable=jump_var, width=6).pack(side=tk.LEFT, padx=3)

tk.Button(controls, text="<<", command=lambda: jump_frames(-jump_var.get())).pack(side=tk.LEFT, padx=4)
tk.Button(controls, text=">>", command=lambda: jump_frames(jump_var.get())).pack(side=tk.LEFT, padx=4)

pause_button = tk.Button(controls, text="Pause", command=toggle_pause)
pause_button.pack(side=tk.LEFT, padx=5)

tk.Button(controls, text="Save Range", command=save_range_label).pack(side=tk.LEFT, padx=5)
tk.Button(controls, text="Quit", command=quit_program).pack(side=tk.LEFT, padx=5)

# ---------------- START ----------------
update_frames()
root.mainloop()
