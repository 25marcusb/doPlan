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

# ---------------- FUNCTIONS ----------------
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

    root.after(30, update_frames)


def save_text():
    label = entry.get()
    commentary = commentary_entry.get()
    frame_number = int(caps["F0"].get(cv2.CAP_PROP_POS_FRAMES))

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([frame_number, label, commentary])

    entry.delete(0, tk.END)
    commentary_entry.delete(0, tk.END)


def toggle_pause():
    global paused
    paused = not paused
    pause_button.config(text="Play" if paused else "Pause")


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

# ---------------- TKINTER GUI ----------------
root = tk.Tk()
root.title("Multi-View Video Labeler")
root.geometry("1100x820")

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

back_button = tk.Button(button_frame, text="<< 100", command=lambda: jump_frames(-100))
back_button.pack(side=tk.LEFT, padx=5)

forward_button = tk.Button(button_frame, text="100 >>", command=lambda: jump_frames(100))
forward_button.pack(side=tk.LEFT, padx=5)

quit_button = tk.Button(button_frame, text="Quit", command=quit_program)
quit_button.pack(side=tk.LEFT, padx=5)

# Start
update_frames()
root.mainloop()
