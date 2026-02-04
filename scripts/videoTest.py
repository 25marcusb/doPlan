import tkinter as tk
import csv
import cv2
from PIL import Image, ImageTk
import os

# --- VIDEO SETUP ---
video_path = "/home/cvrr/Desktop/VLM Competition/doPlan/videos/2021.05.12.22.28.35_veh-35_00620_01164.mp4"
cap = cv2.VideoCapture(video_path)
paused = False

# --- CSV SETUP ---
output_folder = "/home/cvrr/Desktop/VLM Competition/doPlan/outputs"
os.makedirs(output_folder, exist_ok=True)

video_filename = os.path.basename(video_path)
csv_filename = os.path.splitext(video_filename)[0] + ".csv"
csv_path = os.path.join(output_folder, csv_filename)

# --- FUNCTIONS ---
def update_frame():
    """Update the video frame in the Tkinter window."""
    global paused, frame_image

    if not paused:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 360))  # Resize video to fixed size
            frame_image = ImageTk.PhotoImage(Image.fromarray(frame))
            video_label.config(image=frame_image)
            video_label.image = frame_image  # keep reference
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    root.after(30, update_frame)

def save_text():
    user_input = entry.get()
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([current_frame, user_input])

    entry.delete(0, tk.END)

def quit_program():
    cap.release()
    root.destroy()

def toggle_pause():
    global paused
    paused = not paused
    pause_button.config(text="Play" if paused else "Pause")

# --- TKINTER GUI ---
root = tk.Tk()
root.title("Video Labeler")
root.geometry("700x550")  # Set a fixed window size so everything fits

# Main frame
main_frame = tk.Frame(root)
main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Video frame
video_frame = tk.Frame(main_frame)
video_frame.pack()
video_label = tk.Label(video_frame)
video_label.pack()

# Entry box
entry = tk.Entry(main_frame, width=60)  # make it wider
entry.pack(pady=10)

# Button frame
button_frame = tk.Frame(main_frame)
button_frame.pack(pady=5)

save_button = tk.Button(button_frame, text="Save", command=save_text)
save_button.pack(side=tk.LEFT, padx=5)

pause_button = tk.Button(button_frame, text="Pause", command=toggle_pause)
pause_button.pack(side=tk.LEFT, padx=5)

quit_button = tk.Button(button_frame, text="Quit", command=quit_program)
quit_button.pack(side=tk.LEFT, padx=5)

# Start video updates
update_frame()
root.mainloop()
