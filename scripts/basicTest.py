import tkinter as tk
import csv

def save_text():
    user_input = entry.get()

    # Write to CSV file
    with open("doPlan/outputs/output.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([user_input])

    # Clear the entry box after saving
    entry.delete(0, tk.END)

def quit_program():
    root.destroy()

# Create main window
root = tk.Tk()
root.title("CSV Input")

# Text entry box
entry = tk.Entry(root, width=40)
entry.pack(padx=10, pady=10)

# Button frame
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

# Save button
save_button = tk.Button(button_frame, text="Save", command=save_text)
save_button.pack(side=tk.LEFT, padx=5)

# Quit button
quit_button = tk.Button(button_frame, text="Quit", command=quit_program)
quit_button.pack(side=tk.LEFT, padx=5)

# Start GUI loop
root.mainloop()
