import csv
import os
import tkinter as tk
from tkinter import messagebox


def get_history_index_path(self):
    if not self.user_output_folder:
        return None
    return os.path.join(self.user_output_folder, self.history_index_file)


def load_history_entries(self):
    self.history_entries = []
    history_path = self.get_history_index_path()
    if not history_path or not os.path.exists(history_path):
        self.refresh_history_listbox()
        return

    with open(history_path, "r", newline="", encoding="utf-8") as history_file:
        reader = csv.DictReader(history_file)
        for row in reader:
            self.history_entries.append(row)

    self.history_entries = list(reversed(self.history_entries))
    self.refresh_history_listbox()


def refresh_history_listbox(self):
    if not hasattr(self, "history_listbox"):
        return

    self.history_listbox.delete(0, tk.END)
    for entry in self.history_entries:
        display_name = self.get_scene_display_name(
            entry.get("base_name", ""),
            entry.get("video_set", ""),
        )
        label_text = entry.get("label", "")
        display = (
            f"{display_name} | "
            f"{entry.get('start_frame', '')}-{entry.get('end_frame', '')} | "
            f"{entry.get('commentary', '')} | {label_text}"
        )
        self.history_listbox.insert(tk.END, display)

    if hasattr(self, "delete_history_button"):
        self.delete_history_button.configure(state="disabled")


def write_history_entries(self):
    history_path = self.get_history_index_path()
    if not history_path:
        return

    rows = list(reversed(self.history_entries))
    with open(history_path, "w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=[
                "video_set",
                "base_name",
                "start_frame",
                "end_frame",
                "label",
                "commentary",
                "csv_path",
                "row_number",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def upsert_history_entry(self, label_text, commentary, row_number, csv_path=None):
    entry = {
        "video_set": self.selected_video_set or "",
        "base_name": self.current_base_name or "",
        "start_frame": str(self.segment_start()),
        "end_frame": str(self.segment_end()),
        "label": label_text,
        "commentary": commentary,
        "csv_path": csv_path if csv_path is not None else (self.csv_path or ""),
        "row_number": "" if row_number is None else str(row_number),
    }

    if self.editing_history_entry:
        for index, existing in enumerate(self.history_entries):
            if (
                existing.get("csv_path") == self.editing_history_entry.get("csv_path")
                and existing.get("row_number") == self.editing_history_entry.get("row_number")
            ):
                self.history_entries[index] = entry
                break
        else:
            self.history_entries.insert(0, entry)
    else:
        self.history_entries.insert(0, entry)

    self.editing_history_entry = None
    self.refresh_history_listbox()
    self.write_history_entries()


def read_csv_rows(self, csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        return list(csv.reader(csv_file))


def write_csv_rows(self, csv_path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(rows)


def on_history_selected(self, event=None):
    selection = self.history_listbox.curselection()
    if not selection:
        if hasattr(self, "delete_history_button"):
            self.delete_history_button.configure(state="disabled")
        return

    if hasattr(self, "delete_history_button"):
        self.delete_history_button.configure(state="normal")

    entry = self.history_entries[selection[0]]
    try:
        if entry.get("video_set"):
            self.set_selected_video_set(entry["video_set"])
            self.video_set_var.set(entry["video_set"])
        self.load_base_name(entry["base_name"])
        self.set_segment_bounds(int(entry["start_frame"]), int(entry["end_frame"]))
        self.set_segment_start_frame()
        self.paused = True
        self.update_playback_state()
        self.update_status()

        self.clear_label_text()
        saved_label = entry.get("label", "")
        if saved_label != self.no_label_token:
            self.label_entry.insert("1.0", saved_label)
        self.commentary_var.set(
            self.decode_commentary_value(entry.get("commentary", ""))
        )
        if entry.get("csv_path") and entry.get("row_number"):
            self.editing_history_entry = entry
        else:
            self.editing_history_entry = None
        self.update_label_field_state()
        self.focus_label_entry()
    except Exception as exc:
        messagebox.showerror("History Error", str(exc))


def add_annotation_from_selected_history(self):
    self.editing_history_entry = None
    self.history_listbox.selection_clear(0, tk.END)
    if hasattr(self, "delete_history_button"):
        self.delete_history_button.configure(state="disabled")
    self.focus_label_entry()


def delete_selected_history_entry(self):
    selection = self.history_listbox.curselection()
    if not selection:
        return

    entry = self.history_entries[selection[0]]
    should_delete = messagebox.askyesno(
        "Delete Annotation",
        "Delete the selected annotation from recent history?",
    )
    if not should_delete:
        return

    csv_path = entry.get("csv_path", "")
    row_number = entry.get("row_number", "")
    if csv_path and row_number and os.path.exists(csv_path):
        rows = self.read_csv_rows(csv_path)
        row_index = int(row_number)
        if 0 < row_index < len(rows):
            del rows[row_index]
            self.write_csv_rows(csv_path, rows)
            for existing in self.history_entries:
                if existing is entry:
                    continue
                if existing.get("csv_path") != csv_path:
                    continue
                existing_row = existing.get("row_number", "")
                if existing_row and int(existing_row) > row_index:
                    existing["row_number"] = str(int(existing_row) - 1)

    del self.history_entries[selection[0]]
    self.editing_history_entry = None
    self.clear_label_text()
    self.commentary_var.set(self.commentary_placeholder)
    self.update_label_field_state()
    self.write_history_entries()
    self.refresh_history_listbox()
    self.update_welcome_totals()
