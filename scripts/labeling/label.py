import csv
import os
import random
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk
from scene_assigner import run_assignment
from services.catalog import (
    focus_welcome_history as svc_focus_welcome_history,
    get_earliest_unlabeled_clip_entry as svc_get_earliest_unlabeled_clip_entry,
    get_global_annotation_history as svc_get_global_annotation_history,
    get_next_ordered_clip_entry as svc_get_next_ordered_clip_entry,
    get_ordered_clip_entries as svc_get_ordered_clip_entries,
    get_scene_display_name as svc_get_scene_display_name,
    get_scene_id as svc_get_scene_id,
    get_selected_username_candidate as svc_get_selected_username_candidate,
    get_valid_base_names as svc_get_valid_base_names,
    get_valid_clip_entries as svc_get_valid_clip_entries,
    get_video_set_for_base_name as svc_get_video_set_for_base_name,
    has_saved_labels_for_base as svc_has_saved_labels_for_base,
    on_user_selection_changed as svc_on_user_selection_changed,
    refresh_choices as svc_refresh_choices,
    set_selected_username as svc_set_selected_username,
    set_selected_video_set as svc_set_selected_video_set,
    update_welcome_stats as svc_update_welcome_stats,
    update_welcome_totals as svc_update_welcome_totals,
)
from services.history import (
    add_annotation_from_selected_history as svc_add_annotation_from_selected_history,
    delete_selected_history_entry as svc_delete_selected_history_entry,
    get_history_index_path as svc_get_history_index_path,
    load_history_entries as svc_load_history_entries,
    on_history_selected as svc_on_history_selected,
    read_csv_rows as svc_read_csv_rows,
    refresh_history_listbox as svc_refresh_history_listbox,
    upsert_history_entry as svc_upsert_history_entry,
    write_csv_rows as svc_write_csv_rows,
    write_history_entries as svc_write_history_entries,
)
from services.playback import (
    current_frame as svc_current_frame,
    load_base_name as svc_load_base_name,
    load_clip_entry as svc_load_clip_entry,
    load_next_ordered_base_and_segment as svc_load_next_ordered_base_and_segment,
    load_random_base_and_segment as svc_load_random_base_and_segment,
    load_starting_base_and_segment as svc_load_starting_base_and_segment,
    random_segment as svc_random_segment,
    redraw_current_frames as svc_redraw_current_frames,
    refresh_play_pause_button as svc_refresh_play_pause_button,
    release_caps as svc_release_caps,
    resize_frame_to_fit as svc_resize_frame_to_fit,
    resize_video_layout as svc_resize_video_layout,
    set_all_caps_to_frame as svc_set_all_caps_to_frame,
    set_segment_start_frame as svc_set_segment_start_frame,
    sync_speed_preset as svc_sync_speed_preset,
    update_playback_state as svc_update_playback_state,
    update_segment_progress as svc_update_segment_progress,
    update_speed_text as svc_update_speed_text,
    update_status as svc_update_status,
    on_speed_selected as svc_on_speed_selected,
)
from ui.label_panel import build_label_panel
from ui.top_controls import build_top_controls
from ui.video_board import build_video_board
from ui.welcome_screen import build_welcome_screen


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.txt")
HISTORY_INDEX_FILE = "_history_index.csv"
SCENE_ASSIGNMENTS_FILE = os.path.join(os.path.dirname(__file__), "scene_assignments.txt")
SPEED_OPTIONS = {
    "Slower": 60,
    "Slow": 45,
    "Normal": 33,
    "Fast": 24,
    "Faster": 16,
}
NO_LABEL_TOKEN = "[NO_LABEL_NEEDED]"
COMMENTARY_PLACEHOLDER = "Select commentary tag"
NEW_USER_OPTION = "New User"
COMMENTARY_CODE_MAP = {
    "Non-Referential": "",
    "Static Referential": "(s)",
    "Dynamic Referential": "(d)",
    "Both": "ds",
}
LAYOUT_BASE_WIDTH = 240 + 680 + 240
LAYOUT_BASE_HEIGHT = 382 + 186

if not os.path.exists(SETTINGS_FILE):
    raise FileNotFoundError(f"{SETTINGS_FILE} not found!")


def load_settings():
    video_dir = ""
    min_segment_length = 150
    output_folder = os.path.join(BASE_DIR, "labeling", "output")
    commentary_options = [
        "Non-Referential",
        "Static Referential",
        "Dynamic Referential",
        "Both",
    ]

    with open(SETTINGS_FILE, "r", encoding="utf-8") as settings_file:
        for raw_line in settings_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key == "video_dir":
                video_dir = value
            elif key == "min_segment_length":
                min_segment_length = int(value)
            elif key == "output_folder":
                output_folder = value
            elif key == "commentary_options":
                commentary_options = [v.strip() for v in value.split(",") if v.strip()]

    if not video_dir:
        raise ValueError("video_dir must be set in settings.txt")

    return video_dir, min_segment_length, output_folder, commentary_options


VIDEO_ROOT_DIR, MIN_SEGMENT_LENGTH, OUTPUT_FOLDER, COMMENTARY_OPTIONS = load_settings()
CAMERA_IDS = ["L0", "L1", "L2", "F0", "B0", "R0", "R1", "R2"]
CAMERA_LAYOUT = {
    "L0": {"row": 0, "column": 0, "size": (240, 135)},
    "L1": {"row": 1, "column": 0, "size": (240, 135)},
    "L2": {"row": 2, "column": 0, "size": (240, 135)},
    "F0": {"row": 0, "column": 1, "rowspan": 2, "size": (680, 382)},
    "B0": {"row": 2, "column": 1, "size": (330, 186)},
    "R0": {"row": 0, "column": 2, "size": (240, 135)},
    "R1": {"row": 1, "column": 2, "size": (240, 135)},
    "R2": {"row": 2, "column": 2, "size": (240, 135)},
}


def sanitize_username(name):
    cleaned = name.strip()
    for invalid_char in '<>:"/\\|?*':
        cleaned = cleaned.replace(invalid_char, "_")
    return cleaned


def get_existing_usernames():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    names = [
        name
        for name in os.listdir(OUTPUT_FOLDER)
        if os.path.isdir(os.path.join(OUTPUT_FOLDER, name))
    ]
    return sorted(names, key=str.lower)


def get_available_video_sets():
    if not os.path.isdir(VIDEO_ROOT_DIR):
        raise NotADirectoryError(f"video_dir is not a valid directory: {VIDEO_ROOT_DIR}")

    names = [
        name
        for name in os.listdir(VIDEO_ROOT_DIR)
        if os.path.isdir(os.path.join(VIDEO_ROOT_DIR, name))
    ]
    return sorted(names, key=str.lower)


def load_scene_assignments():
    assignments = {}
    if not os.path.exists(SCENE_ASSIGNMENTS_FILE):
        return assignments

    with open(SCENE_ASSIGNMENTS_FILE, "r", encoding="utf-8") as assignment_file:
        for raw_line in assignment_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) != 3:
                continue

            video_set, base_name, scene_id = parts
            assignments[(video_set, base_name)] = scene_id

    return assignments


def scene_sort_key(scene_id):
    if scene_id and scene_id.startswith("scene-"):
        suffix = scene_id.split("-", 1)[1]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, scene_id or "")


def resolve_video_set_dir(video_root_dir, video_set):
    set_dir = os.path.join(video_root_dir, video_set)
    if not os.path.isdir(set_dir):
        return set_dir

    child_dirs = [
        name
        for name in os.listdir(set_dir)
        if os.path.isdir(os.path.join(set_dir, name))
    ]
    nested_dir = os.path.join(set_dir, video_set)
    if child_dirs == [video_set] and os.path.isdir(nested_dir):
        return nested_dir

    return set_dir


class SimpleModernLabeler:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MI3 Labeler")
        self.root.geometry("1580x980")
        self.root.minsize(1380, 860)
        self.root.configure(bg="#f3f5f7")
        self.random = random

        self.caps = {}
        self.frame_images = {}
        self.video_labels = {}
        self.video_cards = {}
        run_assignment(verbose=False)
        self.scene_assignments = load_scene_assignments()
        self.base_name_to_video_set = {}
        self.current_base_name = None
        self.total_frames = 0
        self.csv_path = None
        self.selected_username = None
        self.user_output_folder = None
        self.selected_video_set = None
        self.current_video_dir = VIDEO_ROOT_DIR
        self.video_root_dir = VIDEO_ROOT_DIR
        self.output_folder = OUTPUT_FOLDER
        self.history_index_file = HISTORY_INDEX_FILE
        self.new_user_option = NEW_USER_OPTION
        self.commentary_placeholder = COMMENTARY_PLACEHOLDER
        self.no_label_token = NO_LABEL_TOKEN
        self.camera_ids = CAMERA_IDS
        self.camera_layout = CAMERA_LAYOUT
        self.layout_base_width = LAYOUT_BASE_WIDTH
        self.layout_base_height = LAYOUT_BASE_HEIGHT
        self.min_segment_length = MIN_SEGMENT_LENGTH
        self.speed_options = SPEED_OPTIONS
        self.sanitize_username = sanitize_username
        self.valid_base_names = []
        self.skipped_base_names = set()
        self.history_entries = []
        self.editing_history_entry = None
        self.paused = True
        self.app_started = False
        self._segment_start = 0
        self._segment_end = 0

        self.video_set_var = tk.StringVar(value="")
        self.existing_user_var = tk.StringVar(value="")
        self.new_user_var = tk.StringVar(value="")
        self.label_text_var = tk.StringVar(value="")
        self.commentary_var = tk.StringVar(value=COMMENTARY_PLACEHOLDER)
        self.auto_next_var = tk.BooleanVar(value=False)
        self.confirm_skip_var = tk.BooleanVar(value=True)
        self.jump_var = tk.IntVar(value=100)
        self.playback_delay_var = tk.IntVar(value=SPEED_OPTIONS["Normal"])
        self.playback_state_var = tk.StringVar(value="Paused")
        self.speed_var = tk.StringVar(value=f"Speed {SPEED_OPTIONS['Normal']} ms")
        self.speed_preset_var = tk.StringVar(value="Normal")
        self.segment_progress_value = tk.DoubleVar(value=0.0)

        self.session_var = tk.StringVar(value="Choose a set and user, then start.")
        self.clip_var = tk.StringVar(value="No clip loaded")
        self.frame_var = tk.StringVar(value="Frame 0 / 0")
        self.progress_var = tk.StringVar(value="Segment 0 - 0")
        self.welcome_totals_var = tk.StringVar(
            value="Total users: 0\nLabels created: 0\nHistory entries shown: 0"
        )
        self.welcome_history_button_var = tk.StringVar(value="Show Annotation History")
        self.welcome_stats_var = tk.StringVar(
            value="Select a user to see progress, then continue into labeling."
        )
        self.hint_var = tk.StringVar(
            value=(
                "Taxi test: write what you would tell the driver to make the scene happen. "
                "If nothing needs to be said, use No Label Needed."
            )
        )

        self.configure_style()
        self.build_ui()
        self.refresh_choices()
        self.sync_speed_preset()
        self.update_label_field_state()
        self.root.after_idle(self.resize_video_layout)
        self.existing_user_var.trace_add("write", self.update_welcome_stats)
        self.new_user_var.trace_add("write", self.update_welcome_stats)
        self.existing_user_var.trace_add("write", self.on_user_selection_changed)

        self.root.protocol("WM_DELETE_WINDOW", self.quit_program)
        self.root.bind("<space>", self.on_spacebar)
        self.root.bind("<p>", self.on_p_key)
        self.root.bind("<P>", self.on_p_key)
        self.root.bind("<Return>", self.on_return_key)
        self.root.bind("<Configure>", self.on_window_resize)

        self.update_frames()

    def configure_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        bg = "#f4f6f8"
        panel = "#ffffff"
        border = "#d6dce3"
        text = "#18212b"
        muted = "#5f6874"
        accent = "#0a7c6b"
        accent_active = "#086556"
        soft = "#eef2f5"
        strong = "#102033"

        self.colors = {
            "bg": bg,
            "panel": panel,
            "border": border,
            "text": text,
            "muted": muted,
            "accent": accent,
            "accent_active": accent_active,
            "soft": soft,
            "strong": strong,
        }

        style.configure("App.TFrame", background=bg)
        style.configure("Panel.TFrame", background=panel)
        style.configure(
            "Title.TLabel",
            background=bg,
            foreground=strong,
            font=("Aptos Display", 24, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background=bg,
            foreground=muted,
            font=("Aptos", 11),
        )
        style.configure(
            "PanelTitle.TLabel",
            background=panel,
            foreground=strong,
            font=("Aptos", 12, "bold"),
        )
        style.configure(
            "PanelText.TLabel",
            background=panel,
            foreground=muted,
            font=("Aptos", 11),
        )
        style.configure(
            "Info.TLabel",
            background=panel,
            foreground=text,
            font=("Aptos", 12),
        )
        style.configure(
            "Primary.TButton",
            background=accent,
            foreground="#ffffff",
            padding=(14, 9),
            borderwidth=0,
            font=("Aptos", 11, "bold"),
        )
        style.map("Primary.TButton", background=[("active", accent_active)])
        style.configure(
            "Secondary.TButton",
            background=soft,
            foreground=text,
            padding=(12, 8),
            borderwidth=0,
            font=("Aptos", 11),
        )
        style.map("Secondary.TButton", background=[("active", "#dde6eb")])
        style.configure(
            "Playback.TButton",
            background=strong,
            foreground="#ffffff",
            padding=(22, 14),
            borderwidth=0,
            font=("Aptos", 14, "bold"),
        )
        style.map("Playback.TButton", background=[("active", "#1d2d45")])
        style.configure(
            "App.TEntry",
            fieldbackground="#ffffff",
            foreground=text,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            padding=8,
            font=("Aptos", 12),
        )
        style.map(
            "App.TEntry",
            fieldbackground=[("disabled", "#e6ebf0")],
            foreground=[("disabled", muted)],
        )
        style.configure(
            "App.TCombobox",
            fieldbackground="#ffffff",
            background="#ffffff",
            foreground=text,
            bordercolor=border,
            arrowcolor=muted,
            padding=6,
            font=("Aptos", 11),
        )
        style.map(
            "App.TCombobox",
            fieldbackground=[("readonly", "#ffffff")],
            foreground=[("readonly", text)],
            selectbackground=[("readonly", "#ffffff")],
            selectforeground=[("readonly", text)],
        )
        style.configure(
            "Invalid.TCombobox",
            fieldbackground="#fff1f1",
            background="#fff1f1",
            foreground=text,
            bordercolor="#cc4b4b",
            arrowcolor="#cc4b4b",
            padding=6,
            font=("Aptos", 11),
        )
        style.map(
            "Invalid.TCombobox",
            fieldbackground=[("readonly", "#fff1f1")],
            foreground=[("readonly", text)],
            selectbackground=[("readonly", "#fff1f1")],
            selectforeground=[("readonly", text)],
        )
        style.configure(
            "App.TCheckbutton",
            background=panel,
            foreground=text,
            font=("Aptos", 11),
        )
        style.configure(
            "App.Horizontal.TProgressbar",
            troughcolor=soft,
            bordercolor=soft,
            background=accent,
            lightcolor=accent,
            darkcolor=accent,
        )

    def build_ui(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.outer = ttk.Frame(self.root, style="App.TFrame", padding=20)
        self.outer.grid(sticky="nsew")
        self.outer.grid_columnconfigure(0, weight=1)
        self.outer.grid_rowconfigure(3, weight=1)
        build_welcome_screen(self)
        build_top_controls(self)
        build_label_panel(self, COMMENTARY_PLACEHOLDER, COMMENTARY_OPTIONS, SPEED_OPTIONS)

        build_video_board(self, CAMERA_LAYOUT)

        self.bottom_progress = ttk.Progressbar(
            self.outer,
            style="App.Horizontal.TProgressbar",
            variable=self.segment_progress_value,
            maximum=100,
        )
        self.bottom_progress.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        self.bottom_progress.grid_remove()

    def encode_commentary_value(self, commentary_label):
        cleaned = (commentary_label or "").strip()
        if cleaned in COMMENTARY_CODE_MAP:
            return COMMENTARY_CODE_MAP[cleaned]
        if cleaned in COMMENTARY_CODE_MAP.values():
            return cleaned
        if cleaned == COMMENTARY_PLACEHOLDER:
            return ""
        return cleaned

    def decode_commentary_value(self, commentary_value):
        cleaned = (commentary_value or "").strip()
        reverse_map = {value: key for key, value in COMMENTARY_CODE_MAP.items()}
        if cleaned in reverse_map:
            return reverse_map[cleaned]
        if cleaned in COMMENTARY_CODE_MAP:
            return cleaned
        return COMMENTARY_PLACEHOLDER

    def get_label_text(self):
        return self.label_entry.get("1.0", "end-1c").strip()

    def clear_label_text(self):
        self.label_entry.delete("1.0", tk.END)

    def focus_label_entry(self):
        self.label_entry.focus_set()

    def has_commentary_selection(self):
        commentary = self.commentary_var.get().strip()
        return bool(commentary) and commentary != COMMENTARY_PLACEHOLDER

    def validate_required_fields(self, show_message=True, focus_label=True, allow_no_label=False):
        if not allow_no_label and not self.has_label_text():
            if show_message and self.app_started:
                messagebox.showwarning(
                    "Label Required",
                    "Enter a label or use the 'No Label Needed' button before moving to the next clip.",
                )
            if focus_label:
                self.focus_label_entry()
            return False

        if not allow_no_label and not self.has_commentary_selection():
            if show_message and self.app_started:
                messagebox.showwarning(
                    "Commentary Required",
                    "Select a commentary tag before saving or moving to the next clip.",
                )
            self.commentary_combo.focus_set()
            return False

        return True

    def update_label_field_state(self):
        has_text = self.has_label_text()
        has_commentary = self.has_commentary_selection()
        can_add_another = self.app_started and self.current_base_name is not None
        if has_text:
            self.label_entry.configure(
                bg="#ffffff",
                highlightbackground=self.colors["border"],
                highlightcolor=self.colors["accent"],
            )
        else:
            self.label_entry.configure(
                bg="#fff1f1",
                highlightbackground="#cc4b4b",
                highlightcolor="#cc4b4b",
            )

        if hasattr(self, "commentary_combo"):
            self.commentary_combo.configure(
                style="App.TCombobox" if has_commentary else "Invalid.TCombobox"
            )

        if hasattr(self, "save_button"):
            state = "normal" if has_text and has_commentary else "disabled"
            self.save_button.configure(state=state)
            self.save_next_button.configure(state=state)
        if hasattr(self, "add_history_annotation_button"):
            self.add_history_annotation_button.configure(
                state="normal" if can_add_another else "disabled"
            )

    def on_label_text_changed(self, event=None):
        self.update_label_field_state()

    def on_commentary_changed(self, event=None):
        self.update_label_field_state()

    def has_label_text(self):
        return bool(self.get_label_text())

    def ensure_label_before_advance(self, show_message=True):
        return self.validate_required_fields(show_message=show_message)

    def show_commentary_help(self):
        messagebox.showinfo(
            "Commentary Help",
            "Choose the tag based on what your instruction refers to.\n\n"
            "Non-Referential: no object reference, for example 'turn right here' or 'slow down'. Saves as blank.\n"
            "Static Referential: refers to a fixed thing, for example 'stop at the sign'. Saves as (s).\n"
            "Dynamic Referential: refers to a moving agent, for example 'follow that white car'. Saves as (d).\n"
            "Both: includes both kinds of references. Saves as ds.",
        )

    def show_label_help(self):
        messagebox.showinfo(
            "Label Help",
            "Write the instruction that would cause the behavior in the clip.\n\n"
            "Use the taxi test: what would you tell the driver to make this happen?\n"
            "Good: 'turn right here', 'slow down for the pedestrian', 'follow that white car'.\n"
            "Not the goal: describing the clip like 'the car turns right'.\n"
            "If the car is just driving normally and no guidance is needed, use 'No Label Needed'.\n"
            "Only use 'go straight' when there is an actual choice in the road and straight is the intended path.",
        )

    def segment_start(self):
        return self._segment_start

    def segment_end(self):
        return self._segment_end

    def set_segment_bounds(self, start, end):
        self._segment_start = start
        self._segment_end = end
        self.progress_var.set(f"Segment {start} - {end}")

    def start_or_toggle(self):
        if not self.app_started:
            self.start_labeling()
        else:
            self.toggle_pause()

    def restart_session(self):
        if self.app_started:
            self.replay_segment()
            self.update_playback_state()
            self.update_status()
        else:
            self.refresh_choices()
            self.update_status()

    def start_labeling(self):
        existing_choice = self.existing_user_var.get().strip()
        new_user = self.new_user_var.get().strip()

        if existing_choice == NEW_USER_OPTION:
            if new_user:
                final_user = new_user
            else:
                messagebox.showerror("Missing Username", "Enter a name after choosing 'New User'.")
                self.new_user_entry.focus_set()
                return
        elif existing_choice:
            final_user = existing_choice
        elif new_user:
            final_user = new_user
        else:
            messagebox.showerror("Missing Username", "Pick an existing username or enter a new one.")
            return

        try:
            self.set_selected_username(final_user)
            self.load_starting_base_and_segment()
        except Exception as exc:
            messagebox.showerror("Startup Error", str(exc))
            return

        self.app_started = True
        self.skipped_base_names.clear()
        self.paused = False
        self.load_history_entries()
        self.welcome_frame.grid_remove()
        self.main_header.grid()
        self.setup_panel.grid()
        self.label_panel.grid()
        self.info_panel.grid()
        self.bottom_progress.grid()
        self.update_playback_state()
        self.update_status()
        self.focus_label_entry()

    def toggle_pause(self):
        if not self.app_started:
            return
        self.paused = not self.paused
        self.update_playback_state()
        self.update_status()

    def replay_segment(self):
        if not self.app_started:
            return
        self.set_segment_start_frame()
        self.paused = False
        self.update_playback_state()
        self.update_status()

    def slow_down_video(self):
        self.playback_delay_var.set(min(300, self.playback_delay_var.get() + 10))
        self.update_speed_text()
        self.sync_speed_preset()
        self.update_status()

    def speed_up_video(self):
        self.playback_delay_var.set(max(1, self.playback_delay_var.get() - 10))
        self.update_speed_text()
        self.sync_speed_preset()
        self.update_status()

    def reset_video_speed(self):
        self.playback_delay_var.set(SPEED_OPTIONS["Normal"])
        self.update_speed_text()
        self.sync_speed_preset()
        self.update_status()

    def next_random_scenario(self):
        if not self.app_started:
            return
        if not self.ensure_label_before_advance():
            return
        self.load_next_ordered_base_and_segment()

    def skip_current_clip(self):
        if not self.app_started or self.current_base_name is None:
            return
        if self.confirm_skip_var.get():
            should_skip = messagebox.askyesno(
                "Confirm Skip",
                "Skip this clip and move to the next scene without saving a label?",
            )
            if not should_skip:
                return
        self.skipped_base_names.add(self.current_base_name)
        self.load_next_ordered_base_and_segment()

    def submit_no_label_and_next(self):
        if not self.app_started:
            return
        if self.save_segment_label(force_no_label=True):
            self.load_next_ordered_base_and_segment()

    def save_segment_label(self, force_no_label=False):
        if self.current_base_name is None or self.selected_username is None:
            return False

        label_text = NO_LABEL_TOKEN if force_no_label else self.get_label_text()
        if not label_text:
            messagebox.showwarning(
                "Label Required",
                "Enter a label or use the 'No Label Needed' button before saving this clip.",
            )
            self.focus_label_entry()
            return False
        commentary = "" if force_no_label else self.encode_commentary_value(self.commentary_var.get())
        if not self.validate_required_fields(
            show_message=True,
            focus_label=not force_no_label,
            allow_no_label=force_no_label,
        ):
            return False
        start_frame = self.segment_start()
        end_frame = self.segment_end()

        file_exists = os.path.exists(self.csv_path)
        file_empty = (not file_exists) or (os.path.getsize(self.csv_path) == 0)
        row_number = None

        if self.editing_history_entry and os.path.exists(self.csv_path):
            rows = self.read_csv_rows(self.csv_path)
            row_number = int(self.editing_history_entry["row_number"])
            rows[row_number] = [
                self.current_base_name,
                start_frame,
                end_frame,
                label_text,
                commentary,
                self.selected_username,
            ]
            self.write_csv_rows(self.csv_path, rows)
        else:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                if file_empty:
                    writer.writerow(
                        ["base_name", "start_frame", "end_frame", "label", "commentary", "username"]
                    )
                    row_number = 1
                else:
                    with open(self.csv_path, "r", newline="", encoding="utf-8") as read_file:
                        row_number = len(list(csv.reader(read_file)))
                writer.writerow(
                    [
                        self.current_base_name,
                        start_frame,
                        end_frame,
                        label_text,
                        commentary,
                        self.selected_username,
                    ]
                )

        self.upsert_history_entry(label_text, commentary, row_number)

        self.clear_label_text()
        self.commentary_var.set(COMMENTARY_PLACEHOLDER)
        self.update_label_field_state()
        self.focus_label_entry()
        self.update_welcome_totals()
        return True

    def save_and_next(self):
        if not self.validate_required_fields(show_message=True):
            return
        if self.save_segment_label():
            self.load_next_ordered_base_and_segment()

    def update_frames(self):
        if self.app_started and not self.paused:
            self.redraw_current_frames()

            if self.current_frame() >= self.segment_end():
                if self.auto_next_var.get() and self.has_label_text() and self.has_commentary_selection():
                    self.save_and_next()
                else:
                    self.paused = True
                    self.update_playback_state(ended=True)

        if self.app_started:
            self.update_status()

        self.root.after(self.playback_delay_var.get(), self.update_frames)

    def entry_has_focus(self):
        focused = self.root.focus_get()
        return focused in {self.label_entry, self.new_user_entry}

    def on_spacebar(self, event=None):
        if not self.app_started or self.entry_has_focus():
            return
        self.next_random_scenario()

    def on_p_key(self, event=None):
        if not self.app_started or self.entry_has_focus():
            return
        self.toggle_pause()

    def on_return_key(self, event=None):
        focused = self.root.focus_get()
        if focused == self.new_user_entry and not self.app_started:
            self.start_labeling()
            return "break"
        elif focused == self.label_entry and self.app_started:
            self.save_and_next()
            return "break"

    def on_window_resize(self, event=None):
        if event is not None and event.widget is not self.root:
            return
        self.resize_video_layout()
        if self.app_started and self.caps:
            self.set_all_caps_to_frame(self.current_frame())
            self.redraw_current_frames()
            self.update_status()

    def quit_program(self):
        self.release_caps()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


SimpleModernLabeler.refresh_choices = svc_refresh_choices
SimpleModernLabeler.get_selected_username_candidate = svc_get_selected_username_candidate
SimpleModernLabeler.on_user_selection_changed = svc_on_user_selection_changed
SimpleModernLabeler.update_welcome_stats = svc_update_welcome_stats
SimpleModernLabeler.get_global_annotation_history = svc_get_global_annotation_history
SimpleModernLabeler.update_welcome_totals = svc_update_welcome_totals
SimpleModernLabeler.focus_welcome_history = svc_focus_welcome_history

SimpleModernLabeler.get_history_index_path = svc_get_history_index_path
SimpleModernLabeler.load_history_entries = svc_load_history_entries
SimpleModernLabeler.refresh_history_listbox = svc_refresh_history_listbox
SimpleModernLabeler.write_history_entries = svc_write_history_entries
SimpleModernLabeler.upsert_history_entry = svc_upsert_history_entry
SimpleModernLabeler.read_csv_rows = svc_read_csv_rows
SimpleModernLabeler.write_csv_rows = svc_write_csv_rows
SimpleModernLabeler.on_history_selected = svc_on_history_selected
SimpleModernLabeler.add_annotation_from_selected_history = svc_add_annotation_from_selected_history
SimpleModernLabeler.delete_selected_history_entry = svc_delete_selected_history_entry

SimpleModernLabeler.set_selected_video_set = svc_set_selected_video_set
SimpleModernLabeler.set_selected_username = svc_set_selected_username
SimpleModernLabeler.get_video_set_for_base_name = svc_get_video_set_for_base_name
SimpleModernLabeler.get_scene_display_name = svc_get_scene_display_name
SimpleModernLabeler.get_scene_id = svc_get_scene_id
SimpleModernLabeler.has_saved_labels_for_base = svc_has_saved_labels_for_base
SimpleModernLabeler.get_valid_clip_entries = svc_get_valid_clip_entries
SimpleModernLabeler.get_earliest_unlabeled_clip_entry = svc_get_earliest_unlabeled_clip_entry
SimpleModernLabeler.get_ordered_clip_entries = svc_get_ordered_clip_entries
SimpleModernLabeler.get_next_ordered_clip_entry = svc_get_next_ordered_clip_entry
SimpleModernLabeler.get_valid_base_names = svc_get_valid_base_names

SimpleModernLabeler.release_caps = svc_release_caps
SimpleModernLabeler.current_frame = svc_current_frame
SimpleModernLabeler.set_all_caps_to_frame = svc_set_all_caps_to_frame
SimpleModernLabeler.resize_video_layout = svc_resize_video_layout
SimpleModernLabeler.resize_frame_to_fit = svc_resize_frame_to_fit
SimpleModernLabeler.redraw_current_frames = svc_redraw_current_frames
SimpleModernLabeler.load_base_name = svc_load_base_name
SimpleModernLabeler.load_clip_entry = svc_load_clip_entry
SimpleModernLabeler.update_status = svc_update_status
SimpleModernLabeler.refresh_play_pause_button = svc_refresh_play_pause_button
SimpleModernLabeler.update_playback_state = svc_update_playback_state
SimpleModernLabeler.update_speed_text = svc_update_speed_text
SimpleModernLabeler.update_segment_progress = svc_update_segment_progress
SimpleModernLabeler.sync_speed_preset = svc_sync_speed_preset
SimpleModernLabeler.on_speed_selected = svc_on_speed_selected
SimpleModernLabeler.set_segment_start_frame = svc_set_segment_start_frame
SimpleModernLabeler.random_segment = svc_random_segment
SimpleModernLabeler.load_next_ordered_base_and_segment = svc_load_next_ordered_base_and_segment
SimpleModernLabeler.load_random_base_and_segment = svc_load_random_base_and_segment
SimpleModernLabeler.load_starting_base_and_segment = svc_load_starting_base_and_segment


if __name__ == "__main__":
    app = SimpleModernLabeler()
    app.run()
