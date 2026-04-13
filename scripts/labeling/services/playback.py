import os

import cv2
from PIL import Image, ImageTk


def release_caps(self):
    for cap in self.caps.values():
        try:
            cap.release()
        except Exception:
            pass
    self.caps = {}


def current_frame(self):
    if "F0" not in self.caps:
        return 0
    return int(self.caps["F0"].get(cv2.CAP_PROP_POS_FRAMES))


def set_all_caps_to_frame(self, frame_idx):
    for cap in self.caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


def resize_video_layout(self):
    if not hasattr(self, "video_board"):
        return

    board_width = max(960, self.video_board.winfo_width())
    board_height = max(520, self.video_board.winfo_height())
    horizontal_gap = 24
    vertical_gap = 24

    scale = min(
        (board_width - horizontal_gap) / self.layout_base_width,
        (board_height - vertical_gap) / self.layout_base_height,
    )
    scale = max(0.6, scale)

    for cam, spec in self.camera_layout.items():
        card_width = max(120, int(spec["size"][0] * scale) + 16)
        card_height = max(90, int(spec["size"][1] * scale) + 38)
        self.video_cards[cam].configure(width=card_width, height=card_height)


def resize_frame_to_fit(self, frame, target_width, target_height):
    source_height, source_width = frame.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height))

    canvas = Image.new("RGB", (target_width, target_height), "#f8fafb")
    image = Image.fromarray(resized)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas.paste(image, (offset_x, offset_y))
    return canvas


def redraw_current_frames(self):
    for cam, cap in self.caps.items():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        card_width = self.video_cards[cam].winfo_width()
        card_height = self.video_cards[cam].winfo_height()
        target_width = max(80, card_width - 16)
        target_height = max(60, card_height - 38)
        image = ImageTk.PhotoImage(
            self.resize_frame_to_fit(frame, target_width, target_height)
        )
        self.frame_images[cam] = image

        label = self.video_labels[cam]
        label.configure(image=image, text="")
        label.image = image


def load_base_name(self, base_name):
    self.release_caps()
    scenario_path = os.path.join(self.current_video_dir, base_name)

    new_caps = {}
    for camera_id in self.camera_ids:
        path = os.path.join(scenario_path, f"{camera_id}.mp4")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {path}")
        new_caps[camera_id] = cap

    total_frames = int(new_caps["F0"].get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < self.min_segment_length:
        for cap in new_caps.values():
            try:
                cap.release()
            except Exception:
                pass
        raise RuntimeError(
            f"Video '{base_name}' too short for minimum segment length of {self.min_segment_length} frames."
        )

    self.caps = new_caps
    self.current_base_name = base_name
    self.total_frames = total_frames
    self.csv_path = os.path.join(self.user_output_folder, base_name + ".csv")
    self.clip_var.set(self.get_scene_display_name(base_name))
    self.update_playback_state()
    self.update_status()


def load_clip_entry(self, video_set, base_name):
    self.set_selected_video_set(video_set)
    self.load_base_name(base_name)


def update_status(self):
    scene_id = self.get_scene_id(self.current_base_name or "", self.selected_video_set or "")
    scene_text = scene_id if scene_id else "-"
    session_text = (
        f"Set: {self.selected_video_set or '-'}   |   Scene: {scene_text}   |   User: {self.selected_username or '-'}"
    )
    frame_text = f"Frame {self.current_frame()} / {self.total_frames}"
    segment_text = (
        f"Segment {self.segment_start()} - {self.segment_end()}   |   "
        f"{self.playback_state_var.get()}   |   {self.speed_var.get()}   |   "
        f"Auto Next: {'On' if self.auto_next_var.get() else 'Off'}"
    )
    self.session_var.set(session_text)
    self.frame_var.set(frame_text)
    self.progress_var.set(segment_text)
    self.update_segment_progress()
    self.refresh_play_pause_button()


def refresh_play_pause_button(self):
    if not hasattr(self, "start_pause_button"):
        return
    if not self.app_started:
        self.start_pause_button.configure(text="Start")
        return
    self.start_pause_button.configure(text="Pause" if not self.paused else "Resume")


def update_playback_state(self, ended=False):
    if ended:
        self.playback_state_var.set("Clip ended")
    elif self.paused:
        self.playback_state_var.set("Paused")
    else:
        self.playback_state_var.set("Playing")


def update_speed_text(self):
    self.speed_var.set(f"Speed {self.playback_delay_var.get()} ms")


def update_segment_progress(self):
    segment_length = max(1, self.segment_end() - self.segment_start())
    progress_frames = max(0, self.current_frame() - self.segment_start())
    progress_percent = min(100.0, (progress_frames / segment_length) * 100.0)
    self.segment_progress_value.set(progress_percent)


def sync_speed_preset(self):
    current_delay = self.playback_delay_var.get()
    for label, delay in self.speed_options.items():
        if delay == current_delay:
            self.speed_preset_var.set(label)
            return
    self.speed_preset_var.set("Normal")


def on_speed_selected(self, event=None):
    selected = self.speed_preset_var.get()
    if selected in self.speed_options:
        self.playback_delay_var.set(self.speed_options[selected])
        self.update_speed_text()
        self.update_status()


def set_segment_start_frame(self):
    self.set_all_caps_to_frame(self.segment_start())
    self.redraw_current_frames()
    self.update_playback_state()
    self.update_status()


def random_segment(self):
    start = self.random.randint(0, self.total_frames - self.min_segment_length)
    end = self.random.randint(start + self.min_segment_length, self.total_frames - 1)
    self.set_segment_bounds(start, end)
    self.set_segment_start_frame()
    self.paused = False
    self.update_playback_state()
    self.update_status()


def load_random_base_and_segment(self, exclude_current=False):
    self.editing_history_entry = None
    valid_entries = self.get_valid_clip_entries()
    choices = valid_entries

    current_entry = (self.selected_video_set, self.current_base_name)
    if exclude_current and current_entry in valid_entries and len(valid_entries) > 1:
        choices = [entry for entry in valid_entries if entry != current_entry]

    non_skipped_choices = [entry for entry in choices if entry[1] not in self.skipped_base_names]
    if non_skipped_choices:
        choices = non_skipped_choices

    chosen_video_set, chosen_base_name = self.random.choice(choices)
    self.load_clip_entry(chosen_video_set, chosen_base_name)
    self.random_segment()
    self.clear_label_text()
    self.commentary_var.set(self.commentary_placeholder)
    self.update_label_field_state()
    self.focus_label_entry()


def load_starting_base_and_segment(self):
    self.editing_history_entry = None
    chosen = self.get_earliest_unlabeled_clip_entry()
    if chosen and chosen[1] in self.skipped_base_names:
        self.load_random_base_and_segment(exclude_current=False)
        return
    if not chosen:
        raise RuntimeError("No valid clips found to start labeling.")

    chosen_video_set, chosen_base_name = chosen
    self.load_clip_entry(chosen_video_set, chosen_base_name)
    self.random_segment()
    self.clear_label_text()
    self.commentary_var.set(self.commentary_placeholder)
    self.update_label_field_state()
    self.focus_label_entry()


def load_next_ordered_base_and_segment(self, include_skipped=False):
    self.editing_history_entry = None
    chosen = self.get_next_ordered_clip_entry(include_skipped=include_skipped)
    if not chosen:
        raise RuntimeError("No additional unlabeled clips are available.")

    chosen_video_set, chosen_base_name = chosen
    self.load_clip_entry(chosen_video_set, chosen_base_name)
    self.random_segment()
    self.clear_label_text()
    self.commentary_var.set(self.commentary_placeholder)
    self.update_label_field_state()
    self.focus_label_entry()
