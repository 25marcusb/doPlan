import csv
import os


def _scene_sort_key(scene_id):
    if scene_id and scene_id.startswith("scene-"):
        suffix = scene_id.split("-", 1)[1]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, scene_id or "")


def _resolve_video_set_dir(video_root_dir, video_set):
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


def _get_existing_usernames(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    names = [
        name
        for name in os.listdir(output_folder)
        if os.path.isdir(os.path.join(output_folder, name))
    ]
    return sorted(names, key=str.lower)


def _get_available_video_sets(video_root_dir):
    if not os.path.isdir(video_root_dir):
        raise NotADirectoryError(f"video_dir is not a valid directory: {video_root_dir}")

    names = [
        name
        for name in os.listdir(video_root_dir)
        if os.path.isdir(os.path.join(video_root_dir, name))
    ]
    return sorted(names, key=str.lower)


def refresh_choices(self):
    users = _get_existing_usernames(self.output_folder)
    user_choices = [*users, self.new_user_option]
    self.existing_user_combo["values"] = user_choices

    if self.existing_user_var.get() not in user_choices:
        self.existing_user_var.set(users[0] if users else self.new_user_option)

    self.on_user_selection_changed()
    self.update_welcome_stats()
    self.update_welcome_totals()


def get_selected_username_candidate(self):
    if self.existing_user_var.get().strip() == self.new_user_option:
        return self.sanitize_username(self.new_user_var.get().strip())
    return self.sanitize_username(self.existing_user_var.get().strip())


def on_user_selection_changed(self, *args):
    selected = self.existing_user_var.get().strip()
    is_new_user = selected == self.new_user_option
    self.new_user_entry.configure(state="normal" if is_new_user else "disabled")
    if not is_new_user and self.new_user_var.get():
        self.new_user_var.set("")


def update_welcome_stats(self, *args):
    username = self.get_selected_username_candidate()
    if not username:
        self.welcome_stats_var.set(
            "Enter or choose a name. Last scene: none yet. Labels created: 0."
        )
        return

    user_output_folder = os.path.join(self.output_folder, username)
    if not os.path.isdir(user_output_folder):
        self.welcome_stats_var.set(
            f"User: {username}\nLast scene: none yet\nLabels created: 0"
        )
        return

    labeled_scenes = []
    label_count = 0
    for filename in os.listdir(user_output_folder):
        if not filename.endswith(".csv") or filename == self.history_index_file:
            continue
        csv_path = os.path.join(user_output_folder, filename)
        if os.path.getsize(csv_path) == 0:
            continue
        with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
            rows = list(csv.reader(csv_file))
        if len(rows) > 1:
            labeled_scenes.append(filename[:-4])
            label_count += len(rows) - 1

    if not labeled_scenes:
        self.welcome_stats_var.set(
            f"User: {username}\nLast scene: none yet\nLabels created: 0"
        )
        return

    labeled_scenes_sorted = sorted(
        labeled_scenes,
        key=lambda base_name: (
            _scene_sort_key(self.get_scene_id(base_name, self.get_video_set_for_base_name(base_name))),
            base_name.lower(),
        ),
    )
    last_scene = self.get_scene_display_name(
        labeled_scenes_sorted[-1],
        self.get_video_set_for_base_name(labeled_scenes_sorted[-1]),
    )
    self.welcome_stats_var.set(
        f"User: {username}\nLast scene: {last_scene}\nLabels created: {label_count}"
    )


def get_global_annotation_history(self, limit=200):
    entries = []
    for username in _get_existing_usernames(self.output_folder):
        history_path = os.path.join(self.output_folder, username, self.history_index_file)
        if not os.path.exists(history_path):
            continue
        with open(history_path, "r", newline="", encoding="utf-8") as history_file:
            reader = csv.DictReader(history_file)
            for row in reader:
                row["_username"] = username
                entries.append(row)

    entries.sort(
        key=lambda row: (
            row.get("video_set", ""),
            row.get("base_name", ""),
            int(row.get("row_number") or 0),
        ),
        reverse=True,
    )
    return entries[:limit]


def update_welcome_totals(self):
    users = _get_existing_usernames(self.output_folder)
    label_count = 0

    for username in users:
        user_dir = os.path.join(self.output_folder, username)
        for filename in os.listdir(user_dir):
            if not filename.endswith(".csv") or filename == self.history_index_file:
                continue
            csv_path = os.path.join(user_dir, filename)
            with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
                label_count += max(0, sum(1 for _ in csv.reader(csv_file)) - 1)

    history_entries = self.get_global_annotation_history()
    self.welcome_totals_var.set(
        f"Total users: {len(users)}\n"
        f"Labels created: {label_count}\n"
        f"History entries shown: {len(history_entries)}"
    )

    if not hasattr(self, "welcome_history_listbox"):
        return

    self.welcome_history_listbox.delete(0, "end")
    for entry in history_entries:
        display_name = self.get_scene_display_name(
            entry.get("base_name", ""),
            entry.get("video_set", ""),
        )
        label_text = entry.get("label", "")
        display = (
            f"{entry.get('_username', '')} | {display_name} | "
            f"{entry.get('start_frame', '')}-{entry.get('end_frame', '')} | "
            f"{entry.get('commentary', '')} | {label_text}"
        )
        self.welcome_history_listbox.insert("end", display)


def focus_welcome_history(self):
    if not hasattr(self, "welcome_history_panel"):
        return

    if self.welcome_history_panel.winfo_ismapped():
        self.welcome_history_panel.grid_remove()
        self.welcome_history_button_var.set("Show Annotation History")
        return

    self.welcome_history_panel.grid()
    self.welcome_history_button_var.set("Hide Annotation History")
    if hasattr(self, "welcome_history_listbox"):
        self.welcome_history_listbox.focus_set()
        if self.welcome_history_listbox.size() > 0:
            self.welcome_history_listbox.see(0)


def set_selected_video_set(self, name):
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("A video set must be selected.")

    candidate_dir = os.path.join(self.video_root_dir, cleaned)
    if not os.path.isdir(candidate_dir):
        raise NotADirectoryError(f"Selected video set directory not found: {candidate_dir}")

    self.selected_video_set = cleaned
    self.current_video_dir = _resolve_video_set_dir(self.video_root_dir, cleaned)


def set_selected_username(self, name):
    cleaned = self.sanitize_username(name)
    if not cleaned:
        raise ValueError("Username cannot be empty.")

    self.selected_username = cleaned
    self.user_output_folder = os.path.join(self.output_folder, cleaned)
    os.makedirs(self.user_output_folder, exist_ok=True)


def get_video_set_for_base_name(self, base_name):
    if base_name in self.base_name_to_video_set:
        return self.base_name_to_video_set[base_name]

    assigned_matches = [
        video_set
        for (video_set, assigned_base_name), _scene_id in self.scene_assignments.items()
        if assigned_base_name == base_name
    ]
    if len(assigned_matches) == 1:
        self.base_name_to_video_set[base_name] = assigned_matches[0]
        return assigned_matches[0]

    for video_set in _get_available_video_sets(self.video_root_dir):
        candidate_dir = _resolve_video_set_dir(self.video_root_dir, video_set)
        scenario_path = os.path.join(candidate_dir, base_name)
        if os.path.isdir(scenario_path):
            self.base_name_to_video_set[base_name] = video_set
            return video_set

    return ""


def get_scene_display_name(self, base_name, video_set=None):
    selected_set = video_set or self.selected_video_set or ""
    scene_id = self.scene_assignments.get((selected_set, base_name))
    if scene_id:
        return f"{scene_id} ({base_name})"
    return base_name


def get_scene_id(self, base_name, video_set=None):
    selected_set = video_set or self.selected_video_set or ""
    return self.scene_assignments.get((selected_set, base_name), "")


def has_saved_labels_for_base(self, base_name):
    csv_path = os.path.join(self.user_output_folder, base_name + ".csv")
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return False

    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))
    return len(rows) > 1


def get_valid_clip_entries(self):
    entries = []
    for video_set in _get_available_video_sets(self.video_root_dir):
        video_dir = _resolve_video_set_dir(self.video_root_dir, video_set)
        if not os.path.isdir(video_dir):
            continue

        for name in os.listdir(video_dir):
            scenario_path = os.path.join(video_dir, name)
            if not os.path.isdir(scenario_path):
                continue

            if (video_set, name) not in self.scene_assignments:
                continue

            if all(
                os.path.exists(os.path.join(scenario_path, f"{camera_id}.mp4"))
                for camera_id in self.camera_ids
            ):
                entries.append((video_set, name))
                self.base_name_to_video_set[name] = video_set

    if not entries:
        raise RuntimeError(
            "No assigned scenarios are available for labeling. "
            "Only clips listed in scene_assignments.txt are eligible, and each one must contain all 8 camera .mp4 files."
        )

    return entries


def get_earliest_unlabeled_clip_entry(self):
    ordered_entries = self.get_ordered_clip_entries()

    for video_set, base_name in ordered_entries:
        if not self.has_saved_labels_for_base(base_name):
            return (video_set, base_name)

    return ordered_entries[0] if ordered_entries else None


def get_ordered_clip_entries(self):
    valid_entries = self.get_valid_clip_entries()
    return sorted(
        valid_entries,
        key=lambda item: (
            _scene_sort_key(self.get_scene_id(item[1], item[0])),
            item[0].lower(),
            item[1].lower(),
        ),
    )

 
def get_next_ordered_clip_entry(self, include_skipped=False):
    ordered_entries = self.get_ordered_clip_entries()
    if not ordered_entries:
        return None

    current_entry = (self.selected_video_set, self.current_base_name)
    if current_entry in ordered_entries:
        start_index = ordered_entries.index(current_entry) + 1
        candidate_entries = ordered_entries[start_index:] + ordered_entries[:start_index]
    else:
        candidate_entries = ordered_entries

    for video_set, base_name in candidate_entries:
        if self.has_saved_labels_for_base(base_name):
            continue
        if not include_skipped and base_name in self.skipped_base_names:
            continue
        return (video_set, base_name)

    if include_skipped:
        return None

    for video_set, base_name in candidate_entries:
        if self.has_saved_labels_for_base(base_name):
            continue
        return (video_set, base_name)

    return None


def get_valid_base_names(self):
    return [base_name for _video_set, base_name in self.get_valid_clip_entries()]
