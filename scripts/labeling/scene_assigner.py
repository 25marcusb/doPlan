import argparse
import os


LABELING_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(LABELING_DIR, "settings.txt")
ASSIGNMENTS_FILE = os.path.join(LABELING_DIR, "scene_assignments.txt")
SCENE_NUMBER_MIN = 0
SCENE_NUMBER_MAX = 1000


def load_video_root_dir():
    video_dir = ""
    with open(SETTINGS_FILE, "r", encoding="utf-8") as settings_file:
        for raw_line in settings_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            if key.strip() == "video_dir":
                video_dir = value.strip()

    if not video_dir:
        raise ValueError("video_dir must be set in settings.txt")
    return video_dir


VIDEO_ROOT_DIR = load_video_root_dir()


def get_video_sets():
    return sorted(
        [
            name
            for name in os.listdir(VIDEO_ROOT_DIR)
            if os.path.isdir(os.path.join(VIDEO_ROOT_DIR, name))
        ],
        key=str.lower,
    )


def resolve_video_set_dir(video_set):
    set_dir = os.path.join(VIDEO_ROOT_DIR, video_set)
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


def get_scenarios(video_set):
    set_dir = resolve_video_set_dir(video_set)
    if not os.path.isdir(set_dir):
        return []

    return sorted(
        [
            name
            for name in os.listdir(set_dir)
            if os.path.isdir(os.path.join(set_dir, name))
        ],
        key=str.lower,
    )


def read_assignments():
    assignments = {}
    if not os.path.exists(ASSIGNMENTS_FILE):
        return assignments

    with open(ASSIGNMENTS_FILE, "r", encoding="utf-8") as assignment_file:
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


def write_assignments(assignments):
    rows = sorted(assignments.items(), key=lambda item: (item[0][0].lower(), item[0][1].lower()))
    with open(ASSIGNMENTS_FILE, "w", encoding="utf-8") as assignment_file:
        assignment_file.write("# video_set<TAB>base_name<TAB>scene_id\n")
        for (video_set, base_name), scene_id in rows:
            assignment_file.write(f"{video_set}\t{base_name}\t{scene_id}\n")


def get_existing_scene_numbers(assignments):
    numbers = set()
    for scene_id in assignments.values():
        if not scene_id.startswith("scene-"):
            continue
        suffix = scene_id.split("-", 1)[1]
        if suffix.isdigit():
            numbers.add(int(suffix))
    return numbers


def build_unassigned_clip_list(assignments, only_video_set=None):
    clips = []
    video_sets = [only_video_set] if only_video_set else get_video_sets()

    for video_set in video_sets:
        for scenario in get_scenarios(video_set):
            key = (video_set, scenario)
            if key not in assignments:
                clips.append(key)

    return clips


def assign_scene_ids(assignments, clips):
    if not clips:
        return 0

    used_numbers = get_existing_scene_numbers(assignments)
    available_numbers = [
        number
        for number in range(SCENE_NUMBER_MIN, SCENE_NUMBER_MAX + 1)
        if number not in used_numbers
    ]

    if len(available_numbers) < len(clips):
        raise RuntimeError("Not enough unused scene numbers available for all unassigned clips.")

    for index, clip in enumerate(clips):
        scene_number = available_numbers[index]
        assignments[clip] = f"scene-{scene_number:04d}"

    return len(clips)


def run_assignment(only_video_set=None, seed=None, verbose=True):
    assignments = read_assignments()
    clips = build_unassigned_clip_list(assignments, only_video_set=only_video_set)
    assigned_count = assign_scene_ids(assignments, clips)
    write_assignments(assignments)

    if verbose:
        scope = only_video_set if only_video_set else "all video sets"
        print(f"Assignment scope: {scope}")
        print(f"New assignments created: {assigned_count}")
        print(f"Total assignments stored: {len(assignments)}")
        print(f"File: {ASSIGNMENTS_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Automatically assign unique scene-XXXX ids to unassigned clips in increasing order and store them in labeling/scene_assignments.txt"
        ),
    )
    parser.add_argument(
        "--video-set",
        help="Restrict assignment to a single top-level video set, for example las_vegas_1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Unused compatibility option. Scene ids are now assigned in increasing order.",
    )
    args = parser.parse_args()

    run_assignment(only_video_set=args.video_set, seed=args.seed)


if __name__ == "__main__":
    main()
