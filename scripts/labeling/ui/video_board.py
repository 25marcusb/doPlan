import tkinter as tk
from tkinter import ttk


def build_video_board(app, camera_layout):
    info_panel = tk.Frame(
        app.outer,
        bg=app.colors["panel"],
        highlightbackground=app.colors["border"],
        highlightthickness=1,
        padx=16,
        pady=12,
    )
    info_panel.grid(row=3, column=0, sticky="nsew")
    info_panel.grid_remove()
    info_panel.grid_columnconfigure(0, weight=1)
    info_panel.grid_rowconfigure(1, weight=1)
    app.info_panel = info_panel

    top_info = tk.Frame(info_panel, bg=app.colors["panel"])
    top_info.grid(row=0, column=0, sticky="ew", pady=(0, 12))
    top_info.grid_columnconfigure(0, weight=1)

    ttk.Label(top_info, textvariable=app.session_var, style="Info.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(top_info, textvariable=app.frame_var, style="PanelText.TLabel").grid(row=0, column=1, sticky="e")
    ttk.Label(top_info, textvariable=app.clip_var, style="PanelText.TLabel").grid(
        row=1, column=0, sticky="w", pady=(4, 0)
    )
    ttk.Label(top_info, textvariable=app.progress_var, style="PanelText.TLabel").grid(
        row=1, column=1, sticky="e", pady=(4, 0)
    )

    video_board = tk.Frame(info_panel, bg=app.colors["bg"])
    video_board.grid(row=1, column=0, sticky="nsew")

    for cam, spec in camera_layout.items():
        card = tk.Frame(
            video_board,
            bg="#ffffff",
            highlightbackground=app.colors["border"],
            highlightthickness=1,
            padx=8,
            pady=8,
        )
        card.grid(
            row=spec["row"],
            column=spec["column"],
            rowspan=spec.get("rowspan", 1),
            padx=6,
            pady=6,
            sticky="nsew",
        )
        card.grid_propagate(False)
        card.pack_propagate(False)
        card.configure(width=spec["size"][0] + 16, height=spec["size"][1] + 38)
        app.video_cards[cam] = card

        tk.Label(
            card,
            text=cam,
            bg="#ffffff",
            fg=app.colors["muted"],
            font=("Aptos", 10, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 6))

        label = tk.Label(
            card,
            bg="#f8fafb",
            fg=app.colors["muted"],
            text="Waiting for clip",
            anchor="center",
            justify="center",
        )
        label.pack(fill="both", expand=True)
        app.video_labels[cam] = label

    video_board.grid_columnconfigure(0, weight=1)
    video_board.grid_columnconfigure(1, weight=3)
    video_board.grid_columnconfigure(2, weight=1)
    for row in range(3):
        video_board.grid_rowconfigure(row, weight=1)
    app.video_board = video_board
