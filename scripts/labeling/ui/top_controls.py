import tkinter as tk
from tkinter import ttk


def build_top_controls(app):
    app.main_header = ttk.Frame(app.outer, style="App.TFrame")
    app.main_header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
    app.main_header.grid_remove()
    app.main_header.grid_columnconfigure(0, weight=1)
    ttk.Label(app.main_header, text="MI3 Labeler", style="Title.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(app.main_header, textvariable=app.hint_var, style="Body.TLabel").grid(
        row=1, column=0, sticky="w", pady=(4, 0)
    )

    app.setup_panel = tk.Frame(
        app.outer,
        bg=app.colors["panel"],
        highlightbackground=app.colors["border"],
        highlightthickness=1,
        padx=16,
        pady=14,
    )
    app.setup_panel.grid(row=1, column=0, sticky="ew", pady=(0, 14))
    app.setup_panel.grid_remove()

    app.start_pause_button = ttk.Button(
        app.setup_panel,
        text="Start",
        style="Primary.TButton",
        command=app.start_or_toggle,
    )
    app.start_pause_button.grid(row=1, column=3, sticky="ew", padx=(0, 8), pady=(6, 0))

    ttk.Button(
        app.setup_panel,
        text="Restart",
        style="Secondary.TButton",
        command=app.restart_session,
    ).grid(row=1, column=4, sticky="ew", padx=(0, 8), pady=(6, 0))

    ttk.Button(
        app.setup_panel,
        text="Skip",
        style="Secondary.TButton",
        command=app.skip_current_clip,
    ).grid(row=1, column=5, sticky="ew", padx=(0, 8), pady=(6, 0))

    ttk.Checkbutton(
        app.setup_panel,
        text="Confirm Skip",
        variable=app.confirm_skip_var,
        style="App.TCheckbutton",
    ).grid(row=1, column=6, sticky="w", padx=(0, 12), pady=(6, 0))

    for column in range(3):
        app.setup_panel.grid_columnconfigure(column, weight=1)
