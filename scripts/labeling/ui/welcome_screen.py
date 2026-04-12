import tkinter as tk
from tkinter import ttk


def build_welcome_screen(app):
    welcome_frame = tk.Frame(
        app.outer,
        bg=app.colors["panel"],
        highlightbackground=app.colors["border"],
        highlightthickness=1,
        padx=28,
        pady=24,
    )
    welcome_frame.grid(row=0, column=0, sticky="nsew")
    welcome_frame.grid_columnconfigure(0, weight=1)
    welcome_frame.grid_rowconfigure(6, weight=1)
    app.welcome_frame = welcome_frame

    ttk.Label(welcome_frame, text="MI3 Labeler", style="Title.TLabel").grid(
        row=0, column=0, sticky="w"
    )
    ttk.Label(
        welcome_frame,
        text=(
            "doPlan links language to driving behavior. Watch the clip, imagine you are a "
            "passenger in a taxi, and write what you would tell the driver to make that scene happen. "
            "Do not describe the clip. Write the command that would cause it. If normal driving needs "
            "no instruction, use No Label Needed."
        ),
        style="Body.TLabel",
    ).grid(row=1, column=0, sticky="w", pady=(8, 18))

    context_panel = tk.Frame(
        welcome_frame,
        bg=app.colors["panel"],
        highlightbackground=app.colors["border"],
        highlightthickness=1,
        padx=16,
        pady=16,
    )
    context_panel.grid(row=2, column=0, sticky="ew")
    ttk.Label(context_panel, text="Task Context", style="PanelTitle.TLabel").grid(
        row=0, column=0, sticky="w"
    )
    ttk.Label(
        context_panel,
        text=(
            "Core rule: 'If you were in a taxi, what would you tell the driver to make this scene happen?'\n"
            "Examples: 'turn right here', 'slow down for the pedestrian', 'follow that white car'.\n"
            "Leave it blank only when the car is just driving normally and no guidance is needed.\n"
            "Use commentary tags after labeling: blank = non-referential, (s) = static, (d) = dynamic, ds = both."
        ),
        style="Info.TLabel",
        justify="left",
    ).grid(row=1, column=0, sticky="w", pady=(8, 0))

    welcome_form = tk.Frame(
        welcome_frame,
        bg=app.colors["panel"],
        highlightbackground=app.colors["border"],
        highlightthickness=1,
        padx=16,
        pady=16,
    )
    welcome_form.grid(row=3, column=0, sticky="ew")
    for column in range(2):
        welcome_form.grid_columnconfigure(column, weight=1)

    ttk.Label(welcome_form, text="User", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
    app.existing_user_combo = ttk.Combobox(
        welcome_form,
        textvariable=app.existing_user_var,
        state="readonly",
        width=18,
        style="App.TCombobox",
    )
    app.existing_user_combo.grid(row=1, column=0, sticky="ew", padx=(0, 12), pady=(6, 0))

    ttk.Label(welcome_form, text="New User Name", style="PanelTitle.TLabel").grid(row=0, column=1, sticky="w")
    app.new_user_entry = ttk.Entry(
        welcome_form,
        textvariable=app.new_user_var,
        width=18,
        style="App.TEntry",
    )
    app.new_user_entry.grid(row=1, column=1, sticky="ew", pady=(6, 0))

    stats_panel = tk.Frame(
        welcome_frame,
        bg=app.colors["panel"],
        highlightbackground=app.colors["border"],
        highlightthickness=1,
        padx=16,
        pady=16,
    )
    stats_panel.grid(row=4, column=0, sticky="ew", pady=(14, 0))
    ttk.Label(stats_panel, text="Your Progress", style="PanelTitle.TLabel").grid(
        row=0, column=0, sticky="w"
    )
    ttk.Label(stats_panel, textvariable=app.welcome_stats_var, style="Info.TLabel").grid(
        row=1, column=0, sticky="w", pady=(8, 0)
    )

    totals_panel = tk.Frame(
        welcome_frame,
        bg=app.colors["panel"],
        highlightbackground=app.colors["border"],
        highlightthickness=1,
        padx=16,
        pady=16,
    )
    totals_panel.grid(row=5, column=0, sticky="ew", pady=(14, 0))
    ttk.Label(totals_panel, text="Overall Totals", style="PanelTitle.TLabel").grid(
        row=0, column=0, sticky="w"
    )
    ttk.Label(totals_panel, textvariable=app.welcome_totals_var, style="Info.TLabel").grid(
        row=1, column=0, sticky="w", pady=(8, 0)
    )

    welcome_history_panel = tk.Frame(
        welcome_frame,
        bg=app.colors["panel"],
        highlightbackground=app.colors["border"],
        highlightthickness=1,
        padx=16,
        pady=16,
    )
    welcome_history_panel.grid(row=6, column=0, sticky="nsew", pady=(14, 0))
    welcome_history_panel.grid_remove()
    welcome_history_panel.grid_columnconfigure(0, weight=1)
    welcome_history_panel.grid_rowconfigure(1, weight=1)
    app.welcome_history_panel = welcome_history_panel
    ttk.Label(welcome_history_panel, text="Annotation History", style="PanelTitle.TLabel").grid(
        row=0, column=0, sticky="w"
    )

    welcome_history_list_panel = tk.Frame(
        welcome_history_panel,
        bg=app.colors["panel"],
        highlightbackground=app.colors["border"],
        highlightthickness=1,
        padx=8,
        pady=8,
    )
    welcome_history_list_panel.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
    welcome_history_list_panel.grid_columnconfigure(0, weight=1)
    welcome_history_list_panel.grid_rowconfigure(0, weight=1)

    app.welcome_history_listbox = tk.Listbox(
        welcome_history_list_panel,
        height=12,
        font=("Aptos", 10),
        activestyle="none",
        bg="#ffffff",
        fg=app.colors["text"],
        highlightthickness=0,
        selectbackground=app.colors["accent"],
        selectforeground="#ffffff",
    )
    app.welcome_history_listbox.grid(row=0, column=0, sticky="nsew")
    welcome_history_scrollbar = ttk.Scrollbar(
        welcome_history_list_panel,
        orient="vertical",
        command=app.welcome_history_listbox.yview,
    )
    welcome_history_scrollbar.grid(row=0, column=1, sticky="ns")
    app.welcome_history_listbox.configure(yscrollcommand=welcome_history_scrollbar.set)

    welcome_actions = ttk.Frame(welcome_frame, style="App.TFrame")
    welcome_actions.grid(row=7, column=0, sticky="w", pady=(18, 0))
    ttk.Button(
        welcome_actions,
        text="Continue",
        style="Primary.TButton",
        command=app.start_labeling,
    ).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(
        welcome_actions,
        textvariable=app.welcome_history_button_var,
        style="Secondary.TButton",
        command=app.focus_welcome_history,
    ).grid(row=0, column=1)
