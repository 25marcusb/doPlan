import tkinter as tk
from tkinter import ttk


def build_label_panel(app, commentary_placeholder, commentary_options, speed_options):
    label_panel = tk.Frame(
        app.outer,
        bg=app.colors["panel"],
        highlightbackground=app.colors["border"],
        highlightthickness=1,
        padx=16,
        pady=14,
    )
    label_panel.grid(row=2, column=0, sticky="ew", pady=(0, 14))
    label_panel.grid_remove()
    app.label_panel = label_panel

    label_header = tk.Frame(label_panel, bg=app.colors["panel"])
    label_header.grid(row=0, column=0, sticky="w")
    ttk.Label(label_header, text="Label", style="PanelTitle.TLabel").pack(side="left")
    ttk.Button(
        label_header,
        text="?",
        width=2,
        style="Secondary.TButton",
        command=app.show_label_help,
    ).pack(side="left", padx=(6, 0))
    app.label_entry = tk.Text(
        label_panel,
        height=3,
        wrap="word",
        font=("Aptos", 12),
        bg="#fff1f1",
        fg=app.colors["text"],
        insertbackground=app.colors["text"],
        relief="solid",
        bd=1,
        highlightthickness=2,
        highlightbackground="#cc4b4b",
        highlightcolor="#cc4b4b",
    )
    app.label_entry.grid(row=1, column=0, sticky="ew", padx=(0, 12), pady=(8, 0))
    app.label_entry.bind("<KeyRelease>", app.on_label_text_changed)
    app.label_entry.bind("<<Paste>>", app.on_label_text_changed)

    commentary_header = tk.Frame(label_panel, bg=app.colors["panel"])
    commentary_header.grid(row=0, column=1, sticky="w")
    ttk.Label(commentary_header, text="Commentary", style="PanelTitle.TLabel").pack(side="left")
    ttk.Button(
        commentary_header,
        text="?",
        width=2,
        style="Secondary.TButton",
        command=app.show_commentary_help,
    ).pack(side="left", padx=(6, 0))
    app.commentary_combo = ttk.Combobox(
        label_panel,
        textvariable=app.commentary_var,
        state="readonly",
        values=[commentary_placeholder, *commentary_options],
        width=22,
        style="App.TCombobox",
    )
    app.commentary_combo.grid(row=1, column=1, sticky="ew", padx=(0, 12), pady=(8, 0))
    app.commentary_combo.bind("<<ComboboxSelected>>", app.on_commentary_changed)

    speed_header = tk.Frame(label_panel, bg=app.colors["panel"])
    speed_header.grid(row=0, column=2, sticky="w")
    ttk.Label(speed_header, text="Speed", style="PanelTitle.TLabel").pack(side="left")
    app.speed_combo = ttk.Combobox(
        label_panel,
        textvariable=app.speed_preset_var,
        state="readonly",
        values=list(speed_options.keys()),
        width=12,
        style="App.TCombobox",
    )
    app.speed_combo.grid(row=1, column=2, sticky="ew", padx=(0, 12), pady=(8, 0))
    app.speed_combo.bind("<<ComboboxSelected>>", app.on_speed_selected)

    ttk.Checkbutton(
        label_panel,
        text="Auto Next",
        variable=app.auto_next_var,
        style="App.TCheckbutton",
    ).grid(row=1, column=3, sticky="w", padx=(0, 12), pady=(8, 0))

    app.save_button = ttk.Button(
        label_panel,
        text="Save",
        style="Secondary.TButton",
        command=app.save_segment_label,
    )
    app.save_button.grid(row=1, column=4, sticky="ew", padx=(0, 8), pady=(8, 0))

    app.save_next_button = ttk.Button(
        label_panel,
        text="Save + Next",
        style="Primary.TButton",
        command=app.save_and_next,
    )
    app.save_next_button.grid(row=1, column=5, sticky="ew", pady=(8, 0))

    app.no_label_button = ttk.Button(
        label_panel,
        text="No Label Needed",
        style="Secondary.TButton",
        command=app.submit_no_label_and_next,
    )
    app.no_label_button.grid(row=1, column=6, sticky="ew", padx=(12, 0), pady=(8, 0))

    app.add_history_annotation_button = ttk.Button(
        label_panel,
        text="Add Another",
        style="Secondary.TButton",
        command=app.add_annotation_from_selected_history,
        state="disabled",
    )
    app.add_history_annotation_button.grid(row=1, column=7, sticky="ew", padx=(12, 0), pady=(8, 0))

    label_panel.grid_columnconfigure(0, weight=1)
    label_panel.grid_columnconfigure(1, weight=1)

    ttk.Label(label_panel, text="Recent History", style="PanelTitle.TLabel").grid(
        row=2, column=0, sticky="w", pady=(16, 0)
    )
    history_panel = tk.Frame(
        label_panel,
        bg=app.colors["panel"],
        highlightbackground=app.colors["border"],
        highlightthickness=1,
        padx=8,
        pady=8,
    )
    history_panel.grid(row=3, column=0, columnspan=8, sticky="ew", pady=(8, 0))
    history_panel.grid_columnconfigure(0, weight=1)
    history_panel.grid_rowconfigure(0, weight=1)

    app.history_listbox = tk.Listbox(
        history_panel,
        height=6,
        font=("Aptos", 10),
        activestyle="none",
        bg="#ffffff",
        fg=app.colors["text"],
        highlightthickness=0,
        selectbackground=app.colors["accent"],
        selectforeground="#ffffff",
    )
    app.history_listbox.grid(row=0, column=0, sticky="ew")
    app.history_listbox.bind("<<ListboxSelect>>", app.on_history_selected)
    app.delete_history_button = ttk.Button(
        history_panel,
        text="Delete Selected",
        style="Secondary.TButton",
        command=app.delete_selected_history_entry,
        state="disabled",
    )
    app.delete_history_button.grid(row=1, column=0, sticky="e", pady=(8, 0))
