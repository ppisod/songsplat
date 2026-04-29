"""songsplat - main application window."""

from __future__ import annotations

import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
from typing import Optional

from songsplat.audio.playback import AudioPlayer
from songsplat.core.models import Project, Song
from songsplat.core.project_io import (
    PROJECT_EXTENSION,
    load_project,
    new_project,
    save_project,
)
from songsplat.core.undo import UndoStack
from songsplat.ui import theme as T

APP_NAME    = "songsplat"
AUTOSAVE_MS = 60_000

NAV_ITEMS = [
    ("Songs",   "songs"),
    ("Splats",  "splats"),
    ("Label",   "label"),
    ("Train",   "train"),
    ("Predict", "predict"),
    ("Export",  "export"),
]


class App(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        T.configure_ttk()
        self.title(APP_NAME)
        self.geometry("1400x860")
        self.minsize(900, 600)
        self.configure(bg=T.BG)

        self.player     = AudioPlayer()
        self.undo_stack = UndoStack()
        self.undo_stack.on_change = self._on_undo_change

        self._project:      Optional[Project] = None
        self._project_path: Optional[str]     = None
        self._dirty                            = False
        self._active_song:  Optional[Song]     = None

        self._build_layout()
        self._bind_keys()
        self._autosave_id = self.after(AUTOSAVE_MS, self._autosave_tick)
        self._project = new_project()
        self._apply_project()
        self._show_view("songs")

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._sidebar = _Sidebar(self, self._show_view)
        self._sidebar.grid(row=0, column=0, sticky="nsew", rowspan=2)

        content = tk.Frame(self, bg=T.BG)
        content.grid(row=0, column=1, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)
        self._content = content

        from songsplat.ui.transport_bar import TransportBar
        self.transport = TransportBar(self, self.player)
        self.transport.grid(row=1, column=1, sticky="ew")
        self.grid_rowconfigure(1, minsize=52)

        self._views: dict[str, BaseView] = {}
        self._build_views()

    def _build_views(self) -> None:
        from songsplat.ui.views.songs_view   import SongsView
        from songsplat.ui.views.label_view   import LabelView
        from songsplat.ui.views.splats_view  import SplatsView
        from songsplat.ui.views.train_view   import TrainView
        from songsplat.ui.views.predict_view import PredictView
        from songsplat.ui.views.export_view  import ExportView

        for key, cls in [
            ("songs",   SongsView),
            ("label",   LabelView),
            ("splats",  SplatsView),
            ("train",   TrainView),
            ("predict", PredictView),
            ("export",  ExportView),
        ]:
            view = cls(self._content, self)
            view.grid(row=0, column=0, sticky="nsew")
            view.grid_remove()
            self._views[key] = view

    def _show_view(self, name: str) -> None:
        for key, view in self._views.items():
            if key == name:
                view.grid()
                view.on_show()
            else:
                view.grid_remove()
        self._sidebar.set_active(name)

    def _bind_keys(self) -> None:
        for seq in ("<Command-z>", "<Control-z>"):
            self.bind(seq, lambda _e: self.undo_stack.undo())
        for seq in ("<Command-Z>", "<Control-Z>"):
            self.bind(seq, lambda _e: self.undo_stack.redo())
        for seq in ("<Command-s>", "<Control-s>"):
            self.bind(seq, lambda _e: self._save())
        for seq in ("<Command-n>", "<Control-n>"):
            self.bind(seq, lambda _e: self.new_project())
        for seq in ("<Command-o>", "<Control-o>"):
            self.bind(seq, lambda _e: self.open_project_dialog())
        self.bind("<Up>",    lambda _e: self._label_key("nudge_up"))
        self.bind("<Down>",  lambda _e: self._label_key("nudge_down"))
        self.bind("<Left>",  lambda _e: self._label_key("prev_chunk"))
        self.bind("<Right>", lambda _e: self._label_key("next_chunk"))
        self.bind("<space>", lambda _e: self.player.toggle_play_pause())

    def _label_key(self, action: str) -> None:
        v = self._views.get("label")
        if v and hasattr(v, "handle_key"):
            v.handle_key(action)

    def new_project(self) -> None:
        """Create a blank project, discarding unsaved changes after confirmation."""
        if not self._confirm_discard():
            return
        self._project      = new_project()
        self._project_path = None
        self._dirty        = False
        self.undo_stack.clear()
        self._apply_project()
        self._show_view("songs")

    def open_project_dialog(self) -> None:
        """Show an open-file dialog and load the selected project."""
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[
                ("songsplat projects", f"*{PROJECT_EXTENSION}"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._open_project(path)

    def open_recent(self, path: str) -> None:
        """Open a recently used project file."""
        if self._confirm_discard():
            self._open_project(path)

    def set_active_song(self, song: Song) -> None:
        """Switch the active song and notify all views."""
        self._active_song = song
        try:
            self.player.load_song(song)
        except Exception:
            pass
        for view in self._views.values():
            view.set_song(song)
        self.transport.set_song(song)

    def mark_dirty(self) -> None:
        """Mark the project as having unsaved changes."""
        if not self._dirty:
            self._dirty = True
            self._update_title()

    def _open_project(self, path: str) -> None:
        try:
            proj = load_project(path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open project:\n{e}", parent=self)
            return
        self._project      = proj
        self._project_path = path
        self._dirty        = False
        self.undo_stack.clear()
        self._apply_project()
        self._show_view("songs")

    def _save(self) -> None:
        if self._project is None:
            return
        if self._project_path is None:
            self._save_as()
        else:
            self._save_to(self._project_path)

    def _save_as(self) -> None:
        if self._project is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save Project As",
            defaultextension=PROJECT_EXTENSION,
            initialfile=self._project.name + PROJECT_EXTENSION,
            filetypes=[
                ("songsplat projects", f"*{PROJECT_EXTENSION}"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._save_to(path)

    def _save_to(self, path: str) -> None:
        try:
            save_project(self._project, path)
            self._project_path = path
            self._dirty        = False
            self._update_title()
        except Exception as e:
            messagebox.showerror("Save Error", str(e), parent=self)

    def _autosave_tick(self) -> None:
        if self._project and self._project_path and self._dirty:
            self._save_to(self._project_path)
        self._autosave_id = self.after(AUTOSAVE_MS, self._autosave_tick)

    def _apply_project(self) -> None:
        for view in self._views.values():
            view.set_project(self._project)
        self._active_song = (
            self._project.songs[0]
            if self._project and self._project.songs
            else None
        )
        if self._active_song:
            self.set_active_song(self._active_song)
        self._update_title()

    def _update_title(self) -> None:
        name = self._project.name if self._project else APP_NAME
        self.title(f"{name} *" if self._dirty else name)

    def _on_undo_change(self) -> None:
        v = self._views.get("label")
        if v and hasattr(v, "refresh_labels"):
            v.refresh_labels()
        self.mark_dirty()

    def _confirm_discard(self) -> bool:
        if not self._dirty or self._project is None:
            return True
        return messagebox.askyesno(
            "Unsaved changes", "Discard unsaved changes?", parent=self
        )

    def destroy(self) -> None:
        try:
            self.after_cancel(self._autosave_id)
        except Exception:
            pass
        self.player.stop()
        super().destroy()


class _Sidebar(tk.Frame):
    """Left navigation panel."""

    def __init__(self, master: App, on_nav):
        super().__init__(master, bg=T.SIDEBAR, width=180)
        self.pack_propagate(False)
        self._btns: dict[str, _NavButton] = {}
        self._build(master, on_nav)

    def _build(self, app: App, on_nav) -> None:
        tk.Label(
            self, text="songsplat", bg=T.SIDEBAR, fg=T.FG,
            font=("TkDefaultFont", 18, "bold"), anchor="w", padx=16,
        ).pack(fill="x", pady=(20, 16))

        for label, key in NAV_ITEMS:
            b = _NavButton(self, label, lambda k=key: on_nav(k))
            b.pack(fill="x", padx=8, pady=1)
            self._btns[key] = b

        tk.Frame(self, bg=T.SIDEBAR).pack(fill="y", expand=True)
        tk.Frame(self, bg=T.BORDER, height=1).pack(fill="x", padx=12, pady=(0, 6))

        for text, cmd in [
            ("New project",  app.new_project),
            ("Open project", app.open_project_dialog),
            ("Save  ⌘S", app._save),
        ]:
            _NavButton(self, text, cmd, small=True).pack(fill="x", padx=8, pady=1)

        tk.Frame(self, bg=T.SIDEBAR, height=12).pack()

    def set_active(self, key: str) -> None:
        for k, btn in self._btns.items():
            btn.set_active(k == key)


class _NavButton(tk.Label):
    """Sidebar item with hover and active-state highlighting."""

    def __init__(self, master, text: str, command, small=False):
        font = ("TkDefaultFont", 11) if small else ("TkDefaultFont", 13)
        super().__init__(
            master, text=f"  {text}", bg=T.SIDEBAR, fg=T.FG_DIM,
            font=font, anchor="w", padx=8, pady=4 if small else 6,
            cursor="hand2",
        )
        self._active = False
        self.bind("<Button-1>", lambda _e: command())
        self.bind("<Enter>",    lambda _e: self._hover(True))
        self.bind("<Leave>",    lambda _e: self._hover(False))

    def set_active(self, active: bool) -> None:
        self._active = active
        self.configure(
            bg=T.SEL if active else T.SIDEBAR,
            fg=T.FG if active else T.FG_DIM,
        )

    def _hover(self, entering: bool) -> None:
        if not self._active:
            self.configure(bg=T.SEL_HOVER if entering else T.SIDEBAR)


class BaseView(tk.Frame):
    """Base class for all content views."""

    def __init__(self, master, app: App):
        super().__init__(master, bg=T.BG)
        self.app = app

    @property
    def project(self) -> Optional[Project]:
        return self.app._project

    def set_project(self, project: Optional[Project]) -> None:
        """Called when a project is opened or created."""

    def set_song(self, song: Optional[Song]) -> None:
        """Called when the active song changes."""

    def on_show(self) -> None:
        """Called each time this view is made visible."""


def run() -> None:
    """Launch the application."""
    App().mainloop()
