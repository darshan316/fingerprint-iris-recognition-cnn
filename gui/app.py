"""
The desktop app.

Two upload wells (a fingerprint, an iris), a Verify button, and a result card
that turns green / red / amber depending on whether the two samples come from
the same enrolled person, a different person, or nobody we know.

Model loading and inference both happen on worker threads so the window stays
responsive — TensorFlow takes a few seconds to wake up and we don't want the
UI frozen while it does.
"""

from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image

import config
from gui import theme

PREVIEW = 230          # px, side of the square image well


class UploadCard(ctk.CTkFrame):
    """A titled image well with an upload button; remembers its file path."""

    def __init__(self, master, title, hint, accent):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=16)
        self.path: Path | None = None
        self._img_ref = None       # keep a reference or Tk garbage-collects it

        self.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 6))
        head.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(head, text="●", text_color=accent,
                     font=ctk.CTkFont(size=16)).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkLabel(head, text=title, text_color=theme.TEXT,
                     font=ctk.CTkFont(theme.FONT_FAMILY, 17, "bold")
                     ).grid(row=0, column=1, sticky="w")

        well = ctk.CTkFrame(self, fg_color=theme.SURFACE_2, corner_radius=12,
                            width=PREVIEW, height=PREVIEW)
        well.grid(row=1, column=0, padx=18, pady=6)
        well.grid_propagate(False)
        well.grid_rowconfigure(0, weight=1)
        well.grid_columnconfigure(0, weight=1)
        self.preview = ctk.CTkLabel(well, text=hint, text_color=theme.MUTED,
                                    font=ctk.CTkFont(theme.FONT_FAMILY, 12),
                                    cursor="hand2")
        self.preview.grid(row=0, column=0)
        self.preview.bind("<Button-1>", lambda _e: self.browse())

        self.caption = ctk.CTkLabel(self, text="No file selected",
                                    text_color=theme.MUTED,
                                    font=ctk.CTkFont(theme.FONT_FAMILY, 11))
        self.caption.grid(row=2, column=0, pady=(2, 6))

        ctk.CTkButton(self, text="Upload image", height=36, corner_radius=10,
                      fg_color=theme.SURFACE_2, hover_color=theme.BORDER,
                      text_color=theme.TEXT, command=self.browse
                      ).grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 16))

    def browse(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All files", "*.*")],
        )
        if path:
            self.set_path(Path(path))

    def set_path(self, path: Path):
        self.path = path
        try:
            img = Image.open(path).convert("L")
        except Exception as exc:                       # unreadable file
            self.caption.configure(text=f"Could not open: {exc}",
                                   text_color=theme.BAD)
            return
        self._img_ref = ctk.CTkImage(light_image=img, dark_image=img,
                                     size=(PREVIEW - 16, PREVIEW - 16))
        self.preview.configure(image=self._img_ref, text="")
        name = path.name if len(path.name) < 30 else path.name[:27] + "..."
        self.caption.configure(text=name, text_color=theme.TEXT)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("BioVerify  ·  Fingerprint & Iris Recognition")
        self.geometry("1000x720")
        self.minsize(900, 660)
        self.configure(fg_color=theme.BG)

        self.matcher = None
        self._busy = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)      # the result panel gets the slack

        self._build_header()
        self._build_body()
        self._build_result()
        self._build_status()

        self._load_matcher_async()

    # ---------------------------------------------------------------- header
    def _build_header(self):
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=26, pady=(22, 4))
        bar.grid_columnconfigure(1, weight=1)

        icon_path = config.ROOT / "assets" / "app_icon.png"
        if icon_path.exists():
            ico = Image.open(icon_path)
            self._icon = ctk.CTkImage(light_image=ico, dark_image=ico, size=(46, 46))
            ctk.CTkLabel(bar, text="", image=self._icon).grid(row=0, column=0,
                                                              rowspan=2, padx=(0, 14))
        ctk.CTkLabel(bar, text="BioVerify",
                     font=ctk.CTkFont(theme.FONT_FAMILY, 26, "bold"),
                     text_color=theme.TEXT).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(bar, text="Multimodal biometric verification · fingerprint + iris fusion",
                     font=ctk.CTkFont(theme.FONT_FAMILY, 13),
                     text_color=theme.MUTED).grid(row=1, column=1, sticky="w")

        self.demo_btn = ctk.CTkButton(
            bar, text="Load demo pair", width=130, height=34, corner_radius=10,
            fg_color=theme.SURFACE, hover_color=theme.BORDER, text_color=theme.TEXT,
            command=self._load_demo)
        self.demo_btn.grid(row=0, column=2, rowspan=2, padx=(10, 0))

    # ------------------------------------------------------------------ body
    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew", padx=26, pady=10)
        body.grid_columnconfigure((0, 1), weight=1, uniform="cards")

        self.fp_card = UploadCard(body, "Fingerprint", "Click to upload\na fingerprint",
                                  theme.ACCENT)
        self.fp_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.iris_card = UploadCard(body, "Iris", "Click to upload\nan iris image",
                                    "#a78bfa")
        self.iris_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # the Verify button sits on its own row between the cards and the result
        self.verify_btn = ctk.CTkButton(
            self, text="Verify identity", width=220, height=48, corner_radius=12,
            font=ctk.CTkFont(theme.FONT_FAMILY, 16, "bold"),
            fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
            text_color="#06231f", command=self._on_verify)
        self.verify_btn.grid(row=2, column=0, pady=(4, 10))

    # ---------------------------------------------------------------- result
    def _build_result(self):
        panel = ctk.CTkFrame(self, fg_color=theme.SURFACE, corner_radius=16)
        panel.grid(row=3, column=0, sticky="nsew", padx=26, pady=(6, 8))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(3, weight=1)
        self.result_panel = panel

        self.badge = ctk.CTkLabel(panel, text="AWAITING INPUT",
                                  font=ctk.CTkFont(theme.FONT_FAMILY, 20, "bold"),
                                  text_color=theme.MUTED)
        self.badge.grid(row=0, column=0, sticky="w", padx=22, pady=(18, 0))

        self.headline = ctk.CTkLabel(
            panel, text="Upload a fingerprint and an iris, then press Verify.",
            font=ctk.CTkFont(theme.FONT_FAMILY, 14), text_color=theme.TEXT,
            anchor="w", justify="left")
        self.headline.grid(row=1, column=0, sticky="ew", padx=22, pady=(2, 10))

        conf = ctk.CTkFrame(panel, fg_color="transparent")
        conf.grid(row=2, column=0, sticky="ew", padx=22)
        conf.grid_columnconfigure(0, weight=1)
        self.conf_bar = ctk.CTkProgressBar(conf, height=12, corner_radius=6,
                                           progress_color=theme.ACCENT,
                                           fg_color=theme.SURFACE_2)
        self.conf_bar.set(0)
        self.conf_bar.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        self.conf_pct = ctk.CTkLabel(conf, text="—", width=58,
                                     font=ctk.CTkFont(theme.MONO_FAMILY, 14, "bold"),
                                     text_color=theme.MUTED)
        self.conf_pct.grid(row=0, column=1, padx=(12, 0))

        # per-modality breakdown
        grid = ctk.CTkFrame(panel, fg_color="transparent")
        grid.grid(row=3, column=0, sticky="nsew", padx=22, pady=14)
        grid.grid_columnconfigure((0, 1), weight=1, uniform="bk")
        self.fp_box = self._mod_box(grid, "Fingerprint match", 0)
        self.iris_box = self._mod_box(grid, "Iris match", 1)

        self.detail = ctk.CTkLabel(panel, text="", text_color=theme.MUTED,
                                   font=ctk.CTkFont(theme.FONT_FAMILY, 12),
                                   anchor="w", justify="left", wraplength=900)
        self.detail.grid(row=4, column=0, sticky="ew", padx=22, pady=(0, 18))

    def _mod_box(self, master, title, col):
        box = ctk.CTkFrame(master, fg_color=theme.SURFACE_2, corner_radius=12)
        box.grid(row=0, column=col, sticky="nsew", padx=(0, 8) if col == 0 else (8, 0))
        box.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(box, text=title, text_color=theme.MUTED,
                     font=ctk.CTkFont(theme.FONT_FAMILY, 12)
                     ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 0))
        name = ctk.CTkLabel(box, text="—", text_color=theme.TEXT,
                            font=ctk.CTkFont(theme.FONT_FAMILY, 16, "bold"))
        name.grid(row=1, column=0, sticky="w", padx=16)
        score = ctk.CTkLabel(box, text="", text_color=theme.MUTED,
                             font=ctk.CTkFont(theme.MONO_FAMILY, 12))
        score.grid(row=2, column=0, sticky="w", padx=16, pady=(0, 12))
        return {"name": name, "score": score}

    # ---------------------------------------------------------------- status
    def _build_status(self):
        self.status = ctk.CTkLabel(self, text="", text_color=theme.MUTED,
                                   font=ctk.CTkFont(theme.FONT_FAMILY, 11), anchor="w")
        self.status.grid(row=4, column=0, sticky="ew", padx=28, pady=(0, 10))

    # ------------------------------------------------------------ model load
    def _load_matcher_async(self):
        if not self._models_ready():
            self._set_status("Models not found. Run  python scripts/run_all.py  "
                             "to generate data, train and enrol.", theme.WARN)
            self.verify_btn.configure(state="disabled", text="Setup required")
            return
        self._set_status("Loading models...", theme.MUTED)
        self.verify_btn.configure(state="disabled", text="Loading models...")
        threading.Thread(target=self._load_matcher, daemon=True).start()

    def _load_matcher(self):
        try:
            from src.matcher import BiometricMatcher       # heavy import, off-thread
            matcher = BiometricMatcher()
        except Exception as exc:
            self.after(0, self._set_status, f"Failed to load models: {exc}", theme.BAD)
            return
        self.matcher = matcher
        n = len(matcher.ids)
        self.after(0, lambda: self.verify_btn.configure(state="normal",
                                                        text="Verify identity"))
        self.after(0, self._set_status, f"Ready · {n} subjects enrolled.", theme.ACCENT)

    @staticmethod
    def _models_ready():
        ok = config.ENROLL_DB.exists()
        for m in ("fingerprint", "iris"):
            ok = ok and (config.MODEL_DIR / f"{m}.keras").exists()
        return ok

    # ---------------------------------------------------------------- verify
    def _on_verify(self):
        if self._busy:
            return
        if self.matcher is None:
            self._set_status("Models are still loading...", theme.WARN)
            return
        if not self.fp_card.path or not self.iris_card.path:
            self._idle_result("Please upload both a fingerprint and an iris image.",
                              theme.WARN)
            return

        self._busy = True
        self.verify_btn.configure(state="disabled", text="Verifying...")
        self._set_status("Running inference...", theme.MUTED)
        threading.Thread(target=self._verify_worker, daemon=True).start()

    def _verify_worker(self):
        try:
            res = self.matcher.verify(str(self.fp_card.path), str(self.iris_card.path))
        except Exception as exc:
            self.after(0, self._idle_result, f"Verification error: {exc}", theme.BAD)
            self.after(0, self._reset_button)
            return
        self.after(0, self._show_result, res)
        self.after(0, self._reset_button)

    def _reset_button(self):
        self._busy = False
        self.verify_btn.configure(state="normal", text="Verify identity")

    # --------------------------------------------------------- result render
    def _idle_result(self, message, color):
        self.badge.configure(text="—", text_color=color)
        self.headline.configure(text=message, text_color=theme.TEXT)
        self.conf_bar.set(0)
        self.conf_bar.configure(progress_color=color)
        self.conf_pct.configure(text="—", text_color=theme.MUTED)
        for box in (self.fp_box, self.iris_box):
            box["name"].configure(text="—")
            box["score"].configure(text="")
        self.detail.configure(text="")

    def _show_result(self, res):
        color = theme.DECISION_COLOR[res.decision]
        self.badge.configure(text=theme.DECISION_LABEL[res.decision], text_color=color)
        self.headline.configure(text=res.headline, text_color=theme.TEXT)

        self.conf_bar.configure(progress_color=color)
        self.conf_bar.set(res.confidence)
        self.conf_pct.configure(text=f"{res.confidence*100:.0f}%", text_color=color)

        for box, m in ((self.fp_box, res.fingerprint), (self.iris_box, res.iris)):
            tag = m.best_name if m.recognized else "No confident match"
            box["name"].configure(
                text=f"{tag}",
                text_color=theme.TEXT if m.recognized else theme.MUTED)
            box["score"].configure(
                text=f"{m.best_id} · {m.best_score*100:.1f}%",
                text_color=color if m.recognized else theme.MUTED)

        self.detail.configure(text=res.detail)
        self._set_status(f"Decision: {res.decision} · fused {res.fused_score*100:.1f}%",
                         color)

    def _load_demo(self):
        s = config.ROOT / "samples"
        fp = s / "genuine_fingerprint_SUBJ_02.png"
        ir = s / "genuine_iris_SUBJ_02.png"
        if fp.exists() and ir.exists():
            self.fp_card.set_path(fp)
            self.iris_card.set_path(ir)
            self._set_status("Loaded a genuine demo pair — press Verify.", theme.ACCENT)
        else:
            self._set_status("No demo samples found. Run scripts/generate_data.py.",
                             theme.WARN)

    def _set_status(self, text, color=theme.MUTED):
        self.status.configure(text=text, text_color=color)


def launch():
    App().mainloop()
