"""
Voice to English: Bangla/English speech -> English text.
faster-whisper (OpenAI Whisper) + PySide6.

Features: light/dark/system theme, responsive layout, drag & drop, live progress with ETA,
cancel, copy, save (.txt / .srt), model auto-update from Hugging Face, GPU->CPU fallback,
user-selectable models folder (with optional move of existing models).

Fast startup: the heavy libraries (ctranslate2, faster-whisper, huggingface_hub, onnxruntime, av)
are NOT imported at launch. The window opens first, and they load in a background thread.

pip install faster-whisper PySide6
"""

import gc
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from string import Template

# --- Windowed (--windowed) builds have no console: stdout/stderr are None and tqdm/huggingface would crash ---
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from PySide6.QtCore import QPointF, QRectF, QSettings, Qt, QThread, QTimer, QUrl, Signal  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QColor, QDesktopServices, QFont, QGuiApplication, QIcon, QKeySequence, QPainter, QPainterPath,
    QPalette, QPen, QPixmap, QPolygonF, QShortcut, QTextCursor,
)
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QBoxLayout, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

# ----------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------
APP = "BanglaWhisper"
DEFAULT_MODELS_DIR = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / ".cache")) / APP / "models"
_DIR = {"models": DEFAULT_MODELS_DIR}
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".oga", ".flac", ".opus", ".webm",
              ".aac", ".wma", ".mkv", ".mov", ".amr", ".3gp")
MODELS = [  # (name, label, approx download size text, approx size in GB)
    ("large-v3", "large-v3  ·  most accurate", "3 GB", 3.1),
    ("large-v2", "large-v2  ·  alternative", "3 GB", 3.1),
    ("medium", "medium  ·  faster", "1.5 GB", 1.6),
    ("small", "small  ·  fastest", "0.5 GB", 0.5),
]
SIZES = {m[0]: m[2] for m in MODELS}
SIZES_GB = {m[0]: m[3] for m in MODELS}
PATTERNS = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]
LANG_NAMES = {"bn": "Bangla", "en": "English", "hi": "Hindi", "ur": "Urdu", "as": "Assamese"}

STATE = {"cuda": None}   # None = not detected yet (engine still loading)


def get_models_dir():
    return _DIR["models"]


def set_models_dir(p):
    _DIR["models"] = Path(p)


class _Cancelled(Exception):
    pass


# ----------------------------------------------------------------------------------------------
# Lazy engine loading (the slow part of startup)
# ----------------------------------------------------------------------------------------------
def _setup_dll_paths():
    """Let CTranslate2 find cuBLAS/cuDNN DLLs (pip 'nvidia-*' wheels, or DLLs placed next to the exe)."""
    dirs = []
    if getattr(sys, "frozen", False):
        dirs.append(Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)))
    else:
        for p in sys.path:
            try:
                nv = Path(p) / "nvidia"
                if nv.is_dir():
                    dirs += [d for d in nv.glob("*/bin") if d.is_dir()]
            except OSError:
                pass
    for d in dirs:
        try:
            os.add_dll_directory(str(d))
        except (AttributeError, OSError):
            pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


_ENGINE = {}
_ENGINE_LOCK = threading.Lock()


def load_engine():
    """Import the heavy ML libraries once. Safe to call from any thread, any number of times."""
    if _ENGINE:
        return _ENGINE
    with _ENGINE_LOCK:
        if not _ENGINE:
            _setup_dll_paths()
            import ctranslate2
            import huggingface_hub
            from faster_whisper import WhisperModel
            from tqdm.auto import tqdm
            try:
                STATE["cuda"] = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                STATE["cuda"] = False
            _ENGINE.update(ct2=ctranslate2, hf=huggingface_hub, WhisperModel=WhisperModel, tqdm=tqdm)
    return _ENGINE


class WarmupThread(QThread):
    """Loads the engine in the background right after the window appears."""
    ready = Signal(bool)

    def run(self):
        try:
            load_engine()
        except Exception:
            pass   # the real error is shown when the user starts a job
        self.ready.emit(bool(STATE["cuda"]))


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def fmt(seconds):
    s = int(max(0, seconds))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def srt_ts(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def short_path(p, n=46):
    p = str(p)
    return p if len(p) <= n else p[:14] + "…" + p[-(n - 15):]


def friendly(e):
    s = str(e)
    low = s.lower()
    if any(k in low for k in ("connection", "timed out", "timeout", "resolve", "max retries",
                              "offline", "network", "ssl")):
        return "Couldn't reach the model server. Check your internet connection and try again."
    if "out of memory" in low:
        return "Not enough memory for this model. Pick a smaller model (medium or small) and try again."
    if any(k in low for k in ("invalid data", "invaliddataerror", "decode", "error opening")):
        return "This file couldn't be read as audio. It may be corrupted or in an unsupported format."
    if "no space left" in low:
        return "The disk is full. Free up space or choose another models folder in Settings."
    if isinstance(e, PermissionError):
        return f"Permission denied: {s}"
    return s or e.__class__.__name__


def gpu_error(e):
    low = str(e).lower()
    return any(k in low for k in ("cuda", "cudnn", "cublas", "gpu", "out of memory", "dll", "cannot be loaded"))


# ----------------------------------------------------------------------------------------------
# Models folder helpers
# ----------------------------------------------------------------------------------------------
def validate_dir(p):
    """Return None if the folder is usable, otherwise a human-readable problem."""
    p = Path(p)
    if len(str(p)) > 150:
        return "That path is too long (Windows may fail with very long paths). Choose a shorter folder."
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".write_test"
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        return f"Can't write to that folder ({e.strerror or e})."
    return None


def has_models(d):
    try:
        return Path(d).is_dir() and any(Path(d).glob("models--*"))
    except OSError:
        return False


def ensure_space(name):
    need = SIZES_GB.get(name, 3.2) * 1.15 * 1e9
    d = get_models_dir()
    d.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(d).free
    if free < need:
        raise RuntimeError(
            f"Not enough free disk space for this model: about {need / 1e9:.1f} GB needed, "
            f"{free / 1e9:.1f} GB free in {d}. Free up space or choose another models folder in Settings.")


# ----------------------------------------------------------------------------------------------
# Model management (Hugging Face cache inside the chosen models folder)
# ----------------------------------------------------------------------------------------------
def repo_for(name):
    load_engine()
    try:
        from faster_whisper.utils import _MODELS
        return _MODELS[name]
    except Exception:
        return f"Systran/faster-whisper-{name}"


def _valid(p):
    p = Path(p)
    return (p / "model.bin").exists() and (p / "config.json").exists()


def cached_path(name):
    hf = load_engine()["hf"]
    try:
        p = hf.snapshot_download(repo_for(name), cache_dir=str(get_models_dir()),
                                 local_files_only=True, allow_patterns=PATTERNS)
        return p if _valid(p) else None
    except Exception:
        return None


def remote_sha(name):
    hf = load_engine()["hf"]
    try:
        return hf.HfApi().model_info(repo_for(name), timeout=6).sha
    except Exception:
        return None


def check_model(name):
    """-> (kind, local_path, remote_sha); kind: ok | update | missing | offline | offline_missing"""
    local = cached_path(name)
    sha = remote_sha(name)
    if local is None:
        return ("missing" if sha else "offline_missing"), None, sha
    if sha is None:
        return "offline", local, None
    return ("ok" if Path(local).name == sha else "update"), local, sha


_BAR = {}


def get_bar_class():
    """tqdm replacement (built lazily) that reports aggregate download bytes and supports cancelling."""
    if "cls" not in _BAR:
        base = load_engine()["tqdm"]

        class _Bar(base):
            hook = None
            is_cancelled = None
            bars = {}

            def __init__(self, *a, **kw):
                kw.pop("name", None)
                super().__init__(*a, **kw)

            def update(self, n=1):
                if _Bar.is_cancelled and _Bar.is_cancelled():
                    raise _Cancelled()
                out = super().update(n)
                if self.unit == "B" and self.total and _Bar.hook:
                    _Bar.bars[id(self)] = (self.n, self.total)
                    done = sum(v[0] for v in _Bar.bars.values())
                    total = sum(v[1] for v in _Bar.bars.values())
                    if total > 0:
                        _Bar.hook(done, total)
                return out

        _BAR["cls"] = _Bar
    return _BAR["cls"]


def download_model(name, on_progress, is_cancelled):
    hf = load_engine()["hf"]
    Bar = get_bar_class()
    d = get_models_dir()
    d.mkdir(parents=True, exist_ok=True)
    Bar.bars, Bar.hook, Bar.is_cancelled = {}, on_progress, is_cancelled
    try:
        try:
            return hf.snapshot_download(repo_for(name), cache_dir=str(d),
                                        allow_patterns=PATTERNS, tqdm_class=Bar)
        except TypeError:  # very old huggingface_hub without tqdm_class
            return hf.snapshot_download(repo_for(name), cache_dir=str(d), allow_patterns=PATTERNS)
    finally:
        Bar.hook = Bar.is_cancelled = None


_MODEL = {"key": None, "obj": None}


def load_model(name, path, dev):
    key = (name, str(path), dev)
    if _MODEL["key"] == key:
        return _MODEL["obj"]
    _MODEL.update(key=None, obj=None)
    gc.collect()
    WhisperModel = load_engine()["WhisperModel"]
    try:
        obj = WhisperModel(str(path), device=dev, compute_type="auto" if dev == "cuda" else "int8")
    except Exception as e:
        if dev == "cuda":
            raise
        raise RuntimeError(f"Couldn't load the model files ({e}). If this keeps happening, "
                           f"delete this folder and run again:\n{get_models_dir()}") from e
    _MODEL.update(key=key, obj=obj)
    return obj


# ----------------------------------------------------------------------------------------------
# Background threads
# ----------------------------------------------------------------------------------------------
class CheckWorker(QThread):
    """Read-only: is the model installed / is there a newer version?"""
    result = Signal(str, str)

    def __init__(self, name):
        super().__init__()
        self.name = name

    def run(self):
        try:
            kind = check_model(self.name)[0]
        except Exception:
            kind = "offline"
        self.result.emit(self.name, kind)


class Job(QThread):
    status = Signal(str)
    busy = Signal()
    dl = Signal(int, str)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self._cancel = False
        self._skip = False
        self._last = -1

    def cancel(self):
        self._cancel = True

    def abort(self):
        self._cancel = True

    def cancelled(self):
        return self._cancel or self._skip

    def download(self, name, label):
        ensure_space(name)
        self.status.emit(label)
        self.busy.emit()
        self._last = -1

        def hook(done, total):
            pct = int(done * 100 / total)
            if pct != self._last:
                self._last = pct
                self.dl.emit(pct, f"{done / 1e9:.2f} / {total / 1e9:.2f} GB downloaded")

        try:
            return download_model(name, hook, self.cancelled)
        except _Cancelled:
            raise
        except Exception:
            if self.cancelled():
                raise _Cancelled() from None
            raise


class UpdateJob(Job):
    done = Signal(bool)

    def __init__(self, name):
        super().__init__()
        self.name = name

    def run(self):
        try:
            self.download(self.name, "Updating the model to the latest version...  (Cancel keeps the current one)")
            self.done.emit(False)
        except _Cancelled:
            self.done.emit(True)
        except Exception as e:
            self.failed.emit("Model update failed: " + friendly(e))


class MoveJob(Job):
    """Moves downloaded models from the old folder to the new one."""
    done = Signal(bool, str)

    def __init__(self, src, dst):
        super().__init__()
        self.src, self.dst = Path(src), Path(dst)

    def run(self):
        try:
            self.busy.emit()
            self.status.emit("Moving the downloaded models to the new folder...")
            self.dst.mkdir(parents=True, exist_ok=True)
            items = [p for p in self.src.iterdir() if p.name.startswith("models--")]
            skipped = 0
            for i, p in enumerate(items, 1):
                if self._cancel:
                    raise _Cancelled()
                target = self.dst / p.name
                if target.exists():
                    skipped += 1
                    continue
                shutil.move(str(p), str(target))
                self.dl.emit(int(i * 100 / len(items)), f"{i} of {len(items)} model folders moved")
            msg = "Models moved to the new folder."
            if skipped:
                msg += f" {skipped} already existed there and were left untouched."
            self.done.emit(False, msg)
        except _Cancelled:
            self.done.emit(True, "Move cancelled. Models that weren't moved yet stay in the old folder.")
        except Exception as e:
            self.failed.emit("Couldn't move the models: " + friendly(e))


class Worker(Job):
    segment = Signal(float, float, str)
    progress = Signal(float, float)
    reset = Signal()
    notice = Signal(str, str)
    device = Signal(str)
    lang = Signal(str, float)
    model_state = Signal(str, str)
    done = Signal(bool)

    def __init__(self, path, model_name, language, task, auto_update):
        super().__init__()
        self.path, self.model_name = path, model_name
        self.language, self.task, self.auto_update = language, task, auto_update
        self.phase = ""

    def cancel(self):
        if self.phase == "update":      # cancelling an update just skips it
            self._skip = True
        else:
            self._cancel = True

    # -- model resolution ------------------------------------------------------------------
    def _get_model(self):
        self.status.emit("Checking the model...")
        kind, local, _ = check_model(self.model_name)
        if kind == "offline_missing":
            raise RuntimeError("The model hasn't been downloaded yet and there's no internet connection. "
                               "Connect once to download it; after that it works offline.")
        if kind == "offline":
            self.notice.emit("Offline: using the installed model.", "info")
            return local
        if kind == "ok":
            return local
        if kind == "missing":
            self.phase = "download"
            path = self.download(self.model_name,
                                 f"Downloading the model (about {SIZES.get(self.model_name, '')}, one time only)...")
            self.phase = ""
            self.model_state.emit(self.model_name, "ok")
            return path
        # kind == "update"
        if not self.auto_update:
            self.notice.emit("A newer model version is available (auto-update is off).", "info")
            return local
        self.phase = "update"
        try:
            path = self.download(self.model_name,
                                 "A newer model version is available, updating...  (Cancel skips the update)")
            self.model_state.emit(self.model_name, "ok")
            return path
        except _Cancelled:
            if self._skip and not self._cancel:
                self._skip = False
                self.notice.emit("Update skipped, using the installed version.", "info")
                self.model_state.emit(self.model_name, "update")
                return local
            raise
        except RuntimeError as e:   # e.g. not enough disk space for the update: keep working with what we have
            if "disk space" in str(e):
                self.notice.emit("Not enough disk space to update the model, using the installed version.", "warn")
                self.model_state.emit(self.model_name, "update")
                return local
            raise
        finally:
            self.phase = ""

    # -- transcription ---------------------------------------------------------------------
    def _transcribe(self, model_path, dev):
        self.status.emit("Loading the model on the GPU..." if dev == "cuda" else "Loading the model on the CPU...")
        self.busy.emit()
        model = load_model(self.model_name, model_path, dev)
        self.device.emit(dev)
        if self._cancel:
            raise _Cancelled()
        self.status.emit("Reading the audio...")
        segments, info = model.transcribe(
            self.path,
            task=self.task,
            language=self.language,
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6],
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
            initial_prompt="This is a Bangla and English mixed conversation.",
        )
        total = float(info.duration or 0.0)
        self.lang.emit(info.language, float(info.language_probability))
        self.progress.emit(0.0, total)
        for seg in segments:
            if self._cancel:
                raise _Cancelled()
            text = seg.text.strip()
            if text:
                self.segment.emit(seg.start, seg.end, text)
            self.progress.emit(seg.end, total)
        self.progress.emit(total, total)

    def run(self):
        try:
            if not Path(self.path).is_file():
                raise FileNotFoundError("The selected file was moved or deleted. Please select it again.")
            self.busy.emit()
            self.status.emit("Starting the engine...")
            load_engine()       # instant if the background warm-up already finished
            model_path = self._get_model()
            for dev in (["cuda", "cpu"] if STATE["cuda"] else ["cpu"]):
                try:
                    self._transcribe(model_path, dev)
                    break
                except _Cancelled:
                    raise
                except Exception as e:
                    if dev == "cuda" and gpu_error(e):
                        STATE["cuda"] = False
                        _MODEL.update(key=None, obj=None)
                        gc.collect()
                        self.notice.emit("The GPU couldn't be used, so this ran on the CPU instead (slower).", "warn")
                        self.device.emit("cpu")
                        self.reset.emit()
                        continue
                    raise
            self.done.emit(False)
        except _Cancelled:
            self.done.emit(True)
        except Exception as e:
            self.failed.emit(friendly(e))


# ----------------------------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------------------------
def rgba(h, a):
    c = QColor(h)
    return f"rgba({c.red()},{c.green()},{c.blue()},{int(a * 255)})"


DARK = dict(name="dark", bg="#0e1014", card="#161920", card2="#1e222b", input="#11141a", border="#2a2f3b",
            text="#e9ebf1", sub="#8d95a8", accent="#7c9bff", accent_h="#9ab1ff", accent_text="#0b0e16",
            danger="#ff6b6b", ok="#46d39a", warn="#f2b84b", track="#262b36", dis_bg="#262d44", dis_text="#7482ad")
LIGHT = dict(name="light", bg="#f3f5f9", card="#ffffff", card2="#f1f3f8", input="#f8f9fc", border="#e0e4ec",
             text="#1a1e29", sub="#667085", accent="#4a67ea", accent_h="#3a55d4", accent_text="#ffffff",
             danger="#d03f3f", ok="#16905f", warn="#a96b0c", track="#e3e7f0", dis_bg="#c5cdf5", dis_text="#f6f7fe")
for _p in (DARK, LIGHT):
    _p["accent_soft"] = rgba(_p["accent"], .10)
    _p["info_bg"] = rgba(_p["accent"], .14)
    _p["warn_bg"] = rgba(_p["warn"], .16)
    _p["err_bg"] = rgba(_p["danger"], .14)
    _p["ok_bg"] = rgba(_p["ok"], .14)

QSS = Template("""
QMainWindow, QScrollArea { background:$bg; border:none; }
QWidget#canvas { background:$bg; }
QWidget { color:$text; font-size:13px; }
QLabel { background:transparent; }
QToolTip { background:$card; color:$text; border:1px solid $border; padding:4px 6px; }

QFrame#card { background:$card; border:1px solid $border; border-radius:14px; }
QFrame#drop { background:$card; border:2px dashed $border; border-radius:14px; }
QFrame#drop:hover, QFrame#drop[drag="true"] { border-color:$accent; background:$accent_soft; }
QFrame#drop[loaded="true"] { border:2px solid $accent; }

QLabel#title { font-size:22px; font-weight:700; }
QLabel#section { font-size:13px; font-weight:700; }
QLabel#muted { color:$sub; font-size:12px; }
QLabel#status { font-size:13px; font-weight:600; }
QLabel#percent { font-size:28px; font-weight:700; }
QLabel#dropIcon { font-size:28px; color:$accent; }
QLabel#dropTitle { font-size:14px; font-weight:700; }
QLabel#chip { background:$card2; border:1px solid $border; border-radius:11px; padding:3px 10px;
              color:$sub; font-weight:600; font-size:12px; }
QLabel#chip[gpu="true"] { color:$ok; border-color:$ok; }
QLabel#modelStatus { color:$sub; font-size:12px; }
QLabel#modelStatus[state="ok"] { color:$ok; }
QLabel#modelStatus[state="warn"] { color:$warn; }
QLabel#modelStatus[state="bad"] { color:$danger; }
QLabel#pathLabel { background:$input; border:1px solid $border; border-radius:8px; padding:6px 10px;
                   color:$sub; font-size:12px; }

QLabel#banner { border-radius:9px; padding:9px 12px; }
QLabel#banner[kind="info"] { background:$info_bg; color:$text; }
QLabel#banner[kind="warn"] { background:$warn_bg; color:$warn; }
QLabel#banner[kind="error"] { background:$err_bg; color:$danger; }
QLabel#banner[kind="ok"] { background:$ok_bg; color:$ok; }

QPushButton { background:$card2; border:1px solid $border; border-radius:9px; padding:8px 16px; font-weight:600; }
QPushButton:hover { border-color:$accent; }
QPushButton:disabled { color:$sub; background:$card; }
QPushButton#small { padding:5px 12px; font-size:12px; border-radius:8px; }
QPushButton#primary { background:$accent; color:$accent_text; border:none; padding:11px 18px; font-size:14px; }
QPushButton#primary:hover { background:$accent_h; }
QPushButton#primary:disabled { background:$dis_bg; color:$dis_text; }
QPushButton#danger { background:transparent; color:$danger; border:1px solid $danger; padding:11px 18px; }
QPushButton#danger:hover { background:$danger; color:$accent_text; }
QPushButton#danger:disabled { color:$sub; border-color:$border; }
QPushButton#accent { background:$accent; color:$accent_text; border:none; }
QPushButton#accent:hover { background:$accent_h; }
QPushButton#accent:disabled { background:$dis_bg; color:$dis_text; }

QComboBox { background:$input; border:1px solid $border; border-radius:8px; padding:6px 10px; min-height:20px; }
QComboBox:hover { border-color:$accent; }
QComboBox:disabled { color:$sub; }
QComboBox::drop-down { border:none; width:26px; }
QComboBox::down-arrow { image:url("$arrow"); width:12px; height:12px; }
QComboBox QAbstractItemView { background:$card; border:1px solid $border; outline:0; padding:4px;
                              selection-background-color:$accent; selection-color:$accent_text; }

QCheckBox { spacing:8px; background:transparent; }
QCheckBox::indicator { width:18px; height:18px; border-radius:5px; border:1px solid $border; background:$input; }
QCheckBox::indicator:hover { border-color:$accent; }
QCheckBox::indicator:checked { background:$accent; border-color:$accent; image:url("$check"); }

QPlainTextEdit { background:$input; border:1px solid $border; border-radius:10px; padding:10px;
                 font-size:14px; selection-background-color:$accent; selection-color:$accent_text; }
QPlainTextEdit:focus { border-color:$accent; }

QScrollBar:vertical { background:transparent; width:10px; margin:2px; }
QScrollBar::handle:vertical { background:$border; border-radius:3px; min-height:30px; }
QScrollBar::handle:vertical:hover { background:$sub; }
QScrollBar:horizontal { background:transparent; height:10px; margin:2px; }
QScrollBar::handle:horizontal { background:$border; border-radius:3px; min-width:30px; }
QScrollBar::add-line, QScrollBar::sub-line { width:0; height:0; }
QScrollBar::add-page, QScrollBar::sub-page { background:none; }
""")


def make_icons(pal):
    """Checkmark + dropdown-arrow PNGs for the stylesheet. Cached in the temp folder after the first run."""
    d = Path(tempfile.gettempdir())
    cp, ap = d / f"bw_check_v2_{pal['name']}.png", d / f"bw_arrow_v2_{pal['name']}.png"
    if cp.exists() and ap.exists():
        return cp.as_posix(), ap.as_posix()
    chk = QPixmap(18, 18)
    chk.fill(Qt.transparent)
    p = QPainter(chk)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor(pal["accent_text"]), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.drawPolyline(QPolygonF([QPointF(4.5, 9.5), QPointF(7.8, 12.8), QPointF(13.5, 5.8)]))
    p.end()
    arr = QPixmap(12, 12)
    arr.fill(Qt.transparent)
    p = QPainter(arr)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor(pal["sub"]), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.drawPolyline(QPolygonF([QPointF(2.5, 4.5), QPointF(6, 8), QPointF(9.5, 4.5)]))
    p.end()
    chk.save(str(cp))
    arr.save(str(ap))
    return cp.as_posix(), ap.as_posix()


def make_palette(pal):
    p = QPalette()
    for role, key in ((QPalette.Window, "bg"), (QPalette.WindowText, "text"), (QPalette.Base, "input"),
                      (QPalette.AlternateBase, "card"), (QPalette.Text, "text"), (QPalette.Button, "card2"),
                      (QPalette.ButtonText, "text"), (QPalette.ToolTipBase, "card"), (QPalette.ToolTipText, "text"),
                      (QPalette.Highlight, "accent"), (QPalette.HighlightedText, "accent_text"),
                      (QPalette.PlaceholderText, "sub")):
        p.setColor(role, QColor(pal[key]))
    return p


def system_is_dark():
    try:
        return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except Exception:
        return QGuiApplication.palette().window().color().lightness() < 128


def set_prop(w, name, value):
    w.setProperty(name, value)
    w.style().unpolish(w)
    w.style().polish(w)
    w.update()


# ----------------------------------------------------------------------------------------------
# Widgets
# ----------------------------------------------------------------------------------------------
class Bar(QWidget):
    """Slim custom progress bar with a smooth indeterminate mode."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(10)
        self._v, self._busy, self._phase = 0.0, False, 0.0
        self.track, self.fill = QColor("#262b36"), QColor("#7c9bff")
        self._t = QTimer(self)
        self._t.timeout.connect(self._step)

    def set_colors(self, track, fill):
        self.track, self.fill = QColor(track), QColor(fill)
        self.update()

    def set_value(self, v):
        self._busy = False
        self._t.stop()
        self._v = max(0.0, min(1.0, v))
        self.update()

    def set_busy(self):
        self._busy = True
        if not self._t.isActive():
            self._t.start(16)

    def _step(self):
        self._phase = (self._phase + 0.015) % 1.0
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        rad = r.height() / 2
        p.setPen(Qt.NoPen)
        p.setBrush(self.track)
        p.drawRoundedRect(r, rad, rad)
        path = QPainterPath()
        path.addRoundedRect(r, rad, rad)
        p.setClipPath(path)
        p.setBrush(self.fill)
        if self._busy:
            w = r.width() * 0.3
            x = -w + self._phase * (r.width() + w)
            p.drawRoundedRect(QRectF(x, 0, w, r.height()), rad, rad)
        else:
            p.drawRoundedRect(QRectF(0, 0, r.width() * self._v, r.height()), rad, rad)


class DropZone(QFrame):
    dropped = Signal(str, int)
    rejected = Signal(str)
    clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("drop")
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(120)
        self.drop_enabled = True
        v = QVBoxLayout(self)
        v.setAlignment(Qt.AlignCenter)
        v.setSpacing(3)
        self.icon = QLabel("♫")
        self.icon.setObjectName("dropIcon")
        self.title = QLabel("Drop an audio file here")
        self.title.setObjectName("dropTitle")
        self.sub = QLabel("or click to browse")
        self.sub.setObjectName("muted")
        for w in (self.icon, self.title, self.sub):
            w.setAlignment(Qt.AlignCenter)
            v.addWidget(w)

    def set_file(self, path, size_bytes):
        name = Path(path).name
        if len(name) > 38:
            name = name[:20] + "…" + name[-15:]
        self.icon.setText("✓")
        self.title.setText(name)
        self.sub.setText(f"{size_bytes / 1_048_576:.1f} MB  ·  click to change")
        self.setToolTip(str(path))
        set_prop(self, "loaded", True)

    def reset(self):
        self.icon.setText("♫")
        self.title.setText("Drop an audio file here")
        self.sub.setText("or click to browse")
        self.setToolTip("")
        set_prop(self, "loaded", False)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()

    def dragEnterEvent(self, e):
        if self.drop_enabled and e.mimeData().hasUrls():
            e.acceptProposedAction()
            set_prop(self, "drag", True)
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        set_prop(self, "drag", False)

    def dropEvent(self, e):
        set_prop(self, "drag", False)
        if not self.drop_enabled:
            e.ignore()
            return
        files = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        good = [f for f in files if f.lower().endswith(AUDIO_EXTS)]
        if good:
            self.dropped.emit(good[0], len(files))
        else:
            self.rejected.emit("That doesn't look like an audio or video file "
                               "(mp3, wav, m4a, mp4, ogg, flac, opus, ...).")


# ----------------------------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice to English")
        self.setMinimumSize(520, 560)
        self.settings = QSettings("MH", APP)
        self.path = None
        self.job = None
        self.segments = []
        self.words = 0
        self.t0 = 0.0
        self.t_tr = None
        self.declined = set()
        self._wide = None
        self._threads = set()
        self.theme_mode = self.settings.value("theme", "system")

        self._build_ui()
        self._load_settings()
        self._connect()

        geo = self.settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        else:
            self.resize(1060, 760)
        self._apply_mode()
        self.apply_theme()
        # Heavy libraries load AFTER the window has been painted.
        QTimer.singleShot(60, self._warm_up)

    # ---- startup -----------------------------------------------------------------------------
    def _warm_up(self):
        t = WarmupThread()
        t.ready.connect(self.on_engine_ready)
        self._track(t)
        t.start()

    def on_engine_ready(self, cuda):
        if not self.busy_now():
            self.on_device("cuda" if cuda else "cpu")
        self.check_model_status()

    # ---- UI construction ---------------------------------------------------------------------
    def _card(self):
        f = QFrame()
        f.setObjectName("card")
        return f

    def _label(self, text, name):
        l = QLabel(text)
        l.setObjectName(name)
        return l

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setCentralWidget(scroll)
        canvas = QWidget()
        canvas.setObjectName("canvas")
        scroll.setWidget(canvas)
        root = QVBoxLayout(canvas)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(14)

        # Header
        head = QHBoxLayout()
        tb = QVBoxLayout()
        tb.setSpacing(0)
        tb.addWidget(self._label("Voice to English", "title"))
        sub = self._label("Bangla + English speech, transcribed and translated with Whisper", "muted")
        sub.setWordWrap(True)
        tb.addWidget(sub)
        head.addLayout(tb, 1)
        self.chip = self._label("Starting…", "chip")
        set_prop(self.chip, "gpu", False)
        self.chip.setToolTip("Device used for transcription")
        self.theme_box = QComboBox()
        self.theme_box.addItems(["System", "Light", "Dark"])
        self.theme_box.setToolTip("Theme")
        head.addWidget(self.chip, 0, Qt.AlignVCenter)
        head.addWidget(self.theme_box, 0, Qt.AlignVCenter)
        root.addLayout(head)

        # Body (switches between side-by-side and stacked)
        self.body = QBoxLayout(QBoxLayout.LeftToRight)
        self.body.setSpacing(16)
        root.addLayout(self.body, 1)

        # ---- left column
        self.left = QWidget()
        lv = QVBoxLayout(self.left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(12)

        self.drop = DropZone()
        lv.addWidget(self.drop)

        sc = self._card()
        g = QGridLayout(sc)
        g.setContentsMargins(16, 14, 16, 14)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(8)
        g.addWidget(self._label("Settings", "section"), 0, 0, 1, 2)
        self.model_box = QComboBox()
        for name, label, _, _ in MODELS:
            self.model_box.addItem(label, name)
        self.model_status = self._label("Starting...", "modelStatus")
        self.model_status.setWordWrap(True)
        self.lang_box = QComboBox()
        self.lang_box.addItems(["Auto-detect", "Bangla (bn)", "English (en)"])
        self.lang_box.setToolTip("If most of the speech is Bangla, choose Bangla for better accuracy.")
        self.task_box = QComboBox()
        self.task_box.addItems(["Translate to English", "Transcribe only (original language)"])
        self.auto_check = QCheckBox("Keep the model up to date automatically")

        self.path_label = self._label("", "pathLabel")
        self.folder_btn = QPushButton("Change...")
        self.open_btn = QPushButton("Open")
        self.reset_btn = QPushButton("Default")
        for b in (self.folder_btn, self.open_btn, self.reset_btn):
            b.setObjectName("small")
        self.folder_btn.setToolTip("Choose where models are downloaded and stored")
        self.open_btn.setToolTip("Open the models folder in File Explorer")
        self.reset_btn.setToolTip("Go back to the default folder")
        fb = QHBoxLayout()
        fb.setSpacing(8)
        fb.addWidget(self.folder_btn)
        fb.addWidget(self.open_btn)
        fb.addWidget(self.reset_btn)
        fb.addStretch(1)

        g.addWidget(self._label("Model", "muted"), 1, 0)
        g.addWidget(self.model_box, 1, 1)
        g.addWidget(self.model_status, 2, 1)
        g.addWidget(self._label("Language", "muted"), 3, 0)
        g.addWidget(self.lang_box, 3, 1)
        g.addWidget(self._label("Mode", "muted"), 4, 0)
        g.addWidget(self.task_box, 4, 1)
        g.addWidget(self._label("Models folder", "muted"), 5, 0)
        g.addWidget(self.path_label, 5, 1)
        g.addLayout(fb, 6, 1)
        g.addWidget(self.auto_check, 7, 0, 1, 2)
        g.setColumnStretch(1, 1)
        lv.addWidget(sc)

        ar = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.setToolTip("Ctrl+Enter")
        self.start_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.setToolTip("Esc")
        self.cancel_btn.setEnabled(False)
        ar.addWidget(self.start_btn, 2)
        ar.addWidget(self.cancel_btn, 1)
        lv.addLayout(ar)

        pc = self._card()
        pv = QVBoxLayout(pc)
        pv.setContentsMargins(16, 14, 16, 14)
        pv.setSpacing(8)
        pr = QHBoxLayout()
        self.pct = self._label("0%", "percent")
        self.elapsed = self._label("", "muted")
        pr.addWidget(self.pct)
        pr.addStretch(1)
        pr.addWidget(self.elapsed, 0, Qt.AlignBottom)
        pv.addLayout(pr)
        self.bar = Bar()
        pv.addWidget(self.bar)
        self.status = self._label("Select an audio file to begin.", "status")
        self.status.setWordWrap(True)
        self.detail = self._label("", "muted")
        self.detail.setWordWrap(True)
        pv.addWidget(self.status)
        pv.addWidget(self.detail)
        lv.addWidget(pc)
        lv.addStretch(1)

        # ---- right column (output)
        self.right = self._card()
        rv = QVBoxLayout(self.right)
        rv.setContentsMargins(16, 14, 16, 14)
        rv.setSpacing(10)
        hr = QHBoxLayout()
        hr.addWidget(self._label("English text", "section"))
        hr.addStretch(1)
        self.ts_check = QCheckBox("Timestamps")
        hr.addWidget(self.ts_check)
        rv.addLayout(hr)
        self.banner = QLabel()
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.banner.hide()
        rv.addWidget(self.banner)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMinimumHeight(260)
        self.out.setPlaceholderText("Your English text will appear here as it is processed.")
        rv.addWidget(self.out, 1)
        fr = QHBoxLayout()
        self.count = self._label("", "muted")
        self.clear_btn = QPushButton("Clear")
        self.save_btn = QPushButton("Save...")
        self.copy_btn = QPushButton("Copy text")
        self.copy_btn.setObjectName("accent")
        self.copy_btn.setToolTip("Ctrl+Shift+C")
        fr.addWidget(self.count)
        fr.addStretch(1)
        for b in (self.clear_btn, self.save_btn, self.copy_btn):
            fr.addWidget(b)
        rv.addLayout(fr)
        self.set_output_buttons(False)

        self.body.addWidget(self.left, 0)
        self.body.addWidget(self.right, 1)

        # Elapsed-time ticker
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)

    def _connect(self):
        self.drop.clicked.connect(self.browse)
        self.drop.dropped.connect(self.on_dropped)
        self.drop.rejected.connect(lambda m: self.show_banner(m, "warn"))
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn.clicked.connect(self.cancel)
        self.copy_btn.clicked.connect(self.copy)
        self.save_btn.clicked.connect(self.save)
        self.clear_btn.clicked.connect(self.clear)
        self.folder_btn.clicked.connect(self.choose_models_dir)
        self.open_btn.clicked.connect(self.open_models_dir)
        self.reset_btn.clicked.connect(lambda: self.apply_models_dir(DEFAULT_MODELS_DIR))
        self.ts_check.toggled.connect(self.on_ts_toggled)
        self.theme_box.currentIndexChanged.connect(self.on_theme_changed)
        self.model_box.currentIndexChanged.connect(self.on_model_changed)
        self.lang_box.currentIndexChanged.connect(self._save)
        self.task_box.currentIndexChanged.connect(self._save)
        self.auto_check.toggled.connect(self.on_auto_toggled)
        for key, fn in (("Ctrl+O", self.browse), ("Ctrl+Return", self.start), ("Ctrl+Enter", self.start),
                        ("Esc", self.cancel), ("Ctrl+Shift+C", self.copy)):
            QShortcut(QKeySequence(key), self, activated=fn)
        try:
            QGuiApplication.styleHints().colorSchemeChanged.connect(
                lambda *_: self.apply_theme() if self.theme_mode == "system" else None)
        except Exception:
            pass

    # ---- settings / theme / layout -----------------------------------------------------------
    def _load_settings(self):
        s = self.settings
        i = self.model_box.findData(s.value("model", "large-v3"))
        self.model_box.setCurrentIndex(max(0, i))
        self.lang_box.setCurrentIndex(int(s.value("lang", 0)))
        self.task_box.setCurrentIndex(int(s.value("task", 0)))
        self.auto_check.setChecked(s.value("auto_update", True, type=bool))
        self.ts_check.setChecked(s.value("timestamps", False, type=bool))
        self.theme_box.setCurrentIndex({"system": 0, "light": 1, "dark": 2}.get(self.theme_mode, 0))

        saved = s.value("models_dir")
        if saved:
            err = validate_dir(saved)
            if err:
                # e.g. an external drive that isn't plugged in. Keep the saved setting untouched.
                self.show_banner(f"Your models folder ({saved}) isn't available: {err} "
                                 f"Using the default folder for now.", "warn")
            else:
                set_models_dir(saved)
        self.refresh_folder_ui()

    def _save(self, *_):
        s = self.settings
        s.setValue("model", self.model_box.currentData())
        s.setValue("lang", self.lang_box.currentIndex())
        s.setValue("task", self.task_box.currentIndex())
        s.setValue("auto_update", self.auto_check.isChecked())
        s.setValue("timestamps", self.ts_check.isChecked())
        s.setValue("theme", self.theme_mode)

    def apply_theme(self):
        dark = system_is_dark() if self.theme_mode == "system" else self.theme_mode == "dark"
        pal = DARK if dark else LIGHT
        chk, arr = make_icons(pal)
        app = QApplication.instance()
        app.setPalette(make_palette(pal))
        app.setStyleSheet(QSS.safe_substitute(pal, check=chk, arrow=arr))
        self.bar.set_colors(pal["track"], pal["accent"])

    def on_theme_changed(self, i):
        self.theme_mode = ("system", "light", "dark")[i]
        self._save()
        self.apply_theme()

    def _apply_mode(self):
        wide = self.width() >= 920
        if wide == self._wide:
            return
        self._wide = wide
        self.body.setDirection(QBoxLayout.LeftToRight if wide else QBoxLayout.TopToBottom)
        if wide:
            self.left.setFixedWidth(390)
        else:
            self.left.setMinimumWidth(0)
            self.left.setMaximumWidth(16777215)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._apply_mode()

    # ---- banner / small helpers --------------------------------------------------------------
    def show_banner(self, text, kind="info"):
        self.banner.setText(text)
        set_prop(self.banner, "kind", kind)
        self.banner.show()

    def hide_banner(self):
        self.banner.hide()

    def busy_now(self):
        return self.job is not None and self.job.isRunning()

    def set_output_buttons(self, on):
        for b in (self.copy_btn, self.save_btn, self.clear_btn):
            b.setEnabled(on)

    def set_busy(self, running):
        self.drop.drop_enabled = not running
        self.start_btn.setEnabled(not running and self.path is not None)
        self.cancel_btn.setEnabled(running)
        for w in (self.model_box, self.lang_box, self.task_box, self.folder_btn, self.reset_btn):
            w.setEnabled(not running)
        if not running:
            self.refresh_folder_ui()

    def tick(self):
        self.elapsed.setText(f"Elapsed {fmt(time.time() - self.t0)}")

    def _track(self, thread):
        self._threads.add(thread)
        thread.finished.connect(lambda t=thread: self._threads.discard(t))

    # ---- models folder -----------------------------------------------------------------------
    def refresh_folder_ui(self):
        d = get_models_dir()
        self.path_label.setText(short_path(d))
        self.path_label.setToolTip(str(d))
        is_default = d == DEFAULT_MODELS_DIR
        self.reset_btn.setEnabled(not is_default and not self.busy_now())

    def open_models_dir(self):
        d = get_models_dir()
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.show_banner(f"Couldn't open the folder: {e.strerror or e}", "error")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    def choose_models_dir(self):
        if self.busy_now():
            return
        d = QFileDialog.getExistingDirectory(self, "Choose where to store the models", str(get_models_dir()))
        if not d:
            return
        new = Path(d)
        # Don't dump model folders into a folder full of other stuff (Documents, a drive root, ...)
        try:
            non_empty = any(new.iterdir())
        except OSError:
            non_empty = False
        if non_empty and not has_models(new):
            new = new / "BanglaWhisper models"
        self.apply_models_dir(new)

    def apply_models_dir(self, new):
        if self.busy_now():
            return
        new = Path(new)
        old = get_models_dir()
        try:
            same = new.resolve() == old.resolve()
            nested = (new.resolve().is_relative_to(old.resolve()) or old.resolve().is_relative_to(new.resolve()))
        except (OSError, ValueError):
            same, nested = False, False
        if same:
            return
        if nested:
            self.show_banner("Choose a folder that isn't inside (or around) the current models folder.", "error")
            return
        err = validate_dir(new)
        if err:
            self.show_banner(err, "error")
            return

        move = False
        if has_models(old):
            box = QMessageBox(QMessageBox.Question, "Move existing models?",
                              "You already have models downloaded in the current folder.\n\n"
                              "Move them to the new folder, or leave them where they are? "
                              "(Models left behind will be downloaded again when needed.)", parent=self)
            mv = box.addButton("Move them", QMessageBox.AcceptRole)
            box.addButton("Leave them", QMessageBox.ActionRole)
            cancel = box.addButton(QMessageBox.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is cancel or clicked is None:
                return
            move = clicked is mv

        set_models_dir(new)
        self.settings.setValue("models_dir", str(new))
        self.refresh_folder_ui()
        _MODEL.update(key=None, obj=None)
        gc.collect()
        if move:
            job = MoveJob(old, new)
            job.done.connect(self.on_move_done)
            self.begin_job(job)
        else:
            self.show_banner(f"Models will now be stored in {new}", "info")
            self.check_model_status()

    def on_move_done(self, cancelled, msg):
        self.status.setText(msg)
        self.show_banner(msg, "info" if cancelled else "ok")
        if not cancelled:
            self.pct.setText("100%")
            self.bar.set_value(1.0)
        self.check_model_status()

    # ---- model status / auto-update ----------------------------------------------------------
    def on_model_changed(self, _):
        self._save()
        self.check_model_status()

    def on_auto_toggled(self, _):
        self._save()
        self.declined.clear()
        self.check_model_status()

    def check_model_status(self):
        name = self.model_box.currentData()
        self.model_status.setText("Checking...")
        set_prop(self.model_status, "state", "")
        t = CheckWorker(name)
        t.result.connect(self.on_model_checked)
        self._track(t)
        t.start()

    def on_model_checked(self, name, kind):
        if name != self.model_box.currentData():
            return
        texts = {
            "ok": "Installed  ·  up to date",
            "update": "Installed  ·  update available",
            "missing": f"Not downloaded yet  ·  about {SIZES.get(name, '')} on first use",
            "offline": "Installed  ·  couldn't check for updates (offline?)",
            "offline_missing": "Not downloaded  ·  needs the internet once",
        }
        self.model_status.setText(texts.get(kind, ""))
        set_prop(self.model_status, "state", {"ok": "ok", "update": "warn", "offline_missing": "bad"}.get(kind, ""))
        if (kind == "update" and self.auto_check.isChecked() and self.job is None
                and name not in self.declined):
            self.start_update(name)

    def start_update(self, name):
        job = UpdateJob(name)
        job.done.connect(lambda cancelled, n=name: self.on_update_done(cancelled, n))
        self.begin_job(job)

    def on_update_done(self, cancelled, name):
        if cancelled:
            self.declined.add(name)
            self.status.setText("Update skipped. Using the installed version.")
            self.model_status.setText("Installed  ·  update available (skipped)")
        else:
            self.status.setText("Model updated to the latest version.")
            self.pct.setText("100%")
            self.bar.set_value(1.0)
            self.model_status.setText("Installed  ·  up to date")
            set_prop(self.model_status, "state", "ok")

    # ---- file selection ----------------------------------------------------------------------
    def browse(self):
        if self.busy_now():
            return
        start = self.settings.value("last_dir", str(Path.home()))
        f, _ = QFileDialog.getOpenFileName(
            self, "Select an audio or video file", start,
            "Audio/Video (" + " ".join("*" + e for e in AUDIO_EXTS) + ");;All files (*)")
        if f:
            self.set_file(f)

    def on_dropped(self, f, count):
        self.set_file(f)
        if count > 1:
            self.show_banner("Only one file can be processed at a time. Using the first audio file you dropped.",
                             "info")

    def set_file(self, f):
        p = Path(f)
        try:
            size = p.stat().st_size
        except OSError:
            self.show_banner("That file can't be opened. It may have been moved or you don't have access.", "error")
            return
        if not p.is_file():
            self.show_banner("That isn't a file.", "error")
            return
        if size == 0:
            self.show_banner("That file is empty (0 bytes).", "error")
            return
        self.path = str(p)
        self.settings.setValue("last_dir", str(p.parent))
        self.drop.set_file(self.path, size)
        self.start_btn.setEnabled(not self.busy_now())
        self.hide_banner()
        self.status.setText("Ready. Press Start.")
        self.detail.setText("")
        self.pct.setText("0%")
        self.elapsed.setText("")
        self.bar.set_value(0)

    # ---- job control -------------------------------------------------------------------------
    def begin_job(self, job):
        self.job = job
        job.status.connect(self.status.setText)
        job.busy.connect(self.on_busy)
        job.dl.connect(self.on_dl)
        job.failed.connect(self.on_failed)
        job.finished.connect(self.on_finished)
        self._track(job)
        self.set_busy(True)
        self.t0 = time.time()
        self.t_tr = None
        self.elapsed.setText("Elapsed 00:00")
        self.timer.start(500)
        job.start()

    def start(self):
        if self.busy_now() or not self.path:
            return
        if not Path(self.path).is_file():
            self.show_banner("The file was moved or deleted. Please select it again.", "error")
            self.path = None
            self.drop.reset()
            self.start_btn.setEnabled(False)
            return
        lang = (None, "bn", "en")[self.lang_box.currentIndex()]
        task = "translate" if self.task_box.currentIndex() == 0 else "transcribe"
        name = self.model_box.currentData()
        self.clear()
        self.hide_banner()
        self.detail.setText("")
        if STATE["cuda"] is False and name in ("large-v3", "large-v2"):
            self.show_banner("No compatible GPU was found, so this will run on the CPU. Large models are slow "
                             "on CPU. 'medium' is a good compromise if this takes too long.", "warn")
        w = Worker(self.path, name, lang, task, self.auto_check.isChecked())
        w.segment.connect(self.on_segment)
        w.progress.connect(self.on_progress)
        w.reset.connect(self.clear)
        w.notice.connect(self.show_banner)
        w.device.connect(self.on_device)
        w.lang.connect(self.on_lang)
        w.model_state.connect(self.on_model_checked)
        w.done.connect(self.on_done)
        self.begin_job(w)

    def cancel(self):
        if self.busy_now():
            self.job.cancel()
            self.status.setText("Cancelling...")

    # ---- job signal handlers -----------------------------------------------------------------
    def on_busy(self):
        self.bar.set_busy()
        self.pct.setText("…")

    def on_dl(self, pct, text):
        self.bar.set_value(pct / 100)
        self.pct.setText(f"{pct}%")
        self.detail.setText(text)

    def on_device(self, dev):
        self.chip.setText("GPU · CUDA" if dev == "cuda" else "CPU")
        set_prop(self.chip, "gpu", dev == "cuda")

    def on_lang(self, code, prob):
        self.status.setText(f"Transcribing  ·  detected language: {LANG_NAMES.get(code, code)} ({prob:.0%})")

    def on_progress(self, pos, total):
        if total <= 0:
            self.bar.set_busy()
            self.pct.setText("…")
            return
        if self.t_tr is None:
            self.t_tr = time.time()
        frac = min(1.0, pos / total)
        self.bar.set_value(frac)
        self.pct.setText(f"{int(frac * 100)}%")
        el = time.time() - self.t_tr
        text = f"{fmt(pos)} / {fmt(total)}"
        if el > 1 and pos > 0:
            speed = pos / el
            text += f"   ·   {speed:.1f}× real-time"
            if frac < 1:
                text += f"   ·   about {fmt((total - pos) / speed)} left"
        self.detail.setText(text)

    def on_segment(self, s, e, text):
        self.segments.append((s, e, text))
        self.words += len(text.split())
        self.append_line(s, e, text)
        self.set_output_buttons(True)
        self.count.setText(f"{self.words:,} words")

    def on_done(self, cancelled):
        took = fmt(time.time() - self.t0)
        if cancelled:
            self.status.setText(f"Cancelled after {took}.")
            if self.segments:
                self.show_banner("Cancelled. The text processed so far is kept below.", "info")
        elif not self.segments:
            self.status.setText("Finished, but no speech was found.")
            self.show_banner("No speech detected. Check that the recording isn't silent, or pick the language "
                             "manually (Bangla or English) and try again.", "warn")
        else:
            self.status.setText(f"Done in {took}.")
            self.pct.setText("100%")
            self.bar.set_value(1.0)
            QApplication.alert(self)

    def on_failed(self, msg):
        self.status.setText("Something went wrong.")
        self.show_banner(msg, "error")
        self.bar.set_value(0)
        self.pct.setText("0%")

    def on_finished(self):
        self.timer.stop()
        self.job = None
        self.set_busy(False)

    # ---- output ------------------------------------------------------------------------------
    def append_line(self, s, e, text):
        sb = self.out.verticalScrollBar()
        stick = sb.value() >= sb.maximum() - 8
        empty = self.out.document().isEmpty()
        if self.ts_check.isChecked():
            piece = ("" if empty else "\n") + f"[{fmt(s)} - {fmt(e)}] {text}"
        else:
            piece = ("" if empty else " ") + text
        cur = self.out.textCursor()
        cur.movePosition(QTextCursor.End)
        cur.insertText(piece)
        if stick:
            sb.setValue(sb.maximum())

    def on_ts_toggled(self, _):
        self._save()
        if self.ts_check.isChecked():
            text = "\n".join(f"[{fmt(s)} - {fmt(e)}] {t}" for s, e, t in self.segments)
        else:
            text = " ".join(t for _, _, t in self.segments)
        self.out.setPlainText(text)

    def copy(self):
        text = self.out.toPlainText().strip()
        if not text:
            return
        QApplication.clipboard().setText(text)
        self.copy_btn.setText("Copied!")
        QTimer.singleShot(1500, lambda: self.copy_btn.setText("Copy text"))

    def save(self):
        if not self.segments:
            return
        base = Path(self.path).stem if self.path else "transcript"
        folder = Path(self.path).parent if self.path else Path.home()
        tag = "english" if self.task_box.currentIndex() == 0 else "transcript"
        f, flt = QFileDialog.getSaveFileName(self, "Save", str(folder / f"{base}_{tag}.txt"),
                                             "Text file (*.txt);;Subtitles (*.srt)")
        if not f:
            return
        p = Path(f)
        if not p.suffix:
            p = p.with_suffix(".srt" if "srt" in flt.lower() else ".txt")
        try:
            if p.suffix.lower() == ".srt":
                data = "\n".join(f"{i}\n{srt_ts(s)} --> {srt_ts(e)}\n{t}\n"
                                 for i, (s, e, t) in enumerate(self.segments, 1))
            else:
                data = self.out.toPlainText()
            p.write_text(data, encoding="utf-8")
            self.show_banner(f"Saved to {p}", "ok")
        except OSError as e:
            self.show_banner(f"Couldn't save the file: {e.strerror or e}", "error")

    def clear(self):
        self.segments.clear()
        self.words = 0
        self.out.clear()
        self.count.setText("")
        self.set_output_buttons(False)

    # ---- shutdown ----------------------------------------------------------------------------
    def closeEvent(self, e):
        if self.busy_now():
            if QMessageBox.question(self, "Quit?", "Work is still running. Cancel it and quit?") != QMessageBox.Yes:
                e.ignore()
                return
            self.job.abort()
        for t in list(self._threads):
            if t.isRunning() and not t.wait(3000):
                t.terminate()
                t.wait(1000)
        self.settings.setValue("geometry", self.saveGeometry())
        self._save()
        e.accept()


def main():
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("mh.banglawhisper")
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI"))
    win = MainWindow()
    # set icon
    icon_path = Path(__file__).parent / "app.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()