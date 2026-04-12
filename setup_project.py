"""
Media Forge - Full Automated Setup Script

Run this ONCE from the project root directory:
    python setup_project.py

What it does (in order):
    1.  Creates Python virtual environment (venv)
    2.  Upgrades pip
    3.  Installs PyTorch (CPU or CUDA, auto-detected)
    4.  Installs all pip packages from requirements.txt
    5.  Installs mediaforge package in editable mode (pip install -e .)
    6.  Creates .env from .env.example
    7.  Creates required directories
    8.  Checks FFmpeg
    9.  Checks Tesseract OCR
    10. Downloads Whisper base model (~142 MB)
    11. Downloads YOLOv8n model (~6 MB)
    12. Downloads U2Net model (~176 MB)
    13. Downloads EasyOCR models (~200 MB)

Works on Windows and Linux. All models pre-downloaded for offline use.
"""

import os
import sys
import shutil
import subprocess
import platform
import tempfile
import time
import threading

# ─── ANSI Colors ─────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IS_WINDOWS = platform.system() == "Windows"

# Fix stdout/stderr encoding for Windows (Turkish locale defaults to cp1254
# which can't handle Unicode block chars or model download output)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if IS_WINDOWS:
    VENV_PYTHON = os.path.join(PROJECT_DIR, "venv", "Scripts", "python.exe")
    VENV_PIP = os.path.join(PROJECT_DIR, "venv", "Scripts", "pip.exe")
else:
    VENV_PYTHON = os.path.join(PROJECT_DIR, "venv", "bin", "python")
    VENV_PIP = os.path.join(PROJECT_DIR, "venv", "bin", "pip")


def banner():
    print(f"""
{CYAN}{BOLD}  ███╗   ███╗███████╗██████╗ ██╗ █████╗
  ████╗ ████║██╔════╝██╔══██╗██║██╔══██╗
  ██╔████╔██║█████╗  ██║  ██║██║███████║
  ██║╚██╔╝██║██╔══╝  ██║  ██║██║██╔══██║
  ██║ ╚═╝ ██║███████╗██████╔╝██║██║  ██║
  ╚═╝     ╚═╝╚══════╝╚═════╝ ╚═╝╚═╝  ╚═╝

  ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
  ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
  █████╗  ██║   ██║██████╔╝██║  ███╗█████╗
  ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
  ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
  ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
{RESET}
  {DIM}Automated Project Setup{RESET}
  {DIM}{'─' * 44}{RESET}
""")


# ─── Output helpers ──────────────────────────────────────────────────────────

def step_header(number, total, message):
    print(f"\n{CYAN}{BOLD}[{number}/{total}]{RESET} {BOLD}{message}{RESET}")
    print("─" * 64)


def ok(msg):
    print(f"  {GREEN}✓ {msg}{RESET}")


def fail(msg):
    print(f"  {RED}✗ {msg}{RESET}")


def warn(msg):
    print(f"  {YELLOW}! {msg}{RESET}")


def info(msg):
    print(f"  {msg}")


# ─── Spinner for long-running tasks ─────────────────────────────────────────

class Spinner:
    """Animated spinner that shows elapsed time while a subprocess runs."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message):
        self.message = message
        self._stop = threading.Event()
        self._thread = None

    def _spin(self):
        i = 0
        start = time.time()
        while not self._stop.is_set():
            elapsed = time.time() - start
            frame = self.FRAMES[i % len(self.FRAMES)]
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            print(f"\r  {CYAN}{frame}{RESET} {self.message} {DIM}({time_str}){RESET}    ", end="", flush=True)
            i += 1
            self._stop.wait(0.1)
        print(f"\r{' ' * 80}\r", end="", flush=True)

    def __enter__(self):
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()


# ─── Command runners ────────────────────────────────────────────────────────

def run_cmd(cmd, spinner_msg=None, show_output=False, timeout=600):
    """
    Run a shell command. Returns True on success.
    Uses UTF-8 encoding with error replacement to avoid codec crashes.
    Shows a spinner with elapsed time while running.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    kwargs = {
        "shell": True,
        "env": env,
        "cwd": PROJECT_DIR,
        "timeout": timeout,
    }

    try:
        if show_output:
            result = subprocess.run(cmd, **kwargs)
            return result.returncode == 0

        if spinner_msg:
            with Spinner(spinner_msg):
                result = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    encoding="utf-8", errors="replace", **kwargs
                )
        else:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                encoding="utf-8", errors="replace", **kwargs
            )

        if result.returncode != 0 and result.stdout:
            last_lines = result.stdout.strip().split("\n")[-3:]
            for line in last_lines:
                info(f"  {DIM}{line}{RESET}")

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        fail(f"Command timed out after {timeout}s")
        return False


def run_pip(args, spinner_msg):
    """Run pip inside venv with progress visible via spinner."""
    cmd = f'"{VENV_PIP}" {args}'
    return run_cmd(cmd, spinner_msg=spinner_msg)


def run_venv_script(script_content, spinner_msg=None, timeout=600):
    """
    Write a temp .py file and run it inside venv.
    Avoids shell quoting issues and encoding problems.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="mf_setup_", dir=PROJECT_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(script_content)

        cmd = f'"{VENV_PYTHON}" "{tmp_path}"'
        return run_cmd(cmd, spinner_msg=spinner_msg, timeout=timeout)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def check_venv_import(module_name, timeout=120):
    """Check if a module can be imported inside venv (with timeout)."""
    cmd = f'"{VENV_PYTHON}" -c "import {module_name}"'
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        result = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="utf-8", errors="replace", timeout=timeout, env=env
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


# ─── Steps ───────────────────────────────────────────────────────────────────

TOTAL = 13


def step_01_venv():
    step_header(1, TOTAL, "Creating virtual environment")

    if os.path.exists(VENV_PYTHON):
        ok("Virtual environment already exists (venv/)")
        return True

    success = run_cmd(
        f'"{sys.executable}" -m venv venv',
        spinner_msg="Creating venv..."
    )

    if success and os.path.exists(VENV_PYTHON):
        ok("Virtual environment created at venv/")
        return True
    else:
        fail("Failed to create virtual environment")
        return False


def step_02_pip():
    step_header(2, TOTAL, "Upgrading pip")

    success = run_cmd(
        f'"{VENV_PYTHON}" -m pip install --upgrade pip',
        spinner_msg="Upgrading pip..."
    )

    if success:
        ok("pip upgraded to latest version")
    else:
        warn("pip upgrade failed, continuing anyway")
    return True


def step_03_pytorch():
    step_header(3, TOTAL, "Installing PyTorch")

    if check_venv_import("torch"):
        result = subprocess.run(
            f'"{VENV_PYTHON}" -c "import torch; print(torch.__version__)"',
            shell=True, capture_output=True, encoding="utf-8", errors="replace"
        )
        ver = result.stdout.strip() if result.returncode == 0 else "unknown"
        ok(f"PyTorch already installed (v{ver})")
        return True

    has_gpu = shutil.which("nvidia-smi") is not None

    if has_gpu:
        info("NVIDIA GPU detected — installing PyTorch with CUDA (~2.5 GB)...")
        index_url = "https://download.pytorch.org/whl/cu121"
    else:
        info("No NVIDIA GPU detected — installing PyTorch CPU (~700 MB)...")
        index_url = "https://download.pytorch.org/whl/cpu"

    success = run_pip(
        f"install torch torchvision --index-url {index_url}",
        spinner_msg="Downloading & installing PyTorch..."
    )

    if success:
        ok(f"PyTorch installed ({'CUDA' if has_gpu else 'CPU'})")
    else:
        fail("PyTorch installation failed")
    return success


def step_04_packages():
    step_header(4, TOTAL, "Installing Python packages from requirements.txt")

    req_file = os.path.join(PROJECT_DIR, "requirements.txt")
    if not os.path.exists(req_file):
        fail("requirements.txt not found!")
        return False

    success = run_pip(
        f'install -r "{req_file}"',
        spinner_msg="Installing packages..."
    )

    if success:
        ok("All Python packages installed")
    else:
        fail("Some packages failed to install")
        info("Run manually to see errors:")
        info(f'  "{VENV_PIP}" install -r requirements.txt')
    return success


def step_05_install_project():
    step_header(5, TOTAL, "Installing mediaforge package (editable mode)")

    setup_file = os.path.join(PROJECT_DIR, "setup.py")
    if not os.path.exists(setup_file):
        fail("setup.py not found!")
        return False

    success = run_pip(
        f'install -e "{PROJECT_DIR}"',
        spinner_msg="Installing mediaforge in editable mode..."
    )

    if success:
        ok("mediaforge package installed (editable)")
    else:
        fail("mediaforge package install failed")
    return success


def step_06_env():
    step_header(6, TOTAL, "Creating .env configuration file")

    env_file = os.path.join(PROJECT_DIR, ".env")
    env_example = os.path.join(PROJECT_DIR, ".env.example")

    if os.path.exists(env_file):
        ok(".env file already exists")
        return True

    if not os.path.exists(env_example):
        fail(".env.example not found!")
        return False

    shutil.copy2(env_example, env_file)
    ok(".env created from .env.example")
    return True


def step_07_dirs():
    step_header(7, TOTAL, "Creating project directories")

    dirs = [
        "output", "temp", "cache", "logs", "models",
        "models/whisper", "models/yolo",
        "models/style_transfer", "models/super_resolution",
    ]

    for d in dirs:
        os.makedirs(os.path.join(PROJECT_DIR, d), exist_ok=True)

    ok(f"Created {len(dirs)} directories")
    return True


def step_08_ffmpeg():
    step_header(8, TOTAL, "Checking FFmpeg")

    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if ffmpeg_path and ffprobe_path:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True,
            encoding="utf-8", errors="replace"
        )
        line = (result.stdout or "").split("\n")[0]
        ok(f"FFmpeg found: {line}")
        return True

    fail("FFmpeg NOT found in PATH!")
    info("")
    info("FFmpeg is REQUIRED for all video/audio operations.")
    if IS_WINDOWS:
        info("  Install:  winget install FFmpeg")
        info("  Or:       https://www.gyan.dev/ffmpeg/builds/")
    else:
        info("  Install:  sudo apt install ffmpeg")
    return False


def step_09_tesseract():
    step_header(9, TOTAL, "Checking Tesseract OCR")

    if shutil.which("tesseract"):
        result = subprocess.run(
            ["tesseract", "--version"], capture_output=True,
            encoding="utf-8", errors="replace"
        )
        line = (result.stdout or result.stderr or "").split("\n")[0]
        ok(f"Tesseract found: {line}")
        return True

    warn("Tesseract OCR not found (optional — only for reading text from images)")
    if IS_WINDOWS:
        info("  Install: https://github.com/UB-Mannheim/tesseract/wiki")
    else:
        info("  Install: sudo apt install tesseract-ocr")
    return True


def step_10_whisper():
    step_header(10, TOTAL, "Downloading Whisper model (speech-to-text, ~142 MB)")

    script = """
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    import whisper
except ImportError:
    print('ERROR: whisper not installed', flush=True)
    print('Fix: pip install openai-whisper', flush=True)
    sys.exit(1)
try:
    print('Downloading Whisper base model...', flush=True)
    whisper.load_model('base')
    print('DONE', flush=True)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr, flush=True)
    sys.exit(1)
"""
    success = run_venv_script(script, spinner_msg="Downloading Whisper base model (~142 MB)...", timeout=300)

    if success:
        ok("Whisper base model downloaded and cached")
    else:
        fail("Whisper model download failed")
    return success


def step_11_yolo():
    step_header(11, TOTAL, "Downloading YOLOv8n model (object detection, ~6 MB)")

    script = """
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    from ultralytics import YOLO
except ImportError:
    print('ERROR: ultralytics not installed', flush=True)
    print('Fix: pip install ultralytics', flush=True)
    sys.exit(1)
try:
    print('Downloading YOLOv8n model...', flush=True)
    YOLO('yolov8n.pt')
    print('DONE', flush=True)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr, flush=True)
    sys.exit(1)
"""
    success = run_venv_script(script, spinner_msg="Downloading YOLOv8n model (~6 MB)...", timeout=120)

    if success:
        ok("YOLOv8n model downloaded and cached")
    else:
        fail("YOLOv8 model download failed")
    return success


def step_12_rembg():
    step_header(12, TOTAL, "Downloading U2Net model (background removal, ~176 MB)")

    script = """
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    import onnxruntime
except ImportError:
    print('ERROR: onnxruntime not installed', flush=True)
    print('Fix: pip install onnxruntime', flush=True)
    sys.exit(1)
try:
    from rembg import new_session
    print('Downloading U2Net model...', flush=True)
    new_session('u2net')
    print('DONE', flush=True)
except ImportError as e:
    print(f'ERROR: {e}', flush=True)
    print('Fix: pip install rembg onnxruntime', flush=True)
    sys.exit(1)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr, flush=True)
    sys.exit(1)
"""
    info("This step may take 1-2 minutes (large model + slow import)...")
    success = run_venv_script(script, spinner_msg="Downloading U2Net model (~176 MB)...", timeout=300)

    if success:
        ok("U2Net model downloaded and cached")
    else:
        fail("U2Net model download failed")
    return success


def step_13_easyocr():
    step_header(13, TOTAL, "Downloading EasyOCR models (text recognition, ~200 MB)")

    script = """
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    import easyocr
except ImportError:
    print('ERROR: easyocr not installed', flush=True)
    print('Fix: pip install easyocr', flush=True)
    sys.exit(1)
try:
    print('Downloading EasyOCR models...', flush=True)
    easyocr.Reader(['en'], gpu=False, verbose=False)
    print('DONE', flush=True)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr, flush=True)
    sys.exit(1)
"""
    success = run_venv_script(script, spinner_msg="Downloading EasyOCR models (~200 MB)...", timeout=300)

    if success:
        ok("EasyOCR models downloaded and cached")
    else:
        fail("EasyOCR model download failed")
    return success


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary(results):
    names = [
        "Virtual environment",
        "Pip upgrade",
        "PyTorch",
        "Python packages",
        "mediaforge package",
        ".env config",
        "Directories",
        "FFmpeg",
        "Tesseract OCR",
        "Whisper model",
        "YOLOv8 model",
        "U2Net model",
        "EasyOCR models",
    ]

    print(f"\n{CYAN}{BOLD}{'=' * 64}{RESET}")
    print(f"{BOLD}  SETUP SUMMARY{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 64}{RESET}\n")

    all_ok = True
    for name, passed in zip(names, results):
        icon = f"{GREEN}OK{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  [{icon}]  {name}")
        if not passed and name != "Tesseract OCR":
            all_ok = False

    print()

    if all_ok:
        print(f"  {GREEN}{BOLD}Setup complete! The project is ready to use offline.{RESET}")
        print()
        if IS_WINDOWS:
            print(f"    1. Activate venv :  {CYAN}venv\\Scripts\\activate{RESET}")
        else:
            print(f"    1. Activate venv :  {CYAN}source venv/bin/activate{RESET}")
        print(f"    2. Start API     :  {CYAN}python run_api.py{RESET}")
        print(f"       Docs at       :  {CYAN}http://localhost:8000/docs{RESET}")
        print(f"    3. CLI help      :  {CYAN}python -m mediaforge --help{RESET}")
        print(f"    4. Run tests     :  {CYAN}pytest tests/ -v{RESET}")
    else:
        print(f"  {RED}{BOLD}Setup finished with errors. Fix the issues above and re-run.{RESET}")

    print(f"\n{CYAN}{BOLD}{'=' * 64}{RESET}\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if IS_WINDOWS:
        os.system("")  # enable ANSI escape codes on Windows

    banner()

    info(f"Project  : {PROJECT_DIR}")
    info(f"Python   : {sys.executable} ({platform.python_version()})")
    info(f"Platform : {platform.system()} {platform.release()}")

    results = []

    # Phase 1 — Environment
    r = step_01_venv()
    results.append(r)
    if not r:
        fail("Cannot continue without virtual environment.")
        sys.exit(1)

    results.append(step_02_pip())

    r = step_03_pytorch()
    results.append(r)

    r = step_04_packages()
    results.append(r)
    if not r:
        fail("Package installation failed. Fix errors and re-run.")
        sys.exit(1)

    # Phase 2 — Project install
    results.append(step_05_install_project())

    # Phase 3 — Configuration
    results.append(step_06_env())
    results.append(step_07_dirs())
    results.append(step_08_ffmpeg())
    results.append(step_09_tesseract())

    # Phase 4 — AI Models (pre-download for offline use)
    results.append(step_10_whisper())
    results.append(step_11_yolo())
    results.append(step_12_rembg())
    results.append(step_13_easyocr())

    print_summary(results)


if __name__ == "__main__":
    main()
