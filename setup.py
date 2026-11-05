"""
setup.py — One-Click Environment Setup
=======================================
Run this script ONCE to install all dependencies for the CarRacing-v3 project.

Usage:
    python setup.py

This will:
  1. Upgrade pip
  2. Install all packages from requirements.txt
  3. Verify each critical import works
  4. Test that CarRacing-v3 environment loads correctly
  5. Register a Jupyter kernel so the notebook can find this Python
"""

import subprocess
import sys
import importlib


def run_cmd(cmd, label=""):
    """Run a shell command and return success status."""
    print(f"\n{'─' * 60}")
    print(f"▶  {label or ' '.join(cmd)}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd, text=True)
    return result.returncode == 0


def check_import(module_name, display_name=None):
    """Try importing a module and report success/failure."""
    label = display_name or module_name
    try:
        importlib.import_module(module_name)
        print(f"  ✅  {label}")
        return True
    except ImportError:
        print(f"  ❌  {label}  (not importable)")
        return False


def main():
    print("=" * 60)
    print("  CarRacing-v3 Project — Dependency Installer")
    print("=" * 60)
    print(f"  Python version : {sys.version}")
    print(f"  Executable     : {sys.executable}")
    print("=" * 60)

    # ── Step 1: Upgrade pip ──────────────────────────────────────
    run_cmd(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "Step 1/5: Upgrading pip"
    )

    # ── Step 2: Install from requirements.txt ────────────────────
    run_cmd(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        "Step 2/5: Installing all packages from requirements.txt"
    )

    # ── Step 3: Verify critical imports ──────────────────────────
    print(f"\n{'=' * 60}")
    print("  Step 3/5: Verifying critical imports")
    print(f"{'=' * 60}")

    checks = [
        ("numpy",            "NumPy"),
        ("pandas",           "Pandas"),
        ("matplotlib",       "Matplotlib"),
        ("cv2",              "OpenCV"),
        ("gymnasium",        "Gymnasium"),
        ("Box2D",            "Box2D (CarRacing physics)"),
        ("torch",            "PyTorch"),
        ("stable_baselines3","Stable-Baselines3"),
        ("imageio",          "ImageIO (video recording)"),
        ("PIL",              "Pillow (image processing)"),
        ("IPython",          "IPython / Jupyter"),
        ("reportlab",        "ReportLab (PDF generation)"),
    ]

    all_ok = all(check_import(mod, lbl) for mod, lbl in checks)

    # ── Step 4: Test CarRacing-v3 ────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Step 4/5: Testing CarRacing-v3 environment")
    print(f"{'=' * 60}")

    try:
        import gymnasium as gym
        env = gym.make(
            "CarRacing-v3",
            domain_randomize=False,
            continuous=False,
            render_mode="rgb_array"
        )
        obs, _ = env.reset()
        env.close()
        print(f"  ✅  CarRacing-v3 works!  Observation shape = {obs.shape}")
    except Exception as e:
        print(f"  ❌  CarRacing-v3 failed: {e}")
        all_ok = False

    # ── Step 5: Register Jupyter kernel ──────────────────────────
    print(f"\n{'=' * 60}")
    print("  Step 5/5: Registering Jupyter kernel")
    print(f"{'=' * 60}")

    run_cmd(
        [sys.executable, "-m", "ipykernel", "install",
         "--user", "--name", "carracing",
         "--display-name", "Python (CarRacing-v3)"],
        "Registering Jupyter kernel 'Python (CarRacing-v3)'"
    )

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if all_ok:
        print("  ✅  ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
        print()
        print("  Next steps:")
        print("    1. Open the notebook:")
        print("       jupyter notebook CarRacing_v3_Complete_Guide.ipynb")
        print()
        print("    2. In the notebook, select kernel: 'Python (CarRacing-v3)'")
        print()
        print("    3. Or run the trained agents directly:")
        print("       python src/main.py --agent PPO")
    else:
        print("  ⚠️  Some packages had issues. Check the errors above.")
        print("  You can try running this script again, or install manually:")
        print(f"    {sys.executable} -m pip install -r requirements.txt")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

