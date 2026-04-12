"""
Media Forge — API Server Launcher

Usage (from project root, with venv activated):
    python run_api.py

API docs will be available at:
    http://localhost:8000/docs
"""

import sys
import os


def check_environment():
    """Verify the runtime environment before starting."""
    in_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )

    if not in_venv:
        print("\n  [ERROR] Virtual environment is not activated!\n")
        if sys.platform == "win32":
            print("  Run this first:  venv\\Scripts\\activate")
        else:
            print("  Run this first:  source venv/bin/activate")
        print("  Then retry:      python run_api.py\n")
        sys.exit(1)

    try:
        import mediaforge  # noqa: F401
    except ImportError:
        print("\n  [ERROR] mediaforge package not found!\n")
        print("  Run this inside your activated venv:")
        print("    pip install -e .")
        print("  Then retry:  python run_api.py\n")
        sys.exit(1)


if __name__ == "__main__":
    check_environment()

    import uvicorn
    from mediaforge.api.app import create_app

    app = create_app()
    uvicorn.run("run_api:app", host="0.0.0.0", port=8000, reload=True)
else:
    from mediaforge.api.app import create_app
    app = create_app()
