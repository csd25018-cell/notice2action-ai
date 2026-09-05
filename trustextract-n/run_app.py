#!/usr/bin/env python3
"""
run_app.py – Convenient One-Command Application Launcher for TrustExtract-N
=============================================================================
Usage:
    python run_app.py [--port 8501] [--host 0.0.0.0]
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def main():
    port = os.getenv("APP_PORT", "8501")
    host = os.getenv("APP_HOST", "0.0.0.0")
    
    app_path = root_dir / "app" / "app.py"
    
    print("=" * 60)
    print("      LAUNCHING TRUSTEXTRACT-N INFORMATION CARD APP      ")
    print("=" * 60)
    print(f"  App Path    : {app_path}")
    print(f"  Local Access: http://localhost:{port}")
    print("=" * 60 + "\n")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root_dir) + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", port,
        "--server.address", host
    ]
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")

if __name__ == "__main__":
    main()
