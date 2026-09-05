#!/usr/bin/env python3
"""
scripts/run.py – CLI Launcher Script for TrustExtract-N
========================================================
"""

import sys
import os
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from run_app import main

if __name__ == "__main__":
    main()
