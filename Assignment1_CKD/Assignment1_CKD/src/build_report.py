"""Rebuild the APA figures and student report from saved verified results."""
import subprocess
import sys
from pathlib import Path
from report_apa import build
if __name__ == '__main__':
    subprocess.run([sys.executable, str(Path(__file__).with_name('build_apa_figures.py'))], check=True)
    build()
