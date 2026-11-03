"""Run analysis and Word report with a persistent log and execution manifest.

python src/run_all.py --with-importer additionally refreshes the UCI cross-check.
The reviewed PowerPoint is separate and is not silently rebuilt by this runner.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--with-importer', action='store_true')
args = ap.parse_args()
packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'jupyter',
            'ucimlrepo', 'python-docx', 'python-pptx']
manifest = {'started_utc': datetime.now(timezone.utc).isoformat(),
            'python_executable': sys.executable, 'python': platform.python_version(),
            'in_virtual_environment': sys.prefix != sys.base_prefix,
            'packages': {p: importlib.metadata.version(p) for p in packages},
            'raw_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (root / 'data/raw').rglob('*') if p.is_file()},
            'stages': [], 'importer_refresh_requested': args.with_importer}
reports = root / 'reports'
reports.mkdir(exist_ok=True)
with (reports / 'run_log.txt').open('w', encoding='utf-8') as log:
    def write(s):
        print(s, end='', flush=True)
        log.write(s)
        log.flush()
    write(json.dumps(manifest, indent=2) + '\n')
    stages = [('01_audit_raw.py', *(['--with-importer'] if args.with_importer else [])),
              ('02_prepare_split.py',), ('03_compare_models.py',), ('build_report.py',)]
    for stage, *extra in stages:
        write('\n########## ' + stage + ' ##########\n')
        proc = subprocess.Popen([sys.executable, '-u', str(root / 'src' / stage), *extra],
                                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
        for line in proc.stdout:
            write(line)
        code = proc.wait()
        manifest['stages'].append({'script': stage, 'exit_code': code})
        if code:
            manifest['success'] = False
            break
    else:
        manifest['success'] = True
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (reports / 'execution_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    write('\nSUCCESS\n' if manifest['success'] else '\nFAILED: see stage exit code\n')
sys.exit(0 if manifest['success'] else 1)
