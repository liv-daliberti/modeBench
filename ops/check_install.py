"""Install a distribution into a clean venv and exercise it outside the checkout."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--profile', choices=('core', 'data'), default='core')
    parser.add_argument('--minimum', action='store_true')
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    if not artifact.is_file():
        parser.error('artifact must be an existing wheel or source distribution')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env['PIP_NO_CACHE_DIR'] = '1'
    with tempfile.TemporaryDirectory(prefix='modebench-installed-') as temporary:
        work = Path(temporary)
        shutil.copyfile(ROOT/'ops/test_installed.py', work/'smoke.py')
        shutil.copyfile(ROOT/'tests/fixtures/conformance-v1.json', work/'conformance.json')
        subprocess.run([sys.executable, '-m', 'venv', str(work/'venv')], check=True, env=env)
        python = str(work/'venv/bin/python')
        package = str(artifact) + ('[data]' if args.profile == 'data' else '')
        install = [python, '-m', 'pip', 'install', '--no-cache-dir', package]
        if args.minimum:
            install += ['-c', str(ROOT/'provenance/constraints-minimum.txt')]
        subprocess.run(install, cwd=work, check=True, env=env)
        subprocess.run([python, '-m', 'pip', 'check'], cwd=work, check=True, env=env)
        command = [python, '-I', str(work/'smoke.py'), '--fixtures', str(work/'conformance.json')]
        if args.profile == 'data':
            shutil.copytree(ROOT/'data', work/'frozen-data')
            command += ['--data-source', str(work/'frozen-data')]
        subprocess.run(command, cwd=work, check=True, env=env)


if __name__ == '__main__':
    main()
