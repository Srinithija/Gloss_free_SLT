"""Snapshot and verify the existing sltbaselines project, which this experiment must never change.

    python tools/protect.py snapshot     # once, before any work (refuses to overwrite an existing snapshot)
    python tools/protect.py verify       # before and after every long run

Records: SHA-256 of every checkpoint/weight file, git HEAD and `git status --porcelain`, and the size and
modification time of every other file (excluding .git and __pycache__). Only reads the old project.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

OLD = r'C:\Users\SRINITHIJA\Desktop\Sign_processing\sltbaselines'
SNAPSHOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'protection', 'sltbaselines_snapshot.json')
WEIGHT_EXT = ('.pt', '.pth', '.bin', '.safetensors', '.ckpt')


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.run(['git', '-C', OLD, *args], capture_output=True, text=True, check=True).stdout


def take():
    weights, files = {}, {}
    for root, dirs, names in os.walk(OLD):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__')]
        for n in names:
            p = os.path.join(root, n)
            rel = os.path.relpath(p, OLD)
            st = os.stat(p)
            files[rel] = [st.st_size, int(st.st_mtime)]
            if n.lower().endswith(WEIGHT_EXT):
                weights[rel] = sha256(p)
    status = git('status', '--porcelain')
    return {'project': OLD, 'git_head': git('rev-parse', 'HEAD').strip(), 'git_status': status,
            'weights_sha256': weights, 'files': files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['snapshot', 'verify'])
    args = ap.parse_args()
    if args.action == 'snapshot':
        if os.path.exists(SNAPSHOT):
            raise SystemExit(f'{SNAPSHOT} already exists; it is the reference and is never overwritten')
        os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
        snap = take()
        with open(SNAPSHOT, 'w', encoding='utf-8') as f:
            json.dump(snap, f, indent=1)
        print(f"snapshot: HEAD {snap['git_head'][:10]}, {len(snap['git_status'].splitlines())} git status lines, "
              f"{len(snap['weights_sha256'])} weight files hashed, {len(snap['files'])} files recorded -> {SNAPSHOT}")
        return
    ref = json.load(open(SNAPSHOT, encoding='utf-8'))
    now = take()
    problems = []
    if now['git_head'] != ref['git_head']:
        problems.append(f"git HEAD changed: {ref['git_head'][:10]} -> {now['git_head'][:10]}")
    if now['git_status'] != ref['git_status']:
        problems.append('git status --porcelain changed')
    for rel, digest in ref['weights_sha256'].items():
        if now['weights_sha256'].get(rel) != digest:
            problems.append(f'weight file changed or missing: {rel}')
    for rel, meta in ref['files'].items():
        if now['files'].get(rel) != meta:
            problems.append(f'file changed or missing: {rel}')
    for rel in sorted(set(now['files']) - set(ref['files'])):
        problems.append(f'new file in the old project: {rel}')
    if problems:
        print('OLD PROJECT CHANGED:\n  ' + '\n  '.join(problems[:40]))
        sys.exit(1)
    print(f"old project unchanged: HEAD {now['git_head'][:10]}, git status identical, "
          f"{len(ref['weights_sha256'])} weight files and {len(ref['files'])} files match the snapshot")


if __name__ == '__main__':
    main()
