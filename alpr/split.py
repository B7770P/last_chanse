"""Group by plate identity, image hash and optional acquisition session."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from .common import read_meta


def split_dataset(root, out, val=.15, test=.15, seed=42, sessions=None):
    root = Path(root)
    images = sorted(p.relative_to(root).as_posix() for p in (root / 'images').rglob('*')
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    parent = {x: x for x in images}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(b)] = find(a)

    owner = {}
    def join(key, image):
        if key in owner:
            union(owner[key], image)
        else:
            owner[key] = image

    for row in read_meta(root):
        if row['image'] not in parent:
            raise ValueError(f'Missing image: {row["image"]}')
        # Non-target motifs repeat often; do not group all foreign plates by text.
        if row['plate_type'] != 'other' and '#' not in row['plate_num']:
            join(('number', row['plate_num']), row['image'])
    session_ids = json.loads(Path(sessions).read_text(encoding='utf-8')) if sessions else {}
    for image in images:
        join(('hash', hashlib.sha256((root / image).read_bytes()).hexdigest()), image)
        if image in session_ids:
            join(('session', session_ids[image]), image)
    groups = {}
    for image in images:
        groups.setdefault(find(image), []).append(image)
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 3:
        raise ValueError('Need at least three independent groups for train/val/test')
    result = {'train': [], 'val': [], 'test': []}
    targets = {'train': len(images) * (1 - val - test), 'val': len(images) * val, 'test': len(images) * test}
    # Preserve groups while keeping reasonable proportions; large sessions may skew sizes.
    for group in shuffled:
        part = max(result, key=lambda k: (targets[k] - len(result[k])) / max(targets[k], 1))
        result[part].extend(group)
    if any(not result[x] for x in result):
        raise ValueError('Grouping produced an empty split; use more independent examples')
    result.update({'seed': seed, 'groups': len(groups), 'note': 'No target number/exact image duplicate crosses partitions. Sessions grouped if supplied.'})
    for part in ('train', 'val', 'test'):
        result[part].sort()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val', type=float, default=.15)
    p.add_argument('--test', type=float, default=.15)
    p.add_argument('--sessions', help='JSON mapping image relative paths to capture session IDs')
    a = p.parse_args()
    if not (0 < a.val < 1 and 0 < a.test < 1 and a.val + a.test < 1):
        p.error('val/test must be positive and sum to less than 1')
    result = split_dataset(a.data, a.out, a.val, a.test, a.seed, a.sessions)
    print({x: len(result[x]) for x in ('train', 'val', 'test')})


if __name__ == '__main__':
    main()
