"""Copy a selected partition into a flat directory for inference."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--part', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        p.error('Output must be empty so stale images cannot enter evaluation')
    images = json.loads(Path(a.split).read_text(encoding='utf-8'))[a.part]
    names = [Path(image).name for image in images]
    if len(names) != len(set(names)):
        p.error('Duplicate basenames in partition; rename images before running flat-directory inference')
    out.mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copy2(Path(a.data) / image, out / Path(image).name)
    print(f'Copied {len(images)} images')


if __name__ == '__main__':
    main()
