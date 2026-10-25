"""Per-image rights registry support; never infer a license from a folder name."""
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def output_name(path):
    path = Path(path)
    return path.name + '.jpg' if path.suffix.lower() == '.webp' else path.name


def read_registry(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    records = payload.get('files') if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError('Rights registry must be a list or an object with files')
    result = {}
    for record in records:
        name = record.get('file', '')
        key = PurePosixPath(name)
        if not name or key.is_absolute() or '..' in key.parts or '\\' in name or ':' in name:
            raise ValueError(f'Unsafe registry filename: {name!r}')
        if name in result:
            raise ValueError(f'Duplicate registry filename: {name}')
        result[name] = record
    return result


def metadata(record):
    license_name = record.get('license', '').strip()
    if not license_name or license_name.upper() in {'UNKNOWN', 'NOASSERTION', 'TODO', 'UNSPECIFIED'}:
        raise ValueError(f'License not assigned: {record["file"]}')
    source = record.get('source_url') or record.get('original_publication_url')
    if not source and record.get('rights_status', '').startswith('own_photo_'):
        author = record.get('author')
        if author:
            source = 'own_photo:' + author
    if not source:
        raise ValueError(f'Source not assigned: {record["file"]}')
    return source, license_name


def verify_file(path, record):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if not record.get('sha256') or digest != record['sha256']:
        raise ValueError(f'File differs from rights registry: {record["file"]}')


def prepare_jobs(folder, registry_path=None, recursive=False):
    folder = Path(folder)
    records = read_registry(registry_path) if registry_path else None
    paths = folder.rglob('*') if recursive else folder.iterdir()
    jobs = []
    for path in sorted(paths):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(folder).as_posix()
        record = records.get(relative) if records is not None else None
        if records is not None:
            if record is None:
                raise ValueError(f'No rights record for {relative}; --input must be the registry image root')
            if record.get('excluded'):
                continue
            verify_file(path, record)
            metadata(record)
        jobs.append({'path': path, 'relative': relative, 'record': record})
    # TXT labels share a stem, so foo.jpg and foo.png also need distinct names.
    counts = Counter(Path(output_name(j['path'])).stem.casefold() for j in jobs)
    destinations = set()
    for job in jobs:
        name = output_name(job['path'])
        if counts[Path(name).stem.casefold()] > 1:
            token = hashlib.sha256(job['relative'].encode()).hexdigest()[:12]
            name = f'{Path(name).stem[:100]}__{token}{Path(name).suffix}'
        job['dest'] = 'images/real/' + name
        if job['dest'].casefold() in destinations:
            raise ValueError(f'Duplicate destination: {name}')
        destinations.add(job['dest'].casefold())
    if not jobs:
        raise ValueError(f'No supported, non-excluded images in {folder}')
    return jobs


def read_annotation_image(path):
    from .common import read_image
    try:
        return read_image(path)
    except ValueError:
        # Some downloaded .jpg files actually contain AVIF. Keep originals intact.
        import numpy as np
        from PIL import Image
        with Image.open(path) as image:
            return np.array(image.convert('RGB'))


def matches_saved(original, saved, covered_faces=()):
    """Reproduce the annotator output, including face covers, before trusting a filename."""
    import cv2
    image = read_annotation_image(original)
    for x1, y1, x2, y2 in covered_faces:
        image[y1:y2 + 1, x1:x2 + 1] = 110
    saved = Path(saved)
    params = [cv2.IMWRITE_JPEG_QUALITY, 88] if saved.suffix.lower() in ('.jpg', '.jpeg') else []
    ok, encoded = cv2.imencode(saved.suffix, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params)
    return bool(ok) and encoded.tobytes() == saved.read_bytes()


def read_audit(root):
    path = Path(root) / 'annotation_audit.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []
    return {r['image']: r for r in rows}


def save_provenance(root, updates):
    path = Path(root) / 'rights_provenance.json'
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    data.update(updates)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)
