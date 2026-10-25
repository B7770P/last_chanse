"""Safely update only source/license in existing annotations; dry run unless --apply."""
import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from .rights import prepare_jobs, output_name, metadata, read_audit, matches_saved, save_provenance


def sync_rights(data, folder, rights, apply=False):
    root = Path(data)
    path = root / 'meta.csv'
    original = path.read_bytes()
    reader = csv.DictReader(io.StringIO(original.decode('utf-8-sig')), delimiter=';')
    fields = reader.fieldnames
    rows = list(reader)
    before = [dict(r) for r in rows]
    candidates = defaultdict(list)
    for job in prepare_jobs(folder, rights, recursive=True):
        for name in {job['dest'], 'images/real/' + output_name(job['path'])}:
            candidates[name].append(job)
    audit = read_audit(root)
    updates, matches, skipped = {}, {}, []
    # Include negative frames, whose provenance otherwise has no CSV row.
    images = {r['image'] for r in rows} | set(audit)
    for image in sorted(images):
        options = candidates.get(image, [])
        if not options:
            skipped.append(image)
            continue
        entry = audit.get(image)
        if entry is None:
            raise ValueError(f'No annotation audit for {image}; cannot verify previous image transformations')
        accepted = [j for j in options if matches_saved(j['path'], root / image, entry.get('covered_faces', []))]
        if len(accepted) != 1:
            raise ValueError(f'Expected one verified source for {image}, found {len(accepted)}; nothing updated')
        job = accepted[0]
        matches[image] = metadata(job['record'])
        updates[image] = {'registry': str(Path(rights).absolute()), 'record': job['record'],
                          'matching_method': 'exact_reproduction_of_saved_image'}
    changed = 0
    for row in rows:
        if row['image'] in matches:
            source, license_name = matches[row['image']]
            changed += (row['source'], row['license']) != (source, license_name)
            row['source'], row['license'] = source, license_name
    for old, new in zip(before, rows):
        assert all(new[k] == v for k, v in old.items() if k not in ('source', 'license'))
    report = {'matched_images': len(matches), 'changed_rows': changed, 'total_rows': len(rows),
              'unmatched_images_unchanged': skipped, 'applied': False,
              'image_matches': {name: value['record']['file'] for name, value in updates.items()}}
    if apply:
        if path.read_bytes() != original:
            raise ValueError('meta.csv changed during verification; close the annotator and retry')
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        backup = root / ('rights_backup_' + stamp)
        backup.mkdir()
        (backup / 'meta.csv').write_bytes(original)
        provenance = root / 'rights_provenance.json'
        if provenance.exists():
            (backup / provenance.name).write_bytes(provenance.read_bytes())
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=fields, delimiter=';')
        writer.writeheader()
        writer.writerows(rows)
        temp = path.with_suffix('.csv.rights.tmp')
        with temp.open('w', encoding='utf-8', newline='') as handle:
            handle.write(buffer.getvalue())
        if path.read_bytes() != original:
            temp.unlink()
            raise ValueError('meta.csv changed before commit; retry after closing the annotator')
        temp.replace(path)
        save_provenance(root, updates)
        report.update(applied=True, backup=str(backup), previous_meta_sha256=hashlib.sha256(original).hexdigest())
        (root / 'rights_sync_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Image root matching registry relative filenames')
    parser.add_argument('--data', required=True)
    parser.add_argument('--rights', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    report = sync_rights(args.data, args.input, args.rights, args.apply)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
