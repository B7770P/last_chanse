"""Local set metrics, not official scoring. Handles duplicate plates with one-to-one matching."""
import argparse
import csv
import json
from pathlib import Path
from .common import TYPES, valid_number


def matches(truth, prediction):
    return (truth['plate_type'] == prediction['plate_type'] and
            len(truth['plate_num']) == len(prediction['plate_num']) and
            all(a == '#' or a == b for a, b in zip(truth['plate_num'], prediction['plate_num'])))


def score(truth, predicted):
    # Kuhn bipartite matching: prediction order cannot alter TP count.
    result = {}
    for kind in TYPES[:3]:
        gt = [x for x in truth if x['plate_type'] == kind and x.get('is_vehicle', '1') == '1']
        pr = [x for x in predicted if x['plate_type'] == kind]
        adjacency = [[j for j, b in enumerate(pr) if a['image'] == b['image'] and matches(a, b)] for a in gt]
        owners = {}

        def augment(i, seen):
            for j in adjacency[i]:
                if j in seen:
                    continue
                seen.add(j)
                if j not in owners or augment(owners[j], seen):
                    owners[j] = i
                    return True
            return False

        tp = sum(augment(i, set()) for i in range(len(gt)))
        fp, fn = len(pr) - tp, len(gt) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        result[kind] = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
                        'f1': 2 * precision * recall / max(precision + recall, 1e-12)}
    return {'metric': 'local one-to-one exact plate+type match; # only in truth is wildcard; no bbox matching',
            'per_type': result, 'macro_f1': sum(x['f1'] for x in result.values()) / 3,
            'other_predictions': sum(x['plate_type'] == 'other' for x in predicted)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--truth', required=True, help='Organizer CSV or meta.csv')
    p.add_argument('--pred', required=True)
    p.add_argument('--split', help='If truth is meta.csv, filter to this split JSON')
    p.add_argument('--part', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--out', default='evaluation.json')
    a = p.parse_args()
    def read(path):
        with open(path, encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f, delimiter=';'))
    truth, predicted = read(a.truth), read(a.pred)
    if a.split:
        chosen = set(json.loads(Path(a.split).read_text())[a.part])
        truth = [r for r in truth if r['image'] in chosen]
    for row in truth:
        row['image'] = Path(row['image']).name
    for row in predicted:
        if row['plate_type'] not in TYPES or not 0 <= float(row['confidence']) <= 1:
            raise ValueError(f'Bad prediction: {row}')
        if not valid_number(row['plate_num'], row['plate_type']):
            raise ValueError(f'Bad output number: {row}')
    report = score(truth, predicted)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
