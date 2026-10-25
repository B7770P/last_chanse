"""Итоговая оценка на fin_test: сравнивает CSV решения с ручной разметкой.

    powershell -ExecutionPolicy Bypass -File .\\run_final.ps1 -InputDir fin_test -Output fin_test_pred.csv -Benchmark fin_test_speed.json
    python eval_fin_test.py --labels fin_test_labels.csv --pred fin_test_pred.csv --out fin_test_report.json

Разметка: image;plate_num;plate_type;note — одна строка на номер (на одном фото может
быть несколько строк). Для type1/type1a/type1b номер считается найденным, если на том же
фото есть предсказание с тем же текстом и тем же типом. Для other правильно, если решение
не выдало на этом фото type1 (по заданию other можно не выводить или выводить как other).
Строки с пустым plate_num для type1/type1a/type1b в подсчёт не идут (нечитаемые).
Рамки (IoU) не проверяются: в CSV решения их нет. Строки с заметкой «исключить» пропускаются.
"""
import argparse
import csv
import json
from collections import defaultdict

CYR = str.maketrans('АВЕКМНОРСТУХ', 'ABEKMHOPCTYX')


def norm(text):
    return text.strip().upper().replace(' ', '').translate(CYR)


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=';'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labels', required=True)
    p.add_argument('--pred', required=True)
    p.add_argument('--out', default='fin_test_report.json')
    a = p.parse_args()

    # строки с заметкой «исключить» не учитываются
    labels = [r for r in read_csv(a.labels) if 'исключ' not in (r.get('note') or '').lower()]
    preds = defaultdict(list)
    for r in read_csv(a.pred):
        preds[r['image']].append((norm(r['plate_num']), r['plate_type'].strip()))

    stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    errors = []
    other_images = set()
    for r in labels:
        image, text, kind = r['image'], norm(r['plate_num']), r['plate_type'].strip()
        if kind == 'other':
            other_images.add(image)
            continue
        if not text:
            continue
        stats[kind]['total'] += 1
        if (text, kind) in preds.get(image, []):
            stats[kind]['correct'] += 1
        else:
            errors.append({'image': image, 'expected': f'{text} {kind}',
                           'predicted': [f'{t} {k}' for t, k in preds.get(image, [])]})

    # other: ошибка, если на фото, где размечен только other, решение выдало type1
    target_images = {r['image'] for r in labels if r['plate_type'].strip() != 'other'}
    only_other = other_images - target_images
    other_as_type1 = sorted(i for i in only_other if any(k == 'type1' for _, k in preds.get(i, [])))
    for image in other_as_type1:
        errors.append({'image': image, 'expected': 'other (не type1)',
                       'predicted': [f'{t} {k}' for t, k in preds.get(image, [])]})

    labelled = {r['image'] for r in labels}
    result = {
        'images': len(labelled),
        'by_type': {k: {**v, 'accuracy': round(v['correct'] / v['total'], 3) if v['total'] else None}
                    for k, v in sorted(stats.items())},
        'all_targets': {'total': sum(v['total'] for v in stats.values()),
                        'correct': sum(v['correct'] for v in stats.values())},
        'other_images': len(only_other),
        'other_output_as_type1': len(other_as_type1),
        'predictions_on_unlabelled_images': sorted(set(preds) - labelled),
        'errors': errors,
    }
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    for k, v in result['by_type'].items():
        print(f"{k}: {v['correct']} / {v['total']}")
    t = result['all_targets']
    print(f"всего type1/1a/1b: {t['correct']} / {t['total']}")
    print(f"other выдан как type1: {result['other_output_as_type1']} из {result['other_images']} фото")
    print(f'подробности: {a.out}')


if __name__ == '__main__':
    main()
