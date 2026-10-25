# Распознавание нестандартных ГРЗ (полуфинал «ИИ и анализ данных»)

Офлайн-решение: на входе папка с фотографиями, на выходе CSV с номерами и типами.
Типы: `type1` — обычный однострочный, `type1a` — квадратный двухстрочный, `type1b` — жёлтый (формат `AA12377`), `other` — прочие (по умолчанию не выводятся).

Цепочка: два детектора YOLO11n → выравнивание номера по контуру → CRNN OCR (общий + отдельный для type1).

> Прежний README базового проекта (до реальных данных) сохранён в `backups/readme_requirements_20261025/README.md`.

## Запуск

Одна команда из папки решения (окружение — раздел «Установка» ниже):

```powershell
powershell -ExecutionPolicy Bypass -File .\run_final.ps1 -InputDir "C:\путь\к\фото" -Output predictions.csv
# с замером времени:
powershell -ExecutionPolicy Bypass -File .\run_final.ps1 -InputDir "C:\путь\к\фото" -Output predictions.csv -Benchmark speed.json
```

`-ExecutionPolicy Bypass` нужен, если Windows запрещает запуск скриптов `.ps1`; настройки системы он не меняет.

Вход — JPG/JPEG/PNG прямо в папке (без подпапок), кириллица в путях допускается. Выход — UTF-8 CSV с `;`:

```csv
image;plate_num;plate_type;confidence
```

Несколько номеров на кадре — несколько строк; кадр без номеров — без строк. `#` — символ, в котором модель не уверена. Confidence — эвристика ранжирования, не откалиброванная вероятность. Интернет и внешние API не используются (`YOLO_OFFLINE=true`).

Документы: [пояснительная записка](EXPLANATORY_NOTE.md) (PDF: `EXPLANATORY_NOTE.pdf`), [технические подробности](TECHNICAL_DETAILS.md), [отчёт по критериям](REPORT.md), отчёт валидатора датасета — `dataset_validation_report.json`.

## Скачать датасет

Датасет в репозиторий не входит из-за размера. Он лежит на [Google Drive](https://drive.google.com/drive/folders/1YMq7VnpGj-LJNcLBbss0RtCgfDtchrLs?usp=sharing) архивом `dataset_v8.zip` (≈ 1,24 ГБ): 606 реальных и 5000 синтетических изображений, `labels/`, `meta.csv`, `generator/`, `README.md`, `LICENSE`, `validation_report.json`.

Распакуйте архив в папку решения: внутри архива уже лежит папка `dataset/`, она должна оказаться рядом с `run_final.ps1`. Проверка датасета валидатором:

```powershell
python -m alpr.validate --data dataset --report dataset_validation_report.json
```

Скрипты обучения принимают путь к датасету параметром `--data` (например, `--data dataset --split splits/real606.json`). Для запуска распознавания датасет не нужен.

## Установка (Windows, NVIDIA)

Python 3.9 (проверено на 3.9.13), 64-bit. Из папки проекта:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Версии совпадают с теми, на которых обучены модели (`models/*/MODEL_PROVENANCE.json`). Полный снимок окружения — `requirements-lock.txt` в собранной сдаче. Без NVIDIA: ставь torch с `--index-url https://download.pytorch.org/whl/cpu` и запускай с `-Device cpu`.

Офлайн-установка на ПК жюри (wheelhouse готовится на такой же ОС и версии Python):

```powershell
python -m pip download -r requirements.txt -d wheelhouse
python -m pip download torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118 -d wheelhouse
# на ПК жюри:
python -m pip install --no-index --find-links wheelhouse torch==2.7.1 torchvision==0.22.1 -r requirements.txt
```

## Итоговая конфигурация

| Компонент | Веса |
|---|---|
| Основной детектор (все 4 типа) | `models/yolo11n_real_20261024/weights/best.pt` |
| Детектор белых номеров (type1/type1a) | `models/yolo11n_white_20261024/weights/best.pt` |
| OCR type1a/type1b | `models/ocr_synth2_20261025/best.pt` (два раунда дообучения из `models/ocr_crops_20261024` на реальных обрезках и процедурной синтетике) |
| OCR type1 (мелкие номера) | `models/ocr_type1_small_20261025/best.pt` |

Параметры: порог детекции 0.35, порог символа 0.40 (для type1 — 0.5; выбран на val, `sweep_thresholds.py`), второй проход type1 на 960 px (`--multiscale-type1`) с ранним выходом и только если на 640 px замечен type1 (`--gate-second-pass`), дополнительный проход основного детектора на 960 px на кадрах с длинной стороной ≥ 1600 px или без найденных рамок (`--base-extra-size 960 --base-extra-gate large_or_empty`; отдельная копия весов со своим CUDA graph), удаление одного лишнего символа, если строка длиннее маски (`--length-repair`), `--threads 1`. Ускорение без смены моделей: прямой вызов YOLO (`--direct-yolo`), квадратный вход и CUDA graphs (`--cuda-graphs`), декодирование JPEG с уменьшением до длинной стороны ≥ 2048 px (`--decode-max-side 2048`), чтение следующего фото в фоне (`--prefetch`). Полная команда — в `run_final.ps1`. Если на видеокарте возникнет ошибка CUDA, запускай с `-NoGraphs`. Остальные папки `models/` — промежуточные эксперименты, в сдачу не входят.

## Сборка сдачи

```powershell
python make_submission.py --with-dataset --zip
```

Создаёт новую папку `submission_<дата>/` (и zip): код `alpr/`, тесты, 4 итоговые модели с лицензиями и происхождением, `requirements-lock.txt`, `SHA256SUMS.txt`, документы; с `--with-dataset` — датасет как `dataset/` и сплиты. Проект при этом не меняется.

## Датасет

Итоговый датасет — папка `dataset/` из архива на [Google Drive](https://drive.google.com/drive/folders/1YMq7VnpGj-LJNcLBbss0RtCgfDtchrLs?usp=sharing) (в проекте — `dataset_composite5000/`), структура по заданию: `images/real`, `images/synthetic`, `labels/real`, `labels/synthetic`, `meta.csv`, `generator/`, `README.md`, `LICENSE`. Локальный валидатор: ошибок нет (`validation_report.json`).

| | type1 | type1a | type1b | other |
|---|---:|---:|---:|---:|
| Реальные номера | 100 | 151 | 334 | 56 |
| Реальные изображения | 100 | 151 | 302 | 53 |
| Уникальные номера (реальные) | 98 | 87 | 275 | 53 |
| Рекомендация задания (изобр. / уник.) | — | 150 / 50 | 300 / 100 | 50 |
| Синтетические номера | 900 | 1900 | 2082 | 323 |

606 реальных изображений, 641 номер; разбивка 424 train / 91 val / 91 test (`splits/real606.json`). 5000 синтетических — производные от **обучающих** фотографий (новый текст номера с наследованием перспективы и освещения), val/test не используются. Источник и лицензия — для каждого изображения в `meta.csv` и `rights_provenance.json`.

Формат `.txt`: `class cx cy w h x1 y1 x2 y2 x3 y3 x4 y4 is_vehicle`, координаты нормированы; классы 0=type1, 1=type1a, 2=type1b, 3=other. Это расширенный формат, совместимость с официальным валидатором нужно проверить.

## Качество (что измерено и как)

Все цифры — на RTX 3060, на **validation, которая использовалась для выбора моделей и порогов**. Это не независимая итоговая оценка. Критерий: вся строка + тип + IoU ≥ 0.5.

| Проверка | Результат |
|---|---|
| Детектор, val 91 фото: рамка и тип верны | 81 / 99 |
| Белый детектор, расширенная val: type1 | 57 / 63 |
| Полная цепочка, type1 (44 номера) | 39 / 44 |
| Полная цепочка, type1a (27) | 10 / 27 |
| Полная цепочка, type1b (63) | 25 / 63 |
| Итоговая версия (v8), только строка + тип, без IoU: type1a / type1b | 19 / 27, 47 / 63 (`detection_variants_V6_2048.json`) |

Подробности по каждому этапу: `YOLO_DETECTOR.md`, `DETECTOR_SPECIALIST.md`, `OCR_IMPROVEMENT.md`, `OCR_TYPE1.md`, `TYPE1_FAST.md`.

## Скорость

Итоговый режим на RTX 3060 (73 фото итоговой проверки, модели прогреты, с чтением и записью CSV): среднее ≈ 22,5 мс, p95 ≈ 49 мс. Видеопамяти до 750 МБ с CUDA graphs и около 300–350 МБ с `-NoGraphs` (пик по nvidia-smi сверх фоновой загрузки, 91 фото val). **На GTX 1050 Ti / i5-7600 не измерено, а оценено** по профайлеру CUDA (`estimate_speed.py`): среднее ≈ 55–85 мс. Подробности и метод — в `TECHNICAL_DETAILS.md`, раздел 6. Разбивка времени по этапам: `python profile_stages.py --data dataset_composite5000 --split splits/real606.json --part val`.

## Известные ограничения

- Квадратные и жёлтые номера читаются плохо (≈ 55 % строк целиком на новых фото), хотя детектор их находит — основная проблема в обрезке и OCR.
- У YOLO нет признака «номер на автомобиле», поэтому запуск идёт с `--keep-nonvehicle`: номер вне ТС (на табличке, рекламе) тоже может попасть в вывод. Негативных примеров вне ТС в данных почти нет.
- Итоговая проверка на 72 новых фото (`eval_fin_test.py`, `fin_test_report_v8.json`): прочитано целиком 33 номера из 57 (type1 9/14, type1a 11/21, type1b 13/22; предыдущие версии: v7 — 32/57, v6 — 29/57); ни один `other` не выдан как type1.
- Вторая независимая проверка на 32 новых фото квадратных номеров (`test22_report_v8_sub.json`, 23 разные машины): 18/32 (56 %). Вместе с итоговой проверкой 51/89 (57 %), квадратные 29/53 (55 %); ошибки в основном в цифрах региона, ночью, в дыму и при сильном наклоне. Подробности и список отклонённых вариантов — в `TECHNICAL_DETAILS.md`, раздел 5.
- Шрифт и геометрия синтетики приблизительные, не сертифицированный ГОСТ.

## Проверки кода

```powershell
python -m unittest discover -s tests -v
python -m alpr.validate --data dataset_composite5000 --report reports/dataset_validation.json
```

## Лицензии

- Код проекта — MIT (`LICENSE`).
- Ultralytics YOLO11 и обученные на нём веса — AGPL-3.0 (`models/*/ULTRALYTICS_LICENSE.txt`). При передаче решения выполняются условия AGPL: исходный код прилагается.
- Фотографии — лицензия источника для каждого файла (CC BY 4.0, CC BY 2.0, CC0, CC BY-SA 2.0/3.0/4.0, public domain и др.); синтетика наследует лицензию родительской фотографии. Для CC BY-SA производные распространяются на тех же условиях.
- Шрифт DejaVu — `assets/FONT-LICENSE.txt`.
