"""Простая разметка fin_test: показывает фото, ты вписываешь номер и тип, всё сохраняется в CSV.

    python label_fin_test.py
    python label_fin_test.py --images fin_test --labels fin_test_labels.csv

Управление:
  Enter                — сохранить и перейти к следующему номеру
  щелчок по фото       — увеличенный фрагмент справа (для мелких номеров)
  кнопка «+ ещё номер» — добавить ещё один номер на этом же фото
  кнопка «Удалить»     — убрать текущую строку (например, лишний номер)
Файл сохраняется после каждого перехода. Номера — латиницей, жёлтые в формате AA12377.
"""
import argparse
import csv
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

TYPES = ['type1', 'type1a', 'type1b', 'other']
FIELDS = ['image', 'plate_num', 'plate_type', 'note']
CYR = str.maketrans('АВЕКМНОРСТУХ', 'ABEKMHOPCTYX')


class Labeler:
    def __init__(self, images, labels):
        self.images, self.labels = Path(images), Path(labels)
        with open(self.labels, encoding='utf-8-sig', newline='') as f:
            self.rows = [{k: r.get(k, '') or '' for k in FIELDS} for r in csv.DictReader(f, delimiter=';')]
        if not self.rows:
            raise SystemExit(f'В {self.labels} нет строк')
        # начинаем с первой строки без номера
        self.i = next((n for n, r in enumerate(self.rows) if not r['plate_num'] and r['plate_type'] != 'other'), 0)

        self.win = tk.Tk()
        self.win.title('Разметка fin_test')
        top = tk.Frame(self.win)
        top.pack(fill=tk.X)
        self.info = tk.Label(top, text='', anchor='w', font=('Segoe UI', 11))
        self.info.pack(side=tk.LEFT, padx=6)
        body = tk.Frame(self.win)
        body.pack()
        self.canvas = tk.Canvas(body, width=1000, height=700, bg='black')
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind('<Button-1>', self.zoom)
        self.zoom_canvas = tk.Canvas(body, width=420, height=300, bg='gray20')
        self.zoom_canvas.pack(side=tk.LEFT, anchor='n', padx=6)

        form = tk.Frame(self.win)
        form.pack(pady=6)
        tk.Label(form, text='Номер:').pack(side=tk.LEFT)
        self.num = tk.Entry(form, width=14, font=('Consolas', 16))
        self.num.pack(side=tk.LEFT, padx=4)
        tk.Label(form, text='Тип:').pack(side=tk.LEFT)
        self.kind = ttk.Combobox(form, values=TYPES, width=8, state='readonly', font=('Segoe UI', 12))
        self.kind.pack(side=tk.LEFT, padx=4)
        tk.Label(form, text='Заметка:').pack(side=tk.LEFT)
        self.note = tk.Entry(form, width=40)
        self.note.pack(side=tk.LEFT, padx=4)

        buttons = tk.Frame(self.win)
        buttons.pack(pady=4)
        for title, action in [('← Назад', self.prev), ('Сохранить и дальше (Enter) →', self.next),
                              ('+ ещё номер на этом фото', self.add), ('Удалить строку', self.delete)]:
            tk.Button(buttons, text=title, command=action).pack(side=tk.LEFT, padx=4)
        self.win.bind('<Return>', lambda e: self.next())
        self.win.protocol('WM_DELETE_WINDOW', self.close)
        self.show()

    def show(self):
        row = self.rows[self.i]
        path = self.images / row['image']
        self.original = Image.open(path).convert('RGB')
        w, h = self.original.size
        self.scale = min(1000 / w, 700 / h)
        shown = self.original.resize((max(1, int(w * self.scale)), max(1, int(h * self.scale))))
        self.photo = ImageTk.PhotoImage(shown)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        self.zoom_canvas.delete('all')
        same = sum(1 for r in self.rows if r['image'] == row['image'])
        left = sum(1 for r in self.rows if not r['plate_num'] and r['plate_type'] != 'other')
        self.info.config(text=f'{self.i + 1}/{len(self.rows)}  {row["image"]}  '
                              f'(номеров на фото: {same})   без номера осталось: {left}')
        for entry, value in [(self.num, row['plate_num']), (self.note, row['note'])]:
            entry.delete(0, tk.END)
            entry.insert(0, value)
        self.kind.set(row['plate_type'] if row['plate_type'] in TYPES else '')
        self.num.focus_set()

    def zoom(self, event):
        x, y = event.x / self.scale, event.y / self.scale
        crop = self.original.crop((int(x - 70), int(y - 50), int(x + 70), int(y + 50))).resize((420, 300))
        self.zoom_photo = ImageTk.PhotoImage(crop)
        self.zoom_canvas.delete('all')
        self.zoom_canvas.create_image(0, 0, image=self.zoom_photo, anchor='nw')

    def store(self):
        row = self.rows[self.i]
        row['plate_num'] = self.num.get().strip().upper().replace(' ', '').translate(CYR)
        row['plate_type'] = self.kind.get()
        row['note'] = self.note.get().strip()
        if row['plate_num']:  # подсказка больше не нужна
            row['note'] = row['note'].replace('; впиши номер', '').replace('впиши номер', '').strip()
        if row['plate_type'] not in TYPES:
            messagebox.showerror('Тип', 'Выбери тип: type1, type1a, type1b или other')
            return False
        self.save()
        return True

    def save(self):
        tmp = self.labels.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, delimiter=';')
            w.writeheader()
            w.writerows(self.rows)
        tmp.replace(self.labels)

    def next(self):
        if not self.store():
            return
        if self.i + 1 < len(self.rows):
            self.i += 1
            self.show()
        else:
            messagebox.showinfo('Готово', f'Это последняя строка. Всё сохранено в {self.labels}')

    def prev(self):
        if self.store() and self.i > 0:
            self.i -= 1
            self.show()

    def add(self):
        if not self.store():
            return
        image = self.rows[self.i]['image']
        self.rows.insert(self.i + 1, {'image': image, 'plate_num': '', 'plate_type': '', 'note': ''})
        self.i += 1
        self.save()
        self.show()

    def delete(self):
        image = self.rows[self.i]['image']
        if sum(1 for r in self.rows if r['image'] == image) == 1:
            messagebox.showerror('Удаление', 'Это единственная строка для фото. Чтобы исключить фото, '
                                             'напиши в заметке «исключить».')
            return
        del self.rows[self.i]
        self.i = min(self.i, len(self.rows) - 1)
        self.save()
        self.show()

    def close(self):
        row = self.rows[self.i]
        if self.kind.get() in TYPES:
            row.update(plate_num=self.num.get().strip().upper().replace(' ', '').translate(CYR),
                       plate_type=self.kind.get(), note=self.note.get().strip())
            self.save()
        self.win.destroy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--images', default='fin_test')
    p.add_argument('--labels', default='fin_test_labels.csv')
    a = p.parse_args()
    Labeler(a.images, a.labels).win.mainloop()


if __name__ == '__main__':
    main()
