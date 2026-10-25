"""Manual annotation with Tkinter. Original files are never modified; WebP input is saved as JPEG."""
import argparse
import json
from pathlib import Path
try:
    import tkinter as tk
    from tkinter import simpledialog, messagebox
    from PIL import ImageTk
except ImportError:
    tk = simpledialog = messagebox = None
    ImageTk = None
import numpy as np
from PIL import Image
from .common import read_image, write_image, read_meta, write_meta, write_labels, TYPES, valid_number, number_format_hint
from .rights import prepare_jobs, metadata, matches_saved, read_audit, save_provenance, verify_file, read_annotation_image


class Annotator:
    def __init__(self, source, root, attribution=None, license_name=None, rights=None, recursive=False):
        if tk is None:
            raise RuntimeError('Tkinter/Tcl is unavailable. Install standard Python with Tcl/Tk (Windows), or python3-tk (Linux).')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.source, self.license = attribution, license_name
        self.rights_path = str(Path(rights).absolute()) if rights else None
        self.jobs = prepare_jobs(source, rights, recursive)
        self.files = [j['path'] for j in self.jobs]
        self.audit = read_audit(root)
        self.rows = read_meta(root) if (self.root / 'meta.csv').exists() else []
        self.index, self.points, self.boxes, self.current = 0, [], [], []
        self.window = tk.Tk()
        self.window.title('Manual plate annotation')
        self.info = tk.Label(self.window, text='')
        self.info.pack()
        self.canvas = tk.Canvas(self.window, width=1100, height=650)
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.click)
        self.canvas.bind('<Button-3>', self.undo)
        controls = tk.Frame(self.window)
        controls.pack()
        for title, action in [('Undo corner', self.undo), ('Blur face (2 clicks)', self.blur_mode),
                              ('Undo last plate', self.undo_plate), ('Save and next', self.save),
                              ('Skip image', self.skip)]:
            tk.Button(controls, text=title, command=action).pack(side=tk.LEFT)
        self.mode = 'plate'
        self.load()

    def load(self):
        while self.index < len(self.jobs):
            job = self.jobs[self.index]
            saved = self.root / job['dest']
            if not saved.exists():
                break
            if self.rights_path:
                audit = self.audit.get(job['dest'])
                if audit is None or not matches_saved(job['path'], saved, audit.get('covered_faces', [])):
                    raise ValueError(f'Existing output cannot be verified: {saved}. Use another output name/dataset.')
                print(f'Already saved, skipped: {job["dest"]}', flush=True)
            else:
                messagebox.showerror('Duplicate name', f'{saved} exists. Rename the source image or use another dataset.')
            self.index += 1
        if self.index == len(self.files):
            messagebox.showinfo('Complete', 'Annotation complete. Run the local validator.')
            self.window.destroy()
            return
        self.job = self.jobs[self.index]
        self.path, self.dest = self.job['path'], self.job['dest']
        if self.job['record']:
            self.source, self.license = metadata(self.job['record'])
        self.im = read_annotation_image(self.path)
        h, w = self.im.shape[:2]
        self.scale = min(1100 / w, 650 / h, 1)
        self.points, self.current, self.boxes, self.mode = [], [], [], 'plate'
        self.display()

    def display(self):
        h, w = self.im.shape[:2]
        picture = Image.fromarray(self.im).resize((round(w * self.scale), round(h * self.scale)))
        self.photo = ImageTk.PhotoImage(picture)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        for row in self.current:
            q = [float(x) * self.scale for x in row['quad'].split(',')]
            self.canvas.create_polygon(*q, fill='', outline='lime', width=2)
        for x, y in self.points:
            self.canvas.create_oval(x * self.scale - 3, y * self.scale - 3, x * self.scale + 3, y * self.scale + 3, fill='red')
        self.info.config(text=f'{self.index + 1}/{len(self.files)} {self.path.name} | {self.mode} | corners: TL, TR, BR, BL | plates: {len(self.current)} | {self.license}')

    def undo(self, event=None):
        if self.points:
            self.points.pop()
        self.display()

    def undo_plate(self):
        if self.current:
            self.current.pop()
        self.display()

    def blur_mode(self):
        self.points, self.mode = [], 'blur'
        self.display()

    def click(self, event):
        x, y = event.x / self.scale, event.y / self.scale
        h, w = self.im.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        self.points.append((x, y))
        if self.mode == 'blur' and len(self.points) == 2:
            # Solid cover avoids leaving identifiable features in small Gaussian blur.
            a, b = np.asarray(self.points).astype(int)
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            self.im[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1] = 110
            self.boxes.append([*map(int, lo), *map(int, hi)])
            self.points, self.mode = [], 'plate'
        elif self.mode == 'plate' and len(self.points) == 4:
            kind = simpledialog.askstring('Type', 'type1 / type1a / type1b / other', parent=self.window)
            if kind not in TYPES:
                messagebox.showerror('Invalid type', 'Plate not saved. Use a supported type.')
                self.points = []
                self.display()
                return
            text = simpledialog.askstring('Number', 'Latin uppercase: ' + number_format_hint(kind), parent=self.window)
            if text is None:
                self.points = []
                self.display()
                return
            text = text.strip().upper()
            if not valid_number(text, kind):
                messagebox.showerror('Invalid number', 'Use ' + number_format_hint(kind))
                self.points = []
                self.display()
                return
            vehicle = messagebox.askyesno('Vehicle', 'Is the plate mounted on a vehicle?')
            conditions = simpledialog.askstring('Conditions', 'day,night,rain,snow,dirt,glare,motion_blur,angle', initialvalue='day') or 'day'
            q = np.asarray(self.points)
            lo, hi = q.min(0), q.max(0)
            self.current.append({'image': self.dest, 'plate_num': text, 'plate_type': kind,
                                 'bbox': ','.join(f'{v:.3f}' for v in [*lo, *(hi - lo)]),
                                 'quad': ','.join(f'{v:.3f}' for v in q.ravel()), 'is_vehicle': str(int(vehicle)),
                                 'is_synthetic': '0', 'source': self.source, 'license': self.license,
                                 'conditions': conditions})
            self.points = []
        self.display()

    def save(self):
        if self.points:
            messagebox.showerror('Unfinished plate', 'Finish or undo the corner selection before saving.')
            return
        if (self.root / self.dest).exists():
            messagebox.showerror('Already exists', 'Output appeared during annotation; it will not be overwritten.')
            return
        if self.job['record']:
            verify_file(self.path, self.job['record'])
        write_image(self.root / self.dest, self.im)
        self.rows.extend(self.current)
        write_meta(self.root, self.rows)
        write_labels(self.root, self.current, [self.dest])
        # A real negative frame has an empty TXT and zero meta.csv rows.
        audit = self.root / 'annotation_audit.jsonl'
        with audit.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'image': self.dest, 'method': 'manual', 'covered_faces': self.boxes,
                                'input_file': self.job['relative'], 'rights_registry': self.rights_path,
                                'source': self.source, 'license': self.license}) + '\n')
        if self.job['record']:
            save_provenance(self.root, {self.dest: {'registry': self.rights_path, 'record': self.job['record']}})
        self.index += 1
        self.load()

    def skip(self):
        self.index += 1
        self.load()

    def run(self):
        self.window.mainloop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--source', help='own_photo:author or exact source URL; use with --license')
    p.add_argument('--license', help='Actual license for all input images; use with --source')
    p.add_argument('--rights', help='Per-image JSON rights registry; replaces --source/--license')
    p.add_argument('--recursive', action='store_true', help='Include images in subdirectories of --input')
    a = p.parse_args()
    if a.rights and (a.source or a.license):
        p.error('--rights cannot be combined with --source or --license')
    if not a.rights and (not a.source or not a.license):
        p.error('Provide --rights, or both --source and --license')
    if tk is None:
        p.error('Tkinter/Tcl is unavailable. Install standard Python with Tcl/Tk (Windows), or python3-tk (Linux).')
    Annotator(a.input, a.data, a.source, a.license, a.rights, a.recursive).run()


if __name__ == '__main__':
    main()
