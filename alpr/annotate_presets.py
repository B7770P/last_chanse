"""Review existing four-corner annotations and enter missing plate metadata."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from .annotate import Annotator, tk, messagebox


class PresetAnnotator(Annotator):
    def __init__(self, source, root, rights, presets):
        self.presets = json.loads(Path(presets).read_text(encoding='utf-8'))
        self.pending = []
        super().__init__(source, root, rights=rights)
        if self.index < len(self.jobs):
            controls = tk.Frame(self.window)
            controls.pack()
            tk.Button(controls, text='Use blue outline / Заполнить номер', command=self.accept_outline).pack(side=tk.LEFT)
            tk.Button(controls, text='Discard blue outline / Обвести заново', command=self.discard_outline).pack(side=tk.LEFT)

    def load(self):
        self.pending = []
        super().load()
        if self.index < len(self.jobs):
            self.pending = list(self.presets.get(self.job['relative'], []))
            self.display()

    def display(self):
        super().display()
        for quad in self.pending:
            self.canvas.create_polygon(*[float(v) * self.scale for pt in quad for v in pt], fill='', outline='cyan', width=3)

    def accept_outline(self):
        if not self.pending:
            return
        if self.points:
            messagebox.showerror('Unfinished selection', 'Finish or undo the current corner/face selection first.')
            return
        quad = self.pending[0]
        self.mode = 'plate'
        self.points = [tuple(p) for p in quad[:3]]
        before = len(self.current)
        # Reuse the existing dialogs and validation; never infer text from filenames.
        super().click(SimpleNamespace(x=quad[3][0] * self.scale, y=quad[3][1] * self.scale))
        if len(self.current) > before:
            self.pending.pop(0)
        self.points = []
        self.display()

    def discard_outline(self):
        if self.pending:
            self.pending.pop(0)
            self.display()

    def save(self):
        if self.pending:
            messagebox.showerror('Review outline', 'Use or discard the blue outline before saving.')
            return
        if not self.current:
            messagebox.showerror('No plate', 'Label the plate before saving, or use Skip image.')
            return
        super().save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='type1/ready100/images')
    p.add_argument('--data', default='dataset_type1')
    p.add_argument('--rights', default='type1/ready100/sources.json')
    p.add_argument('--presets', default='type1/ready100/presets.json')
    args = p.parse_args()
    PresetAnnotator(args.input, args.data, args.rights, args.presets).run()


if __name__ == '__main__':
    main()
