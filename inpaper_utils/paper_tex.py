#!/usr/bin/env python3
"""The paper's source as one string, with every \\input{...} expanded in place.

Since the paper split its experiments into sections/*.tex, the figures, tables and
labels no longer sit in iclr2027_conference.tex itself. Every tool that reads the
paper (sync_figures, sync_captions, table_rows) goes through `flat()`, so they see
what LaTeX sees instead of a main file with the experiments cut out of it.

A missing input is an error, not an empty string: silently skipping it would make
sync_figures report that the paper includes no figures at all.
"""
from pathlib import Path
import re

PAPER = Path(__file__).resolve().parents[1] / '__X_files/iclr2027'
MAIN = PAPER / 'iclr2027_conference.tex'
_INPUT = re.compile(r'\\input\{([^}]+)\}')


def _code(line):
    """The part of a line before its first unescaped %, i.e. what LaTeX reads."""
    m = re.search(r'(?<!\\)%', line)
    return line if m is None else line[:m.start()]


def flat(path=MAIN, _seen=None):
    _seen = set() if _seen is None else _seen
    path = Path(path).resolve()
    if path in _seen:
        raise SystemExit(f'paper_tex: {path} inputs itself')
    _seen = _seen | {path}
    out, missing = [], []
    for line in path.read_text().split('\n'):
        code, pos, pieces = _code(line), 0, []
        for m in _INPUT.finditer(code):
            name = m.group(1).strip()
            f = PAPER / (name if name.endswith('.tex') else name + '.tex')
            if not f.exists():
                missing.append(f.relative_to(PAPER))
                continue
            pieces += [line[pos:m.start()], flat(f, _seen)]
            pos = m.end()
        out.append(''.join(pieces) + line[pos:])
    if missing:
        raise SystemExit(f'paper_tex: {path.name} inputs files that are not in {PAPER}: '
                         + ', '.join(map(str, missing)))
    return '\n'.join(out)


if __name__ == '__main__':
    print(f'{len(flat().splitlines())} lines after expanding every \\input of {MAIN.name}')
