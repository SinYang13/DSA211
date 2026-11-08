"""Draws the chronological holdout and the expanding-window CV folds from the real splitter."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_utils import load_train, load_test
from src.cv_utils import get_cv_splitter

d = pd.to_datetime(load_train()['dteday']).reset_index(drop=True)
t = pd.to_datetime(load_test()['dteday'])
folds = list(get_cv_splitter(d).split(d))


def bar(ax, y, start, end, **kw):
    # Bars are drawn on ordinal days so each covers whole calendar days
    ax.barh(y, end.toordinal() - start.toordinal() + 1, left=start.toordinal(), height=.6, **kw)


fig, ax = plt.subplots(figsize=(12, 5))
for i, (tr, va) in enumerate(folds):
    y = len(folds) - i
    bar(ax, y, d[tr].min(), d[tr].max(), color='tab:blue', label='Train (fold)' if i == 0 else None)
    bar(ax, y, d[va].min(), d[va].max(), color='tab:orange', label='Validation (1 month)' if i == 0 else None)
bar(ax, 0, d.min(), d.max(), color='tab:blue', alpha=.5, label='Refit on all training data')
bar(ax, 0, t.min(), t.max(), color='tab:red', label='Locked holdout test')
ax.axvline(t.min().toordinal() - .5, color='k', ls='--', lw=1)
ax.set_yticks(range(len(folds) + 1))
ax.set_yticklabels(['Final refit / test'] + [f'Fold {i}' for i in range(len(folds), 0, -1)])
ticks = pd.date_range(d.min(), t.max(), freq='3MS')
ax.set_xticks([x.toordinal() for x in ticks])
ax.set_xticklabels([x.strftime('%b %Y') for x in ticks], rotation=45)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=4, fontsize=9)
ax.set_title(f'Chronological holdout (test from {t.min():%d %b %Y}) and expanding-window CV (k = {len(folds)})')
fig.tight_layout()
fig.savefig(ROOT / 'figures' / '07_cv_split_scheme.png', dpi=150)
print(len(folds), 'folds saved')
