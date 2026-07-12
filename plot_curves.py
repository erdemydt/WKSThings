#!/usr/bin/env python3
"""Plot 1D descriptor signatures exported by the C++ driver.

Reads wks_curves.csv / hks_curves.csv (column 0 = sample index, remaining
columns = one chosen vertex each) and saves PNGs.

Note the two descriptors run their scale axis in OPPOSITE directions:
  WKS: sample 0 = low energy  = GLOBAL  ...  high sample = LOCAL
  HKS: sample 0 = small t     = LOCAL   ...  high sample = GLOBAL
Curves that agree at one end and diverge at the other tell you the scale at
which those two points differ -- the whole point of the descriptor.
"""

import csv
import matplotlib.pyplot as plt


def load(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    data = [[float(x) for x in r] for r in rows[1:]]
    cols = list(zip(*data))
    return header, cols  # header[0]="sample"; cols[0]=x; cols[i]=curve i


def plot(path, title, xlabel, out, logy=False):
    header, cols = load(path)
    x = cols[0]
    plt.figure(figsize=(7, 4))
    for name, y in zip(header[1:], cols[1:]):
        plt.plot(x, y, label=name, linewidth=1.5)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("descriptor value")
    plt.legend(title="vertex")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print("wrote", out)


# HKS spans a huge dynamic range across time -> log-y makes the curves comparable.
plot("hks_curves.csv", "HKS signatures",
     "time sample  (0 = small t / LOCAL  ...  high = large t / GLOBAL)",
     "hks_curves.png", logy=True)

# WKS is already Ce-normalized -> linear y is fine.
plot("wks_curves.csv", "WKS signatures",
     "energy sample  (0 = low E / GLOBAL  ...  high = high E / LOCAL)",
     "wks_curves.png", logy=False)
