import argparse
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from quick_knn.lsh import integrate, fp_prob, fn_prob

parser = argparse.ArgumentParser()
parser.add_argument("--thresh", "-t", default=0.51, type=float)
parser.add_argument("--bits", "-b", default=4, type=int)
parser.add_argument("--fp_weight", "-fp", default=0.5, type=float)
args = parser.parse_args()

bits = args.bits
thresh = args.thresh
fp_weight = args.fp_weight
fn_weight = 1.0 - fp_weight

bs = []
rs = []

for b in range(1, bits + 1):
    max_r = bits // b
    for r in range(1, max_r + 1):
        bs.append(b)
        rs.append(r)

bs = np.array(bs)
rs = np.array(rs)

fig, ax = plt.subplots(2, len(bs) // 2)
ax = ax.ravel()

xp = np.linspace(0, thresh, 500)
xn = np.linspace(thresh, 1.0, 500)

fp_ = partial(fp_prob, bs, rs)
fn_ = partial(fn_prob, bs, rs)
error = integrate(fp_, 0.0, thresh) * fp_weight + integrate(fn_, thresh, 1.0) * fn_weight

for i, (b, r) in enumerate(zip(bs, rs)):

    yp = fp_prob(b, r, xp)
    yn = fn_prob(b, r, xn)

    ax[i].plot(xp, yp)
    ax[i].plot(xn, yn)
    ax[i].set_title(f"b: {b} r: {r} error: {error[i]:.4f}{'  BEST!!!!!' if error[i] == np.min(error) else ''}")

plt.show()
