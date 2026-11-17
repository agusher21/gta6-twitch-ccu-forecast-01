#!/usr/bin/env python3
"""GTA VI Twitch hour-1 -> hours 2-24 quantile forecast (make.fun).

Model: spline-template of log(viewers(h)/viewers(1)) over past big launches,
fitted by weighted least squares (weights = p1-similarity x unlock match),
times an amplitude a(log p1) regressed from the comparables. Spread from
weighted residual sd per hour. Early-access launches are excluded from the
pool (their release-day curve is contaminated by pre-launch play).

Stdlib only, deterministic, no network. Reads /input/<id>.in (or
MAKEFUN_INPUT_DIR), writes /output/<id>.out (or MAKEFUN_OUTPUT_DIR).
"""
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

KNOTS = [2, 4, 6, 9, 12, 15, 18, 20, 22, 24]
TAU_P1 = 1.0
UNLOCK_BOOST = 4.0
SPREAD_MULT = 1.0
RIDGE = 1.0
DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def classify_unlock(text):
    if not text:
        return "unknown"
    t = text.lower()
    # strip negated mentions ("no early access" is NOT early access)
    t = t.replace("no early access", "").replace("no early-access", "")
    t = t.replace("no simultaneous launch", "").replace("no simultaneous", "")
    if "early access" in t or "early-access" in t:
        return "early"
    if "simultaneous" in t or "single global" in t:
        return "simultaneous"
    if "midnight local" in t or "rolling" in t:
        return "rolling"
    return "unknown"


def norm_ppf(p):
    """Acklam inverse normal CDF, good to ~1e-9."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def basis(h):
    out = []
    for i, k in enumerate(KNOTS):
        if h <= KNOTS[0]:
            v = 1.0 if i == 0 else 0.0
        elif h >= KNOTS[-1]:
            v = 1.0 if i == len(KNOTS) - 1 else 0.0
        else:
            v = 0.0
            for j in range(len(KNOTS) - 1):
                if KNOTS[j] <= h <= KNOTS[j + 1]:
                    f = (h - KNOTS[j]) / (KNOTS[j + 1] - KNOTS[j])
                    if i == j:
                        v = 1 - f
                    elif i == j + 1:
                        v = f
                    break
        out.append(v)
    return out


def solve_ls(X, y, w, ridge=1e-6):
    n = len(X[0])
    AtA = [[0.0] * n for _ in range(n)]
    Aty = [0.0] * n
    for row, yi, wi in zip(X, y, w):
        for i in range(n):
            Aty[i] += wi * row[i] * yi
            for j in range(n):
                AtA[i][j] += wi * row[i] * row[j]
    for i in range(n):
        AtA[i][i] += ridge
    M = [AtA[i] + [Aty[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for cc in range(col, n + 1):
                M[r][cc] -= f * M[col][cc]
    beta = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * beta[j] for j in range(i + 1, n))
        beta[i] = s / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0
    return beta


def comparable_curve(comp):
    """Return {hour: viewers} for the release day, or None if hour 1 missing."""
    try:
        ds = parse_ts(comp["day_start"])
    except Exception:
        return None
    by_hour = {}
    for pt in comp.get("hourly") or []:
        v = pt.get("viewers")
        if v is None or v <= 0:
            continue
        try:
            t = parse_ts(pt["from"])
        except Exception:
            continue
        by_hour[t] = v
    curve = {}
    for h in range(1, 25):
        v = by_hour.get(ds + timedelta(hours=h - 1))
        if v is not None:
            curve[h] = v
    if 1 not in curve:
        return None
    return curve


def forecast(case):
    p1 = case["known"][0]["viewers"]
    logp1 = math.log(p1)
    levels = case.get("levels") or DEFAULT_LEVELS
    z = [norm_ppf(t) for t in levels]

    tu = classify_unlock(case.get("notes") or "")
    if tu == "unknown":
        # default: big console launches roll at midnight local
        tu = "rolling"

    pool = []
    for comp in case.get("comparables") or []:
        if classify_unlock(comp.get("unlock")) == "early":
            continue
        curve = comparable_curve(comp)
        if curve is None:
            continue
        pool.append({"name": comp["name"], "curve": curve,
                     "p1": curve[1],
                     "unlock": classify_unlock(comp.get("unlock"))})

    weights = {}
    for c in pool:
        w = math.exp(-abs(math.log(c["p1"]) - logp1) / TAU_P1)
        if c["unlock"] == tu:
            w *= UNLOCK_BOOST
        weights[c["name"]] = w

    if len(pool) < 5 or not weights:
        # fallback: persistence with widening spread
        return {h: [max(1.0, p1 * math.exp(zz * 0.15 * math.sqrt(h - 1)))
                    for zz in z] for h in range(2, 25)}

    X, y, w = [], [], []
    for c in pool:
        for h in range(2, 25):
            if h in c["curve"]:
                X.append(basis(h))
                y.append(math.log(c["curve"][h] / c["curve"][1]))
                w.append(weights[c["name"]])
    beta = solve_ls(X, y, w, ridge=RIDGE)
    tmpl = {h: sum(b * bb for b, bb in zip(beta, basis(h)))
            for h in range(2, 25)}

    def amp(c):
        num = den = 0.0
        for h in range(2, 25):
            if h in c["curve"]:
                num += tmpl[h] * math.log(c["curve"][h] / c["curve"][1])
                den += tmpl[h] ** 2
        return num / den if den > 1e-9 else 1.0

    xs, ys, ws = [], [], []
    for c in pool:
        xs.append(math.log(c["p1"]))
        ys.append(amp(c))
        ws.append(weights[c["name"]])
    sw = sum(ws)
    mx = sum(wi * x for wi, x in zip(ws, xs)) / sw
    my = sum(wi * yy for wi, yy in zip(ws, ys)) / sw
    den = sum(wi * (x - mx) ** 2 for wi, x in zip(ws, xs))
    beta_a = (sum(wi * (x - mx) * (yy - my)
                  for wi, x, yy in zip(ws, xs, ys)) / den) if den > 1e-9 else 0.0
    alpha_a = my - beta_a * mx
    a_t = min(3.0, max(0.0, alpha_a + beta_a * logp1))

    out = {}
    for h in range(2, 25):
        res = []
        for c in pool:
            if h in c["curve"]:
                res.append((math.log(c["curve"][h] / c["curve"][1])
                            - amp(c) * tmpl[h], weights[c["name"]]))
        if len(res) >= 5:
            tot = sum(wi for _, wi in res)
            m = sum(v * wi for v, wi in res) / tot
            sd = math.sqrt(sum(wi * (v - m) ** 2 for v, wi in res) / tot)
        else:
            sd = 0.15 * math.sqrt(h - 1)
        sd = max(sd, 0.05)
        med = a_t * tmpl[h]
        qs = [p1 * math.exp(med + zz * sd * SPREAD_MULT) for zz in z]
        for i in range(1, len(qs)):
            qs[i] = max(qs[i], qs[i - 1])
        out[h] = [max(1.0, round(q, 2)) for q in qs]
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    if not files:
        print("no .in files in", in_dir, file=sys.stderr)
        return 1
    for fname in files:
        cid = fname[:-3]
        with open(os.path.join(in_dir, fname)) as fh:
            case = json.load(fh)
        fc = forecast(case)
        doc = {"hours": [{"hour": h, "q": fc[h]} for h in range(2, 25)]}
        with open(os.path.join(out_dir, cid + ".out"), "w") as fh:
            json.dump(doc, fh)
        print(f"{cid}: wrote {len(doc['hours'])} hours")
    return 0


if __name__ == "__main__":
    sys.exit(main())
