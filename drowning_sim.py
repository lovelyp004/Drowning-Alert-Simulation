"""
Track-level simulation of a vision-based early drowning alert pipeline (v2).

IMPORTANT: this is a SIMULATION. No real video is used. All numbers depend on the
assumed parameters in COND and on the behaviour models below. They demonstrate
metric trade-offs, not field performance.

Pipeline: detector + tracker output (head height above waterline h, ground-frame
speed v, visibility) -> persistence rule -> alert.
  suspicious frame : h_obs < H_THR and v_obs < V_THR
  alert            : suspicious for T seconds (gaps <= GAP_OK tolerated, track loss
                     <= LOST_OK tolerated, an ID switch resets the timer)
Usage: python drowning_sim.py main | sens
"""
import json, sys, time
import numpy as np
from numba import njit

FPS = 5
DT = 1.0 / FPS
H_THR, V_THR = 0.40, 0.25
GAP_OK, LOST_OK, COOLDOWN = 1.5, 3.0, 30.0
T_LIST = [5, 10, 15, 20]
OP_T = 10
N_EVENTS = 200

# Operational criteria (assumptions, derived in the paper, Section 3.4)
CRIT = dict(recall=0.90, fah=4.5, lat_med=15.0)

# sig: position noise (fraction of head height); drop: fraction of frames lost;
# burst: mean loss burst (s); ghost: spurious low static tracks per camera-hour;
# idsw: ID switches per swimmer-minute; n: swimmers in view;
# drift: water current (m/s) seen as ground-frame speed by non-swimming people;
# comp: fraction of the current removed by drift compensation.
COND = {
    "Overcast, clear pool (reference)": dict(sig=0.05, drop=0.02, burst=0.8, ghost=0.2, idsw=0.01, n=8, drift=0.0, comp=0.0),
    "Glare, calm pool":                 dict(sig=0.10, drop=0.10, burst=1.0, ghost=2.0, idsw=0.01, n=8, drift=0.0, comp=0.0),
    "Glare, rippled pool":              dict(sig=0.15, drop=0.20, burst=1.2, ghost=4.0, idsw=0.02, n=8, drift=0.0, comp=0.0),
    "Artificial light, turbid pool":    dict(sig=0.12, drop=0.25, burst=1.5, ghost=1.0, idsw=0.01, n=8, drift=0.0, comp=0.0),
    "Crowded pool, any light":          dict(sig=0.08, drop=0.15, burst=1.5, ghost=0.5, idsw=0.15, n=20, drift=0.0, comp=0.0),
    "Riverbank ghat, current uncompensated": dict(sig=0.15, drop=0.30, burst=2.0, ghost=3.0, idsw=0.20, n=20, drift=0.30, comp=0.0),
    "Riverbank ghat, current compensated":   dict(sig=0.15, drop=0.30, burst=2.0, ghost=3.0, idsw=0.20, n=20, drift=0.30, comp=0.8),
}


@njit(cache=True)
def detect(h, v, vis, reset, tp, start, fps, hthr, vthr, gapok, lostok, cool):
    dt = 1.0 / fps
    timer = 0.0; gap = 0.0; lost = 0.0; cd = 0.0
    n = 0; first = -1
    for i in range(len(h)):
        if reset[i]:
            timer = 0.0; gap = 0.0; lost = 0.0
        if cd > 0:
            cd -= dt
            continue
        if not vis[i]:
            lost += dt
            if lost > lostok:
                timer = 0.0; gap = 0.0
            continue
        lost = 0.0
        if h[i] < hthr and v[i] < vthr:
            timer += dt; gap = 0.0
            if timer >= tp:
                n += 1
                if first < 0 and i >= start:
                    first = i
                timer = 0.0; cd = cool
        else:
            gap += dt
            if gap > gapok:
                timer = 0.0; gap = 0.0
    return n, first


def burst_mask(rng, T, frac, mean_len):
    mask = np.zeros(T, bool)
    if frac <= 0:
        return mask
    nb = rng.poisson(frac / mean_len * T * DT)
    if nb == 0:
        return mask
    starts = rng.integers(0, T, nb)
    lens = np.maximum(1, (rng.exponential(mean_len, nb) * FPS).astype(int))
    d = np.zeros(T + 1, int)
    np.add.at(d, starts, 1)
    np.add.at(d, np.minimum(starts + lens, T), -1)
    return np.cumsum(d[:T]) > 0


def gen_normal(rng, T, pf=0.01):
    """True head height h (1 = high, 0 = waterline), speed v, submerged, drifting flags.
    drifting = person is not swimming or holding a fixed support, so a current moves them."""
    h = np.empty(T); v = np.empty(T)
    sub = np.zeros(T, bool); dr = np.zeros(T, bool)
    p = np.array([0.46 - pf, 0.20, 0.15, 0.10, pf, 0.09]); p = p / p.sum()
    i = 0
    while i < T:
        k = rng.choice(6, p=p)  # swim, play, tread, rest at support, float on back, dive
        d = [rng.uniform(30, 180), rng.uniform(20, 120), rng.uniform(20, 300),
             rng.uniform(30, 300), rng.uniform(10, 90), rng.uniform(4, 12)][k]
        n = min(int(d * FPS), T - i)
        t = np.arange(n) * DT
        sl = slice(i, i + n)
        if k == 0:
            h[sl] = 0.8 + 0.08 * np.sin(2 * np.pi * t / 1.2) + rng.normal(0, 0.04, n)
            v[sl] = np.clip(rng.normal(0.8, 0.15), 0.4, 1.3) + rng.normal(0, 0.05, n)
        elif k == 1:
            nb = int(np.ceil(n / (2 * FPS))) + 1
            h[sl] = np.repeat(rng.uniform(0.2, 0.95, nb), 2 * FPS)[:n]
            v[sl] = np.repeat(rng.uniform(0.0, 0.6, nb), 2 * FPS)[:n]
            dr[sl] = True
        elif k == 2:
            h[sl] = 0.55 + rng.normal(0, 0.06, n)
            v[sl] = np.abs(rng.normal(0.08, 0.04, n)); dr[sl] = True
        elif k == 3:
            h[sl] = 0.70 + rng.normal(0, 0.05, n)
            v[sl] = np.abs(rng.normal(0.02, 0.02, n))
        elif k == 4:
            h[sl] = 0.30 + rng.normal(0, 0.04, n)
            v[sl] = np.abs(rng.normal(0.05, 0.03, n)); dr[sl] = True
        else:
            h[sl] = 0.0; v[sl] = 0.2; sub[sl] = True; dr[sl] = True
        i += n
    return h, v, sub, dr


def gen_distress(rng, dis_s=40, pf=0.01, pre_s=20, post_s=20):
    T = int((pre_s + dis_s + post_s) * FPS)
    h, v, sub, dr = gen_normal(rng, T, pf)
    a, b = int(pre_s * FPS), int((pre_s + dis_s) * FPS)
    h[a:b] = rng.normal(0.20, 0.07, b - a)
    v[a:b] = np.abs(rng.normal(0.05, 0.03, b - a))
    sub[a:b] = False; dr[a:b] = True
    for _ in range(rng.poisson(dis_s / 6.0)):
        s = rng.integers(a, b)
        sub[s:s + int(rng.uniform(1, 3) * FPS)] = True
    h[b:] = 0.0; v[b:] = 0.0; sub[b:] = True; dr[b:] = True
    return h, v, sub, dr, a


def observe(rng, h, v, sub, dr, c):
    T = len(h)
    vis = (~sub) & (~burst_mask(rng, T, c["drop"], c["burst"]))
    h_obs = h + rng.normal(0, c["sig"], T)
    resid = c["drift"] * (1.0 - c["comp"])
    v_obs = np.abs(v + resid * dr + rng.normal(0, 1.5 * c["sig"], T))
    reset = np.zeros(T, bool)
    nsw = rng.poisson(c["idsw"] * T * DT / 60.0)
    if nsw:
        reset[rng.integers(0, T, nsw)] = True
    return h_obs, v_obs, vis, reset


def run(h, v, vis, reset, tp, start=0):
    return detect(h, v, vis, reset, float(tp), int(start), FPS, H_THR, V_THR,
                  GAP_OK, LOST_OK, COOLDOWN)


def simulate(c, seed, hours, dis_s=40, pf=0.01):
    rng = np.random.default_rng(seed)
    hits = {t: 0 for t in T_LIST}; lat = {t: [] for t in T_LIST}
    for _ in range(N_EVENTS):
        h, v, sub, dr, onset = gen_distress(rng, dis_s, pf)
        ho, vo, vis, rs = observe(rng, h, v, sub, dr, c)
        for t in T_LIST:
            _, first = run(ho, vo, vis, rs, t, onset)
            if first >= 0:
                hits[t] += 1; lat[t].append((first - onset) * DT)
    alerts = {t: 0 for t in T_LIST}
    T = int(hours * 3600 * FPS)
    for _ in range(c["n"]):
        h, v, sub, dr = gen_normal(rng, T, pf)
        ho, vo, vis, rs = observe(rng, h, v, sub, dr, c)
        for t in T_LIST:
            alerts[t] += run(ho, vo, vis, rs, t, 0)[0]
    for _ in range(rng.poisson(c["ghost"] * hours)):  # spurious static low tracks
        d = int(rng.uniform(8, 30) * FPS)
        ho = rng.normal(0.25, c["sig"], d); vo = np.abs(rng.normal(0, 0.1, d))
        vis = np.ones(d, bool); rs = np.zeros(d, bool)
        for t in T_LIST:
            alerts[t] += run(ho, vo, vis, rs, t, 0)[0]
    return hits, lat, alerts


def evaluate(c, seeds, hours, base_seed, dis_s=40, pf=0.01):
    rec = {t: [] for t in T_LIST}; fah = {t: [] for t in T_LIST}; lats = {t: [] for t in T_LIST}
    for s in range(seeds):
        hits, lat, alerts = simulate(c, base_seed + s, hours, dis_s, pf)
        for t in T_LIST:
            rec[t].append(hits[t] / N_EVENTS); fah[t].append(alerts[t] / hours); lats[t] += lat[t]
    out = {}
    for t in T_LIST:
        r, f = np.array(rec[t]), np.array(fah[t])
        L = np.array(lats[t]) if lats[t] else np.array([np.nan])
        sd = lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0
        out[str(t)] = dict(recall=float(r.mean()), recall_sd=sd(r), fah=float(f.mean()), fah_sd=sd(f),
                           precision=float(r.mean() / (r.mean() + 10 * f.mean())) if (r.mean() + f.mean()) > 0 else 0.0,
                           lat_med=(None if np.isnan(L).all() else float(np.nanmedian(L))), lat_p90=(None if np.isnan(L).all() else float(np.nanpercentile(L, 90))),
                           meets=bool(r.mean() >= CRIT["recall"] and f.mean() <= CRIT["fah"]
                                      and (not np.isnan(L).all()) and np.nanmedian(L) <= CRIT["lat_med"]))
    return out


def scaled(c, **k):
    c2 = dict(c)
    for key, f in k.items():
        c2[key] = c[key] * f
    c2["drop"] = min(c2["drop"], 0.6)
    return c2


def main():
    res = {}
    for ci, (name, c) in enumerate(COND.items()):
        t0 = time.time()
        res[name] = evaluate(c, seeds=10, hours=10, base_seed=1000 * ci)
        print(name, round(time.time() - t0), "s", flush=True)
    json.dump(res, open("sim_results.json", "w"), indent=2)


def sens():
    out = {"joint": {}, "one_at_a_time": {}, "behaviour": {}}
    S, HRS = 5, 10
    for ci, (name, c) in enumerate(COND.items()):
        out["joint"][name] = {}
        for f in (0.5, 2.0):
            cc = scaled(c, sig=f, drop=f, ghost=f, idsw=f)
            out["joint"][name][str(f)] = evaluate(cc, S, HRS, 50000 + 100 * ci)
        print("joint", name, flush=True)
    for name in ("Overcast, clear pool (reference)", "Riverbank ghat, current uncompensated"):
        c = COND[name]; out["one_at_a_time"][name] = {}
        for p in ("sig", "drop", "ghost", "idsw"):
            for f in (0.5, 2.0):
                out["one_at_a_time"][name][f"{p}x{f}"] = evaluate(scaled(c, **{p: f}), S, HRS, 70000)
        if c["drift"] > 0:
            for d in (0.15, 0.45):
                cc = dict(c); cc["drift"] = d
                out["one_at_a_time"][name][f"drift={d}"] = evaluate(cc, S, HRS, 70000)
        print("oat", name, flush=True)
    for name in ("Overcast, clear pool (reference)", "Riverbank ghat, current uncompensated"):
        c = COND[name]; out["behaviour"][name] = {}
        for label, kw in (("distress 20 s", dict(dis_s=20)), ("distress 60 s", dict(dis_s=60)),
                          ("no floaters", dict(pf=0.0)), ("3x floaters", dict(pf=0.03))):
            out["behaviour"][name][label] = evaluate(c, S, HRS, 90000, **kw)
        print("beh", name, flush=True)
    json.dump(out, open("sens_results.json", "w"), indent=2)


if __name__ == "__main__":
    {"main": main, "sens": sens}[sys.argv[1]]()
