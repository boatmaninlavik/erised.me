"""Graph v2 — adds interval-detail features without changing structure.

NEW edge features on note_consecutive:
  - interval_quality_category (0=unison, 1=step, 2=consonant_leap, 3=dissonant_leap, 4=octave+)
  - interval_direction (-1/0/+1, explicit)
  - interval_magnitude_within_phrase (z-scored vs other intervals in same phrase)

NEW note-node feature:
  - pitch_deviation_from_phrase_mean (this MIDI - phrase mean MIDI)

All existing features and edges PRESERVED.

Run: python build_graph_v2.py
Output: /tmp/graph_poc/output_v2/
"""
import re, json, pathlib, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

ROOT = pathlib.Path("/Users/thuscodedzara/Desktop/phrase_demo")
OUT  = pathlib.Path("/tmp/graph_poc/output_v2"); OUT.mkdir(parents=True, exist_ok=True)

SONGS = [
    {"id":"W041","slug":"cruel_summer",  "label":"Cruel Summer (T5)",  "tier":5},
    {"id":"M023","slug":"but_daddy",     "label":"But Daddy I Love Him (T0)","tier":0},
]

def interval_quality(semitones: int) -> int:
    """0=unison 1=step 2=consonant_leap 3=dissonant_leap 4=octave+"""
    a = abs(int(semitones))
    if a == 0: return 0
    if a <= 2: return 1
    if a in {3, 4, 5, 7, 12}: return 2
    if a in {6, 8, 9, 10, 11}: return 3
    return 4  # > 12

def load_notes(slug):
    p = ROOT/"melody_v2"/f"{slug}_notes_v2.txt"
    rows = []
    for line in p.read_text().splitlines():
        m = re.match(r'\s*(\d+)\s*\|\s*([\d.]+)s\s*\|\s*([\d.]+)s\s*\|\s*(\d+)\s*\|', line)
        if m:
            rows.append({"idx":int(m[1]), "start":float(m[2]), "dur":float(m[3]), "midi":int(m[4])})
    return pd.DataFrame(rows)

def load_phrases(slug):
    p = ROOT/f"{slug}_phrases_v2.txt"
    rows = []
    for line in p.read_text().splitlines():
        m = re.match(r'\s*(\d+)\s+([\d.]+)s\s+([\d.]+)s\s+(\d+)\s+([\d.]+)\s+([\d\s]+)', line)
        if m:
            rows.append({"idx":int(m[1]), "start":float(m[2]), "end":float(m[3]),
                         "n_notes":int(m[4]), "range_st":float(m[5]),
                         "pitches":[int(p) for p in m[6].split()]})
    return pd.DataFrame(rows)

def build_song_graph_v2(notes_df, phrases_df, song_meta):
    G = nx.MultiDiGraph(song=song_meta["label"], tier=song_meta["tier"])
    song_dur = max(notes_df["start"]+notes_df["dur"]) if len(notes_df) else 1

    note_phrase = []
    for _, n in notes_df.iterrows():
        n_mid = n["start"] + n["dur"]/2
        match = phrases_df[(phrases_df["start"] <= n_mid) & (phrases_df["end"] >= n_mid)]
        note_phrase.append(int(match.iloc[0]["idx"]) if len(match) else -1)
    notes_df = notes_df.copy(); notes_df["phrase_idx"] = note_phrase

    # phrase mean pitches (for new note-level feature)
    phrase_mean = {}
    for _, p in phrases_df.iterrows():
        in_ph = notes_df[notes_df["phrase_idx"] == p["idx"]]
        phrase_mean[p["idx"]] = float(in_ph["midi"].mean()) if len(in_ph) else 0.0

    # phrase interval-magnitude stats (for new edge-level feature: z-score within phrase)
    phrase_intervals = {}  # phrase_idx -> list of consecutive intervals
    valid = notes_df[notes_df["phrase_idx"] >= 0].reset_index(drop=True)
    for i in range(len(valid) - 1):
        a, b = valid.iloc[i], valid.iloc[i+1]
        if a["phrase_idx"] != b["phrase_idx"]: continue
        phrase_intervals.setdefault(int(a["phrase_idx"]), []).append(int(b["midi"] - a["midi"]))
    phrase_int_stats = {pid: (np.mean(np.abs(ints)) if ints else 1.0,
                              np.std(np.abs(ints)) if len(ints) > 1 else 1.0)
                         for pid, ints in phrase_intervals.items()}

    # ── note nodes (with NEW pitch_deviation_from_phrase_mean) ──
    for _, n in notes_df.iterrows():
        if int(n["phrase_idx"]) < 0: continue
        ph = phrases_df.iloc[int(n["phrase_idx"])]
        onset_in_phrase = (n["start"] - ph["start"]) / max(ph["end"] - ph["start"], 1e-3)
        pmean = phrase_mean.get(int(n["phrase_idx"]), float(n["midi"]))
        G.add_node(f"n{n['idx']}",
                   kind="note",
                   pitch_midi=int(n["midi"]),
                   pitch_class=int(n["midi"]) % 12,
                   octave=int(n["midi"]) // 12 - 1,
                   duration_s=float(n["dur"]),
                   onset_in_phrase=float(onset_in_phrase),
                   onset_in_song=float(n["start"] / song_dur),
                   phrase_idx=int(n["phrase_idx"]),
                   # NEW v2 feature:
                   pitch_deviation_from_phrase_mean=float(n["midi"] - pmean))

    # ── phrase nodes (unchanged) ──
    for _, p in phrases_df.iterrows():
        in_phrase = notes_df[notes_df["phrase_idx"] == p["idx"]]
        if len(in_phrase) == 0:
            mp = float(np.mean(p["pitches"])) if p["pitches"] else 0
        else:
            mp = float(in_phrase["midi"].mean())
        contour = list(in_phrase["midi"]) if len(in_phrase) else p["pitches"]
        slope = float(np.polyfit(range(len(contour)), contour, 1)[0]) if len(contour) > 1 else 0.0
        G.add_node(f"p{p['idx']}",
                   kind="phrase",
                   n_notes=int(p["n_notes"]),
                   mean_pitch=mp,
                   pitch_range=float(p["range_st"]),
                   contour_slope=slope,
                   duration_s=float(p["end"] - p["start"]),
                   position_in_song=float(p["start"] / song_dur))

    # ── note→next-note edges (with NEW interval_quality_category, interval_direction, magnitude_within_phrase) ──
    for i in range(len(valid) - 1):
        a, b = valid.iloc[i], valid.iloc[i+1]
        semi = int(b["midi"] - a["midi"])
        ph_id = int(a["phrase_idx"]) if a["phrase_idx"] == b["phrase_idx"] else -1
        if ph_id >= 0 and ph_id in phrase_int_stats:
            mu, sd = phrase_int_stats[ph_id]
            mag_z = (abs(semi) - mu) / max(sd, 1e-6)
        else:
            mag_z = 0.0
        G.add_edge(f"n{a['idx']}", f"n{b['idx']}",
                   etype="note_consecutive",
                   interval_semitones=semi,
                   duration_ratio=float(b["dur"] / max(a["dur"], 1e-3)),
                   time_gap_s=float(b["start"] - (a["start"] + a["dur"])),
                   # NEW v2 edge features:
                   interval_quality_category=interval_quality(semi),
                   interval_direction=int(np.sign(semi)),
                   interval_magnitude_within_phrase=float(mag_z))

    # ── note→phrase membership ──
    for _, n in valid.iterrows():
        G.add_edge(f"n{n['idx']}", f"p{int(n['phrase_idx'])}", etype="note_in_phrase")

    # ── recurrence edges (unchanged) ──
    n_list = valid.reset_index(drop=True)
    for i in range(len(n_list)):
        a = n_list.iloc[i]
        for j in range(i+2, min(i+30, len(n_list))):
            b = n_list.iloc[j]
            if b["start"] - a["start"] > 8.0: break
            if int(a["midi"]) % 12 == int(b["midi"]) % 12:
                G.add_edge(f"n{a['idx']}", f"n{b['idx']}",
                           etype="note_recurrence",
                           time_gap_s=float(b["start"] - a["start"]))
                break

    # ── phrase→next-phrase edges (unchanged) ──
    for i in range(len(phrases_df) - 1):
        a, b = phrases_df.iloc[i], phrases_df.iloc[i+1]
        an, bn = G.nodes[f"p{a['idx']}"], G.nodes[f"p{b['idx']}"]
        G.add_edge(f"p{a['idx']}", f"p{b['idx']}",
                   etype="phrase_consecutive",
                   pitch_shift=float(bn["mean_pitch"] - an["mean_pitch"]),
                   range_ratio=float(bn["pitch_range"] / max(an["pitch_range"], 1e-3)),
                   duration_ratio=float(bn["duration_s"] / max(an["duration_s"], 1e-3)))

    # ── phrase-repetition (unchanged) ──
    def cv(ph_idx, length=8):
        ip = notes_df[notes_df["phrase_idx"] == ph_idx]
        if len(ip) == 0: return np.zeros(length)
        c = ip["midi"].values.astype(float); c = c - c.mean()
        if len(c) == 1: return np.full(length, c[0])
        xs = np.linspace(0, 1, len(c)); xt = np.linspace(0, 1, length)
        return np.interp(xt, xs, c)
    per_cap = {}
    for i in range(len(phrases_df)):
        a_vec = cv(i)
        for j in range(i+2, len(phrases_df)):
            b_vec = cv(j)
            if np.linalg.norm(a_vec) < 1e-6 or np.linalg.norm(b_vec) < 1e-6: continue
            sim = float(np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)))
            if sim > 0.85:
                G.add_edge(f"p{i}", f"p{j}", etype="phrase_repetition", similarity=sim, bars_apart=int(j-i))
                per_cap[i] = per_cap.get(i, 0) + 1
                if per_cap[i] >= 3: break
    return G

# ── run ──
song_summaries = []
for song in SONGS:
    notes = load_notes(song["slug"]); phrases = load_phrases(song["slug"])
    G = build_song_graph_v2(notes, phrases, song)

    # dump CSVs with ALL features (v1 + v2)
    rows_n = [{"node_id":k, **{kk:vv for kk,vv in v.items() if kk!='kind'}} for k,v in G.nodes(data=True) if v.get("kind")=="note"]
    rows_p = [{"node_id":k, **{kk:vv for kk,vv in v.items() if kk!='kind'}} for k,v in G.nodes(data=True) if v.get("kind")=="phrase"]
    rows_e = [{"src":u, "dst":v, **d} for u,v,d in G.edges(data=True)]
    pd.DataFrame(rows_n).to_csv(OUT/f"{song['slug']}_notes_v2.csv", index=False)
    pd.DataFrame(rows_p).to_csv(OUT/f"{song['slug']}_phrases_v2.csv", index=False)
    pd.DataFrame(rows_e).to_csv(OUT/f"{song['slug']}_edges_v2.csv", index=False)

    # aggregate stats for tier-comparison
    cons_edges = [e for _,_,e in G.edges(data=True) if e.get("etype")=="note_consecutive"]
    iq_dist = {q: sum(1 for e in cons_edges if e["interval_quality_category"]==q) for q in range(5)}
    n_total = max(1, sum(iq_dist.values()))
    iq_pct = {q: 100*c/n_total for q,c in iq_dist.items()}
    note_devs = [d.get("pitch_deviation_from_phrase_mean",0) for _,d in G.nodes(data=True) if d.get("kind")=="note"]
    mag_zs = [e.get("interval_magnitude_within_phrase",0) for e in cons_edges]
    summary = {
        "song": song["label"],
        "tier": song["tier"],
        "n_notes": sum(1 for _,d in G.nodes(data=True) if d.get("kind")=="note"),
        "n_phrases": sum(1 for _,d in G.nodes(data=True) if d.get("kind")=="phrase"),
        "n_consecutive_edges": len(cons_edges),
        "pct_unison": iq_pct[0],
        "pct_step":   iq_pct[1],
        "pct_consonant_leap": iq_pct[2],
        "pct_dissonant_leap": iq_pct[3],
        "pct_octave_plus":    iq_pct[4],
        "mean_abs_pitch_deviation_from_phrase_mean": float(np.mean(np.abs(note_devs))),
        "std_pitch_deviation_from_phrase_mean": float(np.std(note_devs)),
        "mean_interval_magnitude_z": float(np.mean(mag_zs)),
        "mean_phrase_pitch_range": float(np.mean([d.get("pitch_range",0) for _,d in G.nodes(data=True) if d.get("kind")=="phrase"])),
        "mean_contour_slope_abs": float(np.mean([abs(d.get("contour_slope",0)) for _,d in G.nodes(data=True) if d.get("kind")=="phrase"])),
        "n_phrase_repetition_edges": sum(1 for _,_,e in G.edges(data=True) if e.get("etype")=="phrase_repetition"),
        "n_recurrence_edges": sum(1 for _,_,e in G.edges(data=True) if e.get("etype")=="note_recurrence"),
    }
    song_summaries.append(summary)
    print(f"\n=== {song['label']} ===")
    for k,v in summary.items():
        if isinstance(v, float): print(f"  {k:50s} = {v:.3f}")
        else: print(f"  {k:50s} = {v}")

# T0 vs T5 comparison
print("\n" + "="*80)
print("T5 vs T0 DIFFERENTIATING FEATURES (does v2 surface real differences?)")
print("="*80)
print(f"{'feature':<50s} | {'T5 (Cruel)':>12s} | {'T0 (Daddy)':>12s} | {'delta':>10s}")
print("-"*100)
t5 = next(s for s in song_summaries if s["tier"] == 5)
t0 = next(s for s in song_summaries if s["tier"] == 0)
for k in t5:
    if k in ("song","tier"): continue
    v5, v0 = t5[k], t0[k]
    if isinstance(v5, (int,float)):
        d = v5 - v0
        print(f"{k:<50s} | {v5:>12.3f} | {v0:>12.3f} | {d:+10.3f}")

pd.DataFrame(song_summaries).to_csv(OUT/"v2_song_summary.csv", index=False)
print(f"\nCSVs → {OUT}")
