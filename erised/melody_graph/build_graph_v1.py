"""Graph proof-of-concept for the melody RM.

For each of 3 songs (Cruel Summer T5, 青花瓷 T5, But Daddy T0):
  1. Load notes (from melody_v2/*_notes_v2.txt) + phrases (from *_phrases_v2.txt).
  2. Build a HETEROGENEOUS graph with note-nodes + phrase-nodes and 5 edge types:
       - note → next note            (consecutive, melodic interval + duration ratio)
       - note → same-pitch-class note (recurrence, time gap)
       - note → its phrase           (membership)
       - phrase → next phrase        (consecutive, pitch_shift / range_ratio / contour_sim)
       - phrase → similar earlier phrase (repetition, contour cosine)
  3. Render a PNG with notes colored by pitch / sized by duration, phrase boxes around groups.
  4. Dump full feature CSVs (one for nodes, one for edges) so you can SEE the spreadsheet.
  5. Produce side-by-side T5 (Cruel Summer) vs T0 (But Daddy) render.

Run: python build_graph.py
Outputs land in /tmp/graph_poc/output/
"""
import re, json, pathlib, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.patches import FancyBboxPatch

ROOT = pathlib.Path("/Users/thuscodedzara/Desktop/phrase_demo")
OUT  = pathlib.Path("/tmp/graph_poc/output"); OUT.mkdir(parents=True, exist_ok=True)

SONGS = [
    {"id":"W041","slug":"cruel_summer",  "label":"Cruel Summer (T5 Masterpiece)",  "tier":5, "artist":"Taylor Swift"},
    {"id":"W089","slug":"qinghuaci",     "label":"青花瓷 (T5 Masterpiece)",          "tier":5, "artist":"Jay Chou"},
    {"id":"M023","slug":"but_daddy",     "label":"But Daddy I Love Him (T0 Horrible)","tier":0, "artist":"Taylor Swift"},
]

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def midi_to_name(m):
    m = int(round(m)); return f"{NOTE_NAMES[m%12]}{m//12 - 1}"

# ---------- loaders ----------
def load_notes(slug):
    """notes_v2 file format: 'idx | start | dur | MIDI | name | int' rows."""
    p = ROOT/"melody_v2"/f"{slug}_notes_v2.txt"
    rows = []
    for line in p.read_text().splitlines():
        m = re.match(r'\s*(\d+)\s*\|\s*([\d.]+)s\s*\|\s*([\d.]+)s\s*\|\s*(\d+)\s*\|', line)
        if m:
            rows.append({"idx":int(m[1]), "start":float(m[2]), "dur":float(m[3]), "midi":int(m[4])})
    return pd.DataFrame(rows)

def load_phrases(slug):
    """phrases_v2 file format: 'idx start end n_notes range_st pitches' rows."""
    p = ROOT/f"{slug}_phrases_v2.txt"
    rows = []
    for line in p.read_text().splitlines():
        m = re.match(r'\s*(\d+)\s+([\d.]+)s\s+([\d.]+)s\s+(\d+)\s+([\d.]+)\s+([\d\s]+)', line)
        if m:
            rows.append({"idx":int(m[1]), "start":float(m[2]), "end":float(m[3]),
                         "n_notes":int(m[4]), "range_st":float(m[5]),
                         "pitches":[int(p) for p in m[6].split()]})
    return pd.DataFrame(rows)

# ---------- graph build ----------
def build_song_graph(notes_df, phrases_df, song_meta):
    G = nx.MultiDiGraph(song=song_meta["label"], tier=song_meta["tier"])
    song_dur = max(notes_df["start"]+notes_df["dur"]) if len(notes_df) else 1

    # assign each note to a phrase by time overlap (note must fall inside phrase window)
    note_phrase = []
    for _, n in notes_df.iterrows():
        n_mid = n["start"] + n["dur"]/2
        match = phrases_df[(phrases_df["start"] <= n_mid) & (phrases_df["end"] >= n_mid)]
        note_phrase.append(int(match.iloc[0]["idx"]) if len(match) else -1)
    notes_df = notes_df.copy(); notes_df["phrase_idx"] = note_phrase

    # ── note nodes
    for _, n in notes_df.iterrows():
        if int(n["phrase_idx"]) < 0: continue
        ph = phrases_df.iloc[int(n["phrase_idx"])]
        onset_in_phrase = (n["start"] - ph["start"]) / max(ph["end"] - ph["start"], 1e-3)
        G.add_node(f"n{n['idx']}",
                   kind="note",
                   pitch_midi=int(n["midi"]),
                   pitch_class=int(n["midi"]) % 12,
                   octave=int(n["midi"]) // 12 - 1,
                   duration_s=float(n["dur"]),
                   onset_in_phrase=float(onset_in_phrase),
                   onset_in_song=float(n["start"] / song_dur),
                   phrase_idx=int(n["phrase_idx"]))

    # ── phrase nodes
    for _, p in phrases_df.iterrows():
        in_phrase = notes_df[notes_df["phrase_idx"] == p["idx"]]
        if len(in_phrase) == 0:
            mean_pitch = float(np.mean(p["pitches"])) if p["pitches"] else 0
        else:
            mean_pitch = float(in_phrase["midi"].mean())
        contour = list(in_phrase["midi"]) if len(in_phrase) else p["pitches"]
        slope = float(np.polyfit(range(len(contour)), contour, 1)[0]) if len(contour) > 1 else 0.0
        G.add_node(f"p{p['idx']}",
                   kind="phrase",
                   n_notes=int(p["n_notes"]),
                   mean_pitch=mean_pitch,
                   pitch_range=float(p["range_st"]),
                   contour_slope=slope,
                   duration_s=float(p["end"] - p["start"]),
                   position_in_song=float(p["start"] / song_dur))

    # ── edges: note → next note (consecutive within song)
    valid = notes_df[notes_df["phrase_idx"] >= 0].reset_index(drop=True)
    for i in range(len(valid) - 1):
        a, b = valid.iloc[i], valid.iloc[i+1]
        G.add_edge(f"n{a['idx']}", f"n{b['idx']}",
                   etype="note_consecutive",
                   interval_semitones=int(b["midi"] - a["midi"]),
                   duration_ratio=float(b["dur"] / max(a["dur"], 1e-3)),
                   time_gap_s=float(b["start"] - (a["start"] + a["dur"])))

    # ── edges: note → membership in phrase
    for _, n in valid.iterrows():
        G.add_edge(f"n{n['idx']}", f"p{n['phrase_idx']}",
                   etype="note_in_phrase")

    # ── edges: note recurrence (same pitch-class within 8s window, non-consecutive)
    n_list = valid.reset_index(drop=True)
    recur_count = 0
    for i in range(len(n_list)):
        a = n_list.iloc[i]
        for j in range(i+2, min(i+30, len(n_list))):  # skip immediate next (already consecutive)
            b = n_list.iloc[j]
            if b["start"] - a["start"] > 8.0: break
            if int(a["midi"]) % 12 == int(b["midi"]) % 12:  # same pitch class
                G.add_edge(f"n{a['idx']}", f"n{b['idx']}",
                           etype="note_recurrence",
                           time_gap_s=float(b["start"] - a["start"]))
                recur_count += 1
                break  # only nearest recurrence; otherwise edge explosion

    # ── edges: phrase → next phrase (consecutive)
    for i in range(len(phrases_df) - 1):
        a, b = phrases_df.iloc[i], phrases_df.iloc[i+1]
        a_node, b_node = G.nodes[f"p{a['idx']}"], G.nodes[f"p{b['idx']}"]
        G.add_edge(f"p{a['idx']}", f"p{b['idx']}",
                   etype="phrase_consecutive",
                   pitch_shift=float(b_node["mean_pitch"] - a_node["mean_pitch"]),
                   range_ratio=float(b_node["pitch_range"] / max(a_node["pitch_range"], 1e-3)),
                   duration_ratio=float(b_node["duration_s"] / max(a_node["duration_s"], 1e-3)))

    # ── edges: phrase repetition (cosine similarity of pitch contour vectors, non-consecutive)
    def contour_vec(ph_idx, length=8):
        in_ph = notes_df[notes_df["phrase_idx"] == ph_idx]
        if len(in_ph) == 0: return np.zeros(length)
        c = in_ph["midi"].values.astype(float)
        c = c - c.mean()  # normalize
        # resample to fixed length
        if len(c) == 1: return np.full(length, c[0])
        xs = np.linspace(0, 1, len(c)); xt = np.linspace(0, 1, length)
        return np.interp(xt, xs, c)
    rep_count = 0
    rep_edges_per = {}  # to limit
    for i in range(len(phrases_df)):
        a_vec = contour_vec(i)
        for j in range(i+2, len(phrases_df)):  # non-consecutive
            b_vec = contour_vec(j)
            if np.linalg.norm(a_vec) < 1e-6 or np.linalg.norm(b_vec) < 1e-6: continue
            sim = float(np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)))
            if sim > 0.85:  # high-similarity only
                G.add_edge(f"p{i}", f"p{j}",
                           etype="phrase_repetition",
                           similarity=sim,
                           bars_apart=int(j - i))
                rep_count += 1
                rep_edges_per[i] = rep_edges_per.get(i,0) + 1
                if rep_edges_per[i] >= 3: break  # cap per phrase
    return G, notes_df, recur_count, rep_count

# ---------- rendering ----------
PITCH_CMAP = plt.cm.viridis
def pitch_color(midi, lo=48, hi=84):
    t = max(0, min(1, (midi - lo) / (hi - lo)))
    return PITCH_CMAP(t)

def render_song(G, notes_df, phrases_df, song_meta, ax):
    """Notes laid out left→right by start_time, y = pitch; phrases as background boxes; edges colored by type."""
    # plot notes
    note_xy = {}
    for _, n in notes_df.iterrows():
        if n["phrase_idx"] < 0: continue
        x = float(n["start"])
        y = float(n["midi"])
        sz = 30 + 600 * min(n["dur"], 2.0)  # size by duration
        ax.scatter(x, y, s=sz, c=[pitch_color(y)], alpha=0.85, edgecolor='k', linewidth=0.5, zorder=3)
        note_xy[f"n{n['idx']}"] = (x, y)

    # phrase boxes (background)
    for _, p in phrases_df.iterrows():
        in_ph = notes_df[(notes_df["phrase_idx"] == p["idx"])]
        if len(in_ph) == 0: continue
        x0, x1 = float(p["start"]) - 0.3, float(p["end"]) + 0.3
        y0, y1 = float(in_ph["midi"].min()) - 1, float(in_ph["midi"].max()) + 1
        rect = mpatches.FancyBboxPatch((x0, y0), x1-x0, y1-y0,
                                       boxstyle="round,pad=0.2", linewidth=1,
                                       edgecolor='#888', facecolor='#f0f0f0', alpha=0.4, zorder=1)
        ax.add_patch(rect)
        ax.text(x0+0.1, y1+0.3, f"P{int(p['idx'])}", fontsize=6, color='#666', zorder=2)

    # consecutive note edges (gray)
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") == "note_consecutive" and u in note_xy and v in note_xy:
            x0,y0 = note_xy[u]; x1,y1 = note_xy[v]
            ax.plot([x0,x1],[y0,y1], color='#aaa', linewidth=0.4, alpha=0.7, zorder=2)
    # recurrence edges (red arcs)
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") == "note_recurrence" and u in note_xy and v in note_xy:
            x0,y0 = note_xy[u]; x1,y1 = note_xy[v]
            ax.plot([x0,x1],[y0,y1], color='crimson', linewidth=0.4, alpha=0.35, zorder=2, linestyle=':')

    # phrase repetition arcs (top of plot)
    pitch_max = notes_df["midi"].max()
    arc_y = pitch_max + 5
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") == "phrase_repetition":
            i, j = int(u[1:]), int(v[1:])
            pi, pj = phrases_df.iloc[i], phrases_df.iloc[j]
            mid_i = (pi["start"] + pi["end"]) / 2
            mid_j = (pj["start"] + pj["end"]) / 2
            ax.annotate('', xy=(mid_j, arc_y), xytext=(mid_i, arc_y),
                        arrowprops=dict(arrowstyle='-', color='blue', alpha=0.4,
                                       connectionstyle=f"arc3,rad=-0.3", linewidth=0.8), zorder=2)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("pitch (MIDI)")
    ax.set_title(f"{song_meta['label']} — {song_meta['artist']}\n"
                 f"{len(notes_df[notes_df['phrase_idx']>=0])} notes · {len(phrases_df)} phrases · "
                 f"{sum(1 for _,_,e in G.edges(data=True) if e.get('etype')=='note_recurrence')} recurrence edges · "
                 f"{sum(1 for _,_,e in G.edges(data=True) if e.get('etype')=='phrase_repetition')} phrase-repetition edges",
                 fontsize=11)
    ax.grid(alpha=0.2)

# ---------- main ----------
all_results = []
for song in SONGS:
    print(f"\n=== {song['label']} ===")
    try:
        notes_df = load_notes(song["slug"])
        phrases_df = load_phrases(song["slug"])
    except FileNotFoundError as e:
        print(f"  SKIP — missing input: {e}")
        continue
    print(f"  loaded {len(notes_df)} notes, {len(phrases_df)} phrases")
    G, notes_df_aug, n_rec, n_rep = build_song_graph(notes_df, phrases_df, song)
    n_note_nodes = sum(1 for _,d in G.nodes(data=True) if d.get("kind")=="note")
    n_phrase_nodes = sum(1 for _,d in G.nodes(data=True) if d.get("kind")=="phrase")
    print(f"  graph: {n_note_nodes} note-nodes + {n_phrase_nodes} phrase-nodes, "
          f"{n_rec} recurrence edges, {n_rep} phrase-repetition edges")

    # single-song render
    fig, ax = plt.subplots(figsize=(20, 6))
    render_song(G, notes_df_aug, phrases_df, song, ax)
    fig.tight_layout()
    png = OUT / f"{song['slug']}_graph.png"
    fig.savefig(png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {png}")

    # feature CSVs
    rows_n = [{"node_id":k, **{kk:vv for kk,vv in v.items() if kk!='kind'}} for k,v in G.nodes(data=True) if v.get("kind")=="note"]
    rows_p = [{"node_id":k, **{kk:vv for kk,vv in v.items() if kk!='kind'}} for k,v in G.nodes(data=True) if v.get("kind")=="phrase"]
    pd.DataFrame(rows_n).to_csv(OUT/f"{song['slug']}_notes_features.csv", index=False)
    pd.DataFrame(rows_p).to_csv(OUT/f"{song['slug']}_phrases_features.csv", index=False)
    rows_e = [{"src":u, "dst":v, **d} for u,v,d in G.edges(data=True)]
    pd.DataFrame(rows_e).to_csv(OUT/f"{song['slug']}_edges_features.csv", index=False)
    print(f"  wrote 3 CSVs in {OUT}")

    all_results.append({"meta":song, "G":G, "notes":notes_df_aug, "phrases":phrases_df})

# ── side-by-side T5 (Cruel Summer) vs T0 (But Daddy) ──
t5 = next((r for r in all_results if r["meta"]["slug"]=="cruel_summer"), None)
t0 = next((r for r in all_results if r["meta"]["slug"]=="but_daddy"),    None)
if t5 and t0:
    fig, axes = plt.subplots(2, 1, figsize=(22, 11))
    render_song(t5["G"], t5["notes"], t5["phrases"], t5["meta"], axes[0])
    render_song(t0["G"], t0["notes"], t0["phrases"], t0["meta"], axes[1])
    fig.suptitle("Taylor Swift A/B: Masterpiece vs Horrible — same-artist controlled contrast", fontsize=13, y=1.005)
    fig.tight_layout()
    png = OUT/"AB_t5_vs_t0_taylor_swift.png"
    fig.savefig(png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\nwrote SIDE-BY-SIDE: {png}")

print(f"\nALL DONE. outputs in {OUT}/")
