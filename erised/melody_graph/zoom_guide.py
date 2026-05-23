"""Reading guide: zoom into ONE single phrase from each song and label every
   note (pitch name + duration in seconds) and every interval on the edges."""
import re, pathlib, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = pathlib.Path("/Users/thuscodedzara/Desktop/phrase_demo")
OUT  = pathlib.Path("/tmp/graph_poc/output"); OUT.mkdir(parents=True, exist_ok=True)

NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def midi_to_name(m):
    m = int(round(m)); return f"{NOTES[m%12]}{m//12 - 1}"

def load_notes(slug):
    p = ROOT/"melody_v2"/f"{slug}_notes_v2.txt"
    rows = []
    for line in p.read_text().splitlines():
        m = re.match(r'\s*(\d+)\s*\|\s*([\d.]+)s\s*\|\s*([\d.]+)s\s*\|\s*(\d+)\s*\|', line)
        if m: rows.append({"idx":int(m[1]), "start":float(m[2]), "dur":float(m[3]), "midi":int(m[4])})
    return pd.DataFrame(rows)

def load_phrases(slug):
    p = ROOT/f"{slug}_phrases_v2.txt"
    rows = []
    for line in p.read_text().splitlines():
        m = re.match(r'\s*(\d+)\s+([\d.]+)s\s+([\d.]+)s\s+(\d+)\s+([\d.]+)', line)
        if m: rows.append({"idx":int(m[1]), "start":float(m[2]), "end":float(m[3]),
                          "n_notes":int(m[4]), "range_st":float(m[5])})
    return pd.DataFrame(rows)

def plot_phrase(ax, slug, song_label, phrase_idx, tier_color):
    notes_df = load_notes(slug); phrases_df = load_phrases(slug)
    ph = phrases_df[phrases_df["idx"] == phrase_idx].iloc[0]
    ns = notes_df[(notes_df["start"] >= ph["start"]-0.1) & (notes_df["start"] < ph["end"]+0.1)].reset_index(drop=True)
    if len(ns) == 0:
        ax.text(0.5, 0.5, "no notes in phrase", ha='center', transform=ax.transAxes); return

    # phrase box (background)
    ymin = ns["midi"].min() - 2; ymax = ns["midi"].max() + 4
    rect = mpatches.FancyBboxPatch((ph["start"]-0.05, ymin), ph["end"]-ph["start"]+0.1, ymax-ymin,
                                   boxstyle="round,pad=0.05", linewidth=2,
                                   edgecolor=tier_color, facecolor=tier_color, alpha=0.08, zorder=1)
    ax.add_patch(rect)
    ax.text(ph["start"], ymax-0.3, f"  Phrase P{int(ph['idx'])}   (lyric line)",
            fontsize=11, color=tier_color, weight='bold', zorder=5)

    # notes — each as a horizontal BAR (height = pitch, width = duration)
    for i, n in ns.iterrows():
        bar = mpatches.Rectangle((n["start"], n["midi"]-0.35), n["dur"], 0.7,
                                 facecolor=plt.cm.viridis((n["midi"]-48)/36),
                                 edgecolor='black', linewidth=1.2, alpha=0.9, zorder=3)
        ax.add_patch(bar)
        # label: pitch name + duration
        ax.text(n["start"] + n["dur"]/2, n["midi"] + 0.6, midi_to_name(n["midi"]),
                ha='center', fontsize=10, weight='bold', zorder=5)
        ax.text(n["start"] + n["dur"]/2, n["midi"] - 0.9, f"{n['dur']:.2f}s",
                ha='center', fontsize=8, color='#444', zorder=5)

    # consecutive-edge interval arrows + labels
    for i in range(len(ns) - 1):
        a, b = ns.iloc[i], ns.iloc[i+1]
        x0 = a["start"] + a["dur"]
        x1 = b["start"]
        y0 = a["midi"]
        y1 = b["midi"]
        interval = int(b["midi"] - a["midi"])
        sign = "+" if interval > 0 else ""
        color = '#d62728' if interval > 0 else ('#1f77b4' if interval < 0 else '#888')
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.85), zorder=2)
        if interval != 0:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx, my+0.3, f"{sign}{interval}", fontsize=9, color=color,
                    weight='bold', ha='center', zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor='white', edgecolor=color, lw=0.7))

    ax.set_xlim(ph["start"]-0.2, ph["end"]+0.4)
    ax.set_ylim(ymin-0.5, ymax+0.5)
    ax.set_xlabel("time (seconds)"); ax.set_ylabel("pitch (MIDI)")
    ax.set_title(f"{song_label} — phrase P{phrase_idx}   "
                 f"({len(ns)} notes, {ph['end']-ph['start']:.1f}s long)",
                 fontsize=12, color=tier_color)
    ax.grid(alpha=0.25)

# pick one expressive phrase from each song
# Cruel Summer P11 has the chorus reach (range ~17 semitones) — chosen as a "winner-pattern" example
# But Daddy P15 has comparable note count for fair compare
fig, axes = plt.subplots(2, 1, figsize=(18, 10))
plot_phrase(axes[0], "cruel_summer", "Cruel Summer (T5 Masterpiece)",   11, tier_color='#2ca02c')
plot_phrase(axes[1], "but_daddy",   "But Daddy I Love Him (T0 Horrible)", 15, tier_color='#d62728')

fig.suptitle("READING GUIDE — one phrase per song, every note / duration / interval labeled",
             fontsize=14, y=0.995)
plt.figtext(0.5, 0.01,
            "Each colored BAR is a note. Its WIDTH = duration in seconds (labeled). "
            "Its HEIGHT (Y-axis) = pitch (note name labeled). "
            "Each ARROW between bars is a consecutive-edge; the NUMBER on the arrow is the interval in semitones "
            "(red = jump up, blue = jump down). The phrase BOX is the lyric line containing those notes.",
            ha='center', fontsize=10, style='italic', wrap=True)
plt.tight_layout(rect=(0, 0.03, 1, 0.97))
png = OUT / "READING_GUIDE_one_phrase_each.png"
fig.savefig(png, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"wrote {png}")
