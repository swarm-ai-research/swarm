#!/usr/bin/env python3
"""Figures for docs/research/hyperspace-two-swarms-lessons.md (bead umzu).

Reads the CSV artifacts of ``experiments/correlation_asymmetry.py`` and writes,
for each figure, a self-contained interactive HTML page (Plotly via CDN, theme
aware, crosshair hover, table view) and a static PNG twin:

    two-swarms-rho-sweep      attacker hit vs defender stages over rho
    two-swarms-verify-rules   defender catch over rho under three aggregation rules
    two-swarms-structured     defender catch over achieved rho by family structure

Every number plotted comes from the run artifacts. The experiment is analytic
and runs in well under a second, so with no ``--run`` this script regenerates
the three runs it needs (unanimous / majority / any_keeps) into a temp dir.

Usage: python scripts/plot_correlation_asymmetry.py [--run DIR] [--out docs/research/figures]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# dataviz reference palette, categorical slots 1-3 (validated light + dark)
LIGHT = {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e0",
         "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a"}
DARK = {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#333331",
        "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70"}
RULES = ("unanimous", "majority", "any_keeps")
RULE_LABEL = {"unanimous": "unanimous keep (any verifier may drop)",
              "majority": "majority keep", "any_keeps": "any verifier keeps"}


def read(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def run_experiment(rule: str, root: Path) -> Path:
    out = root / rule
    subprocess.run([sys.executable, "-m", "experiments.correlation_asymmetry",
                    "--output", str(out), "--seed", "42", "--verify-rule", rule],
                   check=True, capture_output=True)
    return next(out.glob("*correlation_asymmetry*"))


# ---------------------------------------------------------------------------
# series
# ---------------------------------------------------------------------------


def series_rho_sweep(run: Path) -> Dict[str, Any]:
    rows = read(run / "csv" / "exchangeable.csv")
    x = [float(r["rho"]) for r in rows]
    return {
        "title": "Correlation taxes both swarms; the defender's catch is non-monotone",
        "x": x, "xlabel": "pairwise correlation ρ", "ylabel": "probability per round",
        "series": [
            {"name": "attacker: some agent hits (= defender detect)", "short": "attacker hit",
             "y": [float(r["attacker_hit"]) for r in rows]},
            {"name": "defender: true finding survives verification", "short": "retain",
             "y": [float(r["defender_retain"]) for r in rows]},
            {"name": "defender: catch (detect × retain)", "short": "catch",
             "y": [float(r["defender_catch"]) for r in rows]},
        ],
        "note": "q = 0.3, N = 8 detectors, 5 verifiers, false-drop 0.4, unanimous keep. "
                "Detection tracks the attacker exactly; retention moves the other way; "
                "their product peaks at ρ = 0.65.",
    }


def series_rules(runs: Dict[str, Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"title": "The aggregation rule, not ρ, sets the defender's catch rate",
                           "xlabel": "pairwise correlation ρ", "ylabel": "defender catch per round",
                           "series": []}
    for rule in RULES:
        rows = read(runs[rule] / "csv" / "exchangeable.csv")
        out["x"] = [float(r["rho"]) for r in rows]
        out["series"].append({"name": RULE_LABEL[rule], "short": rule.replace("_", " "),
                              "y": [float(r["defender_catch"]) for r in rows]})
    out["note"] = ("Same parameters as the sweep above. Under unanimous keep, independent verifiers "
                   "each get a shot at dropping a true finding, so correlation compensates; majority "
                   "keep restores ρ = 0 as optimal at 2.9× the catch rate.")
    return out


def series_structured(run: Path) -> Dict[str, Any]:
    rows = [r for r in read(run / "csv" / "structured.csv") if r["saturated"] == "0"]
    facets = []
    for fam in ("1", "2", "4"):
        sub = [r for r in rows if r["families"] == fam]
        spreads = ["0.0"] if fam == "1" else ["0.0", "0.5", "1.0"]
        series = []
        for sp in spreads:
            pts = sorted((float(r["rho_achieved_verify"]), float(r["defender_catch"]))
                         for r in sub if r["spread"] == sp)
            name = "exchangeable" if fam == "1" else f"spread {sp}"
            series.append({"name": name, "x": [p[0] for p in pts], "y": [p[1] for p in pts]})
        title = "1 family (exchangeable)" if fam == "1" else f"{fam} families"
        facets.append({"title": title, "series": series})
    return {"title": "ρ* is an artifact of correlation structure, not a tuning target",
            "xlabel": "achieved verifier ρ̄", "ylabel": "defender catch per round", "facets": facets,
            "note": "Two-level shock: 'spread' is the share of correlation held within families. "
                    "Saturated cells (requested ρ̄ unreachable) are omitted. The optimum is interior "
                    "in every topology, but its location moves from 0.10 to 1.00."}


# ---------------------------------------------------------------------------
# HTML (Plotly, theme aware, table view)
# ---------------------------------------------------------------------------

_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root{{color-scheme:light;--surface:{l[surface]};--text:{l[text]};--muted:{l[muted]};--grid:{l[grid]};--s1:{l[s1]};--s2:{l[s2]};--s3:{l[s3]}}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{color-scheme:dark;--surface:{d[surface]};--text:{d[text]};--muted:{d[muted]};--grid:{d[grid]};--s1:{d[s1]};--s2:{d[s2]};--s3:{d[s3]}}}}}
:root[data-theme=dark]{{color-scheme:dark;--surface:{d[surface]};--text:{d[text]};--muted:{d[muted]};--grid:{d[grid]};--s1:{d[s1]};--s2:{d[s2]};--s3:{d[s3]}}}
body{{margin:0;background:var(--surface);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
#plot{{width:100%;height:{height}px}}
p.note{{margin:4px 16px 8px;color:var(--muted);font-size:13px}}
details{{margin:0 16px 12px;color:var(--muted);font-size:13px}}
table{{border-collapse:collapse;margin-top:6px}} td,th{{padding:2px 10px 2px 0;text-align:right;font-variant-numeric:tabular-nums}} th:first-child,td:first-child{{text-align:left}}
</style></head><body>
<div id="plot" role="img" aria-label="{title}"></div>
<p class="note">{note}</p>
<details><summary>Data table</summary>{table}</details>
<script>
const SPEC={spec};
const q=new URLSearchParams(location.search).get('theme'); if(q==='dark'||q==='light') document.documentElement.dataset.theme=q;
function css(v){{return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}}
function draw(){{
  const c=['--s1','--s2','--s3'].map(css), text=css('--text'), muted=css('--muted'), grid=css('--grid'), surface=css('--surface');
  const traces=[], ann=[], layout={{paper_bgcolor:surface, plot_bgcolor:surface, font:{{color:text,size:13}},
    margin:{{l:56,r:24,t:SPEC.facets?68:44,b:48}}, hovermode:'x unified', showlegend:true,
    legend:{{orientation:'h',x:0,y:-0.18,font:{{color:muted}}}},
    title:{{text:SPEC.title,font:{{size:15}},x:0,xanchor:'left'}}, annotations:ann}};
  const axis=(t)=>({{title:{{text:t,font:{{color:muted,size:12}}}},gridcolor:grid,zeroline:false,linecolor:grid,tickfont:{{color:muted}},showline:false}});
  if(SPEC.facets){{
    const n=SPEC.facets.length, gap=0.06, w=(1-gap*(n-1))/n;
    SPEC.facets.forEach((f,i)=>{{
      const xa='x'+(i?i+1:''), ya='y'+(i?i+1:'');
      const xk='xaxis'+(i?i+1:''), yk='yaxis'+(i?i+1:'');
      layout[xk]=Object.assign(axis(SPEC.xlabel),{{domain:[i*(w+gap),i*(w+gap)+w],range:[-0.02,1.02]}});
      layout[yk]=Object.assign(axis(i?'':SPEC.ylabel),{{range:[0,0.26],anchor:xa}});
      if(i) layout[yk].showticklabels=false;
      ann.push({{text:f.title,x:i*(w+gap),y:1.0,xref:'paper',yref:'paper',showarrow:false,xanchor:'left',yanchor:'bottom',yshift:4,font:{{color:muted,size:12}}}});
      f.series.forEach((s,j)=>traces.push({{type:'scatter',mode:'lines+markers',name:s.name,x:s.x,y:s.y,xaxis:xa,yaxis:ya,
        line:{{color:c[j],width:2}},marker:{{size:5,color:c[j]}},showlegend:i===1||(i===0&&n===1),legendgroup:s.name,
        hovertemplate:'ρ̄ %{{x:.2f}}: %{{y:.3f}}<extra>'+s.name+'</extra>'}}));
    }});
  }} else {{
    layout.xaxis=Object.assign(axis(SPEC.xlabel),{{range:[-0.02,1.02]}});
    layout.yaxis=Object.assign(axis(SPEC.ylabel),{{range:[0,1.0]}});
    SPEC.series.forEach((s,j)=>{{
      traces.push({{type:'scatter',mode:'lines',name:s.name,x:SPEC.x,y:s.y,line:{{color:c[j],width:2}},
        hovertemplate:'%{{y:.3f}}<extra>'+s.name+'</extra>'}});
      // direct label at the right end; contrast relief for the aqua slot on light
      const k=SPEC.label_at, left=k===0;
      ann.push({{text:s.short,x:SPEC.x[k],y:s.y[k],xanchor:left?'left':'right',yanchor:'bottom',xshift:left?6:-6,showarrow:false,font:{{color:text,size:11}}}});
    }});
    if(SPEC.peak) ann.push({{text:'ρ* = '+SPEC.peak.x,x:SPEC.peak.x,y:SPEC.peak.y,ax:0,ay:-32,showarrow:true,arrowcolor:muted,font:{{color:text,size:11}}}});
  }}
  Plotly.react('plot',traces,layout,{{displayModeBar:false,responsive:true}});
}}
draw(); matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
</script></body></html>
"""


def label_end(spec: Dict[str, Any]) -> int:
    """Index (0 or last) of the x-end whose closest pair of series is farthest apart."""
    ys = [s["y"] for s in spec["series"]]

    def min_gap(i: int) -> float:
        v = sorted(y[i] for y in ys)
        return min(b - a for a, b in zip(v, v[1:], strict=False)) if len(v) > 1 else 1.0

    return 0 if min_gap(0) > min_gap(-1) else len(spec["x"]) - 1


def table_html(spec: Dict[str, Any]) -> str:
    if spec.get("facets"):
        head = "<tr><th>facet</th><th>series</th><th>ρ̄</th><th>catch</th></tr>"
        body = "".join(
            f"<tr><td>{f['title']}</td><td>{s['name']}</td><td>{x:.2f}</td><td>{y:.3f}</td></tr>"
            for f in spec["facets"] for s in f["series"] for x, y in zip(s["x"], s["y"], strict=True))
        return f"<table>{head}{body}</table>"
    head = "<tr><th>ρ</th>" + "".join(f"<th>{s['name']}</th>" for s in spec["series"]) + "</tr>"
    body = "".join("<tr><td>%.2f</td>" % x + "".join("<td>%.3f</td>" % s["y"][i] for s in spec["series"]) + "</tr>"
                   for i, x in enumerate(spec["x"]))
    return f"<table>{head}{body}</table>"


def write_html(spec: Dict[str, Any], path: Path, height: int = 420) -> None:
    js_spec = {k: v for k, v in spec.items() if k != "note"}
    if not spec.get("facets"):
        js_spec["label_at"] = label_end(spec)
    path.write_text(_HTML.format(title=spec["title"], note=spec["note"], table=table_html(spec),
                                 spec=json.dumps(js_spec), height=height, l=LIGHT, d=DARK))


# ---------------------------------------------------------------------------
# PNG twins (light theme)
# ---------------------------------------------------------------------------


def _style(ax, p, xlabel, ylabel):
    ax.set_facecolor(p["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(p["grid"])
    ax.grid(True, color=p["grid"], linewidth=0.8)
    ax.tick_params(colors=p["muted"], labelsize=9)
    ax.set_xlabel(xlabel, color=p["muted"], fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=p["muted"], fontsize=10)


def write_png(spec: Dict[str, Any], path: Path) -> None:
    p = LIGHT
    colors = [p["s1"], p["s2"], p["s3"]]
    if spec.get("facets"):
        n = len(spec["facets"])
        fig, axes = plt.subplots(1, n, figsize=(10, 3.6), sharey=True, facecolor=p["surface"])
        for i, (ax, f) in enumerate(zip(axes, spec["facets"], strict=True)):
            for j, s in enumerate(f["series"]):
                ax.plot(s["x"], s["y"], color=colors[j], lw=2, marker="o", ms=3.5, label=s["name"])
            ax.set_title(f["title"], color=p["muted"], fontsize=10, loc="left")
            _style(ax, p, spec["xlabel"], spec["ylabel"] if i == 0 else "")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0, 0.26)
        axes[1].legend(frameon=False, fontsize=9, labelcolor=p["text"], loc="upper left")
    else:
        fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=p["surface"])
        k = label_end(spec)
        for j, s in enumerate(spec["series"]):
            ax.plot(spec["x"], s["y"], color=colors[j], lw=2, label=s["name"])
            ax.annotate(s["short"], (spec["x"][k], s["y"][k]), xytext=(6 if k == 0 else -6, 3),
                        textcoords="offset points", ha="left" if k == 0 else "right", va="bottom",
                        fontsize=8.5, color=p["text"])
        if spec.get("peak"):
            ax.annotate(f"ρ* = {spec['peak']['x']}", (spec["peak"]["x"], spec["peak"]["y"]),
                        xytext=(0, 22), textcoords="offset points", ha="center", fontsize=9, color=p["text"],
                        arrowprops={"arrowstyle": "-", "color": p["muted"]})
        _style(ax, p, spec["xlabel"], spec["ylabel"])
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=p["text"], loc="upper right", bbox_to_anchor=(1, 0.92))
    fig.suptitle(spec["title"], x=0.01, ha="left", fontsize=11, color=p["text"])
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=p["surface"])
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, help="unanimous-rule run dir (other rules regenerated)")
    ap.add_argument("--out", type=Path, default=Path("docs/research/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        runs = {r: run_experiment(r, Path(tmp)) for r in RULES}
        if args.run:
            runs["unanimous"] = args.run

        sweep = series_rho_sweep(runs["unanimous"])
        catch = sweep["series"][2]["y"]
        k = max(range(len(catch)), key=catch.__getitem__)
        sweep["peak"] = {"x": sweep["x"][k], "y": round(catch[k], 4)}
        rules = series_rules(runs)
        struct = series_structured(runs["unanimous"])

    for name, spec, h in (("two-swarms-rho-sweep", sweep, 440),
                          ("two-swarms-verify-rules", rules, 440),
                          ("two-swarms-structured", struct, 400)):
        write_html(spec, args.out / f"{name}.html", h)
        write_png(spec, args.out / f"{name}.png")
        print(args.out / f"{name}.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
