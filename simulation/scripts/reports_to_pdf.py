"""Render the Markdown gate reports as PDFs.

Markdown -> self-contained HTML (Japanese-capable serif/sans stack, print CSS)
-> PDF via headless Chrome's --print-to-pdf.

Produces one PDF per report plus a combined document with a title page, a table
of contents and the figures embedded.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser",
]

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
@page { @bottom-center { content: counter(page); } }
:root {
  --ink: #1a1a1a; --muted: #5a5f66; --rule: #d6dade; --accent: #1f4e79;
  --code-bg: #f4f6f8; --head-bg: #eef2f6;
}
* { box-sizing: border-box; }
body {
  font-family: "Yu Gothic", "Meiryo", "Hiragino Kaku Gothic ProN",
               "Noto Sans CJK JP", "Helvetica Neue", Arial, sans-serif;
  font-size: 9.6pt; line-height: 1.65; color: var(--ink);
  margin: 0; -webkit-font-smoothing: antialiased;
  font-feature-settings: "palt" 1;
}
h1, h2, h3, h4 { line-height: 1.35; font-weight: 700; color: var(--ink); }
h1 { font-size: 17pt; margin: 0 0 2mm; letter-spacing: .01em; }
h2 { font-size: 12.5pt; margin: 8mm 0 2.5mm; padding-bottom: 1.2mm;
     border-bottom: 1.2px solid var(--accent); }
h3 { font-size: 10.6pt; margin: 5.5mm 0 1.8mm; color: var(--accent); }
h4 { font-size: 9.8pt; margin: 4mm 0 1.5mm; color: var(--muted); }
p { margin: 0 0 2.6mm; }
ul, ol { margin: 0 0 2.8mm; padding-left: 5.5mm; }
li { margin: 0 0 1.1mm; }
li > ul, li > ol { margin-top: 1.1mm; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 6mm 0; }
code, kbd {
  font-family: "Consolas", "SFMono-Regular", "Menlo", monospace;
  font-size: 0.87em; background: var(--code-bg); color: #0f3d5c;
  padding: 0.5px 2.5px; border-radius: 2.5px;
  word-break: break-word;
}
pre {
  background: var(--code-bg); border: 1px solid var(--rule); border-radius: 3px;
  padding: 2.6mm 3mm; overflow-x: auto; margin: 0 0 3.2mm;
  font-size: 8.2pt; line-height: 1.45;
}
pre code { background: none; padding: 0; color: var(--ink); }
table {
  border-collapse: collapse; width: 100%; margin: 1.5mm 0 4mm;
  font-size: 8.3pt; line-height: 1.4;
}
th, td {
  border: 0; border-bottom: 0.6px solid var(--rule);
  padding: 1.15mm 1.8mm; text-align: left; vertical-align: top;
}
thead { display: table-header-group; }   /* repeat the header when a table breaks */
thead th {
  background: var(--head-bg); border-bottom: 1.1px solid var(--accent);
  font-weight: 700; white-space: nowrap;
}
tr { break-inside: avoid; }
tbody tr:last-child td { border-bottom: 1.1px solid var(--accent); }
td code, th code { font-size: 0.9em; background: none; padding: 0; }
strong { font-weight: 700; }
blockquote {
  margin: 0 0 3mm; padding: 1.5mm 0 1.5mm 3.5mm;
  border-left: 2.5px solid var(--rule); color: var(--muted);
}
pre, figure { break-inside: avoid; }
h2, h3 { break-after: avoid; }
figure { margin: 4mm 0 5.5mm; }
figure img { max-width: 100%; height: auto; display: block; }
figcaption { font-size: 8.2pt; color: var(--muted); margin-top: 1.8mm;
             text-align: left; max-width: 150mm; }
.docmeta { color: var(--muted); font-size: 8.6pt; margin: 0 0 5mm;
           padding-bottom: 2mm; border-bottom: 0.6px solid var(--rule); }
.titlepage { text-align: left; padding-top: 38mm; }
.titlepage .big { font-size: 24pt; font-weight: 700; line-height: 1.3;
                  margin-bottom: 4mm; }
.titlepage .sub { font-size: 12pt; color: var(--muted); margin-bottom: 14mm; }
.titlepage dl { font-size: 9.5pt; }
.titlepage dt { font-weight: 700; color: var(--accent); margin-top: 3mm; }
.titlepage dd { margin: 0.6mm 0 0 0; }
.pagebreak { break-after: page; }
.toc ol { list-style: none; padding-left: 0; }
.toc > ol > li { margin-bottom: 1.6mm; font-weight: 700; }
.toc ol ol { padding-left: 5mm; font-weight: 400; margin-top: 1mm; }
.callout {
  background: #fdf6e7; border: 1px solid #e8d9ae; border-radius: 3px;
  padding: 2.6mm 3.2mm; margin: 0 0 4mm; font-size: 9.2pt;
}
"""

MD_EXT = ["tables", "fenced_code", "sane_lists", "attr_list", "md_in_html"]


def md_to_html(text: str) -> str:
    return markdown.markdown(text, extensions=MD_EXT, output_format="html5")


def page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang='ja'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
        w = shutil.which(c)
        if w:
            return w
    sys.exit("no Chrome/Chromium found for --print-to-pdf")


def to_pdf(chrome: str, html_text: str, out_pdf: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "doc.html"
        src.write_text(html_text, encoding="utf-8")
        cmd = [
            chrome, "--headless", "--disable-gpu", "--no-sandbox",
            # a throwaway profile: without it Chrome silently refuses to print
            # when the user has a normal Chrome window open (the default
            # profile directory is locked) and exits 0 with no output
            f"--user-data-dir={Path(td) / 'profile'}",
            "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=10000",
            f"--print-to-pdf={out_pdf}", src.as_uri(),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if not out_pdf.exists():
            sys.exit(f"chrome failed for {out_pdf}:\n{r.stdout}\n{r.stderr}")


def data_uri(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


TEXT_WIDTH_MM = 178.0          # A4 minus the 16 mm side margins
FIG_DPI = 300                  # figures are saved at 300 dpi
FIG_MAGNIFY = 1.35             # slightly larger than journal size, still legible


def figure_block(fig_dir: Path, fig_id: str, caption: str) -> str:
    """Size each figure by its true physical width so that one-column and
    two-column figures keep the same font size on the page."""
    png = fig_dir / f"{fig_id}.png"
    if not png.exists():
        return ""
    try:
        from PIL import Image
        with Image.open(png) as im:
            width_mm = im.width / FIG_DPI * 25.4 * FIG_MAGNIFY
        style = f" style='width: min({width_mm:.1f}mm, {TEXT_WIDTH_MM}mm)'"
    except Exception:                                   # noqa: BLE001
        style = ""
    return (f"<figure><img src='{data_uri(png)}' alt='{fig_id}'{style}>"
            f"<figcaption><b>{fig_id}</b> — {caption}</figcaption></figure>")


REPORTS = [
    ("00_READ_ME_FIRST.md", "00 — 最初に読む", []),
    ("Z0_final_report.md", "Z0 — 統合報告", []),
    ("G0_validation.md", "G0 — 内部整合性テスト", []),
    ("G3_milp_budget.md", "G3 — MILP ベンチマークと計算設計", []),
    ("G1_muH_peak.md", "G1 — 主結果", [
        ("R4_fig4", "Fig.4。A: 進化後 μ^H vs r（散布・平均・ブートストラップ 95% CI・"
                    "μ^H_0 の水平線）。B: 進化後 μ^S（生値）。C: r∈{5,10,20,40} の "
                    "μ^S–μ^H 最高密度領域（25/50/75%、bw=0.2）。"),
        ("R4_fig4A_detlimit", "平均 μ^H の detlimit 込み／除き。"),
        ("R4_post_structure", "進化後の構造量（改訂版）。A: 提示代謝物数と b（両規約）"
                              "＋進化前の b。B: 排出代謝物数。C: ub。D: 中央値 s。"
                              "zero_cH_rule の効果は N1 §2 を参照。"),
        ("R4_fig5A_max", "Fig.5A（max 版、5≤r≤25）。破線は μ^H_0。"),
        ("R4_figS2A", "Fig.S2A。進化前の平均 ub vs 進化後 μ^H。"),
        ("R4_figS1", "Fig.S1。進化前に寄生的だったペアに限定（n = 2,585 / 4,500）。"),
        ("R4_mutualism_fraction", "相利（μ^H > μ^H_0 かつ μ^S > θ）の成立割合。"),
    ]),
    ("G5_conventions_and_correlations.md", "G5 — Phase A：規約・相関・相利割合", [
        ("A3a_fig4B_raw_vs_convention", "A3-a。生値（C0_raw）と3規約すべてを適用した版"
                                        "（C123_all）の平均曲線とブートストラップ 95% CI。"
                                        "A: μ^S。B: μ^H（規約による差は厳密に 0）。"),
        ("A3e_mutualism_fraction", "A3-e。A: 相利成立割合（μ^H > μ^H_0 + ε かつ μ^S > θ）。"
                                   "太線は既定 (ε, θ) = (0, 10⁻⁶) で 95% CI つき、"
                                   "細線は他の5通り。B: 零過剰の実態 — 宿主が何も得ない"
                                   "（μ^H = μ^H_0 ちょうど）割合と、共生者が何も得ない割合。"),
    ]),
    ("G6_diagnostics_and_host_sampling.md", "G6 — Phase B：診断・Jacobian・宿主受容率", [
        ("A2_D5_jacobian", "A2-D5。A: 数値 ‖J_ub‖_F と解析 s の一致（δ/|c^H|₁ = 10⁻⁴）。"
                           "B: ‖J_ub − S‖_F の中央値（δ ごと）。C: 摂動で活性集合が"
                           "切り替わった cell の割合。"),
        ("A2_D6_host_acceptance", "A2-D6。A: ランダム生成した宿主の μ^H_0 の分布"
                                  "（6,000 試行、対数軸）。0 と 1 に集中し、採択窓は"
                                  "その間の薄い平坦部にある。B: 採択された 91 宿主の"
                                  "窓内での分布。破線は第1弾の宿主。"),
        ("B23_detlimit_interval", "2.3。detlimit の incumbent（下限）と best bound（上限）"
                                  "で挟んだ平均 μ^H。区間の両端でピーク位置は r = 12 のまま。"),
    ]),
    ("N1_solver_and_convention_notes.md", "N1 — ソルバと規約の追加検討", [
        ("R4_post_structure", "改訂版。A: 提示代謝物数（＝代替規約の b）と既定規約の b、"
                              "および進化前の b。B: 排出代謝物数（規約に依らない）。"
                              "C: ub（両規約）。D: 中央値 s（両規約）。"),
    ]),
    ("G4_R1_R2_R3.md", "G4 — R1 / R2 / R3（進化前）", [
        ("R1_fig3", "Fig.3。A: μ^S vs r。B: μ^H vs r。C: r∈{5,10,20,40} の "
                    "μ^S–Δμ^H 最高密度領域（bw=2×10⁻³）。"),
        ("R2_fig6AB_S2B", "Fig.6A/6B/S2B。A: 感度 s vs r（中央値・平均・トリム平均と "
                          "10–90 / 25–75 パーセンタイル帯）。B: 平均 b と平均排出本数。"
                          "C: ub vs r、破線は最大値 b(m−1−b)。"),
        ("R2_b_axis", "Fig.6A と関連量を b 軸で再プロット。破線は (m−1)/2。"),
        ("R3_fig5DE", "Fig.5D/5E（5≤r≤25）。各点は 1 つの R^S。"),
    ]),
    ("G2a_fixed_repertoire_lp.md", "G2a — A1-LP：固定レパートリー対照（速報）", [
        ("A1_repertoire", "A1。実効レパートリー（R^S の列に現れるユニーク反応数）vs r。"
                          "paper では r とともに増えるが、fixed_repertoire では定数。"),
        ("A1_mechanism_r", "A1。機構量の r 依存をスキーム別に重ねたもの。"
                           "A: 中央値 s。B: 排出本数。C: ub。D: b。"),
        ("A1_mechanism_b", "A1。同じ量を b 軸で再プロット。破線は (m−1)/2 = 4.5。"),
    ]),
    ("G2b_fixed_repertoire_milp.md", "G2b — A1-MILP：固定レパートリー対照（主判定）", [
        ("A1_muH_verdict", "A1-MILP。A: 進化後 μ^H の平均と 95% CI（両アーム、破線は μ^H_0）。"
                           "B: 相利成立割合。C: 宿主が何も得られない割合。"
                           "D: 同じ量を b 軸で（b ≥ 4 のみ）。三角はピーク、横棒はその CI。"),
    ]),
    ("G8_r5_probe.md", "G8 — R5 受容率プローブ（2 設計）", [
        ("R5_probe_acceptance", "R5 §2。A: 宿主受容率 vs m（両設計、Wilson 95% CI）。"
                                "B: μ^H_0 が厳密に 0 / 1 になる割合。"
                                "C: 採択窓の下・上に外れる割合。"),
    ]),
    ("G7_multi_host.md", "G7 — A4：複数宿主", [
        ("A4_mu_H_by_host", "A4。平均 μ^H vs r。細線 = 宿主 1 個体（20 本）、"
                            "太線 = 宿主間平均。生の値。"),
        ("A4_mu_H_by_host_rel_mu0", "同じ曲線を各宿主の μ^H_0 で割った版。"),
        ("A4_mu_H_by_host_rel_max", "同じ曲線を各宿主の最大値で割った版（形の比較）。"),
        ("A4_rstar_vs_mu0", "宿主ごとのピーク位置 r* を μ^H_0 に対して回帰したもの。"),
    ]),
    ("G9_m_sweep.md", "G9 — R5：m スイープ（2 設計）", [
        ("R5_rstar_vs_m", "R5 §4。ピーク位置 r* vs m。点 = 宿主 1 個体、線 = 平均、"
                          "破線 = r = m−1 の参照。青 = R5-A（r^H=2.4m）、橙 = R5-B（r^H=24）。"),
        ("R5_rstar_vs_m_dense", "同じ図を R5-B の m 1 刻み（15 点・宿主 150 個体）で描いたもの。"),
        ("R5_b_vs_m", "b 軸版。r* における平均 b vs m。破線は (m−1)/2 の参照。"),
        ("R5_muH_curves", "m ごとの平均 μ^H vs r。縦破線は r = m−1。"),
        ("R5_probe_acceptance", "G8 の受容率プローブ（再掲）。"),
    ]),
    ("Z1_for_the_author.md", "Z1 — 著者が決める必要がある事項", []),
]


# gates that may not exist yet, and what they are.  Anything in this map whose
# file is absent is listed as 未実施 on the title page, so the cover never
# claims more than the directory actually contains.
NOT_YET = {
    "G2a_fixed_repertoire_lp.md": "G2a（A1-LP 固定レパートリー対照・速報）",
    "G2b_fixed_repertoire_milp.md": "G2b（A1-MILP 固定レパートリー対照・主判定）",
    "G7_multi_host.md": "G7（A4 複数宿主）",
}


def today_version() -> str:
    return "v" + dt.datetime.now().strftime("%Y%m%d-%H%M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None,
                    help="report version directory under reports/, "
                         "e.g. v20260917-1647.  Defaults to the newest one.")
    ap.add_argument("--reports", default=None)
    ap.add_argument("--figures", default=str(ROOT / "figures"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = Path(a.reports) if a.reports else ROOT / "reports"
    if a.reports is None:
        vers = sorted(d for d in base.glob("v*") if d.is_dir())
        if a.version:
            rep_dir = base / a.version
            if not rep_dir.is_dir():
                sys.exit(f"no such version: {rep_dir}")
        elif vers:
            rep_dir = vers[-1]
            print("version (newest):", rep_dir.name)
        else:
            rep_dir = base
    else:
        rep_dir = base
    fig_dir = Path(a.figures)
    out_dir = Path(a.out) if a.out else rep_dir      # PDFs land beside the .md
    out_dir.mkdir(parents=True, exist_ok=True)
    version = rep_dir.name if rep_dir.name.startswith("v") else today_version()
    chrome = find_chrome()
    print("chrome:", chrome)
    today = dt.date.today().isoformat()

    combined_parts = []
    toc = []

    for fname, short, figs in REPORTS:
        src = rep_dir / fname
        if not src.exists():
            print("skip (missing):", fname)
            continue
        text = src.read_text(encoding="utf-8")
        body = md_to_html(text)
        figs_html = ""
        if figs:
            figs_html = "<h2>図</h2>" + "".join(
                figure_block(fig_dir, fid, cap) for fid, cap in figs)

        # standalone PDF
        standalone = page(short, f"<article>{body}{figs_html}</article>")
        out_pdf = out_dir / (Path(fname).stem + ".pdf")
        to_pdf(chrome, standalone, out_pdf)
        print("wrote", out_pdf)

        anchor = Path(fname).stem
        combined_parts.append(
            f"<section id='{anchor}' class='pagebreak'>{body}{figs_html}</section>")
        heads = re.findall(r"^##\s+(.+)$", text, flags=re.M)
        toc.append((short, anchor, heads))

    # combined document
    toc_html = ["<h1>目次</h1><nav class='toc'><ol>"]
    for short, anchor, heads in toc:
        toc_html.append(f"<li>{html.escape(short)}<ol>")
        for h in heads:
            toc_html.append(f"<li>{md_to_html(h)[3:-4]}</li>")
        toc_html.append("</ol></li>")
    toc_html.append("</ol></nav>")

    included = "／".join(short for short, _, _ in toc)
    missing = [lab for fn, lab in NOT_YET.items() if not (rep_dir / fn).exists()]
    pending = "、".join(missing) if missing else "なし"
    title = f"""
<div class='titlepage pagebreak'>
  <div class='big'>GRHD — 独立再実装 レポート</div>
  <div class='sub'>Genome reduction / endosymbiosis の三層最適化モデル<br>
  ブラインド再実装</div>
  <div class='callout'>本レポートは <code>instructions/01_model_spec.md</code> のみを一次情報源とした
  独立再実装の結果である。<code>ref/</code> と <code>review/</code> は参照していない。
  原稿との一致・不一致の解釈は行っていない。</div>
  <dl>
    <dt>日付</dt><dd>{today}</dd>
    <dt>版</dt><dd><code>{version}</code></dd>
    <dt>収録</dt><dd>{included}</dd>
    <dt>計算環境</dt>
    <dd>furusawalab-cell2 / cell3 / cell6、Python 3.11.16、
        HiGHS 1.15.1（LP）、quadprog 0.1.13（min-norm QP）、
        CPLEX 22.2.0.0 Community Edition（Level 3 MILP）</dd>
    <dt>規模</dt>
    <dd>進化前 3,000,000 cell（2.1 コア時間）／進化後 4,500 MILP（13.4 コア時間）</dd>
    <dt>未実施</dt><dd>{pending}</dd>
  </dl>
</div>
<div class='pagebreak'>{''.join(toc_html)}</div>
"""
    combined = page("GRHD 独立再実装レポート", title + "".join(combined_parts))
    out_pdf = out_dir / f"GRHD_reproduction_report_{version}.pdf"
    to_pdf(chrome, combined, out_pdf)
    print("wrote", out_pdf)


if __name__ == "__main__":
    main()
