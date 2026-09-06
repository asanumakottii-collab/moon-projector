#!/usr/bin/env python3
"""Create coarser fixed-diameter Moon dots and an A4 print comparison."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas

from generate_moon_plate import (
    ROOT, geometry_summary, number, quantize_areas, sample_image,
    write_detail_pdf, write_pdf, write_svg,
)

OUT = ROOT / "output" / "density"
TMP = ROOT / "tmp" / "pdfs"


def make_pattern(label, dot_mm, pitch_mm):
    args = SimpleNamespace(
        input=ROOT / "GSFC_20171208_Archive_e001982~orig.jpg",
        diameter_mm=50., pitch_mm=pitch_mm,
        min_hole_mm=dot_mm, max_hole_mm=dot_mm,
        border_mm=2., gamma=1., output_dir=OUT,
    )
    axis, levels, radii, source = sample_image(args)
    # Merge the equivalent minimum/maximum radius types into a single dot type.
    levels[levels > 0] = 1
    radii[:] = 0
    radii[1] = dot_mm/2
    summary = geometry_summary(args, axis, levels, radii, source)
    summary["dot_diameter_mm"] = dot_mm
    summary["border_mm"] = summary.pop("black_border_mm")
    summary["dot_color"] = "#000000"
    summary["background_color"] = "#ffffff"
    summary["tone"] = "Fixed dot diameter; local occupied-grid fraction follows source grayscale/255"
    summary["polarity"] = "Black dots on white background, matching the previous inverted version"
    summary["density_method"] = "Deterministic serpentine error diffusion; binary occupied/empty grid"
    summary.pop("too_small_holes")
    summary["occupied_grid_fraction"] = summary["hole_count"]/source["valid_grid_cell_count"]
    summary["previous_grid_pitch_mm"] = 0.1
    summary["previous_max_actual_diameter_mm"] = 0.076798
    stem = f"moon_50mm_fixed_{str(dot_mm).replace('.', 'p')}mm_pitch_{str(pitch_mm).replace('.', 'p')}mm"
    write_svg(OUT / (stem+".svg"), 54., axis, levels, radii, summary, dot_gray=0)
    # The extra-coarse standalone PDF is a rendering intermediate; its artwork
    # is available as SVG and at actual size in the A4 comparison PDF.
    pdf_dir = OUT if label == "STANDARD" else TMP
    pdf = pdf_dir / (stem+".pdf")
    write_pdf(pdf, 54., axis, levels, radii, summary, dot_gray=0)
    crop = (22., 22., 4.)
    write_detail_pdf(TMP / (stem+"_detail.pdf"), axis, levels, radii, crop, dot_gray=0)
    summary["detail_preview_crop_mm"] = {"x": crop[0], "y": crop[1], "width": 4., "height": 4.}
    (OUT / (stem+"_spec.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    points = [(float(axis[ix]), float(axis[iy])) for iy, ix in np.argwhere(levels > 0)]
    assert set(np.unique(radii[levels[levels > 0]])) == {dot_mm/2}
    print(json.dumps({"label": label, "stem": stem, "dots": len(points),
                      "diameter_mm": dot_mm, "pitch_mm": pitch_mm,
                      "minimum_edge_gap_mm": summary["actual_min_neighbor_edge_gap_mm"],
                      "density_sum_error": summary["normalized_area_sum_error"]}), flush=True)
    return dict(label=label, stem=stem, points=points, summary=summary, args=args, pdf=pdf)


def make_print_sheet(patterns):
    c = Canvas(str(OUT / "moon_density_print_test_A4.pdf"),
               pagesize=(210*mm, 297*mm), pageCompression=1, invariant=1)
    c.setTitle("Moon - fixed diameter dots - A4 print test at 100%")
    c.setAuthor("Tsukitou")
    c.scale(mm, mm)

    def text(x, top, value, size=10, bold=False):
        c.setFillGray(0)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size/mm)
        c.drawString(x, 297-top, value)

    def line(x1, y1, x2, y2, gray=0, width=.2):
        c.setStrokeGray(gray)
        c.setLineWidth(width)
        c.line(x1, 297-y1, x2, 297-y2)

    text(24, 21, "MOON / UNIFORM DOTS", 18, True)
    text(24, 29, "Actual Moon diameter: 50 mm   |   Print at actual size / 100%", 10)
    line(24, 35, 186, 35, .7)
    for index, (p, left) in enumerate(zip(patterns, [27., 129.]), 1):
        d, pitch = p["args"].min_hole_mm, p["args"].pitch_mm
        text(left, 46, f"0{index} / {p['label']}", 11, True)
        text(left, 53, f"Dot {d:.2f} mm / Grid {pitch:.2f} mm", 9)
        c.setFillGray(0)
        for x, y in p["points"]:
            c.circle(left+x, 297-61-y, d/2, stroke=0, fill=1)
        text(left, 122, f"Clear gap: {pitch-d:.2f} mm minimum", 9)
        text(left, 128, f"{len(p['points']):,} equal-size dots", 9)
        text(left, 145, "GRID OCCUPANCY", 9, True)
        lib = np.zeros(256)
        lib[1] = d/2
        for i, fraction in enumerate([.2, .4, .6, .8, 1.]):
            w, h = 10., 8.
            nx, ny = int(w/pitch), int(h/pitch)
            present = quantize_areas(np.full((ny, nx), fraction*255),
                                     np.ones((ny, nx), dtype=bool), lib, d, 1.)
            xx = (w-(nx-1)*pitch)/2 + np.arange(nx)*pitch
            yy = (h-(ny-1)*pitch)/2 + np.arange(ny)*pitch
            for row, col in np.argwhere(present > 0):
                c.circle(left+i*11+xx[col], 297-150-yy[row], d/2, stroke=0, fill=1)
            text(left+i*11, 164, f"{fraction:.0%}", 8)
    text(24, 178, "Dots should stay round and separate, including in the densest patches.", 10)
    line(24, 187, 186, 187, .7)
    text(27, 200, "50 mm CALIBRATION", 10, True)
    line(27, 209, 77, 209, 0, .2)
    line(27, 207, 27, 211, 0, .2)
    line(77, 207, 77, 211, 0, .2)
    text(93, 205, "Measure between the two end ticks.", 9)
    text(93, 212, "The distance must be 50 mm.", 9)
    text(24, 236, "PRINTING", 10, True)
    for y, value in [(244, "Use actual size (100%). Disable Fit to page / Shrink to fit."),
                     (251, "Each Moon is a separate pattern. Each pattern uses one dot diameter."),
                     (258, "Black dots on white background. Density alone carries the lunar detail.")]:
        text(24, y, value, 9)
    c.showPage()
    c.save()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    patterns = [make_pattern("STANDARD", .15, .25),
                make_pattern("EXTRA COARSE", .20, .35)]
    make_print_sheet(patterns)
    normal, coarse = patterns
    (OUT / "README.md").write_text(f'''# 月の原図：一定径ドット・密度表現版

ドットの大きさを一定にし、ドットを配置する数だけで月の模様を表現しました。
月径50 mm、白い背景に黒いドットです。前回の白黒反転版と同じ濃淡方向です。
元写真で明るい場所ほどドットが密になり、暗い海では疎になります。

|設定|標準版（STANDARD）|さらに粗い版（EXTRA COARSE）|
|---|---|---|
|月の直径|50 mm|50 mm|
|すべてのドットの直径|0.15 mm|0.20 mm|
|縦横の格子中心間隔|0.25 mm|0.35 mm|
|隣接ドットの縁どうしの最小隙間|0.10 mm|0.15 mm|
|ドット数|{len(normal['points']):,}|{len(coarse['points']):,}|
|データ外寸|54 × 54 mm|54 × 54 mm|

中心間隔はドットを配置できる格子の間隔です。間引いた場所では実際のドット間は広くなります。
ドットの径は各設定内で完全に同じです。半透明・濃い灰色・薄い灰色は使っていません。
密度は元画像のグレースケールに対応し、誤差拡散でドットの有無に変換しています。
元画像をセルの面積で平均してから変換し、月の外側にはドットを配置していません。

## 印刷

`moon_density_print_test_A4.pdf` はA4サイズです。
左右に2種類の月を直径50 mmで並べ、下に密度見本と50 mmの確認線を付けています。
「実際のサイズ / 100%」で印刷し、「用紙に合わせる」「自動縮小」を無効にしてください。
確認線の端の目盛り間が50 mmになり、密な部分でも点が分離している方を使用してください。
格子占有率100%の見本も白い隙間が残る設計です。100%は紙面を黒く塗りつぶす意味ではありません。
実際の印刷の再現性はプリンター・用紙・印刷設定によるため、比較用紙で確認してください。

## データ

- `{normal['stem']}.svg` / `.svgz`：標準版の実寸ベクトル原図。
- `{normal['stem']}.pdf`：標準版だけの実寸PDF（用紙外寸54 × 54 mm）。
- `{coarse['stem']}.svg` / `.svgz`：さらに粗い版の実寸ベクトル原図。
- `moon_density_print_test_A4.pdf`：2種類を並べた実寸の印刷比較用紙。
- `*_preview.png`：全体確認画像。`*_detail.png`：4 × 4 mm範囲の拡大確認画像。
- `*_spec.json` / `*_verification.json`：寸法、配置方法、検証結果。

元の細かい原図は別フォルダに残しています。
再生成にはプロジェクト直下の `generate_moon_density.py` を実行します。
寸法を変えるときはコードの設定値を変えて再生成してください。
印刷画面で拡大縮小すると、ドットの直径・間隔も変わります。
''', encoding="utf-8")
    print("A4 comparison and README written", flush=True)


if __name__ == "__main__":
    main()
