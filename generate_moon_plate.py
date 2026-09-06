#!/usr/bin/env python3
"""Photograph luminance -> dimensioned circular apertures for an etching mask.

The photograph is read as measurement data; the deliverables are vector circles.
No generated imagery, raster embedding, or random point placement is used.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import gzip
import hashlib
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageOps
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parent


def number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "GSFC_20171208_Archive_e001982~orig.jpg")
    parser.add_argument("--diameter-mm", type=float, default=50.0)
    parser.add_argument("--pitch-mm", type=float, default=0.1,
                        help="Horizontal and vertical centre-to-centre spacing.")
    parser.add_argument("--min-hole-mm", type=float, default=0.04)
    parser.add_argument("--max-hole-mm", type=float, default=0.08)
    parser.add_argument("--border-mm", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Aperture area is proportional to (gray / 255)^gamma.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    for name in ["diameter_mm", "pitch_mm", "min_hole_mm", "max_hole_mm", "gamma"]:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if not math.isfinite(args.border_mm) or args.border_mm < 0:
        parser.error("border_mm must be finite and nonnegative")
    if not 0 < args.min_hole_mm <= args.max_hole_mm < args.pitch_mm:
        parser.error("Require 0 < min-hole <= max-hole < pitch")
    if args.pitch_mm > args.diameter_mm:
        parser.error("pitch must not exceed the Moon diameter")
    if math.ceil(args.diameter_mm / args.pitch_mm) > 6000:
        parser.error("This run is limited to 6000 grid positions per side")
    return args


def cell_averages(array, centers, span, axis):
    """Exact box averages of piecewise constant source pixels along one axis."""
    a = np.moveaxis(array, axis, -1)
    integral = np.concatenate([np.zeros(a.shape[:-1] + (1,)), np.cumsum(a, axis=-1)], axis=-1)

    def at(points):
        points = np.clip(points, 0, a.shape[-1])
        indices = np.minimum(np.floor(points).astype(int), a.shape[-1]-1)
        return integral[..., indices] + (points-indices)*a[..., indices]

    left, right = centers+0.5-span/2, centers+0.5+span/2
    result = (at(right)-at(left))/span
    return np.moveaxis(result, -1, axis)


def quantize_areas(values, valid, radii, max_diameter, gamma):
    """Keep small-aperture tone with deterministic serpentine error diffusion."""
    types = [0] + [i for i in range(1, 256) if radii[i] > 0]
    types.sort(key=lambda i: radii[i])
    area = [(2*float(radii[i])/max_diameter)**2 for i in types]
    work = (values/255)**gamma
    levels = np.zeros(values.shape, dtype=np.uint8)
    height, width = values.shape
    for y in range(height):
        direction = 1 if y % 2 == 0 else -1
        cols = range(width) if direction == 1 else range(width-1, -1, -1)
        for x in cols:
            if not valid[y, x]:
                continue
            value = float(work[y, x])
            upper = min(bisect_left(area, value), len(area)-1)
            lower = max(0, upper-1)
            chosen = lower if abs(area[lower]-value) <= abs(area[upper]-value) else upper
            levels[y, x] = types[chosen]
            error = value-area[chosen]
            neighbors = [(y, x+direction, 7), (y+1, x-direction, 3),
                         (y+1, x, 5), (y+1, x+direction, 1)]
            neighbors = [(ny, nx, weight) for ny, nx, weight in neighbors
                         if 0 <= ny < height and 0 <= nx < width and valid[ny, nx]]
            total = sum(weight for _, _, weight in neighbors)
            if total:
                for ny, nx, weight in neighbors:
                    work[ny, nx] += error*weight/total
    return levels


def sample_image(args: argparse.Namespace) -> tuple:
    with Image.open(args.input) as source:
        gray = np.asarray(ImageOps.exif_transpose(source).convert("L"), dtype=np.float64)
    # The supplied photo has a black sky. Ignore JPEG ringing below 12/255.
    ys, xs = np.nonzero(gray > 12)
    if len(xs) == 0:
        raise ValueError("No Moon found against the black background")
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)
    span = max(right-left, bottom-top)
    source_cx = (left + right - 1) / 2
    source_cy = (top + bottom - 1) / 2
    count = math.ceil(args.diameter_mm / args.pitch_mm - 1e-10)
    axis = np.round(args.border_mm + args.diameter_mm/2
                    + (np.arange(count) - (count-1)/2) * args.pitch_mm, 6)
    # Source array indices represent pixel centres. Keep the source orientation.
    source_x = source_cx + (axis-args.border_mm-args.diameter_mm/2)*span/args.diameter_mm
    source_y = source_cy + (axis-args.border_mm-args.diameter_mm/2)*span/args.diameter_mm
    cell_span_px = args.pitch_mm*span/args.diameter_mm
    values = cell_averages(cell_averages(gray, source_x, cell_span_px, 1),
                          source_y, cell_span_px, 0)
    # A radius library permits exact reuse in PDF. Type 1 is the exact minimum.
    # All other nonzero types follow the continuous area-to-tone relation.
    radii = np.round(args.max_hole_mm/2
                     * (np.arange(256, dtype=float)/255)**(args.gamma/2), 6)
    radii[2*radii < args.min_hole_mm] = 0
    radii[1] = round(args.min_hole_mm/2, 6)
    center = args.border_mm + args.diameter_mm/2
    distance = np.hypot(axis[:, None]-center, axis[None, :]-center)
    valid = ((values > 12)
             & (distance + args.max_hole_mm/2 <= args.diameter_mm/2 + 1e-10))
    levels = quantize_areas(values, valid, radii, args.max_hole_mm, args.gamma)
    radii[0] = 0
    if not np.any(levels):
        raise ValueError("No apertures survive the requested size limits")
    source_info = {
        "filename": args.input.name,
        "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "size_px": [gray.shape[1], gray.shape[0]],
        "moon_bbox_px_exclusive": [left, top, right, bottom],
        "moon_reference_diameter_px": span,
        "source_center_px": [source_cx, source_cy],
        "sampling": "source-pixel area average over each grid cell; EXIF orientation applied",
        "background_threshold_8bit": 12,
        "target_normalized_area_sum": float(np.sum((values[valid]/255)**args.gamma)),
        "valid_grid_cell_count": int(valid.sum()),
    }
    return axis, levels, radii, source_info


def geometry_summary(args, axis, levels, radii, source_info) -> dict:
    radius_map = radii[levels]
    counts = np.bincount(levels.ravel(), minlength=256)
    active = counts.copy()
    active[0] = 0
    holes = int(active.sum())
    used_radii = radii[active > 0]
    minimum_gap = math.inf
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        if dx >= 0:
            a = radius_map[:radius_map.shape[0]-dy or None, :radius_map.shape[1]-dx or None]
            b = radius_map[dy:, dx:]
        else:
            a = radius_map[:-1, 1:]
            b = radius_map[1:, :-1]
        valid = (a > 0) & (b > 0)
        if valid.any():
            minimum_gap = min(minimum_gap, float((math.hypot(dx, dy)*args.pitch_mm-a-b)[valid].min()))
    area = float(np.sum(active*math.pi*radii**2))
    return {
        "source": source_info,
        "moon_diameter_mm": args.diameter_mm,
        "artwork_width_mm": args.diameter_mm + 2*args.border_mm,
        "artwork_height_mm": args.diameter_mm + 2*args.border_mm,
        "black_border_mm": args.border_mm,
        "pitch_mm": args.pitch_mm,
        "pitch_definition": "horizontal and vertical centre-to-centre, square grid",
        "diagonal_grid_distance_mm": args.pitch_mm*math.sqrt(2),
        "grid_size": [len(axis), len(axis)],
        "grid_first_center_mm": float(axis[0]),
        "grid_last_center_mm": float(axis[-1]),
        "hole_count": holes,
        "requested_min_hole_diameter_mm": args.min_hole_mm,
        "requested_max_hole_diameter_mm": args.max_hole_mm,
        "actual_min_hole_diameter_mm": float(2*used_radii.min()),
        "actual_max_hole_diameter_mm": float(2*used_radii.max()),
        "guaranteed_edge_gap_mm": args.pitch_mm-args.max_hole_mm,
        "actual_min_neighbor_edge_gap_mm": minimum_gap,
        "total_open_area_mm2": area,
        "open_area_fraction_of_moon_disk": area/(math.pi*(args.diameter_mm/2)**2),
        "polarity": "black = retained/opaque material; white = open aperture",
        "tone": "aperture area proportional to (source grayscale/255)^gamma; no contrast stretch",
        "gamma": args.gamma,
        "too_small_holes": "represented by fewer minimum-size holes using serpentine error diffusion",
        "boundary": "whole circle kept inside nominal Moon disk; no holes in the sky",
        "orientation": "same as source; no mirroring or rotation",
        "coordinate_precision_mm": 0.000001,
        "geometry_checks": {
            "positive_apertures": holes > 0,
            "non_overlapping": minimum_gap > 0,
            "pitch_preserved": bool(np.allclose(np.diff(axis), args.pitch_mm, atol=1e-9, rtol=0)),
        },
        "normalized_area_sum_error": float(np.sum(active*(2*radii/args.max_hole_mm)**2)
                                            - source_info["target_normalized_area_sum"]),
        "hole_count_by_type": {str(i): int(v) for i, v in enumerate(active) if v},
    }


def write_svg(path, size, axis, levels, radii, summary):
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{number(size)}mm" '
                  f'height="{number(size)}mm" viewBox="0 0 {number(size)} {number(size)}">\n')
        out.write('<title>Moon etching mask - white apertures on black retained material</title>\n')
        out.write('<metadata>'+escape(json.dumps(summary, ensure_ascii=False))+'</metadata>\n')
        out.write(f'<rect id="retained-material" x="0" y="0" width="{number(size)}" '
                  f'height="{number(size)}" fill="#000000"/>\n')
        out.write('<g id="apertures" fill="#ffffff" stroke="none">\n')
        radius_text = [number(r) for r in radii]
        axis_text = [number(v) for v in axis]
        for row_index, row in enumerate(levels):
            y = axis_text[row_index]
            out.writelines(f'<circle cx="{axis_text[col]}" cy="{y}" r="{radius_text[row[col]]}"/>\n'
                           for col in np.flatnonzero(row))
        out.write('</g>\n</svg>\n')
    with path.open("rb") as src, gzip.GzipFile(filename=str(path.with_suffix(".svgz")),
                                              mode="wb", mtime=0) as dst:
        while chunk := src.read(1024*1024):
            dst.write(chunk)


def write_pdf(path, size, axis, levels, radii, summary):
    canvas = Canvas(str(path), pagesize=(size*mm, size*mm), pageCompression=1, invariant=1)
    canvas.setTitle(f'Moon - diameter {number(summary["moon_diameter_mm"])} mm - pitch {number(summary["pitch_mm"])} mm')
    canvas.setAuthor("Tsukitou")
    canvas.setSubject("White = apertures; black = retained material. Print at actual size / 100%.")
    # Reusing vector circle forms keeps a million-aperture PDF reasonably small.
    for level in sorted(int(k) for k in summary["hole_count_by_type"]):
        r = float(radii[level])
        canvas.beginForm(f"h{level}", -r, -r, r, r)
        canvas.setFillGray(1)
        canvas.circle(0, 0, r, stroke=0, fill=1)
        canvas.endForm()
    canvas.saveState()
    canvas.scale(mm, mm)
    canvas.translate(0, size)
    canvas.scale(1, -1)
    canvas.setFillGray(0)
    canvas.rect(0, 0, size, size, stroke=0, fill=1)
    prev_x = prev_y = 0.0
    for row_index, row in enumerate(levels):
        y = float(axis[row_index])
        for col in np.flatnonzero(row):
            x = float(axis[col])
            canvas.translate(round(x-prev_x, 6), round(y-prev_y, 6))
            canvas.doForm(f"h{row[col]}")
            prev_x, prev_y = x, y
    canvas.restoreState()
    canvas.showPage()
    canvas.save()


def write_detail_pdf(path, axis, levels, radii, crop):
    """Temporary QA page: exactly the same circles from a 2 mm square crop."""
    x0, y0, size = crop
    c = Canvas(str(path), pagesize=(size*mm, size*mm), pageCompression=1, invariant=1)
    c.scale(mm, mm)
    c.setFillGray(0)
    c.rect(0, 0, size, size, fill=1, stroke=0)
    c.setFillGray(1)
    rmax = float(radii.max())
    for iy in np.flatnonzero((axis >= y0-rmax) & (axis <= y0+size+rmax)):
        for ix in np.flatnonzero((axis >= x0-rmax) & (axis <= x0+size+rmax) & (levels[iy] > 0)):
            r = float(radii[levels[iy, ix]])
            c.circle(float(axis[ix])-x0, size-(float(axis[iy])-y0), r, fill=1, stroke=0)
    c.showPage()
    c.save()


def write_readme(args, summary, stem):
    s = summary
    text = f'''# 月のエッチング原図

指定画像 `{args.input.name}` を円形の白い穴に変換しました。
「こうとう」の原板と同じく、**黒は残す部分、白い円は穴**です。
月面の明るい場所ほど穴が大きくなり、暗い海では小さくなります。
月の外の黒い背景に穴はありません。

## 仕様

|項目|値|
|---|---|
|月の直径|{s['moon_diameter_mm']:g} mm（ユーザー指定）|
|データ外寸|{s['artwork_width_mm']:g} × {s['artwork_height_mm']:g} mm|
|月の周囲の黒い余白|各辺 {s['black_border_mm']:g} mm|
|ドット中心間隔|縦・横とも {s['pitch_mm']:g} mm、正方格子|
|穴の個数|{s['hole_count']:,} 個|
|穴径の設定範囲|{args.min_hole_mm:g} ～ {args.max_hole_mm:g} mm|
|実際の穴径|{s['actual_min_hole_diameter_mm']:.6f} ～ {s['actual_max_hole_diameter_mm']:.6f} mm|
|実際の最小隙間（穴の縁どうし）|{s['actual_min_neighbor_edge_gap_mm']:.6f} mm|
|穴の総面積 / 月の円の面積|{100*s['open_area_fraction_of_moon_disk']:.2f}%|
|向き|元画像と同じ。反転・回転なし|

最小穴径 {args.min_hole_mm:g} mmと中心間隔 {args.pitch_mm:g} mmは、追加指定に合わせています。
間引いた場所には穴がないため、その前後の実際の穴どうしは格子間隔の整数倍などになります。
最大穴径と周囲の余白は今回の設計値です。

## ファイル

- `{stem}.svg`：全ドットが独立した円の実寸ベクトル原図。
- `{stem}.svgz`：同じSVGのgzip圧縮版。
- `{stem}.pdf`：実寸の1ページPDF。用紙自体が原図と同じ外寸です。
- `{stem}_preview.png`：PDFを描画した全体確認画像。加工にはSVG/PDFを使用。
- `{stem}_detail.png`：月面の2 × 2 mm部分を拡大した確認画像。
- `{stem}_spec.json`：寸法、元画像のSHA-256、穴数、計算上の検証結果。
- `{stem}_verification.json`：書き出したSVG/PDFを読み直した寸法・一致検証の結果。

## 印刷・加工時

PDFは「実際のサイズ / 100%」で印刷し、「用紙に合わせる」を無効にしてください。
月の直径は {s['moon_diameter_mm']:g} mm、黒い正方形の一辺は {s['artwork_width_mm']:g} mmです。
これは入稿図形の寸法であり、印刷・転写・エッチング後の穴径の保証ではありません。
最小穴径 {args.min_hole_mm*1000:g} µmの細部が残るか、同じ設定の小片で試作して確認してください。
黒い正方形の外枠は背景の範囲であり、外周切断線ではありません。
転写や投影で必要な反転方向は装置に依存するため、元画像の向きで出力しています。

## 濃淡の変換

元画像の空の余白から月を抽出し、月の直径を指定寸法に対応付けます。
格子の各セルに対応する元画像の画素面積で、グレースケール値を平均します。
8 bit濃淡値を `L`、設定最大穴径を `dmax` とすると、
`穴径 = dmax × (L / 255)^(gamma / 2)` です。標準の `gamma=1` では
**穴の面積が元画像の明るさに比例**します。濃淡の自動引き伸ばしは行いません。
最小穴径に届かない暗部では、誤差拡散で最小径の穴を間引き、局所的な穴面積を保ちます。
穴を最小径に一律に引き上げて暗部を明るくしたり、暗部を一律に消したりしません。
月の外へはみ出す穴は作りません。
全円は白一色、背景は黒一色で、画像の埋め込み・半透明・ぼかしはありません。
背景が残るため、ドット全体を遠目に見た明るさは元写真より低くなります。

## 寸法の変更・再生成

`generate_moon_plate.py` と元画像を同じフォルダに置きます。
Pythonの依存パッケージは `numpy`, `Pillow`, `reportlab` です。

```sh
python3 generate_moon_plate.py --diameter-mm {args.diameter_mm:g} --pitch-mm {args.pitch_mm:g} --min-hole-mm {args.min_hole_mm:g} --max-hole-mm {args.max_hole_mm:g}
```

直径を変える場合は `--diameter-mm` を変更して再生成してください。
SVG/PDFを後から拡大縮小すると、{args.pitch_mm:g} mmの間隔も変わります。
穴径を変える場合も `max-hole-mm < pitch-mm` を満たす設定で再生成します。
`verify_moon_plate.py` でSVG/PDFを読み直して検証できます（追加依存は `pypdf`）。

## 参照した既存資料

- `こうとう/star30_03_0.svg`：黒地と白い円、mm単位のベクトル原図。
- `こうとう/srcOriginal_python/plate_writer.py`：SVG/PDFの色と穴の表現。
- `こうとう/引き継ぎ資料/こうとう引継ぎ28→29.pdf` の5ページ：黒が残る部分、白が穴になる原理。

既存資料の原板径や加工条件は、今回の月の直径・間隔の指定とは別です。
'''
    (args.output_dir / "README.md").write_text(text, encoding="utf-8")


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "tmp" / "pdfs"
    tmp.mkdir(parents=True, exist_ok=True)
    stem = f"moon_{number(args.diameter_mm)}mm_pitch_{number(args.pitch_mm).replace('.', 'p')}mm"
    axis, levels, radii, source_info = sample_image(args)
    summary = geometry_summary(args, axis, levels, radii, source_info)
    print(f"Geometry: {summary['hole_count']:,} apertures; {len(axis)} x {len(axis)} grid", flush=True)
    if not all(summary["geometry_checks"].values()):
        raise ValueError("Geometry checks failed")
    write_svg(args.output_dir / f"{stem}.svg", summary["artwork_width_mm"], axis, levels, radii, summary)
    print("SVG and SVGZ written", flush=True)
    write_pdf(args.output_dir / f"{stem}.pdf", summary["artwork_width_mm"], axis, levels, radii, summary)
    print("PDF written", flush=True)
    # A lunar mare/highland boundary: crop kept in physical mask coordinates.
    crop = (args.border_mm+args.diameter_mm*0.55,
            args.border_mm+args.diameter_mm*0.55, 2.0)
    write_detail_pdf(tmp / f"{stem}_detail.pdf", axis, levels, radii, crop)
    summary["detail_preview_crop_mm"] = {"x": crop[0], "y": crop[1], "width": crop[2], "height": crop[2]}
    (args.output_dir / f"{stem}_spec.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    write_readme(args, summary, stem)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ["hole_count_by_type", "source"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
