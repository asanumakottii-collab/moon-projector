#!/usr/bin/env python3
"""Read back finished SVG/PDF files and verify their physical aperture geometry."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from pypdf import PdfReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", type=Path,
                        default=Path(__file__).resolve().parent / "output/moon_50mm_pitch_0p1mm")
    args = parser.parse_args()
    stem = args.stem
    spec = json.loads(Path(str(stem)+"_spec.json").read_text())
    source = Path(__file__).resolve().parent / spec["source"]["filename"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == spec["source"]["sha256"]
    svg_path = stem.with_suffix(".svg")
    svg_data = svg_path.read_bytes()
    assert gzip.decompress(stem.with_suffix(".svgz").read_bytes()) == svg_data
    ns = "{http://www.w3.org/2000/svg}"
    root = ET.fromstring(svg_data)
    width, height = [float(root.attrib[k].removesuffix("mm")) for k in ["width", "height"]]
    assert width == spec["artwork_width_mm"] and height == spec["artwork_height_mm"]
    assert list(map(float, root.attrib["viewBox"].split())) == [0, 0, width, height]
    assert not root.findall(".//"+ns+"image") and not root.findall(".//"+ns+"filter")
    group = root.find(ns+"g")
    assert group.attrib["fill"] == "#ffffff" and group.attrib["stroke"] == "none"
    circles = list(group)
    assert all(c.tag == ns+"circle" for c in circles)
    geometry = np.array([[float(c.attrib[k]) for k in ["cx", "cy", "r"]] for c in circles])
    assert len(circles) == spec["hole_count"]
    assert np.isfinite(geometry).all()
    pitch, phase = spec["pitch_mm"], spec["grid_first_center_mm"]
    indices_float = (geometry[:, :2]-phase)/pitch
    indices = np.rint(indices_float).astype(int)
    assert np.max(abs(indices_float-indices)) < 1e-7
    assert len(np.unique(indices, axis=0)) == len(indices)
    radii = geometry[:, 2]
    assert radii.min()*2 >= spec["requested_min_hole_diameter_mm"]-1e-9
    assert radii.max()*2 <= spec["requested_max_hole_diameter_mm"]+1e-9
    center = spec["black_border_mm"] + spec["moon_diameter_mm"]/2
    extent = np.hypot(geometry[:, 0]-center, geometry[:, 1]-center) + radii
    assert extent.max() <= spec["moon_diameter_mm"]/2 + 1e-8
    # The diameter ceiling is below the nearest possible lattice distance.
    assert 2*radii.max() < pitch
    actual_area = float(np.sum(np.pi*radii**2))
    assert abs(actual_area-spec["total_open_area_mm2"]) < 1e-7

    reader = PdfReader(stem.with_suffix(".pdf"))
    assert len(reader.pages) == 1
    page = reader.pages[0]
    page_mm = np.array([float(page.mediabox.width), float(page.mediabox.height)])*25.4/72
    assert np.max(abs(page_mm-[width, height])) < 0.0001
    forms = page["/Resources"]["/XObject"]
    radius_by_form = {}
    for key, value in forms.items():
        form = value.get_object()
        assert form["/Subtype"] == "/Form"
        bounds = np.array(list(map(float, form["/BBox"])))
        r = bounds[2]
        assert np.allclose(bounds, [-r, -r, r, r], rtol=0, atol=1e-9)
        commands = form.get_data().decode("ascii")
        assert "1 g" in commands and commands.count(" c\n") == 4
        radius_by_form[key] = r

    stream = page.get_contents().get_data().decode("ascii")
    matrix = re.search(r"q\s+([-\d.]+) 0 0 ([-\d.]+) ([-\d.]+) ([-\d.]+) cm", stream)
    sx, sy, tx, ty = map(float, matrix.groups())
    assert abs(sx-72/25.4) < 1e-6 and abs(sy+72/25.4) < 1e-6 and tx == 0
    assert abs(ty-float(page.mediabox.height)) < 1e-4
    pattern = re.compile(r"1 0 0 1 ([-\d.]+) ([-\d.]+) cm\s+(/FormXob\.h\d+) Do")
    x = y = 0.0
    max_position_error = max_radius_error = 0.0
    pdf_type_counts = Counter()
    pdf_count = 0
    for i, match in enumerate(pattern.finditer(stream)):
        x += float(match[1])
        y += float(match[2])
        r = radius_by_form[match[3]]
        max_position_error = max(max_position_error,
                                 abs(x-geometry[i, 0]), abs(y-geometry[i, 1]))
        max_radius_error = max(max_radius_error, abs(r-geometry[i, 2]))
        pdf_type_counts[match[3].split(".h")[1]] += 1
        pdf_count += 1
    assert pdf_count == len(circles) == stream.count(" Do")
    assert max_position_error < 1e-7 and max_radius_error < 1e-9
    assert dict(pdf_type_counts) == spec["hole_count_by_type"]
    result = {
        "passed": True,
        "source_unchanged": True,
        "svgz_matches_svg": True,
        "svg_pdf_hole_count": pdf_count,
        "svg_physical_size_mm": [width, height],
        "pdf_physical_size_mm": page_mm.tolist(),
        "all_centers_on_requested_grid": True,
        "no_duplicate_centers": True,
        "no_overlapping_holes": True,
        "all_holes_within_moon": True,
        "all_diameters_within_requested_limits": True,
        "max_svg_pdf_center_difference_mm": max_position_error,
        "max_svg_pdf_radius_difference_mm": max_radius_error,
        "pdf_vector_circle_types": len(radius_by_form),
        "pdf_contains_only_vector_forms": True,
    }
    Path(str(stem)+"_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
