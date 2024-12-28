import os
import csv
import re
from lxml import etree
from svgpathtools import svg2paths2, Path, parse_path
from svgpathtools import CubicBezier, Line
# from svgpathtools import Transform
from tqdm import tqdm

#############################
# Helper functions
#############################

def apply_transform_to_path(path_obj, transform_str):
    """
    Manually parse scale(...) and translate(...).
    Extend it for rotate, skew if you need them.
    """
    import re
    from svgpathtools import Path
    
    translate_pattern = r'translate\(\s*([-0-9.]+)\s*(?:,\s*([-0-9.]+))?\s*\)'
    scale_pattern     = r'scale\(\s*([-0-9.]+)\s*(?:,\s*([-0-9.]+))?\s*\)'
    
    tx, ty = 0.0, 0.0
    match_t = re.search(translate_pattern, transform_str)
    if match_t:
        tx = float(match_t.group(1))
        if match_t.group(2):
            ty = float(match_t.group(2))
    
    sx, sy = 1.0, 1.0
    match_s = re.search(scale_pattern, transform_str)
    if match_s:
        sx = float(match_s.group(1))
        if match_s.group(2):
            sy = float(match_s.group(2))

    # Transform each segment’s start/end (and control points if cubic)
    new_segments = []
    for seg in path_obj:
        seg.start = complex(seg.start.real * sx + tx, seg.start.imag * sy + ty)
        seg.end   = complex(seg.end.real   * sx + tx, seg.end.imag   * sy + ty)
        if hasattr(seg, 'control1'):
            seg.control1 = complex(seg.control1.real * sx + tx,
                                   seg.control1.imag * sy + ty)
        if hasattr(seg, 'control2'):
            seg.control2 = complex(seg.control2.real * sx + tx,
                                   seg.control2.imag * sy + ty)
        new_segments.append(seg)

    return Path(*new_segments)


def flatten_path(path_obj, num_samples=1):
    """
    Given a Path object composed of Bézier segments,
    sample each segment so that we get a series of points.
    Returns a list of complex points (x+iy).
    """
    points = []
    for seg in path_obj:
        length = seg.length()
        # We sample each segment at num_samples steps
        for i in range(num_samples + 1):
            # Parameter t goes from 0.0 to 1.0
            t = i / num_samples
            pt = seg.point(t)
            points.append(pt)
    return points

def path_to_stroke_points(path_obj, pen_down_value=1, samples_per_curve=1):
    """
    Convert an svgpathtools Path object into a list of [x, y, penDown, endStroke].
    We do so by flattening the path (sampling it) into discrete points.
    The first point in each path will be penDown=0 (like 'M' command, pen-lifted),
    subsequent points penDown=1, except that the last point of the path will
    have endStroke=1.
    """
    # Flatten
    sampled_pts = flatten_path(path_obj, num_samples=samples_per_curve)
    stroke_array = []

    if len(sampled_pts) == 0:
        return stroke_array

    # The first sample is "move-to" => penDown=0, endStroke=0
    x0, y0 = sampled_pts[0].real, sampled_pts[0].imag
    stroke_array.append([x0, y0, 0, 0])  # M command

    # The rest are penDown=1 except the very last is also endStroke=1
    for i, pt in enumerate(sampled_pts[1:], start=1):
        x, y = pt.real, pt.imag
        is_last = (i == len(sampled_pts) - 1)
        stroke_array.append([x, y, pen_down_value, int(is_last)])

    return stroke_array

def extract_all_strokes_from_svg(svg_file):
    """
    Parse an SVG file with potential group transforms.
    For each <path> in order, produce a series of [x,y,penDown,endStroke] points.
    Then return a single combined list for the entire file.
    """
    # svg2paths2 returns top-level paths and their attributes, 
    # but doesn't automatically apply group transforms. We'll parse
    # with lxml to handle <g> transforms properly if needed.
    # If your files have deeply nested groups, you can walk the tree more thoroughly.
    
    doc = etree.parse(svg_file)
    root = doc.getroot()
    nsmap = root.nsmap.copy()
    if None in nsmap:
        # Sometimes the default namespace is None, let's fix that
        nsmap['svg'] = nsmap.pop(None)

    all_strokes = []

    # Recursively find all <g> elements and track cumulative transforms
    # Then find <path> children in those <g> with the correct transform stack.
    def traverse(node, parent_transform=""):
        # Merge parent's transform with this node's transform
        node_transform = node.get("transform") or ""
        combined_transform = (parent_transform + " " + node_transform).strip()

        # ❗ Use 'svg:path' instead of './/svg:path'
        #    to avoid picking up paths in nested groups more than once.
        for path_el in node.findall('svg:path', nsmap):
            d_str = path_el.get('d')
            if not d_str:
                continue
            
            path_obj = parse_path(d_str)
            if combined_transform:
                path_obj = apply_transform_to_path(path_obj, combined_transform)

            stroke_points = path_to_stroke_points(path_obj)
            all_strokes.extend(stroke_points)

        # Now recurse only into direct child groups
        for g_el in node.findall('svg:g', nsmap):
            traverse(g_el, combined_transform)

    traverse(root)
    return all_strokes


#############################
# Main driver
#############################

def main(data_dir, output_csv):
    """
    data_dir: directory containing subfolders named by class labels 
              (e.g. 'apple', 'ant', etc.).
              Each subfolder has .svg files of that class.
    output_csv: path to the CSV file to create.
    """
    # Count total files for the progress bar
    total_files = sum(
        len(files)
        for _, _, files in os.walk(data_dir)
        if any(file.endswith('.svg') for file in files)
    )

    # Open CSV in write mode
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Create a progress bar
        with tqdm(total=total_files, desc="Processing SVG files", unit="file") as pbar:
            # Walk through each subfolder in data_dir
            for class_name in os.listdir(data_dir):
                class_path = os.path.join(data_dir, class_name)
                if not os.path.isdir(class_path):
                    continue  # skip files at top level, only use subdirectories as classes

                # For each SVG file in that subfolder
                for filename in os.listdir(class_path):
                    if not filename.lower().endswith('.svg'):
                        continue
                    svg_file = os.path.join(class_path, filename)
                    try:
                        # Extract strokes for the entire drawing
                        strokes = extract_all_strokes_from_svg(svg_file)
                        # Convert to string form
                        # e.g. "[[0,5,0,0],[27,144,1,0], ...]"
                        strokes_str = str(strokes)

                        # Write one CSV row: <className>, "<strokes_str>"
                        # We put quotes around the strokes_str to keep it intact
                        writer.writerow([class_name, strokes_str])
                    except Exception as e:
                        print(f"Failed parsing {svg_file}: {e}")

                    # Update progress bar
                    pbar.update(1)

    print(f"Done! Wrote CSV to {output_csv}")


#############################
# If you just want to run it
#############################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert SVG sketches into CSV stroke format.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Directory with subfolders, each subfolder is a class name containing .svg files.")
    parser.add_argument('--output_csv', type=str, default="output.csv",
                        help="Output CSV file path.")
    args = parser.parse_args()
    main(args.data_dir, args.output_csv)
