import os
import csv
import numpy as np
from PIL import Image
from skimage.measure import find_contours

# -----------------------------
# PARAMÈTRES
# -----------------------------
image_name = "Lake_Aydat_traitee.png"
pas_m = 50.0

data_name = image_name.replace("_traitee.png","")
data_path = "../raw/map/" + data_name + "/"
image_path = data_path + image_name
csv_path = data_path + "légende.csv"
output_dir = "../point_cloud/"
os.makedirs(output_dir, exist_ok=True)

output_csv = output_dir + data_name + ".csv"

# -----------------------------
# OUTILS
# -----------------------------
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def get_closest_depth(pixel_color, color_depth_dict):
    min_dist = float("inf")
    closest_depth = None
    for ref_color, depth in color_depth_dict.items():
        d = color_distance(pixel_color, ref_color)
        if d < min_dist:
            min_dist = d
            closest_depth = depth
    return closest_depth

def dist(p0, p1):
    return np.hypot(p1[0] - p0[0], p1[1] - p0[1])

def interpolate(p0, p1, t):
    return (
        p0[0] + t * (p1[0] - p0[0]),
        p0[1] + t * (p1[1] - p0[1])
    )

# -----------------------------
# LECTURE CSV LÉGENDE
# -----------------------------
color_depth = {}
depth_values = []

with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    first_row = next(reader)
    scale = float(first_row[1])
    next(reader)
    for row in reader:
        rgb = hex_to_rgb(row[0])
        depth = -float(row[1])
        color_depth[rgb] = depth
        depth_values.append(depth)
depth_values = sorted(depth_values)

# -----------------------------
# LECTURE IMAGE
# -----------------------------
img = Image.open(image_path).convert("RGB")
img_array = np.array(img)
height, width, _ = img_array.shape

# -----------------------------
# CARTE DE PROFONDEUR
# -----------------------------
depth_map = np.zeros((height, width))
for y in range(height):
    for x in range(width):
        depth_map[y, x] = get_closest_depth(tuple(img_array[y, x]), color_depth)


# -----------------------------
# EXTRACTION DROITES
# -----------------------------
points_list = []

for level in depth_values:
    contours = find_contours(depth_map, level)

    for contour in contours:
        if len(contour) < 2:
            continue

        prev_kept_point = None
        dist_acc = 0.0

        for i in range(len(contour) - 1):
            y0, x0 = contour[i]
            y1, x1 = contour[i + 1]

            p0 = (x0 * scale, (height - y0) * scale)
            p1 = (x1 * scale, (height - y1) * scale)

            seg_len = dist(p0, p1)
            dist_acc += seg_len

            if prev_kept_point is None:
                # premier point du contour
                points_list.append([p0[0], p0[1], level])
                prev_kept_point = p0
                continue

            if dist_acc >= pas_m:
                # on garde p1 TEL QUEL
                points_list.append([p1[0], p1[1], level])
                prev_kept_point = p1
                dist_acc = 0.0
        

# -----------------------------
# ÉCRITURE CSV
# -----------------------------
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["x","y","z"])
    writer.writerows(points_list)


print(f"✅ Nuage de points enregistrés : {output_csv}")
print(f"Nombre de points : {len(points_list)}")