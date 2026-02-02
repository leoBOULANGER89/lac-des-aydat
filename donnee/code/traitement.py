import os
import csv
import numpy as np
from PIL import Image

# -----------------------------
# PARAMÈTRES
# -----------------------------
data_path = "../raw/map/LakeAydat/"
image_path = data_path + "Lake_Aydat_traitee.png"
csv_path = data_path + "légende.csv"
output_dir = "../point_cloud/"

# -----------------------------
# FONCTIONS
# -----------------------------
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def get_closest_depth(pixel_color, color_depth_dict):
    """
    Retourne la profondeur correspondant à la couleur
    la plus proche (distance RGB minimale).
    """
    min_dist = float("inf")
    closest_depth = None

    for ref_color, depth in color_depth_dict.items():
        dist = color_distance(pixel_color, ref_color)

        if dist < min_dist:
            min_dist = dist
            closest_depth = depth

    return closest_depth

# -----------------------------
# LECTURE DE L'IMAGE
# -----------------------------
img = Image.open(image_path).convert("RGB")
img_array = np.array(img)
height, width, _ = img_array.shape

# -----------------------------
# LECTURE DU CSV (scale + couleurs)
# -----------------------------
color_depth = {}
scale = None

with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)

    # Lecture de l'échelle
    first_row = next(reader)
    if first_row[0].lower() != "scale":
        raise ValueError("La première ligne du CSV doit être : scale,<valeur>")
    scale = float(first_row[1])

    # Saut de l'en-tête couleur,profondeur
    next(reader)

    # Lecture des couleurs
    for row in reader:
        rgb = hex_to_rgb(row[0])
        depth = float(row[1])
        color_depth[rgb] = -depth


# --- pré-calcul des profondeurs ---
depth_map = np.zeros((height, width))

for y in range(height):
    for x in range(width):
        depth_map[y, x] = get_closest_depth(
            tuple(img_array[y, x]),
            color_depth
        )

# -----------------------------
# DÉTECTION DES JONCTIONS
# -----------------------------
points = []

for y in range(height):
    for x in range(width):
        current_depth = depth_map[y, x]
        not_append_point = False
        same_lvl = 0

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1),
                       (1,1), (-1,1), (1,-1), (-1,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbor_depth = depth_map[ny, nx]

                if neighbor_depth < current_depth:
                    not_append_point = True

                if neighbor_depth == current_depth:
                    same_lvl += 1

        if (not not_append_point) and same_lvl < 8:
            points.append((x * scale, (height-y) * scale, current_depth))

# -----------------------------
# NOM DU CSV DE SORTIE
# -----------------------------
png_name = os.path.splitext(os.path.basename(image_path))[0]
output_name = png_name.replace("_traitee", "") + ".csv"

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_name)

# -----------------------------
# ÉCRITURE DU CSV FINAL
# -----------------------------
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "z"])
    writer.writerows(points)

print(f"✅ Nuage de points créé : {output_path}")
