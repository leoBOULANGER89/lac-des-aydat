import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# -----------------------------
# PARAMÈTRES
# -----------------------------
data_name = "Lake_Aydat"

data_path = "../curves/"
points_csv = data_path + data_name + "_points.csv"
lines_csv  = data_path + data_name + "_lines.csv"

output_dir = "../resultat/" + data_name + "/"
os.makedirs(output_dir, exist_ok=True)
output_image = output_dir + data_name + "_curves.png"

# -----------------------------
# LECTURE DES POINTS
# -----------------------------
points_list = []
with open(points_csv, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)  # ignorer la première ligne
    for row in reader:
        x, y = float(row[0]), float(row[1])
        points_list.append((x, y))
points = np.array(points_list)

# -----------------------------
# LECTURE DES DROITES
# -----------------------------
lines_list = []
depths_list = []
with open(lines_csv, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        ix, iy, depth = int(row[0]), int(row[1]), float(row[2])
        lines_list.append((ix, iy))
        depths_list.append(depth)
lines = np.array(lines_list)
depths = np.array(depths_list)

# -----------------------------
# NORMALISATION COULEUR
# -----------------------------
norm = colors.Normalize(vmin=min(depths), vmax=max(depths))
cmap = cm.viridis

# -----------------------------
# PLOT 3D
# -----------------------------
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for (ix, iy), depth in zip(lines, depths):
    x_vals = [points[ix, 0], points[iy, 0]]
    y_vals = [points[ix, 1], points[iy, 1]]
    z_vals = [depth, depth]
    ax.plot(x_vals, y_vals, z_vals, color=cmap(norm(depth)), linewidth=1)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (profondeur)")
ax.set_title("Contours 3D — Marching Squares")

# Optionnel : vue du dessus
# ax.view_init(elev=90, azim=-90)

# -----------------------------
# ENREGISTREMENT IMAGE
# -----------------------------
plt.savefig(output_image, dpi=300)
plt.close(fig)

print(f"✅ Visualisation 3D enregistrée : {output_image}")
print(f"Nombre de points : {len(points)}, Nombre de droites : {len(lines)}")
