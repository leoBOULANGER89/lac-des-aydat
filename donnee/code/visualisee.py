import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour 3D

# -----------------------------
# PARAMÈTRES
# -----------------------------

data_path = "../point_cloud/"
csv_path = data_path + "Lake_Aydat.csv"
output_image = data_path + "nuage_points.png"

# -----------------------------
# LECTURE DU CSV
# -----------------------------
points = []

with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x = float(row["x"])
        y = float(row["y"])
        z = -float(row["z"])
        points.append((x, y, z))

points = np.array(points)

# -----------------------------
# PLOT 3D
# -----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# nuage de points
ax.scatter(points[:,0], points[:,1], points[:,2],
           c=points[:,2], cmap='viridis', marker='x', s=1)  # couleur selon z, taille 1

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (profondeur)")
ax.set_title("Nuage de points 3D")

ax.view_init(elev=90, azim=-90)
# -----------------------------
# ENREGISTREMENT DIRECT
# -----------------------------
plt.savefig(output_image, dpi=300)
plt.close(fig)  # ferme la figure pour libérer la mémoire

print(f"✅ Nuage de points enregistré : {output_image}")
