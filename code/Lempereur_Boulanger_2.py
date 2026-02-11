import os
import csv
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def read_point_cloud(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    points = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])
                points.append((x, y, z))
            except KeyError:
                raise KeyError("Le CSV doit contenir les colonnes x, y, z")
            except ValueError:
                raise ValueError("Valeurs numériques invalides dans le CSV")

    if len(points) == 0:
        raise ValueError("Le fichier CSV du nuage de points est vide")

    return np.array(points)

def save_point_cloud(points, output_csv):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        for p in points:
            writer.writerow(p)

def generate_grid(points, M, N):
    x_min, y_min = np.min(points[:, 0]), np.min(points[:, 1])
    x_max, y_max = np.max(points[:, 0]), np.max(points[:, 1])
    x_grid = np.linspace(x_min, x_max, M)
    y_grid = np.linspace(y_min, y_max, N)
    return x_grid, y_grid

def apply_spline_and_gradient(points, M=50, N=50, gradient_threshold=0.01):
    x_grid, y_grid = generate_grid(points, M, N)
    final_points = []

    # Variables pour trouver la spline avec le plus grand gradient
    max_grad = 0
    best_spline = None  # stocke (x_fine, z_fine, dz, coord, orientation)

    # --- Spline selon les lignes (y constant) ---
    for yi in y_grid:
        line_points = points[np.abs(points[:, 1] - yi) < (y_grid[1] - y_grid[0])/2]
        if len(line_points) < 2:
            continue

        line_points = line_points[np.argsort(line_points[:, 0])]
        xs, zs = line_points[:, 0], line_points[:, 2]

        _, unique_indices = np.unique(xs, return_index=True)
        xs, zs = xs[unique_indices], zs[unique_indices]

        if len(xs) < 2:
            continue

        cs = CubicSpline(xs, zs, bc_type='clamped')
        x_fine = np.linspace(xs[0], xs[-1], 10*len(xs))
        z_fine = cs(x_fine)
        dzdx = cs(x_fine, 1)

        # Ajouter les points selon le gradient
        for x_val, z_val, dz in zip(x_fine, z_fine, dzdx):
            if abs(dz) > gradient_threshold:
                final_points.append((x_val, yi, z_val))

        # Vérifier si cette spline a le gradient le plus grand
        local_max = np.max(np.abs(dzdx))
        if local_max > max_grad:
            max_grad = local_max
            best_spline = (x_fine, z_fine, dzdx, yi, 'ligne')

    # --- Spline selon les colonnes (x constant) ---
    for xi in x_grid:
        col_points = points[np.abs(points[:, 0] - xi) < (x_grid[1] - x_grid[0])/2]
        if len(col_points) < 2:
            continue

        col_points = col_points[np.argsort(col_points[:, 1])]
        ys, zs = col_points[:, 1], col_points[:, 2]

        _, unique_indices = np.unique(ys, return_index=True)
        ys, zs = ys[unique_indices], zs[unique_indices]

        if len(ys) < 2:
            continue

        cs = CubicSpline(ys, zs, bc_type='natural')
        y_fine = np.linspace(ys[0], ys[-1], 10*len(ys))
        z_fine = cs(y_fine)
        dzdy = cs(y_fine, 1)

        # Ajouter les points selon le gradient
        for y_val, z_val, dz in zip(y_fine, z_fine, dzdy):
            if abs(dz) > gradient_threshold:
                final_points.append((xi, y_val, z_val))

        # Vérifier si cette spline a le gradient le plus grand
        local_max = np.max(np.abs(dzdy))
        if local_max > max_grad:
            max_grad = local_max
            best_spline = (y_fine, z_fine, dzdy, xi, 'colonne')

    # --- Affichage du graphique pour la spline avec le plus grand gradient ---
    if best_spline is not None:
        x_fine, z_fine, dz, coord, orientation = best_spline
        plt.figure(figsize=(10,5))

        # 1️⃣ Spline et points originaux du nuage initial
        plt.subplot(1,2,1)
        plt.plot(x_fine, z_fine, label=f"Spline {orientation} coord={coord:.2f}", color='blue')
        
        # Ajouter les points originaux (avant ajout)
        if orientation == 'ligne':
            line_points = points[np.abs(points[:,1] - coord) < (y_grid[1]-y_grid[0])/2]
            plt.scatter(line_points[:,0], line_points[:,2], color='red', label='Points d’origine')
            plt.xlabel("x")
        else:
            col_points = points[np.abs(points[:,0] - coord) < (x_grid[1]-x_grid[0])/2]
            plt.scatter(col_points[:,1], col_points[:,2], color='red', label='Points d’origine')
            plt.xlabel("y")
        
        plt.title("Spline avec le plus grand gradient")
        plt.ylabel("z")
        plt.grid(True)
        plt.legend()

        # 2️⃣ Gradient
        plt.subplot(1,2,2)
        plt.plot(x_fine, dz, color='green', label="Dérivée")
        plt.title("Gradient de la spline")
        plt.xlabel("x" if orientation=='ligne' else 'y')
        plt.ylabel("dz/dx" if orientation=='ligne' else 'dz/dy')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("Aucune spline valide trouvée pour affichage.")

    # Fusionner avec les points originaux
    return np.array(np.concatenate([points, final_points]))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spline cubique et ajout de points selon gradient")
    parser.add_argument("input_csv", type=str, help="Fichier CSV du nuage de points")
    parser.add_argument("output_csv", type=str, help="Fichier CSV pour sauvegarder le résultat")
    parser.add_argument("--M", type=int, default=50, help="Nombre de lignes dans X")
    parser.add_argument("--N", type=int, default=50, help="Nombre de lignes dans Y")
    parser.add_argument("--gradient_threshold", type=float, default=0.1, help="Seuil de gradient pour ajouter des points")
    args = parser.parse_args()

    points = read_point_cloud(args.input_csv)
    final_points = apply_spline_and_gradient(points, M=args.M, N=args.N, gradient_threshold=args.gradient_threshold)
    save_point_cloud(final_points, args.output_csv)
    print(f"Nuage de points final sauvegardé dans {args.output_csv}")
