import os
import csv
import numpy as np
from scipy.interpolate import RectBivariateSpline
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

def apply_bicubic_interpolation(points, M=50, N=50):
    x = np.unique(points[:, 0])
    y = np.unique(points[:, 1])
    
    # Création d'une grille régulière à partir des points existants
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # Créer une matrice z correspondant à la grille
    zz = np.full_like(xx, np.nan, dtype=float)
    for xi, yi, zi in points:
        ix = np.where(x == xi)[0][0]
        iy = np.where(y == yi)[0][0]
        zz[ix, iy] = zi

    # Interpoler les valeurs manquantes avec bicubic
    mask = ~np.isnan(zz)
    x_valid = xx[mask]
    y_valid = yy[mask]
    z_valid = zz[mask]
    
    # On peut utiliser griddata pour bicubic sur l'ensemble
    from scipy.interpolate import griddata
    x_grid, y_grid = generate_grid(points, M, N)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    Z_grid = griddata(
        points[:, :2], points[:, 2],
        (X_grid, Y_grid),
        method='cubic'
    )

    # Conversion en liste de points
    final_points = np.column_stack((X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()))
    
    # Affichage rapide pour vérification
    plt.figure(figsize=(8,6))
    cp = plt.contourf(X_grid, Y_grid, Z_grid, cmap='viridis')
    plt.scatter(points[:,0], points[:,1], c='red', s=10, label='Points originaux')
    plt.colorbar(cp, label='z')
    plt.title("Interpolation Bicubique du fond de lac")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    return final_points

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interpolation bicubique 2D sur nuage de points")
    parser.add_argument("input_csv", type=str, help="Fichier CSV du nuage de points")
    parser.add_argument("output_csv", type=str, help="Fichier CSV pour sauvegarder le résultat")
    parser.add_argument("--M", type=int, default=50, help="Nombre de points sur X")
    parser.add_argument("--N", type=int, default=50, help="Nombre de points sur Y")
    args = parser.parse_args()

    points = read_point_cloud(args.input_csv)
    final_points = apply_bicubic_interpolation(points, M=args.M, N=args.N)
    save_point_cloud(final_points, args.output_csv)
    print(f"Nuage de points final sauvegardé dans {args.output_csv}")
