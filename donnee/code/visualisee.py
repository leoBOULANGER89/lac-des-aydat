#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualisation 3D d'un nuage de points et/ou de courbes bathymétriques.

Ce programme permet de :
- visualiser un nuage de points 3D à partir d'un fichier CSV,
- visualiser des courbes (issues par exemple de Marching Squares),
- enregistrer automatiquement les visualisations sous forme d'images.

Par défaut, les deux visualisations sont produites.
L'utilisateur peut choisir de ne produire que le nuage de points (--p)
ou que les courbes (--c).

Entrées attendues
-----------------
Nuage de points :
    CSV contenant les colonnes : x, y, z

Courbes :
    - <name>_points.csv : liste des points 2D
    - <name>_lines.csv  : indices des segments + profondeur

Utilisation
-----------
python visualisation.py
python visualisation.py --name Lake_Aydat
python visualisation.py --p
python visualisation.py --c
python visualisation.py --o ../../resultat/custom/
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (nécessaire pour 3D)


# =============================================================================================================
# LECTURE DES DONNÉES
# =============================================================================================================

def read_point_cloud(csv_path):
    """
    Lit un fichier CSV contenant un nuage de points 3D.

    Parameters
    ----------
    csv_path : str
        Chemin vers le fichier CSV contenant les colonnes x, y, z.

    Returns
    -------
    points : numpy.ndarray
        Tableau de forme (N, 3) contenant les coordonnées des points.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    KeyError
        Si les colonnes x, y, z sont absentes.
    ValueError
        Si le fichier est vide ou mal formé.
    """
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


def read_curves(points_csv, lines_csv):
    """
    Lit les fichiers CSV décrivant les courbes.

    Parameters
    ----------
    points_csv : str
        Fichier CSV contenant les coordonnées des points (x, y).
    lines_csv : str
        Fichier CSV contenant les segments (index point A, index point B, profondeur).

    Returns
    -------
    points : numpy.ndarray
        Tableau (N, 2) des points 2D.
    lines : numpy.ndarray
        Tableau (M, 2) des indices de segments.
    depths : numpy.ndarray
        Tableau (M,) des profondeurs associées.

    Raises
    ------
    FileNotFoundError
        Si un des fichiers est introuvable.
    ValueError
        Si les fichiers sont vides ou mal formés.
    """
    if not os.path.isfile(points_csv):
        raise FileNotFoundError(f"Fichier introuvable : {points_csv}")
    if not os.path.isfile(lines_csv):
        raise FileNotFoundError(f"Fichier introuvable : {lines_csv}")

    pts = []
    with open(points_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            pts.append((float(row[0]), float(row[1])))

    segs, depths = [], []
    with open(lines_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            segs.append((int(row[0]), int(row[1])))
            depths.append(float(row[2]))

    if len(pts) == 0 or len(segs) == 0:
        raise ValueError("Les fichiers de courbes sont vides")

    return np.array(pts), np.array(segs), np.array(depths)


# =============================================================================================================
# VISUALISATION
# =============================================================================================================

def plot_point_cloud(points, output_path):
    """Génère et sauvegarde la visualisation du nuage de points."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap="viridis", s=1, marker="x")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (profondeur)")
    ax.set_title("Nuage de points 3D")

    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_curves(points, lines, depths, output_path):
    """Génère et sauvegarde la visualisation des courbes 3D."""
    norm = colors.Normalize(vmin=min(depths), vmax=max(depths))
    cmap = cm.viridis

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    for (i, j), d in zip(lines, depths):
        ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            [d, d],
            color=cmap(norm(d)),
            linewidth=1
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (profondeur)")
    ax.set_title("Courbes 3D — Marching Squares")

    #ax.view_init(elev=90, azim=0)

    plt.savefig(output_path, dpi=300)
    plt.close(fig)


# =============================================================================================================
# MAIN
# =============================================================================================================
def main_visualise(
    name="Lake_Aydat",
    show_points=True,
    show_curves=True
):
    """
    Génère les visualisations 3D du nuage de points et/ou des courbes
    sans affichage, sans sauvegarde, et retourne les figures matplotlib.

    Parameters
    ----------
    name : str, optional
        Nom du jeu de données (sans extension).
    show_points : bool, optional
        Si True, génère la figure du nuage de points.
    show_curves : bool, optional
        Si True, génère la figure des courbes.

    Returns
    -------
    figures : dict
        Dictionnaire contenant les figures matplotlib :
        - "points" : matplotlib.figure.Figure ou None
        - "curves" : matplotlib.figure.Figure ou None

    Raises
    ------
    FileNotFoundError
        Si un fichier requis est introuvable.
    ValueError
        Si les données sont invalides ou vides.
    """

    figures = {
        "points": None,
        "curves": None
    }

    # chemins d'entrée
    cloud_csv = f"../point_cloud/{name}.csv"
    curves_dir = "../curves/"
    curves_points_csv = os.path.join(curves_dir, f"{name}_points.csv")
    curves_lines_csv = os.path.join(curves_dir, f"{name}_lines.csv")

    # ----------------------------------------------------------------
    # NUAGE DE POINTS
    # ----------------------------------------------------------------
    if show_points:
        points = read_point_cloud(cloud_csv)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=points[:, 2],
            cmap="viridis",
            s=1,
            marker="x"
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (profondeur)")
        ax.set_title("Nuage de points 3D")

        figures["points"] = fig

    # ----------------------------------------------------------------
    # COURBES
    # ----------------------------------------------------------------
    if show_curves:
        pts, lines, depths = read_curves(
            curves_points_csv,
            curves_lines_csv
        )

        norm = colors.Normalize(vmin=min(depths), vmax=max(depths))
        cmap = cm.viridis

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        for (i, j), d in zip(lines, depths):
            ax.plot(
                [pts[i, 0], pts[j, 0]],
                [pts[i, 1], pts[j, 1]],
                [d, d],
                color=cmap(norm(d)),
                linewidth=1
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (profondeur)")
        ax.set_title("Courbes 3D — Marching Squares")

        figures["curves"] = fig

    return figures



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualisation 3D d'un nuage de points et/ou de courbes."
    )

    parser.add_argument(
        "--name",
        type=str,
        default="Lake_Aydat",
        help="Nom du jeu de données (sans extension)."
    )

    parser.add_argument(
        "--p",
        action="store_true",
        help="Afficher uniquement le nuage de points."
    )

    parser.add_argument(
        "--c",
        action="store_true",
        help="Afficher uniquement les courbes."
    )

    parser.add_argument(
        "--o",
        "--output",
        type=str,
        default=None,
        help="Dossier de sortie des images."
    )

    args = parser.parse_args()

    # logique p / c
    do_points = args.p or not args.c
    do_curves = args.c or not args.p

    name = args.name

    # chemins
    cloud_csv = f"../point_cloud/{name}.csv"
    curves_dir = "../curves/"
    curves_points_csv = os.path.join(curves_dir, f"{name}_points.csv")
    curves_lines_csv = os.path.join(curves_dir, f"{name}_lines.csv")

    output_dir = args.o if args.o else f"../../resultat/{name}/"
    os.makedirs(output_dir, exist_ok=True)

    try:
        if do_points:
            print("📊 Nuage de points...")
            points = read_point_cloud(cloud_csv)
            out_img = os.path.join(output_dir, f"{name}_nuage_points.png")
            plot_point_cloud(points, out_img)
            print(f"✅ Nuage enregistré : {out_img}")

        if do_curves:
            print("📈 Courbes...")
            pts, lines, depths = read_curves(curves_points_csv, curves_lines_csv)
            out_img = os.path.join(output_dir, f"{name}_curves.png")
            plot_curves(pts, lines, depths, out_img)
            print(f"✅ Courbes enregistrées : {out_img}")

    except Exception as e:
        print("\n❌ ERREUR :", e)
        sys.exit(1)