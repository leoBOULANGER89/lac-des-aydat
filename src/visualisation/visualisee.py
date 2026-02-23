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
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from ..io import RAndW


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
    ax.set_zlabel("Z (m)")
    ax.set_title("Nuage de points 3D")

    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_curves(pts, curves, depths, output_path):
    """
    Génère et sauvegarde la visualisation des courbes 3D
    à partir des données issues de read_curves().
    """

    # Filtrer les profondeurs valides
    valid_depths = [d for d in depths if d is not None]
    if not valid_depths:
        raise ValueError("Aucune profondeur valide à tracer")

    norm = colors.Normalize(vmin=min(valid_depths), vmax=max(valid_depths))
    cmap = cm.viridis

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Parcours des courbes
    for curve_id, segments in enumerate(curves):

        if not segments:
            continue

        depth = depths[curve_id]
        if depth is None:
            continue

        color = cmap(norm(depth))

        # Tracé de chaque segment de la courbe
        for i0, i1 in segments:
            ax.plot(
                [pts[i0, 0], pts[i1, 0]],
                [pts[i0, 1], pts[i1, 1]],
                [depth, depth],
                color=color,
                linewidth=1
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Courbes 3D — Marching Squares")

    # Vue optionnelle en plan
    # ax.view_init(elev=90, azim=0)

    plt.tight_layout()
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
        points = RAndW.read_points(cloud_csv)

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
        pts, lines, depths = RAndW.read_curves(
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
    cloud_csv = f"data/processed/point_cloud/{name}.csv"
    curves_dir = "data/processed/curves/"
    curves_points_csv = os.path.join(curves_dir, f"{name}_points.csv")
    curves_lines_csv = os.path.join(curves_dir, f"{name}_lines.csv")

    output_dir = args.o if args.o else f"data/final/{name}/"
    os.makedirs(output_dir, exist_ok=True)

    try:
        if do_points:
            print("📊 Nuage de points...")
            points = RAndW.read_points(cloud_csv)
            out_img = os.path.join(output_dir, f"{name}_nuage_points.png")
            plot_point_cloud(points, out_img)
            print(f"✅ Nuage enregistré : {out_img}")

        if do_curves:
            print("📈 Courbes...")
            pts, lines, depths = RAndW.read_curves(curves_points_csv, curves_lines_csv)
            out_img = os.path.join(output_dir, f"{name}_curves.png")
            plot_curves(pts, lines, depths, out_img)
            print(f"✅ Courbes enregistrées : {out_img}")

    except Exception as e:
        print("\n❌ ERREUR :", e)
        sys.exit(1)