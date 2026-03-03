#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'interpolation bicubique 2D d'un nuage de points.

Ce module permet de générer un nuage de points interpolé via une interpolation bicubique à partir d'un fichier CSV contenant
les coordonnées 3D (x, y, z). Le résultat est sauvegardé au format CSV.

Utilisation en ligne de commande
--------------------------------

Traitement d'un jeu de données par défaut (Lake_Aydat) :

    python script.py

Spécifier un fichier CSV à traiter :

    python script.py --name chemin/vers/points.csv

Optionnel :

    --o chemin/fichier.csv
        Définit le chemin du fichier CSV de sortie.
        Par défaut : data/processed/point_cloud/{name}_Bicubic.csv

    --M int
        Nombre de points interpolés le long de l'axe X (défaut 50).

    --N int
        Nombre de points interpolés le long de l'axe Y (défaut 50).

Logique des options
-------------------

- Si --name n'est pas fourni :
       le jeu de données par défaut "Lake_Aydat" est utilisé.
- Si --o n'est pas fourni :
       le fichier de sortie est généré automatiquement dans data/processed/point_cloud/{name}_Bicubic.csv.
- Les paramètres --M et --N définissent le maillage de la grille pour l'interpolation bicubique.

Organisation des fichiers
--------------------------

Entrée :
    - Fichier CSV contenant le nuage de points (colonnes x, y, z).

Sortie :
    - Fichier CSV contenant le nuage de points interpolé.

Les dossiers de sortie sont créés automatiquement si nécessaires.

Gestion des erreurs
-------------------

FileNotFoundError
    Si le fichier CSV spécifié n'existe pas.

pd.errors.EmptyDataError
    Si le fichier CSV est vide.

KeyError
    Si le CSV ne contient pas les colonnes attendues.

ValueError
    Si le CSV contient moins de 4 points ou un nombre de colonnes incorrect.

Exception
    Pour toute autre erreur inattendue, un message détaillé est affiché.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from ..io import RAndW


# =============================================================================================================
# PARAMÈTRES
# =============================================================================================================
def determine_param(name, output_override=None):
    """
    Détermine les chemins d'entrée et de sortie pour l'interpolation Bicubique d'un nuage de points 3D.

    Parameters
    ----------
    name : str
        Nom du jeu de données (sans extension). Utilisé pour construire automatiquement le chemin vers le fichier CSV d'entrée :
            data/processed/point_cloud/{name}.csv
    output_override : str or None, optional
        Chemin complet du fichier CSV de sortie.
        Si fourni, ce chemin est utilisé directement et son dossier parent est créé si nécessaire.
        Si None, le fichier de sortie est généré automatiquement dans :
            data/processed/point_cloud/{name}_Bicubic.csv

    Returns
    -------
    INPUT_CSV : str
        Chemin vers le fichier CSV contenant le nuage de points d'entrée.
    OUTPUT_CSV : str
        Chemin vers le fichier CSV de sortie contenant les points interpolés via Bicubic.

    Notes
    -----
    - Les dossiers de sortie sont créés automatiquement s'ils n'existent pas.
    - Cette fonction centralise la logique des chemins pour éviter la duplication dans le reste du programme.
    """

    INPUT_CSV = f"data/processed/point_cloud/{name}.csv"

    if output_override is not None:
        OUTPUT_CSV = output_override
        output_dir = os.path.dirname(OUTPUT_CSV)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = f"data/processed/point_cloud/"
        os.makedirs(output_dir, exist_ok=True)
        OUTPUT_CSV = os.path.join(output_dir, f"{name}_Bicubic.csv")

    return INPUT_CSV, OUTPUT_CSV

# =============================================================================================================
# GÉNÉRATION DE GRILLE
# =============================================================================================================
def generate_grid(points, M, N):
    """
    Génère une grille régulière dans le plan XY pour l'interpolation.

    Parameters
    ----------
    points : np.ndarray
        Tableau de shape (N_points, 3) contenant les coordonnées x, y, z.
    M : int
        Nombre de points de la grille le long de l'axe X.
    N : int
        Nombre de points de la grille le long de l'axe Y.

    Returns
    -------
    X_grid, Y_grid : np.ndarray
        Grilles 2D pour X et Y, de shape (M, N), utilisables pour l'interpolation Bicubic.

    Notes
    -----
    - La grille couvre l'étendue minimale et maximale des points dans les directions X et Y.
    - indexing="ij" garantit que X_grid varie le long de l'axe 0 et Y_grid le long de l'axe 1.
    """

    x_min, y_min = np.min(points[:, 0]), np.min(points[:, 1])
    x_max, y_max = np.max(points[:, 0]), np.max(points[:, 1])

    x_grid = np.linspace(x_min, x_max, M)
    y_grid = np.linspace(y_min, y_max, N)

    return np.meshgrid(x_grid, y_grid, indexing="ij")


# =============================================================================================================
# INTERPOLATION BICUBIQUE
# =============================================================================================================
def apply_bicubic_interpolation(points, M=50, N=50):
    """
    Applique une interpolation Bicubique sur un nuage de points 3D.

    Parameters
    ----------
    points : np.ndarray
        Tableau de shape (N_points, 3) contenant les coordonnées x, y, z.
    M : int, optional
        Nombre de points de la grille interpolée le long de l'axe X (défaut 50).
    N : int, optional
        Nombre de points de la grille interpolée le long de l'axe Y (défaut 50).

    Returns
    -------
    final_points : np.ndarray
        Tableau de shape (M*N, 3) contenant les points interpolés
        avec les colonnes [x, y, z].

    Notes
    -----
    - Utilise scipy.interpolate.griddata avec method="cubic".
    - La grille XY est générée automatiquement selon l'étendue des points originaux.
    - Les valeurs Z sont interpolées sur chaque point de la grille.
    """

    X_grid, Y_grid = generate_grid(points, M, N)

    Z_grid = griddata(
        points[:, :2],
        points[:, 2],
        (X_grid, Y_grid),
        method="cubic"
    )

    final_points = np.column_stack(
        (X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel())
    )

    return final_points


# =============================================================================================================
# SAUVEGARDE CSV
# =============================================================================================================
def save_csv(output_csv, points):
    """
    Sauvegarde un nuage de points interpolés au format CSV.

    Parameters
    ----------
    output_csv : str
        Chemin complet du fichier CSV de sortie.
    points : np.ndarray
        Tableau de shape (N_points, 3) contenant les coordonnées x, y, z des points interpolés.

    Returns
    -------
    None
        La fonction écrit directement le fichier CSV sur le disque.

    Notes
    -----
    - Les colonnes sont nommées "x", "y" et "z".
    - Le CSV peut ensuite être utilisé pour reconstruction 3D ou visualisation dans d'autres scripts.
    """

    df = pd.DataFrame(points, columns=["x", "y", "z"])
    df.to_csv(output_csv, index=False)


# =============================================================================================================
# MAIN
# =============================================================================================================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Interpolation bicubique 2D d'un nuage de points."
    )

    parser.add_argument(
        "--name",
        type=str,
        default="Lake_Aydat",
        help="Nom du fichier CSV à traiter (sans extension)."
    )

    parser.add_argument(
        "--o",
        "--output",
        type=str,
        default=None,
        help="Chemin du fichier CSV de sortie."
    )

    parser.add_argument(
        "--M",
        type=int,
        default=50,
        help="Nombre de points sur X."
    )

    parser.add_argument(
        "--N",
        type=int,
        default=50,
        help="Nombre de points sur Y."
    )

    args = parser.parse_args()

    print("Utilisation de l'interpolation bicubique :")

    try:
        INPUT_CSV, OUTPUT_CSV = determine_param(args.name, args.o)

        if not os.path.isfile(INPUT_CSV):
            raise FileNotFoundError(f"Fichier CSV introuvable : {INPUT_CSV}")

        print("\t📊 Lecture des points")
        points = RAndW.read_points(INPUT_CSV)

        if points.shape[0] < 4:
            raise ValueError("Au moins 4 points sont nécessaires pour l'interpolation cubic")

        print("\t🧮 Interpolation bicubique")
        final_points = apply_bicubic_interpolation(points, args.M, args.N)

        print("\t💾 Sauvegarde CSV")
        save_csv(OUTPUT_CSV, final_points)

        print(f"\t✅ Nuage interpolé enregistré : {OUTPUT_CSV}")

    except FileNotFoundError as e:
        print(f"\n❌ ERREUR : {e}")
        sys.exit(1)

    except pd.errors.EmptyDataError:
        print("\n❌ ERREUR : Le fichier CSV est vide")
        sys.exit(1)

    except KeyError as e:
        print(f"\n❌ ERREUR : {e}")
        sys.exit(1)

    except ValueError as e:
        print(f"\n❌ ERREUR : {e}")
        sys.exit(1)

    except Exception as e:
        print("\n❌ ERREUR INATTENDUE")
        print(f"   {type(e).__name__} : {e}")
        sys.exit(1)