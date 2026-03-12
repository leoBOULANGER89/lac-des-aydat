# Reconstruction 3D d'une surface à partir d'une carte bathymétrique

## Description du projet
Projet réalisé dans le cadre de nos études d'ingénieur informatique en 2ème année à l'ISIMA.

Ce projet vise à reconstruire une **surface 3D** à partir d'une **carte bathymétrique**.  
Le pipeline permet de transformer une **image contenant des courbes de niveau** en un **maillage 3D triangulé** exploitable en simulation.

> **Cas pratique appliqué : le lac d'Aydat**

---

## Pipeline

Le processus comprend plusieurs étapes :

1. Extraction des profondeurs depuis l'image
2. Reconstruction des courbes de niveau
3. Génération d'un nuage de points
4. Triangulation de Delaunay contrainte
5. Export du maillage au format `.obj`

---

## Structure du projet

```
./
├── data/
│   ├── final/
│   │   ├── Lake_Aydat_Bicubic_Delaunay.obj
│   │   ├── Lake_Aydat_CDT.obj
│   │   ├── Lake_Aydat_curves.png
│   │   ├── Lake_Aydat_Delaunay.obj
│   │   ├── Lake_Aydat_mesure_simu.png
│   │   └── Lake_Aydat_nuage_points.png
│   │
│   ├── processed/
│   │   ├── curves/
│   │   │   ├── Lake_Aydat_lines.csv
│   │   │   └── Lake_Aydat_points.csv
│   │   └── point_cloud/
│   │       └── Lake_Aydat.csv
│   │
│   └── raw/
│       └── map/
│           └── Lake_Aydat/
│               ├── Lake_Aydat_traitee.png
│               └── légende.csv
│
└── src/
    ├── extraction/
    │   └── traitement.py
    ├── io/
    │   └── RAndW.py
    ├── mesure/
    │   └── mesure.py
    ├── processing/
    │   ├── Bicubic.py
    │   ├── CDT.py
    │   └── Delaunay.py
    └── visualisation/
        └── visualisee.py
```

### Détail des dossiers

- **`./data/`** — Dossier principal contenant l'ensemble des données du projet.
  - **`./data/raw/`** — Données brutes, non modifiées par les scripts.
    - **`./data/raw/map/x/`** — Carte bathymétrique de **x**. Contient une version corrigée de la carte où chaque niveau (défini par les lignes de niveau) est colorié d'une couleur différente (de préférence en niveaux de gris), sans lignes de séparation. Contient également un fichier `légende.csv` associant chaque couleur à sa profondeur minimale.
  - **`./data/processed/`** — Données ayant déjà été traitées.
    - **`./data/processed/point_cloud/`** — Nuages de points générés par `./src/extraction/traitement.py`, au format `x.csv`.
    - **`./data/processed/curves/`** — Courbes de niveau approximées par des segments, générées par `./src/extraction/traitement.py`.
      - `x_lines.csv` — Segments d'approximation.
      - `x_points.csv` — Points utilisés pour l'approximation.
  - **`./data/final/x/`** — Résultats finaux : courbes de mesure générées par `./src/mesure/mesure.py` et surfaces produites par les scripts de `./src/processing/`.

- **`./src/`** — Scripts du projet.
  - **`./src/extraction/traitement.py`** — Extraction des informations des cartes situées dans `./data/raw/map/x/`, avec résultats placés dans `./data/processed/point_cloud/` ou `./data/processed/curves/`.
  - **`./src/io/RAndW.py`** — Fonctions de lecture et d'écriture des fichiers `.csv` et `.obj`.
  - **`./src/mesure/mesure.py`** — Calcul de métriques sur les fichiers `.obj`.
  - **`./src/processing/`** — Scripts de reconstruction de surface à partir du nuage de points.
    - **`./src/processing/y.py`** — Applique la méthode **y** pour passer des données de `./data/processed/curves/` ou `./data/processed/point_cloud/` à une surface 3D stockée dans `./data/final/x/`.
  - **`./src/visualisation/visualisee.py`** — Visualisation des données issues de `./data/processed/curves/` ou `./data/processed/point_cloud/`.

---

## Exécution des scripts

Pour exécuter les scripts, se placer à la racine du projet et utiliser la commande :

```bash
python -m <chemin.du.script.séparé.par.des.points>
```

Par exemple, pour exécuter `./src/extraction/traitement.py` :

```bash
python -m src.extraction.traitement
```

---

## Extraction des courbes de niveau

**Script :** `./src/extraction/traitement.py`

Ce script permet de **reconstruire des courbes de niveau (isobathes)** à partir d'une **image bathymétrique RGB** où chaque couleur représente une profondeur.

L'objectif est de transformer une **image traitée** en **données géométriques exploitables** pour la reconstruction d'une surface 3D (nuage de points ou segments de courbes).

L'extraction des contours est réalisée avec la fonction `skimage.measure.find_contours`, basée sur l'algorithme **Marching Squares**.

### Fonctionnalités

- Conversion **pixel → coordonnées métriques** (mètre/pixel)
- Association robuste **couleur → profondeur** via distance RGB minimale
- Extraction automatique des **courbes de niveau**
- **Échantillonnage métrique régulier** des contours
- Reconstruction des **boucles fermées** via Shapely
- Export des données pour reconstruction 3D

### Utilisation

```bash
python -m src.extraction.traitement <chemin vers l'image>
```

Le script recherche automatiquement `légende.csv` dans le même dossier que l'image.

**Options disponibles :**
- `--pas y` : impose un échantillonnage avec un pas de `y` mètres (par défaut : 10 m)
- `--p` : extrait uniquement les points, sans approximation des courbes

### Entrée

Le fichier `légende.csv` doit respecter le format suivant :

```csv
scale;0.5
color;depth
#0000FF;-5
#00FFFF;-10
```

- `scale` : facteur mètre/pixel
- `color` : couleur hexadécimale
- `depth` : profondeur associée

Un exemple est disponible dans `./data/raw/map/Lake_Aydat/légende.csv`.

### Sortie

- Avec l'option `--p` : `./data/processed/point_cloud/<nom_du_lieu>.csv`
- Sans l'option `--p` : deux fichiers sont créés :
  - `./data/processed/curves/<nom_du_lieu>_lines.csv`
  - `./data/processed/curves/<nom_du_lieu>_points.csv`

**Format du nuage de points :**
```csv
x,y,z
1,2,3
```
Exemple : `./data/processed/point_cloud/Lake_Aydat.csv`

**Format des courbes :**
```csv
id,idx,idy,depth
1,2,3,4
```
- `id` : identifiant de la courbe (`-1` si le segment est isolé)
- `idx`, `idy` : indices des points extrémités du segment
- `depth` : profondeur du segment

---

## Calcul des métriques

**Script :** `./src/mesure/mesure.py`

Ce script permet d'analyser la **qualité géométrique de maillages triangulés 3D** au format `.obj`.

Il calcule plusieurs **métriques classiques utilisées en simulation numérique et en géométrie des maillages**.

### Métriques calculées

- **Aspect Ratio** : mesure l'allongement des triangles.
- **Mean Ratio** : indicateur global de la qualité de forme d'un triangle.
- **Distribution des angles** : histogramme des angles internes des triangles.
- **Condition Number** : mesure la distorsion géométrique des éléments.

Les résultats sont visualisés sous forme d'**histogrammes comparatifs**.

### Utilisation

Analyse d'un seul maillage :
```bash
python -m src.mesure.mesure --name <chemin/vers/maillage.obj>
```
Le résultat est sauvegardé dans `<fichier>_mesure_simu.png`.

Analyse de tous les maillages `.obj` d'un dossier :
```bash
python -m src.mesure.mesure --all chemin/vers/dossier/
```
Le résultat est sauvegardé dans `chemin/vers/dossier/<dossier>_mesure_simu.png`.

L'option `-o` permet de spécifier un nom de fichier de sortie personnalisé.

---

## Triangulation de Delaunay

**Script :** `./src/processing/Delaunay.py`

Ce script permet de **générer un maillage triangulé 3D** à partir d'un **nuage de points CSV**.

La triangulation est réalisée sur la projection plane **(x, y)** à l'aide de l'algorithme de **Delaunay**, tandis que les **coordonnées z originales sont conservées** pour reconstruire la surface 3D.

Le résultat est exporté au format **`.obj`** via la bibliothèque `scipy.spatial.Delaunay`.

### Utilisation

```bash
python -m src.processing.Delaunay --name y
```

Le script recherche `./data/processed/point_cloud/y.csv` et génère `./data/final/y/y_Delaunay.obj`.

**Options disponibles :**
- `--o <chemin/vers/sortie.obj>` : chemin de sortie personnalisé (par défaut : `./data/final/y/y_Delaunay.obj`)

### Entrée

Fichier CSV produit par `./src/extraction/traitement.py`.  
Exemple : `./data/processed/point_cloud/Lake_Aydat.csv`

### Sortie

Fichier `.obj` contenant :
```obj
v x y z  # sommets
f i j k  # faces triangulaires
```
- `x`, `y`, `z` : coordonnées du sommet en mètres
- `i`, `j`, `k` : indices des sommets du triangle

---

## Triangulation de Delaunay Contrainte (CDT)

**Script :** `./src/processing/CDT.py`

Ce script permet de générer un **maillage triangulé 3D** à partir :

- d'un **nuage de points 3D**
- de **segments représentant des courbes de niveau**

La triangulation utilisée est une **Constrained Delaunay Triangulation (CDT)**. Contrairement à une triangulation de Delaunay classique, cette méthode **respecte des segments imposés** (PSLG : *Planar Straight Line Graph*), ce qui permet de conserver la géométrie des courbes de niveau dans le maillage final.

### Utilisation

```bash
python -m src.processing.CDT --name y
```

Le script recherche `./data/processed/curves/y_lines.csv` et `./data/processed/curves/y_points.csv`, puis génère `./data/final/y/y_CDT.obj`.

**Options disponibles :**
- `--o <chemin/vers/sortie.obj>` : chemin de sortie personnalisé (par défaut : `./data/final/y/y_CDT.obj`)

### Entrée

Fichiers CSV produits par `./src/extraction/traitement.py`.  
Exemples : `./data/processed/curves/Lake_Aydat_lines.csv` et `./data/processed/curves/Lake_Aydat_points.csv`

### Sortie

Fichier `.obj` contenant :
```obj
v x y z  # sommets
f i j k  # faces triangulaires
```
- `x`, `y`, `z` : coordonnées du sommet en mètres
- `i`, `j`, `k` : indices des sommets du triangle

---

## Interpolation bicubique du nuage de points

**Script :** `./src/processing/Bicubic.py`

Ce script permet de générer un **nuage de points interpolé** à partir d'un **nuage de points 3D (x, y, z)** en utilisant une **interpolation bicubique 2D**.

L'interpolation est réalisée sur le plan **(x, y)** afin d'estimer les valeurs **z** sur une **grille régulière**, ce qui permet de densifier le nuage de points avant une étape de triangulation ou de reconstruction de surface.

### Utilisation

```bash
python -m src.processing.Bicubic --name y
```

Le script recherche `./data/processed/point_cloud/y.csv` et génère `./data/processed/point_cloud/y_Bicubic.csv`.

**Options disponibles :**
- `--M x` : nombre de points interpolés sur l'axe X (par défaut : 50)
- `--N x` : nombre de points interpolés sur l'axe Y (par défaut : 50)

### Entrée

Fichier CSV produit par `./src/extraction/traitement.py`.  
Exemple : `./data/processed/point_cloud/Lake_Aydat.csv`

### Sortie

```csv
x,y,z
1,2,3
```
en mètres.  
Exemple : `./data/processed/point_cloud/Lake_Aydat_Bicubic.csv`

---

## Visualisation des données 3D

**Script :** `./src/visualisation/visualisee.py`

Ce script permet de **visualiser les données générées dans le pipeline** sous forme de figures 3D :

- **Nuage de points 3D**
- **Courbes de niveau 3D** issues du Marching Squares

Les figures sont automatiquement **sauvegardées au format PNG**.

### Utilisation

```bash
python -m src.visualisation.visualisee --name y
```

Le script recherche `./data/processed/point_cloud/y.csv`, `./data/processed/curves/y_lines.csv` et `./data/processed/curves/y_points.csv`, puis génère deux images `.png`.

**Options disponibles :**
- `--c` : affiche uniquement la représentation des courbes d'approximation
- `--p` : affiche uniquement la représentation du nuage de points
- `--o <chemin/vers/dossier/>` : dossier de sortie personnalisé

### Entrée

Fichiers CSV produits par `./src/extraction/traitement.py`.  
Exemples : `./data/processed/point_cloud/Lake_Aydat.csv`, `./data/processed/curves/Lake_Aydat_points.csv` et `./data/processed/curves/Lake_Aydat_lines.csv`

### Sortie

Images `.png` représentant les données extraites.
