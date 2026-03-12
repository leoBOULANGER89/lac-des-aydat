# Reconstruction 3D d'une surface avec une carte des courbe de niveaux.

## Description du projet
Projet effectuer dans le cadre de nos études d'ingénieur informatique en 2ème année à l'ISIMA.
Ce projet vise à reconstruire une **surface 3D** à partir d'une **carte bathymétrique**.  
Le pipeline permet de transformer une **image contenant des courbes de niveau** en **maillage 3D triangulé** exploitable en simulation.
    **Nous avont appliquer ce processus afin au cas pratique du lac d'aydat**

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
./  
├── data/  
│ ├── final/  
| | ├── Lake_Aydat_Bicubic_Delaunay.obj  
| | ├── Lake_Aydat_CDT.obj  
| | ├── Lake_Aydat_curves.png  
| | ├── Lake_Aydat_Delaunay.obj  
| | ├── Lake_Aydat_mesure_simu.png  
| | └── Lake_Aydat_nuage_points.png  
| |  
│ ├── processed/  
│ | ├── curves/  
│ | | ├── Lake_Aydat_lines.csv  
│ | | └── Lake_Aydat_points.csv  
│ | └── point_cloud/  
│ |   └── Lake_Aydat.csv  
| |
│ └── raw/  
|   └── map/  
|     └── Lake_Aydat/  
|       ├── Lake_Aydat_traitee.png  
|       └── légende.csv  
|  
└── src/  
  ├── extraction/  
  │ └── traitement.py  
  ├── io/  
  │ └── RAndW.py  
  ├── mesure/  
  │ └── mesure.py  
  ├── processing/  
  │ ├── Bicubic.py  
  │ ├── CDT.py  
  │ └── Delaunay.py  
  └── visualisation/  
    └── visualisee.py  


- **./data/**  
  Dossier principal contenant l’ensemble des données du projet.  
    - **./data/raw/**  
      Contient les données brutes, non modifiées par les scriptes.  
        - **./data/raw/map/x**  
          Contien la carte bathymétrique de **x**.   
          Une version corriger de cet carte; chaque niveaux (définit par les ligne de niveaux) est colorier d'une couleurs différente (on recommende de le faire avec un niveaux de gris) sans ligne de separation de niveaux.  
          Ainsi qu'un ficher légende.txt qui définit fait correspondre a chaque couleurs, le niveaux le plus bas possible..  
    - **./data/processed/**  
      Dossier contenant l’ensemble des données qui ont deja ete traiter.  
        - **./data/processed/point_cloud**  
          Dossier contenant l’ensemble des cartes de niveaux transformer en nuage de point par **./src/extraction/traitement.py** sous forme de `x.csv`.  
        - **./data/processed/curves**  
          Dossier contenant l’ensemble des cartes de niveaux approximer par des droites via **./src/extraction/traitement.py**.  
          **x__lines.csv** contien les droites d'approximation.  
          **x_points.csv** contien les points servant a l'approximation.  
    - **./data/final/x**  
      Dossier contenant l’ensemble des resultats de x que ce soit les courbes de mesure fait via **./src/mesure/mesure.py** ou toutes les surfaces final fait a partir des scriptes de **./src/processing/**  

- **./src**  
  Dossiée renferment les scriptes utiliser.  
    - **./src/extraction/traitement.py**  
      Scripte permettant d'extraire les information des cartes present en **./data/raw/map/x** et mes les resultat dans le dossier correspondant : **./data/processed/point_cloud** ou **./data/processed/curves**.  
    - **./src/io/RAndW.py**   
      Scripte renferment les fonction de lecture et d'ecriture des ficher `.csv` et `.obj`  
    - **./src/mesure/mesure.py**   
      Scripte renferment permettant d'effectuer les mesure sur les `.obj`.  
    - **./src/processing/**   
      Dossier renferment les different scripte pour passer du nuage de point a la surface.  
        - **./src/processing/y.py**   
          Scripte permettant d'appliquer la methode y pour passer des donner de **x** dans **./data/processed/curves** ou **./data/processed/point_cloud**  a une surface 3D stoquer dans **./data/final/x**.  
    - **./src/visualisation/visualisee.py**   
      Scripte renferment permettant d'effectuer une visualisation des donner mis dans **./data/processed/curves** ou **./data/processed/point_cloud**

---

## Execution des scriptes

Pour excuter les scripte il faut se placer a l'origine et faire cet commande  
```bash
python -m < chemain du scripte a executer separer par des point >
```  

Par exemple, pour executer **./src/extraction/traitement.py** il faut se placer a l'origine et executer  
```bash
python -m src.extraction.traitement
```

---

## Extraction des courbes de niveau
Scripte : **./src/extraction/traitement.py**  
Ce script permet de **reconstruire des courbes de niveau (isobathes)** à partir d'une **image bathymétrique RGB** où chaque couleur représente une profondeur.

L'objectif est de transformer une **image traitée** en **données géométriques exploitables** pour la reconstruction d'une surface 3D (nuage de points ou segments de courbes).

L'extraction des contours est réalisée avec la fonction  
`skimage.measure.find_contours`, basée sur l'algorithme **Marching Squares**.

### Fonctionnalités

Le script fournit plusieurs fonctionnalités :

- conversion **pixel → coordonnées métriques** (mètre/pixel)
- association robuste **couleur → profondeur** via distance RGB minimale
- extraction automatique des **courbes de niveau**
- **échantillonnage métrique régulier** des contours
- reconstruction des **boucles fermées** via Shapely
- export des données pour reconstruction 3D

### Utilisation
```bash
python src.extraction.traitement < chemin vers l'image >
```
Le scripte vas chercher dans le même dossier `légende.csv` afin de correctement lire l'image.

differentes option sont possible :
- `--pas y` : pour imposer un echantillonage avec un pas de y. Par défaut, le pas est de 10 m
- `--p` : pour extraire uniquement les points et non l'approximation des courbes 

### Entré
Le scripte a besoin que `légende.csv` soit dans le format attendue, soit :
```csv
scale;0.5
color;depth
#0000FF;-5
#00FFFF;-10
```
où:
- `scale` : facteur mètre/pixel
- `color` : couleur hexadécimale
- `depth` : profondeu associée
**./data/raw/map/Lake_Aydat/légende.csv** est un exemple du format nécessaire

### Sortie
si l'option `--p` est présent, le ficher de sortie est **./data/processed/point_cloud/< nom du lieux >.csv**  
sinon cela cree deux ficher; **./data/processed/curves/< nom du lieux >_lines.csv** et **./data/processed/curves/< nom du lieux >_points.csv**  

- Le nuage de points est stoquer comme suit :  
  ```csv
  x,y,z
  1,2,3
  ```  
  en m  
  Voici un exemple : **./data/processed/curves/Lake_Aydat_points.csv**  
  
- les courbes sont stoquer comme suit :  
  ```csv
  id,idx,idy,depth
  1,2,3,4
  ```  
  - `id` correspond a l'id de la courbe. il vaux -1 si le segment est seul.  
  - `idx` et `idy` correspond aux l'indice des points d'extremiter du segment.  
  - `depth` correspond a la prondondeur du segment.  
  
---
  
## Calcule des mesures  
Scripte : **./src/mesure/mesure.py**  
Ce script permet d'analyser la **qualité géométrique de maillages triangulés 3D** au format `.obj`.   
  
Il calcule plusieurs **métriques classiques utilisées en simulation numérique et en géométrie des maillages**.  
  
### Métriques calculées  
  
Les métriques suivantes sont évaluées pour chaque triangle du maillage :  
  
- **Aspect Ratio** : mesure l'allongement des triangles.  
- **Mean Ratio** : indicateur global de la qualité de forme d'un triangle.  
- **Distribution des angles** : histogramme des angles internes des triangles.  
- **Condition Number** : mesure la distorsion géométrique des éléments.  
  
Les résultats sont visualisés sous forme **d'histogrammes comparatifs**.  
  
### Utilisation  
  
Analyse d'un seul maillage  
```bash
python -m src.mesure.mesure --name < chemin vers maillage.obj >
```  
le resultat seras alors en `<fichier>_mesure_simu.png`  

Analyse de tous les maillages en `.obj` d'un dossier  
```bash
python -m src.mesure.mesure --all chemin/vers/dossier/
```  
le resultat seras alors en `chemin/vers/dossier/<dossier>_mesure_simu.png`  
  
Il est possible d'ajouter l'option `-o`pour donner le nom de sortie.  
  
---
  
## Application de la triangulation de Delaunay  
Scripte : **./src/processing/Delaunay.py**  
Ce script permet de **générer un maillage triangulé 3D** à partir d'un **nuage de points CSV**.  
  
La triangulation est réalisée sur la projection plane **(x, y)** à l'aide de l'algorithme de **Delaunay**, tandis que les **coordonnées z originales sont conservées** pour reconstruire la surface 3D.  
  
Le résultat est exporté sous forme de **maillage au format `.obj`**.  
La triangulation est réalisée avec la bibliothèque `scipy.spatial.Delaunay`.  
  
### Utilisation  
```bash
python -m src.processing.Delaunay --name y
```  
Le scripte vas alors chercher **./data/processed/point_cloud/y.csv**  
  
differentes option sont possible :  
- `--o < chemin vers le ficher de sortie >.obj` : pour modifier le nom du ficher de sortie. Par defaut : **./data/final/y/y_Delaunay.obj**  
  
### Entré  
Le script attend le fichier `CSV` sortie de **./src/extraction/traitement.py**  
Exemple : **./data/processed/point_cloud/Lake_Aydat.csv**  
  
### Sortie  
Le script génère un fichier `.obj` contenant :  
```obj
v x y z # sommets
f i j k # faces triangulaires
```  
où:  
- `x`, `y` et `z` sont les cordoner du sommet en m  
- `i`, `j` et `k` sont les indices des sommet du triangles  
