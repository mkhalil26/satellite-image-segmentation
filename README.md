# Projet Math-appli 1A — Segmentation d’images de lacs de montagne

Projet réalisé par **l’équipe 28** composée de Mostafa E.J, Obaida E.J, Paul B et Mohamed Khalil B

## Description
Ce projet vise la **segmentation automatique de surfaces d’eau** à partir d'images colocalisées et d’images **SAR** suivant **l'indice OASIS**.

Plusieurs méthodes de segmentation sont implémentées et comparées sur différentes  
**zones géographiques**, **mois** et **années**, à l’aide de métriques quantitatives.

L’objectif est d’évaluer la **robustesse des méthodes** selon les conditions
temporelles et spatiales.

Les méthodes principales étudiées sont :
- la **segmentation par la méthode du seuilage fixe**,
- la **segmentation par la méthode de Chan–Vese**,
- la **segmentation par la méthode Random Forest**

---

## Table des matières

- [Structure du dépôt](#structure-du-dépôt)
- [Modules utilisés](#modules-utilisés)
- [Scores de performances](#scores-de-performances)
- [Utilisation](#utilisation)

---

## Structure du dépôt

### dossier `Data/` (non présent)

  Images SAR Sentinel-1 (selon l'indice OASIS) au format `tif`, organisées par zone géographique (zones 1 à 8). Nous avons plus précisément utilisé les sous-dossiers `MeanMonthly` représentant les moyennes mensuelles.

---

### dossier `GroundTruth_DYN/` (non présent)

  Images de référence (vérité terrain) sous forme de moyennes mensuelles utilisées pour l’apprentissage (période 2023-2024) et l’évaluation des performances (période 2021-2022).

---

### dossier `NDWI_colocalise_avec_sar/` (non présent)

  Images colocalisées entre les images SAR et optiques utiles à l'apprentissage des modèles de la méthode **Random Forest** dans les scripts `RF_Z.py` et `RF_ZM.py`.

---

### dossier `codes/`

Les scripts sont conçus pour fonctionner sur **8 zones** et sur plusieurs
**années (2021–2024)**.

Contient l’ensemble des **scripts Python** du projet :
- implémentation des méthodes de segmentation,
- calcul des métriques,
- génération automatique des graphes.

Scripts Python du projet :
- **`fonctions_images.py`**  
  Fonctions utilitaires pour le traitement et l'affichage des images tif.
- **`fonctions_tests.py`**
  Fonctions de tests et d'affichage pour les fonctions de segmentation.
- **`fonctions_scores.py`**  
  Fonctions de calcul des scores et métriques d’évaluation.
- **`fonctions_RF.py`**
  Fonctions communes aux implémentations de la méthode Random Forest.

- **`seuilF.py`**  
  Implémentation de la méthode de segmentation par seuillage fixe avec filtre gaussien.
- **`CVF_seuil.py`**  
  Implémentation de la méthode de segmentation de **Chan–Vese** avec initialisation par seuillage.
- **`CV_scikit_modifie.py`**
  Implémentation modifiée de l’algorithme **Chan–Vese** du module scikit.

- **`RF_SAR.py`**  
  Implémentation de la méthode **Random Forest** à partir des images SAR : entraînement, prédiction et évaluation.
- **`RF_Z.py`**
  Seconde implémentation de la méthode **Random Forest** à partir des images colocalisées entre SAR et optiques prenant en compte la zone où se situe l'image.
- **`RF_ZM.py`**
  Implémentation de la méthode **Random Forest** à partir d'images colocalisées prenant en compte la zone et le mois.
- **`RF_C.py`**
  Prototype de la méthode **Random Forest** appliqué à des images SAR et à leurs versions colocalisées.  
  Ce script a été utilisé à des fins de test sur un jeu de données restreint et n’a pas été retenu pour l’analyse finale.

---

### dossier `resultats/`
Résultats générés automatiquement par les scripts :
- **`entrainements_RF/`** : modèles Random Forest sauvegardés  
- **`graphes_scores/`** : graphes d’évaluation des moyennes des scores pour une méthode de segmentation au cours du temps et en fonction des zones
- **`graphes_segmentation/`** : visualisations des segmentations sur une année 
- **`scores/`** : affichages des différents scores pour une méthode de segmentation donnée
- **`figures_rapport/`** : les figures produites qui ont servi dans la rédaction du rapport

---

### dossier `planning/`
  Contient le planning du projet (diagramme de Gantt) illustrant l’organisation du travail.

---

### Fichiers principaux
- **`rapport_PMA.pdf`** : rapport final du projet
- **`sujet.pdf`** : sujet officiel du projet

---

## Modules utilisés

Nous avons principalement utilisé les modules :
- [scikit-learn](https://scikit-learn.org/)
- [scikit-image](https://scikit-image.org/)
- [scipy](https://docs.scipy.org/doc/scipy/#)
- [numpy](https://numpy.org/)
- [rasterio](https://rasterio.readthedocs.io/en/stable/)
- [geopandas](https://geopandas.org/en/stable/)
- [joblib](https://joblib.readthedocs.io/en/stable/)
- [matplotlib](https://matplotlib.org/)
---


## Scores de performances

Les performances sont évaluées selon plusieurs scores :

Scores qui doivent tendre vers **0** pour une bonne segmentation (en rouge dans les sorties numériques) :

- distance de Hamming
- différence d’aire
- fausse détection

Scores qui doivent tendre vers **1** pour une bonne segmentation (en bleu dans les sorties numériques) :

- vraie détection
- corrélation
- similarité structurelle

---

## Utilisation

Les scripts principaux se trouvent dans le dossier `codes/`.  

Ils permettent :

- de manipuler des images
- de faire des segmentations
- de calculer les scores et générer des graphes.

### Récupération des images

Les images fournies ne sont pas dans ce dépôt car elles sont trop volumineuses mais nous avons utilisé les dossiers `Data/`, `GroundTruth_DYN/` et `NDWI_colocalise_avec_sar/` qui contiennent les images **tif** des moyennes mensuelles, les images de référence et les images colocalisées.

Les images fournies des zones 1 à 8 sont récupérables depuis la fonction

```python
  def recuperer_images(
    mean_monthly : bool = True,
    zone : int = 2,
    selected_dates : list[str] = ['20210816', '20210828']
  
  ) -> list[tuple[MaskedArray, MaskedArray | None]]:
```

ou l'on peut indiquer si l'on veut la moyenne mensuelle, la zone et les dates voulues. On récupère alors une liste de couples d'images avec l'image récupérée sous forme de variable numpy et l'image de référence associée si trouvée dans le cas contraire on aura **None**.

On peut également récupérer une image quelconque au format **tif** avec la fonction :

```python
  def recuperer_image(
    path : str
  
  ) -> MaskedArray:
```

à partir d'un chemin et qui renvoie la matrice de valeurs.

### méthodes de segmentation

Nos méthodes de segmentation prennent en paramètre une image **SAR** représentée avec l'indice **OASIS** et renvoient une autre image segmentée, remplie de valeurs, **1** pour un pixel désignant une surface d'eau du lac et **0** pour ce qui n'en est pas. Souvent il y a des paramètres supplémentaires spécifiques à la méthode.

- #### Seuillage fixe

  cette méthode a un fonctionnement plutôt simple qui utilise un filtre gaussien, elle est disponible dans le script `seuilF.py`. Voici la fonction associée :

  ```python

  def segmentation_seuillage_fixe(
    image: MaskedArray,
    seuil: float = 0.2,
    sigma: int = 2
  
  ) -> ndarray:
  
  ```
  Avec le seuil entre **0** et **1** et **sigma** l'écart-type de la fonction gaussienne.

- #### Chan–Vese

  pour cette méthode, nous avons modifié la méthode de Chan-Vese (dans le script `CV_scikit_modifie.py`) implémentée par le module scikit-image pour y intégrer une **initialisation par seuillage**. Cette méthode est coûteuse en temps de calcul, sa fonction est disponible dans le script `CVF_seuil.py` :

  ```python

  def segmentation_chan_vese_seuil(
    image: MaskedArray,
    sigma : int = 2,
    mu : float = 0.07,
    lambda1 : float = 1,
    lambda2 : float = 1,
    max_num_iter : int = 200,
    tol : float = 5e-4
  
  ) -> ndarray:

  ```

  Avec **sigma** l'écart-type de l'initialisation par seuillage, **mu** le coefficient de régularisation du contour, **lambda1** et **lambda2** les coefficients de l'intérieur et de l'extérieur du contour. On peut également modifier le nombre maximum d'itérations pour que la fonction converge ainsi que la tolérance.

- #### Random Forest

  Pour cette méthode, les images sont classées selon des caractéristiques et ont été entraînées à l'aide du contenu du dossier `GroundTruth_DYN/` contenant les vérités terrain. 
  Nous avons implémenté 3 variantes de fonctions de segmentation disponibles dans les fichiers `RF_SAR.py`, `RF_Z.py` et `RF_ZM.py` 

  ```python
  def segmentation_random_forest_SAR(
    image : MaskedArray,
    modele : RandomForestClassifier,
    *args
    
  ) -> np.ndarray:

  ```
  
  Cette première variante utilise uniquement les images **SAR** et son modèle comporte seulement 3 caractéristiques : **l'intensité des pixels**, **leur variance** et **le seuil fixe**.  

  ```python
  def segmentation_random_forest_Z(
    image : MaskedArray,
    modele : RandomForestClassifier,
    colocated : list,
    zone : int,
    *args
  
  ) -> ndarray:
  ```

  Cette seconde implémentation prend en compte également les zones colocalisées en fournissant la zone colocalisée correspondante à celle de l'image au modèle. Ce qui apporte plus de contexte.

  ```python
  def segmentation_random_forest_ZM(
    image : MaskedArray,
    modele : RandomForestClassifier,
    colocated : list,
    zone : int,
    mois : int
  
  ) -> ndarray:
  ```

  Enfin cette fonction prend également en compte les images colocalisées et le mois associé à l'image **SAR** pour fournir l'image colocalisée la plus proche en terme de temps. Cela apporte plus de précision.


Random Forest : nombre d’arbres, profondeur, taille minimale des feuilles

### Évaluation des scores et affichages

Les sorties numériques sont enregistrés dans le dossier `resultats/`.

- On peut afficher une image numpy avec la fonction suivante :
  
  ```python
  def afficher_image(
    image : MaskedArray,
    cmap = INDICATEUR_OASIS
  
  ) -> None:
  ```
  On peut mettre un indicateur, par défaut l'image sera affichée avec les couleurs du format **OASIS**. Les zones représentant un lac sont en **bleu** et celles qui n'en font pas partie sont en **blanc**.

- Pour expérimenter une segmentation et la comparer à la référence on utilise la fonction :

  ```python
  def test_segmentation(
    image_ref : tuple[MaskedArray, MaskedArray | None],
    fonction_segmentation : Callable[[MaskedArray], ndarray]
  
  )-> None:
  ```
  
  qui prend en argument l'image non segmentée, son image de référence et la fonction de segmentation et qui affiche l'image après la segmentation comparée à sa référence. On a également la fonction suivante :
  
  ```python
  def tests_segmentation(
    fonction_segmentation : Callable[[MaskedArray], ndarray],
    annee : int = 2021,
    mois : list[int] = np.arange(1,13),
    zones : list[int] = np.arange(1,9),
    mean_monthly : bool = True,
    resolution : int = 300
  
  ) -> None:
  ```
  qui effectue les mêmes opérations mais qui applique la segmentation pour toutes les images d'une année. Cet affichage est sauvegardé dans le dossier **`graphes_segmentation/`**.

- Pour obtenir les scores d'une segmentation nous avons appliqué la fonction :

  ```python
  def moyenne_scores_annees(
    fonction_segmentation: Callable[[MaskedArray], ndarray],
    annees: list[int] = [2021, 2022, 2023, 2024],
    mois : list[int] = np.arange(1,13),
    zones : list[int] = np.arange(1,9)
  
  ) -> None:
  ```

  qui calcule et affiche la moyenne des 6 scores présentés précédemment pour les zones, mois et années spécifiées, les trois scores qui doivent tendre vers **0** pour une bonne segmentation sont affichés sur la courbe rouge et ceux qui doivent tendre vers **1** sont représentés en bleu. Cet affichage est sauvegardé dans le dossier **`scores/`**.

  Un autre moyen de visualiser les scores dans le temps est la fonction suivante :
    
  ```python

  def graphe_scores(
    fonction_segmentation: Callable[[MaskedArray], ndarray],
    zones : list[int] = np.arange(1,9),
    mois : list[int] = np.arange(1,13),
    annees: list[int] = [2021, 2022, 2023, 2024],
    mean_monthly: bool = True,
    resolution: int = 250,
    figsize: tuple[int, int] = (16, 14)
  
  ) -> None:
  ```

  qui représente de manière chronologique la moyenne des 3 scores bleus et ceux en rouge pour les années, mois et zones données. Cet affichage est sauvegardé dans le dossier **`graphes_scores/`**.

- Pour ce qui est de la méthode Random Forest utilisant les images colocalisées nous avons des variantes des fonctions `tests_segmentation_ZM()`, `moyenne_scores_annees_ZM()` et `graphe_scores()` qui ont les mêmes comportement que les fonctions précédentes mais qui passent en plus en paramètre les modèles et images colocalisées :

  ```python
    def tests_segmentation_ZM(
      fonction_segmentation : Callable[[MaskedArray, RandomForestClassifier, list, int, int], ndarray],
      modele : RandomForestClassifier,
      colocated : list = [],
      annee : int = 2021,
      mois : list[int] = np.arange(1,13),
      zones : list[int] = np.arange(1,9),
      mean_monthly : bool = True,
      resolution : int = 300

  ) -> None: 

  ```

  ```python
  def moyenne_scores_annees_ZM(
    fonction_segmentation: Callable[[MaskedArray], ndarray],
    modele : RandomForestClassifier,
    colocated : list = [],
    annees: list[int] = [2021, 2022, 2023, 2024],
    mois : list[int] = np.arange(1,13),
    zones : list[int] = np.arange(1,9)

  ) -> None:
  ```

  ```python
  def graphe_scores_ZM(
    fonction_segmentation: Callable[[MaskedArray], ndarray],
    modele : RandomForestClassifier,
    colocated : list = [],
    zones : list[int] = np.arange(1,9),
    mois : list[int] = np.arange(1,13),
    annees: list[int] = [2021, 2022, 2023, 2024],
    mean_monthly: bool = True,
    resolution: int = 250,
    figsize: tuple[int, int] = (16, 14)
  
  ) -> None:
  ```

