import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from numpy import ndarray
from numpy.ma import MaskedArray
from skimage.metrics import structural_similarity
from fonctions_images import *


DOSSIER_SCORES = "scores"
DOSSIER_GRAPHES_SCORES = "graphes scores"


# calculs des scores
def distance_hamming(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 0 signifie parfait
    
    if image_segmentee_1.shape != image_segmentee_2.shape:
        return float('nan')
    
    return np.sum(image_segmentee_1 != image_segmentee_2) / (image_segmentee_1.shape[0] * image_segmentee_2.shape[1])

def difference_aire(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 0 signifie parfait
    
    aire_1 = np.sum(image_segmentee_1)
    aire_2 = np.sum(image_segmentee_2)

    return abs(aire_1 - aire_2) / max(aire_1, aire_2)

def fausse_detection(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 0 signifie parfait
    
    ref = np.asarray(image_segmentee_1).ravel()
    seg = np.asarray(image_segmentee_2).ravel()

    if ref.shape != seg.shape:
        return float('nan')

    ref_pos = ref > 0
    seg_pos = seg > 0

    vrais_negatifs = np.sum(~ref_pos & ~seg_pos) # vrais négatifs
    faux_positifs = np.sum(~ref_pos & seg_pos) # faux positifs

    denom = vrais_negatifs + faux_positifs
    if denom == 0:

        if faux_positifs == 0:
            return 1.0
        
        return float('nan')

    return float(faux_positifs / denom)

def score_correlation(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 1 signifie parfait

    if image_segmentee_1.shape != image_segmentee_2.shape:
        return float('nan')
    
    arr1 = np.asarray(image_segmentee_1).ravel()
    arr2 = np.asarray(image_segmentee_2).ravel()

    mask = ~np.isnan(arr1) & ~np.isnan(arr2)

    arr1 = arr1[mask]
    arr2 = arr2[mask]

    std1 = arr1.std()
    std2 = arr2.std()

    if std1 == 0 or std2 == 0:
        return 1.0 if np.array_equal(arr1, arr2) else 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.corrcoef(arr1, arr2)[0, 1]

    return float(corr)

def similarite_structurelle(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 1 signifie parfait
    
    if image_segmentee_1.shape != image_segmentee_2.shape:
        return float('nan')

    score, _ = structural_similarity(image_segmentee_1.astype(float), image_segmentee_2.astype(float), data_range=1., full=True)
    return score

def vraie_detection(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray) -> float: # 1 signifie parfait
    
    ref = np.asarray(image_segmentee_1).ravel()
    seg = np.asarray(image_segmentee_2).ravel()

    if ref.shape != seg.shape:
        return float('nan')

    ref_pos = ref > 0
    seg_pos = seg > 0

    vrais_positifs = np.sum(ref_pos & seg_pos) # vrais positifs
    faux_negatifs = np.sum(ref_pos & ~seg_pos) # faux négatifs

    denom = vrais_positifs + faux_negatifs
    if denom == 0:
        return float('nan')

    return float(vrais_positifs / denom)

def affichage_liste(liste : list) -> str:

    if len(liste) == 0:
        return ""
    
    s = ""
    for i in range(len(liste) -1):
        s += str(liste[i]) + ", "

    s += str(liste[len(liste) -1]) 
    return s


# affichage des scores
def print_scores(image_segmentee_1 : ndarray, image_segmentee_2 : ndarray):

    print("Résultat des scores :")

    print(f"Distance de Hamming : {round(distance_hamming(image_segmentee_1, image_segmentee_2),3)}")
    print(f"Différence d'aire : {round(difference_aire(image_segmentee_1, image_segmentee_2),3)}")
    print(f"Fausse détection : {round(fausse_detection(image_segmentee_1, image_segmentee_2),3)}\n")

    print(f"Vraie détection : {round(vraie_detection(image_segmentee_1, image_segmentee_2),3)}")
    print(f"Corrélation : {round(score_correlation(image_segmentee_1, image_segmentee_2),3)}")
    print(f"Similarité structurelle : {round(similarite_structurelle(image_segmentee_1, image_segmentee_2),3)}")

def moyenne_scores_annees(fonction_segmentation: Callable[[MaskedArray], ndarray], annees: list[int] = [2021, 2022, 2023, 2024], mois : list[int] = np.arange(1,13), zones : list[int] = np.arange(1,9)) -> None:

    temps_execution = []

    hamming_vals = []
    diff_aire_vals = []
    fausse_vals = []

    vraie_vals = []
    corr_vals = []
    ssim_vals = []

    for annee in annees:

        print(f"Année {annee} ", end="")

        for zone in zones:

            print(f".", end="")

            dir_oasis = f'./Data/Test_zone{zone}/STATS/MeanMonthly/'
            dir_gt = f'./GroundTruth_DYN/Test_zone{zone}/'

            for m in mois:

                date = f"{annee}{m:02d}"
                img_path = premier_fichier_dossier(f"{dir_oasis}*{date}*.tif")

                if img_path is None:
                    continue

                gt_path = premier_fichier_dossier(f"{dir_gt}*{date}*.tif")

                if gt_path is None:
                    continue

                image_oasis = recuperer_image(img_path)
                image_gt = recuperer_image(gt_path).astype(int)

                start = time.time()
                image_seg = fonction_segmentation(image_oasis)
                end = time.time()

                temps_execution.append(end - start)

                hamming_vals.append(distance_hamming(image_seg, image_gt))
                diff_aire_vals.append(difference_aire(image_seg, image_gt))
                fausse_vals.append(fausse_detection(image_seg, image_gt))

                vraie_vals.append(vraie_detection(image_seg, image_gt))
                corr_vals.append(score_correlation(image_seg, image_gt))
                ssim_vals.append(similarite_structurelle(image_seg, image_gt))

        print()

    temps_execution_moyen = round(np.mean(temps_execution),3)

    noms_scores_moyens = [
        "Distance de Hamming\nmoyenne",
        "Différence d'aire\nmoyenne",
        "Fausse détection\nmoyenne",
        "Vraie détection\nmoyenne",
        "Corrélation\nmoyenne",
        "Similarité structurelle\nmoyenne"
    ]

    scores_moyens = [
        round(np.nanmean(hamming_vals), 3),
        round(np.nanmean(diff_aire_vals), 3),
        round(np.nanmean(fausse_vals), 3),
        round(np.nanmean(vraie_vals), 3),
        round(np.nanmean(corr_vals), 3),
        round(np.nanmean(ssim_vals), 3)
    ]

    for i in range(len(noms_scores_moyens)):
        noms_scores_moyens[i] = noms_scores_moyens[i] + "\n" + str(scores_moyens[i])  

    plt.figure(figsize=(18, 7))
    plt.bar(noms_scores_moyens, scores_moyens, color=['red', 'red', 'red', 'blue', 'blue', 'blue'])
    plt.ylabel("Score moyen")
    plt.title(f"Scores moyens sur les années {affichage_liste(annees)}\nmois de {affichage_liste([MOIS_ANNEE[m - 1] for m in mois])}\ndans les zones {affichage_liste(zones)}\nTemps d'éxécution moyen : {temps_execution_moyen} secondes")
    plt.savefig(f"{DOSSIER_SORTIE}/{DOSSIER_SCORES}/score {fonction_segmentation.__name__}.png", dpi=150)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def scores_0(pred, gt):

    scores = [
        distance_hamming(pred, gt),
        difference_aire(pred, gt),
        fausse_detection(pred, gt)
    ]

    return np.nanmean(scores)

def scores_1(pred, gt):

    scores = [
        vraie_detection(pred, gt),
        score_correlation(pred, gt),
        similarite_structurelle(pred, gt)
    ]

    return np.nanmean(scores)

def graphe_scores(fonction_segmentation: Callable[[MaskedArray], ndarray],
                  zones : list[int] = np.arange(1,9),
                  mois : list[int] = np.arange(1,13),
                  annees: list[int] = [2021, 2022, 2023, 2024],
                  mean_monthly: bool = True,
                  resolution: int = 250,
                  figsize: tuple[int, int] = (16, 14)) -> None:
    
    x = np.arange(len(annees) * len(mois))
    fig, axes = plt.subplots(len(zones), 1, figsize=figsize, sharex=True)

    for i, zone in enumerate(zones):
        
        print(f"Zone {zone}", end="")

        dir_oasis = f'./Data/Test_zone{zone}/{"STATS/MeanMonthly" if mean_monthly else "OASIS"}/'
        dir_gt = f'./GroundTruth_DYN/Test_zone{zone}/'
        ax = axes[i]
        s1, s0 = [], []

        for annee in annees:

            print(f".", end="")

            for m in mois:
                
                date = f"{annee}{m:02d}"
                img_path = premier_fichier_dossier(f"{dir_oasis}*{date}*.tif")
                gt_path = premier_fichier_dossier(f"{dir_gt}*{date}*.tif")

                if img_path is None or gt_path is None:
                    continue

                image = recuperer_image(img_path)
                image_gt = image_reference_binaire(recuperer_image(gt_path))
                image_seg = fonction_segmentation(image)

                s1.append(scores_1(image_seg, image_gt))
                s0.append(scores_0(image_seg, image_gt))

        print()

        ax.plot(x, s1, marker="o", color="blue", label="moyenne des scores qui tendent vers 1")
        ax.plot(x, s0, marker="o", color="red", label="moyenne des scores qui tendent vers 0")
        ax.set_ylim(0, 1)
        
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
        
        elif i == (len(zones) -1):

            x_str = []
            for annee in annees:
            
                for m in mois:
                    x_str.append(f"{MOIS_ANNEE[m - 1]} {annee}")

            x_positions = [j for j in range(len(annees) * len(mois))]

            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_str, ha="right", rotation = 45)
        
        ax.set_ylabel(f"Zone {zone}\n", fontsize=9)    


    fig.suptitle(f"{fonction_segmentation.__name__} — Scores mensuels par zone et par année", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"{DOSSIER_SORTIE}/{DOSSIER_GRAPHES_SCORES}/graphe_scores {fonction_segmentation.__name__}.png", dpi=resolution)
    plt.show()


if __name__ == "__main__": # tests


    def segmentation_test(image : MaskedArray) -> ndarray:

        image = np.nan_to_num(image, nan=0)
        image[image < 0.5] = 0

        return image

    image_ref = recuperer_images(zone = 2, selected_dates=['20210816'])[0]
    image_segmentee = segmentation_test(image_ref)


    #moyenne_scores_annees(segmentation_test,annees=[2024,2021,2022],zones=[4,1,2],mois=[5,2,11])
    #graphe_scores(segmentation_test,annees=[2024,2021,2022],zones=[4,1,2],mois=[5,2,11])
    graphe_scores(segmentation_test)