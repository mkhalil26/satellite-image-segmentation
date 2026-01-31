import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from fonctions_images import *
from fonctions_tests import *


DOSSIER_ENTRAINEMENT = "modeles RF"


def save_model(model : RandomForestClassifier, filemane : str) -> None:

    joblib.dump(model, f"{DOSSIER_SORTIE}/{DOSSIER_ENTRAINEMENT}/{filemane}.pkl")

def load_model(filemane : str) -> RandomForestClassifier:

    return joblib.load(f"{DOSSIER_SORTIE}/{DOSSIER_ENTRAINEMENT}/{filemane}.pkl")

def verify_features(model : RandomForestClassifier) -> None:

    for i, imp in enumerate(model.feature_importances_):

        print(f"Importance feature {i+1} : {round(imp,3)}")

def nb_elements(liste : list) -> int :
    
    n = 0
    for elem in liste:
        
        if isinstance(elem, list):
            n += nb_elements(elem)
        
        else:
            n += 1
    
    return n



def tests_segmentation_ZM(
        fonction_segmentation : Callable[[MaskedArray, RandomForestClassifier, list, int, int], ndarray],
        modele : RandomForestClassifier,
        colocated : list = [],
        annee : int = 2021,
        mois : list[int] = np.arange(1,13),
        zones : list[int] = np.arange(1,9),
        mean_monthly : bool = True,
        resolution : int = 300) -> None:   

    if mean_monthly :
        fig, plots = plt.subplots(3 * len(zones),len(mois),figsize=(10, 14))

    else :
        fig, plots = plt.subplots(2 * len(zones),len(mois),figsize=(12, 8))
    
    plt.suptitle(f"Segmentation des Images pour l'Année {annee}\nimage non segmentée, image segmentée, image de référence", fontsize=14, fontweight='bold')
    temps_execution = []

    for x,zone in enumerate(zones):

        print(f"segmentation Zone {zone} ", end="")
        dir = f'./Data/Test_zone{zone}/{"STATS/MeanMonthly" if mean_monthly else "OASIS"}/'

        if mean_monthly :
            dir_mean_monthly = f"./GroundTruth_DYN/Test_zone{zone}/"

        for y,m in enumerate(mois):
            
            print(f".", end="")

            chemin_image = None
            chemin_image_ref = None
            date = f"{annee}{m:02d}"
            chemin_image = premier_fichier_dossier(f"{dir}*{date}*.tif")
            
            if mean_monthly :

                chemin_image_ref = premier_fichier_dossier(f"{dir_mean_monthly}*{date}*.tif")
  
            image = recuperer_image(chemin_image)

            start = time.time()
            image_segmentee = fonction_segmentation(image, modele, colocated, zone, m)
            end = time.time()
            temps_execution.append(end-start)
            
            if chemin_image_ref is None :
                plot_oasis = plots[x * 2, y]
                plot_segmente = plots[x * 2 + 1, y]

            else :
                plot_oasis = plots[x * 3, y]
                plot_segmente = plots[x * 3 + 1, y]
                plot_ref = plots[x * 3 + 2, y]

                plot_ref.imshow(image_reference_binaire(recuperer_image(chemin_image_ref)), cmap=INDICATEUR_BINAIRE, origin='upper')
                plot_ref.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                
            plot_oasis.imshow(image, cmap=INDICATEUR_OASIS, origin='upper')
            plot_oasis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
            plot_segmente.imshow(image_segmentee, cmap=INDICATEUR_BINAIRE, origin='upper')
            plot_segmente.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            
        print()

    dx = len(zones)
    for x,zone in enumerate(zones):    
        fig.text(0.02, 0.94 * (1 - (x / dx) - (1 / (2 * dx))), f"Zone {zone}", ha='center', va='center', rotation='vertical', fontsize=9, fontweight='bold')
    
    dy = len(mois)
    for y,m in enumerate(mois):
        fig.text((y / dy) + (1 / (2 * dy)), 0.925, f"{MOIS_ANNEE[m - 1]}", ha='center', va='center', fontsize=9, fontweight='bold')

    print(f"temps d'éxécution moyen de {fonction_segmentation.__name__} : {round(np.mean(temps_execution),3)} secondes")
    print("affichage et sauvegarde du graphique")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{DOSSIER_SORTIE}/{DOSSIER_GRAPHES_SEGMENTATION}/{fonction_segmentation.__name__}_{annee}.png", dpi=resolution)
    plt.show()

def moyenne_scores_annees_ZM(
        fonction_segmentation: Callable[[MaskedArray, RandomForestClassifier, list, int, int], ndarray],
        modele : RandomForestClassifier,
        colocated : list = [],
        annees: list[int] = [2021, 2022, 2023, 2024],
        mois : list[int] = np.arange(1,13),
        zones : list[int] = np.arange(1,9)) -> None:

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
                image_seg = fonction_segmentation(image_oasis, modele, colocated, zone, m)
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

def graphe_scores_ZM(
        fonction_segmentation: Callable[[MaskedArray, RandomForestClassifier, list, int, int], ndarray],
        modele : RandomForestClassifier,
        colocated : list = [],
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
                image_seg = fonction_segmentation(image, modele, colocated, zone, m)

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
