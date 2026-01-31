import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from numpy import ndarray
from numpy.ma import MaskedArray
from fonctions_images import *
from fonctions_scores import *


DOSSIER_GRAPHES_SEGMENTATION = "graphes segmentation"


# affichage des tests
def test_segmentation(image_ref : tuple[MaskedArray, MaskedArray | None], fonction_segmentation : Callable[[MaskedArray], ndarray]) -> None:

    image = image_ref[0]
    image_reference = image_ref[1]

    start = time.time()
    image_segmentee = fonction_segmentation(image)
    end = time.time()

    print(f"fonction {fonction_segmentation.__name__} terminée en : {round(end - start,3)} secondes")
    print_scores(image_segmentee, image_reference)

    if image_reference is not None :
        _, plots = plt.subplots(1, 3, figsize=(14, 6))

    else :
        _, plots = plt.subplots(1, 2, figsize=(10, 6))

    plot_oasis = plots[0]
    plot_segmente = plots[1]

    im = plot_oasis.imshow(image, cmap=INDICATEUR_OASIS, origin='upper')
    plt.colorbar(im, ax=plot_oasis)
    plot_oasis.set_title("Image au format OASIS")
    plot_oasis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plot_segmente.imshow(image_segmentee,cmap=INDICATEUR_BINAIRE , origin='upper')
    plot_segmente.set_title(f"Segmentation de l'image avec\n{fonction_segmentation.__name__}")
    plot_segmente.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if image_reference is not None :
        plot_reference = plots[2]
        plot_reference.imshow(image_reference,cmap=INDICATEUR_BINAIRE , origin='upper')
        plot_reference.set_title(f"Image de référence")
        plot_reference.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.show()

def tests_segmentation(fonction_segmentation : Callable[[MaskedArray], ndarray], annee : int = 2021, mois : list[int] = np.arange(1,13), zones : list[int] = np.arange(1,9), mean_monthly : bool = True, resolution : int = 300) -> None:   

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
            image_segmentee = fonction_segmentation(image)
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


if __name__ == "__main__": # tests


    def segmentation_test(image : MaskedArray) -> ndarray:

        image = np.nan_to_num(image, nan=0)
        image[image < 0.5] = 0

        return image


    image_ref = recuperer_images(zone = 2, selected_dates=['20210816'])[0]
    image_segmentee = segmentation_test(image_ref)

    #test_segmentation(image_ref, segmentation_test)
    tests_segmentation(segmentation_test)
    tests_segmentation(segmentation_test,zones=[4,1,2],mois=[5,2,11])