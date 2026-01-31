
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.exposure import rescale_intensity
from seuilF import segmentation_seuillage_fixe
from sklearn.ensemble import RandomForestClassifier
from fonctions_RF import *



def extract_features(image : MaskedArray, colocated : MaskedArray) -> ndarray:
    
    image_no_nan = image.filled(np.nan)
    image_no_nan = np.nan_to_num(image_no_nan, nan=0)
    image_gray = image_no_nan.astype(float)

    image_filtered = gaussian_filter(image_gray, sigma=2)
    image_normalisee = rescale_intensity(image_filtered, in_range=(np.min(image_filtered),np.max(image_filtered)),out_range=(0,1))

    region_locale = 20
    moyenne_locale = uniform_filter(image_gray, size=region_locale)
    variance_locale = uniform_filter(image_gray**2, size=region_locale) - moyenne_locale**2

    seuil = segmentation_seuillage_fixe(image)

    features = np.stack([
        
        colocated,
        image_normalisee,
        variance_locale,
        seuil

    ], axis=-1)

    return features.reshape(-1, features.shape[-1])

def train_random_forest(
    images : list[MaskedArray],
    masks : list[MaskedArray],
    colocated : list[MaskedArray],
    nb_arbres : int = 20,
    profondeur_max_arbre : int = 10,
    pixels_min_feuilles : int = 1,
    nb_threads : int = 8) -> RandomForestClassifier:

    x_train = []
    y_train = []

    for img, mask, col in zip(images, masks, colocated):

        x = extract_features(img, col)
        y = mask.filled(0).astype(int).reshape(-1)

        x_train.append(x)
        y_train.append(y)
        
    x_train = np.vstack(x_train)
    y_train = np.hstack(y_train)

    print(f"Taille du dataset d'entraînement : {x_train.shape[0]}")

    model = RandomForestClassifier(
        n_estimators=nb_arbres,
        max_depth=profondeur_max_arbre,
        min_samples_leaf=pixels_min_feuilles,
        random_state=0,
        n_jobs=nb_threads,
        verbose=2
    )
    model.fit(x_train, y_train)

    return model

def predict_segmentation(model : RandomForestClassifier , image : MaskedArray, colocated : MaskedArray) -> ndarray:

    x = extract_features(image, colocated)
    y_pred = model.predict(x)
    
    return y_pred.reshape(image.shape)

def load_data(mois : list[int] = np.arange(1,13), zones : list[int] = np.arange(1,9)) -> tuple[list[MaskedArray], list[MaskedArray], list[MaskedArray]]:
    
    annee = 2024
    images = []
    images_mask = []
    images_colocated = []
    
    for zone in zones:
        
        dir = f"./Data/Test_zone{zone}/STATS/MeanMonthly/"
        dir_mask = f'./GroundTruth_DYN/Test_zone{zone}/'
        dir_colocated = f"./NDWI_colocalise_avec_sar/Colocated_Images/Zone{zone}"

        for m in mois:
            
            chemin = premier_fichier_dossier(f"{dir}/*{annee}{m:02d}*")
            chemin_mask = premier_fichier_dossier(f"{dir_mask}/*{annee}{m:02d}*")
            chemin_colocated = premier_fichier_dossier(f"{dir_colocated}/*{annee}-{m:02d}*")

            if chemin_colocated is not None and chemin is not None and chemin_mask is not None:
                images.append(recuperer_image(chemin))
                images_mask.append(image_reference_binaire(recuperer_image(chemin_mask)))
                images_colocated.append(recuperer_image(chemin_colocated))

    return images, images_mask, images_colocated

def entrainer_modele(nom : str, mois : list[int] = np.arange(1,13), zones : list[int] = np.arange(1,9), nb_arbres : int = 20, profondeur_max_arbre : int = 10, pixels_min_feuilles : int = 1, nb_threads : int = 8) -> None:

    images, masks, colocated = load_data(mois=mois, zones=zones)

    print(f"images d'entraînement : {nb_elements(images)}")

    start = time.time()
    modele = train_random_forest(images, masks, colocated, nb_arbres=nb_arbres, profondeur_max_arbre=profondeur_max_arbre, pixels_min_feuilles=pixels_min_feuilles, nb_threads=nb_threads)
    end=time.time()

    print(f"temps d'entrainement {round(end-start,3)} secondes")

    save_model(modele, nom)


def segmentation_random_forest_C(image : MaskedArray, modele : RandomForestClassifier, colocated : MaskedArray) -> ndarray:

    return predict_segmentation(modele, image, colocated)


def tests_segmentation_C(
        fonction_segmentation : Callable[[MaskedArray, RandomForestClassifier, MaskedArray], ndarray],
        modele : RandomForestClassifier,
        annee : int = 2024,
        mois : list[int] = np.arange(1,13),
        zones : list[int] = np.arange(1,9),
        resolution : int = 300) -> None:   

    fig, plots = plt.subplots(3 * len(zones),len(mois),figsize=(10, 14))
    
    plt.suptitle(f"Segmentation des Images pour l'Année {annee}\nimage non segmentée, image segmentée, image de référence", fontsize=14, fontweight='bold')
    temps_execution = []

    for x,zone in enumerate(zones):

        dir = f'./Data/Test_zone{zone}/STATS/MeanMonthly/'
        dir_mean_monthly = f"./GroundTruth_DYN/Test_zone{zone}/"
        dir_colocated = f"./NDWI_colocalise_avec_sar/Colocated_Images/Zone{zone}"

        for y,m in enumerate(mois):

            chemin_image = None
            chemin_image_ref = None
            date = f"{annee}{m:02d}"
            
            chemin_image = premier_fichier_dossier(f"{dir}/*{date}*.tif")
            chemin_image_ref = premier_fichier_dossier(f"{dir_mean_monthly}/*{date}*.tif")
            chemin_colocated = premier_fichier_dossier(f"{dir_colocated}/*{annee}-{m:02d}*")

            plot_oasis = plots[x * 3, y]
            plot_segmente = plots[x * 3 + 1, y]
            plot_ref = plots[x * 3 + 2, y]
            
            if chemin_image is not None and chemin_image_ref is not None and chemin_colocated is not None:

                colocated = recuperer_image(chemin_colocated)
                image = recuperer_image(chemin_image)

                start = time.time()
                image_segmentee = fonction_segmentation(image, modele, colocated)
                end = time.time()
                temps_execution.append(end-start)
                
                plot_ref.imshow(image_reference_binaire(recuperer_image(chemin_image_ref)), cmap=INDICATEUR_BINAIRE, origin='upper')
                plot_ref.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    
                plot_oasis.imshow(image, cmap=INDICATEUR_OASIS, origin='upper')
                plot_oasis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            
                plot_segmente.imshow(image_segmentee, cmap=INDICATEUR_BINAIRE, origin='upper')
                plot_segmente.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            
            else:
                plot_oasis.axis("off")
                plot_segmente.axis("off")
                plot_ref.axis("off")

            
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

def moyenne_scores_annees_C(
    fonction_segmentation: Callable[[MaskedArray, RandomForestClassifier, MaskedArray], ndarray],
    modele : RandomForestClassifier,
    annees: list[int] = [2024],
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
            dir_colocated = f"./NDWI_colocalise_avec_sar/Colocated_Images/Zone{zone}"

            for m in mois:

                date = f"{annee}{m:02d}"
                img_path = premier_fichier_dossier(f"{dir_oasis}*{date}*.tif")

                if img_path is None:
                    continue

                gt_path = premier_fichier_dossier(f"{dir_gt}*{date}*.tif")

                if gt_path is None:
                    continue

                colocated_path = premier_fichier_dossier(f"{dir_colocated}/*{annee}-{m:02d}*")

                if colocated_path is None:
                    continue

                image_oasis = recuperer_image(img_path)
                image_gt = recuperer_image(gt_path).astype(int)
                image_colocated = recuperer_image(colocated_path)

                start = time.time()
                image_seg = fonction_segmentation(image_oasis, modele, image_colocated)
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


if __name__ == "__main__":

    #entrainer_modele("modele RF C 2024", zones=np.arange(5,9), mois=np.arange(1,13))


    modele = load_model("modele RF C 2024")
    verify_features(modele)
    

    tests_segmentation_C(segmentation_random_forest_C, modele, zones=np.arange(1,9), mois=np.arange(5, 11))
    moyenne_scores_annees_C(segmentation_random_forest_C, modele, zones=np.arange(1,9), mois=np.arange(5, 11))