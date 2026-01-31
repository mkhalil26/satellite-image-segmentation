
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.exposure import rescale_intensity
from seuilF import segmentation_seuillage_fixe
from sklearn.ensemble import RandomForestClassifier
from fonctions_RF import *


def extract_features(image : MaskedArray, colocated : MaskedArray, mois : int, zone : int) -> ndarray:
    
    image_no_nan = image.filled(np.nan)
    image_no_nan = np.nan_to_num(image_no_nan, nan=0)
    image_gray = image_no_nan.astype(float)

    image_filtered = gaussian_filter(image_gray, sigma=2)
    image_normalisee = rescale_intensity(image_filtered, in_range=(np.min(image_filtered),np.max(image_filtered)),out_range=(0,1))

    region_locale = 20
    moyenne_locale = uniform_filter(image_gray, size=region_locale)
    variance_locale = uniform_filter(image_gray**2, size=region_locale) - moyenne_locale**2
    
    seuil = segmentation_seuillage_fixe(image)

    image_mois = np.full(image.shape, (mois-1) / 11)
    image_zone = np.full(image.shape, (zone-1) / 7)

    features = np.stack([
        
        colocated,
        image_mois,
        image_zone,
        image_normalisee,
        variance_locale,
        seuil

    ], axis=-1)

    return features.reshape(-1, features.shape[-1])

def train_random_forest(
    images : list[list[list[MaskedArray]]],
    masks : list[list[list[MaskedArray]]],
    colocated : list[list[MaskedArray]],
    nb_arbres : int = 20,
    profondeur_max_arbre : int = 10,
    pixels_min_feuilles : int = 1,
    nb_threads : int = 8) -> RandomForestClassifier:

    x_train = []
    y_train = []

    for zone in range(1,9):

        for mois in range(1,13):

            for img, mask in zip(images[zone -1][mois -1], masks[zone -1][mois -1]):

                x = extract_features(img, colocated[zone -1][mois -1], mois, zone)
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

def predict_segmentation(model : RandomForestClassifier , image : MaskedArray, colocated : MaskedArray, mois : int, zone : int) -> ndarray:

    x = extract_features(image, colocated, mois, zone)
    y_pred = model.predict(x)
    
    return y_pred.reshape(image.shape)

def load_training_data(annees : list[int] = [2021]) -> tuple[list[list[list[MaskedArray]]], list[list[list[MaskedArray]]]]:

    images_x = []
    images_y = []

    for zone in range(1, 9):

        dir_y = f'./GroundTruth_DYN/Test_zone{zone}/'
        dir_x = f'./Data/Test_zone{zone}/STATS/MeanMonthly/'
        images_zone_x = []
        images_zone_y = []

        for mois in range(1,13):

            images_mois_x = []
            images_mois_y = []

            for annee in annees:

                date = f"{annee}{mois:02d}"
                chemin_image_x = premier_fichier_dossier(f"{dir_x}*{date}*.tif")
                chemin_image_y = premier_fichier_dossier(f"{dir_y}*{date}*.tif")
                
                if chemin_image_x is not None and chemin_image_y is not None :
                    
                    images_mois_x.append(recuperer_image(chemin_image_x))
                    images_mois_y.append(image_reference_binaire(recuperer_image(chemin_image_y)))
            
            images_zone_x.append(images_mois_x)
            images_zone_y.append(images_mois_y)

        images_x.append(images_zone_x)
        images_y.append(images_zone_y)

    return images_x, images_y

def load_colocated_data() -> list[list[MaskedArray]]:
    
    annee = 2024
    images = [[None for _ in range(12)] for _ in range(8)]
    
    for zone in range(1, 9):
        
        dir = f"./NDWI_colocalise_avec_sar/Colocated_Images/Zone{zone}"
        for mois in range(1,13):

            chemin = premier_fichier_dossier(f"{dir}/*{annee}-{mois:02d}*")

            if chemin is not None :
                images[zone -1][mois -1] = recuperer_image(chemin)
    
    for zone in range(8):
        mois_valides = [i for i, img in enumerate(images[zone]) if img is not None]

        if not mois_valides:
            continue

        for mois in range(12):
            if images[zone][mois] is None:
                
                plus_proche = min(mois_valides, key=lambda x: abs(x - mois))
                images[zone][mois] = images[zone][plus_proche].copy()

    return images

def entrainer_modele(colocated : list, nom : str, annees : list[int] = [2023,2024], nb_arbres : int = 20, profondeur_max_arbre : int = 10, pixels_min_feuilles : int = 1, nb_threads : int = 8) -> None:

    images, masks = load_training_data(annees=annees)

    print(f"images d'entraînement : {nb_elements(images)}")

    start = time.time()
    modele = train_random_forest(images, masks, colocated, nb_arbres=nb_arbres, profondeur_max_arbre=profondeur_max_arbre, pixels_min_feuilles=pixels_min_feuilles, nb_threads=nb_threads)
    end=time.time()

    print(f"temps d'entrainement {round(end-start,3)} secondes")

    save_model(modele, nom)


def segmentation_random_forest_ZM(image : MaskedArray, modele : RandomForestClassifier, colocated : list, zone : int, mois : int) -> ndarray:

        return predict_segmentation(modele, image, colocated[zone -1][mois -1], mois, zone)


if __name__ == "__main__":

    colocated = load_colocated_data()
    entrainer_modele(colocated, "modele RF ZM 2023-2024")


    modele = load_model("modele RF ZM 2023-2024")
    verify_features(modele)
    

    tests_segmentation_ZM(segmentation_random_forest_ZM, modele, colocated, annee=2021)
    moyenne_scores_annees_ZM(segmentation_random_forest_ZM, modele, colocated, annees=[2021,2022])
    graphe_scores_ZM(segmentation_random_forest_ZM, modele, colocated, annees=[2021,2022])