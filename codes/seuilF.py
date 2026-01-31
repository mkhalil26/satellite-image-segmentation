from scipy.ndimage import gaussian_filter
import numpy as np
from fonctions_images import *
from fonctions_tests import *


def segmentation_seuillage_fixe(image: MaskedArray, seuil: float = 0.2, sigma: int = 2) -> ndarray:
    
    image_no_nan = image.filled(np.nan)
    image_no_nan = np.nan_to_num(image_no_nan, nan=0)

    image_gray = image_no_nan.astype(float)

    image_filtered = gaussian_filter(image_gray, sigma=sigma)

    segmentation_result_boolean = image_filtered > seuil

    segmentation_result = segmentation_result_boolean.astype(int)

    return segmentation_result



if __name__ == "__main__": # tests

    #image_ref = recuperer_images(zone=5, selected_dates=['202102'])[0]
    #test_segmentation(image_ref, segmentation_seuillage_fixe)
    
    tests_segmentation(segmentation_seuillage_fixe, annee=2021)
    moyenne_scores_annees(segmentation_seuillage_fixe, annees=[2021,2022])
    graphe_scores(segmentation_seuillage_fixe, annees=[2021,2022])