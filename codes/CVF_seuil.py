from scipy.ndimage import gaussian_filter
from CV_scikit_modifie import chan_vese
from skimage.exposure import rescale_intensity
from numpy import ndarray
import numpy as np
from fonctions_images import *
from fonctions_tests import *


def segmentation_chan_vese_seuil(
    image: MaskedArray,
    sigma : int = 2,
    mu : float = 0.07,
    lambda1 : float = 1,
    lambda2 : float = 1,
    max_num_iter : int = 200,
    tol : float = 5e-4) -> ndarray:
    
    image_no_nan = image.filled(np.nan)
    image_no_nan = np.nan_to_num(image_no_nan, nan=np.nanmean(image_no_nan))

    image_gray = image_no_nan.astype(float)

    image_filtered = gaussian_filter(image_gray, sigma=sigma)

    image_normalisee = rescale_intensity(image_filtered, in_range=(np.min(image_filtered),np.max(image_filtered)),out_range=(0,1))

    image_segmentee = chan_vese(

        image_normalisee,
        mu=mu,
        lambda1=lambda1,
        lambda2=lambda2,
        max_num_iter=max_num_iter,
        tol=tol,
        dt=0.5,
        init_level_set="threshold",
        extended_output=False

    )

    image_segmentee[np.isnan(image)] = 0

    return image_segmentee



if __name__ == "__main__": # tests

    #image_ref = recuperer_images(zone=2, selected_dates=['202108'])[0]
    #test_segmentation(image_ref, segmentation_chan_vese_seuil)
    
    tests_segmentation(segmentation_chan_vese_seuil, annee=2021)
    moyenne_scores_annees(segmentation_chan_vese_seuil, annees=[2021,2022])
    graphe_scores(segmentation_chan_vese_seuil, annees=[2021,2022])