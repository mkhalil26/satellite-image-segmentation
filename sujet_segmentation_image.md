# Segmentation d'images satellitaires pour la détection des lacs en montagne

**Contact: Guillaume James (guillaume.james@univ-grenoble-alpes.fr)**

## Contexte

Les lacs et réservoirs jouent un rôle fondamental dans l’équilibre écologique, l’approvisionnement en eau potable et la production d’énergie hydroélectrique. Afin de cartographier leur étendue et étudier leurs variations, les images satellites de haute définition sont des outils essentiels, notamment celles des satellites Sentinel-1 et 2 de l'agence spatiale européenne. Cependant, des défis persistent pour délimiter précisément les contours des lacs de montagne par télédétection spatiale, qu'il s'agisse de mesures optiques ou radar.
Les images optiques sont restreintes par la couverture nuageuse et les ombres projetées par les reliefs ou les nuages. De plus, les variations de teintes des lacs, influencées par leur turbidité et leur milieu, rendent complexe l’utilisation de techniques automatisées de détection. Les images radar comme celles du Radar à Synthèse d’Ouverture (SAR) de Sentinel-1 sont quant à elles insensibles aux nuages, mais souffrent de distorsions radar en terrains accidentés, rendant difficile l’identification des surfaces d’eau.

La segmentation des images est un processus utilisé dans de nombreux domaines (comme la biomédecine, la télédétection spatiale, la communication vidéo) pour analyser les images et en extraire une information pertinente. Elle correspond à un découpage d'une image en régions disjointes et homogènes sur la base d'un ensemble de critères. Il existe de très nombreuses méthodes de segmentation, basées par exemple sur une combinaison de seuillage et filtrage, la minimisation d'une énergie associée à des régions et des contours de l'image, la décomposition en ondelettes, la segmentation de graphes, l'apprentissage automatique,... 


## Objectifs

Le but de ce projet est d'évaluer le potentiel de plusieurs méthodes de segmentation existantes pour délimiter les lacs de montagne sur les images SAR de Sentinel-1 et optique de Sentinel-2. Dans des recherches récentes, différentes méthodes de segmentation ont été testées sur ces images pour isoler les signaux neige (Guiot et al. 2023, Kilias et al. 2025). Pour la segmentation des lacs, nous allons nous baser sur des indices dérivés des observations satellites qui ont montré leur efficacité pour faciliter l'identification des lacs (Karbou et al. 2025).

Pour ces tests, la zone d'étude sera centrée sur le secteur de la Bérarde qui a connu en 2024 un événement tragique avec la vidange du lac supraglaciaire du glacier de Bonne Pierre. Cette zone d'étude comprend un grand nombre de lacs de montagne de différentes formes et superficies ainsi que des lacs glaciaires.

<p align="center">
<img src="Zone_Test_Berarde.jpg" width="800px"/>  
</p>
<p align="center"> 
Délimitation de la zone d'étude centrée sur le secteur de la Bérarde.
</p>

## Etapes du projet

<ol>
  <li>Revue bibliographique de méthodes de segmentation existantes avec examen des différences entre ces méthodes, du point de vue des notions mathématiques mises en jeu et de leur intérêt pour la télédétection. On pourra consulter par exemple l'article de Guiot et al. 2023 dans lequel un échantillon de méthodes de segmentation est décrit et testé (des articles de revue plus exhaustifs sont également indiqués dans la bibliographie).
Les équipes se familiariseront également avec l'utilisation de librairies adaptées comme
scikit-image et scikit-learn.</li>
<br/>

  <li>Sélection de quelques méthodes de segmentation pour les tester sur des images satellites d'intérêt. On se focalisera dans un premier temps sur des images acquises sur les Alpes en périodes estivales. L'enjeu de cette étape est de tester différentes approches de segmentation et d'évaluer leur pertinence pour détecter les contours de différents lacs d'altitude (superficies et tailles variables). Nous examinerons notamment la sensibilité des résultats aux choix d'images optiques ou radar, la fusion des deux sources si la méthode de segmentation le permet, et l'apport de la prise en compte des textures.</li>
<br/>

  <li>Tests de détection des lacs dans des situations complexes: milieux glaciaires, périodes pendant lesquelles les lacs peuvent être en partie gelés.</li>
</ol>


## Données

Dans ce projet les équipes analyseront des images SAR de Sentinel-1 (résolution de 20 mètres) et des images optiques de Sentinel-2 (résolution de 10 à 20 mètres selon les bandes).


## Références bibliographiques

V. Dey, Y. Zhang, M. Zhong (2010), [A Review on Image Segmentation Techniques with Remote Sensing Perspective](https://www.isprs.org/proceedings/XXXVIII/part7/a/pdf/31_XXXVIII-part7A.pdf) ISPRS TC VII Symposium – 100 Years ISPRS, Vol. XXXVIII, Part 7A.

A. Guiot, F. Karbou, G. James, P. Durand (2023) [Insights into Segmentation Methods Applied to Remote Sensing SAR Images for Wet Snow Detection.](https://www.mdpi.com/2076-3263/13/7/193) Geosciences, 13, 193.

F. Karbou, G. James, A. Mauss, P. Verry, B. Demolis, R., Martin (2025) Unlocking lakes in all terrain with Sentinel-1 SAR imagery, soumis.

K. Kilias, G. James, F. Karbou, C. Turbé, A. Mauss and G. Eynard-Bontemps (2025), [Snowline Altitude Estimation in Sentinel-1 Satellite Images Using an Extended Chan-Vese Method: Proof of Concept and Prototype of Fusion with Optical Data.](https://ieeexplore.ieee.org/document/11142876) IEEE Transactions on Geoscience and Remote Sensing, volume 63, pages 1-13.

A. Yu, Y. Quan, R. Yu, W. Guo, X. Wang, D. Hong, H. Zhang, J. Chen, Q. Hu, P. He (2023), [Deep Learning Methods for Semantic Segmentation in Remote Sensing with Small Data: A Survey](https://www.mdpi.com/2072-4292/15/20/4987) Remote Sensing, 15(20), 4987.

X. Yuan, J. Shi, L. Gu (2021), [A review of deep learning methods for semantic segmentation of remote sensing imagery](https://www.sciencedirect.com/science/article/abs/pii/S0957417420310836) Expert Systems with Applications 169, 114417.

Librairies Python :

-  [scikit-learn](https://scikit-learn.org/)

-  [scikit-image](https://scikit-image.org/)
