# Structure des Notebooks

## 1. EDA
Analyse exploratoire du dataset : distribution des classes, visualisation des images, statistiques descriptives.

## 2. Embedding
Extraction de features via deux architectures pré-entraînées :
- **ResNet-50**
- **EfficientNet-B4**

## 3. Clustering — Approche non supervisée

| Version | Description |
|---------|-------------|
| **V1** | Approche de référence — clustering GMM (DBSCAN et K-Means écartés pour performances inférieures) |
| **V2** | Optimisation par seuil de confiance sur la distance euclidienne aux centroïdes |
| **V3** | Optimisation par seuil de confiance via régression logistique (Out-Of-Fold) |
| **V4** | Optimisation avec k=4 clusters fixes et nettoyage des pseudo-labels par Random Forest (Out-Of-Fold) |

## 4. Modélisation — Approches semi-supervisées et supervisées (CNN)

Chaque version utilise les pseudo-labels produits par la version de clustering correspondante :

| Version | Pseudo-labels sources |
|---------|-----------------------|
| **V1** | Clustering V1 |
| **V2** | Clustering V2 |
| **V3** | Clustering V3 |
| **V4** | Clustering V4 |
