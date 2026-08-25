# Avancement — Notebook Semi-Supervisé Optimisé

**Dernière mise à jour** : 2026-04-17
**Notebook** : [`Semi_Supervise_Optimise.ipynb`](Semi_Supervise_Optimise.ipynb) (35 cellules, 12 md + 23 code)

## Statut : NOTEBOOK ÉCRIT + CODE REVIEW APPLIQUÉE, PRÊT À EXÉCUTION

## Choix validés avec Kevin
| # | Question | Décision |
|---|----------|----------|
| 1 | Split val/test | **Identique à Modelisations** : 70 train + 30 val (`random_state=42`, stratify) |
| 2 | IMG_SIZE | **224** (cohérent embeddings, ~55 min GPU) |
| 3 | Backbone | **EfficientNet-B4 ImageNet, gel total sauf classifier** (cohérent Phase 2/3) |
| 4 | tqdm | Ajouté via `uv add tqdm` (4.67.3) |
| 5 | Comparaison anciens modèles | MyCNN redéfini dans le notebook + reload des 3 .pth pour eval apple-to-apple |

## Sanity checks effectués
- CUDA disponible : OK (RTX 5070)
- Splits identiques à Modelisations : 1406 unlabeled, 70 train, 30 val
- Chemins images résolus correctement (PROJECT_ROOT relatif)
- EfficientNet-B4 build : 3 586 params entraînables / 17.5M total (classifier head uniquement)
- Forward pass GPU OK : input (4,3,224,224) → output (4,2)

## Structure du notebook

| Section | Cellules | Contenu |
|---------|----------|---------|
| 1. Setup | 1 code | Imports, seeds, DEVICE, paths |
| 2. Data | 1 code | Chargement CSVs + split identique à Modelisations |
| 3. Transforms | 2 code | weak / strong / inference + Cutout + viz |
| 4. Datasets | 2 code | LabeledMRI, UnlabeledMRITwin, UnlabeledMRISingle + DataLoaders |
| 5. Backbone + utils | 3 code | build_efficientnet_b4, evaluate, predict_loader, GradCAM, save/load_checkpoint |
| 6. Baseline EffNet-B4 | 2 code | train_supervised + eval + Grad-CAM (30 epochs, AdamW lr=1e-3) |
| 7. Pseudo-Labeling | 3 code | generate_pseudo_labels + boucle 5 iter × 5 epochs (τ=0.95, lr=5e-5) + eval |
| 8. Mean Teacher | 3 code | EMA helpers + training 30 epochs (α=0.999, rampup=10) + eval |
| 9. FixMatch | 2 code | Training 30 epochs (SGD lr=0.03, τ=0.95, λ_u=1.0) + eval |
| 10. Comparaison | 4 code | MyCNN + reload .pth + table + bar plots + Grad-CAM comparatif |
| 11. Conclusion | 1 md | Points à analyser après run |

## Ordre d'exécution recommandé
Le notebook s'exécute linéairement. Les méthodes SSL (sections 7, 8, 9) **dépendent** du checkpoint `baseline_supervised.pth` produit en section 6 (warm start). La section 10 dépend de toutes les précédentes.

## Estimation temps GPU (RTX 5070, batch labeled=16, batch unlabeled=64)
- Section 6 : Baseline supervisé — ~2 min (30 epochs sur 4 batches/epoch)
- Section 7 : Pseudo-Labeling — ~5 min
- Section 8 : Mean Teacher — ~15-20 min (30 epochs × ~4 batches × forward sur 16+64+64)
- Section 9 : FixMatch — ~15-20 min (30 epochs × forward concaténé)
- Section 10 : eval anciens modèles + plots — <2 min
- **Total : ~45 min**

## Checkpoints qui seront créés (dans `checkpoints/`)
- `baseline_supervised.pth` — meilleur sur val durant training baseline
- `pseudo_labeling_best.pth` — meilleur sur val parmi 5 itérations
- `mean_teacher_best.pth` — **Teacher** EMA au best val
- `fixmatch_best.pth` — best val
- `history_pseudo_labeling.csv`, `history_mean_teacher.csv`, `history_fixmatch.csv` — losses/métriques par epoch
- `comparison_results.csv` — tableau récap final

## Commandes utiles
```powershell
cd c:\Users\Kevin\projects\OC_P7
uv run jupyter lab
# Puis ouvrir Optimisation/Semi_Supervise_Optimise.ipynb et exécuter "Run all"
```

## Code review appliquée (sous-agent `code-reviewer`)

**6 fixes prioritaires appliqués via `NotebookEdit`** :

| # | Sévérité | Cellule | Fix |
|---|----------|---------|-----|
| 1 | CRITICAL | `aa08986d6e2e` | Helper `freeze_backbone_bn(model)` ajouté — force BN du backbone gelé en eval mode pour éviter la dérive des running stats ImageNet |
| 2 | CRITICAL | `9ad727383b07`, `0a68d17affcc`, `719542e22ecb`, `f453f7a49827` | Appel `freeze_backbone_bn()` après chaque `.train()` dans baseline / pseudo-labeling / mean teacher / fixmatch |
| 3 | CRITICAL | `f453f7a49827` | FixMatch : `logits_w.detach()` explicite pour la génération des pseudo-labels |
| 4 | HIGH | `719542e22ecb` | Mean Teacher : supervisé utilise `loader_train_strong` (cohérent avec baseline) au lieu de `loader_train_weak` |
| 5 | HIGH | `7252c60a7a88` | `run_gradcam_analysis` : `try/finally` + `model.zero_grad(set_to_none=True)` pour cleanup |
| 6 | MEDIUM | `41a319ff5420` | Suppression `drop_last=True` sur les loaders labelisés (évite de perdre 6 images / epoch sur 70) |

Validation post-fix : tous les checks passent (helper défini + appelé partout, detach présent, MT utilise strong loader).

## TODO restant
- [ ] Exécuter le notebook end-to-end et vérifier les résultats (~45 min RTX 5070)
- [ ] Compléter la section 11 conclusion avec les vrais chiffres
- [ ] (Si SSL n'améliore pas) Tester variante avec backbone partiellement dégelé (`features[5:]`)
