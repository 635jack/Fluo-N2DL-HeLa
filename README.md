# 🔬 Segmentation et Tracking de Cellules HeLa par CNN

Pipeline complet de segmentation d'instances et tracking de cellules HeLa utilisant un reseau U-Net pour le Cell Tracking Challenge.

## 📊 Dataset

**Fluo-N2DL-HeLa** - Cellules HeLa avec marqueur nucleaire H2b-GFP

- **Microscope**: Olympus IX81
- **Objectif**: Plan 10x/0.4
- **Resolution spatiale**: 0.645 x 0.645 µm/pixel
- **Resolution temporelle**: 30 minutes entre frames
- **Source**: [Cell Tracking Challenge](https://celltrackingchallenge.net/2d-datasets/)

## 🎯 Approche

### 1. Segmentation par U-Net

**Architecture**: U-Net avec encodeur-decodeur
- **Entree**: Images grayscale (1 canal)
- **Sortie**: 3 classes (background, cellule, bordure)
- **Base filters**: 32
- **Total parametres**: ~7.7M

**Strategie 3 classes**:
- Classe 0: Background
- Classe 1: Interieur des cellules (apres erosion)
- Classe 2: Bordures entre cellules → permet de separer les instances qui se touchent

### 2. Post-traitement

**Watershed pour separation d'instances**:
- Marqueurs: interieurs des cellules (classe 1 sans bordures)
- Elevation: probabilite de la classe cellule
- Nettoyage: suppression des petits objets (<150 pixels)

### 3. Tracking

**Association par IoU + distance**:
- Score combine: `iou - (distance/max_distance) * 0.3`
- Seuil IoU: 0.35
- Distance max: 80 pixels (~52 µm)

## ⚙️ Parametres de Tuning

### Entrainement

| Parametre | Valeur | Role |
|-----------|--------|------|
| Learning rate | 1e-3 | Vitesse convergence |
| Optimizer | Adam | Optimisation adaptative |
| Weight decay | 1e-5 | Regularisation L2 |
| Batch size | 2 | Limite memoire GPU |
| Epochs | 50 | Iterations |
| Class weights | [0.1, 1.0, 2.0] | Balance bg/cell/border |
| Scheduler | ReduceLROnPlateau | Adaptation LR |

### Post-traitement

| Parametre | Valeur | Role |
|-----------|--------|------|
| threshold_cell | 0.5 | Seuil probabilite cellule |
| threshold_border | 0.3 | Seuil probabilite bordure |
| min_size | 150 pixels | Taille min cellule |
| erosion_radius | 2 | Rayon pour marqueurs |

### Tracking

| Parametre | Valeur | Role |
|-----------|--------|------|
| iou_threshold | 0.35 | Seuil chevauchement |
| max_distance | 80 pixels | Distance max centroide |
| min_track_length | 5 frames | Longueur min track |

## 🚀 Installation et Utilisation

### Installation des dependances

```bash
pip install -r requirements.txt
```

### Lancement du pipeline complet

```bash
python cell_segmentation_tracking.py
```

Le script va:
1. Charger les donnees d'entrainement (Trainset/01 et 02)
2. Entrainer le modele U-Net (50 epochs)
3. Sauvegarder le meilleur modele (`best_unet_model.pth`)
4. Effectuer l'inference sur Testset/01
5. Generer les segmentations et tracks dans `Results/`

### Exploration avec Jupyter

Pour visualiser et analyser les donnees:

```bash
jupyter notebook exploration_visualization.ipynb
```

Le notebook contient:
- Exploration des donnees
- Analyse statistique des cellules
- Visualisation des ground truth
- Explication detaillee des parametres
- Visualisation des resultats

## 📁 Structure des Resultats

```
Results/
├── SEG/
│   ├── mask000.tif
│   ├── mask001.tif
│   └── ...
├── tracks.txt
└── training_curves.png
```

- **SEG/**: Masques de segmentation (format uint16, 1 label par cellule)
- **tracks.txt**: Format Cell Tracking Challenge (id, start, end, parent)
- **training_curves.png**: Courbes de loss train/val

## 📊 Metriques d'Evaluation

Pour evaluer avec le Cell Tracking Challenge:

1. **SEG**: Segmentation accuracy (IoU-based)
2. **DET**: Detection accuracy (TP, FP, FN)
3. **TRA**: Tracking accuracy (associations correctes)

## 🔧 Ameliorations Possibles

### Court terme
- ✅ Augmentation de donnees (rotations, flips)
- ✅ Tester Dice Loss ou Focal Loss
- ✅ Ajuster les seuils de post-traitement

### Long terme
- 🔄 Utiliser ResNet/EfficientNet comme encodeur
- 🎯 Ajouter des modules d'attention (Attention U-Net)
- 🌱 Detecter les divisions cellulaires
- 🔮 Kalman filter pour prediction de trajectoire
- 🌐 Graph Neural Networks pour tracking global

## 📚 References

- **U-Net**: Ronneberger et al. (2015) - "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **Cell Tracking Challenge**: Ulman et al. (2017) - "An objective comparison of cell-tracking algorithms"
- **Dataset**: Mitocheck Consortium

## 👨‍💻 Auteur

Cree avec ❤️ pour le TP3 d'imagerie biomedicale

## 📝 Commentaires Techniques

### Choix d'architecture
- U-Net est le standard pour la segmentation biomedicale
- 3 classes permettent de gerer les cellules adjacentes
- Base filters = 32 est un bon compromis capacite/memoire

### Choix d'hyperparametres
- LR = 1e-3 avec ReduceLROnPlateau: convergence stable
- Class weights adaptes au desequilibre (beaucoup de bg)
- Batch size = 2: limite par la taille des images (~1k x 1k)

### Choix de tracking
- IoU + distance: simple et efficace
- Seuil 0.35: balance entre precision et rappel
- Max distance 80px: couvre les mouvements biologiques

### Limitations
- Pas de detection de divisions cellulaires
- Tracking local (frame-to-frame) sans optimisation globale
- Pas de gestion des occlusions



