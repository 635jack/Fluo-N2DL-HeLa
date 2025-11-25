"""
⚙️ fichier de configuration pour la segmentation et tracking
permet d'ajuster facilement tous les parametres de tuning
"""

from pathlib import Path

# ============================================================================
# 📂 chemins des donnees
# ============================================================================

PATHS = {
    'train_01': Path('Trainset/01'),
    'train_01_gt_seg': Path('Trainset/01_GT/SEG'),
    'train_01_gt_tra': Path('Trainset/01_GT/TRA'),
    'train_02': Path('Trainset/02'),
    'train_02_gt_seg': Path('Trainset/02_GT/SEG'),
    'train_02_gt_tra': Path('Trainset/02_GT/TRA'),
    'test_01': Path('Testset/01'),
    'test_02': Path('Testset/02'),
    'results': Path('Results'),
}

# ============================================================================
# 🏗️ parametres du modele u-net
# ============================================================================

MODEL = {
    'n_channels': 1,          # images grayscale
    'n_classes': 3,           # background, cellule, bordure
    'base_filters': 32,       # nombre de filtres de base (16, 32, 64)
    'depth': 4,               # profondeur du u-net (nombre de niveaux)
}

# ============================================================================
# 🎓 parametres d'entrainement
# ============================================================================

TRAINING = {
    # optimisation
    'num_epochs': 50,         # nombre d'iterations (50-100)
    'batch_size': 2,          # taille de batch (1, 2, 4, 8)
    'learning_rate': 1e-3,    # lr initial (1e-2, 1e-3, 1e-4)
    'weight_decay': 1e-5,     # regularisation l2 (0, 1e-5, 1e-4)
    'optimizer': 'adam',      # 'adam', 'sgd', 'adamw'
    
    # loss function
    'loss_type': 'ce',        # 'ce' (cross entropy), 'dice', 'focal'
    'class_weights': [0.1, 1.0, 2.0],  # poids pour [bg, cell, border]
    
    # scheduler
    'scheduler_type': 'plateau',  # 'plateau', 'step', 'cosine'
    'scheduler_factor': 0.5,      # facteur de reduction du lr
    'scheduler_patience': 5,      # patience avant reduction
    
    # split des donnees
    'train_ratio': 0.8,       # ratio train/val (0.7-0.9)
    'random_seed': 42,        # seed pour reproductibilite
    
    # workers
    'num_workers': 2,         # threads pour chargement donnees
}

# ============================================================================
# 🔄 augmentation de donnees (optionnel)
# ============================================================================

AUGMENTATION = {
    'enabled': False,         # activer/desactiver augmentation
    'rotation': 15,           # rotation aleatoire en degres (±15)
    'horizontal_flip': True,  # flip horizontal
    'vertical_flip': True,    # flip vertical
    'brightness': 0.1,        # variation luminosite (±10%)
    'contrast': 0.1,          # variation contraste (±10%)
    'zoom': 0.1,              # zoom aleatoire (±10%)
}

# ============================================================================
# 🎨 parametres de post-traitement
# ============================================================================

POSTPROCESS = {
    # seuils de probabilite (OPTIMISES - version amelioree)
    'threshold_cell': 0.55,    # 0.5 → 0.55 (plus strict, moins faux positifs)
    'threshold_border': 0.35,  # 0.3 → 0.35 (bordures mieux definies)
    
    # watershed
    'erosion_radius': 2,       # rayon erosion pour marqueurs (1-3)
    'min_size': 250,           # 192 → 250 (filtrer petits objets)
    'min_markers_size': 30,    # taille min des marqueurs (10-50)
    
    # nettoyage
    'remove_border_cells': False,  # supprimer cellules touchant bords
}

# ============================================================================
# 🔍 parametres de tracking
# ============================================================================

TRACKING = {
    # association entre frames (OPTIMISES - version amelioree)
    'iou_threshold': 0.25,     # 0.35 → 0.25 (plus permissif, moins fragmentations)
    'max_distance': 120,       # 80 → 120 (cellules bougent plus que prevu)
    'distance_weight': 0.2,    # 0.3 → 0.2 (moins penaliser mouvement)
    
    # filtrage des tracks
    'min_track_length': 3,     # longueur minimale track (3-10 frames)
    'max_gap': 2,              # nombre max de frames manquants (1-5)
    
    # division cellulaire (experimental)
    'detect_division': False,  # detecter les divisions
    'division_area_ratio': 0.4,  # ratio aire fille/mere (0.3-0.6)
}

# ============================================================================
# 💾 parametres de sauvegarde
# ============================================================================

SAVE = {
    'save_best_only': True,    # sauvegarder seulement le meilleur modele
    'save_interval': 10,       # sauvegarder tous les N epochs
    'save_predictions': True,  # sauvegarder predictions sur val set
    'save_overlays': True,     # sauvegarder overlays image+mask
    'visualization_frames': [0, 20, 40, 60, 80],  # frames a visualiser
}

# ============================================================================
# 🖥️ parametres systeme
# ============================================================================

SYSTEM = {
    'device': 'auto',          # 'auto', 'cuda', 'cpu'
    'gpu_id': 0,               # id du gpu si plusieurs disponibles
    'mixed_precision': False,  # utiliser mixed precision (fp16)
    'deterministic': True,     # reproductibilite (plus lent)
}

# ============================================================================
# 📊 parametres d'evaluation
# ============================================================================

EVALUATION = {
    'compute_metrics': True,   # calculer metriques pendant validation
    'metrics': ['iou', 'dice', 'precision', 'recall'],
    'save_confusion_matrix': True,
}


# ============================================================================
# 🎯 presets pour differents scenarios
# ============================================================================

def get_preset(preset_name):
    """
    retourne une configuration preset
    
    presets disponibles:
    - 'fast': entrainement rapide pour test
    - 'balanced': configuration equilibree (par defaut)
    - 'quality': maximiser la qualite (plus lent)
    - 'memory': minimiser utilisation memoire
    """
    
    if preset_name == 'fast':
        return {
            'TRAINING': {**TRAINING, 'num_epochs': 20, 'batch_size': 4},
            'MODEL': {**MODEL, 'base_filters': 16},
            'AUGMENTATION': {**AUGMENTATION, 'enabled': False},
        }
    
    elif preset_name == 'quality':
        return {
            'TRAINING': {**TRAINING, 'num_epochs': 100, 'batch_size': 2, 'learning_rate': 5e-4},
            'MODEL': {**MODEL, 'base_filters': 64},
            'AUGMENTATION': {**AUGMENTATION, 'enabled': True},
            'POSTPROCESS': {**POSTPROCESS, 'min_size': 100},
        }
    
    elif preset_name == 'memory':
        return {
            'TRAINING': {**TRAINING, 'batch_size': 1, 'num_workers': 1},
            'MODEL': {**MODEL, 'base_filters': 16},
            'SYSTEM': {**SYSTEM, 'mixed_precision': True},
        }
    
    else:  # 'balanced' par defaut
        return {
            'TRAINING': TRAINING,
            'MODEL': MODEL,
            'AUGMENTATION': AUGMENTATION,
            'POSTPROCESS': POSTPROCESS,
            'TRACKING': TRACKING,
            'SAVE': SAVE,
            'SYSTEM': SYSTEM,
        }


# ============================================================================
# 📝 notes et justifications des parametres
# ============================================================================

NOTES = """
🔧 guide de tuning des parametres

LEARNING RATE:
- 1e-3: bon point de depart pour adam
- 1e-4: si oscillations dans la loss
- 1e-2: convergence rapide mais instable

BATCH SIZE:
- limite par la memoire gpu
- images ~1k x 1k → batch_size = 2-4
- batch_size plus grand = entrainement plus stable

CLASS WEIGHTS [bg, cell, border]:
- [0.1, 1.0, 2.0]: background majoritaire, bordures rares
- ajuster si trop de faux positifs/negatifs
- border weight = 2.0: critique pour separer cellules

MIN_SIZE:
- base sur analyse des donnees (5e percentile)
- trop petit: beaucoup de faux positifs
- trop grand: perte de petites cellules

IOU_THRESHOLD:
- 0.35: bon compromis
- plus bas (0.2): plus de faux positifs tracking
- plus haut (0.5): risque de perdre des cellules

MAX_DISTANCE:
- base sur biologie: vitesse cellules + interval temps
- 80 pixels ≈ 52 µm ≈ deplacement max en 30 min
- ajuster selon vitesse deplacement des cellules
"""


if __name__ == '__main__':
    # afficher la configuration actuelle
    print("📋 configuration actuelle:\n")
    print(f"modele: {MODEL}")
    print(f"\nentrainement: {TRAINING}")
    print(f"\npost-traitement: {POSTPROCESS}")
    print(f"\ntracking: {TRACKING}")
    print(f"\n{NOTES}")

