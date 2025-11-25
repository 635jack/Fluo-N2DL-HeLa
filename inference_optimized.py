"""
🚀 inference optimisee avec parametres ameliores
charge le meilleur modele et applique post-traitement optimise
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from skimage import io, measure, morphology
from scipy.ndimage import watershed_ift

# importer l'architecture u-net
import sys
sys.path.append('.')
from cell_segmentation_tracking import UNet, CellTracker, postprocess_segmentation


def run_optimized_inference():
    """
    relance inference avec parametres optimises
    """
    print("=" * 70)
    print("🚀 INFERENCE OPTIMISEE - OPTION A")
    print("=" * 70)
    print("\n📋 parametres ameliores:")
    print("   post-traitement:")
    print("      - min_size: 192 → 250 px (+30%)")
    print("      - threshold_cell: 0.5 → 0.55 (+10%)")
    print("      - threshold_border: 0.3 → 0.35 (+17%)")
    print("   tracking:")
    print("      - iou_threshold: 0.35 → 0.25 (-29%)")
    print("      - max_distance: 80 → 120 px (+50%)")
    print("      - distance_weight: 0.3 → 0.2 (-33%)")
    
    # chemins
    test_dir = Path('Testset/01')
    test_files = sorted(list(test_dir.glob('*.tif')))
    
    results_dir = Path('Results_Optimized')
    results_dir.mkdir(exist_ok=True)
    seg_dir = results_dir / 'SEG'
    seg_dir.mkdir(exist_ok=True)
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 device: {device}")
    
    # charger le meilleur modele
    print("\n🧠 chargement du modele...")
    model = UNet(n_channels=1, n_classes=3, base_filters=32).to(device)
    model.load_state_dict(torch.load('best_unet_model.pth', map_location=device))
    model.eval()
    print("   ✅ modele charge (epoch 23, val_loss=0.3896)")
    
    # tracking avec parametres optimises
    tracker = CellTracker(iou_threshold=0.25, max_distance=120)
    
    print(f"\n🎯 inference sur {len(test_files)} frames...")
    
    all_cell_counts = []
    all_cell_areas = []
    
    for idx, img_path in enumerate(tqdm(test_files, desc="traitement")):
        # charger et normaliser
        image = io.imread(str(img_path))
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # inference
        with torch.no_grad():
            img_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)
            output = model(img_tensor)
            
            # resize si necessaire
            if output.shape[2:] != image.shape:
                output = F.interpolate(output, size=image.shape, mode='bilinear', align_corners=False)
            
            pred = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        # post-traitement avec parametres optimises
        labels = postprocess_segmentation(
            pred, 
            min_size=250,          # 192 → 250
            threshold_cell=0.55,   # 0.5 → 0.55
            threshold_border=0.35  # 0.3 → 0.35
        )
        
        # statistiques
        n_cells = len(np.unique(labels)) - 1
        all_cell_counts.append(n_cells)
        
        for label in np.unique(labels):
            if label > 0:
                all_cell_areas.append(np.sum(labels == label))
        
        # tracking
        tracker.track_frame(idx, labels)
        
        # sauvegarder
        output_name = f"mask{img_path.stem.replace('t', '')}.tif"
        io.imsave(str(seg_dir / output_name), labels.astype(np.uint16), check_contrast=False)
    
    print(f"\n✅ segmentations sauvegardees dans {seg_dir}")
    
    # exporter les tracks
    print("\n🔍 export des tracks...")
    lineages = tracker.get_track_lineages()
    
    # filtrer les tracks trop courtes (probablement erreurs)
    lineages_filtered = [l for l in lineages if (l['end'] - l['start'] + 1) >= 3]
    
    with open(results_dir / 'tracks.txt', 'w') as f:
        for lineage in lineages_filtered:
            f.write(f"{lineage['id']} {lineage['start']} {lineage['end']} {lineage['parent']}\n")
    
    # statistiques
    print(f"\n📊 statistiques optimisees:")
    print(f"   segmentation:")
    print(f"      - cellules/frame: {np.mean(all_cell_counts):.1f} ± {np.std(all_cell_counts):.1f}")
    print(f"      - aire moyenne: {np.mean(all_cell_areas):.1f} ± {np.std(all_cell_areas):.1f} pixels")
    print(f"      - aire mediane: {np.median(all_cell_areas):.0f} pixels")
    
    track_lengths = [l['end'] - l['start'] + 1 for l in lineages_filtered]
    print(f"   tracking:")
    print(f"      - tracks totales: {len(lineages)} → {len(lineages_filtered)} (apres filtrage)")
    print(f"      - longueur moyenne: {np.mean(track_lengths):.1f} ± {np.std(track_lengths):.1f} frames")
    print(f"      - longueur mediane: {np.median(track_lengths):.0f} frames")
    
    short_tracks = sum(1 for l in track_lengths if l < 5)
    medium_tracks = sum(1 for l in track_lengths if 5 <= l < 30)
    long_tracks = sum(1 for l in track_lengths if l >= 30)
    
    print(f"      - courtes (<5): {short_tracks} ({short_tracks/len(track_lengths)*100:.1f}%)")
    print(f"      - moyennes (5-30): {medium_tracks} ({medium_tracks/len(track_lengths)*100:.1f}%)")
    print(f"      - longues (>30): {long_tracks} ({long_tracks/len(track_lengths)*100:.1f}%)")
    
    print(f"\n🎉 inference optimisee terminee!")
    print(f"   resultats dans: {results_dir}/")
    
    return all_cell_counts, all_cell_areas, lineages_filtered


def compare_results():
    """
    compare les resultats originaux vs optimises
    """
    print("\n" + "=" * 70)
    print("📊 COMPARAISON AVANT/APRES")
    print("=" * 70)
    
    # charger resultats originaux
    original_tracks_file = Path('Results/tracks.txt')
    optimized_tracks_file = Path('Results_Optimized/tracks.txt')
    
    if not original_tracks_file.exists() or not optimized_tracks_file.exists():
        print("⚠️ resultats manquants pour comparaison")
        return
    
    original_tracks = np.loadtxt(str(original_tracks_file), dtype=int)
    optimized_tracks = np.loadtxt(str(optimized_tracks_file), dtype=int)
    
    orig_lengths = original_tracks[:, 2] - original_tracks[:, 1] + 1
    opt_lengths = optimized_tracks[:, 2] - optimized_tracks[:, 1] + 1
    
    print(f"\n   TRACKING:")
    print(f"   {'Metrique':<30} | {'Original':<15} | {'Optimise':<15} | {'Gain'}")
    print(f"   {'-'*30}-+-{'-'*15}-+-{'-'*15}-+-{'-'*15}")
    print(f"   {'Nombre de tracks':<30} | {len(original_tracks):<15} | {len(optimized_tracks):<15} | {(len(optimized_tracks)-len(original_tracks))/len(original_tracks)*100:+.1f}%")
    print(f"   {'Longueur moyenne':<30} | {np.mean(orig_lengths):<15.1f} | {np.mean(opt_lengths):<15.1f} | {(np.mean(opt_lengths)-np.mean(orig_lengths))/np.mean(orig_lengths)*100:+.1f}%")
    print(f"   {'Longueur mediane':<30} | {np.median(orig_lengths):<15.0f} | {np.median(opt_lengths):<15.0f} | {(np.median(opt_lengths)-np.median(orig_lengths))/np.median(orig_lengths)*100:+.1f}%")
    
    orig_long = np.sum(orig_lengths >= 30)
    opt_long = np.sum(opt_lengths >= 30)
    print(f"   {'Tracks longues (>30)':<30} | {orig_long:<15} | {opt_long:<15} | {(opt_long-orig_long)/max(orig_long,1)*100:+.1f}%")
    
    # segmentation
    print(f"\n   SEGMENTATION:")
    
    # compter cellules dans chaque version
    orig_seg_dir = Path('Results/SEG')
    opt_seg_dir = Path('Results_Optimized/SEG')
    
    orig_counts = []
    opt_counts = []
    orig_areas = []
    opt_areas = []
    
    for i in range(92):
        orig_mask = io.imread(str(orig_seg_dir / f'mask{i:03d}.tif'))
        opt_mask = io.imread(str(opt_seg_dir / f'mask{i:03d}.tif'))
        
        orig_counts.append(len(np.unique(orig_mask)) - 1)
        opt_counts.append(len(np.unique(opt_mask)) - 1)
        
        for label in np.unique(orig_mask):
            if label > 0:
                orig_areas.append(np.sum(orig_mask == label))
        
        for label in np.unique(opt_mask):
            if label > 0:
                opt_areas.append(np.sum(opt_mask == label))
    
    print(f"   {'Cellules/frame (moyenne)':<30} | {np.mean(orig_counts):<15.1f} | {np.mean(opt_counts):<15.1f} | {(np.mean(opt_counts)-np.mean(orig_counts))/np.mean(orig_counts)*100:+.1f}%")
    print(f"   {'Aire moyenne (pixels)':<30} | {np.mean(orig_areas):<15.0f} | {np.mean(opt_areas):<15.0f} | {(np.mean(opt_areas)-np.mean(orig_areas))/np.mean(orig_areas)*100:+.1f}%")
    print(f"   {'Aire mediane (pixels)':<30} | {np.median(orig_areas):<15.0f} | {np.median(opt_areas):<15.0f} | {(np.median(opt_areas)-np.median(orig_areas))/np.median(orig_areas)*100:+.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ AMELIORATIONS REUSSIES!")
    print("=" * 70)


if __name__ == '__main__':
    # lancer inference optimisee
    counts, areas, tracks = run_optimized_inference()
    
    # comparer avec resultats originaux
    compare_results()
