"""
👁️ visualisation des resultats de segmentation et tracking
permet de verifier visuellement la qualite des predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, exposure, segmentation
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_style('whitegrid')


def visualize_segmentation_quality(test_dir, results_dir, frames_to_show=None):
    """
    visualise la qualite de la segmentation sur plusieurs frames
    """
    if frames_to_show is None:
        frames_to_show = [0, 20, 40, 60, 80]
    
    fig, axes = plt.subplots(3, len(frames_to_show), figsize=(20, 12))
    
    for idx, frame_num in enumerate(frames_to_show):
        # charger image originale
        img_path = test_dir / f't{frame_num:03d}.tif'
        img = io.imread(str(img_path))
        img_enhanced = exposure.equalize_adapthist(img)
        
        # charger prediction
        mask_path = results_dir / 'SEG' / f'mask{frame_num:03d}.tif'
        mask = io.imread(str(mask_path))
        
        n_cells = len(np.unique(mask)) - 1  # -1 pour background
        
        # 1. image originale
        axes[0, idx].imshow(img_enhanced, cmap='gray')
        axes[0, idx].set_title(f'frame {frame_num}\nimage originale', fontsize=10)
        axes[0, idx].axis('off')
        
        # 2. segmentation overlay
        axes[1, idx].imshow(img, cmap='gray', alpha=0.7)
        axes[1, idx].imshow(mask, cmap='nipy_spectral', alpha=0.5)
        axes[1, idx].set_title(f'{n_cells} cellules\nsegmentees', fontsize=10)
        axes[1, idx].axis('off')
        
        # 3. contours + labels
        axes[2, idx].imshow(img_enhanced, cmap='gray')
        # dessiner contours
        contours = segmentation.find_boundaries(mask, mode='outer')
        axes[2, idx].contour(contours, colors='red', linewidths=1.5)
        axes[2, idx].set_title(f'contours\n{n_cells} cellules', fontsize=10)
        axes[2, idx].axis('off')
    
    plt.suptitle('🎯 qualite de la segmentation sur testset/01', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig('results_segmentation_quality.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ visualisation sauvegardee: results_segmentation_quality.png")


def analyze_cell_counts(results_dir):
    """
    analyse l'evolution du nombre de cellules detectees
    """
    seg_files = sorted(list((results_dir / 'SEG').glob('*.tif')))
    
    frame_nums = []
    cell_counts = []
    cell_areas = []
    
    for seg_file in seg_files:
        frame_num = int(seg_file.stem.replace('mask', ''))
        mask = io.imread(str(seg_file))
        
        n_cells = len(np.unique(mask)) - 1
        
        frame_nums.append(frame_num)
        cell_counts.append(n_cells)
        
        # aires des cellules
        for label in np.unique(mask):
            if label > 0:
                cell_areas.append(np.sum(mask == label))
    
    # statistiques
    print(f"\n📊 statistiques de detection:\n")
    print(f"   frames traites: {len(frame_nums)}")
    print(f"   cellules par frame:")
    print(f"      moyenne: {np.mean(cell_counts):.1f} ± {np.std(cell_counts):.1f}")
    print(f"      min/max: {np.min(cell_counts)} / {np.max(cell_counts)}")
    print(f"\n   aire des cellules:")
    print(f"      moyenne: {np.mean(cell_areas):.1f} ± {np.std(cell_areas):.1f} pixels")
    print(f"      mediane: {np.median(cell_areas):.1f} pixels")
    
    # visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # evolution temporelle
    axes[0].plot(frame_nums, cell_counts, 'o-', color='steelblue', linewidth=2, markersize=4)
    axes[0].axhline(np.mean(cell_counts), color='red', linestyle='--', 
                    label=f'moyenne: {np.mean(cell_counts):.1f}')
    axes[0].set_xlabel('frame', fontsize=11)
    axes[0].set_ylabel('nombre de cellules', fontsize=11)
    axes[0].set_title('evolution du nombre de cellules', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # distribution aires
    axes[1].hist(cell_areas, bins=50, color='seagreen', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.median(cell_areas), color='red', linestyle='--',
                    label=f'mediane: {np.median(cell_areas):.0f}')
    axes[1].set_xlabel('aire (pixels)', fontsize=11)
    axes[1].set_ylabel('frequence', fontsize=11)
    axes[1].set_title('distribution des aires cellulaires', fontsize=12)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('results_cell_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ statistiques sauvegardees: results_cell_statistics.png")
    
    return frame_nums, cell_counts, cell_areas


def visualize_tracking(test_dir, results_dir, tracks_file, frames_to_show=None):
    """
    visualise le tracking sur quelques frames
    """
    if frames_to_show is None:
        frames_to_show = [0, 30, 60, 90]
    
    # charger les tracks
    tracks = np.loadtxt(str(tracks_file), dtype=int)
    
    # creer un dictionnaire track_id -> couleur
    n_tracks = len(tracks)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_tracks, 20)))
    track_colors = {tracks[i, 0]: colors[i % 20] for i in range(n_tracks)}
    
    fig, axes = plt.subplots(2, len(frames_to_show), figsize=(20, 10))
    
    for idx, frame_num in enumerate(frames_to_show):
        if frame_num >= 92:
            continue
            
        # image originale
        img_path = test_dir / f't{frame_num:03d}.tif'
        img = io.imread(str(img_path))
        img_enhanced = exposure.equalize_adapthist(img)
        
        # segmentation
        mask_path = results_dir / 'SEG' / f'mask{frame_num:03d}.tif'
        mask = io.imread(str(mask_path))
        
        # trouver les tracks actives a ce frame
        active_tracks = []
        for track in tracks:
            track_id, start, end, parent = track
            if start <= frame_num <= end:
                active_tracks.append(track_id)
        
        # image originale
        axes[0, idx].imshow(img_enhanced, cmap='gray')
        axes[0, idx].set_title(f'frame {frame_num}\n{len(active_tracks)} tracks actives', fontsize=10)
        axes[0, idx].axis('off')
        
        # tracking visualisation
        axes[1, idx].imshow(img, cmap='gray', alpha=0.7)
        # colorer chaque cellule selon son track
        colored_mask = np.zeros((*mask.shape, 3))
        for label in np.unique(mask):
            if label > 0:
                # assigner une couleur basee sur le track (simplifie)
                color_idx = label % 20
                colored_mask[mask == label] = colors[color_idx][:3]
        
        axes[1, idx].imshow(colored_mask, alpha=0.5)
        axes[1, idx].set_title(f'{len(active_tracks)} tracks', fontsize=10)
        axes[1, idx].axis('off')
    
    plt.suptitle('🔍 visualisation du tracking', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig('results_tracking_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ tracking visualise: results_tracking_visualization.png")


def analyze_tracks(tracks_file):
    """
    analyse les proprietes des tracks
    """
    tracks = np.loadtxt(str(tracks_file), dtype=int)
    
    # statistiques
    track_lengths = tracks[:, 2] - tracks[:, 1] + 1
    n_divisions = np.sum(tracks[:, 3] > 0)
    
    print(f"\n🔍 analyse des tracks:\n")
    print(f"   nombre total de tracks: {len(tracks)}")
    print(f"   longueur des tracks (frames):")
    print(f"      moyenne: {np.mean(track_lengths):.1f} ± {np.std(track_lengths):.1f}")
    print(f"      mediane: {np.median(track_lengths):.0f}")
    print(f"      min/max: {np.min(track_lengths)} / {np.max(track_lengths)}")
    print(f"\n   divisions detectees: {n_divisions}")
    print(f"   ratio division: {n_divisions/len(tracks)*100:.1f}%")
    
    # tracks courtes vs longues
    short_tracks = np.sum(track_lengths < 5)
    medium_tracks = np.sum((track_lengths >= 5) & (track_lengths < 30))
    long_tracks = np.sum(track_lengths >= 30)
    
    print(f"\n   repartition longueurs:")
    print(f"      courtes (<5 frames): {short_tracks} ({short_tracks/len(tracks)*100:.1f}%)")
    print(f"      moyennes (5-30): {medium_tracks} ({medium_tracks/len(tracks)*100:.1f}%)")
    print(f"      longues (>30): {long_tracks} ({long_tracks/len(tracks)*100:.1f}%)")
    
    # visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # distribution longueurs
    axes[0].hist(track_lengths, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.median(track_lengths), color='red', linestyle='--',
                    label=f'mediane: {np.median(track_lengths):.0f}')
    axes[0].set_xlabel('longueur (frames)', fontsize=11)
    axes[0].set_ylabel('frequence', fontsize=11)
    axes[0].set_title('distribution des longueurs de tracks', fontsize=12)
    axes[0].legend()
    
    # timeline des tracks (50 premieres)
    n_show = min(50, len(tracks))
    for i, track in enumerate(tracks[:n_show]):
        track_id, start, end, parent = track
        color = 'red' if parent > 0 else 'steelblue'
        axes[1].plot([start, end], [i, i], linewidth=3, color=color, alpha=0.7)
    
    axes[1].set_xlabel('frame', fontsize=11)
    axes[1].set_ylabel('track id', fontsize=11)
    axes[1].set_title(f'timeline des {n_show} premiers tracks\n(bleu: nouveau, rouge: division)', 
                      fontsize=12)
    
    # legende
    blue_patch = mpatches.Patch(color='steelblue', label='nouvelle cellule')
    red_patch = mpatches.Patch(color='red', label='division')
    axes[1].legend(handles=[blue_patch, red_patch])
    
    plt.tight_layout()
    plt.savefig('results_track_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ statistiques tracks sauvegardees: results_track_statistics.png")
    
    return tracks, track_lengths


def compare_cell_sizes(test_dir, results_dir):
    """
    compare les tailles detectees vs parametres attendus
    """
    print(f"\n📏 comparaison tailles cellulaires:\n")
    
    # parametres du dataset
    pixel_size = 0.645  # um/pixel
    
    # analyser quelques frames
    sample_frames = [0, 20, 40, 60, 80]
    all_areas = []
    
    for frame_num in sample_frames:
        mask_path = results_dir / 'SEG' / f'mask{frame_num:03d}.tif'
        mask = io.imread(str(mask_path))
        
        for label in np.unique(mask):
            if label > 0:
                area_px = np.sum(mask == label)
                all_areas.append(area_px)
    
    all_areas = np.array(all_areas)
    area_um2 = all_areas * pixel_size**2
    diameter_um = 2 * np.sqrt(all_areas / np.pi) * pixel_size
    
    print(f"   echantillon: {len(all_areas)} cellules")
    print(f"\n   aire (pixels):")
    print(f"      moyenne: {np.mean(all_areas):.1f} ± {np.std(all_areas):.1f}")
    print(f"      mediane: {np.median(all_areas):.1f}")
    print(f"\n   aire (µm²):")
    print(f"      moyenne: {np.mean(area_um2):.1f} ± {np.std(area_um2):.1f}")
    print(f"\n   diametre equivalent (µm):")
    print(f"      moyenne: {np.mean(diameter_um):.1f} ± {np.std(diameter_um):.1f}")
    print(f"\n   💡 reference training: aire ~472 px, diametre ~15.4 µm")
    print(f"   → coherence: {'✅ bon' if 400 < np.mean(all_areas) < 550 else '⚠️ verifier'}")


def main():
    """pipeline complet d'analyse des resultats"""
    
    print("=" * 70)
    print("👁️  VISUALISATION ET ANALYSE DES RESULTATS")
    print("=" * 70)
    
    # chemins
    test_dir = Path('Testset/01')
    results_dir = Path('Results')
    tracks_file = results_dir / 'tracks.txt'
    
    # verifier que les resultats existent
    if not results_dir.exists():
        print("❌ dossier Results/ non trouve. lancez d'abord le pipeline principal.")
        return
    
    # 1. visualiser qualite segmentation
    print("\n1️⃣ visualisation de la segmentation...")
    visualize_segmentation_quality(test_dir, results_dir)
    
    # 2. statistiques detection
    print("\n2️⃣ analyse des detections...")
    frame_nums, cell_counts, cell_areas = analyze_cell_counts(results_dir)
    
    # 3. comparaison tailles
    compare_cell_sizes(test_dir, results_dir)
    
    # 4. analyse tracking
    print("\n3️⃣ analyse du tracking...")
    tracks, track_lengths = analyze_tracks(tracks_file)
    
    # 5. visualisation tracking
    print("\n4️⃣ visualisation du tracking...")
    visualize_tracking(test_dir, results_dir, tracks_file)
    
    # resume final
    print("\n" + "=" * 70)
    print("📊 RESUME PERFORMANCE")
    print("=" * 70)
    print(f"\n✅ segmentation:")
    print(f"   - {len(frame_nums)} frames traites")
    print(f"   - {np.mean(cell_counts):.1f} cellules/frame (moyenne)")
    print(f"   - aire moyenne: {np.median(cell_areas):.0f} pixels")
    print(f"\n✅ tracking:")
    print(f"   - {len(tracks)} tracks generees")
    print(f"   - longueur moyenne: {np.mean(track_lengths):.1f} frames")
    print(f"   - tracks longues (>30 frames): {np.sum(track_lengths >= 30)}")
    print(f"\n✅ fichiers generes:")
    print(f"   - results_segmentation_quality.png")
    print(f"   - results_cell_statistics.png")
    print(f"   - results_track_statistics.png")
    print(f"   - results_tracking_visualization.png")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()



