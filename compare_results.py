"""
📊 compare visuellement les resultats avant/apres optimisation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, exposure
import seaborn as sns

sns.set_style('whitegrid')


def compare_segmentation_visual():
    """compare segmentation sur quelques frames"""
    
    frames_to_show = [20, 40, 60, 80]
    
    fig, axes = plt.subplots(3, len(frames_to_show), figsize=(20, 12))
    
    test_dir = Path('Testset/01')
    orig_dir = Path('Results/SEG')
    opt_dir = Path('Results_Optimized/SEG')
    
    for idx, frame_num in enumerate(frames_to_show):
        # image originale
        img = io.imread(str(test_dir / f't{frame_num:03d}.tif'))
        img_enh = exposure.equalize_adapthist(img)
        
        # masques
        mask_orig = io.imread(str(orig_dir / f'mask{frame_num:03d}.tif'))
        mask_opt = io.imread(str(opt_dir / f'mask{frame_num:03d}.tif'))
        
        n_orig = len(np.unique(mask_orig)) - 1
        n_opt = len(np.unique(mask_opt)) - 1
        
        # image originale
        axes[0, idx].imshow(img_enh, cmap='gray')
        axes[0, idx].set_title(f'frame {frame_num}', fontsize=11)
        axes[0, idx].axis('off')
        
        # version originale
        axes[1, idx].imshow(img, cmap='gray', alpha=0.7)
        axes[1, idx].imshow(mask_orig, cmap='nipy_spectral', alpha=0.5)
        axes[1, idx].set_title(f'original\n{n_orig} cellules', fontsize=11, color='red')
        axes[1, idx].axis('off')
        
        # version optimisee
        axes[2, idx].imshow(img, cmap='gray', alpha=0.7)
        axes[2, idx].imshow(mask_opt, cmap='nipy_spectral', alpha=0.5)
        axes[2, idx].set_title(f'optimise\n{n_opt} cellules', fontsize=11, color='green')
        axes[2, idx].axis('off')
    
    plt.suptitle('comparaison segmentation: original vs optimise', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig('comparison_segmentation.png', dpi=150, bbox_inches='tight')
    print("✅ comparaison sauvegardee: comparison_segmentation.png")
    plt.show()


def compare_statistics():
    """compare statistiques globales"""
    
    # charger tous les masques
    orig_dir = Path('Results/SEG')
    opt_dir = Path('Results_Optimized/SEG')
    
    orig_counts = []
    opt_counts = []
    orig_areas = []
    opt_areas = []
    
    for i in range(92):
        mask_orig = io.imread(str(orig_dir / f'mask{i:03d}.tif'))
        mask_opt = io.imread(str(opt_dir / f'mask{i:03d}.tif'))
        
        orig_counts.append(len(np.unique(mask_orig)) - 1)
        opt_counts.append(len(np.unique(mask_opt)) - 1)
        
        for label in np.unique(mask_orig):
            if label > 0:
                orig_areas.append(np.sum(mask_orig == label))
        
        for label in np.unique(mask_opt):
            if label > 0:
                opt_areas.append(np.sum(mask_opt == label))
    
    # visualisation
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # evolution cellules/frame
    frames = np.arange(92)
    axes[0, 0].plot(frames, orig_counts, 'o-', label='original', color='red', alpha=0.6, markersize=3)
    axes[0, 0].plot(frames, opt_counts, 'o-', label='optimise', color='green', alpha=0.6, markersize=3)
    axes[0, 0].axhline(np.mean(orig_counts), color='red', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(np.mean(opt_counts), color='green', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('frame', fontsize=11)
    axes[0, 0].set_ylabel('nombre de cellules', fontsize=11)
    axes[0, 0].set_title('evolution temporelle', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # distribution aires
    axes[0, 1].hist(orig_areas, bins=50, alpha=0.5, label='original', color='red', edgecolor='darkred')
    axes[0, 1].hist(opt_areas, bins=50, alpha=0.5, label='optimise', color='green', edgecolor='darkgreen')
    axes[0, 1].axvline(np.median(orig_areas), color='red', linestyle='--', linewidth=2)
    axes[0, 1].axvline(np.median(opt_areas), color='green', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('aire (pixels)', fontsize=11)
    axes[0, 1].set_ylabel('frequence', fontsize=11)
    axes[0, 1].set_title('distribution des aires cellulaires', fontsize=12)
    axes[0, 1].legend()
    
    # boxplot cellules/frame
    axes[1, 0].boxplot([orig_counts, opt_counts], labels=['original', 'optimise'], 
                        patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
    axes[1, 0].set_ylabel('cellules/frame', fontsize=11)
    axes[1, 0].set_title('distribution cellules par frame', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # boxplot aires
    axes[1, 1].boxplot([orig_areas, opt_areas], labels=['original', 'optimise'],
                        patch_artist=True,
                        boxprops=dict(facecolor='lightcoral', alpha=0.7),
                        medianprops=dict(color='darkred', linewidth=2))
    axes[1, 1].set_ylabel('aire (pixels)', fontsize=11)
    axes[1, 1].set_title('distribution aires cellulaires', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparison_statistics.png', dpi=150, bbox_inches='tight')
    print("✅ statistiques comparatives sauvegardees: comparison_statistics.png")
    plt.show()


def compare_tracking():
    """compare les tracks"""
    
    orig_tracks = np.loadtxt('Results/tracks.txt', dtype=int)
    opt_tracks = np.loadtxt('Results_Optimized/tracks.txt', dtype=int)
    
    orig_lengths = orig_tracks[:, 2] - orig_tracks[:, 1] + 1
    opt_lengths = opt_tracks[:, 2] - opt_tracks[:, 1] + 1
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # distribution longueurs
    axes[0].hist(orig_lengths, bins=30, alpha=0.6, label='original', color='red', edgecolor='darkred')
    axes[0].hist(opt_lengths, bins=30, alpha=0.6, label='optimise', color='green', edgecolor='darkgreen')
    axes[0].axvline(np.median(orig_lengths), color='red', linestyle='--', linewidth=2,
                    label=f'mediane orig: {np.median(orig_lengths):.0f}')
    axes[0].axvline(np.median(opt_lengths), color='green', linestyle='--', linewidth=2,
                    label=f'mediane opt: {np.median(opt_lengths):.0f}')
    axes[0].set_xlabel('longueur (frames)', fontsize=11)
    axes[0].set_ylabel('frequence', fontsize=11)
    axes[0].set_title('distribution longueurs tracks', fontsize=12)
    axes[0].legend()
    
    # timeline comparaison (50 premiers de chaque)
    n_show = 50
    offset = 25  # offset pour separer visuellement
    
    for i, track in enumerate(orig_tracks[:n_show]):
        axes[1].plot([track[1], track[2]], [i, i], linewidth=2, color='red', alpha=0.6)
    
    for i, track in enumerate(opt_tracks[:min(n_show, len(opt_tracks))]):
        axes[1].plot([track[1], track[2]], [i + offset, i + offset], linewidth=2, color='green', alpha=0.6)
    
    axes[1].axhline(offset - 2, color='black', linestyle='--', alpha=0.5)
    axes[1].text(0, n_show//2, 'original', fontsize=10, color='red', weight='bold')
    axes[1].text(0, offset + n_show//2, 'optimise', fontsize=10, color='green', weight='bold')
    axes[1].set_xlabel('frame', fontsize=11)
    axes[1].set_ylabel('track id', fontsize=11)
    axes[1].set_title(f'timeline des tracks (50 premiers)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('comparison_tracking.png', dpi=150, bbox_inches='tight')
    print("✅ tracking comparatif sauvegarde: comparison_tracking.png")
    plt.show()


def main():
    print("=" * 70)
    print("📊 VISUALISATION COMPARATIVE AVANT/APRES")
    print("=" * 70)
    
    print("\n1️⃣ comparaison segmentation...")
    compare_segmentation_visual()
    
    print("\n2️⃣ statistiques comparatives...")
    compare_statistics()
    
    print("\n3️⃣ comparaison tracking...")
    compare_tracking()
    
    print("\n" + "=" * 70)
    print("✅ toutes les comparaisons generees!")
    print("   - comparison_segmentation.png")
    print("   - comparison_statistics.png")
    print("   - comparison_tracking.png")
    print("=" * 70)


if __name__ == '__main__':
    main()



