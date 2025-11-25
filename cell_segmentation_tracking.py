"""
🔬 segmentation et tracking de cellules hela par u-net
auteur: assistant ia
dataset: fluo-n2dl-hela (cell tracking challenge)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# deep learning 🧠
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# traitement d'images 📸
from skimage import io, measure, morphology
from skimage.segmentation import watershed
from scipy import ndimage


# ============================================================================
# 🏗️ architecture u-net pour la segmentation
# ============================================================================

class DoubleConv(nn.Module):
    """bloc de double convolution avec batch norm et relu"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    🎯 u-net pour segmentation d'instances
    parametres importants:
    - n_channels: nombre de canaux d'entree (1 pour grayscale)
    - n_classes: nombre de classes de sortie (3: background, cellule, bordure)
    - base_filters: nombre de filtres de la premiere couche (32 par defaut)
    """
    def __init__(self, n_channels=1, n_classes=3, base_filters=32):
        super(UNet, self).__init__()
        
        # encodeur (contraction) 📉
        self.enc1 = DoubleConv(n_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(base_filters, base_filters*2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(base_filters*2, base_filters*4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(base_filters*4, base_filters*8)
        self.pool4 = nn.MaxPool2d(2)
        
        # bottleneck 🍾
        self.bottleneck = DoubleConv(base_filters*8, base_filters*16)
        
        # decodeur (expansion) 📈
        self.upconv4 = nn.ConvTranspose2d(base_filters*16, base_filters*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_filters*16, base_filters*8)
        
        self.upconv3 = nn.ConvTranspose2d(base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_filters*8, base_filters*4)
        
        self.upconv2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_filters*4, base_filters*2)
        
        self.upconv1 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_filters*2, base_filters)
        
        # couche finale 🎯
        self.out = nn.Conv2d(base_filters, n_classes, kernel_size=1)
    
    def forward(self, x):
        # encodeur avec skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # decodeur avec concatenation et crop automatique
        dec4 = self.upconv4(bottleneck)
        dec4 = self._crop_and_concat(dec4, enc4)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = self._crop_and_concat(dec3, enc3)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = self._crop_and_concat(dec2, enc2)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = self._crop_and_concat(dec1, enc1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)
    
    def _crop_and_concat(self, upsampled, encoder_features):
        """crop encoder features pour matcher la taille de upsampled puis concatener"""
        # obtenir les tailles
        _, _, h_up, w_up = upsampled.size()
        _, _, h_enc, w_enc = encoder_features.size()
        
        # crop si necessaire
        if h_enc != h_up or w_enc != w_up:
            # centrer le crop
            dh = (h_enc - h_up) // 2
            dw = (w_enc - w_up) // 2
            encoder_features = encoder_features[:, :, dh:dh+h_up, dw:dw+w_up]
        
        return torch.cat([upsampled, encoder_features], dim=1)


# ============================================================================
# 📦 dataset personnalise pour le chargement des donnees
# ============================================================================

class HeLaDataset(Dataset):
    """
    dataset pour les cellules hela
    charge les images brutes et les masques de segmentation ground truth
    """
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        
        # liste des images disponibles
        self.image_files = sorted(list(self.image_dir.glob('*.tif')))
        
        # si on a des masques gt, ne garder que les images avec masque
        if self.mask_dir:
            mask_files = [f.stem.replace('man_seg', 't') for f in self.mask_dir.glob('*.tif')]
            self.image_files = [f for f in self.image_files if f.stem in mask_files]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # charger l'image
        img_path = self.image_files[idx]
        image = io.imread(str(img_path))
        
        # normalisation 0-1
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # charger le masque si disponible
        if self.mask_dir:
            mask_name = 'man_seg' + img_path.stem.replace('t', '')
            mask_path = self.mask_dir / f'{mask_name}.tif'
            
            if mask_path.exists():
                mask = io.imread(str(mask_path))
                # creer masque 3 classes: background, cellule, bordure
                mask_processed = self.create_border_mask(mask)
            else:
                mask_processed = np.zeros((3, image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            mask_processed = None
        
        # ajouter dimension canal
        image = np.expand_dims(image, 0)
        
        if self.transform:
            image, mask_processed = self.transform(image, mask_processed)
        
        if mask_processed is not None:
            return torch.FloatTensor(image), torch.FloatTensor(mask_processed)
        else:
            return torch.FloatTensor(image)
    
    def create_border_mask(self, mask):
        """
        🎨 creer un masque a 3 classes:
        - classe 0: background
        - classe 1: interieur des cellules
        - classe 2: bordures entre cellules (pour separer les instances)
        """
        h, w = mask.shape
        output = np.zeros((3, h, w), dtype=np.float32)
        
        # classe 0: background
        output[0] = (mask == 0).astype(np.float32)
        
        # pour chaque cellule, calculer l'interieur et la bordure
        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids > 0]
        
        borders = np.zeros_like(mask, dtype=bool)
        interiors = np.zeros_like(mask, dtype=bool)
        
        for cell_id in cell_ids:
            cell_mask = (mask == cell_id)
            
            # eroder pour obtenir l'interieur
            eroded = morphology.binary_erosion(cell_mask, morphology.disk(2))
            interiors |= eroded
            
            # bordure = cellule - interieur
            border = cell_mask & ~eroded
            borders |= border
        
        # classe 1: interieur
        output[1] = interiors.astype(np.float32)
        
        # classe 2: bordures
        output[2] = borders.astype(np.float32)
        
        return output


# ============================================================================
# 🎓 entrainement du modele
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device='cuda'):
    """
    entrainement du u-net
    
    📊 parametres de tuning importants:
    - num_epochs: nombre d'iterations (50-100 recommande)
    - lr: learning rate (1e-3 a 1e-4, avec decay)
    - batch_size: dans le dataloader (4-8 pour images ~1k x 1k)
    - weight_decay: regularisation l2 (1e-5)
    - class_weights: pour gerer le desequilibre background/cellules/bordures
    """
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 1.0, 2.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # learning rate scheduler pour diminuer le lr progressivement
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                      patience=5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # phase d'entrainement 🏋️
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'epoch {epoch+1}/{num_epochs}')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # forward
            optimizer.zero_grad()
            outputs = model(images)
            
            # resizer outputs pour matcher la taille de masks si necessaire
            if outputs.shape[2:] != masks.shape[2:]:
                outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
            
            # calculer la loss
            # convertir masks de (B, 3, H, W) a (B, H, W) avec indices de classe
            masks_labels = torch.argmax(masks, dim=1)
            loss = criterion(outputs, masks_labels)
            
            # backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # phase de validation 📊
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                
                # resizer outputs pour matcher la taille de masks si necessaire
                if outputs.shape[2:] != masks.shape[2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                
                masks_labels = torch.argmax(masks, dim=1)
                loss = criterion(outputs, masks_labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
        
        # sauvegarder le meilleur modele 💾
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print('✅ meilleur modele sauvegarde!')
        
        # ajuster le learning rate
        scheduler.step(val_loss)
    
    return train_losses, val_losses


# ============================================================================
# 🎯 post-traitement pour obtenir des instances
# ============================================================================

def postprocess_segmentation(prediction, min_size=250, threshold_cell=0.55, threshold_border=0.35):
    """
    post-traitement pour convertir la prediction en instances individuelles
    
    🔧 parametres de tuning (OPTIMISES):
    - min_size: taille minimale d'une cellule en pixels (250)
    - threshold_cell: seuil pour la classe cellule (0.55)
    - threshold_border: seuil pour les bordures (0.35)
    """
    # prediction shape: (3, H, W)
    prob_bg = prediction[0]
    prob_cell = prediction[1]
    prob_border = prediction[2]
    
    # masque binaire des cellules (SEUILS OPTIMISES)
    cell_mask = prob_cell > threshold_cell
    border_mask = prob_border > threshold_border
    
    # combiner: cellules sans bordures
    markers_mask = cell_mask & ~border_mask
    
    # marqueurs pour watershed
    markers = measure.label(markers_mask)
    
    # nettoyer les petits objets
    markers = morphology.remove_small_objects(markers, min_size=min_size//4)
    
    # watershed pour separer les cellules qui se touchent
    # utiliser la probabilite cellule comme elevation
    elevation = -prob_cell
    labels = watershed(elevation, markers, mask=cell_mask)
    
    # nettoyer les petites instances
    labels = morphology.remove_small_objects(labels, min_size=min_size)
    
    # re-numeroter les labels de 1 a n
    labels = measure.label(labels)
    
    return labels


# ============================================================================
# 🔍 tracking des cellules entre frames
# ============================================================================

class CellTracker:
    """
    🎯 tracking de cellules par correspondance iou entre frames consecutifs
    
    📐 parametres de tuning (OPTIMISES):
    - iou_threshold: seuil de chevauchement pour associer 2 cellules (0.25)
    - max_distance: distance maximale centroide en pixels (120)
    - min_track_length: longueur minimale d'une trackle (5-10 frames)
    """
    def __init__(self, iou_threshold=0.25, max_distance=120, distance_weight=0.2):
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.distance_weight = distance_weight
        self.tracks = {}  # {track_id: [(frame, cell_id, centroid, area), ...]}
        self.next_track_id = 1
        self.frame_cells = {}  # {frame: {cell_id: properties}}
    
    def compute_iou(self, mask1, mask2):
        """calcule l'intersection over union entre 2 masques"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0
        return intersection / union
    
    def compute_distance(self, centroid1, centroid2):
        """distance euclidienne entre 2 centroides"""
        return np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
    
    def extract_cell_properties(self, labels):
        """extraire les proprietes des cellules d'un frame"""
        props = measure.regionprops(labels)
        cells = {}
        for prop in props:
            cells[prop.label] = {
                'centroid': prop.centroid,
                'area': prop.area,
                'bbox': prop.bbox,
                'mask': labels == prop.label
            }
        return cells
    
    def track_frame(self, frame_idx, labels):
        """
        tracker les cellules d'un nouveau frame
        associe les cellules au frame precedent
        """
        current_cells = self.extract_cell_properties(labels)
        self.frame_cells[frame_idx] = current_cells
        
        if frame_idx == 0:
            # premier frame: creer de nouvelles tracks
            for cell_id, props in current_cells.items():
                self.tracks[self.next_track_id] = [(frame_idx, cell_id, props['centroid'], props['area'])]
                self.next_track_id += 1
        else:
            # frames suivants: associer aux tracks existantes
            prev_frame = frame_idx - 1
            if prev_frame not in self.frame_cells:
                return
            
            prev_cells = self.frame_cells[prev_frame]
            
            # trouver les tracks actives au frame precedent
            active_tracks = {}
            for track_id, track_data in self.tracks.items():
                if track_data[-1][0] == prev_frame:
                    prev_cell_id = track_data[-1][1]
                    if prev_cell_id in prev_cells:
                        active_tracks[track_id] = prev_cells[prev_cell_id]
            
            # associer cellules courantes aux tracks
            matched_tracks = set()
            matched_cells = set()
            
            # calculer matrice de similarite (iou + distance)
            for cell_id, cell_props in current_cells.items():
                best_track = None
                best_score = -1
                
                for track_id, prev_props in active_tracks.items():
                    if track_id in matched_tracks:
                        continue
                    
                    # iou entre masques
                    iou = self.compute_iou(cell_props['mask'], prev_props['mask'])
                    
                    # distance entre centroides
                    dist = self.compute_distance(cell_props['centroid'], prev_props['centroid'])
                    
                    # score combine (OPTIMISE)
                    if iou > self.iou_threshold and dist < self.max_distance:
                        score = iou - (dist / self.max_distance) * self.distance_weight
                        if score > best_score:
                            best_score = score
                            best_track = track_id
                
                # associer a la meilleure track
                if best_track is not None:
                    self.tracks[best_track].append((frame_idx, cell_id, 
                                                     cell_props['centroid'], 
                                                     cell_props['area']))
                    matched_tracks.add(best_track)
                    matched_cells.add(cell_id)
                else:
                    # nouvelle cellule: creer nouvelle track
                    self.tracks[self.next_track_id] = [(frame_idx, cell_id, 
                                                         cell_props['centroid'], 
                                                         cell_props['area'])]
                    self.next_track_id += 1
    
    def get_track_lineages(self):
        """obtenir les lignees de tracking au format cell tracking challenge"""
        lineages = []
        for track_id, track_data in self.tracks.items():
            start_frame = track_data[0][0]
            end_frame = track_data[-1][0]
            lineages.append({
                'id': track_id,
                'start': start_frame,
                'end': end_frame,
                'parent': 0,  # a determiner pour les divisions
                'frames': track_data
            })
        return lineages


# ============================================================================
# 🚀 pipeline principal
# ============================================================================

def main():
    """pipeline complet de segmentation et tracking"""
    
    print("🔬 pipeline de segmentation et tracking de cellules hela")
    print("=" * 60)
    
    # chemins
    train_images_01 = Path('Trainset/01')
    train_masks_01 = Path('Trainset/01_GT/SEG')
    train_images_02 = Path('Trainset/02')
    train_masks_02 = Path('Trainset/02_GT/SEG')
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 device: {device}")
    
    # creer les datasets
    print("\n📦 chargement des donnees...")
    dataset_01 = HeLaDataset(train_images_01, train_masks_01)
    dataset_02 = HeLaDataset(train_images_02, train_masks_02)
    
    # combiner les datasets
    full_dataset = torch.utils.data.ConcatDataset([dataset_01, dataset_02])
    
    # split train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"   train: {len(train_dataset)} images")
    print(f"   val: {len(val_dataset)} images")
    
    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)
    
    # creer le modele
    print("\n🏗️ creation du modele u-net...")
    model = UNet(n_channels=1, n_classes=3, base_filters=32).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   parametres: {total_params:,}")
    
    # entrainement
    print("\n🎓 entrainement du modele...")
    print("   parametres:")
    print("   - epochs: 50")
    print("   - learning rate: 1e-3")
    print("   - optimizer: adam avec weight decay 1e-5")
    print("   - class weights: [0.1, 1.0, 2.0] (bg, cell, border)")
    print("   - scheduler: reduce on plateau")
    
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=50, lr=1e-3, device=device
    )
    
    # sauvegarder les courbes d'entrainement
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('courbes d\'entrainement')
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("\n💾 courbes sauvegardees: training_curves.png")
    
    # inference sur une sequence de test
    print("\n🎯 inference sur testset/01...")
    model.load_state_dict(torch.load('best_unet_model.pth'))
    model.eval()
    
    test_dir = Path('Testset/01')
    test_files = sorted(list(test_dir.glob('*.tif')))
    
    # creer dossier de resultats
    results_dir = Path('Results')
    results_dir.mkdir(exist_ok=True)
    seg_dir = results_dir / 'SEG'
    seg_dir.mkdir(exist_ok=True)
    
    # tracking (PARAMETRES OPTIMISES)
    tracker = CellTracker(iou_threshold=0.25, max_distance=120, distance_weight=0.2)
    
    print(f"   traitement de {len(test_files)} frames...")
    for idx, img_path in enumerate(tqdm(test_files)):
        # charger et normaliser
        image = io.imread(str(img_path))
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # inference
        with torch.no_grad():
            img_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)
            output = model(img_tensor)
            
            # resizer output pour matcher la taille originale de l'image
            if output.shape[2:] != image.shape:
                output = F.interpolate(output, size=image.shape, mode='bilinear', align_corners=False)
            
            pred = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        # post-traitement (PARAMETRES OPTIMISES)
        labels = postprocess_segmentation(pred, min_size=250, threshold_cell=0.55, threshold_border=0.35)
        
        # tracking
        tracker.track_frame(idx, labels)
        
        # sauvegarder
        output_name = f"mask{img_path.stem.replace('t', '')}.tif"
        io.imsave(str(seg_dir / output_name), labels.astype(np.uint16), check_contrast=False)
    
    print(f"✅ segmentations sauvegardees dans {seg_dir}")
    
    # exporter les tracks
    print("\n🔍 export des tracks...")
    lineages = tracker.get_track_lineages()
    
    with open(results_dir / 'tracks.txt', 'w') as f:
        for lineage in lineages:
            f.write(f"{lineage['id']} {lineage['start']} {lineage['end']} {lineage['parent']}\n")
    
    print(f"✅ {len(lineages)} tracks exportees")
    print(f"\n🎉 pipeline termine avec succes!")
    print(f"   resultats dans: {results_dir}")


if __name__ == '__main__':
    main()

