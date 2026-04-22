"""
Dataset loader for Construction Site Object Detection
Supports loading from Arrow files or folder structure (images/ + annotations/)
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from datasets import Dataset as HFDataset


# Class mapping
CLASS_NAMES = ['excavator', 'rebar', 'worker_with_white_hard_hat']
CLASS_TO_IDX = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}  # 0 is background
NUM_CLASSES = len(CLASS_NAMES) + 1  # +1 for background


class ConstructionSiteDataset(Dataset):
    """
    Dataset for construction site object detection.
    
    Supports:
    - Arrow file format (HuggingFace datasets)
    - Folder format (images/ + annotations/)
    """
    
    def __init__(
        self,
        data_source: Union[str, Path, List[Union[str, Path]]],
        transform=None,
        max_samples: Optional[int] = None,
        source_type: str = 'auto'  # 'arrow', 'folder', or 'auto'
    ):
        """
        Args:
            data_source: Path to arrow file, folder, or list of paths
            transform: Image transforms
            max_samples: Maximum number of samples to load
            source_type: 'arrow', 'folder', or 'auto' (detect automatically)
        """
        self.transform = transform
        self.samples = []
        
        # Handle single path or list of paths
        if isinstance(data_source, (str, Path)):
            data_sources = [data_source]
        else:
            data_sources = data_source
        
        # Load from each source
        for source in data_sources:
            source = Path(source)
            
            # Determine source type
            if source_type == 'auto':
                if source.suffix == '.arrow':
                    self._load_from_arrow(source)
                elif (source / 'images').exists():
                    self._load_from_folder(source)
                else:
                    raise ValueError(f"Cannot determine source type for {source}")
            elif source_type == 'arrow':
                self._load_from_arrow(source)
            elif source_type == 'folder':
                self._load_from_folder(source)
        
        # Limit samples if specified
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples")
    
    def _load_from_arrow(self, arrow_path: Path):
        """Load samples from HuggingFace Arrow file."""
        ds = HFDataset.from_file(str(arrow_path))
        
        for idx in range(len(ds)):
            sample = ds[idx]
            self.samples.append({
                'type': 'arrow',
                'dataset': ds,
                'index': idx,
                'image_id': sample['image_id'],
            })
    
    def _load_from_folder(self, folder_path: Path):
        """Load samples from folder structure (images/ + annotations/)."""
        images_dir = folder_path / 'images'
        annotations_dir = folder_path / 'annotations'
        
        if not images_dir.exists():
            print(f"Warning: images folder not found at {images_dir}")
            return
        
        # Get all image files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        for img_path in sorted(image_files):
            image_id = img_path.stem
            ann_path = annotations_dir / f"{image_id}.json"
            
            if ann_path.exists():
                self.samples.append({
                    'type': 'folder',
                    'image_path': img_path,
                    'annotation_path': ann_path,
                    'image_id': image_id,
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        sample_info = self.samples[idx]
        
        if sample_info['type'] == 'arrow':
            return self._get_arrow_item(sample_info)
        else:
            return self._get_folder_item(sample_info)
    
    def _get_arrow_item(self, sample_info: Dict) -> Tuple[torch.Tensor, Dict]:
        """Get item from Arrow dataset."""
        ds = sample_info['dataset']
        sample = ds[sample_info['index']]
        
        # Get image
        image = sample['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image['path']).convert('RGB')
        else:
            image = image.convert('RGB')
        
        # Get image size
        img_w, img_h = image.size
        
        # Build target
        target = self._build_target(
            sample['excavator'],
            sample['rebar'],
            sample['worker_with_white_hard_hat'],
            img_w, img_h,
            sample_info['index']
        )
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
    def _get_folder_item(self, sample_info: Dict) -> Tuple[torch.Tensor, Dict]:
        """Get item from folder structure."""
        # Load image
        image = Image.open(sample_info['image_path']).convert('RGB')
        img_w, img_h = image.size
        
        # Load annotation
        with open(sample_info['annotation_path'], 'r') as f:
            ann = json.load(f)
        
        # Build target
        target = self._build_target(
            ann.get('excavator', []),
            ann.get('rebar', []),
            ann.get('worker_with_white_hard_hat', []),
            img_w, img_h,
            self.samples.index(sample_info)
        )
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
    def _build_target(
        self,
        excavator_boxes: List,
        rebar_boxes: List,
        worker_boxes: List,
        img_w: int,
        img_h: int,
        idx: int
    ) -> Dict:
        """Build target dictionary for object detection."""
        boxes = []
        labels = []
        
        # Process each class
        for box in excavator_boxes:
            if len(box) == 4:
                boxes.append(self._convert_box(box, img_w, img_h))
                labels.append(CLASS_TO_IDX['excavator'])
        
        for box in rebar_boxes:
            if len(box) == 4:
                boxes.append(self._convert_box(box, img_w, img_h))
                labels.append(CLASS_TO_IDX['rebar'])
        
        for box in worker_boxes:
            if len(box) == 4:
                boxes.append(self._convert_box(box, img_w, img_h))
                labels.append(CLASS_TO_IDX['worker_with_white_hard_hat'])
        
        # Handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
        }
        
        return target
    
    def _convert_box(self, box: List, img_w: int, img_h: int) -> List:
        """Convert normalized [x1, y1, x2, y2] to absolute coordinates."""
        x1, y1, x2, y2 = box
        return [
            x1 * img_w,
            y1 * img_h,
            x2 * img_w,
            y2 * img_h
        ]


def collate_fn(batch):
    """Custom collate function for object detection."""
    return tuple(zip(*batch))


def get_transform(train: bool = True):
    """Get image transforms."""
    from torchvision import transforms
    
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


if __name__ == "__main__":
    # Test dataset loading
    from torchvision import transforms
    
    transform = get_transform(train=False)
    
    # Test Arrow loading
    import os
    _data = Path(os.environ.get('CONSYNTH_DATA_ROOT', Path.home() / 'consynth_data'))
    print("Testing Arrow loading...")
    ds_arrow = ConstructionSiteDataset(
        str(_data / 'LouisChen15___construction_site' / 'construction_site-test.arrow'),
        transform=transform,
        max_samples=5
    )
    
    img, target = ds_arrow[0]
    print(f"Image shape: {img.shape}")
    print(f"Boxes: {target['boxes'].shape}")
    print(f"Labels: {target['labels']}")
    
    # Test folder loading
    print("\nTesting Folder loading...")
    ds_folder = ConstructionSiteDataset(
        str(_data / 'augmentation_data' / 'construction_site-test'),
        transform=transform,
        max_samples=5
    )
    
    img, target = ds_folder[0]
    print(f"Image shape: {img.shape}")
    print(f"Boxes: {target['boxes'].shape}")
    print(f"Labels: {target['labels']}")
