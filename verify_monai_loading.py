#!/usr/bin/env python3
"""
Verification script to test MONAI dataset loading.
This script verifies that the dataset is correctly loaded with MONAI
and produces the expected output format.
"""

import os
import torch
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset as MONAIDataset
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped, Lambdad
from monai.transforms.intensity.dictionary import ScaleIntensityd
from monai.transforms.spatial.dictionary import Resized

# Configuration
DATASET_PATH = "dataset"
TRAIN_DIR = "dataset/Training"
VALID_DIR = "dataset/Validation"
TEST_DIR = "dataset/Testing"
IMAGE_SIZE = 224

# Valid image extensions
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def select_rgb_channels(image):
    """Select only the first 3 channels (RGB) from an image tensor."""
    return image[:3, :, :] if image.shape[0] > 3 else image


def discover_files(directory, classes, class_to_idx):
    """
    Discover image files in a directory organized by class folders.
    
    Args:
        directory: Root directory path
        classes: List of class names
        class_to_idx: Dictionary mapping class names to indices
    
    Returns:
        Tuple of (file_paths, labels)
    """
    files, labels = [], []
    for target_class in classes:
        class_dir = os.path.join(directory, target_class)
        if not os.path.isdir(class_dir):
            continue
        
        for fname in os.listdir(class_dir):
            # Filter only valid image files
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_IMAGE_EXTENSIONS:
                files.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[target_class])
    
    return files, labels

def verify_monai_loading():
    """Verify MONAI dataset loading produces expected output."""
    
    print("=" * 80)
    print("MONAI Dataset Loading Verification")
    print("=" * 80)
    
    # Define transforms
    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Lambdad(keys="image", func=select_rgb_channels),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(IMAGE_SIZE, IMAGE_SIZE)),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])
    
    # Discover classes and create file lists
    classes = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    # Discover files for each split
    train_files, train_labels = discover_files(TRAIN_DIR, classes, class_to_idx)
    valid_files, valid_labels = discover_files(VALID_DIR, classes, class_to_idx)
    test_files, test_labels = discover_files(TEST_DIR, classes, class_to_idx)
    
    # Create MONAI Datasets
    train_data = [{"image": img, "label": lab} for img, lab in zip(train_files, train_labels)]
    valid_data = [{"image": img, "label": lab} for img, lab in zip(valid_files, valid_labels)]
    test_data = [{"image": img, "label": lab} for img, lab in zip(test_files, test_labels)]
    
    train_dataset = MONAIDataset(data=train_data, transform=transform)
    valid_dataset = MONAIDataset(data=valid_data, transform=transform)
    test_dataset = MONAIDataset(data=test_data, transform=transform)
    
    # Print summary
    print(f"\nDataset carregado com MONAI: {len(train_dataset)} imagens de treino, "
          f"{len(valid_dataset)} de validação e {len(test_dataset)} de teste.")
    
    print(f"\nDEBUG: Lendo dados de: TRAIN='{TRAIN_DIR}', VALID='{VALID_DIR}', TEST='{TEST_DIR}'")
    print(f"\nDEBUG: Classes encontradas ({len(classes)}): {classes}")
    print(f"\nDEBUG: Mapeamento de classes: {class_to_idx}")
    print(f"\nDEBUG: Total de arquivos de treino: {len(train_files)}, "
          f"Validação: {len(valid_files)}, Teste: {len(test_files)}")
    
    # Verify sample
    train_loader = DataLoader(train_dataset, batch_size=1)
    sample = next(iter(train_loader))
    
    print(f"\nShape do tensor de imagem do MONAI: {sample['image'].shape}")
    print(f"\nTipo de dado do tensor: {sample['image'].dtype}")
    print(f"\nEstrutura do item do Dataset (chaves): {sample.keys()}")
    
    # Verify expected values
    print("\n" + "=" * 80)
    print("Verification Results")
    print("=" * 80)
    
    expected_train = 407
    expected_valid = 48
    expected_test = 54
    expected_classes = 7
    expected_shape = torch.Size([1, 3, 224, 224])
    expected_dtype = torch.float32
    
    checks = {
        "Training images count": len(train_dataset) == expected_train,
        "Validation images count": len(valid_dataset) == expected_valid,
        "Test images count": len(test_dataset) == expected_test,
        "Number of classes": len(classes) == expected_classes,
        "Image tensor shape": sample['image'].shape == expected_shape,
        "Tensor data type": sample['image'].dtype == expected_dtype,
        "Dataset has 'image' key": 'image' in sample,
        "Dataset has 'label' key": 'label' in sample,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All checks passed! MONAI dataset loading is working correctly.")
    else:
        print("✗ Some checks failed. Please review the implementation.")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    try:
        success = verify_monai_loading()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
