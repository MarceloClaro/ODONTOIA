# MONAI Dataset Loading Implementation

## Overview
This document describes the MONAI-based dataset loading implementation for the ODONTO.IA project. The implementation provides robust medical image loading with proper preprocessing and augmentation capabilities.

## Implementation Details

### Dataset Structure
The dataset is organized in the following structure:
```
dataset/
├── Training/
│   ├── CaS/
│   ├── CoS/
│   ├── Gum/
│   ├── MC/
│   ├── OC/
│   ├── OLP/
│   └── OT/
├── Validation/
│   └── [same class structure]
└── Testing/
    └── [same class structure]
```

### Dataset Statistics
- **Training images**: 407
- **Validation images**: 48
- **Testing images**: 54
- **Number of classes**: 7
- **Classes**: CaS, CoS, Gum, MC, OC, OLP, OT

### Class Distribution
| Class | Training | Validation | Testing |
|-------|----------|------------|---------|
| CaS   | 63       | 8          | 8       |
| CoS   | 59       | 8          | 8       |
| Gum   | 47       | 7          | 7       |
| MC    | 72       | 9          | 9       |
| OC    | 42       | 6          | 6       |
| OLP   | 74       | 10         | 10      |
| OT    | 50       | 0          | 6       |

### MONAI Transforms Pipeline

#### Training Transforms
```python
train_transforms = Compose([
    LoadImaged(keys=["image"]),              # Load image from file
    EnsureChannelFirstd(keys=["image"]),     # Ensure channel-first format (C, H, W)
    Lambdad(keys="image", func=...),         # Keep only RGB channels (first 3)
    ScaleIntensityd(keys=["image"]),         # Normalize to [0, 1]
    Resized(keys=["image"], spatial_size=(224, 224)),  # Resize to 224x224
    EnsureTyped(keys=["image"], dtype=torch.float32),  # Ensure float32 dtype
])
```

#### Optional Augmentations (when selected)
```python
RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),     # Random horizontal flip
RandRotate90d(keys=["image"], prob=0.5, max_k=3),        # Random 90° rotations
RandZoomd(keys=["image"], prob=0.5, min_zoom=0.9, max_zoom=1.1),  # Random zoom
```

#### Validation/Test Transforms
Same as training transforms but without augmentations.

### Output Format

#### Dataset Item Structure
Each item in the dataset is a dictionary with the following keys:
- `'image'`: Tensor of shape `[C, H, W]` where C=3, H=224, W=224
- `'label'`: Integer label (0-6 corresponding to class index)

#### Batch Structure
When loaded through a DataLoader with batch_size=1:
- `sample['image']`: Tensor of shape `torch.Size([1, 3, 224, 224])`
- `sample['label']`: Tensor of shape `torch.Size([1])`
- Data type: `torch.float32`

### Expected Output
When running the dataset loading, you should see:

```
Dataset carregado com MONAI: 407 imagens de treino, 48 de validação e 54 de teste.

DEBUG: Lendo dados de: TRAIN='dataset/Training', VALID='dataset/Validation', TEST='dataset/Testing'

DEBUG: Classes encontradas (7): ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

DEBUG: Mapeamento de classes: {'CaS': 0, 'CoS': 1, 'Gum': 2, 'MC': 3, 'OC': 4, 'OLP': 5, 'OT': 6}

DEBUG: Total de arquivos de treino: 407, Validação: 48, Teste: 54

Shape do tensor de imagem do MONAI: torch.Size([1, 3, 224, 224])

Tipo de dado do tensor: torch.float32

Estrutura do item do Dataset (chaves): dict_keys(['image', 'label'])
```

## Verification

To verify the MONAI dataset loading is working correctly, run:

```bash
python verify_monai_loading.py
```

This script will:
1. Load the dataset using MONAI
2. Display dataset statistics
3. Verify tensor shapes and data types
4. Confirm all checks pass

## Benefits of MONAI

1. **Medical Image Optimized**: MONAI is specifically designed for medical imaging
2. **Robust Loading**: Better handling of various image formats and metadata
3. **Dictionary-based**: Clean interface with named keys ('image', 'label')
4. **Extensible**: Easy to add medical-specific transforms
5. **Performance**: Optimized for medical image pipelines
6. **Type Safety**: Explicit type enforcement (EnsureTyped)

## Integration with Training Pipeline

The MONAI datasets integrate seamlessly with the existing training pipeline:

1. **Dataset Creation**: MONAI datasets are created with transforms in `run_training_pipeline()`
2. **DataLoader Creation**: Standard PyTorch DataLoader wraps MONAI datasets
3. **Training Loop**: Accesses data via dictionary keys (`batch['image']`, `batch['label']`)
4. **Metrics Calculation**: Works with standard PyTorch tensor operations

## Code Location

- **Main Implementation**: `app.py` in the `run_training_pipeline()` function
- **Utility Functions**: `utils.py` - `select_rgb_channels()`, `discover_image_files()`
- **Configuration**: `config.py` - dataset path constants (`TRAIN_DIR`, `VALID_DIR`, `TEST_DIR`)
- **Verification Script**: `verify_monai_loading.py`

## References

- MONAI Documentation: https://docs.monai.io/
- MONAI Transforms: https://docs.monai.io/en/stable/transforms.html
- MONAI Dataset: https://docs.monai.io/en/stable/data.html
