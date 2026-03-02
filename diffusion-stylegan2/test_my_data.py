import sys
sys.path.insert(0, 'training')
from custom_dataset import FeatureSubsampledDataset

DATA_PATH = "/myriadfs/home/sejjoff/Scratch/Diffusion-GAN/gene_data"

# Test without subsampling
dataset = FeatureSubsampledDataset(
    path=DATA_PATH,
    file_ext='csv',
    resolution=64,
    subsamples_per_sample=0,
    features_per_subsample=0,
)

image, label = dataset[0]
print(f"No subsampling: {len(dataset)} samples, shape {image.shape}, dtype {image.dtype}")

# Test with subsampling
dataset2 = FeatureSubsampledDataset(
    path=DATA_PATH,
    file_ext='csv',
    resolution=64,
    subsamples_per_sample=5,
    features_per_subsample=2,
    method='one_sample'
)

image2, label2 = dataset2[0]
print(f"With subsampling: {len(dataset2)} samples, shape {image2.shape}, dtype {image2.dtype}")
print("✓ Tests passed")