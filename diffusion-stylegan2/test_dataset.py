import sys
sys.path.insert(0, 'training')
from custom_dataset import FeatureSubsampledDataset

DATA_PATH = "/tmp/george_leilani"

print("Testing dataset...")
dataset = FeatureSubsampledDataset(
    path=DATA_PATH,
    file_ext='csv',
    resolution=64,
    subsamples_per_sample=5,
    features_per_subsample=10,
    method='one_sample'
)

image, label = dataset[0]
print(f"✓ Dataset works! Shape: {image.shape}, dtype: {image.dtype}")
