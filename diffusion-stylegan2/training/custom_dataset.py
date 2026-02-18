
import os
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from typing import List, Tuple
import PIL.Image

class FeatureSubsampledDataset(torch.utils.data.Dataset):
    """
    Dataset with optional feature subsampling for gene/spatial data.
    
    Compatible with Diffusion-GAN StyleGAN2 training pipeline.
    Supports both CSV gene data and image data.
    """

    def __init__(
        self,
        path: str,
        file_ext: str = "csv",
        resolution: int = 64, # GH: Set to None by default in StyleGAN -- should we change?
        subsamples_per_sample: int = 0,
        features_per_subsample: int = 0,
        method: str = "one_sample",
        **super_kwargs,  # Ignore extra kwargs from StyleGAN
    ):
        """
        Args:
            path: Path to data directory
            file_ext: File extension ('csv' or 'png')
            resolution: Image resolution (H x W)
            subsamples_per_sample: Number of subsamples to create per file (0 = disabled)
            features_per_subsample: Number of features/channels to keep per subsample
            method: 'one_sample' or 'two_sample'
        """

        print("GH: HELLO")

        # GH: Missing some other attributes present in the default StyleGAN Dataset class

        self._path = path
        self._file_ext = file_ext
        self._resolution = resolution
        self.subsamples_per_sample = subsamples_per_sample
        self.features_per_subsample = features_per_subsample
        self.method = method
        
        # Find all files
        self._all_fnames = []
        for fname in sorted(os.listdir(self._path)):
            if fname.endswith(f'.{file_ext}'):
                self._all_fnames.append(fname)
        
        if not self._all_fnames:
            raise ValueError(f"No {file_ext} files found in {path}")
        
        # Count channels (genes/features)
        if file_ext == "csv":
            sample_path = os.path.join(self._path, self._all_fnames[0])
            df = pd.read_csv(sample_path)
            self._num_channels = len(df.columns) - 2  # Subtract 'row' and 'col'
        elif file_ext in ["png", "jpg", "jpeg"]:
            sample_path = os.path.join(self._path, self._all_fnames[0])
            with PIL.Image.open(sample_path) as img:
                img_array = np.array(img)
                self._num_channels = 3 if len(img_array.shape) == 3 else 1
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        
        # Validate subsampling settings
        self.use_subsampling = self.subsamples_per_sample > 0
        if self.use_subsampling:
            if self.features_per_subsample <= 0:
                raise ValueError("features_per_subsample must be > 0 when subsampling")
            if self.features_per_subsample > self._num_channels:
                raise ValueError(
                    f"features_per_subsample ({self.features_per_subsample}) must be "
                    f"<= number of channels ({self._num_channels})"
                )
            if self.method not in {"one_sample", "two_sample"}:
                raise ValueError(
                    f"Unknown subsample method: {self.method}. "
                    f"Must be 'one_sample' or 'two_sample'."
                )
        else:
            self.subsamples_per_sample = 1
        
        # Set final number of channels after subsampling
        self._final_channels = (
            self.features_per_subsample if self.use_subsampling else self._num_channels
        )
    
    def __len__(self) -> int:
        return len(self._all_fnames) * self.subsamples_per_sample
    
    def __getitem__(self, idx: int):
        # Map index to base file
        base_index = idx // self.subsamples_per_sample
        fname = self._all_fnames[base_index]
        fpath = os.path.join(self._path, fname)
        
        # Load base sample
        base_sample = self._load_sample(fpath)
        
        # Apply subsampling if enabled
        if not self.use_subsampling:
            sample = base_sample
        elif self.method == "one_sample":
            sample = self._random_feature_subset(
                base_sample, self.features_per_subsample
            )
        else:  # method == "two_sample"
            other_index = random.randint(0, len(self._all_fnames) - 1)
            other_fpath = os.path.join(self._path, self._all_fnames[other_index])
            other_sample = self._load_sample(other_fpath)
            sample = self._random_mixed_feature_subset(
                base_sample, other_sample, self.features_per_subsample
            )
        
        # StyleGAN expects images as uint8 numpy arrays (H, W, C)
        # Convert from (C, H, W) torch tensor to (H, W, C) numpy array
        image = sample.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        image = self._normalize_to_uint8(image)
        
        return image.copy(), 0  # (image, label) - label unused
    
    def _load_sample(self, file_path: str) -> Tensor:
        """Load a sample as a torch tensor (C, H, W)"""
        if self._file_ext == "csv":
            return self._load_csv_as_tensor(file_path)
        else:
            return self._load_image_as_tensor(file_path)
    
    def _load_csv_as_tensor(self, file_path: str) -> Tensor:
        """Load CSV file into tensor (C, H, W)"""
        df = pd.read_csv(file_path)
        max_row = df["row"].max()
        max_col = df["col"].max()
        
        gene_columns = df.columns.difference(["row", "col"])
        nc = len(gene_columns)
        
        # Initialize tensor with expected resolution
        h, w = self._resolution, self._resolution
        tensor = np.zeros((nc, h, w), dtype=np.float32)
        
        # Fill tensor with data
        for i in range(len(df)):
            row, col = int(df["row"].iloc[i]), int(df["col"].iloc[i])
            if row < h and col < w:  # Bounds check
                for j, gene in enumerate(gene_columns):
                    tensor[j, row, col] = df[gene].iloc[i]
        
        return torch.tensor(tensor, dtype=torch.float32)
    
    def _load_image_as_tensor(self, file_path: str) -> Tensor:
        """Load image file into tensor (C, H, W)"""
        with PIL.Image.open(file_path) as img:
            img = img.resize((self._resolution, self._resolution), PIL.Image.LANCZOS)
            img_array = np.array(img)
            
            # Handle grayscale vs RGB
            if len(img_array.shape) == 2:
                img_array = img_array[np.newaxis, :, :]  # (H, W) -> (1, H, W)
            else:
                img_array = img_array.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            return torch.tensor(img_array, dtype=torch.float32)
    
    def _normalize_to_uint8(self, array: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 255] uint8 range"""
        # Normalize to [0, 1]
        min_val = array.min()
        max_val = array.max()
        if max_val > min_val:
            normalized = (array - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(array)
        
        # Scale to [0, 255] and convert to uint8
        return (normalized * 255).astype(np.uint8)
    
    @staticmethod
    def _random_feature_subset(sample: Tensor, features_per_subsample: int) -> Tensor:
        """Randomly select features from a single sample."""
        feature_indices = torch.randperm(sample.shape[0])[:features_per_subsample]
        return sample[feature_indices]
    
    @staticmethod
    def _random_mixed_feature_subset(
        sample_a: Tensor, sample_b: Tensor, features_per_subsample: int
    ) -> Tensor:
        """Randomly mix features from two samples."""
        if sample_a.shape != sample_b.shape:
            raise ValueError("sample_a and sample_b must have the same shape to mix")
        feature_indices = torch.randperm(sample_a.shape[0])[:features_per_subsample]
        subset_a = sample_a[feature_indices]
        subset_b = sample_b[feature_indices]
        choose_b = torch.rand(features_per_subsample) < 0.5
        choose_b = choose_b.view(-1, 1, 1)
        return torch.where(choose_b, subset_b, subset_a)
    
    # Properties required by StyleGAN
    @property
    def name(self):
        return "FeatureSubsampledDataset"
    
    @property
    def image_shape(self):
        """Return (C, H, W) expected by StyleGAN"""
        return [self._final_channels, self._resolution, self._resolution]
    
    @property
    def num_channels(self):
        return self._final_channels
    
    @property
    def resolution(self):
        return self._resolution
    
    @property
    def label_shape(self):
        return [0]  # No labels
    
    @property
    def label_dim(self):
        return 0
    
    @property
    def has_labels(self):
        return False
    
    @property
    def has_onehot_labels(self):
        return False
