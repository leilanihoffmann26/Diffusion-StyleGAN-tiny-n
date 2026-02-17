import os
import random
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import PIL.Image

class FeatureSubsampledDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        file_ext: str = "csv",
        resolution: int = 64,
        subsamples_per_sample: int = 0,
        features_per_subsample: int = 0,
        method: str = "one_sample",
        **super_kwargs,
    ):
        self._path = path
        self._file_ext = file_ext
        self._resolution = resolution
        self.subsamples_per_sample = subsamples_per_sample
        self.features_per_subsample = features_per_subsample
        self.method = method
        
        self._all_fnames = sorted([
            fname for fname in os.listdir(self._path) 
            if fname.endswith(f'.{file_ext}')
        ])
        
        if not self._all_fnames:
            raise ValueError(f"No {file_ext} files found in {path}")
        
        print(f"[Dataset] Found {len(self._all_fnames)} {file_ext} files")
        
        if file_ext == "csv":
            sample_path = os.path.join(self._path, self._all_fnames[0])
            df = pd.read_csv(sample_path)
            self._num_channels = len(df.columns) - 2
            print(f"[Dataset] Detected {self._num_channels} genes/features")
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        
        self.use_subsampling = self.subsamples_per_sample > 0
        if self.use_subsampling:
            if self.features_per_subsample <= 0:
                raise ValueError("features_per_subsample must be > 0 when subsampling")
            if self.features_per_subsample > self._num_channels:
                raise ValueError(
                    f"features_per_subsample ({self.features_per_subsample}) "
                    f"must be <= channels ({self._num_channels})"
                )
            if self.method not in {"one_sample", "two_sample"}:
                raise ValueError(f"Unknown method: {self.method}")
            print(f"[Dataset] Subsampling: {self.subsamples_per_sample} per file, "
                  f"{self.features_per_subsample} features, method={self.method}")
        else:
            self.subsamples_per_sample = 1
            print("[Dataset] Subsampling disabled")
        
        self._final_channels = (
            self.features_per_subsample if self.use_subsampling else self._num_channels
        )
        
        print(f"[Dataset] Total: {len(self)} samples, {self._final_channels} channels each")
    
    def __len__(self) -> int:
        return len(self._all_fnames) * self.subsamples_per_sample
    
    def __getitem__(self, idx: int):
        base_index = idx // self.subsamples_per_sample
        fname = self._all_fnames[base_index]
        fpath = os.path.join(self._path, fname)
        
        base_sample = self._load_csv_as_tensor(fpath)
        
        if not self.use_subsampling:
            sample = base_sample
        elif self.method == "one_sample":
            sample = self._random_feature_subset(base_sample, self.features_per_subsample)
        else:
            other_index = random.randint(0, len(self._all_fnames) - 1)
            other_fpath = os.path.join(self._path, self._all_fnames[other_index])
            other_sample = self._load_csv_as_tensor(other_fpath)
            sample = self._random_mixed_feature_subset(
                base_sample, other_sample, self.features_per_subsample
            )
        
        image = sample.numpy()
        image = self._normalize_to_uint8(image)
        
        return image.copy(), 0
    
    def _load_csv_as_tensor(self, file_path: str) -> Tensor:
        df = pd.read_csv(file_path)
        max_row = df["row"].max()
        max_col = df["col"].max()
        
        gene_columns = df.columns.difference(["row", "col"])
        nc = len(gene_columns)
        
        h, w = self._resolution, self._resolution
        tensor = np.zeros((nc, h, w), dtype=np.float32)
        
        for i in range(len(df)):
            row = int(df["row"].iloc[i])
            col = int(df["col"].iloc[i])
            if row < h and col < w:
                for j, gene in enumerate(gene_columns):
                    tensor[j, row, col] = df[gene].iloc[i]
        
        return torch.tensor(tensor, dtype=torch.float32)
    
    def _normalize_to_uint8(self, array: np.ndarray) -> np.ndarray:
        min_val = array.min()
        max_val = array.max()
        if max_val > min_val:
            normalized = (array - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(array)
        return (normalized * 255).astype(np.uint8)
    
    @staticmethod
    def _random_feature_subset(sample: Tensor, features_per_subsample: int) -> Tensor:
        feature_indices = torch.randperm(sample.shape[0])[:features_per_subsample]
        return sample[feature_indices]
    
    @staticmethod
    def _random_mixed_feature_subset(
        sample_a: Tensor, sample_b: Tensor, features_per_subsample: int
    ) -> Tensor:
        if sample_a.shape != sample_b.shape:
            raise ValueError("Samples must have same shape")
        feature_indices = torch.randperm(sample_a.shape[0])[:features_per_subsample]
        subset_a = sample_a[feature_indices]
        subset_b = sample_b[feature_indices]
        choose_b = torch.rand(features_per_subsample) < 0.5
        choose_b = choose_b.view(-1, 1, 1)
        return torch.where(choose_b, subset_b, subset_a)
    
    @property
    def name(self):
        return "FeatureSubsampledDataset"
    
    @property
    def image_shape(self):
        return [self._final_channels, self._resolution, self._resolution]
    
    @property
    def num_channels(self):
        return self._final_channels
    
    @property
    def resolution(self):
        return self._resolution
    
    @property
    def label_shape(self):
        return [0]
    
    @property
    def label_dim(self):
        return 0
    
    @property
    def has_labels(self):
        return False
    
    @property
    def has_onehot_labels(self):
        return False
