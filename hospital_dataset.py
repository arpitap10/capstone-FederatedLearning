"""
hospital_dataset.py - dataset objects over the federation CSVs.

Two dataset classes, both returning `(image, target, cue_dict, patient_id)`:
  * HospitalDataset(hospital, split)  - a client's train or local-val pool.
  * CsvImageDataset(csv_path)         - any CSV with a `patientId`/`Target`
                                        column (test sets, dev federation, probes).

`cue_dict` carries the three METADATA cues (view / sex / age_band) as integers
in {0,1} or -1 for "missing".  Image-derived cues are NOT computed here (they
are produced in model.predict_proba, which has the de-normalised pixels).

Images are read from a pre-rendered JPG when present, else decoded from DICOM
with pydicom (MONOCHROME1 inverted, per-image min-max to 8-bit).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import config
from cue_bank import METADATA_CUES
from hospital_transforms import get_transform

_METADATA_CUE_NAMES = ["view", "sex", "age_band"]


# ----------------------------------------------------------------------
# IMAGE LOADING
# ----------------------------------------------------------------------
def _load_image(patient_id: str) -> Image.Image:
    path = config.image_path(patient_id)
    if path.endswith(".jpg") or path.endswith(".png"):
        return Image.open(path).convert("L")
    return _load_dicom(path)


def _load_dicom(path: str) -> Image.Image:
    import pydicom

    dcm = pydicom.dcmread(path)
    arr = dcm.pixel_array.astype(np.float32)
    if str(getattr(dcm, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr  # invert so bone is bright
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    return Image.fromarray(arr.astype(np.uint8), mode="L")


# ----------------------------------------------------------------------
# CUE ENCODING
# ----------------------------------------------------------------------
def _encode_metadata_cues(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Vectorised metadata-cue encoding; missing -> -1 (int)."""
    out = {}
    for name in _METADATA_CUE_NAMES:
        fn = METADATA_CUES[name]
        vals = []
        for _, row in df.iterrows():
            v = fn(row)
            vals.append(-1 if v is None else int(v))
        out[name] = np.array(vals, dtype=np.int64)
    return out


# ----------------------------------------------------------------------
# BASE DATASET
# ----------------------------------------------------------------------
class _BaseFrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        assert "patientId" in self.df and "Target" in self.df, \
            "dataframe needs patientId and Target columns"
        self.transform = get_transform(train=train)
        self.ids = self.df["patientId"].astype(str).tolist()
        self.targets = self.df["Target"].astype(int).to_numpy()
        self.cue_arrays = _encode_metadata_cues(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        img = self.transform(_load_image(pid))
        target = float(self.targets[idx])
        cues = {name: int(self.cue_arrays[name][idx]) for name in _METADATA_CUE_NAMES}
        return img, target, cues, pid


class HospitalDataset(_BaseFrameDataset):
    """A client's local pool.  `split` in {'train','val'}."""

    def __init__(self, hospital_name: str, split: str = "train",
                 federation: str = "main", federation_dir: Optional[str] = None):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        # federation_dir, if given, is the full path to a federation (…/federations/<name>);
        # otherwise it is built from the federation name under FEDERATIONS_DIR.
        fed_dir = federation_dir or os.path.join(config.FEDERATIONS_DIR, federation)
        csv = os.path.join(fed_dir, "hospitals", hospital_name, f"{split}.csv")
        if not os.path.exists(csv):
            raise FileNotFoundError(f"{csv} - run build_federation.py first")
        df = pd.read_csv(csv, dtype={"patientId": str})
        super().__init__(df, train=(split == "train"))
        self.hospital_name = hospital_name
        self.split = split


class CsvImageDataset(_BaseFrameDataset):
    """Generic loader for a test/dev/probe CSV (eval transform, no augmentation)."""

    def __init__(self, csv_path: str, train: bool = False):
        df = pd.read_csv(csv_path, dtype={"patientId": str})
        super().__init__(df, train=train)
        self.csv_path = csv_path


# ----------------------------------------------------------------------
# COLLATE + LOADER FACTORY
# ----------------------------------------------------------------------
def collate_batch(batch):
    """Stack images/targets, batch cue dicts into LongTensors, keep ids as tuple."""
    imgs = torch.stack([b[0] for b in batch], 0)
    targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    cue_names = batch[0][2].keys()
    cues = {name: torch.tensor([b[2][name] for b in batch], dtype=torch.long)
            for name in cue_names}
    ids = tuple(b[3] for b in batch)
    return imgs, targets, cues, ids


def make_loader(dataset: Dataset, batch_size: int = 32, shuffle: bool = False,
                num_workers: int = 2) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_batch,
        drop_last=False,
    )
