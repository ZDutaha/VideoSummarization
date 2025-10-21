import os, numpy as np, torch
from typing import Dict, Any
from torch.utils.data import Dataset

class VideoSummaryDataset(Dataset):
    def __init__(self, root_dir:str, split:str="train", dataset_name:str=None):
        self.root_dir = root_dir
        self.split = split
        base = None
        if dataset_name is not None:
            base = os.path.join(root_dir, dataset_name, split)
        if base is None or (not os.path.isdir(base)):
            base = os.path.join(root_dir, "toy", split)
        self.dir = base
        self.files = sorted([f for f in os.listdir(self.dir) if f.endswith(".npz")])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx:int) -> Dict[str, Any]:
        path = os.path.join(self.dir, self.files[idx])
        data = np.load(path, allow_pickle=True)
        feats = data["feats"].astype("float32")
        labels = data["labels"].astype("float32")
        frame2sent = data["frame2sent"].astype("int64")
        texts = data["texts"]
        # texts may be float embedding (S,D) or list[str]
        if texts.dtype == object:
            texts = texts.tolist()
        else:
            texts = texts.astype("float32")
        return {
            "video_id": os.path.splitext(self.files[idx])[0],
            "feats": torch.from_numpy(feats),
            "labels": torch.from_numpy(labels),
            "frame2sent": torch.from_numpy(frame2sent),
            "texts": texts
        }
