#!/usr/bin/env python3
"""_dl_predict.py - DL inference subprocess (runs with higher thread count)
No scipy/sklearn imported here to avoid BLAS conflict.
Usage: python _dl_predict.py <output_dir> <data_npz> <models_json>
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import sys, json, gc, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

NUM_CLASSES = 4

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).unsqueeze(0), torch.tensor(self.y[idx], dtype=torch.long)

class CNN1D(nn.Module):
    def __init__(self, input_len, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 11, padding=5), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 9, padding=4), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.GELU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 5, padding=2), nn.BatchNorm1d(256), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

class Attention(nn.Module):
    def __init__(self, h):
        super().__init__(); self.attn = nn.Linear(h, 1)
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1); return (w * x).sum(dim=1)

class LSTMClassifier(nn.Module):
    def __init__(self, input_len, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding=4), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(4))
        self.lstm = nn.LSTM(32, 128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = Attention(256); self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.4), nn.Linear(64, num_classes))
    def forward(self, x):
        feat = self.features(x).permute(0,2,1); out, _ = self.lstm(feat)
        return self.classifier(self.attention(out))

class GRUClassifier(nn.Module):
    def __init__(self, input_len, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding=4), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(4))
        self.gru = nn.GRU(32, 128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = Attention(256); self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.4), nn.Linear(64, num_classes))
    def forward(self, x):
        feat = self.features(x).permute(0,2,1); out, _ = self.gru(feat)
        return self.classifier(self.attention(out))

DL_MODEL_CLS = {"CNN": CNN1D, "LSTM": LSTMClassifier, "GRU": GRUClassifier}


def main():
    output_dir = Path(sys.argv[1])
    data_npz = sys.argv[2]
    models_json = sys.argv[3]

    data = np.load(data_npz)
    X = data["X"]
    y = data["y"]
    N = len(y)

    with open(models_json) as f:
        model_specs = json.load(f)

    results = {}
    for spec in model_specs:
        name = spec["name"]
        model_path = spec["model_path"]
        t1 = time.time()
        print(f"  [subprocess] Loading {name}...", flush=True)
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            input_len = ckpt.get("input_len", X.shape[1])
            state_dict = ckpt["model_state_dict"]
            del ckpt; gc.collect()

            model = DL_MODEL_CLS[name](input_len, NUM_CLASSES)
            model.load_state_dict(state_dict, assign=True)
            model.eval()
            del state_dict; gc.collect()

            ds = AudioDataset(X, y)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
            all_preds = []
            with torch.no_grad():
                for xb, _ in loader:
                    all_preds.extend(model(xb).argmax(1).cpu().numpy().tolist())
            preds = np.array(all_preds, dtype=np.int64)
            elapsed = time.time() - t1
            print(f"    {name}: done ({elapsed:.1f}s)", flush=True)

            np.save(str(output_dir / f"_dl_preds_{name}.npy"), preds)
            results[name] = {"status": "ok", "time": elapsed}

            del model, ds, loader, all_preds, preds; gc.collect()
        except Exception as e:
            print(f"    {name}: FAILED - {e}", flush=True)
            results[name] = {"status": "error", "error": str(e)}

    with open(str(output_dir / "_dl_results.json"), "w") as f:
        json.dump(results, f)
    print("  [subprocess] Done!", flush=True)


if __name__ == "__main__":
    main()
