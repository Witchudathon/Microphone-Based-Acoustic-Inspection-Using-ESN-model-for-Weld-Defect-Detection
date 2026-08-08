# ==================================================
# compare_models_cnn_esn_lstm_hybrid_viz.py
# เปรียบเทียบผลโมเดล:
#   - Filter + CNN
#   - Filter + ESN
#   - Filter + LSTM
#   - Hybrid CNN + ESN
# และวาด:
#   - Confusion Matrix + per-class accuracy
#   - Waveform ต่อกัน (raw vs filtered)
#   - Waveform ต่อ class (raw + filtered)
#   - Spectrogram ต่อ class (filtered)
# ใช้ข้อมูลจาก dataset/labels.csv ชุดเดียวกัน
# ==================================================

import csv
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import butter, filtfilt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import matplotlib as mpl

# ================== FONT CONFIG ===================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["xtick.labelsize"] = 11
mpl.rcParams["ytick.labelsize"] = 11
mpl.rcParams["legend.fontsize"] = 11

# ============= CONFIG =================
SAMPLE_RATE = 16000
DURATION_SEC = 1.5
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)

BASE_DIR = Path("dataset")
LABELS_CSV = BASE_DIR / "labels.csv"

# ลำดับ canonical ของ class (เวลา sort)
ALL_CLASSES = ["Normal", "High-Low", "HL+RS", "WrongSide"]

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CNN_MODEL_PATH = Path("chain_cnn_only.pth")
ESN_MODEL_PATH = Path("chain_esn_only.pth")
LSTM_MODEL_PATH = Path("chain_lstm_only.pth")
HYBRID_MODEL_PATH = Path("chain_hybrid_cnn_esn.pth")

CM_CNN_PNG = Path("cm_compare_cnn.png")
CM_ESN_PNG = Path("cm_compare_esn.png")
CM_LSTM_PNG = Path("cm_compare_lstm.png")
CM_HYBRID_PNG = Path("cm_compare_hybrid.png")

COMPARE_PNG = Path("compare_per_class_acc_cnn_esn_lstm_hybrid.png")

# รูป wave/spectrogram แบบรวมทุก class ปนกัน
WAVEFORM_PNG = Path("wave_raw_vs_filtered_concat.png")
SPECTRO_PNG = Path("spectrogram_per_class.png")

# รูป wave/spectrogram แบบ "แยกต่อ class" สไตล์ paper
RAW_WAVEFORM_CLASS_PNG = Path("wave_raw_per_class_concat.png")
WAVEFORM_CLASS_PNG = Path("wave_filtered_per_class_concat.png")
SPECTRO_CLASS_PNG = Path("spectrogram_filtered_per_class_concat.png")

# ฟิลเตอร์ที่ใช้กับเสียง
HPF_CUTOFF = 100.0  # Hz
HPF_ORDER = 4

ESN_HIDDEN_SIZE = 256  # default (สำหรับ ESN-only); hybrid จะอ่านจาก checkpoint
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2  # ให้ตรงกับ train_filter_lstm.py
LSTM_BIDIR = True

# จำนวนคลิปสูงสุดต่อ class ที่จะเอามาต่อกัน (เหมือนใน paper: ให้ waveform ยาวพอ)
MAX_SEGMENTS_PER_CLASS = 3


# ============= FILTER =================
def highpass_filter(
    audio: np.ndarray, sr: int, cutoff: float = HPF_CUTOFF, order: int = HPF_ORDER
):
    nyq = 0.5 * sr
    norm_cut = cutoff / nyq
    b, a = butter(order, norm_cut, btype="highpass", analog=False)
    filtered = filtfilt(b, a, audio).astype(np.float32)
    return filtered


# ============= DATASET =================
class ChainAudioDataset(Dataset):
    """
    ใช้ dataset เดียวกับตอนเทรน แต่ไม่มี augment
    เอาไว้สำหรับเทียบโมเดลทั้งหมด
    """

    def __init__(self, rows, class_to_idx):
        self.rows = rows
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.rows)

    def _load_audio_filtered(self, path_str):
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(path)

        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)
        audio = highpass_filter(audio, sr=SAMPLE_RATE)
        return audio

    def _fix_length(self, audio: np.ndarray):
        if len(audio) < NUM_SAMPLES:
            pad_width = NUM_SAMPLES - len(audio)
            audio = np.pad(audio, (0, pad_width), mode="constant")
        elif len(audio) > NUM_SAMPLES:
            # สำหรับ eval ใช้ตัดกลางให้ deterministic
            start = (len(audio) - NUM_SAMPLES) // 2
            audio = audio[start : start + NUM_SAMPLES]
        return audio

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = row["file_path"]
        class_name = row["class_name"]

        x = self._load_audio_filtered(path)
        x = self._fix_length(x)

        x = torch.from_numpy(x).unsqueeze(0)  # [1, T]
        y = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)
        return x, y


# ============= MODELS =================
class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # [B, 64]
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out


class ESNLayer(nn.Module):
    """
    Echo State Network layer (reservoir fixed, train only readout)
    Input:  [B, T, input_size]
    Output: [B, hidden_size]  (last state)
    """

    def __init__(self, input_size, hidden_size, alpha=0.9, spectral_radius=0.9):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha

        w_in = np.random.uniform(-0.5, 0.5, size=(hidden_size, input_size)).astype(
            np.float32
        )
        w_res = np.random.uniform(-0.5, 0.5, size=(hidden_size, hidden_size)).astype(
            np.float32
        )

        eigvals = np.linalg.eigvals(w_res)
        radius = np.max(np.abs(eigvals)).real
        if radius == 0:
            radius = 1.0
        w_res = (w_res / radius * spectral_radius).astype(np.float32)

        self.register_buffer("W_in", torch.from_numpy(w_in))  # [H, D]
        self.register_buffer("W_res", torch.from_numpy(w_res))  # [H, H]

    def forward(self, x):
        B, T, D = x.shape
        H = self.hidden_size

        h = x.new_zeros(B, H)
        for t in range(T):
            u_t = x[:, t, :]
            pre_act = torch.matmul(u_t, self.W_in.T) + torch.matmul(h, self.W_res.T)
            h_new = torch.tanh(pre_act)
            h = (1.0 - self.alpha) * h + self.alpha * h_new
        return h


class ESNClassifier(nn.Module):
    """
    Filter + ESN classifier
    Input:  [B, 1, T] → ESN เห็น [B, T, 1]
    """

    def __init__(self, num_classes: int, esn_hidden_size: int = ESN_HIDDEN_SIZE):
        super().__init__()
        self.esn = ESNLayer(
            input_size=1, hidden_size=esn_hidden_size, alpha=0.9, spectral_radius=0.9
        )
        self.classifier = nn.Sequential(
            nn.Linear(esn_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x_seq = x.permute(0, 2, 1)  # [B, T, 1]
        h = self.esn(x_seq)  # [B, H]
        out = self.classifier(h)
        return out


class LSTMClassifier(nn.Module):
    """
    Filter + BiLSTM classifier
    Input:  [B, 1, T] → LSTM เห็น [B, T, 1]
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_LAYERS,
        bidirectional: bool = LSTM_BIDIR,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        fc_in_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(fc_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [B, 1, T] -> [B, T, 1]
        x_seq = x.permute(0, 2, 1)
        out, (h_n, c_n) = self.lstm(x_seq)  # h_n: [num_layers*num_directions, B, H]

        if self.bidirectional:
            # concat last forward & last backward
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h_last = torch.cat([h_forward, h_backward], dim=1)  # [B, 2H]
        else:
            h_last = h_n[-1]  # [B, H]

        logits = self.classifier(h_last)
        return logits


class HybridCNNESNClassifier(nn.Module):
    """
    Hybrid: 1D-CNN backbone (หลัก) + ESN (เสริม temporal)
    Input:  [B, 1, T]
    Flow:
        x -> CNN features: [B, C, T']
        CNN branch: GAP -> [B, C]
        ESN branch: permute [B, T', C] -> ESN -> [B, H]
        concat [C, H] -> classifier -> logits
    """

    def __init__(
        self,
        num_classes: int,
        esn_hidden_size: int,
        esn_alpha: float = 0.9,
        esn_spectral_radius: float = 0.9,
    ):
        super().__init__()
        # CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        cnn_channels = 64

        # ESN เห็น feature sequence จาก CNN
        self.esn = ESNLayer(
            input_size=cnn_channels,
            hidden_size=esn_hidden_size,
            alpha=esn_alpha,
            spectral_radius=esn_spectral_radius,
        )

        combined_dim = cnn_channels + esn_hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)  # [B, C, T']
        cnn_vec = self.gap(feat).squeeze(-1)  # [B, C]
        seq = feat.permute(0, 2, 1)  # [B, T', C]
        esn_vec = self.esn(seq)  # [B, H]
        combined = torch.cat([cnn_vec, esn_vec], dim=1)
        out = self.classifier(combined)
        return out


# ============= UTILS =================
def load_rows_and_classes(labels_csv: Path):
    rows = []
    class_set = set()

    with labels_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cname = row["class_name"]
            if cname in ALL_CLASSES:
                rows.append(row)
                class_set.add(cname)

    classes_in_use = sorted(list(class_set), key=lambda c: ALL_CLASSES.index(c))
    return rows, classes_in_use


def plot_confusion_matrix(cm, classes, title, save_path: Path):
    plt.figure(figsize=(6, 5), facecolor="white")
    plt.imshow(cm, interpolation="nearest", cmap="Oranges")
    plt.title(title, fontname="Times New Roman")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right", fontname="Times New Roman")
    plt.yticks(tick_marks, classes, fontname="Times New Roman")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                fontname="Times New Roman",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )

    plt.ylabel("True label", fontname="Times New Roman")
    plt.xlabel("Predicted label", fontname="Times New Roman")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor="white")
    plt.close()
    print(f"📊 Saved Confusion Matrix plot -> {save_path}")


def per_class_accuracy(cm):
    per_class_acc = []
    for i in range(cm.shape[0]):
        total = cm[i].sum()
        correct = cm[i, i]
        acc = correct / total if total > 0 else 0.0
        per_class_acc.append(acc)
    return np.array(per_class_acc)


# ============= VISUALIZATION 1:
# raw vs filtered ปนกันหลาย class + spectrogram ทีละคลิป =============
def visualize_raw_vs_filtered_and_spectrogram(rows, classes_in_use):
    """
    เลือกตัวอย่าง 4–5 คลิป (อย่างน้อย 1 ต่อ class ถ้าเป็นไปได้)
    - วาด waveform raw + filtered ต่อกันยาว ๆ
    - วาด spectrogram แยกตาม clip
    """
    print("\n=== Visualizing raw vs filtered waveforms & spectrograms ===")

    max_examples_per_class = 1
    max_total_examples = 5

    selected = []
    picked_per_class = {c: 0 for c in classes_in_use}

    for row in rows:
        cname = row["class_name"]
        if cname not in picked_per_class:
            continue
        if picked_per_class[cname] < max_examples_per_class:
            selected.append(row)
            picked_per_class[cname] += 1
        if len(selected) >= max_total_examples:
            break

    if len(selected) == 0:
        print("⚠ ไม่พบตัวอย่างสำหรับวาดรูปคลื่น/สเปกโทรแกรม")
        return

    print(f"Using {len(selected)} clips for visualization.")

    raw_segments = []
    filt_segments = []
    seg_classes = []

    for row in selected:
        path = Path(row["file_path"])
        cname = row["class_name"]
        if not path.exists():
            print(f"  ⚠ File not found for visualization: {path}")
            continue

        audio_raw, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        audio_raw = audio_raw.astype(np.float32)

        # fix length raw
        if len(audio_raw) < NUM_SAMPLES:
            pad_width = NUM_SAMPLES - len(audio_raw)
            audio_raw_fixed = np.pad(audio_raw, (0, pad_width), mode="constant")
        elif len(audio_raw) > NUM_SAMPLES:
            start = (len(audio_raw) - NUM_SAMPLES) // 2
            audio_raw_fixed = audio_raw[start : start + NUM_SAMPLES]
        else:
            audio_raw_fixed = audio_raw

        # filtered
        audio_filt = highpass_filter(audio_raw_fixed, sr=SAMPLE_RATE)

        raw_segments.append(audio_raw_fixed)
        filt_segments.append(audio_filt)
        seg_classes.append(cname)

    if len(raw_segments) == 0:
        print("⚠ No valid segments for visualization.")
        return

    raw_concat = np.concatenate(raw_segments, axis=0)
    filt_concat = np.concatenate(filt_segments, axis=0)

    total_len = raw_concat.shape[0]
    time_axis = np.arange(total_len) / SAMPLE_RATE

    # ขอบเขตแต่ละ segment
    boundaries = []
    start_idx = 0
    for seg in raw_segments:
        end_idx = start_idx + len(seg)
        boundaries.append((start_idx, end_idx))
        start_idx = end_idx

    # -------- Plot waveform raw vs filtered --------
    plt.figure(figsize=(12, 6), facecolor="white")

    # Raw
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(time_axis, raw_concat, linewidth=0.8)
    ax1.set_title("Concatenated Raw Waveform", fontname="Times New Roman")
    ax1.set_ylabel("Amplitude", fontname="Times New Roman")
    for (s, e), cname in zip(boundaries, seg_classes):
        t_s = s / SAMPLE_RATE
        t_e = e / SAMPLE_RATE
        ax1.axvline(t_s, color="gray", linestyle="--", alpha=0.5)
        ax1.text(
            (t_s + t_e) / 2,
            ax1.get_ylim()[1] * 0.8,
            cname,
            ha="center",
            va="center",
            fontname="Times New Roman",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    ax1.grid(alpha=0.3)

    # Filtered
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(time_axis, filt_concat, linewidth=0.8)
    ax2.set_title(
        "Concatenated High-pass Filtered Waveform (100 Hz cut-off)",
        fontname="Times New Roman",
    )
    ax2.set_xlabel("Time (seconds)", fontname="Times New Roman")
    ax2.set_ylabel("Amplitude", fontname="Times New Roman")
    for (s, _), _cname in zip(boundaries, seg_classes):
        t_s = s / SAMPLE_RATE
        ax2.axvline(t_s, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(WAVEFORM_PNG, dpi=150, facecolor="white")
    plt.close()
    print(f"📈 Saved raw vs filtered waveform plot -> {WAVEFORM_PNG}")

    # -------- Plot spectrogram per clip --------
    n_examples = len(filt_segments)
    n_cols = min(n_examples, 4)
    n_rows = int(np.ceil(n_examples / n_cols))

    plt.figure(figsize=(4 * n_cols, 3 * n_rows), facecolor="white")

    for idx, (audio_filt, cname) in enumerate(zip(filt_segments, seg_classes), start=1):
        plt.subplot(n_rows, n_cols, idx)
        D = librosa.stft(audio_filt, n_fft=512, hop_length=128, win_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        plt.imshow(
            S_db,
            aspect="auto",
            origin="lower",
            extent=[
                0,
                len(audio_filt) / SAMPLE_RATE,
                0,
                SAMPLE_RATE / 2,
            ],
            cmap="inferno",
        )
        plt.title(f"Class: {cname}", fontname="Times New Roman")
        plt.xlabel("Time (s)", fontname="Times New Roman")
        plt.ylabel("Frequency [Hz]", fontname="Times New Roman")
        cbar = plt.colorbar()
        cbar.set_label("Intensity [dB]", fontname="Times New Roman")

    plt.tight_layout()
    plt.savefig(SPECTRO_PNG, dpi=150, facecolor="white")
    plt.close()
    print(f"🎛 Saved spectrogram per clip plot -> {SPECTRO_PNG}")


# ============= VISUALIZATION 2:
# สไตล์ paper: waveform ต่อ class + spectrogram ต่อ class ============
def visualize_per_class_concat_wave_and_spectrogram(
    rows,
    classes_in_use,
    max_segments_per_class: int = MAX_SEGMENTS_PER_CLASS,
):
    """
    วาดรูปแบบคล้าย paper:
      - Waveform ก่อนฟิลเตอร์ ต่อกันสำหรับแต่ละ class (1 แถวต่อ 1 class)
      - Waveform หลังฟิลเตอร์ ต่อกันสำหรับแต่ละ class
      - Spectrogram ต่อ class พร้อม colorbar Intensity [dB]
    """

    print("\n=== Visualizing per-class concatenated waveforms & spectrograms ===")

    # จัดกลุ่ม path ตาม class
    class_to_files = {c: [] for c in classes_in_use}
    for row in rows:
        cname = row["class_name"]
        if cname in class_to_files:
            class_to_files[cname].append(row["file_path"])

    # เตรียมสัญญาณ raw / filtered ต่อ class
    class_wave_raw = {}  # cname -> np.ndarray (ก่อนฟิลเตอร์)
    class_wave_filt = {}  # cname -> np.ndarray (หลังฟิลเตอร์)

    for cname in classes_in_use:
        files = class_to_files.get(cname, [])
        if len(files) == 0:
            print(f"  ⚠ no files for class {cname}, skip.")
            continue

        raw_segments = []
        filt_segments = []
        for path_str in files[:max_segments_per_class]:
            path = Path(path_str)
            if not path.exists():
                print(f"  ⚠ file not found for class {cname}: {path}")
                continue

            audio_raw, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            audio_raw = audio_raw.astype(np.float32)

            # fix length ให้เท่ากับ NUM_SAMPLES ก่อน filter
            if len(audio_raw) < NUM_SAMPLES:
                pad_width = NUM_SAMPLES - len(audio_raw)
                audio_raw_fixed = np.pad(audio_raw, (0, pad_width), mode="constant")
            elif len(audio_raw) > NUM_SAMPLES:
                start = (len(audio_raw) - NUM_SAMPLES) // 2
                audio_raw_fixed = audio_raw[start : start + NUM_SAMPLES]
            else:
                audio_raw_fixed = audio_raw

            audio_filt = highpass_filter(audio_raw_fixed, sr=SAMPLE_RATE)

            raw_segments.append(audio_raw_fixed)
            filt_segments.append(audio_filt)

        if len(raw_segments) == 0:
            print(f"  ⚠ no valid segments for class {cname}")
            continue

        class_wave_raw[cname] = np.concatenate(raw_segments, axis=0)
        class_wave_filt[cname] = np.concatenate(filt_segments, axis=0)

    if len(class_wave_raw) == 0:
        print("⚠ No valid per-class waveforms for visualization.")
        return

    n_classes = len(class_wave_raw)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    # ----------------- Waveform RAW ต่อ class -----------------
    fig, axes = plt.subplots(
        n_classes,
        1,
        figsize=(11, 2.4 * n_classes),
        sharex=True,
        facecolor="white",
    )
    if n_classes == 1:
        axes = [axes]

    fig.suptitle(
        "Waveform Before High-pass Filtering",
        fontname="Times New Roman",
        fontsize=14,
    )

    axis_list = []
    for idx, cname in enumerate(classes_in_use):
        if cname not in class_wave_raw:
            continue
        sig = class_wave_raw[cname]
        t = np.arange(len(sig)) / SAMPLE_RATE
        ax = axes[len(axis_list)]
        axis_list.append(ax)

        ax.plot(t, sig, linewidth=0.8, color=colors[idx % len(colors)])
        ax.set_ylabel("Amplitude", fontname="Times New Roman")
        ax.set_title(f"{cname} Sound", fontname="Times New Roman", fontsize=12)
        ax.grid(alpha=0.3)

    if axis_list:
        axis_list[-1].set_xlabel("Time (s)", fontname="Times New Roman")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(RAW_WAVEFORM_CLASS_PNG, dpi=150, facecolor="white")
    plt.close()
    print(f"📈 Saved per-class RAW waveform plot -> {RAW_WAVEFORM_CLASS_PNG}")

    # ----------------- Waveform FILTERED ต่อ class -----------------
    fig, axes = plt.subplots(
        n_classes,
        1,
        figsize=(11, 2.4 * n_classes),
        sharex=True,
        facecolor="white",
    )
    if n_classes == 1:
        axes = [axes]

    fig.suptitle(
        "Waveform After High-pass Filtering (100 Hz)",
        fontname="Times New Roman",
        fontsize=14,
    )

    axis_list = []
    for idx, cname in enumerate(classes_in_use):
        if cname not in class_wave_filt:
            continue
        sig = class_wave_filt[cname]
        t = np.arange(len(sig)) / SAMPLE_RATE
        ax = axes[len(axis_list)]
        axis_list.append(ax)

        ax.plot(t, sig, linewidth=0.8, color=colors[idx % len(colors)])
        ax.set_ylabel("Amplitude", fontname="Times New Roman")
        ax.set_title(f"{cname} Sound", fontname="Times New Roman", fontsize=12)
        ax.grid(alpha=0.3)

    if axis_list:
        axis_list[-1].set_xlabel("Time (s)", fontname="Times New Roman")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(WAVEFORM_CLASS_PNG, dpi=150, facecolor="white")
    plt.close()
    print(f"📈 Saved per-class filtered waveform plot -> {WAVEFORM_CLASS_PNG}")

    # ----------------- Spectrogram ต่อ class (ใช้สัญญาณ filtered) -----------------
    # เตรียมค่า vmin/vmax จากทุก class เพื่อให้ scale สีเหมือนกัน
    all_S_db = []
    for cname, sig in class_wave_filt.items():
        D = librosa.stft(sig, n_fft=512, hop_length=128, win_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        all_S_db.append(S_db)

    stacked = np.concatenate(all_S_db, axis=1)
    vmin = stacked.min()
    vmax = stacked.max()

    fig, axes = plt.subplots(
        n_classes,
        1,
        figsize=(11, 2.6 * n_classes),
        sharex=True,
        facecolor="white",
    )
    if n_classes == 1:
        axes = [axes]

    fig.suptitle(
        "Spectrograms After High-pass Filtering (100 Hz)",
        fontname="Times New Roman",
        fontsize=14,
    )

    cmap = plt.get_cmap("turbo")

    axis_list = []
    for idx, cname in enumerate(classes_in_use):
        if cname not in class_wave_filt:
            continue

        sig = class_wave_filt[cname]
        D = librosa.stft(sig, n_fft=512, hop_length=128, win_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        ax = axes[len(axis_list)]
        axis_list.append(ax)

        img = ax.imshow(
            S_db,
            aspect="auto",
            origin="lower",
            extent=[0, len(sig) / SAMPLE_RATE, 0, SAMPLE_RATE / 2],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_ylabel("Frequency [Hz]", fontname="Times New Roman")
        ax.set_title(f"{cname} Sound", fontname="Times New Roman", fontsize=12)

        cbar = fig.colorbar(
            img,
            ax=ax,
            location="right",
            fraction=0.025,
            pad=0.01,
        )
        cbar.set_label("Intensity [dB]", fontname="Times New Roman", color="black")
        cbar.ax.set_facecolor("white")
        cbar.outline.set_edgecolor("black")
        for t in cbar.ax.get_yticklabels():
            t.set_color("black")
            t.set_fontname("Times New Roman")
            t.set_fontsize(9)

    if axis_list:
        axis_list[-1].set_xlabel("Time [sec]", fontname="Times New Roman")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(SPECTRO_CLASS_PNG, dpi=150, facecolor="white")
    plt.close()
    print(f"🎛 Saved per-class spectrogram plot -> {SPECTRO_CLASS_PNG}")


# ============= MAIN =================
def main():
    print("\n=== Compare CNN vs ESN vs LSTM vs Hybrid CNN+ESN on same dataset ===")
    print(f"Device: {DEVICE}")

    if not LABELS_CSV.exists():
        print(f"⚠ labels.csv not found at: {LABELS_CSV}")
        return

    if (
        not CNN_MODEL_PATH.exists()
        or not ESN_MODEL_PATH.exists()
        or not LSTM_MODEL_PATH.exists()
        or not HYBRID_MODEL_PATH.exists()
    ):
        print("⚠ ไม่พบไฟล์โมเดลอย่างน้อยหนึ่งตัว:")
        print(f"  CNN   : {CNN_MODEL_PATH.exists()}")
        print(f"  ESN   : {ESN_MODEL_PATH.exists()}")
        print(f"  LSTM  : {LSTM_MODEL_PATH.exists()}")
        print(f"  Hybrid: {HYBRID_MODEL_PATH.exists()}")
        return

    # ----- Load dataset -----
    rows, classes_in_use = load_rows_and_classes(LABELS_CSV)
    print(f"Total clips (valid classes): {len(rows)}")
    print(f"Classes in use            : {classes_in_use}")

    if len(rows) == 0 or len(classes_in_use) == 0:
        print("⚠ No valid data or no classes found. Check labels.csv.")
        return

    class_to_idx = {c: i for i, c in enumerate(classes_in_use)}

    test_ds = ChainAudioDataset(rows, class_to_idx)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ----- Load CNN checkpoint -----
    cnn_ckpt = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
    cnn_classes = cnn_ckpt["classes"]
    if cnn_classes != classes_in_use:
        print("⚠ classes_in_use ของ CNN ไม่ตรงกับ labels.csv ปัจจุบัน")
        print("  cnn_classes:", cnn_classes)
        print("  csv_classes:", classes_in_use)
        return

    cnn_model = CNNClassifier(num_classes=len(classes_in_use)).to(DEVICE)
    cnn_model.load_state_dict(cnn_ckpt["model_state"])
    cnn_model.eval()

    # ----- Load ESN checkpoint -----
    esn_ckpt = torch.load(ESN_MODEL_PATH, map_location=DEVICE)
    esn_classes = esn_ckpt["classes"]
    if esn_classes != classes_in_use:
        print("⚠ classes_in_use ของ ESN ไม่ตรงกับ labels.csv ปัจจุบัน")
        print("  esn_classes:", esn_classes)
        print("  csv_classes:", classes_in_use)
        return

    esn_model = ESNClassifier(num_classes=len(classes_in_use)).to(DEVICE)
    esn_model.load_state_dict(esn_ckpt["model_state"])
    esn_model.eval()

    # ----- Load LSTM checkpoint -----
    lstm_ckpt = torch.load(LSTM_MODEL_PATH, map_location=DEVICE)
    lstm_classes = lstm_ckpt["classes"]
    if lstm_classes != classes_in_use:
        print("⚠ classes_in_use ของ LSTM ไม่ตรงกับ labels.csv ปัจจุบัน")
        print("  lstm_classes:", lstm_classes)
        print("  csv_classes:", classes_in_use)
        return

    lstm_hidden_size = lstm_ckpt.get("lstm_hidden_size", LSTM_HIDDEN_SIZE)
    lstm_num_layers = lstm_ckpt.get("lstm_num_layers", LSTM_LAYERS)
    lstm_bidirectional = lstm_ckpt.get("lstm_bidirectional", LSTM_BIDIR)

    lstm_model = LSTMClassifier(
        num_classes=len(classes_in_use),
        hidden_size=lstm_hidden_size,
        num_layers=lstm_num_layers,
        bidirectional=lstm_bidirectional,
    ).to(DEVICE)
    lstm_model.load_state_dict(lstm_ckpt["model_state"])
    lstm_model.eval()

    # ----- Load Hybrid checkpoint -----
    hybrid_ckpt = torch.load(HYBRID_MODEL_PATH, map_location=DEVICE)
    hybrid_classes = hybrid_ckpt["classes"]
    if hybrid_classes != classes_in_use:
        print("⚠ classes_in_use ของ Hybrid ไม่ตรงกับ labels.csv ปัจจุบัน")
        print("  hybrid_classes:", hybrid_classes)
        print("  csv_classes:", classes_in_use)
        return

    esn_hidden_size_h = hybrid_ckpt.get("esn_hidden_size", 256)
    esn_alpha_h = hybrid_ckpt.get("esn_alpha", 0.9)
    esn_spectral_radius_h = hybrid_ckpt.get("esn_spectral_radius", 0.9)

    hybrid_model = HybridCNNESNClassifier(
        num_classes=len(classes_in_use),
        esn_hidden_size=esn_hidden_size_h,
        esn_alpha=esn_alpha_h,
        esn_spectral_radius=esn_spectral_radius_h,
    ).to(DEVICE)
    hybrid_model.load_state_dict(hybrid_ckpt["model_state"])
    hybrid_model.eval()

    # ----- Run inference for all models -----
    all_y = []
    all_pred_cnn = []
    all_pred_esn = []
    all_pred_lstm = []
    all_pred_hybrid = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits_cnn = cnn_model(x)
            logits_esn = esn_model(x)
            logits_lstm = lstm_model(x)
            logits_hybrid = hybrid_model(x)

            pred_cnn = logits_cnn.argmax(dim=1)
            pred_esn = logits_esn.argmax(dim=1)
            pred_lstm = logits_lstm.argmax(dim=1)
            pred_hybrid = logits_hybrid.argmax(dim=1)

            all_y.extend(y.cpu().numpy().tolist())
            all_pred_cnn.extend(pred_cnn.cpu().numpy().tolist())
            all_pred_esn.extend(pred_esn.cpu().numpy().tolist())
            all_pred_lstm.extend(pred_lstm.cpu().numpy().tolist())
            all_pred_hybrid.extend(pred_hybrid.cpu().numpy().tolist())

    all_y = np.array(all_y)
    all_pred_cnn = np.array(all_pred_cnn)
    all_pred_esn = np.array(all_pred_esn)
    all_pred_lstm = np.array(all_pred_lstm)
    all_pred_hybrid = np.array(all_pred_hybrid)

    # ----- Metrics -----
    labels_idx = list(range(len(classes_in_use)))

    cm_cnn = confusion_matrix(all_y, all_pred_cnn, labels=labels_idx)
    cm_esn = confusion_matrix(all_y, all_pred_esn, labels=labels_idx)
    cm_lstm = confusion_matrix(all_y, all_pred_lstm, labels=labels_idx)
    cm_hybrid = confusion_matrix(all_y, all_pred_hybrid, labels=labels_idx)

    acc_cnn = (all_pred_cnn == all_y).mean()
    acc_esn = (all_pred_esn == all_y).mean()
    acc_lstm = (all_pred_lstm == all_y).mean()
    acc_hybrid = (all_pred_hybrid == all_y).mean()

    print("\n=== Overall Accuracy ===")
    print(f"CNN   : {acc_cnn:.4f}")
    print(f"ESN   : {acc_esn:.4f}")
    print(f"LSTM  : {acc_lstm:.4f}")
    print(f"Hybrid: {acc_hybrid:.4f}")

    print("\n=== Classification Report: CNN ===")
    print(classification_report(all_y, all_pred_cnn, target_names=classes_in_use))

    print("\n=== Classification Report: ESN ===")
    print(classification_report(all_y, all_pred_esn, target_names=classes_in_use))

    print("\n=== Classification Report: LSTM ===")
    print(classification_report(all_y, all_pred_lstm, target_names=classes_in_use))

    print("\n=== Classification Report: Hybrid ===")
    print(classification_report(all_y, all_pred_hybrid, target_names=classes_in_use))

    # ----- Plot confusion matrices -----
    plot_confusion_matrix(cm_cnn, classes_in_use, "Confusion Matrix (CNN)", CM_CNN_PNG)
    plot_confusion_matrix(cm_esn, classes_in_use, "Confusion Matrix (ESN)", CM_ESN_PNG)
    plot_confusion_matrix(
        cm_lstm, classes_in_use, "Confusion Matrix (LSTM)", CM_LSTM_PNG
    )
    plot_confusion_matrix(
        cm_hybrid, classes_in_use, "Confusion Matrix (Hybrid CNN+ESN)", CM_HYBRID_PNG
    )

    # ----- Per-class accuracy comparison -----
    per_cnn = per_class_accuracy(cm_cnn)
    per_esn = per_class_accuracy(cm_esn)
    per_lstm = per_class_accuracy(cm_lstm)
    per_hybrid = per_class_accuracy(cm_hybrid)

    x = np.arange(len(classes_in_use))
    width = 0.2

    plt.figure(figsize=(10, 5), facecolor="white")
    plt.bar(x - 1.5 * width, per_cnn, width, label="CNN")
    plt.bar(x - 0.5 * width, per_esn, width, label="ESN")
    plt.bar(x + 0.5 * width, per_lstm, width, label="LSTM")
    plt.bar(x + 1.5 * width, per_hybrid, width, label="Hybrid")
    plt.xticks(x, classes_in_use, rotation=45, ha="right", fontname="Times New Roman")
    plt.ylim(0, 1.0)
    plt.ylabel("Per-class accuracy", fontname="Times New Roman")
    plt.title(
        "CNN vs ESN vs LSTM vs Hybrid CNN+ESN Per-class Accuracy",
        fontname="Times New Roman",
    )
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMPARE_PNG, dpi=150, facecolor="white")
    plt.close()
    print(f"📈 Saved per-class accuracy comparison -> {COMPARE_PNG}")

    # ----- Visualization: แบบรวม และแบบสไตล์ paper -----
    visualize_raw_vs_filtered_and_spectrogram(rows, classes_in_use)
    visualize_per_class_concat_wave_and_spectrogram(rows, classes_in_use)

    print(
        "\n✅ Done comparing CNN vs ESN vs LSTM vs Hybrid & generating visualizations."
    )


if __name__ == "__main__":
    main()
