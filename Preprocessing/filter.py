# ==================================================
# hpf_sweep_visualize.py
#
# สคริปต์สำหรับลองปรับ High-pass filter cutoff หลาย ๆ ค่า (Hz)
# แล้ววาดกราฟ:
#   - Waveform ต่อ class (หลัง filter)
#   - Spectrogram ต่อ class (หลัง filter)
#
# ใช้ labels.csv ชุดเดียวกับ compare_models_cnn_esn_lstm_hybrid_viz.py
# (ไม่ต้องโหลดโมเดลใด ๆ)
# ==================================================

import csv
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import butter, filtfilt

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

# ลำดับ canonical ของ class (เวลา sort ให้คงที่)
ALL_CLASSES = ["Normal", "High-Low", "HL+RS", "WrongSide"]

# จำนวนคลิปสูงสุดต่อ class ที่จะเอามาต่อกัน
MAX_SEGMENTS_PER_CLASS = 3

# ลิสต์ cutoff ที่อยากลอง (Hz)
# 0 = ไม่ฟิลเตอร์ (ผ่าน raw ตรง ๆ)
HPF_CUTOFF_LIST = [1000, 2000, 3000, 4000]

# default order ของ filter
HPF_ORDER = 4


# ============= FILTER =================
def highpass_filter(
    audio: np.ndarray,
    sr: int,
    cutoff: float,
    order: int = HPF_ORDER,
) -> np.ndarray:
    """
    High-pass filter ถ้า cutoff == 0 จะไม่ฟิลเตอร์ (คืนค่าเดิม)
    """
    if cutoff is None or cutoff <= 0:
        # ไม่ฟิลเตอร์
        return audio.astype(np.float32)

    nyq = 0.5 * sr
    norm_cut = cutoff / nyq
    b, a = butter(order, norm_cut, btype="highpass", analog=False)
    filtered = filtfilt(b, a, audio).astype(np.float32)
    return filtered


# ============= UTILS =================
def load_rows_and_classes(labels_csv: Path):
    rows = []
    class_set = set()

    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found at: {labels_csv}")

    with labels_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cname = row["class_name"]
            if cname in ALL_CLASSES:
                rows.append(row)
                class_set.add(cname)

    classes_in_use = sorted(list(class_set), key=lambda c: ALL_CLASSES.index(c))
    return rows, classes_in_use


def fix_length(audio: np.ndarray, target_len: int = NUM_SAMPLES) -> np.ndarray:
    """
    ทำให้ความยาวของสัญญาณเท่ากับ NUM_SAMPLES:
      - ถ้าสั้น → padding 0 ด้านท้าย
      - ถ้ายาว → ตัดกลาง
    """
    if len(audio) < target_len:
        pad_width = target_len - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="constant")
    elif len(audio) > target_len:
        start = (len(audio) - target_len) // 2
        audio = audio[start : start + target_len]
    return audio


def build_per_class_waveforms(rows, classes_in_use, cutoff_hz: float):
    """
    อ่านไฟล์เสียง แยกตาม class
    แล้วฟิลเตอร์ด้วย high-pass cutoff_hz
    และต่อคลิป (สูงสุด MAX_SEGMENTS_PER_CLASS) เป็น waveform ยาวหนึ่งเส้นต่อ class

    คืนค่า dict:
      class_name -> np.ndarray (หลัง filter)
    """
    class_to_files = {c: [] for c in classes_in_use}
    for row in rows:
        cname = row["class_name"]
        if cname in class_to_files:
            class_to_files[cname].append(row["file_path"])

    class_wave_filtered = {}

    for cname in classes_in_use:
        files = class_to_files.get(cname, [])
        if len(files) == 0:
            print(f"  ⚠ no files for class {cname}, skip.")
            continue

        segments = []
        for path_str in files[:MAX_SEGMENTS_PER_CLASS]:
            path = Path(path_str)
            if not path.exists():
                print(f"  ⚠ file not found for class {cname}: {path}")
                continue

            audio_raw, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            audio_raw = audio_raw.astype(np.float32)

            audio_raw_fixed = fix_length(audio_raw, NUM_SAMPLES)
            audio_filt = highpass_filter(
                audio_raw_fixed, sr=SAMPLE_RATE, cutoff=cutoff_hz
            )

            segments.append(audio_filt)

        if len(segments) == 0:
            print(f"  ⚠ no valid segments for class {cname}")
            continue

        class_wave_filtered[cname] = np.concatenate(segments, axis=0)

    return class_wave_filtered


def plot_waveforms_per_class(class_wave_filt: dict, classes_in_use, cutoff_hz: float):
    """
    วาด waveform ต่อ class หลัง filter ด้วย cutoff_hz
    เซฟเป็นไฟล์ PNG
    """
    if len(class_wave_filt) == 0:
        print("⚠ No filtered waveforms to plot.")
        return

    n_classes = len(class_wave_filt)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, axes = plt.subplots(
        n_classes,
        1,
        figsize=(11, 2.4 * n_classes),
        sharex=True,
        facecolor="white",
    )
    if n_classes == 1:
        axes = [axes]

    if cutoff_hz <= 0:
        title_str = "Waveforms Without High-pass Filtering"
        suffix = "no_hp"
    else:
        title_str = f"Waveforms After High-pass Filtering ({cutoff_hz:.0f} Hz)"
        suffix = f"hp{int(cutoff_hz)}Hz"

    fig.suptitle(title_str, fontname="Times New Roman", fontsize=14)

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

    out_name = f"wave_filtered_per_class_{suffix}.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_name, dpi=150, facecolor="white")
    plt.close()
    print(f"📈 Saved per-class filtered waveform plot -> {out_name}")


def plot_spectrograms_per_class(
    class_wave_filt: dict, classes_in_use, cutoff_hz: float
):
    """
    วาด spectrogram ต่อ class จาก waveform หลัง filter ด้วย cutoff_hz
    ใช้ scale สีร่วมกันทุก class
    เซฟเป็นไฟล์ PNG
    """
    if len(class_wave_filt) == 0:
        print("⚠ No filtered waveforms for spectrogram.")
        return

    # เตรียม S_db รวมทุก class เพื่อหา vmin/vmax
    all_S_db = []
    for cname, sig in class_wave_filt.items():
        D = librosa.stft(sig, n_fft=512, hop_length=128, win_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        all_S_db.append(S_db)

    stacked = np.concatenate(all_S_db, axis=1)
    vmin = stacked.min()
    vmax = stacked.max()

    n_classes = len(class_wave_filt)
    fig, axes = plt.subplots(
        n_classes,
        1,
        figsize=(11, 2.6 * n_classes),
        sharex=True,
        facecolor="white",
    )
    if n_classes == 1:
        axes = [axes]

    if cutoff_hz <= 0:
        title_str = "Spectrograms Without High-pass Filtering"
        suffix = "no_hp"
    else:
        title_str = f"Spectrograms After High-pass Filtering ({cutoff_hz:.0f} Hz)"
        suffix = f"hp{int(cutoff_hz)}Hz"

    fig.suptitle(title_str, fontname="Times New Roman", fontsize=14)
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

    out_name = f"spectrogram_filtered_per_class_{suffix}.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_name, dpi=150, facecolor="white")
    plt.close()
    print(f"🎛 Saved per-class spectrogram plot -> {out_name}")


# ============= MAIN =================
def main():
    print("\n=== High-pass Filter Sweep Visualization ===")

    # โหลด labels
    rows, classes_in_use = load_rows_and_classes(LABELS_CSV)
    print(f"Total clips (valid classes): {len(rows)}")
    print(f"Classes in use            : {classes_in_use}")

    if len(rows) == 0 or len(classes_in_use) == 0:
        print("⚠ No valid data or classes found. Check labels.csv.")
        return

    # loop ไปตาม cutoff แต่ละตัว
    for cutoff in HPF_CUTOFF_LIST:
        if cutoff <= 0:
            print("\n--- Cutoff: NO HPF (raw) ---")
        else:
            print(f"\n--- Cutoff: {cutoff} Hz ---")

        # โหลดและฟิลเตอร์สัญญาณต่อ class
        class_wave_filt = build_per_class_waveforms(rows, classes_in_use, cutoff)

        # วาด waveform ต่อ class
        plot_waveforms_per_class(class_wave_filt, classes_in_use, cutoff)

        # วาด spectrogram ต่อ class
        plot_spectrograms_per_class(class_wave_filt, classes_in_use, cutoff)

    print("\n✅ Done HPF sweep visualization.")


if __name__ == "__main__":
    main()
