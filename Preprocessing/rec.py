# ==================================================
# rec.py
# Chain Audio Data Recorder (RAW waveform only)
# ==================================================

import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000  # Hz (fix ค่าเดียวทั้งโปรเจกต์)
DURATION_SEC = 1.5# วินาทีต่อคลิป
BASE_DIR = Path("dataset")

# Final class names (DO NOT CHANGE)
CLASSES = ["Normal", "High-Low", "HL+RS", "WrongSide"]


# =========================
# LABEL MAP
# =========================
def class_to_labels(class_name: str):
    """
    Map class name -> (y_okng, y_defect)
    """
    if class_name == "Normal":
        return 0, 0
    elif class_name == "High-Low":
        return 1, 1
    elif class_name == "HL+RS":
        return 1, 2
    elif class_name == "WrongSide":
        return 1, 3
    else:
        raise ValueError(f"Unknown class: {class_name}")


# =========================
# INIT PATHS
# =========================
def init_paths(session_name: str):
    audio_dir = BASE_DIR / "audio" / session_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    labels_path = BASE_DIR / "labels.csv"
    if not labels_path.exists():
        with labels_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "file_path",
                    "class_name",
                    "y_okng",
                    "y_defect",
                    "session",
                    "timestamp",
                    "remark",
                ]
            )
    return audio_dir, labels_path


# =========================
# RECORD
# =========================
def record_clip(duration, sample_rate):
    print(f"\n🎙 Recording {duration:.2f}s ...")
    audio = sd.rec(
        int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32"
    )
    sd.wait()
    return audio.squeeze(-1)


def normalize(audio):
    """
    Normalize ONLY to prevent clipping
    (ไม่เปลี่ยนรูป waveform)
    """
    peak = np.max(np.abs(audio)) + 1e-8
    return (audio / peak * 0.95).astype(np.float32)


def save_clip(audio, sample_rate, audio_dir: Path, class_name: str):
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{class_name}_{ts}.wav"
    path = audio_dir / filename

    wav_write(path, sample_rate, audio)
    return path, now.isoformat(timespec="seconds")


def append_label(
    labels_path: Path, file_path: Path, class_name: str, session: str, timestamp: str
):
    y_okng, y_defect = class_to_labels(class_name)
    with labels_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [file_path.as_posix(), class_name, y_okng, y_defect, session, timestamp, ""]
        )


# =========================
# MAIN LOOP
# =========================
def main():
    print("\n=== Chain Audio RAW Recorder ===\n")
    print("Classes:")
    for i, c in enumerate(CLASSES):
        print(f" [{i}] {c}")

    session = input("\nSession name " "(e.g. 2025-12-03_lineA_frame): ").strip()

    if not session:
        session = datetime.now().strftime("session_%Y%m%d")

    audio_dir, labels_path = init_paths(session)

    print(f"\n📂 Audio dir : {audio_dir.resolve()}")
    print(f"🧾 Labels   : {labels_path.resolve()}")

    while True:
        cmd = input(
            "\nSelect class " "(0=Normal, 1=High-Low, 2=HL+RS, 3=WrongSide, q=Exit): "
        ).strip()

        if cmd.lower() == "q":
            print("\n👋 Exit recorder")
            break

        if cmd not in ["0", "1", "2", "3"]:
            print("⚠ Invalid choice")
            continue

        class_name = CLASSES[int(cmd)]
        n = input("How many clips to record?: ").strip()

        if not n.isdigit():
            print("⚠ Must be a number")
            continue

        n = int(n)

        for i in range(n):
            input(f"\nPress ENTER to record {i+1}/{n} ...")
            audio = record_clip(DURATION_SEC, SAMPLE_RATE)
            audio = normalize(audio)

            path, ts = save_clip(audio, SAMPLE_RATE, audio_dir, class_name)
            append_label(labels_path, path, class_name, session, ts)

            print(f"💾 Saved: {path.name}")

    print("\n✅ Data collection finished")


if __name__ == "__main__":
    main()
