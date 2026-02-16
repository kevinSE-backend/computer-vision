from pathlib import Path

for split in ["train", "val"]:
    for cls in ["dogs", "cats"]:
        Path(f"dataset/{split}/{cls}").mkdir(parents=True, exist_ok=True)
