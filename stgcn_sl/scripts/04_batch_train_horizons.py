import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "02_train_stgcn_regression.py"

P = 30
EPOCHS = 30
BATCH = 16
LR = 1e-3

for H in range(1, 8):
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--P", str(P),
        "--H", str(H),
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH),
        "--lr", str(LR),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
