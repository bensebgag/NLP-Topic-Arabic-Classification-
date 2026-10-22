from pathlib import Path

ROOT_DIR    = Path(__file__).resolve().parent.parent   
DATA_DIR    = ROOT_DIR / "Dataset"
RESULTS_DIR = ROOT_DIR / "results"
FIG_DIR     = RESULTS_DIR / "figures"

RANDOM_SEED = 42

CLASSES = ['Culture', 'Finance', 'Medical', 'Politics', 'Religion', 'Sports', 'Tech']