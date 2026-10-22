from pathlib import Path

ROOT_DIR    = Path(__file__).resolve().parent.parent   
DATA_DIR    = ROOT_DIR / "Dataset"
RESULTS_DIR = ROOT_DIR / "results"
FIG_DIR     = RESULTS_DIR / "figures"
PROCESSED_DIR = ROOT_DIR / "data_processed"

RANDOM_SEED = 42
MIN_FREQ = 2
MAX_LEN = 512
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

CLASSES = ['Culture', 'Finance', 'Medical', 'Politics', 'Religion', 'Sports', 'Tech']