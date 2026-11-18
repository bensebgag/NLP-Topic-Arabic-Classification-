from pathlib import Path

ROOT_DIR    = Path(__file__).resolve().parent.parent   
DATA_DIR    = ROOT_DIR / "Dataset"
RESULTS_DIR = ROOT_DIR / "results"
FIG_DIR     = RESULTS_DIR / "figures"
PROCESSED_DIR= ROOT_DIR / "data_processed"
RANDOM_SEED = 42
MAX_LEN=512
MIN_FREQ = 2
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
CLASSES = ['Culture', 'Finance', 'Medical', 'Politics', 'Religion', 'Sports', 'Tech']
LABEL2ID = {name: i for i, name in enumerate(CLASSES)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}
HOLDOUT_FRAC = 0.30
EMB_DIM = 300
FASTTEXT_PATH = PROCESSED_DIR / "cc.ar.300.vec.gz"