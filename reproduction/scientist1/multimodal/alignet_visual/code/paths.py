"""Central path definitions."""
import os

DATA_DIR = "/data/zhenqian/data"
MODEL_DIR = "/data/zhenqian/models"

THINGS_ROOT = os.path.join(DATA_DIR, "things_ooo_root")
THINGS_IMGS = os.path.join(THINGS_ROOT, "imgur_images")
THINGS_META = os.path.join(THINGS_ROOT, "concepts-metadata_things.tsv")
THINGS_MANIFEST = os.path.join(THINGS_ROOT, "imgur_manifest.tsv")
THINGS_OSF = os.path.join(THINGS_ROOT, "osfstorage")
THINGS_SENSEVEC = os.path.join(
    THINGS_OSF,
    "02_object-level/semantic-embedding_things/semantic-embedding_sensevec-augmented-with-wordvec_things.csv",
)
THINGS_WORDVEC = os.path.join(
    THINGS_OSF,
    "02_object-level/semantic-embedding_things/semantic-embedding_wordvec_things.csv",
)
THINGS_CAT27_BU = os.path.join(THINGS_OSF, "03_category-level/category27_bottom-up.tsv")
THINGS_CAT27_TD = os.path.join(THINGS_OSF, "03_category-level/category27_top-down.tsv")
THINGS_CAT27_MAN = os.path.join(THINGS_OSF, "03_category-level/category27_manual.tsv")
THINGS_CAT53_LONG = os.path.join(THINGS_OSF, "03_category-level/category53_long-format.tsv")
THINGS_CAT53_WIDE = os.path.join(THINGS_OSF, "03_category-level/category53_wide-format.tsv")

IMAGENET_VAL = os.path.join(DATA_DIR, "imagenet-val")
IMAGENET_VAL_DATA = os.path.join(IMAGENET_VAL, "data")
IMAGENET_TRAIN_IDS = os.path.join(IMAGENET_VAL, "train_subset_40k.txt")
IMAGENET_EVAL_IDS = os.path.join(IMAGENET_VAL, "eval_subset_10k.txt")
IMAGENET_INDEX = os.path.join(IMAGENET_VAL, "global_index.tsv")

BREEDS_ROOT = os.path.join(DATA_DIR, "breeds")

# Models
SIGLIP_PATH = os.path.join(MODEL_DIR, "siglip-so400m-patch14-384")
DINOV2_BASE = os.path.join(MODEL_DIR, "dinov2-base")
DINOV2_SMALL = os.path.join(MODEL_DIR, "dinov2-small")
DINOV2_LARGE = os.path.join(MODEL_DIR, "dinov2-large")
VITB_PATH = os.path.join(MODEL_DIR, "vit-base-patch16-224")
CLIP_L14 = os.path.join(MODEL_DIR, "clip-vit-l14")

# Working directories (relative to cwd)
WORKDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(WORKDIR, "cache")
LOG_DIR = os.path.join(WORKDIR, "logs")
RESULTS_DIR = os.path.join(WORKDIR, "results")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
