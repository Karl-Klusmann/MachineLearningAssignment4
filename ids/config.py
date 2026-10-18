"""Project-wide constants: file locations, the data schema and the random seeds.
The file defines the path of the data file, the names of the nominal features and the ten class names.
It also defines the share of data held out as the generalisation set and one master seed.
The function rng turns the master seed into separate random streams, so every experiment gives the same result on every run.
Every other module imports from this file.
"""

from pathlib import Path
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "networkTraffic.csv"
RESULTS_DIR = ROOT / "results"

ID_COLUMN = "id"
TARGET = "attack_cat"
MISSING_TOKEN = "?"
NOMINAL_FEATURES = ("proto", "service", "state")
MISSING_CATEGORY = "none"
CLASS_NAMES = ("Normal", "Reconnaissance", "Backdoor", "DoS", "Exploits",
               "Analysis", "Fuzzers", "Worms", "Shellcode", "Generic")
N_CLASSES = len(CLASS_NAMES)

GENERALISATION_FRACTION = 0.2

MASTER_SEED = 20261019


def rng(stream: str, *index: int) -> np.random.Generator:
    """
    Create a reproducible random number generator for one named purpose.

    The master seed, the stream name and the integers together fix the generator.
    Different names or integers give independent streams.
    NumPy pads the key with zeros, so rng("folds") and rng("folds", 0) are the same stream.
    Within one stream name the project always passes the same number of integers.

    Args:
        stream: Name of the purpose, for example "holdout", "folds" or "forest".
        *index: Optional integers that tell repetitions apart, such as the seed number or the tree number.

    Returns:
        A NumPy random generator.
    """
    key = [MASTER_SEED, zlib.crc32(stream.encode("utf-8")), *index]
    return np.random.default_rng(np.random.SeedSequence(key))
