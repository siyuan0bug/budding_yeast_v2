"""Stable dataset command; implementation is in data.systems.generation."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budding_yeast_v2.data.systems.generation import main, generate_dataset


if __name__ == '__main__':
    main()
