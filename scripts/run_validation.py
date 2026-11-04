from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from experimental_validation import run_experimental_validation

if __name__ == '__main__':
    run_experimental_validation()

