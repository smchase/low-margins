import sys
from pathlib import Path
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import ROWS, COLS, COLORS


def generate_test_data_json(filename: str = "test_data.json") -> None:
    test_cases = []
    for i in range(100):
        data = np.random.randint(0, len(COLORS), (ROWS, COLS), dtype=np.int64)
        test_cases.append(data.tolist())
    
    filepath = Path(__file__).parent / filename
    with open(filepath, 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    print(f"Generated 20 test cases and saved to: {filepath}")


if __name__ == "__main__":
    generate_test_data_json()

