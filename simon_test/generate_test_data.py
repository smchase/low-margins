import sys
from pathlib import Path
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import ROWS, COLS, COLORS


def generate_test_data_json(filename: str = "test_data.json") -> None:
    data = np.random.randint(0, len(COLORS), (ROWS, COLS), dtype=np.int64)
    
    filepath = Path(__file__).parent / filename
    with open(filepath, 'w') as f:
        json.dump({
            'rows': ROWS,
            'cols': COLS,
            'num_colors': len(COLORS),
            'data': data.tolist()
        }, f, indent=2)
    

if __name__ == "__main__":
    generate_test_data_json()

