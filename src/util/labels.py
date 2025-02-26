from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

@dataclass
class Labels:
    categories: Dict[str, List[str]]

def parse_labels_file(labels_file: str) -> Labels:
    """Parse labels from a structured text file"""
    categories = {}
    current_category = None
    
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_category = line[1:-1]
                categories[current_category] = []
            elif line and current_category:
                categories[current_category].append(line)
    
    return Labels(categories=categories)