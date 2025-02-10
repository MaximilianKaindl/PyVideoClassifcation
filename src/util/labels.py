from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

class LabelCategory(Enum):
    RECORDING = "recording"
    CONTENT = "content"
    GENRE = "genre"

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

def get_default_labels() -> Labels:
    """Get default hardcoded labels if no file is provided"""
    return Labels(categories={
        LabelCategory.RECORDING.value: [
            "a professional recording with high production value",
            "an amateur recording with low production value"
        ],
        LabelCategory.CONTENT.value: [
            "a video showing a lecture or educational content",
            "a video showing entertainment or performance",
            "a video showing news or journalism",
            "a video showing promotional or advertising content"
        ],
        LabelCategory.GENRE.value: [
            "a comedy video",
            "a drama video",
            "an action video",
            "a documentary video",
            "a horror video",
            "a romance video",
            "a sci-fi video",
            "a thriller video",
            "a mystery video",
            "a musical video",
            "an animation video"
        ]
    })