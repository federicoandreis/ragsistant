import json
from pathlib import Path
from app.models import IngestedDocument

# Save IngestedDocument to JSON

def save_ingested_document(doc: IngestedDocument, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc.dict(), f, ensure_ascii=False, indent=2)

# Load IngestedDocument from JSON

def load_ingested_document(path: str) -> IngestedDocument:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return IngestedDocument.model_validate(data)
