import json
from pathlib import Path
from threading import Lock

_STATUS_FILE = Path("ingest_status.json")
_status_lock = Lock()

def _read_status():
    if not _STATUS_FILE.exists():
        return {}
    with _STATUS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_status(status):
    with _STATUS_FILE.open("w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

def set_backend_status(doc_name: str, backend: str, state: str):
    """Set status for a backend ('vector' or 'graph') for a document."""
    with _status_lock:
        status = _read_status()
        if doc_name not in status:
            status[doc_name] = {}
        status[doc_name][backend] = state
        _write_status(status)

def get_backend_status(doc_name: str):
    """Get status dict for a document, e.g. {'vector': 'ready', 'graph': 'processing'}"""
    with _status_lock:
        status = _read_status()
        return status.get(doc_name, {})

def get_ready_backends(doc_name: str):
    """Return a list of ready backends for the document."""
    s = get_backend_status(doc_name)
    return [k for k, v in s.items() if v == 'ready']
