# ChromaDB Embedding Dimension Errors: Robust Solution

## Problem
You may encounter errors like:

```
chromadb.errors.InvalidArgumentError: Collection expecting embedding with dimension of 1024, got 768
```

This happens when the ChromaDB collection was created with a different embedding dimension than what your current embedding model produces.

## Solution (Implemented in Code)
- The code now checks the embedding dimension of the existing ChromaDB collection on startup.
- If a mismatch is detected, the collection is deleted and recreated with the correct dimension.
- This ensures the collection always matches your configured embedding model.

**Location of logic:**
- `app/db/connections.py`, in the `DatabaseManager.get_chroma_collection()` method.
- The method calls `_validate_collection_dimensions()` and, on mismatch, safely recreates the collection.

## How to Change Embedding Model/Dimension
1. Update your embedding model and/or `EMBEDDING_DIMENSION` in `.env` or config.
2. On next startup, the app will automatically recreate the collection if needed.
3. No more manual intervention required!

## Troubleshooting
If you still get dimension errors:
- Double-check your `.env` and config for `EMBEDDING_MODEL` and `EMBEDDING_DIMENSION`.
- Ensure you are not running multiple apps pointing to the same ChromaDB directory with different embedding models/dimensions.

---

**This behavior is robust and automatic as of May 2025.**

---

_Last updated: 2025-05-23_
