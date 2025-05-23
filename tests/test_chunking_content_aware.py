import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.chunking import chunk_text

TEST_FILE = Path(__file__).parent.parent / "mercer.txt"
with open(TEST_FILE, encoding="utf-8") as f:
    TEST_TEXT = f.read()

def test_chunking_grid():
    for chunk_size in [200, 400]:
        for overlap in [0, 1]:
            print(f"\n=== chunk_size={chunk_size}, overlap={overlap} ===")
            chunks = chunk_text(TEST_TEXT, strategy="content-aware", chunk_size=chunk_size, overlap=overlap)
            print(f"Number of chunks: {len(chunks)}")
            for i, chunk in enumerate(chunks[:3]):
                print(f"  Chunk {i+1}: {chunk.text[:120]}...")
            print("-")

def main():
    test_chunking_grid()

if __name__ == "__main__":
    main()
