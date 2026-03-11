import branchorag
import os
import time
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
SCAN_PATH = r"C:\Users\grint\Documents\OneDrive\roop_row\MONEYMOINE\Web Training"
MEMORY_FILE = "brancho_memory.json"
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, ~80MB, fast and good for code
EXPECTED_DIM = 384
MAX_EMBED_CHARS = 4_000             # ~1000 tokens — long files are truncated before encoding

def run_brain():
    print("--- BranchoRAG v0.03: The Embedder ---")

    if not os.path.isdir(SCAN_PATH):
        raise FileNotFoundError(f"Scan path not found or is not a folder: {SCAN_PATH}")

    try:
        rag = branchorag.BranchoRAG()

        # --- STEP 1: LOAD EXISTING MEMORY (if any) ---
        if os.path.isfile(MEMORY_FILE):
            print(f"Loading existing memory from {MEMORY_FILE}...")
            rag.load_memory(MEMORY_FILE)
            print(f"  Loaded {rag.node_count()} existing node(s).")

        # --- STEP 2: SCAN ---
        print(f"Reading files in: {SCAN_PATH}...")
        start = time.perf_counter()
        rag.scan_folder(SCAN_PATH)
        elapsed = time.perf_counter() - start
        print(f"  {rag.node_count()} total file(s) after scan ({elapsed:.2f}s).")

        # --- STEP 3: EMBED (only files without an embedding) ---
        pending = rag.get_unembedded_contents()  # list of (index, content)

        if not pending:
            print("  All files already embedded — nothing to do.")
        else:
            print(f"Loading embedding model '{EMBED_MODEL}'...")
            model = SentenceTransformer(EMBED_MODEL)

            # Sanity-check: confirm the model outputs the dimension we expect
            probe = model.encode(["test"], convert_to_numpy=True)
            actual_dim = probe.shape[1]
            if actual_dim != EXPECTED_DIM:
                raise ValueError(
                    f"Model output dim is {actual_dim}, expected {EXPECTED_DIM}. "
                    f"Update EXPECTED_DIM or switch models."
                )

            indices, contents = zip(*pending)

            # Truncate very long files — sentence-transformers silently truncates at 256 tokens
            # by default anyway, but being explicit here prevents silent data loss on huge files.
            contents = [c[:MAX_EMBED_CHARS] for c in contents]

            print(f"Embedding {len(contents)} new file(s)...")
            start = time.perf_counter()
            embeddings = model.encode(contents, show_progress_bar=True, convert_to_numpy=True)
            elapsed = time.perf_counter() - start
            print(f"  Embedded in {elapsed:.2f}s.")

            rag.set_embeddings_partial(
                [(int(idx), emb.tolist()) for idx, emb in zip(indices, embeddings)]
            )

        # --- STEP 4: SAVE ---
        rag.save_memory(MEMORY_FILE)
        print(f"✅ Success: Knowledge + embeddings saved to {MEMORY_FILE}.")

    except Exception as e:
        print(f"❌ BranchoRAG failed: {e}")
        raise

if __name__ == "__main__":
    run_brain()
