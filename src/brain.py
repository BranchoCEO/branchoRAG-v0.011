import branchorag
import os

# Using a 'raw' string (r"") to handle Windows backslashes correctly
SCAN_PATH = r"C:\Users\grint\Documents\OneDrive\roop_row\MONEYMOINE\Web Training"
MEMORY_FILE = "brancho_memory.json"

def run_brain():
    print("--- BranchoRAG v0.01: Active ---")

    # Quick check to see if the path exists before we hand it to Rust
    if not os.path.exists(SCAN_PATH):
        print(f"❌ Error: The path '{SCAN_PATH}' does not exist.")
        return

    try:
        # 1. Initialize the RAG system
        rag = branchorag.BranchoRAG()

        # 2. Tell the AI to look at the specific Web Training folder
        print(f"Scanning target folder: {SCAN_PATH}...")
        rag.scan_folder(SCAN_PATH)
        
        print(f"  Found {rag.node_count()} relevant file(s).")

        # 3. Save the findings to your SSD
        rag.save_memory(MEMORY_FILE)
        print(f"✅ Success: Project map saved to {MEMORY_FILE}.")

    except Exception as e:
        print(f"❌ BranchoRAG failed: {e}")
        raise

if __name__ == "__main__":
    run_brain()
