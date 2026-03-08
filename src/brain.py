import branchorag # Updated library name
import torch

print("--- BranchoRAG v0.01: Active ---")

def run_brain():
    # 1. Initialize the RAG system
    rag = branchorag.BranchoRAG()
    
    # 2. Tell the AI to look at its own project folder
    # "." means 'current folder'
    print("Scanning project folders...")
    rag.scan_folder(".") 
    
    # 3. Save the findings to your SSD
    rag.save_memory("brancho_memory.json")
    print("✅ Success: BranchoRAG has mapped your project.")

if __name__ == "__main__":
    run_brain()
