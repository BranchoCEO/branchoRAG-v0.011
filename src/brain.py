print("HELLO BRANCHO - THE SYSTEM IS STARTING")
import branchograph
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

print("BranchoRAG v0.01 Initializing")

# 1. USE THE RUST ENGINE (Lightning Fast)
print("1. Rust building the code graph...")
graph = branchograph.CodeGraph()

# Simulating the Rust engine finding files
node_main = graph.add_node("main.py")
node_auth = graph.add_node("auth.py")
node_db = graph.add_node("db.py")

# Simulating the connections (main imports auth, auth imports db)
graph.add_edge(node_main, node_auth)
graph.add_edge(node_auth, node_db)

# 2. SAVE IT TO DISK (So it survives a reboot)
graph.save_to_disk("brancho_memory.json")
print("2. Memory saved securely to SSD (brancho_memory.json).")

# 3. WAKE UP THE PYTHON BRAIN (The GNN)
print("3. Booting up PyTorch GNN Brain...")

# Format the connections for PyTorch
edge_index = torch.tensor([[node_main, node_auth], 
                           [node_auth, node_db]], dtype=torch.long)

# Create "dummy math" (vectors) to represent the meaning of the 3 files
x = torch.randn(3, 16) 
data = Data(x=x, edge_index=edge_index)

# A simple Graph AI Layer
class BranchoBrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 8) # Compresses the knowledge

    def forward(self, data):
        return self.conv1(data.x, data.edge_index)

# 4. THINK
model = BranchoBrain()
output = model(data)
print("4. Brain has processed the code structure! Output matrix shape:", output.shape)
print("--- Sequence Complete ---")
