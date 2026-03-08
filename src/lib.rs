use pyo3::prelude::*;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;
use walkdir::WalkDir;

// --- BLOCK 1: DATA ---
#[derive(Serialize, Deserialize)]
struct GraphData {
    nodes: Vec<String>,
    edges: Vec<(usize, usize)>, // Reserved for future relationship mapping
}

#[pyclass]
struct BranchoRAG {
    data: GraphData,
}

// --- BLOCK 2: METHODS ---
// This block must be ALONE. No curly braces should be wrapping it.
#[pymethods]
impl BranchoRAG {
    #[new]
    fn new() -> Self {
        BranchoRAG {
            data: GraphData { nodes: Vec::new(), edges: Vec::new() },
        }
    }

    fn scan_folder(&mut self, path: String) -> PyResult<()> {
        for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let name = entry.path().display().to_string();
                if !self.data.nodes.contains(&name) {
                    self.data.nodes.push(name);
                }
            }
        }
        Ok(())
    }

    fn save_memory(&self, filename: String) -> PyResult<()> {
        // Propagate serialization errors to Python instead of panicking
        let json = serde_json::to_string_pretty(&self.data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let mut file = File::create(filename)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Returns the number of file nodes currently held in memory.
    fn node_count(&self) -> usize {
        self.data.nodes.len()
    }
}

// --- BLOCK 3: THE MODULE ---
#[pymodule]
fn branchorag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BranchoRAG>()?;
    Ok(())
}
//python -m maturin develop
