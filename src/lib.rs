use pyo3::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Write, BufReader};
use walkdir::WalkDir;

// --- BLOCK 1: DATA ---
#[derive(Serialize, Deserialize)]
struct FileNode {
    path: String,
    content: String,
    embedding: Vec<f32>,  // 384-dim vector from sentence-transformers; empty until embed() is called
}

#[derive(Serialize, Deserialize)]
struct GraphData {
    nodes: Vec<FileNode>,
    edges: Vec<(usize, usize)>,
}

#[pyclass]
struct BranchoRAG {
    data: GraphData,
}

const IGNORE_LIST: &[&str] = &["target", ".git", ".venv", "__pycache__", "env", "node_modules"];
const MAX_FILE_BYTES: u64 = 1_000_000; // 1MB — skip minified JS, logs, binaries, etc.

// --- BLOCK 2: METHODS ---
#[pymethods]
impl BranchoRAG {
    #[new]
    fn new() -> Self {
        BranchoRAG {
            data: GraphData { nodes: Vec::new(), edges: Vec::new() },
        }
    }

    fn scan_folder(&mut self, path: String) -> PyResult<()> {

        // Track paths we've already seen to avoid duplicates
        let mut seen: HashSet<String> = self.data.nodes.iter().map(|n| n.path.clone()).collect();

        // filter_entry prunes ignored dirs entirely — WalkDir won't descend into them at all,
        // which is much faster than checking every file inside .git, node_modules, etc.
        for entry in WalkDir::new(path).into_iter().filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            !IGNORE_LIST.contains(&name)
        }).filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }

            let path_str = entry.path().display().to_string();
            if seen.contains(&path_str) {
                continue;
            }

            // Check size before reading to avoid loading huge files into memory
            let size = entry.metadata().map(|m| m.len()).unwrap_or(u64::MAX);
            if size > MAX_FILE_BYTES {
                continue;
            }

            // read_to_string naturally skips binary files that aren't valid UTF-8
            if let Ok(content) = fs::read_to_string(entry.path()) {
                seen.insert(path_str.clone());
                self.data.nodes.push(FileNode { path: path_str, content, embedding: Vec::new() });
            }
        }
        Ok(())
    }

    /// Returns (index, content) for every node that has no embedding yet.
    /// Python uses the indices to write back only the new embeddings via set_embeddings_partial().
    fn get_unembedded_contents(&self) -> Vec<(usize, String)> {
        self.data.nodes.iter().enumerate()
            .filter(|(_, n)| n.embedding.is_empty())
            .map(|(i, n)| (i, n.content.clone()))
            .collect()
    }

    /// Accepts (index, embedding) pairs and writes them back into the correct nodes.
    /// Only touches nodes that were returned by get_unembedded_contents().
    fn set_embeddings_partial(&mut self, embeddings: Vec<(usize, Vec<f32>)>) -> PyResult<()> {
        let node_count = self.data.nodes.len();
        for (idx, emb) in embeddings {
            if idx >= node_count {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Index {} out of range (node count: {})", idx, node_count
                )));
            }
            self.data.nodes[idx].embedding = emb;
        }
        Ok(())
    }

    /// Loads a previously saved memory JSON back into this instance.
    /// Safe to call before scan_folder — nodes loaded here participate in the seen-set dedup.
    fn load_memory(&mut self, filename: String) -> PyResult<()> {
        let file = File::open(&filename)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                format!("Could not open '{}': {}", filename, e)
            ))?;
        let reader = BufReader::new(file);
        let loaded: GraphData = serde_json::from_reader(reader)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to parse '{}': {}", filename, e)
            ))?;
        self.data = loaded;
        Ok(())
    }

    fn save_memory(&self, filename: String) -> PyResult<()> {
        let json = serde_json::to_string_pretty(&self.data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let mut file = File::create(filename)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

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
