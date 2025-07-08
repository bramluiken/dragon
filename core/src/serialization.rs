use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read, Write};

use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Write tensors to a `.safetensors` file.
pub fn write_safetensors(
    tensors: &BTreeMap<String, Tensor>,
    path: &str,
    metadata: Option<Value>,
) -> io::Result<()> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, tensor) in tensors {
        let num_bytes = tensor.data.len() * std::mem::size_of::<f32>();
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": tensor.shape,
                "data_offsets": [offset, offset + num_bytes],
            }),
        );
        offset += num_bytes;
    }
    if let Some(meta) = metadata {
        header.insert("__metadata__".to_string(), meta);
    }
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let header_len = header_bytes.len() as u64;

    let mut file = File::create(path)?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(&header_bytes)?;
    for tensor in tensors.values() {
        // Safety: f32 is plain old data
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                tensor.data.as_ptr() as *const u8,
                tensor.data.len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(bytes)?;
    }
    Ok(())
}

/// Read tensors from a `.safetensors` file.
pub fn read_safetensors(path: &str) -> io::Result<(BTreeMap<String, Tensor>, Option<Value>)> {
    let mut file = File::open(path)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&header_bytes).unwrap();

    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let mut tensors = BTreeMap::new();
    let mut metadata = None;
    for (name, val) in header.into_iter() {
        if name == "__metadata__" {
            metadata = Some(val);
            continue;
        }
        let shape = val["shape"].as_array().unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        let start = val["data_offsets"][0].as_u64().unwrap() as usize;
        let end = val["data_offsets"][1].as_u64().unwrap() as usize;
        let num_elems = (end - start) / std::mem::size_of::<f32>();
        let mut tensor_data = vec![0f32; num_elems];
        let bytes = &data[start..end];
        // Safety: alignment is correct as data was written from f32 slices
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                tensor_data.as_mut_ptr() as *mut u8,
                bytes.len(),
            );
        }
        tensors.insert(name, Tensor { shape, data: tensor_data });
    }
    Ok((tensors, metadata))
}

