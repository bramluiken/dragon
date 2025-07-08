use std::os::raw::{c_char, c_ulong};
use std::ffi::CStr;
use crate::model::Model;

/// Opaque handle wrapping a `Model` for FFI usage.
#[repr(C)]
pub struct ModelHandle {
    model: Model,
}

#[no_mangle]
pub extern "C" fn dragon_model_create(
    vocab_size: c_ulong,
    embed_dim: c_ulong,
    hidden_dim: c_ulong,
    num_layers: c_ulong,
) -> *mut ModelHandle {
    let model = Model::new(
        vocab_size as usize,
        embed_dim as usize,
        hidden_dim as usize,
        num_layers as usize,
    );
    Box::into_raw(Box::new(ModelHandle { model }))
}

#[no_mangle]
pub extern "C" fn dragon_model_free(handle: *mut ModelHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

#[no_mangle]
pub extern "C" fn dragon_model_generate(
    handle: *mut ModelHandle,
    tokens_ptr: *const c_ulong,
    len: c_ulong,
    steps: c_ulong,
    out_ptr: *mut c_ulong,
) -> c_ulong {
    assert!(!handle.is_null());
    let model = unsafe { &(*handle).model };
    let raw = unsafe { std::slice::from_raw_parts(tokens_ptr, len as usize) };
    let tokens: Vec<usize> = raw.iter().map(|&v| v as usize).collect();
    let result = model.generate(&tokens, steps as usize);
    unsafe {
        let out_slice = std::slice::from_raw_parts_mut(out_ptr, result.len());
        for (i, &val) in result.iter().enumerate() {
            out_slice[i] = val as c_ulong;
        }
    }
    result.len() as c_ulong
}

#[no_mangle]
pub extern "C" fn dragon_model_save(handle: *mut ModelHandle, path: *const c_char) -> bool {
    if handle.is_null() || path.is_null() {
        return false;
    }
    let path = unsafe { CStr::from_ptr(path) };
    let path_str = match path.to_str() {
        Ok(p) => p,
        Err(_) => return false,
    };
    let model = unsafe { &(*handle).model };
    model.save_safetensors(path_str).is_ok()
}

#[no_mangle]
pub extern "C" fn dragon_model_load(path: *const c_char) -> *mut ModelHandle {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let path = unsafe { CStr::from_ptr(path) };
    let path_str = match path.to_str() {
        Ok(p) => p,
        Err(_) => return std::ptr::null_mut(),
    };
    match Model::load_safetensors(path_str) {
        Ok(model) => Box::into_raw(Box::new(ModelHandle { model })),
        Err(_) => std::ptr::null_mut(),
    }
}
#[no_mangle]
pub extern "C" fn dragon_model_generate_inplace(
    handle: *mut ModelHandle,
    tokens_ptr: *mut c_ulong,
    len: c_ulong,
    steps: c_ulong,
) -> c_ulong {
    assert!(!handle.is_null());
    if tokens_ptr.is_null() {
        return 0;
    }
    let model = unsafe { &(*handle).model };
    let total = (len + steps) as usize;
    let buf = unsafe { std::slice::from_raw_parts_mut(tokens_ptr as *mut usize, total) };
    let mut current = len as usize;
    for _ in 0..steps {
        let logits = model.forward(&buf[..current]);
        if let Some(last) = logits.last() {
            let next = last
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            buf[current] = next;
            current += 1;
        }
    }
    current as c_ulong
}

