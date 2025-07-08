use std::os::raw::{c_char, c_ulong};
use std::ffi::CStr;
use crate::tokenizer::BpeTokenizer;
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

/// Opaque handle wrapping a `BpeTokenizer` for FFI usage.
#[repr(C)]
pub struct TokenizerHandle {
    tok: BpeTokenizer,
}

#[no_mangle]
pub extern "C" fn dragon_tokenizer_create(
    vocab_path: *const c_char,
    merges_path: *const c_char,
    unk_id: c_ulong,
) -> *mut TokenizerHandle {
    if vocab_path.is_null() || merges_path.is_null() {
        return std::ptr::null_mut();
    }
    let vocab_cstr = unsafe { CStr::from_ptr(vocab_path) };
    let merges_cstr = unsafe { CStr::from_ptr(merges_path) };
    let vocab_str = match vocab_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let merges_str = match merges_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let vocab_contents = match std::fs::read_to_string(vocab_str) {
        Ok(c) => c,
        Err(_) => return std::ptr::null_mut(),
    };
    let vocab: Vec<String> = vocab_contents.lines().map(|l| l.to_string()).collect();
    let merges_contents = match std::fs::read_to_string(merges_str) {
        Ok(c) => c,
        Err(_) => return std::ptr::null_mut(),
    };
    let merges: Vec<(String, String)> = merges_contents
        .lines()
        .filter_map(|l| {
            let mut parts = l.split_whitespace();
            let a = parts.next()?.to_string();
            let b = parts.next()?.to_string();
            Some((a, b))
        })
        .collect();
    let tok = BpeTokenizer::new(vocab, merges, unk_id as usize);
    Box::into_raw(Box::new(TokenizerHandle { tok }))
}

#[no_mangle]
pub extern "C" fn dragon_tokenizer_free(handle: *mut TokenizerHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

#[no_mangle]
pub extern "C" fn dragon_tokenizer_encode(
    handle: *const TokenizerHandle,
    text: *const c_char,
    out_ptr: *mut c_ulong,
    out_cap: c_ulong,
) -> c_ulong {
    if handle.is_null() || text.is_null() {
        return 0;
    }
    let tok = unsafe { &(*handle).tok };
    let text_cstr = unsafe { CStr::from_ptr(text) };
    let text_str = match text_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let tokens = tok.encode(text_str);
    let count = std::cmp::min(tokens.len(), out_cap as usize);
    unsafe {
        let out_slice = std::slice::from_raw_parts_mut(out_ptr, count);
        for (i, &t) in tokens.iter().take(count).enumerate() {
            out_slice[i] = t as c_ulong;
        }
    }
    count as c_ulong
}
