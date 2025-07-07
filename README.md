# dragon

Let’s design **dRAGon** from first principles as a *grammar-only Transformer decoder* (for a primitive RAG). We’ll keep it **modular**, **modern**, and aligned with latest best practices. Core in **Rust** (fast, safe for tensor ops), orchestrated in **PHP** (your preference).

---

## 🐉 **dRAGon: Design Overview**

### 🎯 Goal

A **minimal decoder-only LLM**:

* No world knowledge
* Only grammar/syntax
* Built to compose outputs from retrieved chunks in RAG
* Written in **Rust** (core) + PHP (API orchestration)

---

### 📐 Core Architecture

#### 🏗 Decoder-Only Transformer

* **Layers:** 4–8 (adjustable)
* **Hidden size:** 256–512
* **Heads:** 4–8
* **Embedding size:** 256
* **Vocab size:** \~16K (use SentencePiece/BPE)
* **Context length:** 512 tokens (you don’t need >2K)

#### Modules:

1. **Token Embedding**
2. **Positional Encoding**

   * Rotary Embeddings (RoPE) instead of sinusoids (faster for extrapolation)
3. **Multi-Head Self-Attention**

   * Scaled dot-product attention
   * Causal mask (to prevent peeking ahead)
4. **Feedforward Layer (MLP)**

   * GELU activation (modern best practice)
5. **LayerNorm**

   * Pre-norm (Transformer Pre-LN improves stability)
6. **Final Linear Layer**

   * Projects back to vocab size for logits

---

### 🦀 Rust Core

#### Why Rust?

* Low-level tensor ops (via `ndarray`, `tch-rs`, or `burn`)
* Memory safety
* SIMD friendly
* Zero-cost abstractions

---

#### Core Modules in Rust

| Module                   | Crate / Approach                          |
| ------------------------ | ----------------------------------------- |
| Tensor Math              | `ndarray`, `tch-rs`, or `burn`            |
| Tokenizer                | `tokenizers` (Hugging Face Rust bindings) |
| Matrix Multiplication    | BLAS bindings (`ndarray-linalg`)          |
| Attention                | Custom implementation w/ SIMD             |
| Training Loop (optional) | `burn` or raw autograd                    |
| File I/O (weights)       | Flatbuffers/serde                         |
| Model serialization      | HuggingFace `.safetensors` or TorchScript |

---

### 🐘 PHP Orchestration

#### Why PHP?

* For API endpoint serving
* Middleware to accept prompt + retrieved chunks → send to Rust core
* Lightweight async orchestration with `Swoole` or `ReactPHP`

#### Example PHP flow:

1. Accept prompt + retrieved context via HTTP POST
2. Pre-tokenize text (`ffi` to Rust tokenizer)
3. Call Rust core (via FFI, or subprocess with shared memory)
4. Stream logits back, decode tokens to PHP strings
5. Return JSON response

---

---

## 🧠 Training Strategy

#### 📖 Training Data

* Large text corpus (Wikipedia, books)
* **Entity randomization:**

  * Shuffle entity names
  * Replace facts with syntactic equivalents
* Objective: Predict next token (causal LM)

#### 🥋 Best Practices

* **AdamW optimizer**
* **Cosine LR scheduler** with warmup
* **Gradient checkpointing** (if big model)
* **Mixed precision (fp16)**

#### 🥽 Alternative: Pretrain with knowledge -> Forget knowledge

* Pretrain normally (or start with GPT weights)
* Fine-tune on shuffled dataset to destroy factual grounding
* Result: grammar intact, facts erased

---

## 📦 Project Layout

```
dRAGon/
├── core/             # Rust Transformer core
│   ├── src/
│   ├── Cargo.toml
├── php/              # PHP orchestration
│   ├── api/
│   ├── composer.json
├── tokenizer/        # SentencePiece/BPE models
├── weights/          # Saved model weights
└── README.md
```

---

## 🚀 Roadmap

✅ **Phase 1:** Rust core (inference-only)
✅ **Phase 2:** PHP API orchestration
✅ **Phase 3:** Training pipeline (optional)
✅ **Phase 4:** Integration with RAG system

---

### Development Progress

* Implemented a naive self-attention layer in Rust (`core/src/attention.rs`) as the first step toward the full decoder.
* Added a simple two-layer feedforward network (`core/src/feedforward.rs`).
* Created a minimal decoder block chaining attention and feedforward (`core/src/decoder.rs`).
* Introduced a simple multi-layer `Transformer` composed of decoder blocks (`core/src/transformer.rs`).
* Added a basic layer normalization module and integrated it into the decoder blocks (`core/src/layernorm.rs`).
