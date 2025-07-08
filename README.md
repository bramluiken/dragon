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
* Implemented a simple token embedding lookup layer (`core/src/embedding.rs`).
* Wrapped the embedding and transformer with a final linear layer in a new
  `Model` struct (`core/src/model.rs`) to enable end-to-end inference.
* Added a naive rotary positional embedding module (`core/src/rotary.rs`) and
  integrated it into the `Model`.
* Introduced a basic command-line inference tool (`core/src/bin/infer.rs`) demonstrating model usage.
* Added a simple PHP endpoint invoking the Rust inference binary (`php/api/index.php`).
* Implemented a naive whitespace tokenizer module (`core/src/tokenizer.rs`).
* Added a CLI to run inference directly on text input (`core/src/bin/infer_text.rs`).
* Implemented a simple autoregressive text generation CLI (`core/src/bin/generate_text.rs`).
* Added a cross-entropy loss module for evaluation (`core/src/loss.rs`).
* Added a CLI to compute cross-entropy loss for a text prompt (`core/src/bin/eval_loss.rs`).
* Added a CLI to compute perplexity for a text prompt (`core/src/bin/eval_perplexity.rs`).

## \ud83d\udcdd Development To-Do List

### Core (Rust)
- [ ] Replace naive attention with optimized multi-head attention
- [ ] Integrate BLAS-backed matrix multiplication for speed
- [x] Expose FFI-friendly API for PHP integration
- [ ] Support model serialization to `.safetensors`
- [ ] Implement optional quantization for lightweight inference
- [ ] Add training loop using `burn` or custom autograd

### Tokenizer
- [ ] Switch from whitespace tokenizer to SentencePiece/BPE
- [ ] Provide scripts to train and update vocabularies
- [ ] Add FFI bindings so PHP can tokenize directly
- [ ] Include tests for encode/decode round trips
- [ ] Support dynamic vocabulary merges during training
- [ ] Document tokenizer usage in `/tokenizer/README.md`

### PHP API & Integration
- [ ] Implement async HTTP server with Swoole/ReactPHP
- [ ] Stream tokens back to clients during generation
- [ ] Call the Rust core through FFI for zero-copy data flow
- [ ] Add authentication and rate limiting middleware
- [ ] Improve logging and structured error handling
- [x] Provide example client scripts (PHP and JavaScript)

### Training Pipeline
- [ ] Build a dataset loader for large text corpora
- [ ] Implement shuffling and batching dataloader
- [ ] Add mixed-precision and gradient accumulation support
- [ ] Save training checkpoints under `weights/`
- [ ] Provide evaluation metrics and scripts
- [ ] Document full training workflow

### RAG Integration
- [ ] Connect retrieval system for context injection
- [ ] Compose prompts with retrieved chunks prior to inference
- [ ] Allow pluggable retrieval backends (e.g. Elasticsearch, SQLite FTS)
- [ ] Cache retrieval results for repeated queries
- [ ] Demonstrate RAG flow with example documents
- [ ] Benchmark retrieval + generation latency

### Testing and CI
- [ ] Write unit tests for Rust modules
- [ ] Write unit tests for PHP endpoints
- [ ] Set up GitHub Actions for continuous integration
- [ ] Add benchmarks for inference speed
- [ ] Run Clippy and PHPStan as part of CI
- [ ] Create integration tests across the Rust/PHP boundary

### Documentation & Examples
- [ ] Expand main README with command examples
- [ ] Generate Rust docs via `cargo doc` and host them
- [ ] Provide thorough code comments for each module
- [ ] Write a tutorial walking through training and inference
- [ ] Include architecture diagrams and flowcharts
- [ ] Add example notebooks demonstrating usage

### Deployment
- [ ] Create a Dockerfile for the complete stack
- [ ] Provide Kubernetes manifests and Helm chart
- [ ] Document local deployment with Docker Compose
- [ ] Automate releases with versioned Git tags
- [ ] Supply example systemd service files for production

