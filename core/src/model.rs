use crate::{embedding::Embedding, transformer::Transformer, Linear, rotary::RotaryEmbedding};
use crate::serialization::{self, Tensor};
use std::collections::BTreeMap;
use serde_json::json;

/// End-to-end decoder-only model tying together embedding, transformer and output layer.
///
/// All submodules are initialized with identity weights so that the entire model
/// acts as an identity function in tests when given small vocab/embedding sizes.
pub struct Model {
    pub embedding: Embedding,
    pub positional: RotaryEmbedding,
    pub transformer: Transformer,
    pub output_layer: Linear,
}

impl Model {
    /// Creates a new [`Model`] with the specified dimensions.
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize, num_layers: usize, num_heads: usize) -> Self {
        // embedding weights: vocab_size x embed_dim identity-like matrix
        let embed_weights = (0..vocab_size)
            .map(|i| {
                (0..embed_dim)
                    .map(|j| if i == j { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();
        // output weights: embed_dim x vocab_size identity-like matrix
        let output_weights = (0..embed_dim)
            .map(|i| {
                (0..vocab_size)
                    .map(|j| if i == j { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();
        Self {
            embedding: Embedding::new(embed_weights),
            positional: RotaryEmbedding::new(embed_dim),
            transformer: Transformer::new(num_layers, embed_dim, hidden_dim, num_heads),
            output_layer: Linear::new(output_weights, vec![0.0; vocab_size]),
        }
    }

    /// Runs the model on token ids and returns logits over the vocabulary.
    pub fn forward(&self, input: &[usize]) -> Vec<Vec<f32>> {
        let embedded = self.embedding.forward(input);
        let positioned = self.positional.forward(&embedded);
        let transformed = self.transformer.forward(&positioned);
        self.output_layer.forward(&transformed)
    }

    /// Autoregressively generates additional tokens using greedy decoding.
    ///
    /// `steps` specifies how many new tokens to generate beyond the provided
    /// `input`. The returned vector contains the original input followed by the
    /// generated tokens.
    pub fn generate(&self, input: &[usize], steps: usize) -> Vec<usize> {
        let mut tokens = input.to_vec();
        for _ in 0..steps {
            let logits = self.forward(&tokens);
            if let Some(last) = logits.last() {
                let next = last
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                tokens.push(next);
            }
        }
        tokens
    }

    /// Appends a new token to the embedding and output layers and returns its id.
    pub fn add_token(&mut self) -> usize {
        let id = self.embedding.add_token();
        self.output_layer.add_output();
        id
    }

    /// Current size of the model's vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.embedding.weights.len()
    }

    /// Saves the model weights to a `.safetensors` file.
    pub fn save_safetensors(&self, path: &str) -> std::io::Result<()> {
        let mut tensors: BTreeMap<String, Tensor> = BTreeMap::new();

        let vocab_size = self.embedding.weights.len();
        let embed_dim = self.embedding.weights[0].len();
        tensors.insert(
            "embedding.weight".into(),
            Tensor {
                shape: vec![vocab_size, embed_dim],
                data: self
                    .embedding
                    .weights
                    .iter()
                    .flat_map(|v| v.clone())
                    .collect(),
            },
        );

        let out_in = self.output_layer.weight.len();
        let out_out = self.output_layer.weight[0].len();
        tensors.insert(
            "output.weight".into(),
            Tensor {
                shape: vec![out_in, out_out],
                data: self
                    .output_layer
                    .weight
                    .iter()
                    .flat_map(|v| v.clone())
                    .collect(),
            },
        );
        tensors.insert(
            "output.bias".into(),
            Tensor {
                shape: vec![self.output_layer.bias.len()],
                data: self.output_layer.bias.clone(),
            },
        );

        for (i, block) in self.transformer.blocks.iter().enumerate() {
            let prefix = format!("layers.{}", i);
            tensors.insert(
                format!("{}.ln1.gamma", prefix),
                Tensor {
                    shape: vec![block.ln1.gamma.len()],
                    data: block.ln1.gamma.clone(),
                },
            );
            tensors.insert(
                format!("{}.ln1.beta", prefix),
                Tensor {
                    shape: vec![block.ln1.beta.len()],
                    data: block.ln1.beta.clone(),
                },
            );
            tensors.insert(
                format!("{}.ln2.gamma", prefix),
                Tensor {
                    shape: vec![block.ln2.gamma.len()],
                    data: block.ln2.gamma.clone(),
                },
            );
            tensors.insert(
                format!("{}.ln2.beta", prefix),
                Tensor {
                    shape: vec![block.ln2.beta.len()],
                    data: block.ln2.beta.clone(),
                },
            );

            add_linear(&mut tensors, &block.self_attn.w_q, &format!("{}.attn.w_q", prefix));
            add_linear(&mut tensors, &block.self_attn.w_k, &format!("{}.attn.w_k", prefix));
            add_linear(&mut tensors, &block.self_attn.w_v, &format!("{}.attn.w_v", prefix));
            add_linear(&mut tensors, &block.self_attn.w_o, &format!("{}.attn.w_o", prefix));
            add_linear(&mut tensors, &block.feedforward.w1, &format!("{}.ff.w1", prefix));
            add_linear(&mut tensors, &block.feedforward.w2, &format!("{}.ff.w2", prefix));
        }

        let meta = json!({"num_layers": self.transformer.blocks.len()});
        serialization::write_safetensors(&tensors, path, Some(meta))
    }

    /// Loads a model from a `.safetensors` file.
    pub fn load_safetensors(path: &str) -> std::io::Result<Self> {
        let (tensors, meta) = serialization::read_safetensors(path)?;
        let num_layers = meta
            .as_ref()
            .and_then(|m| m.get("num_layers"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;

        let embed = tensors.get("embedding.weight").unwrap();
        let vocab_size = embed.shape[0];
        let embed_dim = embed.shape[1];

        let ff = tensors
            .get("layers.0.ff.w1.weight")
            .expect("missing feedforward weight");
        let hidden_dim = ff.shape[1];

        let num_heads = 1;
        let mut model = Model::new(vocab_size, embed_dim, hidden_dim, num_layers, num_heads);

        model.embedding.weights = matrix(embed);
        model.output_layer.weight = matrix(tensors.get("output.weight").unwrap());
        model.output_layer.bias = tensors.get("output.bias").unwrap().data.clone();

        for i in 0..num_layers {
            let prefix = format!("layers.{}", i);
            let block = &mut model.transformer.blocks[i];
            block.ln1.gamma = tensors.get(&format!("{}.ln1.gamma", prefix)).unwrap().data.clone();
            block.ln1.beta = tensors.get(&format!("{}.ln1.beta", prefix)).unwrap().data.clone();
            block.ln2.gamma = tensors.get(&format!("{}.ln2.gamma", prefix)).unwrap().data.clone();
            block.ln2.beta = tensors.get(&format!("{}.ln2.beta", prefix)).unwrap().data.clone();
            load_linear(&mut block.self_attn.w_q, &tensors, &format!("{}.attn.w_q", prefix));
            load_linear(&mut block.self_attn.w_k, &tensors, &format!("{}.attn.w_k", prefix));
            load_linear(&mut block.self_attn.w_v, &tensors, &format!("{}.attn.w_v", prefix));
            load_linear(&mut block.self_attn.w_o, &tensors, &format!("{}.attn.w_o", prefix));
            load_linear(&mut block.feedforward.w1, &tensors, &format!("{}.ff.w1", prefix));
            load_linear(&mut block.feedforward.w2, &tensors, &format!("{}.ff.w2", prefix));
        }
        Ok(model)
    }
}

fn add_linear(tensors: &mut BTreeMap<String, Tensor>, linear: &Linear, name: &str) {
    let in_dim = linear.weight.len();
    let out_dim = linear.weight[0].len();
    tensors.insert(
        format!("{}.weight", name),
        Tensor {
            shape: vec![in_dim, out_dim],
            data: linear.weight.iter().flat_map(|v| v.clone()).collect(),
        },
    );
    tensors.insert(
        format!("{}.bias", name),
        Tensor {
            shape: vec![linear.bias.len()],
            data: linear.bias.clone(),
        },
    );
}

fn load_linear(linear: &mut Linear, tensors: &BTreeMap<String, Tensor>, name: &str) {
    let w = tensors.get(&format!("{}.weight", name)).unwrap();
    let b = tensors.get(&format!("{}.bias", name)).unwrap();
    linear.weight = matrix(w);
    linear.bias = b.data.clone();
}

fn matrix(t: &Tensor) -> Vec<Vec<f32>> {
    let rows = t.shape[0];
    let cols = t.shape[1];
    t.data.chunks(cols).map(|c| c.to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_forward_shapes() {
        let model = Model::new(2, 2, 2, 1, 1);
        let input = vec![0usize, 1];
        let output = model.forward(&input);
        assert_eq!(output.len(), input.len());
        assert_eq!(output[0].len(), 2);
        assert_eq!(output[1].len(), 2);
    }

    #[test]
    fn model_generate_length() {
        let model = Model::new(2, 2, 2, 1, 1);
        let input = vec![0usize];
        let generated = model.generate(&input, 3);
        assert_eq!(generated.len(), 4);
        assert!(generated.iter().all(|&t| t < 2));
    }

    #[test]
    fn model_save_load_roundtrip() {
        let model = Model::new(2, 2, 2, 1, 1);
        let path = "test_model.safetensors";
        model.save_safetensors(path).unwrap();
        let loaded = Model::load_safetensors(path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(model.embedding.weights, loaded.embedding.weights);
        assert_eq!(model.output_layer.bias, loaded.output_layer.bias);
    }
}
