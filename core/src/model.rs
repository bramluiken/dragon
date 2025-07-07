use crate::{embedding::Embedding, transformer::Transformer, Linear};

/// End-to-end decoder-only model tying together embedding, transformer and output layer.
///
/// All submodules are initialized with identity weights so that the entire model
/// acts as an identity function in tests when given small vocab/embedding sizes.
pub struct Model {
    pub embedding: Embedding,
    pub transformer: Transformer,
    pub output_layer: Linear,
}

impl Model {
    /// Creates a new [`Model`] with the specified dimensions.
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
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
            transformer: Transformer::new(num_layers, embed_dim, hidden_dim),
            output_layer: Linear::new(output_weights, vec![0.0; vocab_size]),
        }
    }

    /// Runs the model on token ids and returns logits over the vocabulary.
    pub fn forward(&self, input: &[usize]) -> Vec<Vec<f32>> {
        let embedded = self.embedding.forward(input);
        let transformed = self.transformer.forward(&embedded);
        self.output_layer.forward(&transformed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_forward_shapes() {
        let model = Model::new(2, 2, 2, 1);
        let input = vec![0usize, 1];
        let output = model.forward(&input);
        assert_eq!(output.len(), input.len());
        assert_eq!(output[0].len(), 2);
        assert_eq!(output[1].len(), 2);
    }
}
