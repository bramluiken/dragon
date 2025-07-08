use crate::decoder::DecoderBlock;

/// Simple Transformer consisting of repeated [`DecoderBlock`]s.
pub struct Transformer {
    blocks: Vec<DecoderBlock>,
}

impl Transformer {
    /// Creates a new [`Transformer`] with `num_layers` blocks.
    pub fn new(num_layers: usize, embed_dim: usize, hidden_dim: usize, num_heads: usize) -> Self {
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(DecoderBlock::new(embed_dim, hidden_dim, num_heads));
        }
        Self { blocks }
    }

    /// Runs the transformer on the provided sequence.
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.blocks.iter().fold(input.to_owned(), |acc, block| block.forward(&acc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformer_forward_shape() {
        let model = Transformer::new(3, 2, 2, 1);
        let input = vec![vec![1.0f32, -1.0]];
        let output = model.forward(&input);
        assert_eq!(output.len(), input.len());
        assert_eq!(output[0].len(), input[0].len());
    }
}
