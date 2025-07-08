use crate::attention::MultiHeadAttention;
use crate::feedforward::FeedForward;
use crate::layernorm::LayerNorm;

/// Simplified decoder block combining self-attention and feedforward layers.
///
/// This block applies self-attention followed by a feedforward network. Both
/// components use identity weights so the block can be unit tested easily.
pub struct DecoderBlock {
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub self_attn: MultiHeadAttention,
    pub feedforward: FeedForward,
}

impl DecoderBlock {
    /// Creates a new [`DecoderBlock`].
    pub fn new(embed_dim: usize, hidden_dim: usize, num_heads: usize) -> Self {
        Self {
            ln1: LayerNorm::new(embed_dim),
            ln2: LayerNorm::new(embed_dim),
            self_attn: MultiHeadAttention::new(embed_dim, num_heads),
            feedforward: FeedForward::new(embed_dim, hidden_dim),
        }
    }

    /// Runs the block on the provided sequence.
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let norm1 = self.ln1.forward(input);
        let attn_out = self.self_attn.forward(&norm1);
        let norm2 = self.ln2.forward(&attn_out);
        self.feedforward.forward(&norm2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_block_forward_shape() {
        let block = DecoderBlock::new(2, 2, 1);
        let input = vec![vec![0.5f32, -0.5]];
        let output = block.forward(&input);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 2);
    }
}
