use crate::attention::SelfAttention;
use crate::feedforward::FeedForward;

/// Simplified decoder block combining self-attention and feedforward layers.
///
/// This block applies self-attention followed by a feedforward network. Both
/// components use identity weights so the block can be unit tested easily.
pub struct DecoderBlock {
    pub self_attn: SelfAttention,
    pub feedforward: FeedForward,
}

impl DecoderBlock {
    /// Creates a new [`DecoderBlock`].
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        Self {
            self_attn: SelfAttention::new(embed_dim),
            feedforward: FeedForward::new(embed_dim, hidden_dim),
        }
    }

    /// Runs the block on the provided sequence.
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let attn_out = self.self_attn.forward(input);
        self.feedforward.forward(&attn_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_block_forward_shape() {
        let block = DecoderBlock::new(2, 2);
        let input = vec![vec![0.5f32, -0.5]];
        let output = block.forward(&input);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 2);
    }
}
