use super::Linear;

/// Simple two-layer feedforward network with GELU activation.
pub struct FeedForward {
    pub w1: Linear,
    pub w2: Linear,
}

impl FeedForward {
    /// Creates a new [`FeedForward`] layer with identity weights.
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        let identity_embed = (0..embed_dim)
            .map(|i| {
                (0..hidden_dim)
                    .map(|j| if i == j { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();
        let identity_hidden = (0..hidden_dim)
            .map(|i| {
                (0..embed_dim)
                    .map(|j| if i == j { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();
        let bias_w1 = vec![0.0f32; hidden_dim];
        let bias_w2 = vec![0.0f32; embed_dim];
        Self {
            w1: Linear::new(identity_embed, bias_w1),
            w2: Linear::new(identity_hidden, bias_w2),
        }
    }

    /// Forward pass through the feedforward network.
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let hidden = self.w1.forward(input);
        let activated = hidden
            .iter()
            .map(|row| row.iter().map(|x| gelu(*x)).collect::<Vec<f32>>())
            .collect::<Vec<_>>();
        self.w2.forward(&activated)
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (x * (2.0_f32 / std::f32::consts::PI).sqrt() * (1.0 + 0.044_715 * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feedforward_identity() {
        let layer = FeedForward::new(2, 2);
        let input = vec![vec![0.5f32, -0.5]];
        let output = layer.forward(&input);
        let expected = vec![vec![gelu(0.5), gelu(-0.5)]];
        assert!((output[0][0] - expected[0][0]).abs() < 1e-5);
        assert!((output[0][1] - expected[0][1]).abs() < 1e-5);
    }
}

