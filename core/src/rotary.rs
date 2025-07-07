pub struct RotaryEmbedding {
    pub dim: usize,
    pub base: f32,
}

impl RotaryEmbedding {
    /// Creates a new `RotaryEmbedding` with the given dimension.
    /// The `base` determines the frequency scaling (default 10000.0).
    pub fn new(dim: usize) -> Self {
        Self { dim, base: 10000.0 }
    }

    /// Applies rotary positional encoding to the input sequence.
    ///
    /// Each token vector in `input` has length `dim` and `dim` must be even.
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut output = vec![vec![0.0f32; self.dim]; input.len()];
        for (pos, token) in input.iter().enumerate() {
            for i in 0..self.dim / 2 {
                let angle =
                    (pos as f32) / self.base.powf(2.0 * i as f32 / self.dim as f32);
                let cos = angle.cos();
                let sin = angle.sin();
                let x1 = token[2 * i];
                let x2 = token[2 * i + 1];
                output[pos][2 * i] = x1 * cos - x2 * sin;
                output[pos][2 * i + 1] = x1 * sin + x2 * cos;
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_forward_shape() {
        let rope = RotaryEmbedding::new(4);
        let input = vec![vec![1.0f32, 0.0, 0.5, -0.5]];
        let output = rope.forward(&input);
        assert_eq!(output.len(), input.len());
        assert_eq!(output[0].len(), input[0].len());
    }
}
