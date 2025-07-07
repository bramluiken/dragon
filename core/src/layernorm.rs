use std::f32;

/// Simple Layer Normalization.
///
/// Each token vector is normalized independently: `y = (x - mean) / sqrt(var + eps) * gamma + beta`.
pub struct LayerNorm {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub eps: f32,
}

impl LayerNorm {
    /// Creates a new [`LayerNorm`] with unit gamma and zero beta.
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            eps: 1e-5,
        }
    }

    /// Applies layer normalization over the last dimension.
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        input
            .iter()
            .map(|row| {
                let len = row.len() as f32;
                let mean = row.iter().sum::<f32>() / len;
                let var = row
                    .iter()
                    .map(|x| {
                        let diff = *x - mean;
                        diff * diff
                    })
                    .sum::<f32>()
                    / len;
                let denom = (var + self.eps).sqrt();
                row.iter()
                    .enumerate()
                    .map(|(i, x)| {
                        self.gamma[i] * ((*x - mean) / denom) + self.beta[i]
                    })
                    .collect::<Vec<f32>>()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layernorm_zero_mean_unit_var() {
        let ln = LayerNorm::new(2);
        let input = vec![vec![1.0f32, -1.0]];
        let output = ln.forward(&input);
        let row = &output[0];
        let mean: f32 = row.iter().sum::<f32>() / row.len() as f32;
        let var: f32 = row
            .iter()
            .map(|x| {
                let diff = *x - mean;
                diff * diff
            })
            .sum::<f32>()
            / row.len() as f32;
        assert!(mean.abs() < 1e-6);
        assert!((var - 1.0).abs() < 1e-4);
    }
}

