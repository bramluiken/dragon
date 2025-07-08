pub mod embedding;
pub mod attention;
pub mod feedforward;
pub mod decoder;
pub mod transformer;
pub mod layernorm;
pub mod rotary;
pub mod model;
pub mod tokenizer;
pub mod loss;
pub mod ffi;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

/// Simple linear layer using nested vectors for storage.
pub struct Linear {
    weight: Vec<Vec<f32>>, // shape: in_dim x out_dim
    bias: Vec<f32>,        // shape: out_dim
}

impl Linear {
    /// Creates a new [`Linear`] layer.
    pub fn new(weight: Vec<Vec<f32>>, bias: Vec<f32>) -> Self {
        Self { weight, bias }
    }

    /// Applies the linear transformation to an input matrix.
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        input
            .iter()
            .map(|row| {
                self.weight[0]
                    .iter()
                    .enumerate()
                    .map(|(j, _)| {
                        row.iter()
                            .enumerate()
                            .map(|(i, x)| x * self.weight[i][j])
                            .sum::<f32>()
                            + self.bias[j]
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
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn linear_forward() {
        let weight = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]]; // 2x2
        let bias = vec![0.5f32, -0.5];
        let layer = Linear::new(weight, bias);
        let input = vec![vec![1.0f32, 1.0]]; // 1x2
        let output = layer.forward(&input);
        let expected = vec![vec![1.0 * 1.0 + 1.0 * 3.0 + 0.5,
                                 1.0 * 2.0 + 1.0 * 4.0 - 0.5]];
        assert_eq!(output, expected);
    }
}
