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
pub mod blas;

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
        let m = input.len();
        let k = input[0].len();
        let n = self.weight[0].len();

        let mut a = Vec::with_capacity(m * k);
        for row in input {
            a.extend_from_slice(row);
        }

        let mut b = Vec::with_capacity(k * n);
        for row in &self.weight {
            b.extend_from_slice(row);
        }

        let mut c = vec![0.0f32; m * n];
        crate::blas::sgemm(m, n, k, &a, &b, &mut c);

        let mut output = vec![vec![0.0f32; n]; m];
        for i in 0..m {
            for j in 0..n {
                output[i][j] = c[i * n + j] + self.bias[j];
            }
        }
        output
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
