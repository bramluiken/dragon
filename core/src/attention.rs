// Naive multi-head self-attention implementation.
// This is a simplified, non-optimized module purely for demonstration.
use std::f32;

/// Computes scaled dot-product attention for a single head.
/// `q`, `k`, `v` are matrices of shape (seq_len x dim).
fn scaled_dot_product_attention(q: &[Vec<f32>], k: &[Vec<f32>], v: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let seq_len = q.len();
    let dim = q[0].len() as f32;
    let scale = (dim).sqrt();
    let mut output = vec![vec![0.0f32; v[0].len()]; seq_len];

    for i in 0..seq_len {
        // compute attention scores for token i
        let mut scores = vec![0.0f32; seq_len];
        for j in 0..=i { // causal mask
            let mut dot = 0.0f32;
            for d in 0..q[i].len() {
                dot += q[i][d] * k[j][d];
            }
            scores[j] = dot / scale;
        }
        // softmax over j
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scores
            .iter()
            .map(|s| if *s != 0.0 { (*s - max_score).exp() } else { 0.0 })
            .sum();
        for j in 0..=i {
            let weight = if exp_sum > 0.0 {
                ((scores[j] - max_score).exp()) / exp_sum
            } else {
                0.0
            };
            for d in 0..v[0].len() {
                output[i][d] += weight * v[j][d];
            }
        }
    }
    output
}

/// Basic self-attention layer with a single head.
pub struct SelfAttention {
    pub w_q: super::Linear,
    pub w_k: super::Linear,
    pub w_v: super::Linear,
    pub w_o: super::Linear,
}

impl SelfAttention {
    pub fn new(embed_dim: usize) -> Self {
        // initialize with identity weights for simplicity
        let identity = (0..embed_dim)
            .map(|i| {
                (0..embed_dim)
                    .map(|j| if i == j { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();
        let bias = vec![0.0f32; embed_dim];
        Self {
            w_q: super::Linear::new(identity.clone(), bias.clone()),
            w_k: super::Linear::new(identity.clone(), bias.clone()),
            w_v: super::Linear::new(identity.clone(), bias.clone()),
            w_o: super::Linear::new(identity, bias),
        }
    }

    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let q = self.w_q.forward(input);
        let k = self.w_k.forward(input);
        let v = self.w_v.forward(input);
        let context = scaled_dot_product_attention(&q, &k, &v);
        self.w_o.forward(&context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_attention_identity() {
        let layer = SelfAttention::new(2);
        // using a single token ensures the causal mask keeps the output equal to input
        let input = vec![vec![1.0f32, 0.0]];
        let output = layer.forward(&input);
        // with identity weights and bias zero, output should equal input
        assert_eq!(output, input);
    }
}

