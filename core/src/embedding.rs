/// Simple token embedding layer.
///
/// Maps token indices to embedding vectors via a lookup table.
pub struct Embedding {
    pub weights: Vec<Vec<f32>>, // shape: vocab_size x embed_dim
}

impl Embedding {
    /// Creates a new [`Embedding`] with the provided weight matrix.
    pub fn new(weights: Vec<Vec<f32>>) -> Self {
        Self { weights }
    }

    /// Looks up embeddings for each token id in `input`.
    pub fn forward(&self, input: &[usize]) -> Vec<Vec<f32>> {
        input
            .iter()
            .map(|&idx| self.weights[idx].clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_lookup() {
        let weights = vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
            vec![0.5, 0.6],
        ];
        let emb = Embedding::new(weights);
        let input = vec![2usize, 0, 1];
        let output = emb.forward(&input);
        assert_eq!(output.len(), 3);
        assert_eq!(output[0], vec![0.5, 0.6]);
        assert_eq!(output[1], vec![0.1, 0.2]);
        assert_eq!(output[2], vec![0.3, 0.4]);
    }
}
