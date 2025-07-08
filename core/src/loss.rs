/// Simple cross-entropy loss for logits and integer targets.
///
/// The function expects `logits` as a sequence of vectors where each
/// vector represents unnormalized log probabilities for one step. The
/// `targets` slice contains the expected token index for each step.
pub fn cross_entropy(logits: &[Vec<f32>], targets: &[usize]) -> f32 {
    assert_eq!(logits.len(), targets.len());
    let mut loss = 0.0f32;
    for (logit, &target) in logits.iter().zip(targets.iter()) {
        let max = logit
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logit.iter().map(|x| (*x - max).exp()).sum();
        let log_prob = logit[target] - max - exp_sum.ln();
        loss -= log_prob;
    }
    loss / logits.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_loss_for_correct_prediction() {
        let logits_good = vec![vec![2.0f32, 0.1]];
        let logits_bad = vec![vec![0.1f32, 2.0]];
        let target = vec![0usize];
        let good = cross_entropy(&logits_good, &target);
        let bad = cross_entropy(&logits_bad, &target);
        assert!(good < bad);
    }
}
