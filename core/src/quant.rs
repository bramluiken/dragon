pub fn quantize_i8(weights: &[Vec<f32>]) -> (Vec<Vec<i8>>, f32) {
    let max_abs = weights
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0f32, |m, &v| m.max(v.abs()));
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    let quant = weights
        .iter()
        .map(|row| {
            row.iter()
                .map(|&v| ((v / scale).round().clamp(-128.0, 127.0) as i8))
                .collect::<Vec<i8>>()
        })
        .collect::<Vec<_>>();
    (quant, scale)
}

pub fn dequantize_i8(weights: &[Vec<i8>], scale: f32) -> Vec<Vec<f32>> {
    weights
        .iter()
        .map(|row| row.iter().map(|&v| v as f32 * scale).collect::<Vec<f32>>())
        .collect::<Vec<_>>()
}

pub struct QuantizedLinear {
    weight: Vec<Vec<i8>>, // shape: in_dim x out_dim
    bias: Vec<f32>,       // shape: out_dim
    scale: f32,
}

impl QuantizedLinear {
    pub fn new(weight: Vec<Vec<i8>>, bias: Vec<f32>, scale: f32) -> Self {
        Self { weight, bias, scale }
    }

    pub fn from_linear(layer: &super::Linear) -> Self {
        let (wq, s) = quantize_i8(&layer.weight);
        Self {
            weight: wq,
            bias: layer.bias.clone(),
            scale: s,
        }
    }

    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        input
            .iter()
            .map(|row| {
                (0..self.weight[0].len())
                    .map(|j| {
                        let mut sum = 0.0f32;
                        for i in 0..row.len() {
                            sum += row[i] * self.weight[i][j] as f32 * self.scale;
                        }
                        sum + self.bias[j]
                    })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Linear;

    #[test]
    fn quant_dequant_roundtrip() {
        let weights = vec![vec![0.5f32, -0.5], vec![1.0, -1.0]];
        let (q, s) = quantize_i8(&weights);
        let deq = dequantize_i8(&q, s);
        for i in 0..weights.len() {
            for j in 0..weights[0].len() {
                assert!((weights[i][j] - deq[i][j]).abs() < 1e-2);
            }
        }
    }

    #[test]
    fn quantized_linear_approx() {
        let weight = vec![vec![0.5f32, -0.5], vec![1.0, -1.0]];
        let bias = vec![0.1f32, -0.1];
        let linear = Linear::new(weight.clone(), bias.clone());
        let qlinear = QuantizedLinear::from_linear(&linear);
        let input = vec![vec![0.2f32, 0.4]];
        let out_f = linear.forward(&input);
        let out_q = qlinear.forward(&input);
        for j in 0..bias.len() {
            assert!((out_f[0][j] - out_q[0][j]).abs() < 1e-1);
        }
    }
}
