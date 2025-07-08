use dragon_core::model::Model;
use dragon_core::tokenizer::BpeTokenizer;
use dragon_core::loss::cross_entropy;
use dragon_core::hyperparams::{EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, LEARNING_RATE};
use half::f16;
use std::env;
use std::fs;

fn main() {
    let mut args = env::args().skip(1);
    let vocab_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: train <vocab.txt> <merges.txt> <text> [epochs] [accum_steps] [--fp16]");
            std::process::exit(1);
        }
    };
    let merges_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: train <vocab.txt> <merges.txt> <text> [epochs] [accum_steps] [--fp16]");
            std::process::exit(1);
        }
    };
    let text = match args.next() {
        Some(t) => t,
        None => {
            eprintln!("Usage: train <vocab.txt> <merges.txt> <text> [epochs] [accum_steps] [--fp16]");
            std::process::exit(1);
        }
    };
    let epochs: usize = args
        .next()
        .unwrap_or_else(|| "5".into())
        .parse()
        .expect("invalid epochs");
    let accum_steps: usize = args
        .next()
        .unwrap_or_else(|| "1".into())
        .parse()
        .expect("invalid accumulation steps");
    let use_fp16 = args.next().map_or(false, |a| a == "--fp16");

    let vocab_contents = fs::read_to_string(&vocab_path).expect("failed to read vocab file");
    let vocab: Vec<String> = vocab_contents.lines().map(|s| s.to_string()).collect();
    let merges_contents = fs::read_to_string(&merges_path).expect("failed to read merges file");
    let merges: Vec<(String, String)> = merges_contents
        .lines()
        .filter_map(|l| {
            let mut parts = l.split_whitespace();
            let a = parts.next()?.to_string();
            let b = parts.next()?.to_string();
            Some((a, b))
        })
        .collect();

    let tokenizer = BpeTokenizer::new(vocab.clone(), merges, 0);
    let tokens = tokenizer.encode(&text);
    if tokens.len() < 2 {
        eprintln!("Need at least two tokens to train");
        std::process::exit(1);
    }

    let inputs = &tokens[..tokens.len() - 1];
    let targets = &tokens[1..];

    let vocab_size = vocab.len();
    let mut model = Model::new(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS);
    let lr = LEARNING_RATE;

    for epoch in 0..epochs {
        let embedded = model.embedding.forward(inputs);
        let positioned = model.positional.forward(&embedded);
        let transformed = model.transformer.forward(&positioned);
        let logits = model.output_layer.forward(&transformed);

        let loss = cross_entropy(&logits, targets);
        println!("epoch {} loss {}", epoch, loss);

        if use_fp16 {
            let mut grad_w = vec![vec![f16::from_f32(0.0); vocab_size]; EMBED_DIM];
            let mut grad_b = vec![f16::from_f32(0.0); vocab_size];
            let mut accum = 0usize;
            for (step, &target) in targets.iter().enumerate() {
                let logit = &logits[step];
                let max = logit.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = logit.iter().map(|x| (*x - max).exp()).sum();
                let softmax: Vec<f32> = logit.iter().map(|x| (*x - max).exp() / exp_sum).collect();
                for i in 0..vocab_size {
                    let grad = softmax[i] - if i == target { 1.0 } else { 0.0 };
                    grad_b[i] = f16::from_f32(grad_b[i].to_f32() + grad);
                    for j in 0..EMBED_DIM {
                        grad_w[j][i] = f16::from_f32(grad_w[j][i].to_f32() + transformed[step][j] * grad);
                    }
                }
                accum += 1;
                if accum == accum_steps {
                    let n = accum as f32;
                    {
                        let bias = model.output_layer.bias_mut();
                        for i in 0..vocab_size {
                            bias[i] -= lr * grad_b[i].to_f32() / n;
                            grad_b[i] = f16::from_f32(0.0);
                        }
                    }
                    {
                        let weight = model.output_layer.weight_mut();
                        for i in 0..vocab_size {
                            for j in 0..EMBED_DIM {
                                weight[j][i] -= lr * grad_w[j][i].to_f32() / n;
                                grad_w[j][i] = f16::from_f32(0.0);
                            }
                        }
                    }
                    accum = 0;
                }
            }
            if accum > 0 {
                let n = accum as f32;
                {
                    let bias = model.output_layer.bias_mut();
                    for i in 0..vocab_size {
                        bias[i] -= lr * grad_b[i].to_f32() / n;
                    }
                }
                {
                    let weight = model.output_layer.weight_mut();
                    for i in 0..vocab_size {
                        for j in 0..EMBED_DIM {
                            weight[j][i] -= lr * grad_w[j][i].to_f32() / n;
                        }
                    }
                }
            }
        } else {
            let mut grad_w = vec![vec![0.0f32; vocab_size]; EMBED_DIM];
            let mut grad_b = vec![0.0f32; vocab_size];
            let mut accum = 0usize;
            for (step, &target) in targets.iter().enumerate() {
                let logit = &logits[step];
                let max = logit.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = logit.iter().map(|x| (*x - max).exp()).sum();
                let softmax: Vec<f32> = logit.iter().map(|x| (*x - max).exp() / exp_sum).collect();
                for i in 0..vocab_size {
                    let grad = softmax[i] - if i == target { 1.0 } else { 0.0 };
                    grad_b[i] += grad;
                    for j in 0..EMBED_DIM {
                        grad_w[j][i] += transformed[step][j] * grad;
                    }
                }
                accum += 1;
                if accum == accum_steps {
                    let n = accum as f32;
                    {
                        let bias = model.output_layer.bias_mut();
                        for i in 0..vocab_size {
                            bias[i] -= lr * grad_b[i] / n;
                            grad_b[i] = 0.0;
                        }
                    }
                    {
                        let weight = model.output_layer.weight_mut();
                        for i in 0..vocab_size {
                            for j in 0..EMBED_DIM {
                                weight[j][i] -= lr * grad_w[j][i] / n;
                                grad_w[j][i] = 0.0;
                            }
                        }
                    }
                    accum = 0;
                }
            }
            if accum > 0 {
                let n = accum as f32;
                {
                    let bias = model.output_layer.bias_mut();
                    for i in 0..vocab_size {
                        bias[i] -= lr * grad_b[i] / n;
                    }
                }
                {
                    let weight = model.output_layer.weight_mut();
                    for i in 0..vocab_size {
                        for j in 0..EMBED_DIM {
                            weight[j][i] -= lr * grad_w[j][i] / n;
                        }
                    }
                }
            }
        }
    }

    // print final loss
    let embedded = model.embedding.forward(inputs);
    let positioned = model.positional.forward(&embedded);
    let transformed = model.transformer.forward(&positioned);
    let logits = model.output_layer.forward(&transformed);
    let loss = cross_entropy(&logits, targets);
    println!("final loss {}", loss);
}
