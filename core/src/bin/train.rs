use dragon_core::model::Model;
use dragon_core::tokenizer::WhitespaceTokenizer;
use dragon_core::loss::cross_entropy;
use std::env;
use std::fs;

fn main() {
    let mut args = env::args().skip(1);
    let vocab_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: train <vocab.txt> <text> [epochs]");
            std::process::exit(1);
        }
    };
    let text = match args.next() {
        Some(t) => t,
        None => {
            eprintln!("Usage: train <vocab.txt> <text> [epochs]");
            std::process::exit(1);
        }
    };
    let epochs: usize = args
        .next()
        .unwrap_or_else(|| "5".into())
        .parse()
        .expect("invalid epochs");

    let vocab_contents = fs::read_to_string(&vocab_path).expect("failed to read vocab file");
    let vocab: Vec<String> = vocab_contents.lines().map(|s| s.to_string()).collect();

    let tokenizer = WhitespaceTokenizer::new(vocab.clone(), 0);
    let tokens = tokenizer.encode(&text);
    if tokens.len() < 2 {
        eprintln!("Need at least two tokens to train");
        std::process::exit(1);
    }

    let inputs = &tokens[..tokens.len() - 1];
    let targets = &tokens[1..];

    let vocab_size = vocab.len();
    let embed_dim = 4;
    let hidden_dim = 4;
    let num_layers = 1;

    let mut model = Model::new(vocab_size, embed_dim, hidden_dim, num_layers);
    let lr = 0.1f32;

    for epoch in 0..epochs {
        let embedded = model.embedding.forward(inputs);
        let positioned = model.positional.forward(&embedded);
        let transformed = model.transformer.forward(&positioned);
        let logits = model.output_layer.forward(&transformed);

        let loss = cross_entropy(&logits, targets);
        println!("epoch {} loss {}", epoch, loss);

        let mut grad_w = vec![vec![0.0f32; vocab_size]; embed_dim];
        let mut grad_b = vec![0.0f32; vocab_size];
        for (step, &target) in targets.iter().enumerate() {
            let logit = &logits[step];
            let max = logit.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logit.iter().map(|x| (*x - max).exp()).sum();
            let softmax: Vec<f32> = logit.iter().map(|x| (*x - max).exp() / exp_sum).collect();
            for i in 0..vocab_size {
                let grad = softmax[i] - if i == target { 1.0 } else { 0.0 };
                grad_b[i] += grad;
                for j in 0..embed_dim {
                    grad_w[j][i] += transformed[step][j] * grad;
                }
            }
        }
        let n = targets.len() as f32;
        {
            let bias = model.output_layer.bias_mut();
            for i in 0..vocab_size {
                bias[i] -= lr * grad_b[i] / n;
            }
        }
        {
            let weight = model.output_layer.weight_mut();
            for i in 0..vocab_size {
                for j in 0..embed_dim {
                    weight[j][i] -= lr * grad_w[j][i] / n;
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
