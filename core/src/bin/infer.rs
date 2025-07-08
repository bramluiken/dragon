use dragon_core::model::Model;
use dragon_core::hyperparams::{DEFAULT_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: infer <token ids>");
        std::process::exit(1);
    }

    let tokens: Vec<usize> = args.iter().map(|a| a.parse::<usize>().expect("invalid token" )).collect();

    // Example model dimensions; in real usage load actual weights.
    let vocab_size = DEFAULT_VOCAB_SIZE;
    let model = Model::new(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS);
    let logits = model.forward(&tokens);

    for (idx, logit) in logits.iter().enumerate() {
        println!("step {} -> {:?}", idx, logit);
    }
}
