use dragon_core::model::Model;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: infer <token ids>");
        std::process::exit(1);
    }

    let tokens: Vec<usize> = args.iter().map(|a| a.parse::<usize>().expect("invalid token" )).collect();

    // Example model dimensions; in real usage load actual weights.
    let vocab_size = 16;
    let embed_dim = 4;
    let hidden_dim = 4;
    let num_layers = 1;

    let model = Model::new(vocab_size, embed_dim, hidden_dim, num_layers);
    let logits = model.forward(&tokens);

    for (idx, logit) in logits.iter().enumerate() {
        println!("step {} -> {:?}", idx, logit);
    }
}
