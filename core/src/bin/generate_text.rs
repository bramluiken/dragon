use dragon_core::model::Model;
use dragon_core::tokenizer::WhitespaceTokenizer;
use std::env;
use std::fs;

fn main() {
    let mut args = env::args().skip(1);
    let vocab_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: generate_text <vocab.txt> <prompt> <steps>");
            std::process::exit(1);
        }
    };
    let prompt = match args.next() {
        Some(t) => t,
        None => {
            eprintln!("Usage: generate_text <vocab.txt> <prompt> <steps>");
            std::process::exit(1);
        }
    };
    let steps: usize = match args.next() {
        Some(s) => s.parse().expect("invalid steps"),
        None => {
            eprintln!("Usage: generate_text <vocab.txt> <prompt> <steps>");
            std::process::exit(1);
        }
    };

    let vocab_contents = fs::read_to_string(vocab_path).expect("failed to read vocab file");
    let vocab: Vec<String> = vocab_contents.lines().map(|s| s.to_string()).collect();

    let tokenizer = WhitespaceTokenizer::new(vocab.clone(), 0);
    let mut tokens = tokenizer.encode(&prompt);

    let vocab_size = vocab.len();
    let embed_dim = 4;
    let hidden_dim = 4;
    let num_layers = 1;
    let num_heads = 1;

    let model = Model::new(vocab_size, embed_dim, hidden_dim, num_layers, num_heads);
    tokens = model.generate(&tokens, steps);

    let out_text = tokenizer.decode(&tokens);
    println!("{}", out_text);
}
