use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;

fn main() {
    let mut args = env::args().skip(1);
    let input_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: train_vocab <input.txt> <output.txt> [limit]");
            std::process::exit(1);
        }
    };
    let output_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: train_vocab <input.txt> <output.txt> [limit]");
            std::process::exit(1);
        }
    };
    let limit: Option<usize> = args.next().and_then(|s| s.parse().ok());

    let contents = fs::read_to_string(&input_path).expect("failed to read input file");
    let mut counts: HashMap<String, usize> = HashMap::new();
    for word in contents.split_whitespace() {
        *counts.entry(word.to_string()).or_insert(0) += 1;
    }

    let mut items: Vec<(String, usize)> = counts.into_iter().collect();
    items.sort_by(|a, b| b.1.cmp(&a.1));
    if let Some(lim) = limit {
        items.truncate(lim);
    }

    let mut file = fs::File::create(&output_path).expect("failed to create output file");
    for (token, _) in items {
        writeln!(file, "{}", token).expect("failed to write token");
    }
}

