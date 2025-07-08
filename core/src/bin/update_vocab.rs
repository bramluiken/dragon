use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::Write;

fn usage() {
    eprintln!("Usage: update_vocab <old_vocab.txt> <new_text.txt> <output_vocab.txt> [limit]");
}

fn main() {
    let mut args = env::args().skip(1);
    let old_vocab_path = match args.next() {
        Some(p) => p,
        None => {
            usage();
            std::process::exit(1);
        }
    };
    let new_text_path = match args.next() {
        Some(p) => p,
        None => {
            usage();
            std::process::exit(1);
        }
    };
    let output_path = match args.next() {
        Some(p) => p,
        None => {
            usage();
            std::process::exit(1);
        }
    };
    let limit: Option<usize> = args.next().and_then(|s| s.parse().ok());

    let vocab_contents = fs::read_to_string(&old_vocab_path).expect("failed to read vocab file");
    let mut vocab: Vec<String> = vocab_contents.lines().map(|s| s.to_string()).collect();
    let mut existing: HashSet<String> = vocab.iter().cloned().collect();

    let new_text = fs::read_to_string(&new_text_path).expect("failed to read text file");
    let mut counts: HashMap<String, usize> = HashMap::new();
    for word in new_text.split_whitespace() {
        if !existing.contains(word) {
            *counts.entry(word.to_string()).or_insert(0) += 1;
        }
    }

    let mut items: Vec<(String, usize)> = counts.into_iter().collect();
    items.sort_by(|a, b| b.1.cmp(&a.1));

    for (token, _) in items {
        if limit.map(|l| vocab.len() < l).unwrap_or(true) {
            vocab.push(token.clone());
            existing.insert(token);
        } else {
            break;
        }
    }

    // if a limit is provided and the vocab exceeds it, truncate
    if let Some(lim) = limit {
        if vocab.len() > lim {
            vocab.truncate(lim);
        }
    }

    let mut file = fs::File::create(&output_path).expect("failed to create output file");
    for token in vocab {
        writeln!(file, "{}", token).expect("failed to write token");
    }
}
