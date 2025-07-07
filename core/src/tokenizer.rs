/// Naive whitespace tokenizer.
///
/// This tokenizer splits text on ASCII whitespace and maps words to
/// integer ids based on a provided vocabulary. Unknown words are mapped
/// to a special `unk_id`.
use std::collections::HashMap;

pub struct WhitespaceTokenizer {
    vocab: HashMap<String, usize>,
    inv_vocab: Vec<String>,
    unk_id: usize,
}

impl WhitespaceTokenizer {
    /// Creates a new tokenizer from a list of vocabulary tokens.
    /// `unk_id` specifies the id returned for unknown words.
    pub fn new(vocab: Vec<String>, unk_id: usize) -> Self {
        let mut map = HashMap::new();
        for (i, tok) in vocab.iter().enumerate() {
            map.insert(tok.clone(), i);
        }
        Self { vocab: map, inv_vocab: vocab, unk_id }
    }

    /// Encodes space-separated text into token ids.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|w| self.vocab.get(w).cloned().unwrap_or(self.unk_id))
            .collect()
    }

    /// Decodes token ids back into a space-separated string.
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&id| self.inv_vocab.get(id).cloned().unwrap_or_else(|| "".into()))
            .collect::<Vec<String>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let vocab = vec!["hello".into(), "world".into()];
        let tok = WhitespaceTokenizer::new(vocab.clone(), 0);
        let text = "hello world";
        let ids = tok.encode(text);
        assert_eq!(ids, vec![0, 1]);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn unknown_token() {
        let vocab = vec!["a".into(), "b".into()];
        let tok = WhitespaceTokenizer::new(vocab, 1);
        let ids = tok.encode("c a");
        assert_eq!(ids, vec![1, 0]);
    }
}
