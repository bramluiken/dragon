use std::collections::HashMap;

/// Simple byte pair encoding (BPE) tokenizer.
///
/// The tokenizer loads a vocabulary and merge operations and applies
/// merges greedily based on their rank. It is intentionally minimal and
/// meant for demonstration only.
pub struct BpeTokenizer {
    vocab: HashMap<String, usize>,
    inv_vocab: Vec<String>,
    merges: HashMap<(String, String), usize>,
    unk_id: usize,
}

impl BpeTokenizer {
    /// Creates a new tokenizer from a list of vocabulary tokens and a list of
    /// merge pairs ordered by priority. `unk_id` specifies the id returned for
    /// unknown tokens.
    pub fn new(
        vocab: Vec<String>,
        merges: Vec<(String, String)>,
        unk_id: usize,
    ) -> Self {
        let mut map = HashMap::new();
        for (i, tok) in vocab.iter().enumerate() {
            map.insert(tok.clone(), i);
        }
        let mut merge_map = HashMap::new();
        for (rank, (a, b)) in merges.iter().enumerate() {
            merge_map.insert((a.clone(), b.clone()), rank);
        }
        Self {
            vocab: map,
            inv_vocab: vocab,
            merges: merge_map,
            unk_id,
        }
    }

    fn encode_word(&self, word: &str) -> Vec<usize> {
        let mut pieces: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if pieces.is_empty() {
            return Vec::new();
        }
        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..pieces.len() - 1 {
                let pair = (pieces[i].clone(), pieces[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    if best.map(|b| rank < b.0).unwrap_or(true) {
                        best = Some((rank, i));
                    }
                }
            }
            match best {
                Some((_rank, idx)) => {
                    let merged = format!("{}{}", pieces[idx], pieces[idx + 1]);
                    pieces[idx] = merged;
                    pieces.remove(idx + 1);
                    if pieces.len() == 1 {
                        break;
                    }
                }
                None => break,
            }
        }
        pieces
            .into_iter()
            .map(|p| self.vocab.get(&p).cloned().unwrap_or(self.unk_id))
            .collect()
    }

    /// Encodes text into token ids using greedy BPE merges. Words are split on
    /// ASCII whitespace before BPE is applied.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            ids.extend(self.encode_word(word));
        }
        ids
    }

    /// Decodes token ids back into text by simply concatenating token strings.
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&id| self.inv_vocab.get(id).cloned().unwrap_or_default())
            .collect::<Vec<String>>()
            .join("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let vocab = vec![
            "<unk>".into(),
            "h".into(),
            "e".into(),
            "l".into(),
            "o".into(),
            "he".into(),
            "hel".into(),
            "hell".into(),
            "hello".into(),
        ];
        let merges = vec![
            ("h".into(), "e".into()),
            ("he".into(), "l".into()),
            ("hel".into(), "l".into()),
            ("hell".into(), "o".into()),
        ];
        let tok = BpeTokenizer::new(vocab.clone(), merges, 0);
        let text = "hello";
        let ids = tok.encode(text);
        assert_eq!(ids, vec![8]);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn unknown_token() {
        let vocab = vec!["<unk>".into(), "h".into()];
        let tok = BpeTokenizer::new(vocab, Vec::new(), 0);
        let ids = tok.encode("xy");
        assert_eq!(ids, vec![0, 0]);
    }
}
