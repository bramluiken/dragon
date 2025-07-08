use std::collections::HashMap;

/// Tokenizer that simply splits text on ASCII whitespace.
///
/// Each whitespace separated token is looked up in the provided vocabulary.
/// Unknown tokens are mapped to `unk_id`.
pub struct WhitespaceTokenizer {
    vocab: HashMap<String, usize>,
    inv_vocab: Vec<String>,
    unk_id: usize,
}

impl WhitespaceTokenizer {
    /// Creates a new [`WhitespaceTokenizer`].
    pub fn new(vocab: Vec<String>, unk_id: usize) -> Self {
        let mut map = HashMap::new();
        for (i, tok) in vocab.iter().enumerate() {
            map.insert(tok.clone(), i);
        }
        Self {
            vocab: map,
            inv_vocab: vocab,
            unk_id,
        }
    }

    /// Encodes `text` by splitting on whitespace and converting each token to an id.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|t| self.vocab.get(t).cloned().unwrap_or(self.unk_id))
            .collect()
    }

    /// Decodes a sequence of ids back into a whitespace separated string.
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&id| self.inv_vocab.get(id).cloned().unwrap_or_default())
            .collect::<Vec<String>>()
            .join(" ")
    }
}

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

    /// Returns the current vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inv_vocab.len()
    }

    /// Adds a new merge pair to the tokenizer and returns the id of the newly
    /// created token. If the merged token already exists, its id is returned
    /// and the merge order is updated accordingly.
    pub fn add_merge(&mut self, a: &str, b: &str) -> usize {
        let merged = format!("{}{}", a, b);
        let id = if let Some(&existing) = self.vocab.get(&merged) {
            existing
        } else {
            let new_id = self.inv_vocab.len();
            self.vocab.insert(merged.clone(), new_id);
            self.inv_vocab.push(merged.clone());
            new_id
        };
        let rank = self.merges.len();
        self.merges
            .entry((a.to_string(), b.to_string()))
            .or_insert(rank);
        id
    }

    /// Learns up to `num_merges` new merges from `text` based on token pair
    /// frequencies. Merges are applied greedily one at a time.
    pub fn learn_merges(&mut self, text: &str, num_merges: usize) {
        for _ in 0..num_merges {
            let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
            for word in text.split_whitespace() {
                let tokens = self.encode(word);
                for pair in tokens.windows(2) {
                    *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
                }
            }
            let ((a, b), _) = match counts.into_iter().max_by_key(|p| p.1) {
                Some(p) => p,
                None => break,
            };
            let a_tok = self.inv_vocab[a].clone();
            let b_tok = self.inv_vocab[b].clone();
            self.add_merge(&a_tok, &b_tok);
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
    fn whitespace_roundtrip() {
        let vocab = vec!["<unk>".into(), "hello".into(), "world".into()];
        let tok = WhitespaceTokenizer::new(vocab.clone(), 0);
        let text = "hello world";
        let ids = tok.encode(text);
        assert_eq!(ids, vec![1, 2]);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

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

    #[test]
    fn roundtrip_all_vocab_tokens() {
        let vocab = vec![
            "<unk>".into(),
            "h".into(),
            "e".into(),
            "l".into(),
            "o".into(),
            "w".into(),
            "r".into(),
            "d".into(),
            "he".into(),
            "hel".into(),
            "hell".into(),
            "hello".into(),
            "wo".into(),
            "wor".into(),
            "worl".into(),
            "world".into(),
        ];
        let merges = vec![
            ("h".into(), "e".into()),
            ("he".into(), "l".into()),
            ("hel".into(), "l".into()),
            ("hell".into(), "o".into()),
            ("w".into(), "o".into()),
            ("wo".into(), "r".into()),
            ("wor".into(), "l".into()),
            ("worl".into(), "d".into()),
        ];
        let tok = BpeTokenizer::new(vocab.clone(), merges, 0);
        for token in vocab.iter().skip(1) {
            let ids = tok.encode(token);
            let decoded = tok.decode(&ids);
            assert_eq!(decoded, *token, "token {}", token);
        }
    }

    #[test]
    fn roundtrip_empty_string() {
        let vocab = vec!["<unk>".into()];
        let tok = BpeTokenizer::new(vocab, Vec::new(), 0);
        let ids = tok.encode("");
        assert!(ids.is_empty());
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "");
    }

    #[test]
    fn dynamic_merge() {
        let vocab = vec!["<unk>".into(), "a".into(), "b".into()];
        let mut tok = BpeTokenizer::new(vocab, Vec::new(), 0);
        let id = tok.add_merge("a", "b");
        assert_eq!(id, 3);
        let encoded = tok.encode("ab");
        assert_eq!(encoded, vec![id]);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, "ab");
    }

    #[test]
    fn learn_merges_from_text() {
        let vocab = vec!["<unk>".into(), "a".into(), "b".into()];
        let mut tok = BpeTokenizer::new(vocab, Vec::new(), 0);
        tok.learn_merges("ab ab", 1);
        assert_eq!(tok.vocab_size(), 3 + 1);
        let encoded = tok.encode("ab");
        assert_eq!(encoded.len(), 1);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, "ab");
    }
}
