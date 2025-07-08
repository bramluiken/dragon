use std::fs::File;
use std::io::{self, BufRead, BufReader};

use crate::tokenizer::WhitespaceTokenizer;

/// Streaming text dataset backed by a file.
///
/// The loader reads the corpus line by line and converts each line
/// into token id sequences using the provided [`WhitespaceTokenizer`].
/// This avoids loading the entire dataset into memory and is suitable
/// for large text corpora.
pub struct TextDataset {
    reader: BufReader<File>,
    tokenizer: WhitespaceTokenizer,
}

impl TextDataset {
    /// Opens the dataset at `path` with the given tokenizer.
    pub fn open<P: AsRef<std::path::Path>>(path: P, tokenizer: WhitespaceTokenizer) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self {
            reader: BufReader::new(file),
            tokenizer,
        })
    }

    /// Returns the next sample as `(input, target)` token id vectors.
    ///
    /// `Ok(None)` is returned when the end of the file is reached.
    pub fn next_sample(&mut self) -> io::Result<Option<(Vec<usize>, Vec<usize>)>> {
        let mut line = String::new();
        loop {
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                return Ok(None);
            }
            let tokens = self.tokenizer.encode(line.trim_end());
            line.clear();
            if tokens.len() > 1 {
                let input = tokens[..tokens.len() - 1].to_vec();
                let target = tokens[1..].to_vec();
                return Ok(Some((input, target)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn iterate_samples() {
        let path = std::env::temp_dir().join("dataset_test.txt");
        {
            let mut f = File::create(&path).unwrap();
            writeln!(f, "hello world").unwrap();
            writeln!(f, "foo bar").unwrap();
        }
        let vocab = vec!["hello".into(), "world".into(), "foo".into(), "bar".into()];
        let tok = WhitespaceTokenizer::new(vocab, 0);
        let mut ds = TextDataset::open(&path, tok).unwrap();

        let s1 = ds.next_sample().unwrap().unwrap();
        assert_eq!(s1.0, vec![0]);
        assert_eq!(s1.1, vec![1]);

        let s2 = ds.next_sample().unwrap().unwrap();
        assert_eq!(s2.0, vec![2]);
        assert_eq!(s2.1, vec![3]);

        assert!(ds.next_sample().unwrap().is_none());
        let _ = std::fs::remove_file(&path);
    }
}
