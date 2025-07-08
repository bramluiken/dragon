use std::fs::File;
use std::io::{self, BufRead, BufReader};
use rand::seq::SliceRandom;
use rand::thread_rng;

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

/// In-memory dataloader that yields shuffled batches of samples.
pub struct DataLoader {
    samples: Vec<(Vec<usize>, Vec<usize>)>,
    batch_size: usize,
    index: usize,
}

impl DataLoader {
    /// Load all samples from `dataset` into memory and optionally shuffle them.
    pub fn new(mut dataset: TextDataset, batch_size: usize, shuffle: bool) -> io::Result<Self> {
        let mut samples = Vec::new();
        while let Some(sample) = dataset.next_sample()? {
            samples.push(sample);
        }
        if shuffle {
            samples.shuffle(&mut thread_rng());
        }
        Ok(Self { samples, batch_size, index: 0 })
    }

    /// Returns the next batch as `(inputs, targets)` or `None` at end of epoch.
    pub fn next_batch(&mut self) -> Option<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        if self.index >= self.samples.len() {
            return None;
        }
        let end = (self.index + self.batch_size).min(self.samples.len());
        let batch = &self.samples[self.index..end];
        let (inputs, targets): (Vec<_>, Vec<_>) = batch.iter().cloned().unzip();
        self.index = end;
        Some((inputs, targets))
    }

    /// Resets the dataloader to the beginning and optionally reshuffles.
    pub fn reset(&mut self, shuffle: bool) {
        self.index = 0;
        if shuffle {
            self.samples.shuffle(&mut thread_rng());
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

    #[test]
    fn dataloader_batches() {
        let path = std::env::temp_dir().join("dataset_loader_test.txt");
        {
            let mut f = File::create(&path).unwrap();
            writeln!(f, "a b c d").unwrap();
            writeln!(f, "e f g h").unwrap();
        }
        let vocab = vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into(), "f".into(), "g".into(), "h".into()];
        let tok = WhitespaceTokenizer::new(vocab, 0);
        let ds = TextDataset::open(&path, tok).unwrap();
        let mut dl = DataLoader::new(ds, 2, true).unwrap();
        let batch = dl.next_batch().unwrap();
        assert_eq!(batch.0.len(), 2);
        assert!(dl.next_batch().is_none());
        let _ = std::fs::remove_file(&path);
    }
}
