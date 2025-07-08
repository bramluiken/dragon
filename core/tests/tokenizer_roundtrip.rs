use dragon_core::tokenizer::BpeTokenizer;

#[test]
fn encode_decode_encode_identity() {
    let vocab = vec![
        "<unk>".to_string(),
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
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
        ("hel".to_string(), "l".to_string()),
        ("hell".to_string(), "o".to_string()),
        ("w".to_string(), "o".to_string()),
        ("wo".to_string(), "r".to_string()),
        ("wor".to_string(), "l".to_string()),
        ("worl".to_string(), "d".to_string()),
    ];
    let tok = BpeTokenizer::new(vocab, merges, 0);
    let text = "hello";
    let ids1 = tok.encode(text);
    let decoded = tok.decode(&ids1);
    let ids2 = tok.encode(&decoded);
    assert_eq!(ids1, ids2);
}
