<?php
// Example of calling Rust tokenizer via FFI.
// Usage: php ffi_tokenize.php "your text"

if (!extension_loaded('FFI')) {
    fwrite(STDERR, "FFI extension not available\n");
    exit(1);
}

$header = "
    typedef unsigned long ulong;
    typedef struct TokenizerHandle TokenizerHandle;
    TokenizerHandle* dragon_tokenizer_create(const char* vocab, const char* merges, ulong unk);
    void dragon_tokenizer_free(TokenizerHandle* handle);
    ulong dragon_tokenizer_encode(const TokenizerHandle* handle, const char* text, ulong* out, ulong cap);
";

$lib = FFI::cdef($header, realpath(__DIR__ . '/../../core/target/debug/libdragon_core.so'));

$vocab = realpath(__DIR__ . '/../../data/tokenizer/vocab.txt');
$merges = realpath(__DIR__ . '/../../data/tokenizer/merges.txt');

if ($vocab === false || $merges === false) {
    fwrite(STDERR, "Tokenizer files not found\n");
    exit(1);
}

$handle = $lib->dragon_tokenizer_create($vocab, $merges, 0);

$input = $argv[1] ?? '';
if ($input === '') {
    fwrite(STDERR, "Usage: php ffi_tokenize.php \"text\"\n");
    exit(1);
}

$cap = strlen($input) + 16; // enough for demo
$out = FFI::new("ulong[$cap]", false);
$len = $lib->dragon_tokenizer_encode($handle, $input, $out, $cap);

$result = [];
for ($i = 0; $i < $len; $i++) {
    $result[] = $out[$i];
}

echo json_encode($result) . PHP_EOL;

$lib->dragon_tokenizer_free($handle);
?>
