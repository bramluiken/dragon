<?php
// Example usage of FFI bindings to dragon-core.
// Usage: php ffi_client.php <token0> <token1> ...

if (!extension_loaded('FFI')) {
    fwrite(STDERR, "FFI extension not available\n");
    exit(1);
}

$header = "
    typedef unsigned long ulong;
    typedef struct ModelHandle ModelHandle;
    ModelHandle* dragon_model_create(ulong vocab, ulong embed, ulong hidden, ulong layers);
    void dragon_model_free(ModelHandle* handle);
    ulong dragon_model_generate_inplace(ModelHandle* handle, ulong* tokens, ulong len, ulong steps);
";

$lib = FFI::cdef($header, realpath(__DIR__ . '/../../core/target/debug/libdragon_core.so'));
require_once __DIR__ . '/../hyperparams.php';

$tokens = array_map('intval', array_slice($argv, 1));
if (empty($tokens)) {
    fwrite(STDERR, "Usage: php ffi_client.php <token0> <token1> ...\n");
    exit(1);
}

$handle = $lib->dragon_model_create(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS);


$steps = 2;
$buf = FFI::new("ulong[".(count($tokens)+$steps)."]", false);
foreach ($tokens as $i => $t) {
    $buf[$i] = $t;
}
$len = $lib->dragon_model_generate_inplace($handle, $buf, count($tokens), $steps);

$result = [];
for ($i = 0; $i < $len; $i++) {
    $result[] = $buf[$i];
}

echo json_encode($result) . PHP_EOL;

$lib->dragon_model_free($handle);
?>
