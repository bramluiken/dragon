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
    ulong dragon_model_generate(ModelHandle* handle, const ulong* tokens, ulong len, ulong steps, ulong* out);
";

$lib = FFI::cdef($header, realpath(__DIR__ . '/../../core/target/debug/libdragon_core.so'));

$tokens = array_map('intval', array_slice($argv, 1));
if (empty($tokens)) {
    fwrite(STDERR, "Usage: php ffi_client.php <token0> <token1> ...\n");
    exit(1);
}

$handle = $lib->dragon_model_create(16, 4, 4, 1);

$in = FFI::new("ulong[".count($tokens)."]", false);
foreach ($tokens as $i => $t) {
    $in[$i] = $t;
}

$steps = 2;
$out = FFI::new("ulong[".(count($tokens)+$steps)."]", false);
$len = $lib->dragon_model_generate($handle, $in, count($tokens), $steps, $out);

$result = [];
for ($i = 0; $i < $len; $i++) {
    $result[] = $out[$i];
}

echo json_encode($result) . PHP_EOL;

$lib->dragon_model_free($handle);
?>
