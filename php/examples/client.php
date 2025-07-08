<?php
// Example PHP client for the dragon API.
// Usage: php client.php <token0> <token1> ...

$url = 'http://localhost:8080/index.php';
$tokens = array_slice($argv, 1);
if (empty($tokens)) {
    fwrite(STDERR, "Usage: php client.php <token0> <token1> ...\n");
    exit(1);
}

$payload = json_encode(['tokens' => array_map('intval', $tokens)], JSON_UNESCAPED_SLASHES);

$ch = curl_init($url);
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
    CURLOPT_POSTFIELDS => $payload,
    CURLOPT_RETURNTRANSFER => true,
]);

$response = curl_exec($ch);
if ($response === false) {
    fwrite(STDERR, 'Request failed: ' . curl_error($ch) . "\n");
    exit(1);
}

curl_close($ch);

echo $response . PHP_EOL;
?>
