<?php
// Example PHP client for the streaming API.
// Usage: php stream_client.php <steps> <token0> [token1 ...]

$steps = intval($argv[1] ?? 1);
$tokens = array_map('intval', array_slice($argv, 2));
if (empty($tokens)) {
    fwrite(STDERR, "Usage: php stream_client.php <steps> <token0> [token1 ...]\n");
    exit(1);
}

$payload = json_encode(['tokens' => $tokens, 'steps' => $steps], JSON_UNESCAPED_SLASHES);

$ch = curl_init('http://localhost:8080/stream.php');
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
    CURLOPT_POSTFIELDS => $payload,
    CURLOPT_WRITEFUNCTION => function($ch, $data) {
        echo $data;
        return strlen($data);
    },
]);

curl_exec($ch);
if (curl_errno($ch)) {
    fwrite(STDERR, 'Request failed: ' . curl_error($ch) . "\n");
}

curl_close($ch);
?>
