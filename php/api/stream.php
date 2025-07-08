<?php
// Streams generated tokens back to the client using Server-Sent Events (SSE).
// Expects JSON: {"tokens": [1,2,3], "steps": 2}

header('Content-Type: text/event-stream');
header('Cache-Control: no-cache');

$raw = file_get_contents('php://input');
if ($raw === false) {
    http_response_code(400);
    echo "event: error\ndata: No input\n\n";
    exit;
}

$data = json_decode($raw, true);
if (!is_array($data) || !isset($data['tokens']) || !is_array($data['tokens'])) {
    http_response_code(400);
    echo "event: error\ndata: Expected JSON with \"tokens\" array\n\n";
    exit;
}

$steps = isset($data['steps']) ? max(1, intval($data['steps'])) : 1;
$tokens = array_map('intval', $data['tokens']);
$binary = realpath(__DIR__ . '/../../core/target/debug/generate_tokens');

if ($binary === false || !is_file($binary)) {
    http_response_code(500);
    echo "event: error\ndata: Generate binary not found\n\n";
    exit;
}

$cmd = escapeshellcmd($binary . ' ' . $steps . ' ' . implode(' ', $tokens));
$proc = popen($cmd, 'r');
if ($proc === false) {
    http_response_code(500);
    echo "event: error\ndata: Failed to execute binary\n\n";
    exit;
}

while (!feof($proc)) {
    $line = fgets($proc);
    if ($line === false) {
        break;
    }
    $line = trim($line);
    if ($line === '') {
        continue;
    }
    echo "data: $line\n\n";
    ob_flush();
    flush();
}

pclose($proc);

echo "event: end\ndata: done\n\n";
?>
