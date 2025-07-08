<?php
// Simple HTTP endpoint to run dragon-core inference.
// Expects JSON: {"tokens": [1,2,3]}

<<<<<< codex/improve-logging-and-error-handling
require_once __DIR__ . '/util.php';

try {
    $raw = file_get_contents('php://input');
    if ($raw === false) {
        respond_error(null, 'No input', 400);
    }

    $data = json_decode($raw, true);
    if (!is_array($data) || !isset($data['tokens']) || !is_array($data['tokens'])) {
        respond_error(null, 'Expected JSON with "tokens" array', 400);
    }

    $tokens = array_map('intval', $data['tokens']);
    log_info('Inference request: ' . json_encode($tokens));

    $binary = realpath(__DIR__ . '/../../core/target/debug/infer');
    if ($binary === false || !is_file($binary)) {
        respond_error(null, 'Inference binary not found', 500);
    }

    $cmd = escapeshellcmd($binary . ' ' . implode(' ', $tokens));
    $output = shell_exec($cmd);
    if ($output === null) {
        respond_error(null, 'Failed to execute inference binary', 500);
    }

    respond_json(null, ['raw' => $output]);
} catch (Throwable $e) {
    respond_error(null, 'Unhandled error: ' . $e->getMessage(), 500);
=======
require_once __DIR__ . '/middleware.php';

header('Content-Type: application/json');

if (!check_auth_header($_SERVER['HTTP_AUTHORIZATION'] ?? '')) {
    http_response_code(401);
    echo json_encode(['error' => 'Unauthorized']);
    exit;
}

if (!update_rate_limit($_SERVER['REMOTE_ADDR'] ?? 'unknown')) {
    http_response_code(429);
    echo json_encode(['error' => 'Rate limit exceeded']);
    exit;
}

$raw = file_get_contents('php://input');
if ($raw === false) {
    http_response_code(400);
    echo json_encode(['error' => 'No input']);
    exit;
}

$data = json_decode($raw, true);
if (!is_array($data) || !isset($data['tokens']) || !is_array($data['tokens'])) {
    http_response_code(400);
    echo json_encode(['error' => 'Expected JSON with "tokens" array']);
    exit;
}

$tokens = array_map('intval', $data['tokens']);
$binary = realpath(__DIR__ . '/../../core/target/debug/infer');

if ($binary === false || !is_file($binary)) {
    http_response_code(500);
    echo json_encode(['error' => 'Inference binary not found']);
    exit;
>>>>>> main
}
?>
