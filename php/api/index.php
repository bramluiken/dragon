<?php
// Simple HTTP endpoint to run dragon-core inference.
// Expects JSON: {"tokens": [1,2,3]}

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
}
?>
