<?php
// Asynchronous HTTP server using Swoole to invoke dragon-core inference.
// Requires the Swoole PHP extension.

require_once __DIR__ . '/util.php';

use Swoole\Http\Server;
use Swoole\Http\Request;
use Swoole\Http\Response;

require_once __DIR__ . '/middleware.php';

$host = '0.0.0.0';
$port = 8080;

$server = new Server($host, $port);

$server->on('start', function (Server $server) use ($host, $port) {
    echo "Swoole HTTP server listening on http://{$host}:{$port}\n";
});

$server->on('request', function (Request $request, Response $response) {
<<<<<< codex/improve-logging-and-error-handling
    try {
        if ($request->server['request_method'] !== 'POST') {
            respond_error($response, 'POST only', 405);
            return;
        }

        $data = json_decode($request->rawContent(), true);
        if (!is_array($data) || !isset($data['tokens']) || !is_array($data['tokens'])) {
            respond_error($response, 'Expected JSON with "tokens" array', 400);
            return;
        }

        $tokens = array_map('intval', $data['tokens']);
        log_info('Inference request: ' . json_encode($tokens));

        $binary = realpath(__DIR__ . '/../../core/target/debug/infer');
        if ($binary === false || !is_file($binary)) {
            respond_error($response, 'Inference binary not found', 500);
            return;
        }

        $cmd = escapeshellcmd($binary . ' ' . implode(' ', $tokens));
        $output = shell_exec($cmd);
        if ($output === null) {
            respond_error($response, 'Failed to execute inference binary', 500);
            return;
        }

        respond_json($response, ['raw' => $output]);
    } catch (Throwable $e) {
        respond_error($response, 'Unhandled error: ' . $e->getMessage(), 500);
=======
    $response->header('Content-Type', 'application/json');

    if (!check_auth_header($request->header['authorization'] ?? '')) {
        $response->status(401);
        $response->end(json_encode(['error' => 'Unauthorized']));
        return;
    }

    $ip = $request->server['remote_addr'] ?? 'unknown';
    if (!update_rate_limit($ip)) {
        $response->status(429);
        $response->end(json_encode(['error' => 'Rate limit exceeded']));
        return;
    }

    if ($request->server['request_method'] !== 'POST') {
        $response->status(405);
        $response->end(json_encode(['error' => 'POST only']));
        return;
    }

    $data = json_decode($request->rawContent(), true);
    if (!is_array($data) || !isset($data['tokens']) || !is_array($data['tokens'])) {
        $response->status(400);
        $response->end(json_encode(['error' => 'Expected JSON with "tokens" array']));
        return;
>>>>>> main
    }
});

$server->start();
