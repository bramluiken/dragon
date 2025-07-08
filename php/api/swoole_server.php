<?php
// Asynchronous HTTP server using Swoole to invoke dragon-core inference.
// Requires the Swoole PHP extension.

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
    }

    $tokens = array_map('intval', $data['tokens']);
    $binary = realpath(__DIR__ . '/../../core/target/debug/infer');

    if ($binary === false || !is_file($binary)) {
        $response->status(500);
        $response->end(json_encode(['error' => 'Inference binary not found']));
        return;
    }

    $cmd = escapeshellcmd($binary . ' ' . implode(' ', $tokens));
    $output = shell_exec($cmd);

    if ($output === null) {
        $response->status(500);
        $response->end(json_encode(['error' => 'Failed to execute inference binary']));
        return;
    }

    $response->end(json_encode(['raw' => $output]));
});

$server->start();
