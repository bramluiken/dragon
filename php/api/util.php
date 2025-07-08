<?php
function log_message(string $level, string $msg): void {
    $time = date('c');
    error_log("[$time][$level] $msg");
}

function log_info(string $msg): void { log_message('INFO', $msg); }
function log_error(string $msg): void { log_message('ERROR', $msg); }

function respond_json($resp, array $payload, int $status = 200): void {
    $body = json_encode($payload, JSON_UNESCAPED_SLASHES);
    if (is_object($resp) && method_exists($resp, 'header')) {
        $resp->status($status);
        $resp->header('Content-Type', 'application/json');
        $resp->end($body);
    } else {
        http_response_code($status);
        header('Content-Type: application/json');
        echo $body;
    }
}

function respond_error($resp, string $message, int $status): void {
    log_error($message);
    respond_json($resp, ['error' => $message, 'code' => $status], $status);
    if (!is_object($resp)) {
        exit;
    }
}
?>
