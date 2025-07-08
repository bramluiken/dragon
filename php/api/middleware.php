<?php
// Simple authentication and rate limiting helpers used by the API.

function check_auth_header(string $header): bool {
    $expected = getenv('DRAGON_API_KEY') ?: 'secret';
    if (preg_match('/Bearer\s+(.*)/', $header, $m)) {
        return hash_equals($expected, trim($m[1]));
    }
    return false;
}

function update_rate_limit(string $ip, int $limit = 60, int $window = 60): bool {
    $file = sys_get_temp_dir() . '/dragon_rate_' . md5($ip);
    $now = time();
    if (is_file($file)) {
        [$start, $count] = array_map('intval', explode(':', file_get_contents($file) ?: '0:0'));
        if ($now - $start >= $window) {
            $start = $now;
            $count = 1;
        } else {
            if ($count >= $limit) {
                return false;
            }
            $count++;
        }
    } else {
        $start = $now;
        $count = 1;
    }
    file_put_contents($file, $start . ':' . $count, LOCK_EX);
    return true;
}
?>
