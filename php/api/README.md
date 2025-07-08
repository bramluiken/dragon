# API

Simple HTTP endpoint that invokes the Rust inference binary.

## Usage

1. Build the Rust core: from the repo root run `cargo build --bin infer`.
2. Start a PHP server inside this directory:
   ```bash
   php -S localhost:8080
   ```
3. Send a POST request with JSON tokens:
   ```bash
   curl -X POST -d '{"tokens": [0,1,2]}' http://localhost:8080/index.php
   ```
   The response contains the raw output from the inference binary.

### Authentication

Requests must include an `Authorization: Bearer <token>` header. Set the
expected token via the `DRAGON_API_KEY` environment variable (default `secret`).

### Rate limiting

Each client IP is limited to 60 requests per minute. Exceeding the limit returns
HTTP `429`.

## Async server (Swoole)

If the [Swoole](https://www.swoole.co.uk/) extension is installed you can run
an asynchronous server:

```bash
php swoole_server.php
```

Requests are identical to the `index.php` example.
