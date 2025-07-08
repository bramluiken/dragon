# PHP Orchestration

Entry point for dragon API.

All API endpoints now emit structured JSON errors and write logs using PHP's
`error_log` facility.

## Example clients

The `examples/` directory contains small scripts that demonstrate how to call
the API from PHP or Node.js:

```bash
# Using PHP
php examples/client.php 0 1 2

# Using Node.js
node examples/client.js 0 1 2

# Using PHP FFI bindings for inference (zero-copy)
php examples/ffi_client.php 0 1 2

# Tokenizing text via FFI
php examples/ffi_tokenize.php "hello"
```
