#!/usr/bin/env node
// Example Node.js client for the dragon API.
// Usage: node client.js <token0> <token1> ...

const http = require('http');

const tokens = process.argv.slice(2).map(t => parseInt(t, 10));
if (tokens.length === 0) {
  console.error('Usage: node client.js <token0> <token1> ...');
  process.exit(1);
}

const data = JSON.stringify({ tokens });

const options = {
  hostname: 'localhost',
  port: 8080,
  path: '/index.php',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(data),
    'Authorization': 'Bearer ' + (process.env.DRAGON_API_KEY || 'secret')
  }
};

const req = http.request(options, res => {
  res.setEncoding('utf8');
  let raw = '';
  res.on('data', chunk => { raw += chunk; });
  res.on('end', () => { console.log(raw); });
});

req.on('error', err => {
  console.error('Request error:', err.message);
});

req.write(data);
req.end();

