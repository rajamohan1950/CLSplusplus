# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public issue
2. Email the maintainer or create a [private security advisory](https://github.com/rajamohan1950/CLSplusplus/security/advisories/new)
3. Include steps to reproduce and impact assessment
4. We will acknowledge within 48 hours and provide an update on remediation

## Security Practices

- API keys are validated with constant-time comparison
- Rate limiting is enforced per key
- Input validation on all endpoints (namespace, item_id, text length)
- Parameterized SQL to prevent injection
