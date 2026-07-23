# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.7.x   | :white_check_mark: |
| 1.6.x   | :white_check_mark: |
| < 1.6   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email the maintainers directly at the address listed in the repository contact information, or use [GitHub's private vulnerability reporting](https://github.com/swarm-ai-research/swarm/security/advisories/new).

### What to include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### What to expect

- Acknowledgment within 48 hours
- Status update within 7 days
- We will coordinate disclosure timing with you

## Scope

This project is a research simulation framework. Security concerns include:

- **Code injection** via YAML scenario files or agent configurations
- **Dependency vulnerabilities** in third-party packages
- **LLM API key exposure** through logging or error messages
- **Arbitrary code execution** through agent or plugin mechanisms

## Security Measures

- CodeQL static analysis runs on every push and weekly
- Dependabot monitors dependencies for known vulnerabilities
- YAML loading uses `yaml.safe_load` (no arbitrary object instantiation)
- LLM API keys are read from environment variables, never stored in config files
