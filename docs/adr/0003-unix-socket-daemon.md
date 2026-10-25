# ADR-003: Daemon over a unix socket, not HTTP

Status: Accepted

## Context

Loading the models takes a while, so a long-lived process that keeps them warm makes each `cyv say` fast. The daemon has one client: the local `cyv` CLI. It should not be reachable from the network, and it should not add much code or many dependencies.

## Decision

The daemon (`cyvd`) speaks length-prefixed JSON frames over a unix socket, using plain `asyncio`. It does not use FastAPI or any HTTP server.

- Socket: `~/.cyv/run/cyvd.sock`. Its directory is created with mode 0700 before the socket is bound. If the path is longer than macOS allows (about 104 bytes), the socket goes in `$TMPDIR/cyv-<uid>/`.
- The daemon checks the peer user ID on each connection.
- Requests: `health`, `say`, `convert`, `reload`, `shutdown`.
- The daemon returns audio as raw bytes. It never takes an output path. The CLI writes the file.
- The daemon exits after an idle timeout (15 minutes by default), only when its queue is empty.
- Spawning is race-free: the CLI takes a file lock before it starts the daemon.

## Consequences

- No network port, so no browser cross-site attacks against a localhost server.
- Less code and a faster import than a web framework.
- The socket permissions and peer-user check stop other users, but not malware running as you. This is stated in SECURITY.md.
- If we ever need a second kind of client, the protocol may need a new version. This is acceptable because the API is not public in v1.
