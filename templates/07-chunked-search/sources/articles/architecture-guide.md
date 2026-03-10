CloudSync Architecture Guide

The CloudSync backend is built on a microservices architecture deployed on Kubernetes. Each service handles a specific domain: file storage, metadata, user authentication, real-time sync, and search indexing.

File storage uses a content-addressable store backed by object storage (S3-compatible). Files are split into variable-size chunks using a rolling hash algorithm (Rabin fingerprint) to enable efficient deduplication and delta syncing. Each chunk is compressed with zstd before storage.

The metadata service tracks file trees, permissions, and sharing state in a PostgreSQL database with logical replication for read scaling. Every file operation produces an event that flows through an Apache Kafka topic, consumed by the sync engine, search indexer, and audit logger.

The sync engine maintains a per-device vector clock to track causality. When two devices produce conflicting edits, the CRDT merge function resolves them deterministically without user intervention. For binary files that cannot be merged, CloudSync creates a conflict copy with both versions preserved.

Authentication supports SAML 2.0, OpenID Connect, and API keys. All API requests are authenticated via short-lived JWT tokens issued by the auth service. Rate limiting is enforced at the API gateway layer using a sliding window algorithm with per-endpoint quotas.

The real-time collaboration layer uses WebSockets with automatic fallback to HTTP long polling. Document operations are transformed using operational transformation (OT) for text documents and last-writer-wins registers for spreadsheet cells. Presence information (cursor position, active selection) is broadcast via a lightweight pub/sub channel.
