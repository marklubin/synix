CloudSync Troubleshooting Guide

Symptom: Files are not syncing between devices.
Check the sync status indicator in the system tray. A yellow icon means the client is connected but has pending changes. A red icon means the client cannot reach the sync server. Verify your network connection and check whether your organization's firewall allows outbound connections on port 443. If using a proxy, configure it in Settings > Network > Proxy Configuration.

Symptom: Conflict copies appearing unexpectedly.
Conflict copies are created when two devices edit the same binary file simultaneously. This is expected behavior for files that cannot be automatically merged (images, PDFs, compiled binaries). To reduce conflicts, enable file locking for binary file types in Settings > Collaboration > File Locking. For text-based files, ensure all team members are running client version 4.2 or later, which includes improved CRDT merge support.

Symptom: Search results are missing recently added files.
The search index updates asynchronously after file uploads. New files typically appear in search results within 30 seconds. If files remain missing after several minutes, check the indexing status in the admin dashboard under System > Search > Index Health. A backlog greater than 1000 documents may indicate the search service needs additional resources. Contact support if the backlog persists.

Symptom: High CPU usage on the client device.
The CloudSync client uses background threads for file watching, chunk hashing, and compression. On initial sync of a large folder (more than 50,000 files), CPU usage may spike temporarily. This is normal and subsides once the initial index is built. If high CPU persists during normal operation, check for filesystem loops (symlinks pointing to parent directories) and exclude them in Settings > Sync > Excluded Paths.
