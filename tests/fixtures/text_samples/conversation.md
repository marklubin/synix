---
title: Debugging Memory Leak in Production
date: 2025-02-01
tags: [debugging, production, memory]
---

User: We have a memory leak in our production Node.js service. RSS keeps growing from 512MB to 2GB over 24 hours. Where should I start investigating

Assistant: Start with heap snapshots. Take three snapshots at 5-minute intervals using Chrome DevTools protocol or the --inspect flag. Compare them to find objects that are growing.

User: Found it - we have an event listener leak in our WebSocket handler. Each reconnection adds a new listener without removing the old one.

Assistant: Classic Node.js memory leak pattern. Fix it by calling removeListener or removeAllListeners before adding new ones. Also consider using the once method for one-shot listeners.
