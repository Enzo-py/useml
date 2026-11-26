---
title: useml monitor
emoji: 📈
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

A private dashboard for experiments run with useml. Machines push snapshots to it through
outgoing HTTPS connections; it keeps them in memory only. Deployed by `useml monitor deploy`.
