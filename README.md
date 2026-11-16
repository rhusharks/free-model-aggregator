[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/rhusharks/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/rhusharks/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

# free-model-aggregator

Entry for make.fun's [free-model-aggregator-02](https://make.fun/bounty/free-model-aggregator-02): one server that puts OpenRouter's free models behind a single endpoint.

- `POST /v1/chat/completions` — OpenAI wire, JSON and SSE streaming
- `POST /v1/messages` — Anthropic wire, JSON and SSE streaming
- `GET /v1/models` — model list in the shape of the auth header

Every completion is a real call to one of the brief's fourteen free models, chosen by best published rating that meets the request's requirements, with failover across the rest on a refusal (retries tracked in `x-makefun-upstream-failures`), per-model cooldowns, and a usage-preserving stream that never carries a duplicate finish chunk.

`node main.ts --list` prints the catalog document byte for byte as the server registers it.

Node 22 standard library alone. MIT licensed.
