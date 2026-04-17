# Hyperspace Pods

Pool your machines into one AI cluster. Distributed and sharded inference across a mesh of consumer devices — laptops, desktops, VMs — with an OpenAI-compatible API.

---

## Install the CLI

```bash
curl -fsSL https://agents.hyper.space/api/install | bash
```

The installer auto-detects your GPU, downloads the best model for your hardware, and starts the daemon.

```bash
hyperspace --version
hyperspace status
```

To update, run the install command again.

---

## What are Pods

A **Pod** is a private compute cluster. Members install the CLI, someone creates a pod, shares an invite link, and the machines form a mesh. Models are served across the mesh — a query routes to whichever node has the best model loaded.

**The core idea**: a 32B parameter model that doesn't fit on any single laptop can run across two 16 GB machines. The CLI auto-detects available VRAM across the pod, splits transformer layers proportionally, and streams activations between nodes over libp2p.

```
  ┌─────────────────────────────────────────────────┐
  │              Hyperspace Pod Mesh                 │
  │                                                 │
  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   │
  │  │ Laptop A │   │ Laptop B │   │   VM C   │   │
  │  │ 16GB GPU │   │ 16GB GPU │   │  8GB GPU │   │
  │  │ layers   │   │ layers   │   │ smaller  │   │
  │  │ 0–31     │◄─►│ 32–63    │   │ models   │   │
  │  └──────────┘   └──────────┘   └──────────┘   │
  │       ▲               ▲              ▲         │
  │       └───────────────┴──────────────┘         │
  │              libp2p shard protocols             │
  │                                                 │
  │  ┌──────────────────────────────────────────┐  │
  │  │   OpenAI-compatible API (pk_* key)       │  │
  │  │   http://localhost:8080/v1               │  │
  │  └──────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────┘
```

Every pod also gets:
- **Shared cloud providers** — pool OpenRouter, Groq, Together, or any cloud API key as a fallback when local VRAM is full.
- **OpenAI-compatible API** — a `pk_*` key that works with any OpenAI SDK client, Claude Code, aider, continue, cursor, or custom scripts.

---

## Quick start

### 1. Create a pod

```bash
hyperspace pod create "my-lab"
```

### 2. Invite members

```bash
hyperspace pod invite
# → Invite code: hp_inv_abc123...
# → Share link:  https://hyper.space/join/abc123
```

Options:

```bash
hyperspace pod invite --role admin      # admin permissions
hyperspace pod invite --ttl 2d          # expires in 2 days
hyperspace pod invite --multi-use       # reusable invite
```

### 3. Members join

```bash
hyperspace pod join hp_inv_abc123
```

### 4. Check what's available

```bash
hyperspace pod status                   # online nodes, total VRAM, models
hyperspace pod models                   # all models across the mesh
hyperspace pod resources                # per-node VRAM, CPU, loaded models
```

### 5. Shard a large model

```bash
hyperspace pod shard qwen3.5-32b
```

The CLI computes the optimal shard plan automatically. Each node downloads its layer range and begins serving.

### 6. Query the pod

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer pk_abc123..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-32b",
    "messages": [{"role":"user","content":"Hello from the pod"}]
  }'
```

---

## Distributed inference — how it works

### Layer sharding

When you run `hyperspace pod shard <model>`, the CLI:

1. **Surveys the pod** — discovers every node's available VRAM and loaded models.
2. **Estimates model size** — extracts parameter count from the model name (7b, 27b, 32b, 70b, 8x7b MoE, etc.) and accounts for quantization (Q4_K_M = 4.5 bits/param) + 20% KV cache overhead.
3. **Computes the shard plan** — splits transformer layers proportional to each node's free VRAM. Two 16 GB nodes sharding a 32B model each get ~half the layers.
4. **Pulls model weights** — each node downloads only its assigned layer range. Sources: Ollama (fastest), HuggingFace GGUF (auto-selects Q4_K_M > Q5_K_M > Q4_K_S), or direct URL.
5. **Activates the ring** — nodes begin listening on the shard protocols.

### Shard protocols

Three libp2p protocols handle inter-node communication:

| Protocol | Purpose |
|---|---|
| `/hyperspace/shard-activation/1.0.0` | Stream binary activations from one node's layers to the next |
| `/hyperspace/shard-request/1.0.0` | Route an incoming inference request to the first shard |
| `/hyperspace/shard-token/1.0.0` | Stream generated tokens from the tail shard back to the head for output |

Inference flow for a sharded request:

```
Request → Node A (layers 0–31)
              │ activations
              ▼
          Node B (layers 32–63)
              │ logits
              ▼
          Token sampling → stream response back
```

Each node uses its local inference backend (Ollama, llama-server, or native engine) for its assigned layers.

### Smart routing

The pod gateway evaluates routing options in priority order:

1. **Pod-distributed** — sharded model running across local nodes (fastest, free)
2. **Pod-peer** — federated pod via alliance (if local is at capacity)
3. **Cloud-BYOK** — admin's own cloud API keys (OpenRouter, Groq, etc.)
4. **Cloud-funded** — platform keys, charges pod treasury

If the requested model isn't sharded but a single node has it loaded, the request routes directly to that node with no sharding overhead.

### Model recommendations

| Combined VRAM | Recommended shard |
|---|---|
| 16 GB (2 × 8 GB) | Gemma 3 12B, Qwen 2.5 14B |
| 32 GB (2 × 16 GB) | Qwen 3.5 32B, DeepSeek Coder V2 Lite |
| 48 GB (3 × 16 GB) | Gemma 3 27B (full precision), Codestral 22B |
| 64 GB (4 × 16 GB) | Qwen 2.5 72B (Q4), Llama 3.1 70B (Q4) |
| 96 GB+ | Qwen 2.5 72B (Q8), DeepSeek V3 (Q4) |

---

## API keys

Every pod gets OpenAI-compatible API keys.

```bash
hyperspace pod keys create --name "dev-key" --scopes inference,embed
# → pk_abc123def456...
```

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="pk_abc123def456..."
)

response = client.chat.completions.create(
    model="qwen3.5-32b",
    messages=[{"role": "user", "content": "Hello from pods"}]
)
print(response.choices[0].message.content)
```

### TypeScript

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8080/v1',
  apiKey: 'pk_abc123def456...',
});

const response = await client.chat.completions.create({
  model: 'qwen3.5-32b',
  messages: [{ role: 'user', content: 'Hello from pods' }],
});
console.log(response.choices[0].message.content);
```

### curl

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer pk_abc123def456..." \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-32b","messages":[{"role":"user","content":"Hello"}]}'
```

Features: rate limiting per key (default 60 RPM), daily/monthly spend limits, model allowlist, usage tracking, and `allow_public_overflow` to fall back to the Hyperspace network when the pod is at capacity.

```bash
hyperspace pod keys list                # list keys with usage stats
```

---

## Command reference

| Command | Description |
|---|---|
| `hyperspace pod create <name>` | Create a new pod |
| `hyperspace pod join <invite-code>` | Join an existing pod |
| `hyperspace pod leave` | Leave the current pod |
| `hyperspace pod status` | Show pod status, online nodes, total VRAM |
| `hyperspace pod members` | List members with roles + online status |
| `hyperspace pod invite` | Generate a shareable invite token |
| `hyperspace pod models` | List all models available across the mesh |
| `hyperspace pod resources` | Per-node breakdown: VRAM, CPU, loaded models |
| `hyperspace pod shard <model>` | Activate distributed inference for a model |
| `hyperspace pod keys create` | Mint a new `pk_*` API key |
| `hyperspace pod keys list` | List all API keys with usage |
| `hyperspace pod providers` | List configured cloud provider fallbacks |
| `hyperspace pod usage` | Current usage + cost breakdown |

All commands support `--json` for structured output (used by MCP / Claude Code).

---

## Using with Claude Code

The Hyperspace CLI exposes itself as an MCP server. Claude Code can manage your pod and route inference through it.

### Setup

Add Hyperspace as an MCP server in your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "hyperspace": {
      "command": "hyperspace",
      "args": ["mcp"]
    }
  }
}
```

### Example prompts in Claude Code

```
"Create a pod called research-lab and invite my teammate"
"What models are available in my pod?"
"Shard qwen3.5-32b across the pod"
"Mint an API key for the frontend app"
```

### Use the pod as Claude Code's inference backend

```bash
export OPENAI_BASE_URL="http://localhost:8080/v1"
export OPENAI_API_KEY="pk_your_pod_key"

# Any tool that uses the OpenAI SDK now talks to your pod:
# aider, continue, cursor, or custom scripts.
```

### Device linking

Link your CLI to your Hyperspace account so the web UI can send commands to your machine:

```bash
hyperspace login
# The CLI heartbeats to your account and polls for remote commands.
```

Remote commands: install model, unload model, restart, shard model, pull URL. All queued with 1-hour TTL.

---

## Links

- **Changelog**: [changelog.hyper.space](https://changelog.hyper.space)
- **Network**: [agents.hyper.space](https://agents.hyper.space)
- **GitHub**: [hyperspaceai/agi](https://github.com/hyperspaceai/agi)
- **CLI Install**: `curl -fsSL https://agents.hyper.space/api/install | bash`
- **Twitter**: [@HyperspaceAI](https://x.com/HyperspaceAI) · [@varun_mathur](https://x.com/varun_mathur)
