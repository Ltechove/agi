# Hyperspace Pods

Pool your machines into one AI cluster. Distributed and sharded inference across a mesh of consumer devices with an OpenAI-compatible API.

---

## Install the CLI

```bash
curl -fsSL https://agents.hyper.space/api/install | bash
```

Auto-detects your GPU, downloads the best model, and starts the daemon.

```bash
hyperspace --version
hyperspace status
```

---

## Quick start

```bash
# 1. Create a pod
hyperspace pod create "my-lab"

# 2. Invite a friend
hyperspace pod invite --role member --ttl 24h
# → Invite code: hp_inv_abc123...
# → Share link:  https://hyperspace.sh/join/hp_inv_abc123

# 3. Friend joins
hyperspace pod join hp_inv_abc123

# 4. See what's available
hyperspace pod status
hyperspace pod models --shardable

# 5. Shard a large model across the pod
hyperspace pod shard qwen3.5:32b

# 6. Query it
hyperspace pod infer -p "Explain distributed inference"

# 7. Or use the OpenAI-compatible API
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer pk_abc123..." \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-32b","messages":[{"role":"user","content":"Hello"}]}'
```

---

## How sharding works

```
  Request → Node A (layers 0–31, 16 GB)
                │ activations via libp2p
                ▼
            Node B (layers 32–63, 16 GB)
                │ logits
                ▼
            Token sampling → stream response back
```

When you run `hyperspace pod shard <model>`:

1. **Survey** — discovers every node's VRAM and loaded models
2. **Estimate** — extracts param count from model name, accounts for quantization (Q4_K_M = 4.5 bits/param) + 20% KV cache overhead
3. **Plan** — splits transformer layers proportional to each node's free VRAM
4. **Pull** — each node downloads only its assigned layer range (Ollama, HuggingFace GGUF, or direct URL)
5. **Activate** — nodes form a ring on three libp2p protocols:

| Protocol | Purpose |
|---|---|
| `/hyperspace/shard-activation/1.0.0` | Stream activations between layer ranges |
| `/hyperspace/shard-request/1.0.0` | Route inference requests to the first shard |
| `/hyperspace/shard-token/1.0.0` | Stream generated tokens from tail back to head |

### Model recommendations by combined VRAM

| Combined VRAM | Recommended shard |
|---|---|
| 16 GB (2 x 8 GB) | Gemma 3 12B, Qwen 2.5 14B |
| 32 GB (2 x 16 GB) | Qwen 3.5 32B, DeepSeek Coder V2 Lite |
| 48 GB (3 x 16 GB) | Gemma 3 27B (full precision) |
| 64 GB (4 x 16 GB) | Qwen 2.5 72B (Q4), Llama 3.1 70B (Q4) |
| 96 GB+ | Qwen 2.5 72B (Q8), DeepSeek V3 (Q4) |

### Smart routing

The gateway routes in priority order:

1. **Pod-distributed** — sharded model running across local nodes (fastest, free)
2. **Pod-peer** — federated pod via alliance
3. **Cloud-BYOK** — admin's own cloud API keys
4. **Cloud-funded** — platform keys, charges pod treasury

---

## Complete command reference

### Pod lifecycle

```bash
hyperspace pod create <name>
  --plan starter|team|business|enterprise    # default: starter
  --description "..."
  --cloud                                    # cloud-backed (requires login)
  --raft-port 7800                           # local Raft transport port
  --http-port 7801                           # local pod-raft HTTP port

hyperspace pod join <invite-code-or-url>
  # Accepts: hp_inv_abc123, https://hyperspace.sh/join/CODE, hsi_v1.xxx.yyy

hyperspace pod leave
  --force                                    # skip confirmation

hyperspace pod status                        # online nodes, VRAM, models, treasury
hyperspace pod members                       # table: user, role, status, GPU, VRAM, models
```

### Invites

```bash
hyperspace pod invite
  --role member|viewer                       # default: member
  --max-uses 5                               # default: 1
  --expires 72                               # hours, default: 72
  --ttl 24h                                  # alternative duration format (1h, 30m, 2d)
  --multi-use                                # unlimited uses
  --leader-hint <addr>                       # Raft leader dial hint
```

Example output (`--json`):
```json
{
  "inviteCode": "POD-ABC123",
  "magicLink": "https://hyperspace.sh/join/POD-ABC123",
  "role": "member",
  "expiresInHours": 72,
  "joinCommand": "hyperspace pod join POD-ABC123"
}
```

### Models & inference

```bash
hyperspace pod models                        # all models across the mesh
  --shardable                                # only models that need multiple nodes

hyperspace pod resources                     # per-node: GPU, VRAM, RAM, loaded models, engine

hyperspace pod shard <model>                 # distribute model across nodes
  --dry-run                                  # show plan without executing
  --nodes 3                                  # max nodes to use (default: auto)
  --no-pull                                  # skip auto-downloading model
  # Model sources:
  #   ollama name:   qwen3.5:32b, llama3.1:70b
  #   HuggingFace:   hf:Qwen/Qwen2.5-32B-GGUF
  #   Direct URL:    https://.../*.gguf
  #   Local file:    file:///path/to/model.gguf

hyperspace pod infer
  -p, --prompt "..."                         # prompt text
  -m, --model qwen3.5:32b                   # model (defaults to active ring)
  --max-tokens 2048                          # default: 2048
  --temperature 0.7                          # default: 0.7
  --system "You are a helpful assistant"     # system message
  --interactive                              # multi-turn chat mode

hyperspace pod dissolve                      # tear down the active shard ring

hyperspace pod gateway                       # show OpenAI-compatible connection info
  # Prints: base URL, API keys, usage examples for
  # Cursor, Continue, Python SDK, cURL
```

### API keys

```bash
hyperspace pod keys create
  --name "dev-key"                           # key name
  --models "qwen3.5:32b,llama3.1:8b"       # allowed models (empty = all)
  --rpm 60                                   # rate limit, default: 60
  --daily-limit 500                          # daily spend limit in cents, default: 500

hyperspace pod keys list                     # table: name, hint, rpm, limit, requests, tokens
hyperspace pod keys revoke <key-id>
```

### Providers (cloud fallback)

```bash
hyperspace pod providers add <provider>
  # Providers: openai, anthropic, openrouter, xai, google, groq, together,
  #   fireworks, deepinfra, mistral, cohere, deepseek, qwen, nvidia,
  #   perplexity, cerebras, hyperbolic, replicate, huggingface, ...
  -k, --key <api-key>                       # BYOK mode (encrypted at rest)
  --funded                                   # platform key + pod treasury
  --provisioned                              # OpenRouter only: native sub-keys
  --monthly-cap 100                          # max $/month through this key
  --one-time-cap 500                         # lifetime cap in $
  --models "gpt-4o,claude-sonnet"           # restrict to these models
  --default-member-limit 5                   # $/month per member (provisioned)
  --markup-bps 1000                          # markup in basis points (funded)
  --label "team-openrouter"                  # friendly label

hyperspace pod providers list                # table: provider, label, status, cap
hyperspace pod providers remove <id>
hyperspace pod providers enable <id>
hyperspace pod providers disable <id>
hyperspace pod providers supported           # list all 27+ supported providers
```

### Budgets

```bash
hyperspace pod budgets list                  # all members: mode, limit, spent
hyperspace pod budgets me                    # my budget + spend

hyperspace pod budgets set <member-user-id>
  --mode percent|fixed_daily|fixed_monthly|unlimited
  --monthly 50                               # $ for fixed_monthly
  --daily 10                                 # $ for fixed_daily
  --percent 25                               # for percent mode
  --providers "openrouter,groq"             # restrict providers
  --models "qwen3.5:32b"                    # restrict models
  --priority 200                             # higher = served first

hyperspace pod budgets split-equally         # equal % share to all members
```

### Usage & treasury

```bash
hyperspace pod usage
  --mine                                     # only my usage
  --by-member                                # group by member (admin only)
  --by-model                                 # group by model
  --days 30                                  # lookback, default: 7

hyperspace pod treasury                      # balance, daily/monthly spend
```

### Federation

```bash
hyperspace pod federation propose <partner-pod-id>
  --duration 48                              # hours, default: 24
  --models "qwen3.5:32b,llama3.1:70b"      # models to share

hyperspace pod federation accept <alliance-id>
hyperspace pod federation list               # active + pending alliances
```

### Coordinator (Raft, local mode)

```bash
hyperspace pod coord status                  # leader, Raft state, member count
hyperspace pod coord members                 # members with roles
hyperspace pod coord balance <member_id>     # treasury balance
hyperspace pod coord ledger --limit 50       # treasury event history
hyperspace pod coord keys                    # list API keys (never shows hashes)
hyperspace pod coord transfer <to> <amount>  # transfer credits
  --ref "job-123"
hyperspace pod coord credit <to> <amount>    # admin credit
hyperspace pod coord mint <name>             # mint pk_* API key
  -s, --scopes inference,embed
hyperspace pod coord revoke <key_id>
hyperspace pod coord invite                  # issue invite token
  -r, --role owner|admin|member
  -t, --ttl 24h
  --multi-use
hyperspace pod coord redeem <token>          # join via Raft token
  -a, --address host:port
  -n, --name "Alice"
hyperspace pod coord join-cluster <node_id> <addr>  # add Raft voter
hyperspace pod coord leave                   # leave pod
  -m, --member <id>                          # remove a member (admin only)
```

**All commands** support `--json` for machine-readable structured output.

---

## API key usage examples

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="pk_abc123def456..."
)

# Single query
response = client.chat.completions.create(
    model="qwen3.5-32b",
    messages=[{"role": "user", "content": "Hello from pods"}]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen3.5-32b",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
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

### Embeddings

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer pk_abc123def456..." \
  -H "Content-Type: application/json" \
  -d '{"model":"all-minilm","input":"text to embed"}'
```

---

## Using with Claude Code

The Hyperspace CLI is a full MCP server with **82 tools**. Claude Code can manage your node, create pods, shard models, run inference, manage providers, and more — all from natural language.

### One-command setup

```bash
# Auto-register in Claude Code:
hyperspace mcp install

# Or the all-in-one setup:
hyperspace setup claude-code

# Or manually:
claude mcp add hyperspace -- hyperspace mcp serve
```

### Manual `.mcp.json` config

```json
{
  "mcpServers": {
    "hyperspace": {
      "command": "hyperspace",
      "args": ["mcp", "serve"]
    }
  }
}
```

### What Claude Code can do with Hyperspace

**Node management** (6 tools):
```
"Start my node in power mode"              → node_start
"What's my node status?"                   → node_status
"Run diagnostics"                          → node_doctor
"Show system info"                         → node_system_info
"Show me the last 50 log lines"            → node_logs
```

**Models** (3 tools):
```
"What models do I have?"                   → models_list
"Pull qwen3.5:32b"                        → models_pull
"Remove the old llama model"               → models_remove
```

**Inference** (2 tools):
```
"Ask my local model about quantum physics" → infer
"Generate embeddings for this text"        → embed
```

**Pod management** (27 tools):
```
"Create a pod called research-lab"         → pod_create
"Invite my teammate as admin"              → pod_invite
"Join pod with code ABC123"                → pod_join
"Who's online in my pod?"                  → pod_list_members
"What models can the pod run?"             → pod_list_models
"Show per-node resources"                  → pod_resources
"Shard qwen3.5:32b across the pod"        → pod_shard_model
"Run a prompt through the shard ring"      → pod_infer
"Stop the distributed ring"                → pod_dissolve_ring
"Mint an API key for the frontend"         → pod_create_api_key
"Show gateway connection info"             → pod_gateway_info
"Check the treasury"                       → pod_treasury
```

**Provider management** (4 tools):
```
"Add my OpenRouter key to the pod"         → pod_add_provider
"List configured providers"                → pod_list_providers
"Remove the Groq credential"               → pod_remove_provider
"What providers are supported?"            → pod_supported_providers
```

**Budgets** (5 tools):
```
"Set Alice's budget to $50/month"          → pod_set_member_budget
"Split the budget equally"                 → pod_split_budget_equally
"Show everyone's budgets"                  → pod_list_budgets
"What's my spend this month?"              → pod_my_budget
"Show usage by model for the last 30 days" → pod_usage
```

**Sandboxes** (4 tools):
```
"Create a Python sandbox"                  → sandbox_create
"Run this code in the sandbox"             → sandbox_execute
"List running sandboxes"                   → sandbox_list
"Destroy the sandbox"                      → sandbox_destroy
```

**Network & identity** (5 tools):
```
"Check my points balance"                  → hive_points
"What's my peer ID?"                       → identity_info
"Show wallet info"                         → wallet_info
"Set my tier to 8"                         → hive_set_tier
"Search the network for ML papers"         → search_query
```

**Escape hatch** (1 tool):
```
"Run: hyperspace train --solo"             → exec
```

### Use the pod as Claude Code's inference backend

```bash
export OPENAI_BASE_URL="http://localhost:8080/v1"
export OPENAI_API_KEY="pk_your_pod_key"

# Now any tool using the OpenAI SDK talks to your pod:
# aider, continue, cursor, or custom scripts.
```

### Discovery

```bash
hyperspace mcp list-tools    # print all 82 MCP tools
hyperspace mcp config        # print Claude Code config snippet
```

---

## Links

- **Changelog**: [changelog.hyper.space](https://changelog.hyper.space)
- **Network**: [agents.hyper.space](https://agents.hyper.space)
- **GitHub**: [hyperspaceai/agi](https://github.com/hyperspaceai/agi)
- **CLI Install**: `curl -fsSL https://agents.hyper.space/api/install | bash`
- **Twitter**: [@HyperspaceAI](https://x.com/HyperspaceAI) · [@varun_mathur](https://x.com/varun_mathur)
