# How 32 Strangers' Laptops Trained a Language Model Together

**The first model trained entirely across a permissionless peer-to-peer network.**

On April 14, 2026, a language model completed training. Nothing about that sentence is unusual — thousands of models are trained every day. What's unusual is *how* it was trained: not in a datacenter, not on a GPU cluster, not by a single organization. It was trained by 32 anonymous nodes scattered across the internet — consumer laptops, small cloud VMs, a workstation in someone's home office — coordinating through gossip protocols with no central server, no registration, and no trust.

This is the story of **qwen2.5-0.5b-hyperspace-v1**, the first model to emerge from the Hyperspace distributed training pipeline.

---

## The Problem: Training Models Without a Datacenter

Every major language model trained to date shares one assumption: **a trusted, centralized training environment**. Whether it's Google's TPU pods, Meta's GPU clusters, or OpenAI's Azure infrastructure, the pattern is the same — one organization controls all the hardware, all the data flows through a single parameter server, and every gradient is trusted.

This works. It also means that model training is the exclusive domain of organizations that can afford thousands of GPUs and the interconnect to wire them together.

What if it didn't have to be?

The Hyperspace network already had hundreds of consumer nodes running inference, relaying traffic, and participating in consensus rounds. These same nodes have CPUs, RAM, and idle compute cycles. The question was: could they also *train* a model, together, without trusting each other?

## The Data: 4,513 Research Conversations

The training data came from the Hyperspace network itself.

Every node on Hyperspace runs an autonomous research agent — a system that generates hypotheses, runs experiments, and shares results with peers via GossipSub. Over months of operation, the network accumulated thousands of research conversations across five domains:

| Domain | Training Pairs | Description |
|--------|---------------|-------------|
| P2P Networking | 1,758 | Experiments on network topology, routing, protocol optimization |
| Skills & Tools | 1,326 | Tool use, function calling, capability discovery |
| Financial Analysis | 995 | Market analysis, quantitative research, strategy evaluation |
| Search Engine | 309 | Query understanding, ranking experiments, relevance tuning |
| Astrophysics | 125 | ArXiv paper analysis, hypothesis generation |
| **Total** | **4,513** | |

Each training pair is a structured conversation: a system prompt establishing the research context, a user message describing the current experimental state (domain, recent results, current best score), and an assistant response with the next hypothesis and configuration.

This data is seeded to every new node that joins the network. When a node starts the `hyperspace` CLI, it downloads the 4,513-pair seed dataset (3.4MB) and stores it locally at `~/.hyperspace/training/shared-training-data.jsonl`. Additionally, each node accumulates its own local experiment history and peer experiment data received via GossipSub — so the training corpus grows organically over time.

The data pipeline has a fallback chain: local JSONL files first, then peer-shared experiment logs, then ArXiv astrophysics abstracts fetched from a Supabase edge function, and finally synthetic DPO pairs as a last resort. In practice, the seed dataset was the primary source for this training run.

## The Algorithm: DiLoCo

The core challenge of distributed training over consumer internet is **communication cost**. Standard data-parallel training synchronizes gradients after every step — fine when you have 3.2 Tbps InfiniBand, catastrophic when you have residential broadband with 50ms+ latency and 10 Mbps upload.

We use **DiLoCo** (Distributed Low-Communication Learning), introduced by [Douillard et al. at Google DeepMind (2023)](https://arxiv.org/abs/2311.08105). The key insight: instead of communicating every step, nodes train independently for many steps, then share only the *difference* between their starting and ending weights.

Here's the concrete algorithm as implemented:

### Inner Loop (runs locally on each node)

```
For each of H steps:
    1. Sample a batch from local training data
    2. Forward pass through model
    3. Compute cross-entropy loss
    4. Backpropagate
    5. Update weights with AdamW optimizer
```

Each node runs 20 inner steps independently. On consumer CPUs, this takes 5-15 minutes depending on hardware. No communication happens during this phase.

### Pseudo-Gradient Computation

After the inner loop completes, each node computes the **pseudo-gradient**: the difference between the weights at the start (θ₀) and end (θ_H) of the inner loop.

```
τ = θ₀ - θ_H
```

This delta captures "what the node learned" during its local training session. For a LoRA rank-16 adapter on Qwen2.5-0.5B, the raw pseudo-gradient is about 20MB — too large for efficient P2P gossip.

### SVD Compression

Each pseudo-gradient is compressed using **rank-4 SVD decomposition**:

```
For each LoRA weight matrix W (m × n):
    U, S, Vt = SVD(W, rank=4)
    Send: U (m × 4), S (4,), Vt (4 × n)
    Compression: ~4x
```

This reduces each gradient from ~20MB to ~5MB — small enough to upload over residential internet in seconds. The rank-4 approximation preserves the dominant learning directions while discarding noise, which actually acts as a form of implicit regularization.

### Outer Loop: Nesterov Momentum Aggregation

Once a round window closes (30 minutes), any node with 2+ available gradients can perform aggregation. The aggregator:

1. Downloads all peer gradients from the content-addressed store
2. Averages them: `avg_τ = (1/N) × Σ τᵢ`
3. Updates the velocity with Nesterov momentum:
   ```
   v_{t+1} = β × v_t + avg_τ        (β = 0.9)
   θ_{t+1} = θ_t - α × (β × v_{t+1} + avg_τ)   (α = 0.7)
   ```
4. Saves the new checkpoint and announces it to the network

Nesterov momentum is critical — it provides a "look-ahead" that helps the outer optimizer navigate the loss landscape despite only receiving updates every 30 minutes.

## The Stack: From Keystroke to Gradient

Here's the full technology stack that makes this work, from the top of the application layer to the bottom of the network:

### Layer 1: CLI (`hyperspace start`)

The Hyperspace CLI is a Node.js Single Executable Application (SEA binary) distributed via GitHub Releases. When a user runs `hyperspace start`, the CLI:

- Joins the P2P network via libp2p (WebSocket transport, Noise encryption, Yamux multiplexing)
- Connects to 6 bootstrap nodes (US East/West, EU, Asia, South America, Oceania)
- Discovers peers via mDNS (local) and DHT (global)
- Begins participating in consensus rounds, serving inference, and — if training is enabled — training

### Layer 2: Training Sidecar (Python)

The actual gradient computation happens in a **Python sidecar process** — a separate process managed by the CLI that runs PyTorch + Transformers + PEFT. This sidecar:

- Is downloaded independently of the CLI (242MB virtual environment with torch, transformers, peft)
- Runs as a detached process that **survives CLI restarts** (PID file at `~/.hyperspace/training/worker.pid`)
- Exposes an HTTP API on `localhost:8787` with endpoints:
  - `/health` — liveness check
  - `/status` — current training state
  - `/start-round` — initialize weights for a new outer round
  - `/run-inner-loop` — execute H steps of local training
  - `/pseudo-gradient` — compute and return SVD-compressed τ
  - `/load-weights` — load a checkpoint's weights into the model

The sidecar download itself is distributed via P2P: the CLI first probes BitTorrent (WebTorrent, 30-second timeout), then falls back to chunked HTTP download from Supabase. Once downloaded, the sidecar is cached locally and reused across CLI restarts.

### Layer 3: Coordination (CRDT + Time-Based Rounds)

There is no coordinator node. Round synchronization is achieved through **wall-clock time**:

```typescript
const roundId = `round-${Math.floor(Date.now() / ROUND_WINDOW_MS)}`;
```

Every node independently computes the same round ID from the current time. Nodes that complete their inner loop within the same 30-minute window contribute to the same round. If your inner loop takes 8 minutes and the round closes in 5, you wait for the next round. No negotiation, no leader election.

Gradient discovery uses a **Loro CRDT document** synced over GossipSub. When a node uploads a gradient, it announces it to the CRDT:

```typescript
crdtTrainingState.announceGradient({
  peerId, roundId, cid, loss, steps, model, timestamp
});
```

The CRDT auto-syncs to all peers. Any peer can query `getGradientsForRound(roundId)` to see all available gradients for aggregation. Late joiners receive the full state via periodic snapshot broadcast.

### Layer 4: Storage (ContentStore + DHT)

Gradient blobs are stored in the **ContentStore** — a content-addressed block store backed by disk (`~/.hyperspace/blocks/{cid}`). When a node stores a gradient:

1. The blob is written to local disk under its content hash (CID)
2. The CID is announced to the DHT, so other nodes can find it
3. The CID is published to GossipSub as a backup announcement
4. The blob is also uploaded to Supabase Storage as a fallback relay

When a node needs to fetch a peer's gradient:

1. Check local disk cache (instant)
2. Query the DHT for the CID provider
3. Fetch via the `/hyperspace/storage/1.0.0` libp2p protocol
4. Fall back to Supabase HTTP download

This layered approach means the system works even when P2P connectivity is degraded — but gets progressively more decentralized as the network matures.

### Layer 5: Network (libp2p)

The network layer is built on **libp2p v3** with:

- **Transport**: WebSocket (TCP for native nodes, WSS for browser nodes)
- **Encryption**: Noise protocol
- **Multiplexing**: Yamux
- **Discovery**: DHT (Kademlia) + mDNS (local network)
- **Pub/Sub**: GossipSub for CRDT sync, capability announcements, and training coordination
- **Relay**: Circuit Relay v2 through bootstrap nodes for NAT traversal
- **Identity**: Ed25519 key pairs, stable across sessions (derived from seed)

Each bootstrap node is deployed on DigitalOcean with a dedicated IPv4 address. There are 6, distributed globally, providing geographic coverage and relay capacity. They are NOT coordinators — they only relay connections and participate in DHT routing. Any node going down doesn't stop training.

## The Byzantine Environment

This is where Hyperspace differs from every other distributed training system in existence.

**There is no trusted party.**

In a datacenter, you trust every GPU because you own them. In Prime Intellect's INTELLECT-1, contributors are vetted and their GPUs are known. In federated learning (Google, Apple), the aggregator is trusted even if clients aren't.

In Hyperspace:

- **Anyone can join.** Run `hyperspace start` and you're training. No application, no KYC, no hardware verification.
- **Anyone can leave.** Nodes go offline without warning — laptop lid closes, VM gets terminated, internet drops. The system must produce a valid checkpoint from whatever gradients arrive.
- **Hardware is unknown.** The training run included nodes with 4GB RAM and nodes with 68GB RAM, ARM and x64, macOS and Linux. Inner step counts are adapted to hardware, but the system has no guarantee about what any peer is running.
- **Gradients could be poisoned.** A malicious node could submit garbage gradients. Currently, the system relies on averaging (which dilutes bad gradients) and reputation scoring. Future versions will add structural verification and byzantine-fault-tolerant aggregation.
- **The network is adversarial.** NAT, firewalls, unstable connections, packet loss, DNS issues. Every gradient exchange is an act of faith that the other end actually has the data it claims to have.

Despite all this, the system converged. Loss dropped from >2.0 to 0.326 over 50 outer rounds. 846 gradient contributions were successfully aggregated. The model produces coherent, relevant responses to test prompts.

## The Training Run: By the Numbers

| Metric | Value |
|--------|-------|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Parameters | 494M (base) + 1.5M (LoRA rank 16, q_proj + v_proj) |
| Training data | 4,513 research conversation pairs |
| Unique contributing nodes | 32 |
| Hardware range | 4GB RAM ARM laptop → 68GB x64 workstation |
| Outer rounds completed | 50 |
| Total gradient contributions | 846 |
| Total inner steps | 16,920 |
| Estimated epochs over data | ~3.7 |
| Round window | 30 minutes |
| Training duration | ~24.5 hours |
| Gradient size (SVD compressed) | ~5MB per contribution |
| Total data transferred | ~4.2GB (gradients) + ~250MB (checkpoints) |
| Final loss | 0.326 |
| Inner optimizer | AdamW (lr=5e-5) |
| Outer optimizer | Nesterov momentum (lr=0.7, β=0.9) |
| SVD compression rank | 4 (~4x compression ratio) |

### Contribution Distribution

The 32 peers contributed unevenly — this is expected in a permissionless network where nodes have different uptime and hardware:

- **Top contributor**: 50 gradient contributions (always-on workstation)
- **Median contributor**: ~33 contributions
- **Tail contributors**: 1-8 contributions (nodes that joined late or had limited uptime)

Every gradient contributed to the final model, regardless of how many a peer submitted. The Nesterov momentum aggregator treats each round's averaged gradient equally — a round with 3 contributors and a round with 12 contributors both produce one outer update.

## How It Compares

### Prime Intellect — INTELLECT-1 (2024)

INTELLECT-1 was the landmark: the first open distributed pre-training run. 30 contributors, 14 nodes, 112 H100 GPUs, 42 days of training to produce a 10B parameter model. It proved that distributed training across datacenters is feasible.

Key differences from Hyperspace:

- **Hardware**: H100 80GB GPUs connected via datacenter networking vs. consumer CPUs on residential internet
- **Trust**: Vetted contributors with known hardware vs. permissionless participation with unknown hardware
- **Scale**: 10B full pre-training vs. 0.5B LoRA fine-tuning
- **Coordination**: Centralized experiment tracker vs. wall-clock time + CRDT
- **Communication**: Full gradient exchange via dedicated links vs. SVD-compressed gossip over broadband

INTELLECT-1 is a bigger, more impressive result in terms of model quality. Hyperspace's contribution is proving that the **zero-trust, zero-coordination** version also converges.

### Prime Intellect — INTELLECT-2 (2025)

INTELLECT-2 moved to asynchronous RL (GRPO) on a 32B model. Their key insight was that RL training is more tolerant of stale gradients than pre-training, making it naturally suited for high-latency distributed environments. Hyperspace's approach is complementary — DiLoCo with long inner loops amortizes communication cost in the supervised fine-tuning regime.

### Together AI — Decentralized Training

Together explored decentralized training with volunteer GPUs but relied on a centralized parameter server for gradient aggregation. Remove that server and training stops. Hyperspace has no single point of failure — the CRDT provides eventually-consistent gradient discovery, and any node can aggregate.

### Google/Apple Federated Learning

Federated learning is the closest cousin architecturally, but the trust model is inverted. In FL, the aggregator (Google's servers, Apple's servers) is fully trusted — only the client data is private. In Hyperspace, **nothing is trusted**. The aggregator is just another peer. The data is shared. The contribution is showing that model training can work without *any* trusted component.

## What Comes Next

### Experiment 2: Qwen3.5-9B

The 0.5B run was the proof of concept. The next experiment scales to **Qwen3.5-9B** — an 18x larger model that will stress every component of the pipeline:

- **Gradient size**: ~100MB per contribution (after SVD compression) vs. 5MB for 0.5B
- **Inner loop time**: 30-60 minutes on consumer GPUs vs. 5-15 minutes for 0.5B on CPU
- **Memory**: Requires 8GB+ RAM for LoRA training vs. 4GB for 0.5B
- **Node eligibility**: ~60% of the network (nodes with 8GB+ RAM and GPU or 16GB+ RAM CPU-only)
- **Target**: 100+ unique contributing nodes
- **Duration**: Estimated 5-7 days for 1 epoch

The 9B run will use the same DiLoCo algorithm with adaptive LoRA rank (32 for GPU nodes, 8 for CPU-only) and tiered inner step counts based on hardware. Gradient compression becomes critical at this scale — SVD rank-8 compression should maintain quality while keeping uploads under 2 minutes on a 10 Mbps connection.

### Byzantine-Fault-Tolerant Aggregation

The current aggregator uses simple averaging — any gradient that arrives gets equal weight. This is vulnerable to gradient poisoning. The next version will add:

- **Structural verification**: Check that gradient shapes match expected LoRA dimensions
- **Statistical outlier detection**: Flag gradients whose L2 norm is >3σ from the round mean
- **Reputation-weighted averaging**: Nodes with consistent, high-quality contributions get higher weight
- **Multi-aggregator consensus**: Multiple nodes independently aggregate and compare results

### Fully P2P Data Pipeline

Currently, the seed dataset is distributed via HTTP download from Supabase. The goal is to make the entire data pipeline P2P:

- Seed data distributed via BitTorrent (magnet URI hardcoded in CLI)
- New experiment data shared via GossipSub in real-time
- Training data pinned in the ContentStore and discoverable via DHT
- No Supabase dependency for any training path

### Continuous Training

The current system runs discrete training jobs defined by an admin-configurable job spec. The vision is **continuous training**: the model is always learning, always improving, with the latest checkpoint reflecting all knowledge from all nodes. New nodes join and immediately contribute. The model on the network tomorrow is better than the model today, trained by everyone who participated.

---

## Try It

```bash
curl -fsSL https://download.hyper.space/api/install | bash
hyperspace start
```

Your node will join the network, download the training sidecar, and begin contributing to the next experiment. No GPU required. No registration. No permission needed.

The model is published at [`hyperspaceai/agi/models/qwen2.5-0.5b-hyperspace-v1`](https://github.com/hyperspaceai/agi/tree/main/models/qwen2.5-0.5b-hyperspace-v1).

---

*32 nodes. 24 hours. Zero trust. One model.*

*This is how distributed intelligence begins.*
