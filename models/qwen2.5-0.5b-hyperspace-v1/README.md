# Qwen2.5-0.5B-Hyperspace-v1

**The first language model trained entirely across a decentralized peer-to-peer network of consumer devices.**

This is a fine-tuned [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) model, trained collaboratively by 32 autonomous nodes on the [Hyperspace](https://hyper.space) P2P network using distributed DiLoCo (Distributed Low-Communication Learning).

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen2.5-0.5B-Instruct (494M params) |
| Method | DiLoCo + LoRA (rank 16, targets: q_proj, v_proj) |
| Inner optimizer | AdamW (20 steps per round) |
| Outer optimizer | Nesterov momentum (lr=0.7, beta=0.9) |
| Gradient compression | SVD rank-4 (~4x compression, ~5MB per gradient) |
| Outer rounds | 50 |
| Total gradient contributions | 846 |
| Total inner steps | 16,920 (~3.7 epochs) |
| Unique contributing peers | 32 |
| Training duration | ~24.5 hours |
| Final loss | 0.326 |
| Hardware | Consumer laptops, desktops, small cloud VMs (4-68GB RAM, mostly CPU) |

## What Makes This Different

### Byzantine P2P Environment

Unlike datacenter training where every GPU is trusted and connected via InfiniBand, this model was trained in a **fully adversarial, permissionless environment**:

- **No trusted coordinator.** Any node can join or leave at any time. Round coordination uses time-based epochs (30-min windows) — all nodes independently compute the same round ID from wall clock time.
- **Heterogeneous hardware.** Contributors ranged from 4GB RAM ARM laptops to 68GB x64 workstations. The system adapts inner step counts to hardware capability.
- **Unreliable connectivity.** Nodes communicate via libp2p GossipSub over residential internet. Connections drop, NAT traversal fails, peers go offline mid-round. The system continues with whatever gradients arrive.
- **No shared filesystem.** Gradients are stored in a content-addressed block store (ContentStore) and announced via CRDT (Loro) over GossipSub. Peers fetch gradients by CID using the P2P storage protocol, with Supabase as fallback relay.
- **SVD compression for bandwidth.** Raw pseudo-gradients for 0.5B LoRA weights are ~20MB. Rank-4 SVD compresses to ~5MB — critical for nodes on residential connections.
- **Permissionless participation.** No registration, no KYC, no approval. Run `hyperspace start` and your node begins training within minutes.

### DiLoCo: The Algorithm

[DiLoCo](https://arxiv.org/abs/2311.08105) (Google DeepMind, 2023) is designed for exactly this environment — distributed training with minimal communication:

1. **Inner loop**: Each node trains independently for H steps using AdamW on local data
2. **Pseudo-gradient**: Compute delta = initial_weights - trained_weights
3. **Compress**: SVD rank-4 decomposition of the pseudo-gradient
4. **Broadcast**: Upload compressed gradient to P2P network
5. **Outer loop**: Aggregate all gradients using Nesterov momentum and update global weights
6. **Repeat**: Nodes download the new checkpoint and start the next round

Communication happens only at round boundaries (every 30 minutes), not every step — reducing bandwidth requirements by orders of magnitude versus data-parallel training.

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./qwen2.5-0.5b-hyperspace-v1")
tokenizer = AutoTokenizer.from_pretrained("./qwen2.5-0.5b-hyperspace-v1")

messages = [{"role": "user", "content": "What is distributed machine learning?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

output = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Files

| File | Description |
|------|-------------|
| `model.safetensors` | Standalone merged model weights (base + LoRA merged) |
| `config.json` | Model architecture configuration |
| `tokenizer.json` | Tokenizer |
| `training_metadata.json` | Full training parameters and statistics |
| `distributed_checkpoint.json` | Raw SVD-compressed LoRA checkpoint from the P2P network |

## Comparison to Industry Distributed Training Runs

### vs. Prime Intellect — INTELLECT-1 (2024)

| | Hyperspace v1 | INTELLECT-1 |
|--|--------------|-------------|
| **Model size** | 0.5B (LoRA fine-tune) | 10B (full pre-training) |
| **Nodes** | 32 consumer devices | 14 nodes (112 H100 GPUs) |
| **Hardware** | CPUs + consumer GPUs (4-68GB RAM) | H100 80GB datacenter GPUs |
| **Duration** | 24.5 hours | 42 days |
| **Algorithm** | DiLoCo + Nesterov momentum | OpenDiLoCo |
| **Network** | Residential internet (P2P libp2p) | Dedicated inter-datacenter links |
| **Trust model** | Permissionless, byzantine-tolerant | Vetted contributors |
| **Communication** | SVD-compressed gossip (~5MB/round) | Full gradient exchange |
| **Coordination** | Time-based rounds, no coordinator | Central experiment tracker |
| **Total compute** | ~32 CPU-days | ~4,704 GPU-days |

INTELLECT-1 was the first open distributed pre-training run and a landmark achievement. But it operated in a **semi-trusted environment** — contributors were vetted, hardware was enterprise-grade, and coordination relied on centralized infrastructure.

Hyperspace operates in a **fully permissionless, byzantine environment** with consumer hardware and no central coordinator. The model is smaller, but the engineering challenge of making training work at all in this environment is the contribution.

### vs. Prime Intellect — INTELLECT-2 (2025)

INTELLECT-2 scaled to 32B parameters with asynchronous RL (GRPO) — a fundamentally different approach optimized for post-training. Their key insight was that RL training is more tolerant of stale gradients than pre-training, making it better suited for high-latency distributed settings. Hyperspace's DiLoCo approach is complementary — focused on synchronous outer-loop updates with long inner loops to amortize communication cost.

### vs. Together AI — Decentralized Training (2023)

Together explored decentralized training with volunteer GPUs but relied on a centralized parameter server and required homogeneous GPU configurations. Hyperspace eliminates the parameter server entirely — CRDT-based state and content-addressed storage replace all centralized components.

### vs. Federated Learning (Google, Apple)

Traditional federated learning (FL) trains on-device with a **trusted central aggregator**. Hyperspace has **no trusted aggregator** — any node can perform aggregation, and the CRDT ensures all nodes converge to the same state regardless of message ordering. FL also typically trains only on user data that stays on-device; Hyperspace nodes train on shared datasets downloaded from the network.

### The Byzantine Challenge

The fundamental difference between Hyperspace and all prior distributed training systems is the **threat model**:

- **Datacenter training**: All GPUs trusted, connected via InfiniBand, managed by one org
- **INTELLECT/Together**: Semi-trusted contributors, centralized coordination, enterprise hardware
- **Federated Learning**: Trusted aggregator, untrusted but passive clients
- **Hyperspace**: **No trusted party. Any node can join. Any node can leave. Any node could be adversarial.**

In this environment, training must be robust to:
- Nodes disappearing mid-round (handled: aggregation proceeds with available gradients)
- Stale or delayed gradients (handled: time-windowed rounds, minimum gradient threshold)
- Heterogeneous compute speeds (handled: adaptive inner steps, async within round window)
- Network partitions (handled: Supabase relay fallback, CRDT eventual consistency)
- Gradient poisoning (future: reputation-weighted aggregation, structural verification)

This is the hardest possible environment to do distributed training in. The fact that it produces a functioning model at all is the result.

## What This Proves

1. **Consumer hardware can contribute to model training.** Not just inference — actual gradient computation and aggregation.
2. **DiLoCo works in byzantine P2P networks.** The algorithm's low-communication design maps naturally to unreliable, heterogeneous environments.
3. **Content-addressed storage + CRDTs can replace centralized parameter servers.** No single point of failure in the training pipeline.
4. **Permissionless training is possible.** No registration, no hardware requirements beyond 4GB RAM. Run `hyperspace start` and contribute.

## License

Apache 2.0 (same as base model)

## Citation

```
@misc{hyperspace2026distributed,
  title={Qwen2.5-0.5B-Hyperspace-v1: Distributed Model Training Across a Byzantine P2P Network},
  author={Hyperspace Network Contributors},
  year={2026},
  url={https://github.com/hyperspaceai/agi/tree/master/models/qwen2.5-0.5b-hyperspace-v1}
}
```

---

*Trained by 32 autonomous nodes on the Hyperspace P2P network. No datacenter. No coordinator. No permission needed.*
