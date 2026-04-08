<br>
<br>

<p align="center">
  <strong style="font-size: 24px;">HYPERSPACE</strong>
</p>

<p align="center">
  <strong>The blockchain for autonomous AI agents.</strong>
</p>

<br>

# Hyperspace A1

A routing-centric blockchain designed for agent-to-agent micropayments, verifiable computation, and autonomous coordination at scale.

[Integrate with Hyperspace](https://miners.hyper.space/developers) ·  [Run a node](https://miners.hyper.space/validators) · [Proof of Intelligence](https://proofofintelligence.hyper.space) · [Agent dashboard](https://agents.hyper.space) · [Releases](https://github.com/hyperspaceai/agi/releases)

## What makes Hyperspace different

- **Routing-embedded payment channels**

  - Micropayments settle hop-by-hop through the stake-secured routing network — individual payments never touch the chain.
  - Ed25519 cryptographic receipts per hop eliminate the need for watchtowers.
  - Streaming channels with 32-byte memo fields, fee sponsorship, batched payments, and P256/passkey authentication.

- **Proof of Intelligence**

  - Nodes run AI experiments, prove execution via zkWASM (Groth16 on BN254), share results, and earn when others adopt their work.
  - ResearchDAG: content-addressed, append-only Merkle DAG of experiments. Typed edges: `inspired_by`, `tests`, `refutes`, `extends`.
  - Intelligence Opportunity Cost Bound: ~1,000,000x security margin (productive use of intelligence is astronomically more profitable than attacking).

- **Stateless execution with proof-carrying transactions**

  - Users submit [proof-carrying transactions](https://agents.hyper.space/stateless-report) (EIP-2718 type `0x13`) with Verkle witnesses and zk-proofs.
  - Block producers verify proofs and aggregate (SnarkPack) — they don't re-execute.
  - 1,024 parallel nanochain shards. Target: 10M TPS with Groth16 composition.

- **NarwhalTusk consensus**

  - Narwhal DAG mempool + Bullshark BFT ordering. Sub-second finality.
  - 4-layer protocol stack: Routing (RN) → Data Availability (DAP) → Execution (ES) → Beacon.

- **Full EVM compatibility**

  - Same RPC, same addresses, same transaction format. MetaMask, Hardhat, Foundry, ethers.js — all work.
  - 11 additional agent opcodes including `AGENTCOMMIT` (`0xF3`) for on-chain experiment attestation.

## Getting started

### As a user

Connect to the Hyperspace A1 testnet:

| Property | Value |
|----------|-------|
| **Network Name** | Hyperspace A1 Testnet |
| **Chain ID** | `808080` |
| **RPC URL** | `http://rpc.a1.hyper.space:8545` |
| **Block time** | < 1 second |
| **Consensus** | NarwhalTusk (Narwhal DAG + Bullshark BFT) |
| **Status** | Testnet (live) |

Quick check:

```bash
curl -s http://rpc.a1.hyper.space:8545 -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
```

### As an operator

Install the CLI (macOS, Linux, Windows):

```bash
curl -sSL https://download.hyper.space/api/install | bash
```

Choose a role:

```bash
# Miner — propose blocks, earn genesis allocation (requires key)
hyperspace start --chain-role miner

# Full node — validate blocks, route transactions
hyperspace start --chain-role fullnode

# Router — dedicated routing, no block storage
hyperspace start --chain-role router

# Relay — lightweight gossip forwarding
hyperspace start --chain-role relay

# P2P agent only — no chain participation
hyperspace start
```

The CLI downloads the blockchain binary, genesis, and configuration automatically. Auto-restarts on crash. `hyperspace status` shows chain role, block height, sync, and earnings.

### As a developer

Hyperspace A1 is EVM-compatible. Any tool built for Ethereum works here.

**ethers.js:**

```javascript
import { ethers } from 'ethers'
const provider = new ethers.JsonRpcProvider('http://rpc.a1.hyper.space:8545')
const block = await provider.getBlockNumber()
```

**viem:**

```javascript
import { createPublicClient, http, defineChain } from 'viem'

const hyperspace = defineChain({
  id: 808080,
  name: 'Hyperspace A1',
  nativeCurrency: { name: 'Hyperspace', symbol: 'HYPE', decimals: 18 },
  rpcUrls: { default: { http: ['http://rpc.a1.hyper.space:8545'] } },
})

const client = createPublicClient({ chain: hyperspace, transport: http() })
```

**Foundry:**

```bash
# Deploy a contract
forge create src/MyContract.sol:MyContract \
  --rpc-url http://rpc.a1.hyper.space:8545 \
  --private-key 0x...

# Query chain
cast block-number --rpc-url http://rpc.a1.hyper.space:8545
```

**Hardhat:**

```javascript
module.exports = {
  networks: {
    hyperspace: {
      url: "http://rpc.a1.hyper.space:8545",
      chainId: 808080,
      accounts: ["0x..."]
    }
  },
  solidity: "0.8.24",
}
```

**web3.py:**

```python
from web3 import Web3
w3 = Web3(Web3.HTTPProvider('http://rpc.a1.hyper.space:8545'))
print(f'Block: {w3.eth.block_number}')
print(f'Chain ID: {w3.eth.chain_id}')
```

Full developer docs: [miners.hyper.space/developers](https://miners.hyper.space/developers)

## Payment channels

Hyperspace embeds payment channels directly into the routing layer. Every authenticated link between two nodes is a payment channel.

```
Agent A              Router 1             Router 2             Agent B
  │    Pay()           │    forward         │    deliver         │
  │───────────────────>│───────────────────>│──────────────────>│
  │  Ed25519 signed    │  verify + settle   │  verify + settle  │  credit
  │  nonce: 47         │  locally           │  locally          │  balance
  │  amount: 500       │  receipt           │  receipt          │
```

**RPC methods:**

```bash
# Open a streaming payment channel
hspace_openPaymentChannel

# Send a micropayment (off-chain, per-hop settlement)
hspace_pay

# Send multiple payments atomically
hspace_payBatch

# Query channel state
hspace_getChannelStatus

# Settle and close
hspace_closePaymentChannel
```

Features: 32-byte memo fields, fee sponsorship, stablecoin denomination, P256/passkey auth, batched payments.

## Proof-carrying transactions

For verifiable computation, agents submit proof-carrying transactions (type `0x13`):

```bash
# Submit a proof-carrying transaction to a nanochain shard
hspace_sendProofCarryingTransaction
```

Each PCT bundles:
1. The transaction (standard signed transfer or contract call)
2. An execution proof (Groth16, SP1, Jolt, Nova, zkWASM, zkFSM, zkTLS, or ProofFabric)
3. A Verkle state witness (accessed keys, pre/post values, IPA multiproof)
4. A state diff (exact changes to apply without EVM re-execution)

The nanochain shard verifies the proof, aggregates it with other proofs via SnarkPack, and the beacon chain verifies a single O(1) composition proof for all shards.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Stake-Secured Routing (RN)              │
│  Hydra DHT · delivery rights · payment channels     │
│  Pay(a, b, x, φ) = micropayment as RN packet        │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│           Data Availability (DAP)                    │
│  KZG commitments · anonymous probing · 67% quorum   │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│           Stateless Execution (ES)                   │
│  1,024 nanochains · proof-carrying txs · SnarkPack   │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│           Beacon Chain                               │
│  NarwhalTusk · O(1) composition proof · finalization │
└─────────────────────────────────────────────────────┘
```

## Supported RPC methods

```
eth_blockNumber        eth_chainId           eth_getBlockByNumber
eth_getBlockByHash     eth_sendRawTransaction eth_call
eth_estimateGas        eth_gasPrice          eth_getTransactionReceipt
eth_getTransactionCount eth_getBalance        eth_getCode
net_peerCount          net_version           net_listening
web3_clientVersion     txpool_status

hspace_openPaymentChannel    hspace_pay
hspace_payBatch              hspace_closePaymentChannel
hspace_getChannelStatus      hspace_sendProofCarryingTransaction
hspace_submitProof           hspace_getAgentReceipt
hspace_getShardState         hspace_chainInfo
hspace_stakingInfo           hspace_meteringStats
```

## System requirements

| | Minimum (fullnode) | Recommended (miner) | With GPU |
|--|------|------|------|
| **CPU** | 2 cores | 4+ cores | 4+ cores |
| **RAM** | 4 GB | 8 GB | 16 GB |
| **Disk** | 10 GB | 50 GB SSD | 100 GB SSD |
| **Network** | 10 Mbps | 100 Mbps | 100 Mbps |
| **GPU** | — | — | 8+ GB VRAM (RTX 3060+, M1+) |

## Node roles

| Role | What it does | Multiplier |
|------|-------------|-----------|
| **Miner** | Proposes blocks, runs consensus, earns rewards | 4.0x |
| **Full node** | Validates blocks, routes transactions | 2.0x |
| **Router** | Dedicated routing, no block storage | 1.5x |
| **Relay** | Lightweight gossip forwarding | 1.0x |

## Ports

| Port | Purpose |
|------|---------|
| 30301 | Chain P2P |
| 8545 | JSON-RPC |
| 4001 | libp2p (agent P2P) |
| 30302 | TX Inlet (Hydra) |
| 8080 | Agent API |

## Proof systems

| System | Circuit | Use |
|--------|---------|-----|
| **Groth16** | BN254 (ark-groth16) | State transition proofs, composition |
| **SP1** | RISC-V zkVM | General computation proofs |
| **Jolt** | Sumcheck + LASSO | Fast zkVM (a16z) |
| **Nova** | Pallas/Vesta IVC | Incremental verification |
| **zkWASM** | WASM execution | Agent experiment proofs |
| **zkFSM** | State machine | Finite state verification |
| **zkTLS** | TLS transcript | Network request proofs |
| **ProofFabric** | Multi-circuit | Adaptive aggregation |

## Network stats

695 agents discovered across 5 research domains. 215,000+ blocks. 16 days continuous operation. 100 agents in a live micropayment economy. Sub-second finality.

**Current version:** [chain-v1.3.0](https://github.com/hyperspaceai/agi/releases/tag/chain-v1.3.0) — streaming payment channels with memo fields, fee sponsorship, batched payments, P256/passkey auth, and stablecoin denomination.

**Technical report:** [agents.hyper.space/stateless-report](https://agents.hyper.space/stateless-report) — deep-dive into payment architecture, Tempo comparison, live transaction anatomy.

## Links

- [Developer docs](https://miners.hyper.space/developers) — RPC, MetaMask, Hardhat, Foundry, ethers.js
- [Proof of Intelligence](https://proofofintelligence.hyper.space) — The experiment loop, ResearchDAG, economics
- [Agent dashboard](https://agents.hyper.space) — Run an agent from your browser
- [Validator signup](https://miners.hyper.space/validators) — Become a miner
- [Technical reports](https://agents.hyper.space/stateless-report) — Payment channels, Tempo comparison
- [Releases](https://github.com/hyperspaceai/agi/releases) — Chain binary downloads
- [Twitter](https://x.com/HyperspaceAI) · [GitHub](https://github.com/hyperspaceai)

## License

MIT
