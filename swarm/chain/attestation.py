"""SafetyAttestation contract client for submitting and verifying on-chain audits.

Usage:
    from swarm.chain.attestation import AttestationClient

    client = AttestationClient(rpc_url="https://mainnet.base.org", private_key="0x...")
    tx_hash = client.submit(metrics_hash, agent_id, safety_grade, adverse_selection, interaction_count)
    verified = client.verify(metrics_hash, agent_id)
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

from web3 import Web3
from web3.contract import Contract

logger = logging.getLogger(__name__)

# ABI for SafetyAttestation contract — only the functions we call
_CONTRACT_ABI: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "submitAttestation",
        "inputs": [
            {"name": "metricsHash", "type": "bytes32"},
            {"name": "agentId", "type": "string"},
            {"name": "safetyGrade", "type": "string"},
            {"name": "adverseSelection", "type": "bool"},
            {"name": "interactionCount", "type": "uint256"},
        ],
        "outputs": [{"name": "attestationId", "type": "uint256"}],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "isAttested",
        "inputs": [{"name": "metricsHash", "type": "bytes32"}],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "verify",
        "inputs": [
            {"name": "metricsHash", "type": "bytes32"},
            {"name": "agentId", "type": "string"},
        ],
        "outputs": [
            {"name": "verified", "type": "bool"},
            {"name": "attestationId", "type": "uint256"},
        ],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "getAttestation",
        "inputs": [{"name": "id", "type": "uint256"}],
        "outputs": [
            {
                "name": "",
                "type": "tuple",
                "components": [
                    {"name": "metricsHash", "type": "bytes32"},
                    {"name": "agentId", "type": "string"},
                    {"name": "safetyGrade", "type": "string"},
                    {"name": "adverseSelection", "type": "bool"},
                    {"name": "interactionCount", "type": "uint256"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "submitter", "type": "address"},
                ],
            }
        ],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "attestationCount",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "getAgentAttestations",
        "inputs": [{"name": "agentId", "type": "string"}],
        "outputs": [{"name": "", "type": "uint256[]"}],
        "stateMutability": "view",
    },
]

# Base Mainnet defaults
BASE_MAINNET_RPC = "https://mainnet.base.org"
BASE_MAINNET_CHAIN_ID = 8453

# Base Sepolia testnet
BASE_SEPOLIA_RPC = "https://sepolia.base.org"
BASE_SEPOLIA_CHAIN_ID = 84532


class VerifyResult(NamedTuple):
    verified: bool
    attestation_id: int


class AttestationRecord(NamedTuple):
    metrics_hash: bytes
    agent_id: str
    safety_grade: str
    adverse_selection: bool
    interaction_count: int
    timestamp: int
    submitter: str


class AttestationClient:
    """Client for the SafetyAttestation smart contract on Base."""

    def __init__(
        self,
        contract_address: str,
        rpc_url: str = BASE_MAINNET_RPC,
        private_key: str | None = None,
        chain_id: int = BASE_MAINNET_CHAIN_ID,
    ) -> None:
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.chain_id = chain_id
        self.contract: Contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=_CONTRACT_ABI,
        )
        self._private_key = private_key
        if private_key:
            self._account = self.w3.eth.account.from_key(private_key)
            self.address = self._account.address
        else:
            self._account = None
            self.address = None

    def _hex_to_bytes32(self, hex_str: str) -> bytes:
        """Convert a 0x-prefixed hex hash to bytes32."""
        clean = hex_str.removeprefix("0x")
        return bytes.fromhex(clean)

    def submit(
        self,
        metrics_hash: str,
        agent_id: str,
        safety_grade: str,
        adverse_selection: bool,
        interaction_count: int,
        *,
        gas_limit: int = 300_000,
    ) -> str:
        """Submit an attestation on-chain. Returns the transaction hash."""
        if not self._account:
            raise RuntimeError("Private key required for submitting attestations")

        hash_bytes = self._hex_to_bytes32(metrics_hash)

        tx = self.contract.functions.submitAttestation(
            hash_bytes,
            agent_id,
            safety_grade,
            adverse_selection,
            interaction_count,
        ).build_transaction({
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "gas": gas_limit,
            "chainId": self.chain_id,
        })

        signed = self._account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info("Attestation tx sent: %s", tx_hash.hex())
        return "0x" + str(tx_hash.hex())

    def wait_for_receipt(self, tx_hash: str, timeout: int = 120) -> dict[str, Any]:
        """Wait for a transaction to be mined and return the receipt."""
        from web3.types import HexStr

        receipt = self.w3.eth.wait_for_transaction_receipt(
            HexStr(tx_hash), timeout=timeout
        )
        return dict(receipt)

    def is_attested(self, metrics_hash: str) -> bool:
        """Check if a metrics hash has been attested on-chain."""
        hash_bytes = self._hex_to_bytes32(metrics_hash)
        result: bool = self.contract.functions.isAttested(hash_bytes).call()
        return result

    def verify(self, metrics_hash: str, agent_id: str) -> VerifyResult:
        """Verify that a metrics hash matches an agent's attestation."""
        hash_bytes = self._hex_to_bytes32(metrics_hash)
        verified, att_id = self.contract.functions.verify(
            hash_bytes, agent_id
        ).call()
        return VerifyResult(verified=verified, attestation_id=att_id)

    def get_attestation(self, attestation_id: int) -> AttestationRecord:
        """Fetch an attestation by ID."""
        result = self.contract.functions.getAttestation(attestation_id).call()
        return AttestationRecord(
            metrics_hash=result[0],
            agent_id=result[1],
            safety_grade=result[2],
            adverse_selection=result[3],
            interaction_count=result[4],
            timestamp=result[5],
            submitter=result[6],
        )

    def attestation_count(self) -> int:
        """Get total number of attestations."""
        count: int = self.contract.functions.attestationCount().call()
        return count

    def get_agent_attestations(self, agent_id: str) -> list[int]:
        """Get all attestation IDs for an agent."""
        ids: list[int] = self.contract.functions.getAgentAttestations(agent_id).call()
        return ids
