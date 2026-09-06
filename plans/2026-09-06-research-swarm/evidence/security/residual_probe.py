import json
from swarm.agentgit.identity import AgentKeypair, DelegationChain, NonceRegistry, sign_link
issuer, subject = AgentKeypair.generate(), AgentKeypair.generate()
a = sign_link(issuer, subject_did=subject.did, permissions=["read"], nonce="a"*32, audience="task-42")
b = sign_link(issuer, subject_did=subject.did, permissions=["write"], nonce="a"*32, audience="task-42")
r=NonceRegistry()
result = {
 "bound_foreign_context": DelegationChain([a]).verify(context="task-99")[0],
 "first_registered": DelegationChain([a]).verify(context="task-42", nonces=r)[0],
 "identical_replay_registered": DelegationChain([a]).verify(context="task-42", nonces=r)[0],
 "different_payload_reuse_registered": DelegationChain([b]).verify(context="task-42", nonces=r)[0],
 "different_payload_reuse_default": DelegationChain([b]).verify(context="task-42")[0],
 "unbound_default_with_context": DelegationChain([sign_link(issuer, subject_did=subject.did, permissions=["read"])]).verify(context="task-99")[0],
}
print(json.dumps(result, indent=2))
assert result == {"bound_foreign_context":False,"first_registered":True,"identical_replay_registered":True,"different_payload_reuse_registered":False,"different_payload_reuse_default":True,"unbound_default_with_context":True}
