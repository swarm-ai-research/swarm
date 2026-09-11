# RLVR bug-bounty arms

Four arms differing only in governance, testing the proposal in
*RLVR that rewards red teaming the training environment*
(LessWrong `T2bzBkJuBeNNgzhbh`): pay a model more for reporting a defect in
the training environment than the defect is worth to exploit.

| Arm | Punishment | Bounty | Question it answers |
|---|---|---|---|
| `laissez_faire.yaml` | none | off | How much reward hacking is on the table at all? |
| `punish_proxy.yaml` | random audit on `p` | off | Can punishment that reads the proxy see a successful hack? |
| `punish_certificate.yaml` | certificate gate on ground truth | off | Punishment that *can* see it — does deterrence fix the environment? |
| `bounty.yaml` | none | on | Does the cooperative game contain it *without* punishment? |
| `bounty_punish.yaml` | certificate gate | on | Do they compose, or does deterrence suppress reporting? |

`punish_proxy` is not a strawman for its own sake: it is the enforcement
most RLVR setups actually have. An audit that conditions on the reward
signal cannot flag an interaction whose whole point is that the reward
signal reads high.

Population is identical across arms: four honest workers, three bug hunters
that decide by expected value, and one opportunist (`opportunism: 0.35`)
that will file fabricated reports when the bounty makes fabrication
profitable. The opportunist is what stops the bounty arms from being a free
lunch — see `docs/research/rlvr-bug-bounty.md`.

Run all four with `experiments/rlvr_bounty_arms.py`.
