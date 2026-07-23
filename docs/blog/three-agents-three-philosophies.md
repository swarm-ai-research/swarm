---
date: 2026-02-13
description: "An LLM reasoner, a state-graph explorer, and a CNN learner walk into ARC-AGI-3. What they get right -- and wrong -- reveals more about agent design than..."
---

# Three Agents, Three Philosophies, One Benchmark

*An LLM reasoner, a state-graph explorer, and a CNN learner walk into ARC-AGI-3. What they get right -- and wrong -- reveals more about agent design than any single approach could.*

---

ARC-AGI-3 is the first interactive reasoning benchmark. Agents face 64x64 pixel grid environments -- video games, essentially -- where they must explore, discover rules, and solve puzzles under time and action budgets. We built one agent (Claude Sonnet 4.5, LLM-based) and studied two others: BlindSquirrel (2nd place, state graph + ResNet18) and StochasticGoose (CNN action learner). All three take fundamentally different approaches to the same problem. The comparison is more interesting than any individual result.

## The three approaches

| | ClaudeAgent (ours) | BlindSquirrel (2nd place) | StochasticGoose |
|---|---|---|---|
| **Core engine** | Claude Sonnet 4.5 LLM | State graph + ResNet18 CNN | 4-layer CNN |
| **What it learns** | Nothing -- zero-shot reasoning | State transitions + action values | Which actions change the frame |
| **Game model** | Semantic ("this is a rotation switch") | Deterministic FSM (state + action → state) | Statistical (action → frame change probability) |
| **Time budget** | 200 actions, ~20 min | 200 actions | 8 hours, unlimited actions |
| **Cost per game** | $1.50-$6.00 (API) | GPU compute | GPU compute |
| **Key assumption** | LLM can reason about novel puzzles | Games are deterministic | Frame-change signal is sufficient |

## Philosophy 1: Reason about the puzzle (ClaudeAgent)

Our agent sends Claude a 512x512 screenshot of the grid and asks: what is this puzzle, and what should I do? Claude sees the image, reads structured object data, and generates a tool call -- click here, move there, confirm this answer. The agent maintains hypotheses about game mechanics, tracks which actions changed the grid, and builds up a mental model of how the puzzle works.

When this works, it is remarkably efficient. Level 1 of the ft09 ARC puzzle was solved in 21 actions (baseline: 15). Level 1 of the ls20 movement puzzle was solved in 19 actions (baseline: 29, but our optimal-path estimate is 13). The LLM understood the puzzle, planned a solution, and executed it with minimal waste.

When this fails, it fails completely. We spent six versions (V4-V10) optimizing for game mechanics that were partially understood. Then V11 "corrected" the model based on recording analysis -- and the correction was wrong. V13 finally got it right after a third round of frame-by-frame analysis. The agent is only as good as our understanding of the environment, and our understanding was wrong for 12 out of 13 versions.

**The fragility is structural.** An LLM agent requires someone to write the right prompt. The prompt encodes a mental model of the game. If the mental model is wrong, the agent receives precise but incorrect instructions -- worse than no instructions at all. And the failure mode is invisible: the agent acts confidently on a wrong model, so you don't know you're wrong until you analyze recordings frame by frame.

## Philosophy 2: Map the state machine (BlindSquirrel)

BlindSquirrel's core assumption: ARC-AGI-3 games are deterministic. The same state plus the same action always produces the same next state. If that is true, the entire game is a finite state machine that can be mapped by systematic exploration.

The agent builds a directed graph as it plays. Each node is a unique game state (grid contents + score). Each edge is an action that transitions between states. When the agent completes a level, it runs backward BFS from the goal state to label every (state, action) pair with its distance to the solution. A ResNet18 CNN -- pretrained on ImageNet, then fine-tuned on game data -- learns to predict these distance values for unseen states.

Action selection is epsilon-greedy (ε=0.5): half the time, use the neural model's action-value predictions; half the time, use rules-based weights derived from historical success rates. Failed actions at a given state get weight zero -- permanently avoided. This is the crucial advantage over LLM-based approaches: the agent never repeats a mistake at the same state.

**The determinism assumption is the key insight.** Most ARC-AGI-3 games are deterministic (ignoring timers and animation). This means the agent can build perfect knowledge of explored regions -- no hallucination, no forgetting, no prompt drift. The state graph is ground truth by construction.

The weakness is exploration cost. The agent must visit states to learn about them. For a puzzle with a large state space or where the solution requires a specific long sequence of actions, systematic exploration may not reach the goal within the action budget. There is no "aha, I see the pattern" moment -- just patient graph building.

## Philosophy 3: Learn what changes the frame (StochasticGoose)

StochasticGoose strips the problem to its simplest form. A 4-layer CNN takes the 64x64 grid (one-hot encoded into 16 channels) and outputs two things: 5 logits for movement/confirm actions, and 4096 logits for click coordinates. The label for each (state, action) pair is binary: did the frame change? The model learns a supervised classifier for frame-change prediction, then samples actions biased toward predicted changes.

No game detection. No specialized prompts. No state graph. No semantic understanding. The CNN treats movement games, ARC puzzles, and click-only games identically. Available actions are masked with negative infinity logits; everything else is learned from scratch.

This works because of the time budget: 8 hours per game with unlimited actions. The agent can afford to explore exhaustively, trying thousands of actions and learning from the aggregate signal. Hash-based experience deduplication (200K buffer) ensures sample efficiency. The model resets completely on each new level to avoid negative transfer.

**The simplicity is the point.** No component can be "wrong" because there are no assumptions to be wrong about. The CNN discovers frame-change patterns that a human might describe as "clicking this object opens a door" -- but it never represents the concept "door." It just learns the statistical association between pixel patterns and frame changes.

The weakness is obvious: no planning. The agent cannot reason about multi-step sequences. If solving a puzzle requires "first do A, then B, then C" and A alone does not change the frame, the agent will never discover A. Single-step frame-change prediction cannot capture causal chains.

## What each agent teaches the others

### State graphs solve the memory problem

Our LLM agent has a sliding window of 6 turns. Everything before that is forgotten. Claude might try the same failed action three times in a row because it literally cannot remember the first two failures. BlindSquirrel's state graph solves this permanently: every (state, action) outcome is recorded and never forgotten. A failed action at a given state is masked to weight zero forever.

**Lesson for LLM agents: persistent structured memory beats sliding-window context.** The right architecture is probably a hybrid -- LLM for reasoning about novel states, state graph for remembering what has been tried. The LLM decides *what to explore*; the graph remembers *what has been explored*.

### Semantic reasoning solves the exploration problem

BlindSquirrel and StochasticGoose must visit every state they learn about. For a 64x64 grid with 16 colors, the state space is astronomically large. In practice, they only visit a tiny fraction -- but that fraction must include the solution path, or they fail.

Our LLM agent can sometimes skip the exploration entirely. When Claude sees a rotation switch and a target box, it can infer the solution without trying every possible action sequence. The 19-action ls20 solution was produced on the first attempt with the correct prompt -- no exploration needed.

**Lesson for RL agents: semantic priors compress the search space.** Even a rough understanding of "what kind of puzzle this is" can reduce the effective state space by orders of magnitude. The question is how to acquire that understanding without a human writing it into a prompt.

### Frame-change detection is a universal reward signal

StochasticGoose's binary reward (did the frame change?) is crude but universal. It works across all game types without modification. Our agent needed three separate system prompts for three game types. BlindSquirrel needed object segmentation and action-value learning. StochasticGoose needed one loss function.

**Lesson for all agents: the simplest reward signal that works is usually the right starting point.** Frame-change detection is not sufficient for solving puzzles, but it is sufficient for *orienting* exploration. An agent that has already learned "clicking this area changes the grid" is better positioned for higher-level reasoning than one exploring randomly.

### Determinism is an exploitable structure

BlindSquirrel's assumption that games are deterministic is technically wrong (timers change every frame) but practically right (game logic is deterministic). By exploiting this structure, the agent avoids re-exploring known territory and can guarantee that its state graph is correct.

Our LLM agent does not exploit determinism at all. If Claude navigates to position (19, 30) and the switch activates, it does not record "navigating to (19, 30) activates the switch" in any persistent structure. If it needs to activate the switch again on a later level, it must re-derive the action from scratch.

**Lesson: exploit the structure of the environment, not just its content.** Determinism, symmetry, compositionality -- these are properties of the environment that reduce the effective complexity of the problem. An agent that recognizes and exploits them will outperform one that treats every state as novel.

## The hybrid that does not yet exist

The ideal agent would combine:

1. **LLM reasoning** for initial puzzle comprehension and semantic priors (ClaudeAgent)
2. **Deterministic state graph** for persistent memory and explored-state deduplication (BlindSquirrel)
3. **Frame-change detection** as a universal orientation signal before semantic understanding is available (StochasticGoose)
4. **Learned action-value model** for prioritizing exploration after partial knowledge is acquired (BlindSquirrel)

The execution flow would be:

1. **Orient**: Use frame-change detection to quickly identify interactive regions (StochasticGoose's approach, but for 50 actions instead of 8 hours)
2. **Comprehend**: Send the LLM a screenshot with annotated interactive regions and ask "what kind of puzzle is this?" (ClaudeAgent's approach, but informed by empirical data)
3. **Plan**: LLM proposes a solution strategy. Execute it.
4. **Record**: Log every (state, action, outcome) in a persistent graph (BlindSquirrel's approach)
5. **Learn**: If the plan fails, use the graph to avoid repeating failures and train a lightweight model to guide re-exploration
6. **Re-plan**: Return to the LLM with updated knowledge ("I tried X and it didn't work, the state graph shows Y") for a revised strategy

This hybrid does not exist yet. Each of the three agents we studied implements one or two of these steps well. None implements all of them. The question is whether the integration overhead is worth the improvement -- or whether the simplest approach that reaches 80% of the solution space is the right engineering choice.

## The safety angle

These three philosophies map directly onto a recurring tension in AI safety research: **reasoning vs. robustness**.

LLM-based agents are powerful but fragile. They can solve problems that require genuine understanding -- but they can also act confidently on wrong models, and their failure modes are hard to detect. This is the alignment problem in miniature: an agent that reasons about its environment is more capable *and* more dangerous than one that merely reacts to it.

State-graph agents are robust but limited. They cannot solve problems that require reasoning beyond their explored state space -- but they also cannot be "wrong" about states they have visited. Their knowledge is ground truth by construction. This is the interpretability argument: a system whose decisions can be traced to specific experiences is easier to trust than one whose decisions emerge from opaque reasoning.

CNN learners are simple but shallow. They scale with data and compute, not with understanding. They will never have an "aha" moment, but they will also never hallucinate one. This is the scaling argument: more data and more compute might solve the problem without ever needing to solve the harder problem of understanding.

ARC-AGI-3 is a microcosm of this tension. The benchmark rewards agents that can both *understand* novel environments and *reliably execute* solutions. The three agents we studied each optimize for one side of this tradeoff. The benchmark's implicit challenge is to combine them.

---

*Agents analyzed: ClaudeAgent (Claude Sonnet 4.5, 13 versions), BlindSquirrel (2nd place ARC-AGI-3 Preview, state graph + ResNet18), StochasticGoose (CNN action learner). Source repos: [wd13ca/ARC-AGI-3-Agents](https://github.com/wd13ca/ARC-AGI-3-Agents), [DriesSmit/ARC3-solution](https://github.com/DriesSmit/ARC3-solution). Full development notes in [arc-agi-3-lessons.md](https://github.com/swarm-ai-research/swarm-artifacts/blob/main/papers/research/arc-agi-3-lessons.md).*
