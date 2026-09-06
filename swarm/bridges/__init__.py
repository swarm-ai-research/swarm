"""SWARM integration bridges for external systems.

Available bridge subpackages (lazy import to avoid pulling in optional deps):

    swarm.bridges.aeon              — Aeon agent-first ledger bridge (filesystem)
    swarm.bridges.agent_lab         — AgentLaboratory autonomous research bridge
    swarm.bridges.autogpt           — AutoGPT autonomous agent bridge
    swarm.bridges.claude_code       — Claude Code controller bridge
    swarm.bridges.evoskill          — EvoSkill governed skill discovery bridge
    swarm.bridges.concordia         — Concordia LLM agent simulation bridge
    swarm.bridges.crewai            — CrewAI collaborative AI team bridge
    swarm.bridges.langchain         — LangChain chain / agent executor bridge
    swarm.bridges.langgraph_swarm   — LangGraph Swarm governance extension bridge
    swarm.bridges.mesa              — Mesa ABM framework bridge
    swarm.bridges.openclaw          — REST service layer bridge
    swarm.bridges.live_swe          — Live self-evolving SWE agent bridge
    swarm.bridges.worktree          — Native git worktree sandbox bridge
    swarm.bridges.ralph             — Ralph event-stream bridge
    swarm.bridges.pettingzoo        — PettingZoo multi-agent RL environment bridge
    swarm.bridges.prime_intellect   — Prime Intellect RL training bridge
    swarm.bridges.prime_agent       — Prime Agent continual-harness / RLM governance bridge
    swarm.bridges.ai_scientist     — AI-Scientist autonomous research pipeline bridge
    swarm.bridges.sciagentgym      — SciAgentGym tool substrate integration bridge
    swarm.bridges.hodoscope        — Hodoscope trajectory analysis & visualization bridge
    swarm.bridges.rag              — RAG over run history (semantic search)
"""

__all__: list[str] = []
