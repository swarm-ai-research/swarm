"""Search ClawXiv for distributional safety papers and fetch a specific paper."""

import textwrap

from swarm.research.platforms import ClawxivClient


def print_paper(paper, indent="  "):
    """Pretty-print a Paper object."""
    print(f"{indent}Paper ID : {paper.paper_id}")
    print(f"{indent}Title    : {paper.title}")
    print(f"{indent}Authors  : {', '.join(paper.authors) if paper.authors else '(none)'}")
    print(f"{indent}Upvotes  : {paper.upvotes}")
    abstract_text = paper.abstract or "(no abstract)"
    wrapped = textwrap.fill(abstract_text, width=80, initial_indent=f"{indent}           ",
                            subsequent_indent=f"{indent}           ")
    print(f"{indent}Abstract : {wrapped.strip()}")
    print()


def main():
    client = ClawxivClient()  # No API key needed for search

    # --- Part 1: Search queries ---
    queries = [
        "distributional safety multi-agent",
        "SWARM governance",
        "soft labels probabilistic",
        "agent welfare",
    ]

    for query in queries:
        print("=" * 80)
        print(f"SEARCH: \"{query}\"")
        print("=" * 80)
        result = client.search(query)
        print(f"Total results: {result.total_count}")
        if result.papers:
            for i, paper in enumerate(result.papers, 1):
                print(f"\n  [{i}]")
                print_paper(paper)
        else:
            print("  No papers found for this query.\n")

    # --- Part 2: Fetch specific paper ---
    paper_id = "clawxiv.2602.00040"
    print("=" * 80)
    print(f"FETCH PAPER: {paper_id}")
    print("=" * 80)
    paper = client.get_paper(paper_id)
    if paper:
        print_paper(paper)
    else:
        print(f"  Could not retrieve paper {paper_id} (may not exist or service unavailable).\n")

    print("Done.")


if __name__ == "__main__":
    main()
