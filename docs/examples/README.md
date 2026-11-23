# Sample outputs

Real outputs from the local Qwen 3.5 9B on a 16 GB M3 MacBook Air, copied unchanged.

| File | What it shows |
|---|---|
| `digest-2026-11-22-single-agent.md` | A scheduled morning digest (single-agent summaries): 8 picks, each with its slot and the scoring reason |
| `digest-2026-11-15-multi-agent.md` | A Critic, Writer and Verifier digest that opened with the "needs review" banner because the Critic hit its drop limit |
| `answer-abstain.md` | The Q&A agent declining to answer a question the store couldn't support |

More results: `evals/results/` (single vs multi-agent comparisons, vector backend comparison, judge hand-check sheets).
