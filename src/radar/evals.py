"""The eval suite (M6): one question file, three things to test, scored the same way locally
and in LangSmith.

**What is tested (targets).** Each target takes `{"question": ...}` and returns outputs:
- `retrieval`: hybrid search alone (no LLM). Fast; isolates the retriever.
- `agent`: the full Q&A graph (ReAct loop, evidence check, citations) on local Qwen.
- `no_retrieval`: the same Qwen answering from memory, with no tools: the baseline that shows
  what retrieval buys.

**How it is scored (evaluators).** Plain functions of `(outputs, reference_outputs)`, the
signature LangSmith's `evaluate()` accepts, so the same code scores both ways:
- `recall@k`: is an expected paper among the top k retrieved? (retrieval and agent)
- `mrr`: 1 / rank of the first expected paper; rewards putting it *first*, not just in the top k
- `cites_expected`: does the answer cite an expected paper?
- `citation_validity`: share of cited IDs that are real papers in the store
- `abstain_correct`: did it abstain exactly when the question is outside the store?
- `claim_support` (optional, `--judge`): a Verifier pass on the answer's claims (LLM-as-judge)
- `seconds`, `tool_steps`: cost, so quality can be weighed against time

**Where results go.** Locally, always: a Markdown report in `evals/results/eval-*.md`. With
LangSmith on, also: the question file is synced to a LangSmith *dataset* (`radar-qa`), each
target x model becomes an *experiment* on it, and the LangSmith UI can put experiments side by
side, question by question, with the trace of every answer one click away.

Questions with `should_abstain` and no `expected_ids` don't count toward retrieval scores (there
is nothing to find); they only test abstention.
"""

import json
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from radar.context import AppContext

DATASET = "radar-qa"
TARGETS = ("retrieval", "agent", "no_retrieval")
_ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
_ABSTAIN_WORDS = re.compile(
    r"couldn't find enough evidence|can't answer|cannot answer|not aware of|don't know|"
    r"no information|not sure which",
    re.IGNORECASE,
)


def load_questions(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    for r in rows:
        r.setdefault("expected_ids", [])
        r.setdefault("should_abstain", not r["expected_ids"])
    return rows


# ---- targets ----------------------------------------------------------------------------------


def make_target(ctx: AppContext, kind: str, k: int = 5, llm=None, judge=None) -> Callable:
    """A function inputs -> outputs for one target. `llm`/`judge` are for tests (fake models);
    normally the Agent builds them from ctx.settings, so a model swap is a settings change."""
    from radar import services

    if kind == "retrieval":

        def retrieval(inputs: dict) -> dict:
            start = time.perf_counter()
            ids = [p.item_id for p in services.search(ctx, inputs["question"], k=k)]
            return {"retrieved": ids, "seconds": round(time.perf_counter() - start, 2)}

        return retrieval

    agent = services.Agent(ctx, llm=llm, judge=judge, checkpointer=_memory_saver())
    no_retrieval = kind == "no_retrieval"

    def answer(inputs: dict) -> dict:
        start = time.perf_counter()
        res = agent.ask(inputs["question"], no_retrieval=no_retrieval, remember=False)
        secs = round(time.perf_counter() - start, 2)
        if no_retrieval:
            # No tools, so no citations in our format; count the arXiv IDs it names instead.
            cited = list(dict.fromkeys(_ARXIV_ID.findall(res.answer)))
            abstained = bool(_ABSTAIN_WORDS.search(res.answer))
        else:
            cited = res.cited
            abstained = not res.cited
        retrieved = list(dict.fromkeys(p["item_id"] for p in res.passages))
        return {
            "answer": res.answer,
            "cited": cited,
            "cited_in_store": [i for i in cited if services.get_item(ctx, i) is not None],
            "retrieved": retrieved,
            "abstained": abstained,
            "tool_steps": sum(1 for t in res.trace if t.get("step") == "observe"),
            "seconds": secs,
            # kept for the claim-support judge; trimmed so the LangSmith run stays readable
            "passages": [{**p, "text": p["text"][:1500]} for p in res.passages],
        }

    return answer


def _memory_saver():
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()  # evals don't need conversations to outlive the run


# ---- evaluators -------------------------------------------------------------------------------
# Each returns {"key", "score"}; score None means "not applicable to this question", and is
# left out of averages (here and in LangSmith).


def _expected(ref: dict) -> list[str]:
    return ref.get("expected_ids") or []


def recall_at_k(outputs: dict, reference_outputs: dict) -> dict:
    exp = set(_expected(reference_outputs))
    score = float(bool(exp & set(outputs.get("retrieved", [])))) if exp else None
    return {"key": "recall@k", "score": score}


def mrr(outputs: dict, reference_outputs: dict) -> dict:
    exp = set(_expected(reference_outputs))
    if not exp:
        return {"key": "mrr", "score": None}
    for rank, i in enumerate(outputs.get("retrieved", []), start=1):
        if i in exp:
            return {"key": "mrr", "score": 1 / rank}
    return {"key": "mrr", "score": 0.0}


def cites_expected(outputs: dict, reference_outputs: dict) -> dict:
    exp = set(_expected(reference_outputs))
    score = float(bool(exp & set(outputs.get("cited", [])))) if exp else None
    return {"key": "cites_expected", "score": score}


def citation_validity(outputs: dict, reference_outputs: dict) -> dict:
    cited = outputs.get("cited", [])
    score = len(outputs.get("cited_in_store", [])) / len(cited) if cited else None
    return {"key": "citation_validity", "score": score}


def abstain_correct(outputs: dict, reference_outputs: dict) -> dict:
    should = bool(reference_outputs.get("should_abstain"))
    return {"key": "abstain_correct", "score": float(outputs.get("abstained") == should)}


def seconds(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "seconds", "score": outputs.get("seconds")}


def tool_steps(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "tool_steps", "score": outputs.get("tool_steps")}


def make_claim_support(ctx: AppContext, judge_llm) -> Callable:
    """LLM-as-judge: the Verifier agent splits the answer into claims and checks each against
    the evidence. For the agent, the evidence is what it retrieved; for no_retrieval (which has
    none), it's the abstracts of the expected papers, i.e. "does the answer match the paper?".
    Caveat (as in evals_agents.py): the judge is the same local Qwen, so treat the number as
    an upper bound."""
    from radar.graphs.agents.verifier import build_verifier, support_rate
    from radar.store import items as item_store

    judge = build_verifier(judge_llm)

    def claim_support(outputs: dict, reference_outputs: dict) -> dict:
        if outputs.get("abstained") or not outputs.get("answer"):
            return {"key": "claim_support", "score": None}
        passages = outputs.get("passages") or []
        if not passages:
            for i in _expected(reference_outputs):
                if item := item_store.get_item(ctx.conn, i):
                    passages.append({"item_id": i, "title": item.title, "text": item.abstract})
        if not passages:
            return {"key": "claim_support", "score": None}
        body = outputs["answer"].split("\n\nSources:")[0]
        results = judge.invoke({"text": body, "passages": passages})["results"]
        return {"key": "claim_support", "score": support_rate(results)}

    return claim_support


EVALUATORS = {
    "retrieval": [recall_at_k, mrr, seconds],
    # For the agent, "retrieved" is every paper it gathered as evidence, in the order found.
    "agent": [
        recall_at_k,
        cites_expected,
        citation_validity,
        abstain_correct,
        tool_steps,
        seconds,
    ],
    "no_retrieval": [cites_expected, citation_validity, abstain_correct, seconds],
}


# ---- running ----------------------------------------------------------------------------------


def run_local(target: Callable, questions: list[dict], evaluators: list[Callable]) -> list[dict]:
    """The same loop LangSmith's evaluate() runs, minus the upload: one row per question."""
    rows = []
    for q in questions:
        ref = {"expected_ids": q["expected_ids"], "should_abstain": q["should_abstain"]}
        try:
            out = target({"question": q["question"]})
            error = ""
        except Exception as e:  # one broken question shouldn't sink the run
            out, error = {}, f"{type(e).__name__}: {e}"
        scores = (
            {} if error else {r["key"]: r["score"] for r in (ev(out, ref) for ev in evaluators)}
        )
        rows.append({"question": q["question"], "outputs": out, "scores": scores, "error": error})
    return rows


def run_langsmith(
    target: Callable,
    evaluators: list[Callable],
    prefix: str,
    metadata: dict,
    dataset: str = DATASET,
    client=None,
    evaluate=None,
) -> tuple[list[dict], str]:
    """Run one experiment on the LangSmith dataset. Returns (rows, experiment name)."""
    if evaluate is None:
        from langsmith import evaluate
    results = evaluate(
        target,
        data=dataset,
        evaluators=evaluators,
        experiment_prefix=prefix,
        metadata=metadata,
        max_concurrency=0,  # one question at a time: Ollama serves one request at a time anyway
        client=client,
    )
    rows = []
    for r in results:
        run = r["run"]
        scores = {e.key: e.score for e in r["evaluation_results"]["results"]}
        rows.append(
            {
                "question": r["example"].inputs["question"],
                "outputs": run.outputs or {},
                "scores": scores,
                "error": run.error or "",
            }
        )
    return rows, getattr(results, "experiment_name", prefix)


def sync_dataset(questions: list[dict], client=None, name: str = DATASET) -> dict:
    """Make the LangSmith dataset match the question file (the file is the source of truth):
    add new questions, update changed answers, delete questions no longer in the file."""
    if client is None:
        from langsmith import Client

        client = Client()
    if client.has_dataset(dataset_name=name):
        ds = client.read_dataset(dataset_name=name)
    else:
        ds = client.create_dataset(
            name, description="AI Research Radar: questions with the papers that answer them"
        )
    existing = {e.inputs["question"]: e for e in client.list_examples(dataset_id=ds.id)}
    wanted = {
        q["question"]: {"expected_ids": q["expected_ids"], "should_abstain": q["should_abstain"]}
        for q in questions
    }
    new = [
        {"inputs": {"question": q}, "outputs": o} for q, o in wanted.items() if q not in existing
    ]
    if new:
        client.create_examples(dataset_id=ds.id, examples=new)
    changed = 0
    for q, o in wanted.items():
        if q in existing and dict(existing[q].outputs or {}) != o:
            client.update_example(existing[q].id, outputs=o)
            changed += 1
    removed = [e for q, e in existing.items() if q not in wanted]
    for e in removed:
        client.delete_example(e.id)
    return {"dataset": name, "added": len(new), "updated": changed, "removed": len(removed)}


def summarize(rows: list[dict]) -> dict:
    """Average each score over the questions where it applies."""
    keys = sorted({k for r in rows for k in r["scores"]})
    out = {}
    for k in keys:
        vals = [r["scores"][k] for r in rows if r["scores"].get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    out["errors"] = sum(1 for r in rows if r["error"])
    return out


def run_suite(
    ctx: AppContext,
    questions_file: Path,
    targets: list[str],
    models: list[str] | None = None,
    k: int = 5,
    judge: bool = False,
    use_langsmith: bool = False,
    limit: int | None = None,
    out_dir: Path | None = Path("evals/results"),
    llm=None,
    judge_llm=None,
    client=None,
    evaluate=None,
) -> dict:
    """Run every target x model and write one report. Retrieval doesn't use the chat model, so
    it runs once whatever `models` says."""
    from radar.context import AppContext as Ctx
    from radar.models import get_chat_model

    questions = load_questions(questions_file)[:limit]
    models = models or [ctx.settings.chat_model]
    report: dict = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "questions": len(questions),
        "backend": ctx.settings.vector_backend,
        "embed_model": ctx.settings.embed_model,
        "k": k,
        "experiments": [],
    }
    if use_langsmith:
        report["dataset"] = sync_dataset(questions, client)

    for target_kind in targets:
        for model in models if target_kind != "retrieval" else [models[0]]:
            settings = ctx.settings.model_copy(update={"chat_model": model, "retrieval_k": k})
            run_ctx = Ctx(settings, ctx.conn, ctx.embeddings, ctx.vectors)
            target = make_target(run_ctx, target_kind, k=k, llm=llm, judge=judge_llm)
            evaluators = list(EVALUATORS[target_kind])
            if judge and target_kind != "retrieval":
                evaluators.append(
                    make_claim_support(run_ctx, judge_llm or get_chat_model(settings))
                )
            label = target_kind if target_kind == "retrieval" else f"{target_kind} / {model}"
            meta = {
                "target": target_kind,
                "model": model,
                "k": k,
                "backend": settings.vector_backend,
                "embed_model": settings.embed_model,
            }
            if use_langsmith:
                prefix = f"{target_kind}-{model}".replace(":", "-").replace("/", "-")
                rows, name = run_langsmith(
                    target, evaluators, prefix, meta, client=client, evaluate=evaluate
                )
            else:
                rows, name = run_local(target, questions, evaluators), ""
            report["experiments"].append(
                {
                    "label": label,
                    **meta,
                    "experiment": name,
                    "summary": summarize(rows),
                    "rows": rows,
                }
            )

    report["markdown"] = to_markdown(report)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"eval-{datetime.now():%Y%m%d-%H%M}.md"
        path.write_text(report["markdown"])
        report["path"] = str(path)
    return report


METRIC_LABELS = {
    "recall@k": "Recall@k (expected paper retrieved)",
    "mrr": "MRR (1 / rank of expected paper)",
    "cites_expected": "Answer cites an expected paper",
    "citation_validity": "Citations that are real store papers",
    "abstain_correct": "Abstains exactly when it should",
    "claim_support": "Claims supported (LLM judge)",
    "tool_steps": "Tool calls per question",
    "seconds": "Seconds per question",
    "errors": "Errors",
}


def _fmt(key: str, v) -> str:
    if v is None:
        return "-"
    if key in ("seconds", "tool_steps"):
        return f"{v:.1f}"
    if key == "errors":
        return str(v)
    return f"{v:.0%}"


def to_markdown(report: dict) -> str:
    exps = report["experiments"]
    lines = [
        f"# Eval run {report['started']}",
        "",
        f"{report['questions']} questions, k={report['k']}, vector backend "
        f"{report['backend']}, embeddings {report['embed_model']}.",
        "",
        "| Metric | " + " | ".join(e["label"] for e in exps) + " |",
        "|---|" + "---|" * len(exps),
    ]
    for key, label in METRIC_LABELS.items():
        if not any(key in e["summary"] for e in exps):
            continue
        cells = [_fmt(key, e["summary"].get(key)) for e in exps]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if any(e["experiment"] for e in exps):
        lines += ["", "LangSmith experiments (Datasets -> radar-qa -> Experiments):"]
        lines += [f"- {e['label']}: `{e['experiment']}`" for e in exps if e["experiment"]]
    for e in exps:
        if e["target"] == "retrieval":
            continue
        lines += [
            "",
            f"## {e['label']}: per question",
            "",
            "| Question | retrieved expected | cited | abstained | s |",
            "|---|---|---|---|---|",
        ]
        for r in e["rows"]:
            o, sc = r["outputs"], r["scores"]
            hit = {1.0: "yes", 0.0: "no"}.get(sc.get("recall@k"), "-")
            cited = ", ".join(o.get("cited", [])) or "-"
            if r["error"]:
                cited = f"error: {r['error'][:60]}"
            lines.append(
                f"| {r['question'][:70]} | {hit} | {cited} | "
                f"{'yes' if o.get('abstained') else 'no'} | {o.get('seconds', '-')} |"
            )
    retrieval = [e for e in exps if e["target"] == "retrieval"]
    for e in retrieval:
        misses = [r for r in e["rows"] if r["scores"].get("recall@k") == 0.0]
        if misses:
            lines += ["", "## Retrieval misses", ""]
            lines += [
                f"- {r['question']} (got {', '.join(r['outputs']['retrieved'])})" for r in misses
            ]
    return "\n".join(lines) + "\n"
