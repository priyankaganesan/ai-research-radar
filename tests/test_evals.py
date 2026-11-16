"""M6 eval suite: targets, evaluators, the local runner and the LangSmith wiring (with a fake
`evaluate` and a fake client, so nothing leaves the machine)."""

import json
import uuid
from types import SimpleNamespace

from conftest import fake_chat, tool_call

from radar import evals

QUESTIONS = [
    {"question": "What is Medusa?", "expected_ids": ["2401.10774"]},
    {"question": "How do draft models speed up inference?", "expected_ids": ["2410.00002"]},
    {"question": "Who won the 1998 World Cup?", "should_abstain": True},
]


def write_questions(tmp_path, rows=QUESTIONS):
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_evaluators():
    ref = {"expected_ids": ["a", "b"], "should_abstain": False}
    out = {"retrieved": ["x", "b", "a"], "cited": ["b", "zz"], "cited_in_store": ["b"]}
    assert evals.recall_at_k(out, ref)["score"] == 1.0
    assert evals.mrr(out, ref)["score"] == 0.5
    assert evals.cites_expected(out, ref)["score"] == 1.0
    assert evals.citation_validity(out, ref)["score"] == 0.5
    assert evals.abstain_correct({**out, "abstained": False}, ref)["score"] == 1.0
    abstain_ref = {"expected_ids": [], "should_abstain": True}
    assert evals.recall_at_k(out, abstain_ref)["score"] is None  # nothing to find: not counted
    assert evals.citation_validity({"cited": []}, ref)["score"] is None


def test_load_questions_defaults_should_abstain(tmp_path):
    rows = evals.load_questions(write_questions(tmp_path))
    assert [r["should_abstain"] for r in rows] == [False, False, True]


def test_retrieval_suite_runs_locally_and_writes_a_report(filled_ctx, tmp_path):
    report = evals.run_suite(
        filled_ctx, write_questions(tmp_path), ["retrieval"], out_dir=tmp_path / "results"
    )
    (exp,) = report["experiments"]
    assert exp["summary"]["recall@k"] == 1.0
    assert exp["summary"]["mrr"] > 0
    assert "Recall@k" in report["markdown"]
    assert (tmp_path / "results").glob("eval-*.md")


def test_agent_and_baseline_targets(filled_ctx, tmp_path):
    rows = QUESTIONS[:1] + QUESTIONS[2:]
    llm = fake_chat(
        tool_call("store_search", query="Medusa"),
        "Medusa adds decoding heads [2401.10774].",
        tool_call("store_search", query="World Cup 1998"),
        tool_call("search_arxiv", query="World Cup 1998"),
        # no_retrieval answers (no tools): one names a real store paper, one a made-up ID
        "Medusa (arXiv 2401.10774) adds heads; see also 2499.99999.",
        "I don't know.",
    )
    judge = fake_chat(
        '{"sufficient": true, "missing": ""}',
        '{"sufficient": false, "missing": "nothing on football"}',
        '{"sufficient": false, "missing": "nothing on football"}',
    )
    import radar.sources.arxiv as arxiv

    filled_ctx.settings = filled_ctx.settings.model_copy(update={"max_tool_steps": 2})
    arxiv_search, arxiv.search = arxiv.search, lambda q, max_results=5: []
    try:
        report = evals.run_suite(
            filled_ctx,
            write_questions(tmp_path, rows),
            ["agent", "no_retrieval"],
            out_dir=None,
            llm=llm,
            judge_llm=judge,
        )
    finally:
        arxiv.search = arxiv_search
    agent, base = report["experiments"]
    assert [r["error"] for e in (agent, base) for r in e["rows"]] == [""] * 4
    assert agent["summary"]["cites_expected"] == 1.0
    assert agent["summary"]["citation_validity"] == 1.0
    assert agent["summary"]["abstain_correct"] == 1.0  # answered one, abstained on the other
    assert base["rows"][0]["outputs"]["cited"] == ["2401.10774", "2499.99999"]
    assert base["summary"]["citation_validity"] == 0.5  # one real, one invented
    assert base["rows"][1]["outputs"]["abstained"] is True
    # evals don't touch long-term memory: no questions logged, no gaps opened
    assert filled_ctx.conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 0


def test_langsmith_path_syncs_dataset_and_reads_results(filled_ctx, tmp_path):
    calls = {}

    class FakeClient:
        def __init__(self):
            self.examples = [
                SimpleNamespace(id=uuid.uuid4(), inputs={"question": "Old question"}, outputs={}),
                SimpleNamespace(
                    id=uuid.uuid4(),
                    inputs={"question": "What is Medusa?"},
                    outputs={"expected_ids": ["wrong"], "should_abstain": False},
                ),
            ]
            self.created, self.updated, self.deleted = [], [], []

        def has_dataset(self, dataset_name):
            return True

        def read_dataset(self, dataset_name):
            return SimpleNamespace(id="ds1")

        def list_examples(self, dataset_id):
            return self.examples

        def create_examples(self, dataset_id, examples):
            self.created += examples

        def update_example(self, example_id, outputs):
            self.updated.append(outputs)

        def delete_example(self, example_id):
            self.deleted.append(example_id)

    class Results(list):  # what langsmith.evaluate returns: iterable rows + experiment name
        experiment_name = "retrieval-abc"

    def evaluate(target, data, evaluators, experiment_prefix, metadata, **kw):
        calls.update(data=data, prefix=experiment_prefix, metadata=metadata)
        out = target({"question": "What is Medusa?"})
        ref = {"expected_ids": ["2401.10774"], "should_abstain": False}
        scores = [SimpleNamespace(**ev(out, ref)) for ev in evaluators]
        return Results(
            [
                {
                    "run": SimpleNamespace(outputs=out, error=None),
                    "example": SimpleNamespace(inputs={"question": "What is Medusa?"}),
                    "evaluation_results": {"results": scores},
                }
            ]
        )

    client = FakeClient()
    report = evals.run_suite(
        filled_ctx,
        write_questions(tmp_path),
        ["retrieval"],
        use_langsmith=True,
        client=client,
        evaluate=evaluate,
        out_dir=None,
    )
    assert report["dataset"] == {"dataset": "radar-qa", "added": 2, "updated": 1, "removed": 1}
    assert client.updated == [{"expected_ids": ["2401.10774"], "should_abstain": False}]
    assert calls["data"] == "radar-qa" and calls["metadata"]["target"] == "retrieval"
    (exp,) = report["experiments"]
    assert exp["experiment"] == "retrieval-abc"
    assert exp["summary"]["recall@k"] == 1.0
    assert "radar-qa" in report["markdown"]
