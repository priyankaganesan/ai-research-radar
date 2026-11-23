# AI Research Radar: 2026-11-15

> **Needs review.** Read the sources before relying on this digest:
> - the Critic hit its drop limit, so some kept items carry its concerns

8 items, ranked by `scoring.py`; summaries by local Qwen (multi).

## 1. Tree-Ising: Repairing Language Models with Bit-Level Ising Edits

**Top pick** · `2611.14210` · 2026-11-10 · [link](https://arxiv.org/abs/2611.14210)

The problem involves repairing reasoning loops in frontier language models without relying on gradient-based training or extensive LLM agent interactions. The key idea is to cast weight-space repair as a sparse discrete optimization problem over BF16 weights, decomposing it into independent QUBO problems solved via simulated annealing or quantum-inspired methods using only sign flips and exponent shifts. The main result shows that Tree-Ising outperforms the original agent edit on math benchmarks with fewer modified weights and significantly lower computational costs, including zero LLM tokens and minimal forward-pass requests [2611.14210].

_Why: Top score (0.63, mostly relevance): relevance 0.77, 45 HF upvotes, 5 days old_

## 2. Infrathon: Benchmarking Language Agents for Real-World, Long-Horizon AI Infra Optimization

**Top pick** · `2611.11995` · 2026-11-09 · [link](https://arxiv.org/abs/2611.11995)

The paper introduces Infrathon, a benchmark designed to evaluate whether frontier language agents can continually optimize real-world AI infrastructure over long horizons. Key findings reveal that model rankings shift depending on the evaluation duration and that most runs achieve further improvements in the second half of their budgets, suggesting short evaluations may miss gains or performance changes. Additionally, agents were able to discover solutions surpassing human-designed baselines, such as significantly increasing KV-cache compression ratios, though barriers like task violations and premature completion hinder sustained progress [2611.11995].

_Why: Top score (0.56, mostly relevance): relevance 0.64, 50 HF upvotes, 6 days old_

## 3. Actor, Judge, or Teacher? Understanding Decision Models in Long-Horizon Agents

**Top pick** · `2611.17167` · 2026-11-12 · [link](https://arxiv.org/abs/2611.17167)

The study investigates whether decision models that perform poorly as actors can still effectively teach policies, contrasting their utility when used for acting versus judging or teaching. The key idea is that evaluating actions based on whether they helped achieve subgoals after seeing the full trajectory separates useful from useless actions better than predicting task success. As a teacher, the JEV model successfully raised policy success rates on ALFWorld and WebShop, demonstrating that a poor actor can still teach well because judging reveals consequences that acting must anticipate [2611.17167].

_Why: Top score (0.56, mostly relevance): relevance 0.55, 35 HF upvotes, 3 days old_

## 4. Factored One-sided Adam-Moment for Practical and Scalable SOAP

**Top pick** · `2611.16905` · 2026-11-12 · [link](https://arxiv.org/abs/2611.16905)

The problem is that while second-order optimizers like SOAP offer faster convergence than AdamW, they remain costly due to substantial memory usage and eigendecomposition overhead. The key idea is Factored One-sided Adam-Moment (FOAM), which retains SOAP's benefits by using input-only preconditioning to drop the output-side Kronecker factor and replacing dense second moments with a factored Adam-moment estimator based on row-by-column statistics. The main result shows that FOAM matches strong SOAP and Muon baselines in validation loss across large language model scales while using half of SOAP's optimizer state and running at Muon-class wall-clock time. [2611.16905]

_Why: Top score (0.53, mostly relevance): relevance 0.58, 14 HF upvotes, 3 days old (replaced hn:50443264, dropped by the critic)_

## 5. SteerDistill: Quantizing LLM Agents via Closed-Loop Teacher Steering

**Top pick** · `2611.13410` · 2026-11-10 · [link](https://arxiv.org/abs/2611.13410)

Low-bit quantization of LLM agents often causes task failures because quantized models overlook requirements, leading to behavioral divergences from successful full-precision references [2611.13410]. SteerDistill addresses this by employing a closed-loop teacher steering approach where a full-precision teacher guides the student to revise failed responses and iteratively refines its guidance based on execution feedback [2611.13410]. This method enables unassisted inference that exceeds the strongest quantized baseline by 1.67–2.18 points in average score while using significantly fewer training GPU hours than prior approaches [2611.13410].

_Why: Top score (0.55, mostly relevance): relevance 0.60, 35 HF upvotes, 5 days old_

## 6. InfraRSI: Harnessing Recursive Self-Improvement for Training Infrastructure

**Top pick** · `2611.11558` · 2026-11-08 · [link](https://arxiv.org/abs/2611.11558)

InfraRSI addresses the challenge of tightly coupled distributed LLM training settings where single edits often cause out-of-memory errors, compiler conflicts, or numerical drift. It employs a five-stage agentic workflow that initializes from diverse experience-seeded configurations and fits global models to price memory costs, pairing memory-hungry optimizations with savers while preventing reward hacking through fixed task contracts. Across four dense and Mixture-of-Experts workloads, the system achieves up to 2.75x speedup and increases mean Model FLOPs Utilization from 17.4% to 30.0% by forming a two-level recursive self-improvement loop that revises priors online and distills experience across clusters [2611.11558].

_Why: Top score (0.54, mostly relevance): relevance 0.65, 33 HF upvotes, 7 days old_

_Critic: The paper focuses on training infrastructure optimization rather than the reader's core interests in inference efficiency, agent tool use, or model reasoning._

## 7. SO-Arena: Benchmarking the Capability Spectrum and Reliability of System One Models

**Early buzz** · `2611.15777` · 2026-11-11 · [link](https://arxiv.org/abs/2611.15777)

Existing evaluations often fail to provide sufficient insight into the breadth of capabilities and the reliability of decisions for System One models. The authors introduce SO-Arena, a benchmark that assesses these dimensions through evidence-grounded evaluation, fine-grained domain-specific criteria, realistic interactive sandboxing, and measures of decision stability and missing-answer recognition. Evaluations across 21 models reveal that scaling benefits vary by domain and that strong overall performance does not guarantee reliable decision-making, with the top-performing model ranking seventh in reliability [2611.15777].

_Why: Early buzz: lots of attention, few citations yet (0.57, mostly attention): relevance 0.47, 138 HF upvotes, 4 days old_

## 8. When One GPU Isn’t Enough: Serving GLM-5.3 Across Two NVIDIA DGX Stations with vLLM

**Lab launch** · `post:dbb6a7059ae0` · 2026-11-13 · [link](https://developer.nvidia.com/blog/when-one-gpu-isnt-enough-serving-glm-5-3-across-two-nvidia-dgx-stations-with-vllm/)

The problem is that frontier-scale models no longer fit entirely within the GPU memory of a single machine. The key idea involves connecting two NVIDIA DGX Stations over ConnectX-8 RDMA to serve one model through a single endpoint using tensor parallelism to split weights and computation across both GPUs. This approach enables local serving of large quantized models like GLM-5.3 when one GPU is insufficient [post:dbb6a7059ae0].

_Why: Lab launch (0.48, mostly relevance): relevance 0.72, 2 days old_

_Critic: This is a system engineering blog post about distributed serving infrastructure rather than a research contribution on model efficiency, architecture, or evaluation._

---

- Critic dropped 2611.16748 (PhysLoop: Harnessing LLMs' Physics Awareness for Self-Improv): The paper focuses on video generation and physical dynamics rather than the reader's core interests in LLM agents, reasoning, efficient inference, or evaluation.
- Critic dropped 2611.17331 (TTB: Test-time Baking for Efficient Rendering of Decoder-onl): The paper focuses on 3D view synthesis rendering rather than the reader's core interests in LLM agents, RAG, reasoning, or efficient language model inference.
- Critic dropped hn:50443264 (TimeNet: Open-source data infrastructure for Temporal Founda): This is a data infrastructure project rather than a research contribution on model architecture, reasoning, or inference efficiency.

Rate an item: `radar rate ITEM_ID useful|not_relevant|skipped`. Ratings tune tomorrow's digest (`radar profile` shows how).
