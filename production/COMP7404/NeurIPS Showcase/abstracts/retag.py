# -*- coding: utf-8 -*-
"""
依据标题+摘要，对 oral / spotlight 全部论文重新判定主题标签（tag1/tag2）。

设计：
  - 24 个既有主题标签，每个标签配一组英文关键词（摘要为英文）。
  - 关键词命中打分：标题权重 TITLE_W，摘要权重 1；按「命中关键词个数」计分。
  - tag1 = 最高分标签；tag2 = 次高分且达到阈值时并列。
  - 歧义保护：若既有 tag1 得分与最高分接近，则保留既有标签，避免无谓抖动。
  - 无任何命中 -> 归入「其他跨领域」。

输出：
  - 覆盖写回 build/NeurIPS2025_oral_spotlight_论文分类清单.csv（先备份 .bak.csv）
  - 诊断报告 abstracts/retag_report.txt（UTF-8）
"""

import argparse
import collections
import csv
import json
import math
import os
import re

BASE = r"D:\HKU\course\COMP7404\NeurIPS Showcase"
CSV_IN = os.path.join(BASE, "build", "NeurIPS2025_oral_spotlight_论文分类清单.csv")
CSV_BAK = os.path.join(BASE, "build", "NeurIPS2025_oral_spotlight_论文分类清单.bak.csv")
ARX = os.path.join(BASE, "abstracts", "arxiv_progress.jsonl")
OLD = os.path.join(BASE, "abstracts", "progress.jsonl")
REPORT = os.path.join(BASE, "abstracts", "retag_report.txt")

TITLE_W = 2.0          # 标题命中相对摘要命中的权重倍数
MIN_SCORE = 3.0        # 认定一个标签所需的最低加权分（约等于 1 个高信息量命中词）
SECOND_RATIO = 0.55    # 次标签需达到主标签分数的比例
KEEP_DOMINANCE = 1.25  # 新标签需超过既有标签该倍数才替换，否则保留既有

# ------------------------------------------------------------------ 标签规则
RULES = {
    "机器学习理论": [
        "theorem", "theoretical", "theoretically", "proof", "prove", "proves", "proven",
        "bound", "bounds", "generalization", "generalize", "sample complexity", "statistical",
        "estimator", "estimation", "minimax", "regret", "pac-bayes", "rademacher",
        "uniform convergence", "asymptotic", "nonparametric", "excess risk", "consistency",
        "learning theory", "high-dimensional", "overparameterized", "over-parameterized",
        "implicit bias", "neural tangent", "mean-field", "mean field", "expressivity",
        "expressive power", "universal approximation", "identifiability", "identifiable",
        "loss landscape", "convergence rate", "lower bound", "upper bound", "information-theoretic",
        "risk bound", "hypothesis class", "vc dimension", "function approximation",
        "kernel method", "recovery guarantee", "provable", "feature learning", "scaling law",
    ],
    "计算机视觉": [
        "image", "images", "visual", "vision", "video", "videos", "3d", "point cloud",
        "point clouds", "object detection", "detection", "segmentation", "recognition",
        "reconstruction", "render", "rendering", "scene", "geometry", "geometric", "camera",
        "pose", "depth", "rgb", "nerf", "neural radiance", "gaussian splatting", "optical flow",
        "tracking", "super-resolution", "super resolution", "inpainting", "image quality",
        "image editing", "image generation", "avatar", "mesh", "novel view", "view synthesis",
        "keypoint", "landmark", "face", "hand", "occlusion", "panoramic", "lidar", "slam",
        "image classification", "object recognition", "visual perception", "texture", "pixels",
    ],
    "安全对齐隐私": [
        "safe", "safety", "safeguard", "alignment", "aligned", "misalignment", "jailbreak",
        "harmful", "harm", "toxic", "toxicity", "poison", "poisoning", "backdoor", "adversarial",
        "attack", "attacks", "attacker", "privacy", "private", "differential privacy", "federated",
        "watermark", "watermarking", "security", "secure", "malicious", "misuse", "bias",
        "fairness", "fair", "ethical", "ethics", "red team", "red-team", "guardrail", "refuse",
        "refusal", "sycophancy", "deception", "deceptive", "unlearning", "membership inference",
        "data leakage", "manipulation", "misinformation", "hallucination", "trustworthy",
        "trustworthiness", "value alignment", "human values", "rights", "anonymity",
        "regulation", "regulate", "regulated", "regulatory", "governance", "legislation",
        "legal", "accountability", "oversight", "addiction", "digital addiction",
    ],
    "LLM与语言模型": [
        "language model", "language models", "large language", "llm", "llms", "text", "token",
        "tokens", "tokenizer", "tokenization", "prompt", "prompts", "prompting", "instruction",
        "instruction tuning", "instruction-following", "fine-tuning", "finetuning", "fine-tune",
        "rlhf", "preference optimization", "dpo", "in-context learning", "chain-of-thought",
        "chain of thought", "natural language", "nlp", "sentence", "dialogue", "chatbot",
        "decoding", "translation", "summarization", "question answering", "pretraining",
        "pretrain", "lora", "language understanding", "text generation", "context window",
        "transformer", "attention", "prompting", "few-shot", "zero-shot",
    ],
    "强化学习": [
        "reinforcement learning", "rl", "policy", "policies", "reward", "rewards",
        "markov decision", "mdp", "mdps", "value function", "q-learning", "q-function",
        "actor-critic", "actor critic", "exploration", "exploit", "multi-armed bandit",
        "bandit", "bandits", "offline reinforcement", "online reinforcement", "trajectory",
        "trajectories", "rollout", "credit assignment", "temporal difference", "goal-conditioned",
        "model-based rl", "model-free", "advantage", "ppo", "grpo", "self-play", "reward model",
    ],
    "生成模型扩散": [
        "diffusion", "denoising", "denoise", "score-based", "score matching", "flow matching",
        "flow model", "generative", "generation", "generative model", "generative models",
        "gan", "gans", "variational autoencoder", "vae", "normalizing flow", "autoregressive",
        "image synthesis", "text-to-image", "text to image", "text-to-video", "synthesis",
        "diffusion model", "diffusion models", "consistency model", "sampler", "denoising diffusion",
        "latent diffusion", "video generation", "3d generation", "sample", "sampling", "noise",
    ],
    "优化与数学": [
        "optimization", "optimizer", "optimizers", "gradient descent", "sgd", "stochastic gradient",
        "adam", "adamw", "momentum", "convex", "non-convex", "nonconvex", "regularization",
        "regularizer", "sparse", "sparsity", "proximal", "duality", "dual", "langevin",
        "learning rate", "second-order", "hessian", "newton", "linear algebra", "matrix",
        "matrices", "tensor", "eigen", "spectral", "frank-parisi", "variational", "gradient norm",
        "clipping", "smoothness", "saddle", "constraint", "constrained optimization", "min-max",
        "bilevel", "bregman", "mirror descent", "monte carlo", "convergence",
    ],
    "高效推理加速": [
        "efficient", "efficiency", "efficiently", "acceleration", "accelerate", "inference",
        "latency", "throughput", "quantization", "quantize", "pruning", "prune", "distillation",
        "distill", "compression", "compress", "sparse attention", "memory footprint", "flops",
        "speedup", "deployment", "serving", "test-time", "on-device", "kv cache", "kv-cache",
        "lightweight", "compute cost", "computational cost", "inference cost", "token budget",
    ],
    "AI4Science": [
        "molecule", "molecules", "molecular", "protein", "proteins", "chemistry", "chemical",
        "physics", "physical", "material", "materials", "biology", "biological", "genomics",
        "gene", "genes", "drug", "climate", "weather", "ocean", "astronomy", "astrophysics",
        "quantum", "scientific", "science", "dna", "rna", "crystal", "atoms", "atom", "pde",
        "partial differential", "fluid", "molecular dynamics", "electron", "neural operator",
        "single-cell", "biomolecular", "conformational", "dark matter", "biomolecule",
    ],
    "训练数据方法": [
        "dataset", "data selection", "data curation", "curate", "data augmentation", "augmentation",
        "synthetic data", "self-supervised", "self supervised", "semi-supervised", "unsupervised",
        "self-training", "data quality", "labeling", "label noise", "annotation", "annotate",
        "active learning", "curriculum", "data contamination", "data mixing", "data mixture",
        "pretraining data", "data distribution", "data pool", "data-centric", "data centric",
        "weak supervision", "pseudo-label", "training data", "data efficiency", "sample selection",
    ],
    "多模态视觉语言": [
        "multimodal", "multi-modal", "vision-language", "vision language", "vlm", "vlms",
        "image-text", "image text", "cross-modal", "cross modal", "clip", "visual question",
        "vqa", "captioning", "caption", "grounding", "video-language", "visual instruction",
        "modalities", "modality", "language-image", "text-image", "mllm", "vision-language model",
    ],
    "评测与基准": [
        "benchmark", "benchmarks", "evaluation", "evaluate", "metric", "metrics", "leaderboard",
        "human evaluation", "testbed", "suite", "comprehensive evaluation", "empirical study",
        "assessment", "quality assessment", "diagnostic benchmark", "error analysis", "we evaluate",
    ],
    "概率因果贝叶斯": [
        "bayesian", "posterior", "prior", "probabilistic", "uncertainty", "calibration",
        "calibrated", "gaussian process", "variational inference", "mcmc", "causal", "causality",
        "causal inference", "treatment effect", "counterfactual", "intervention", "graphical model",
        "latent variable", "latent variables", "stochastic process", "diffusion process",
        "distribution shift", "out-of-distribution", "probability", "random variable", "posterior distribution",
    ],
    "可解释性": [
        "interpretability", "interpretable", "explainability", "explainable", "explanation",
        "attribution", "saliency", "feature attribution", "sparse autoencoder", "sparse autoencoders",
        "sae", "probing", "mechanistic", "circuit", "circuits", "concept bottleneck", "concept",
        "feature visualization", "transparency", "internal representation", "knowledge editing",
    ],
    "神经科学与认知": [
        "brain", "cognitive", "cognition", "cortical", "cortex", "neural activity", "neuronal",
        "neuron", "neurons", "fmri", "eeg", "meg", "spiking", "spike", "neuroscience",
        "hippocampus", "visual cortex", "perception", "perceptual", "human cognition", "psychology",
        "behavioral", "behavioural", "decision-making", "place cells", "grid cells", "working memory",
    ],
    "领域应用": [
        "application", "applications", "applied", "real-world", "real world", "use case",
        "industry", "production", "practical", "in the wild", "case study",
    ],
    "医疗健康": [
        "medical", "clinical", "clinic", "health", "healthcare", "disease", "diagnosis",
        "diagnostic", "patient", "patients", "hospital", "cancer", "tumor", "tumour", "mri",
        "ct scan", "x-ray", "pathology", "histopathology", "biomedical", "biomedicine",
        "electronic health", "ehr", "drug discovery", "therapy", "treatment", "lesion",
        "radiology", "mortality", "anatomical", "clinical trial", "medical imaging",
    ],
    "机器人具身": [
        "robot", "robots", "robotic", "robotics", "manipulation", "manipulator", "embodied",
        "embodiment", "navigation", "locomotion", "actuator", "gripper", "tactile", "drone",
        "drones", "uav", "autonomous driving", "autonomous vehicle", "sim-to-real", "sim to real",
        "dexterous", "grasp", "grasping", "mobile robot", "motion planning", "control policy",
        "hardware", "teleoperation", "quadruped",
    ],
    "Agent智能体": [
        "agent", "agents", "agentic", "tool use", "tool-use", "tool calling", "multi-agent",
        "multiagent", "llm agent", "llm agents", "web agent", "computer use", "task planning",
        "workflow", "orchestration", "autonomous agent", "agent framework", "swe-bench",
        "webarena", "self-reflection", "collaboration among agents", "planning agent", "toolkit",
    ],
    "图与关系学习": [
        "graph", "graphs", "graph neural", "gnn", "gnns", "node", "nodes", "edge", "edges",
        "message passing", "subgraph", "knowledge graph", "graphon", "graph structure", "topology",
        "relational", "link prediction", "hypergraph", "spectral graph", "node classification",
        "graph representation", "graph transformer",
    ],
    "世界模型与模拟": [
        "world model", "world models", "simulator", "simulators", "simulation", "model-based",
        "dynamics model", "predict the future", "video prediction", "environment model", "dreamer",
        "generative environment", "physics simulation", "interactive environment", "game engine",
        "planning", "mental model", "model the world",
    ],
    "语音音频": [
        "speech", "audio", "acoustic", "acoustics", "voice", "sound", "asr", "automatic speech",
        "text-to-speech", "text to speech", "tts", "speaker", "music", "mel", "speech recognition",
        "audio generation", "hearing", "speech synthesis", "speech-to-text",
    ],
    "检索知识与记忆": [
        "retrieval", "retrieve", "retriever", "rag", "retrieval-augmented", "retrieval augmented",
        "knowledge base", "memory", "memorization", "memorize", "vector database",
        "information retrieval", "passage", "external knowledge", "knowledge retrieval", "recall",
    ],
}

TAGS = list(RULES.keys()) + ["其他跨领域"]
OTHER = "其他跨领域"


def kw_re(kw):
    # 左边界同时排除连字符：避免 "in-memory" 命中 "memory"、专有复合词误触发；
    # 右边界只排除字母：保留 "memory-based" / "diffusion-based" 等后缀构词
    parts = [re.escape(p) for p in kw.split()]
    return re.compile(r"(?<![A-Za-z-])" + r"\s+".join(parts) + r"(?![A-Za-z])", re.I)


COMPILED = {tag: [(kw, kw_re(kw)) for kw in kws] for tag, kws in RULES.items()}


def build_idf(rows, abstracts):
    """按命中词在全语料（标题+摘要）中的文档频率计算 IDF 权重，
    使“evaluation/benchmark”等泛词自动降权，专有词保持高权重。"""
    N = len(rows)
    df = collections.Counter()
    for r in rows:
        title = (r["论文标题"] or "").lower()
        ab = (abstracts.get((r["论文标题"] or "").strip(), "") or "").lower()
        text = title + " \n " + ab
        for tag, kws in COMPILED.items():
            for kw, rx in kws:
                if rx.search(text):
                    df[(tag, kw)] += 1
    idf = {}
    for tag, kws in COMPILED.items():
        for kw, rx in kws:
            idf[(tag, kw)] = math.log((N + 1.0) / (df[(tag, kw)] + 1.0))
    return idf


def score_text(text, tag, idf):
    if not text:
        return 0.0
    s = 0.0
    for kw, rx in COMPILED[tag]:
        if rx.search(text):
            s += idf[(tag, kw)]
    return s


def load_abstracts():
    m = {}
    for path in (OLD, ARX):          # 先旧后新，新数据覆盖
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("abstract"):
                    m[d["title"]] = d["abstract"]
    return m


def load_rows():
    with open(CSV_IN, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def decide(title, abstract, old1, old2, idf):
    tl = title.lower()
    al = (abstract or "").lower()
    scores = {}
    for tag in RULES:
        s = 0.0
        for kw, rx in COMPILED[tag]:
            w = idf[(tag, kw)]
            if rx.search(tl):
                s += TITLE_W * w
            elif al and rx.search(al):
                s += w
        if s > 0:
            scores[tag] = s
    if not scores:
        return OTHER, "", {}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_tag, top_s = ranked[0]
    # 歧义保护：既有 tag1 有证据、且新标签未明显更强时，保留既有标签
    if old1 in scores:
        old_s = scores[old1]
        if top_s < KEEP_DOMINANCE * old_s:
            top_tag, top_s = old1, old_s
    if top_s < MIN_SCORE:
        return OTHER, "", scores
    # tag2：与 tag1 不同的次高分，且达到比例阈值
    tag2 = ""
    for tag, s in ranked:
        if tag == top_tag:
            continue
        if s >= max(MIN_SCORE, SECOND_RATIO * top_s):
            tag2 = tag
        break
    return top_tag, tag2, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只生成报告，不覆盖 CSV")
    args = ap.parse_args()
    abstracts = load_abstracts()
    rows = load_rows()
    idf = build_idf(rows, abstracts)
    out = []
    changed1 = 0
    dist = collections.Counter()
    dist2 = collections.Counter()
    missing_abs = 0
    for r in rows:
        title = r["论文标题"].strip()
        old1 = (r["主题标签1"] or "").strip()
        old2 = (r["主题标签2"] or "").strip()
        ab = abstracts.get(title, "")
        if not ab:
            missing_abs += 1
        tag1, tag2, scores = decide(title, ab, old1, old2, idf)
        if tag1 != old1:
            changed1 += 1
        dist[tag1] += 1
        if tag2:
            dist2[tag2] += 1
        out.append({"分组": r["分组"].strip(), "主题标签1": tag1, "主题标签2": tag2, "论文标题": title})

    # 备份 + 写回
    if args.dry_run:
        print("dry-run：跳过 CSV 写入")
    else:
        if not os.path.exists(CSV_BAK):
            os.replace(CSV_IN, CSV_BAK)
        else:
            os.remove(CSV_IN)
        with open(CSV_IN, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["分组", "主题标签1", "主题标签2", "论文标题"])
            w.writeheader()
            w.writerows(out)

    lines = []
    lines.append("total=%d  changed_tag1=%d (%.1f%%)  missing_abstract=%d"
                 % (len(out), changed1, 100.0 * changed1 / max(1, len(out)), missing_abs))
    lines.append("--- tag1 distribution ---")
    for k, v in dist.most_common():
        lines.append("%4d  %s" % (v, k))
    lines.append("--- tag2 distribution ---")
    for k, v in dist2.most_common():
        lines.append("%4d  %s" % (v, k))
    lines.append("--- changed samples (first 40) ---")
    c = 0
    for r, o in zip(rows, out):
        if (r["主题标签1"] or "").strip() != o["主题标签1"]:
            lines.append("%s  [%s]  ->  %s (tag2=%s)"
                         % (r["主题标签1"], r["论文标题"][:70], o["主题标签1"], o["主题标签2"]))
            c += 1
            if c >= 40:
                break
    open(REPORT, "w", encoding="utf-8").write("\n".join(lines))
    print("done. report ->", REPORT, "| changed=", changed1, "| missing_abs=", missing_abs)


if __name__ == "__main__":
    main()
