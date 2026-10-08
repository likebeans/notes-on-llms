"""Dependency-free, deterministic retrieval teaching example."""
import re
import math
from copy import deepcopy
from pathlib import Path
import hashlib
import json


def chunk_document(doc_id, title, text, max_chars=180):
    """Split paragraphs, then long paragraphs by characters; keep source positions."""
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars 必须是正整数")
    chunks = []
    for paragraph, content in enumerate(re.split(r"\n\s*\n", text.strip()), start=1):
        content = content.strip()
        if not content:
            continue
        for offset in range(0, len(content), max_chars):
            segment = offset // max_chars + 1
            chunks.append({
                "chunk_id": f"{doc_id}#p{paragraph:03d}s{segment:03d}",
                "doc_id": doc_id, "title": title, "paragraph": paragraph,
                "start_char": offset, "end_char": min(offset + max_chars, len(content)),
                "text": content[offset:offset + max_chars],
            })
    return chunks


TOP_K = 3
MIN_COVERAGE = 0.60  # Teaching heuristic, not a calibrated answerability probability.


def lexical_terms(text):
    """Chinese character bigrams + lowercased Latin words; no embeddings/tokenizer."""
    text = re.sub(r"是什么|是多少|是多久|有哪些|什么时候|请问|如何|怎么|是否|的|呢", "", text.lower())
    terms = set(re.findall(r"[a-z0-9]+", text))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(run[i:i + 2] for i in range(max(1, len(run) - 1)))
    return terms


class KnowledgeBase:
    def __init__(self, documents):
        self.chunks = {}
        for doc in documents:
            for chunk in chunk_document(doc["doc_id"], doc["title"], doc["text"]):
                if chunk["chunk_id"] in self.chunks:
                    raise ValueError("重复文档 ID")
                self.chunks[chunk["chunk_id"]] = chunk
        self.terms = {key: lexical_terms(chunk["text"]) for key, chunk in self.chunks.items()}

    def search(self, query, limit=TOP_K):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
            raise ValueError("query 必须是 1–500 字符的非空字符串")
        if type(limit) is not int or not 1 <= limit <= 5:
            raise ValueError("limit 必须是 1–5 的整数，不能是布尔值")
        query_terms = lexical_terms(query)
        if not query_terms:
            return []
        weights = {term: 1 + math.log((1 + len(self.chunks)) /
                   (1 + sum(term in terms for terms in self.terms.values())))
                   for term in sorted(query_terms)}
        total_weight = sum(weights.values())
        hits = []
        for chunk_id, terms in self.terms.items():
            matched = query_terms & terms
            if matched:
                hits.append({**self.chunks[chunk_id],
                             "score": sum(weights[t] for t in sorted(matched)) / total_weight,
                             "coverage": len(matched) / len(query_terms),
                             "matched_terms": sorted(matched)})
        return sorted(hits, key=lambda hit: (-hit["score"], hit["chunk_id"]))[:limit]


class ToolRunner:
    """Only two read-only capabilities. Failed attempts also consume the budget."""
    def __init__(self, kb, budget=4):
        if type(budget) is not int or not 1 <= budget <= 10:
            raise ValueError("budget 必须是 1–10 的整数")
        self.kb = kb
        self.budget = budget
        self.trace = []

    def call(self, name, arguments):
        step = len(self.trace) + 1
        error = None
        result = None
        if step > self.budget:
            error = {"code": "budget_exceeded", "message": "工具调用预算已耗尽"}
        elif name not in {"search", "read_evidence"}:
            error = {"code": "unknown_tool", "message": "工具不在白名单内"}
        else:
            try:
                if not isinstance(arguments, dict):
                    raise ValueError("参数必须是对象")
                allowed = {"query", "limit"} if name == "search" else {"chunk_id"}
                required = {"query"} if name == "search" else {"chunk_id"}
                if set(arguments) - allowed or not required <= set(arguments):
                    raise ValueError("缺少必需参数或包含额外参数")
                if name == "search":
                    result = self.kb.search(arguments["query"], arguments.get("limit", TOP_K))
                else:
                    chunk_id = arguments["chunk_id"]
                    if not isinstance(chunk_id, str) or not 1 <= len(chunk_id) <= 100:
                        raise ValueError("chunk_id 必须是 1–100 字符的字符串")
                    if chunk_id not in self.kb.chunks:
                        error = {"code": "not_found", "message": "该证据 ID 不存在"}
                    else:
                        result = deepcopy(self.kb.chunks[chunk_id])
            except ValueError as exc:
                error = {"code": "invalid_arguments", "message": str(exc)}
        row = {"step": step, "tool": name, "arguments": deepcopy(arguments),
               "status": "error" if error else "ok", "result": result, "error": error}
        self.trace.append(row)
        return {"ok": error is None, "result": result, "error": error}


def answer_question(kb, question, tool_budget=4):
    """Fixed search→coverage gate→read→quote workflow; no LLM decisions."""
    runner = ToolRunner(kb, budget=tool_budget)
    response = {"status": "refused", "answer": "资料不足，无法根据现有文档回答。",
                "evidence": [], "coverage": 0.0, "trace": runner.trace}
    found = runner.call("search", {"query": question, "limit": TOP_K})
    if not found["ok"]:
        response.update(status="error", answer="工具调用失败；请查看 trace。")
        return response
    covered = set()
    selected = []
    for hit in found["result"]:
        new_terms = set(hit["matched_terms"]) - covered
        if new_terms:
            selected.append(hit)
            covered.update(new_terms)
    query_terms = lexical_terms(question)
    response["coverage"] = len(covered) / len(query_terms) if query_terms else 0.0
    if not selected or response["coverage"] < MIN_COVERAGE:
        return response
    evidence = []
    for hit in selected:
        read = runner.call("read_evidence", {"chunk_id": hit["chunk_id"]})
        if not read["ok"]:
            response.update(status="error", answer="证据读取未完成；请查看 trace。")
            return response
        evidence.append(read["result"])
    response.update(status="answered", evidence=evidence,
                    answer="检索到的原文摘录（未作推断）：\n" + "\n".join(
                        f"[{item['chunk_id']}] {item['text']}" for item in evidence))
    return response


def score_retrieval(gold, retrieved, k=TOP_K):
    """Deduplicated binary metrics; no-answer queries are a separate bucket."""
    if type(k) is not int or k <= 0:
        raise ValueError("k 必须是正整数")
    gold = set(gold)
    if not gold:
        return {"recall": None, "mrr": None, "ndcg": None}
    retrieved = list(dict.fromkeys(retrieved))[:k]
    hits = [int(item in gold) for item in retrieved]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(gold))))
    return {"recall": sum(hits) / len(gold),
            "mrr": next((1 / (rank + 1) for rank, hit in enumerate(hits) if hit), 0.0),
            "ndcg": dcg / ideal}


DATA_DIR = Path(__file__).resolve().parent / "data"


def load_documents():
    """Read only the bundled fixture directory, never a user-supplied path."""
    documents = []
    for path in sorted((DATA_DIR / "documents").glob("*.md")):
        title, _, text = path.read_text(encoding="utf-8").partition("\n")
        documents.append({"doc_id": path.stem, "title": title.removeprefix("# "),
                          "text": text.strip()})
    return documents


def load_queries():
    return json.loads((DATA_DIR / "queries.json").read_text(encoding="utf-8"))["queries"]


def evaluate(kb, queries):
    """Evaluate retrieval and the extractive workflow separately, with explicit buckets."""
    rows = []
    for query in queries:
        gold = set(query["gold_evidence"])
        if gold - kb.chunks.keys():
            raise ValueError(f"{query['id']}: gold evidence 不在当前语料快照中")
        result = answer_question(kb, query["question"])
        candidates = result["trace"][0]["result"] or []
        retrieved = [item["chunk_id"] for item in candidates]
        cited = {item["chunk_id"] for item in result["evidence"]}
        scores = score_retrieval(gold, retrieved)
        if gold:
            success = (result["status"] == "answered" and gold <= cited and
                       all(fact in result["answer"] for fact in query["expected_answer_contains"]))
            if not gold <= set(retrieved):
                failure = "retrieval_miss"
            elif result["status"] == "refused":
                failure = "coverage_gate_refused"
            elif not success:
                failure = "evidence_or_answer_incomplete"
            else:
                failure = None
        else:
            success = result["status"] == "refused"
            failure = None if success else "false_answer_or_error"
        rows.append({"id": query["id"], "question": query["question"],
                     "bucket": query.get("bucket", "unspecified"), "answerable": bool(gold),
                     "gold_evidence": sorted(gold), "retrieved_ids": retrieved,
                     "cited_ids": sorted(cited), "retrieval": scores,
                     "evidence_recall": len(cited & gold) / len(gold) if gold else None,
                     "citation_precision": len(cited & gold) / len(cited) if cited else None,
                     "success": success, "failure": failure, **result})

    def mean(values):
        present = [value for value in values if value is not None]
        return sum(present) / len(present) if present else None

    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    snapshot = json.dumps(kb.chunks, ensure_ascii=False, sort_keys=True).encode("utf-8")
    summary = {"count": len(rows), "answerable_count": len(answerable),
               "unanswerable_count": len(unanswerable),
               "recall_at_3": mean(row["retrieval"]["recall"] for row in answerable),
               "mrr_at_3": mean(row["retrieval"]["mrr"] for row in answerable),
               "ndcg_at_3": mean(row["retrieval"]["ndcg"] for row in answerable),
               "evidence_recall": mean(row["evidence_recall"] for row in answerable),
               "citation_precision": mean(row["citation_precision"] for row in rows),
               "citation_scored_count": sum(row["citation_precision"] is not None for row in rows),
               "answerable_success_rate": mean(row["success"] for row in answerable),
               "refusal_accuracy": mean(row["success"] for row in unanswerable),
               "task_success_rate": mean(row["success"] for row in rows)}
    return {"config": {"algorithm": "lexical-bigram-extractive-v1", "top_k": TOP_K,
                       "min_coverage": MIN_COVERAGE, "chunk_max_chars": 180,
                       "corpus_sha256": hashlib.sha256(snapshot).hexdigest()},
            "summary": summary, "rows": rows}
