"""Run from any working directory; all default operations stay local and read-only."""
import argparse
import json
import sys

sys.dont_write_bytecode = True

from knowledge_assistant import (KnowledgeBase, ToolRunner, answer_question,
                                 evaluate, load_documents, load_queries)


def show_answer(question, result):
    print(f"\n问题：{question}\n状态：{result['status']} / 词项覆盖率：{result['coverage']:.3f}")
    print(result["answer"])
    print("工具 trace：")
    for row in result["trace"]:
        print(f"  {row['step']}. {row['tool']} {row['arguments']} → {row['status']}")


def show_report(report):
    summary = report["summary"]
    print(f"\n离线评估：{summary['count']} 个问题；有答案 {summary['answerable_count']}；无答案 {summary['unanswerable_count']}")
    print("ID   结果  状态       失败原因")
    for row in report["rows"]:
        print(f"{row['id']}  {'PASS' if row['success'] else 'FAIL'}  {row['status']:<10} {row['failure'] or '-'}")
    for name, value in summary.items():
        if name not in {"count", "answerable_count", "unanswerable_count"}:
            print(f"  {name}: {'N/A' if value is None else f'{value:.4f}'}")
    print(f"语料 SHA-256：{report['config']['corpus_sha256']}")
    print("这是开发回归集的确定性摘录结果，不是 LLM 准确率或生产质量保证。")


def main():
    parser = argparse.ArgumentParser(description="虚构资料的确定性知识助手；无 API key、无默认网络调用")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="演示回答、拒答、参数拒绝和离线评估")
    ask = subparsers.add_parser("ask", help="原文摘录问答")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="输出完整结构化 trace")
    evaluate_parser = subparsers.add_parser("evaluate", help="评估固定 20 条问题")
    evaluate_parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    kb = KnowledgeBase(load_documents())
    if args.command == "ask":
        result = answer_question(kb, args.question)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            show_answer(args.question, result)
        return 2 if result["status"] == "error" else 0
    if args.command == "demo":
        print("模式：中文词法检索 + 原文摘录 + 固定工具工作流（不调用 LLM）")
        print(f"语料：10 份虚构文档 / {len(kb.chunks)} 个片段")
        for question in ["借阅期限是多久？", "借阅押金是多少？"]:
            show_answer(question, answer_question(kb, question))
        invalid = ToolRunner(kb).call("search", {"query": "借阅", "limit": 0})
        print("\n非法工具参数：", json.dumps(invalid, ensure_ascii=False))
    report = evaluate(kb, load_queries())
    if getattr(args, "json", False):
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        show_report(report)
    return 0  # Evaluation reports failures as data; only runtime/argument errors fail the command.


if __name__ == "__main__":
    raise SystemExit(main())
