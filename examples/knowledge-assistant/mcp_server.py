"""Optional FastMCP 2.12.5 adapter. stdout is reserved for the protocol."""
import sys

sys.dont_write_bytecode = True

from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field
from knowledge_assistant import KnowledgeBase, ToolRunner, answer_question, load_documents

mcp = FastMCP("fictional-knowledge-assistant")
kb = KnowledgeBase(load_documents())


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=3, ge=1, le=5)


class EvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    chunk_id: str = Field(min_length=1, max_length=100)


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: str = Field(min_length=1, max_length=500)


def call_core(name, arguments):
    result = ToolRunner(kb, budget=1).call(name, arguments)
    if not result["ok"]:
        raise ValueError(f"{result['error']['code']}: {result['error']['message']}")
    return result["result"]


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False})
def search(request: SearchRequest) -> dict:
    """Search only the bundled fictional documents using lexical terms."""
    return {"hits": call_core("search", request.model_dump())}


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False})
def read_evidence(request: EvidenceRequest) -> dict:
    """Read an existing chunk ID. Paths, URLs and arbitrary files are not accepted."""
    return call_core("read_evidence", request.model_dump())


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False})
def ask(request: QuestionRequest) -> dict:
    """Run the fixed search/read/quote workflow; no LLM or autonomous planning."""
    result = answer_question(kb, request.question)
    if result["status"] == "error":
        raise ValueError(result["answer"])
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
