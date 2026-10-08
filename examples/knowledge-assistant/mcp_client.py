"""A deterministic host and executable integration check over real MCP stdio."""
import asyncio
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError


def payload(result):
    if result.is_error:
        raise AssertionError("MCP returned a tool error")
    return json.loads(result.content[0].text)


async def main():
    # Fixed trusted executable/script configuration, never supplied by a query or document.
    transport = StdioTransport(command=sys.executable,
                               args=["-B", str(Path(__file__).with_name("mcp_server.py").resolve())],
                               keep_alive=False)
    async with Client(transport) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools}
        assert names == {"search", "read_evidence", "ask"}, names
        print("tools/list:", ", ".join(sorted(names)))
        found = payload(await client.call_tool("search", {"request": {"query": "借阅期限是多久？", "limit": 3}}))
        assert found["hits"][0]["chunk_id"] == "loan#p001s001", found
        evidence = payload(await client.call_tool("read_evidence", {"request": {"chunk_id": "loan#p001s001"}}))
        assert "21 天" in evidence["text"], evidence
        answer = payload(await client.call_tool("ask", {"request": {"question": "借阅期限是多久？"}}))
        assert answer["status"] == "answered", answer
        assert answer["evidence"][0]["chunk_id"] == "loan#p001s001", answer
        assert [row["tool"] for row in answer["trace"]] == ["search", "read_evidence"], answer
        refused = payload(await client.call_tool("ask", {"request": {"question": "借阅押金是多少？"}}))
        assert refused["status"] == "refused", refused
        invalid_cases = [
            ("search", {"request": {"query": "借阅", "limit": 0}}),
            ("search", {"request": {"query": "借阅", "limit": True}}),
            ("search", {"request": {"query": "借阅", "path": "/etc/passwd"}}),
            ("search", {"request": {"query": "借阅"}, "unexpected": "field"}),
            ("read_evidence", {"request": {"chunk_id": "../../README.md"}}),
            ("ask", {"request": {"question": ""}}),
            ("shell", {"command": "echo forbidden"}),
        ]
        for name, arguments in invalid_cases:
            try:
                await client.call_tool(name, arguments)
            except ToolError:
                pass
            else:
                raise AssertionError(f"illegal tool call accepted: {name} {arguments}")
        print(answer["answer"])
        print("无答案：", refused["status"])
        print(f"MCP stdio check passed: discovery, search, read, answer, refusal, {len(invalid_cases)} rejected calls.")


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(main(), timeout=30))
