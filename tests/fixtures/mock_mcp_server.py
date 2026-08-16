import json
import os
import sys
import time

shutdown_marker = None


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    request_id = message.get("id")

    if method == "initialize":
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": message["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "flux-test", "version": "1.0.0"},
                },
            }
        )
    elif method == "tools/list":
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo test arguments",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                },
            }
        )
    elif method == "tools/call":
        arguments = message["params"].get("arguments", {})
        shutdown_marker = arguments.get("shutdown_marker", shutdown_marker)
        time.sleep(arguments.get("delay_ms", 0) / 1000)
        if arguments.get("mode") == "text":
            result = {
                "content": [{"type": "text", "text": arguments.get("text", "")}],
                "isError": False,
            }
        else:
            result = {
                "content": [],
                "structuredContent": {
                    "arguments": arguments,
                    "allowed_env": os.environ.get(arguments.get("allowed_env_name", "")),
                    "blocked_env": os.environ.get(arguments.get("blocked_env_name", "")),
                },
                "isError": arguments.get("fail", False),
            }
        send({"jsonrpc": "2.0", "id": request_id, "result": result})


if shutdown_marker:
    with open(shutdown_marker, "w", encoding="utf-8") as file:
        file.write("closed")
