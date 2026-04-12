# Green Agent — ComtradeBench Evaluator

The Green Agent is the **evaluator/judge** component of ComtradeBench. It implements the Agent-to-Agent (A2A) protocol and serves as the referee that Purple agents compete against.

## What it does

1. Receives evaluation requests from the AgentBeats platform
2. Sends tasks (T1-T10) to the Purple agent via A2A
3. Collects the Purple agent's output (data.jsonl, metadata.json, run.log)
4. Scores the output using the 6-dimension judge (correctness, completeness, robustness, efficiency, data quality, observability)
5. Reports results back to the leaderboard

## Running

```bash
# Build and run with Docker
docker build -t comtrade-green:latest -f green/Dockerfile .
docker run -p 9009:9009 -e MOCK_URL=http://mock-comtrade:8000 comtrade-green:latest

# Or run directly
python -m green.agent_a2a --port 9009
```

## A2A Endpoints

- `GET /.well-known/agent.json` — Agent card (discovery)
- `POST /a2a/rpc` — JSON-RPC 2.0 task handling
- `GET /healthz` — Health check

## Integration with AgentBeats

The Green Agent is designed to run alongside the mock Comtrade service in a Docker Compose setup. See the root `docker-compose.yml` for the full deployment configuration.
