FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install A2A SDK and dependencies
RUN pip install --no-cache-dir a2a-sdk httpx requests pydantic uvicorn

# Copy green agent source
COPY green/ /app/green/

ENV MOCK_URL=http://mock-comtrade:8000
ENV PURPLE_OUTPUT_ROOT=/workspace/purple_output
ENV PORT=9009

EXPOSE 9009

ENTRYPOINT ["python", "-m", "green.agent_a2a"]
CMD []
