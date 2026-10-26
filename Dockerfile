# Full pipeline (downloads the Kenneth French data on first run) and the offline tests.
#   docker build -t bolab .
#   docker run --rm -v "$PWD/reports:/app/reports" bolab     # about 45 seconds
#   docker run --rm --entrypoint pytest bolab -q
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY bolab bolab
COPY tests tests
RUN pip install --no-cache-dir -e ".[dev]"
ENTRYPOINT ["python", "-m", "bolab"]
