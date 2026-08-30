"""Integration tests against a running server."""
import httpx
import json
import sys

BASE = "http://localhost:8642"
PASS = 0
FAIL = 0


def test(name, method, path, expected_code, data=None, check_fn=None):
    global PASS, FAIL
    try:
        if method == "GET":
            resp = httpx.get(f"{BASE}{path}", timeout=30)
        else:
            resp = httpx.post(f"{BASE}{path}", json=data, timeout=60)

        if resp.status_code != expected_code:
            print(f"  ❌ {name} (expected {expected_code}, got {resp.status_code})")
            print(f"     {resp.text[:200]}")
            FAIL += 1
            return None

        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text

        if check_fn and not check_fn(body, resp):
            print(f"  ❌ {name} (check function failed)")
            FAIL += 1
            return None

        print(f"  ✅ {name} (HTTP {resp.status_code})")
        PASS += 1
        return body
    except Exception as e:
        print(f"  ❌ {name} (exception: {e})")
        FAIL += 1
        return None


def check_has_models(body, resp):
    return "data" in body and len(body["data"]) > 0


def check_has_providers(body, resp):
    return isinstance(body, dict) and len(body) > 0


def check_has_choices(body, resp):
    return "choices" in body and len(body["choices"]) > 0


def check_has_health_models(body, resp):
    return "models" in body and len(body["models"]) > 0


def check_dashboard(body, resp):
    return "active_models" in body and "jobs" in body


def check_router_headers(body, resp):
    return "x-llm-router-provider" in resp.headers and "x-llm-router-model" in resp.headers


print("═══════════════════════════════════════════════")
print("  Comprehensive Integration Tests")
print("═══════════════════════════════════════════════")

# ── Health & Meta ──
print("\n── Health & Meta ──")
test("GET /health", "GET", "/health", 200, check_fn=lambda b, r: b.get("status") == "ok")
test("GET /v1/models", "GET", "/v1/models", 200, check_fn=check_has_models)
test("GET /v1/quota", "GET", "/v1/quota", 200, check_fn=check_has_providers)
test("GET /v1/providers/health", "GET", "/v1/providers/health", 200, check_fn=check_has_providers)
test("GET /v1/models/health", "GET", "/v1/models/health", 200, check_fn=check_has_health_models)
test("GET /dashboard", "GET", "/dashboard", 200, check_fn=check_dashboard)

# ── Chat Completions: Auto-routing ──
print("\n── Chat Completions: Auto-routing ──")
body = test(
    "Auto-route (general)", "POST", "/v1/chat/completions", 200,
    data={"messages": [{"role": "user", "content": "Say hello in one word"}]},
    check_fn=check_has_choices,
)
if body:
    print(f"     Model: {body.get('model', '?')}")
    content = body["choices"][0]["message"]["content"][:80]
    print(f"     Content: {content}")

body = test(
    "Auto-route (code task)", "POST", "/v1/chat/completions", 200,
    data={"messages": [{"role": "user", "content": "Write a Python function to add two numbers"}]},
    check_fn=check_has_choices,
)
if body:
    print(f"     Model: {body.get('model', '?')}")

body = test(
    "Auto-route (reasoning task)", "POST", "/v1/chat/completions", 200,
    data={"messages": [{"role": "user", "content": "Analyze why water boils at 100°C, step by step"}]},
    check_fn=check_has_choices,
)
if body:
    print(f"     Model: {body.get('model', '?')}")

# ── Chat Completions: Direct routing ──
print("\n── Chat Completions: Direct routing ──")
body = test(
    "Direct: deepseek:deepseek-v4-flash", "POST", "/v1/chat/completions", 200,
    data={"messages": [{"role": "user", "content": "Hi"}], "model": "deepseek:deepseek-v4-flash"},
    check_fn=check_has_choices,
)
if body:
    print(f"     Model: {body.get('model', '?')}")
    content = body["choices"][0]["message"]["content"][:80]
    print(f"     Content: {content}")

# ── Chat Completions: Streaming ──
print("\n── Chat Completions: Streaming ──")
try:
    with httpx.stream("POST", f"{BASE}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Count from 1 to 3"}],
        "stream": True,
    }, timeout=60) as resp:
        chunks = []
        for line in resp.iter_lines():
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        chunks.append(content)
                except json.JSONDecodeError:
                    pass
        if chunks:
            print(f"  ✅ Streaming ({len(chunks)} chunks)")
            print(f"     Content: {''.join(chunks)[:80]}")
            PASS += 1
        else:
            print(f"  ❌ Streaming (no chunks received)")
            FAIL += 1
except Exception as e:
    print(f"  ❌ Streaming ({e})")
    FAIL += 1

# ── Chat Completions: With params ──
print("\n── Chat Completions: With params ──")
body = test(
    "With max_tokens", "POST", "/v1/chat/completions", 200,
    data={"messages": [{"role": "user", "content": "Write a long essay about AI"}], "max_tokens": 10},
    check_fn=check_has_choices,
)
if body:
    usage = body.get("usage", {})
    print(f"     Completion tokens: {usage.get('completion_tokens', '?')}")

# ── Chat Completions: Response headers ──
print("\n── Chat Completions: Response headers ──")
try:
    resp = httpx.post(f"{BASE}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hello"}],
    }, timeout=60)
    if resp.status_code == 200:
        provider = resp.headers.get("x-llm-router-provider", "")
        model = resp.headers.get("x-llm-router-model", "")
        if provider and model:
            print(f"  ✅ Router headers (provider={provider}, model={model})")
            PASS += 1
        else:
            print(f"  ❌ Router headers missing")
            FAIL += 1
    else:
        print(f"  ❌ Router headers (HTTP {resp.status_code})")
        FAIL += 1
except Exception as e:
    print(f"  ❌ Router headers ({e})")
    FAIL += 1

# ── Error handling ──
print("\n── Error handling ──")
test(
    "Invalid model", "POST", "/v1/chat/completions", 502,
    data={"messages": [{"role": "user", "content": "Hi"}], "model": "nonexistent:model"},
)

# Empty messages pass Pydantic validation but fail at provider level (502 is correct)
test(
    "Empty messages", "POST", "/v1/chat/completions", 502,
    data={"messages": []},
)

# ── Jobs ──
print("\n── Jobs ──")
body = test(
    "Submit batch job", "POST", "/jobs", 200,
    data={"messages": [{"role": "user", "content": "What is 2+2?"}], "priority": "batch"},
)
if body and "job_id" in body:
    job_id = body["job_id"]
    test(f"Get job status", "GET", f"/jobs/{job_id}", 200)
    test(f"Process jobs", "POST", "/jobs/process", 200)

# ── Auth middleware (disabled since API_KEY not set) ──
print("\n── Auth middleware ──")
try:
    resp = httpx.get(f"{BASE}/health", timeout=5)
    if resp.status_code == 200:
        print(f"  ✅ No auth required (API_KEY not set)")
        PASS += 1
    else:
        print(f"  ❌ Unexpected status {resp.status_code}")
        FAIL += 1
except Exception as e:
    print(f"  ❌ Auth test ({e})")
    FAIL += 1

# ── Summary ──
print("\n═══════════════════════════════════════════════")
print(f"  Results: {PASS} passed, {FAIL} failed")
print("═══════════════════════════════════════════════")

sys.exit(0 if FAIL == 0 else 1)
