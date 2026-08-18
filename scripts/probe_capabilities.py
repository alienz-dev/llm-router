"""Empirically verify which free-tier models support tool calling + JSON-schema structured output.

Talks DIRECTLY to each provider's OpenAI-compatible endpoint (bypassing llm-router) so the
results are ground truth about the model, not about the router.
"""
import asyncio, json, os, sys, time
import httpx, yaml
from dotenv import load_dotenv
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(f"{ROOT}/.env")
CFG = yaml.safe_load(open(f"{ROOT}/config.yaml"))["providers"]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    },
}]

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "job_fit",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
    },
}

conn = sqlite3.connect(f"{ROOT}/llm_router.db")
MODELS = conn.execute(
    "SELECT provider_id, model_id FROM models WHERE active=1 ORDER BY provider_id, model_id"
).fetchall()

SKIP_PROVIDERS = {"huggingface", "google"}  # no key / not OpenAI-compatible at this base_url

sem = asyncio.Semaphore(4)


async def call(client, provider, model, payload, timeout=90):
    p = CFG[provider]
    key = os.environ.get(p["api_key_env"], "")
    url = p["base_url"].rstrip("/") + "/chat/completions"
    t0 = time.time()
    try:
        r = await client.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        dt = round(time.time() - t0, 1)
        if r.status_code != 200:
            body = r.text[:180].replace("\n", " ")
            return {"ok": False, "status": r.status_code, "err": body, "sec": dt}
        return {"ok": True, "status": 200, "body": r.json(), "sec": dt}
    except Exception as e:
        return {"ok": False, "status": 0, "err": f"{type(e).__name__}: {e}"[:180],
                "sec": round(time.time() - t0, 1)}


def classify_tools(body):
    try:
        msg = body["choices"][0]["message"]
    except Exception:
        return "malformed", ""
    tc = msg.get("tool_calls")
    if tc:
        try:
            fn = tc[0]["function"]
            args = json.loads(fn["arguments"])
            has_id = bool(tc[0].get("id"))
            note = f"{fn['name']}({','.join(args)})" + ("" if has_id else " NO-ID")
            return ("native" if has_id else "native-noid"), note
        except Exception as e:
            return "native-broken", f"{type(e).__name__}"
    content = (msg.get("content") or "")
    if "get_weather" in content:
        return "text-leak", content[:70].replace("\n", " ")
    return "ignored", content[:70].replace("\n", " ")


def classify_json(body):
    try:
        content = body["choices"][0]["message"]["content"] or ""
    except Exception:
        return "malformed", ""
    try:
        obj = json.loads(content)
    except Exception:
        return "not-json", content[:60].replace("\n", " ")
    if isinstance(obj, dict) and {"score", "reason"} <= set(obj):
        return "schema-ok", json.dumps(obj)[:60]
    return "json-off-schema", json.dumps(obj)[:60]


async def probe(client, provider, model):
    async with sem:
        base = {"model": model, "max_tokens": 300}
        r1 = await call(client, provider, model, {
            **base,
            "messages": [{"role": "user", "content": "What is the weather in Sydney? Use the tool."}],
            "tools": TOOLS, "tool_choice": "auto",
        })
        if r1["ok"]:
            tool_v, tool_n = classify_tools(r1["body"])
        else:
            tool_v, tool_n = f"HTTP{r1['status']}", r1["err"]

        r2 = await call(client, provider, model, {
            **base,
            "messages": [{"role": "user",
                          "content": "Rate this job fit 0-100: Python backend role, candidate is a Python dev. Respond as JSON."}],
            "response_format": SCHEMA,
        })
        if r2["ok"]:
            json_v, json_n = classify_json(r2["body"])
        else:
            json_v, json_n = f"HTTP{r2['status']}", r2["err"]

        row = {"provider": provider, "model": model, "tools": tool_v, "tools_note": tool_n,
               "json": json_v, "json_note": json_n, "sec": r1["sec"]}
        print(f"{provider:12} {model[:46]:46} tools={tool_v:14} json={json_v:14} {r1['sec']}s", flush=True)
        return row


async def main():
    targets = [(p, m) for p, m in MODELS if p not in SKIP_PROVIDERS]
    print(f"probing {len(targets)} models\n", flush=True)
    async with httpx.AsyncClient() as client:
        rows = await asyncio.gather(*[probe(client, p, m) for p, m in targets])
    out = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/docs/model-capabilities.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


asyncio.run(main())
