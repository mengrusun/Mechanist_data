"""Thin wrapper for GPT-5 (dmxapi) chat completion with proxy bypass and retry."""
import os
import time
import json


def _drop_proxies():
    for k in ["ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]:
        os.environ.pop(k, None)


_drop_proxies()

from openai import OpenAI  # noqa: E402
from config import API_KEY, BASE_URL, MODEL  # noqa: E402

_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def chat(messages, model=MODEL, temperature=None, max_tokens=None, retries=3, sleep=2.0, json_mode=False):
    """Return the assistant string. On failure, raise after retries."""
    _drop_proxies()
    last_err = None
    for attempt in range(retries):
        try:
            kwargs = dict(model=model, messages=messages)
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            r = _client.chat.completions.create(**kwargs)
            return r.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(sleep * (attempt + 1))
    raise last_err


def chat_json(messages, **kw):
    """Chat expecting a valid JSON object as the reply. Retries with a fixer prompt if needed."""
    txt = chat(messages, json_mode=True, **kw)
    txt = _strip_code_fences(txt)
    try:
        return json.loads(txt)
    except Exception:
        # try once more with reminder
        fix_msgs = messages + [
            {"role": "assistant", "content": txt},
            {"role": "user", "content": "Your previous reply was not valid JSON. Reply ONLY with a valid JSON object. No prose, no code fences."},
        ]
        txt2 = chat(fix_msgs, json_mode=True, **kw)
        return json.loads(_strip_code_fences(txt2))


def _strip_code_fences(txt):
    t = txt.strip()
    if t.startswith("```"):
        # strip first fence line
        parts = t.split("\n", 1)
        if len(parts) == 2:
            t = parts[1]
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


if __name__ == "__main__":
    out = chat_json([
        {"role": "system", "content": "You reply in JSON."},
        {"role": "user", "content": "Return {\"greeting\": \"hi\"}"},
    ])
    print(out)
