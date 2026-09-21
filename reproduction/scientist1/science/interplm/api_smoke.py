import json, urllib.request

API_URL = "https://www.dmxapi.cn/v1/chat/completions"
API_KEY = "<Your_api>"
MODEL = "gpt-5.4"

body = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": "reply with only the word OK"}],
    "temperature": 0.0,
}).encode()
req = urllib.request.Request(
    API_URL, data=body,
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open(req, timeout=60) as f:
        d = json.load(f)
        print(json.dumps(d, indent=2)[:600])
except Exception as e:
    print('ERR:', type(e).__name__, e)
