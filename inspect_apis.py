import httpx
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

client = httpx.Client(timeout=10)
r = client.get('http://192.168.65.157:3002/api/v1/api-directory')
data = r.json()

with open('api_directory.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Total endpoints: {data.get('total_endpoints')}")
for ep in data.get('endpoints', []):
    print(f"\n{'='*60}")
    print(f"{ep['method']} {ep['path']} [{ep.get('authBadge')}]")
    print(f"Category: {ep.get('category')} | Pacing: {ep.get('pacing')}")
    print(f"Description: {ep.get('description')}")
    if ep.get('request_example'):
        print(f"Request: {json.dumps(ep['request_example'], indent=2)}")
    if ep.get('response_example'):
        print(f"Response: {json.dumps(ep['response_example'], indent=2)}")
