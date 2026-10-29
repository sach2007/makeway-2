import httpx
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

base = 'http://192.168.65.157:3003'

# Fetch JS bundle
try:
    r = httpx.get(base + '/assets/index-CKKmFUrH.js', timeout=15)
    print(f'JS Bundle: {r.status_code}, length={len(r.text)}')
    
    # Search for URL patterns - API base URLs
    urls = re.findall(r'https?://[0-9.]+:\d+[^\s"\',;)]*', r.text)
    print('\nURLs found:')
    for u in sorted(set(urls)):
        print(f'  {u}')
    
    # Search for /api patterns
    apis = re.findall(r'/api/v\d+/[\w/.-]+', r.text)
    print('\nAPI routes found:')
    for a in sorted(set(apis)):
        print(f'  {a}')
    
    # Search for localhost or IP patterns
    hosts = re.findall(r'(?:localhost|192\.168\.\d+\.\d+):\d+', r.text)
    print('\nHost:port patterns:')
    for h in sorted(set(hosts)):
        print(f'  {h}')

    # Look for "baseURL", "API_URL", "BASE_URL" etc
    base_urls = re.findall(r'(?:baseURL|BASE_URL|API_URL|apiUrl|baseUrl|api_base)["\s:=]+["\']([^"\']+)["\']', r.text, re.IGNORECASE)
    print('\nBase URL configs:')
    for b in base_urls:
        print(f'  {b}')

    # Search for fetch patterns
    fetches = re.findall(r'fetch\(["`]([^"`]+)["`]', r.text)
    print('\nFetch calls:')
    for f in sorted(set(fetches)):
        print(f'  {f}')

    # Save the bundle
    with open('bundle.js', 'w', encoding='utf-8') as f:
        f.write(r.text)
    print(f'\nSaved bundle ({len(r.text)} bytes)')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
