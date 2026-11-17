import re
import json

text = open('data/metadata/ceda_swagger_init.js', encoding='utf-8').read()
# Find swaggerDoc
start = text.find('"swaggerDoc":')
if start != -1:
    sub = text[start + len('"swaggerDoc":'):]
    end = sub.find(',\n  "customOptions"')
    if end == -1:
        end = sub.find(',\r\n  "customOptions"')
    if end == -1:
        end = sub.find('"customOptions"')
        end = sub.rfind(',', 0, end)
    raw_json = sub[:end].strip()
    spec = json.loads(raw_json)
    print("=== CEDA API SPECIFICATION ===")
    print("Title:", spec.get("info", {}).get("title"))
    print("Version:", spec.get("info", {}).get("version"))
    print("Description:", spec.get("info", {}).get("description"))
    print("Servers:", spec.get("servers"))
    print("\nPaths / Endpoints:")
    for path, methods in spec.get("paths", {}).items():
        for m, details in methods.items():
            print(f"  [{m.upper()}] {path}")
            print(f"    Summary: {details.get('summary', '')}")
            params = [f"{p.get('name')} ({p.get('in')}, req={p.get('required', False)})" for p in details.get('parameters', [])]
            print(f"    Params: {', '.join(params)}")
            print(f"    Security: {details.get('security', spec.get('security', 'None'))}")
    
    with open('data/metadata/ceda_api_spec.json', 'w', encoding='utf-8') as f:
        json.dump(spec, f, indent=2)
    print("\nSaved parsed spec to data/metadata/ceda_api_spec.json")
else:
    print("swaggerDoc not found")
