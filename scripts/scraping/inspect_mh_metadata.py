import json

data = json.load(open('data/metadata/agmarknet_filters.json', encoding='utf-8'))

# Find Maharashtra
mh_states = [s for s in data['state_data'] if 'maharashtra' in s['state_name'].lower()]
print("Maharashtra State:", mh_states)
mh_id = mh_states[0]['state_id'] if mh_states else None

# Find target districts in Maharashtra
target_dist_names = ['pune', 'mumbai', 'nashik', 'nagpur']
target_districts = [d for d in data['district_data'] if d.get('state_id') == mh_id and any(t in d['district_name'].lower() for t in target_dist_names)]
print("\nTarget Districts:", target_districts)

# Find Essential Commodities
target_cmdts = ['rice', 'wheat', 'tomato', 'onion', 'potato', 'arhar', 'tur', 'gram', 'moong', 'urad', 'mustard', 'sunflower', 'sugar', 'milk']
found_cmdts = [c for c in data['cmdt_data'] if any(t in c['cmdt_name'].lower() for t in target_cmdts)]
print(f"\nTarget Commodities Found ({len(found_cmdts)}):")
for c in found_cmdts[:25]:
    print(f"  ID: {c['cmdt_id']} | Name: {c['cmdt_name']}")
