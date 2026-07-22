import requests, re

with open('examples/input.png','rb') as f:
    r = requests.post('http://localhost:8765/api/vectorize', 
        files={'image':f}, 
        data={'maxColors':12,'detail':5,'engine':'vtracer','separateText':'true'})
svg = r.json()['svg']

print(f"Text layer: {'text-layer' in svg}")
print(f"Shape layer: {'shape-layer' in svg}")
print(f"Text regions: {r.json().get('textRegions', '?')}")
print(f"Total SVG size: {len(svg)} chars")

layers = re.findall(r'<!--.*?-->', svg)
for l in layers:
    print(f"  {l}")
