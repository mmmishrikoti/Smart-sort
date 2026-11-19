import json
from pathlib import Path
from collections import Counter

anno_path = Path(r"e:\Hackathons\smart sort\TACO\data\annotations.json")
with open(anno_path, "r", encoding="utf-8") as f:
    data = json.load(f)

cat_by_id = {c["id"]: c for c in data["categories"]}
counts = Counter(ann["category_id"] for ann in data["annotations"])

print("ID | Name | Supercategory | Count")
print("-" * 65)
for cid, c in sorted(cat_by_id.items()):
    cnt = counts.get(cid, 0)
    print(f"{cid:2d} | {c['name']:<28} | {str(c.get('supercategory')):<20} | {cnt}")
