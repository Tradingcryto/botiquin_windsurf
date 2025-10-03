import yaml
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cats = cfg.get("taxonomy", {}).get("categories")
print("categories type:", type(cats).__name__)
assert isinstance(cats, list), "categories has to be a list"
for i, c in enumerate(cats):
    assert isinstance(c, dict) and "name" in c and "keywords" in c, f"bad item at {i}"
    assert isinstance(c["keywords"], list) and c["keywords"], f"empty keywords at {i}"
print("OK")
