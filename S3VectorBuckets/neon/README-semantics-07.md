Looking at our semantic sentence: **"group-inclusive-speaker moved-past-completed bank-money"**

We can extract these semantic triples:

```python
triples = [
    ("group-inclusive-speaker", "performed-action", "moved-past-completed"),
    ("group-inclusive-speaker", "destination", "bank-money"),
    ("moved-past-completed", "target-location", "bank-money"),
    ("group-inclusive-speaker", "temporal-state", "past"),
    ("moved-past-completed", "completion-status", "completed"),
    ("bank-money", "role", "destination"),
    ("group-inclusive-speaker", "cardinality", "multiple"),
    ("group-inclusive-speaker", "includes", "speaker")
]
```

Think of it like extracting all possible relationships from a knowledge graph. Each triple is like an edge in the graph connecting semantic nodes.

The beauty is that our compressed representation actually makes these relationships more explicit! In the original "We went to the bank," the relationships were buried in grammar. Our semantic tokens make them queryable:

```python
# Query examples
if ("group-inclusive-speaker", "cardinality", "multiple"):
    print("Multiple people involved")
    
if ("moved-past-completed", "completion-status", "completed"):
    print("Action is finished")
```

So from 3 semantic tokens, we can extract roughly 8 meaningful triples - that's pretty dense information encoding!

