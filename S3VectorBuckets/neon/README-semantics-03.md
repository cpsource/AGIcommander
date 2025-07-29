Absolutely! "to" is another low-entropy token that's highly predictable from context.

```python
# Further compression
tokens = ["We", "went", "bank-money"]
```

"We went bank-money" still carries the full meaning. The directional relationship is implied by the verb "went" - it's like how in many programming languages you can have implicit operations:

```python
# Explicit
user.go_to(location)

# Implicit - the relationship is understood
user.went(location)
```

Think of it like lossy compression in image processing - you're removing the "noise" (predictable elements) while keeping the signal (meaningful content). 

Many languages actually work this way naturally. Russian, for example, often drops articles and prepositions because case endings on nouns carry that grammatical information. It's like having a more efficient encoding scheme.

The core semantic triple "We → went → bank-money" captures the essential information: who performed the action, what action, and the destination. Everything else was just grammatical scaffolding that context makes redundant.
