Great observation! "We" has many possible replacements that preserve the core meaning. Let's think of this like a substitution cipher where multiple keys map to the same semantic value:

```python
# All these preserve the entropy of "multiple people went to bank"
we_substitutes = [
    "We", "Us", "Together", "Both", "All", "Everyone", 
    "The group", "The team", "The family", "The friends",
    "John and I", "My wife and I", "The kids and I"
]

# Even pronouns with context
contextual_substitutes = [
    "They",  # if previously established who "they" refers to
    "Those two",  # with pointing/context
    "The couple"  # if context established
]
```

But here's the key insight: the entropy isn't just in the word choice - it's in the **cardinality** (how many people) and **identity** (who specifically). 

Think of it like variables in Python:
```python
# High-level abstraction (preserves entropy)
group = ["person1", "person2", "..."]  # multiple people
group.went("bank-money")

# The specific implementation can vary:
we.went("bank-money")
us.went("bank-money") 
together.went("bank-money")
```

You could probably substitute dozens of terms and keep the core entropy: "multiple people performed this action." The constraint is that they all must encode "more than one person, including the speaker."

