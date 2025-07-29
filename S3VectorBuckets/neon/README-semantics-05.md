Perfect! Let's create a semantic token that captures the essence:

```python
# Semantic token for "multiple people including speaker"
semantic_token = "group-inclusive-speaker"

# Our compressed sentence
tokens = ["group-inclusive-speaker", "went", "bank-money"]
```

This token encodes the core semantic information:
- **group**: more than one person
- **inclusive**: includes the speaker 
- **speaker**: the person telling the story

Just like how you created `bank-money` to disambiguate from other meanings of "bank", `group-inclusive-speaker` eliminates ambiguity about who performed the action.

Think of it like defining a class in Python:
```python
class GroupInclusiveSpeaker:
    def __init__(self):
        self.count = "multiple"
        self.includes_narrator = True
        self.perspective = "first_person"
```

Now our ultra-compressed sentence becomes:
**"group-inclusive-speaker went bank-money"**

Three tokens, maximum entropy, zero ambiguity. Each token is semantically atomic and carries distinct, non-redundant information. It's like the most efficient possible encoding of the original meaning!

