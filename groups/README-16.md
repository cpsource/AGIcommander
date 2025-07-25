In the triple **(h, r, t)**, these stand for:

- **h** = **head entity** (the subject)
- **r** = **relation** (the relationship/predicate) 
- **t** = **tail entity** (the object)

This is the standard knowledge graph representation format. Think of it like a sentence structure:

**Examples from the paper:**
- **(Methane, Contains Element, Carbon)** 
  - h = Methane (subject)
  - r = Contains Element (relationship)
  - t = Carbon (object)

- **(Aspirin, may-treat, Myocardial Infarction)**
  - h = Aspirin
  - r = may-treat  
  - t = Myocardial Infarction

**Why this matters for your semantic approach:**

This triple format is **exactly** what you called "axioms" in our earlier discussion! Each triple is a fundamental semantic fact that can be:

1. **Represented as coordinates** in your semantic space
2. **Inherited hierarchically** (if Aspirin treats heart attacks, and heart attacks are cardiovascular events, then Aspirin relates to cardiovascular treatment)
3. **Used for opposite relationships** (if Aspirin treats conditions, then toxins might cause the same conditions)
4. **Composed into paths** for complex reasoning

**The paper's key insight:** They traverse **paths** of these triples:
- (Methane, Contains Bond, C-H Bond) → (C-H Bond, Is Type Of, Sigma Bond) → (Sigma Bond, Has Property, Single Covalent Bond)

**Your innovation:** Instead of just traversing these as discrete graph steps, you're embedding them as **geometric relationships** in coordinate space, where semantic similarity = spatial proximity.

So h-r-t triples are the "atoms" of structured knowledge that both their system and your semantic architecture use as building blocks!

