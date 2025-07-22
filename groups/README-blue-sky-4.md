You're thinking like a true systems architect! Each approach has its place, and I think the **combination approach** is brilliant because different types of conflicts need different resolution strategies.

**Conflict Type → Resolution Strategy:**

**Minor Disagreements (75% vs 85% confidence):**
- Split the difference with decay weighting
- Network says 75%, DB says 85% → average to 80% but trend toward network over time
- Let empirical learning gradually override theoretical knowledge

**Major Conflicts (20% vs 90% confidence):**
- Fork the database! Create parallel semantic paths
- Path A: "classical_semantics" (follows original DB)
- Path B: "learned_semantics" (follows network discoveries)  
- Train both and see which performs better in practice

**Novel Discoveries (network finds 95% relationship that DB doesn't have):**
- Log to operator with evidence strength
- Temporary "pending" status with lower confidence
- If consistently validated across batches → promote to full relationship

**Domain-Specific Conflicts:**
- Medical: bias toward established knowledge (safety critical)
- Creative domains: bias toward network discoveries (innovation valued)
- Scientific: require human expert review

**The Combination Strategy:**
```
if confidence_gap < 15%: split_difference()
elif domain == "safety_critical": bias_toward_database()  
elif novel_relationship: log_and_validate()
elif repeated_conflict: fork_database()
else: decay_database_confidence()
```

This creates a **learning semantic database** that evolves with your network! The conflicts become valuable signals about where human knowledge might be incomplete or domain-specific.

What's your intuition about the confidence thresholds? And do you think certain relationship types (opposites vs siblings vs parents) should have different conflict resolution rules?

