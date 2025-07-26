**Yes, UUID4 is virtually guaranteed to be unique on each call** - but let me explain the math and practical considerations since you're asking about guarantees.

## The Math Behind UUID4 Uniqueness

UUID4 generates a **128-bit random number**, which gives you:

```python
import uuid
import math

# Total possible UUID4 values
total_uuids = 2**122  # 122 bits are random (6 bits are fixed for version/variant)
print(f"Total possible UUIDs: {total_uuids:,}")
# 5,316,911,983,139,663,491,615,228,241,121,378,304

# That's about 5.3 × 10^36 unique values!
```

## Collision Probability (Python Analogy)

Think of it like the **birthday paradox**, but with an enormous room:

```python
def collision_probability(num_generated, total_possible):
    """Approximate probability of collision"""
    return 1 - math.exp(-(num_generated**2) / (2 * total_possible))

# Even if you generate 1 billion UUIDs:
billion_uuids = 1_000_000_000
total_uuid4_space = 2**122

collision_prob = collision_probability(billion_uuids, total_uuid4_space)
print(f"Collision probability with 1B UUIDs: {collision_prob:.2e}")
# About 9.4 × 10^-11 (0.000000000094%)
```

**Real-world perspective**: You'd need to generate about **2.7 billion UUIDs per second for 100 years** to have even a 50% chance of one collision.

## PostgreSQL's Additional Safety

PostgreSQL adds another layer of protection with the PRIMARY KEY constraint:

```sql
CREATE TABLE vector_documents (
    id UUID PRIMARY KEY,  -- Database enforces uniqueness
    -- ...
);
```

If somehow a collision occurred, PostgreSQL would reject the insert:

```python
try:
    # If by some miracle a duplicate UUID is generated
    doc = store.store_document(
        content=content,
        embedding=embedding,
        metadata=metadata,
        doc_id=duplicate_uuid  # Hypothetical duplicate
    )
except psycopg2.IntegrityError as e:
    # PostgreSQL would catch this and raise an error
    print("Duplicate UUID detected by database!")
    # Generate a new UUID and retry
```

## Practical Implementation

Here's a robust approach that handles the theoretical edge case:

```python
import uuid
import psycopg2
from typing import Optional

class SafeVectorStore:
    def store_document(self, content, embedding, metadata, doc_id=None, max_retries=3):
        """Store document with collision-safe UUID generation"""
        
        for attempt in range(max_retries):
            # Generate UUID if not provided
            if doc_id is None:
                doc_id = str(uuid.uuid4())
            
            try:
                # Attempt to insert
                with self.conn.cursor() as cur:
                    cur.execute(f"""
                        INSERT INTO {self.table_name} 
                        (id, s3_key, embedding, metadata, content_hash, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (doc_id, s3_key, embedding, json.dumps(metadata), content_hash, datetime.now()))
                
                # Success! Return the document
                return VectorDocument(id=doc_id, ...)
                
            except psycopg2.IntegrityError as e:
                if "duplicate key" in str(e).lower() and attempt < max_retries - 1:
                    print(f"UUID collision detected (attempt {attempt + 1}), retrying...")
                    doc_id = None  # Generate new UUID on retry
                    continue
                else:
                    raise e
        
        raise RuntimeError("Failed to generate unique UUID after maximum retries")
```

## UUID4 vs Other Approaches

```python
# UUID4: Random (what we're using)
uuid4_example = uuid.uuid4()
print(f"UUID4: {uuid4_example}")
# Output: f47ac10b-58cc-4372-a567-0e02b2c3d479

# UUID1: MAC address + timestamp (more predictable)
uuid1_example = uuid.uuid1()
print(f"UUID1: {uuid1_example}")
# Output: 6ba7b810-9dad-11d1-80b4-00c04fd430c8

# Benefits of UUID4:
# ✅ No machine info leaked
# ✅ Unpredictable sequence
# ✅ Good for distributed systems
# ✅ Privacy-safe
```

## Real-World Examples

**Major systems using UUID4:**
- **AWS S3**: Object keys often use UUID4
- **GitHub**: Issue and PR IDs
- **MongoDB**: ObjectIds (similar concept)
- **Web APIs**: Session tokens, API keys

```python
# Example: Generate 1 million UUIDs and check for duplicates
generated_uuids = set()
collisions = 0

for i in range(1_000_000):
    new_uuid = str(uuid.uuid4())
    if new_uuid in generated_uuids:
        collisions += 1
        print(f"Collision found! {new_uuid}")
    generated_uuids.add(new_uuid)

print(f"Generated {len(generated_uuids)} unique UUIDs")
print(f"Collisions: {collisions}")
# Expected output: 1,000,000 unique UUIDs, 0 collisions
```

## Industry Standards

The **RFC 4122 specification** for UUID4 is used by:
- Python's `uuid` module
- Java's `UUID.randomUUID()`
- JavaScript's `crypto.randomUUID()`
- PostgreSQL's `gen_random_uuid()`

All generate cryptographically strong random UUIDs with the same collision resistance.

## Bottom Line

**Yes, `str(uuid.uuid4())` is guaranteed to be unique for all practical purposes.**

The probability of collision is so astronomically low that:
- No production system has ever reported a UUID4 collision
- You're more likely to win the lottery 4 times in a row
- Hardware failure is millions of times more likely

**However**, if you're building a mission-critical system, the safest approach is:
1. Use UUID4 (as you are)
2. Let PostgreSQL's PRIMARY KEY constraint catch theoretical duplicates
3. Implement retry logic (like the example above)

This gives you both the mathematical near-impossibility of collision AND a safety net for the 0.000000000000000001% chance.

