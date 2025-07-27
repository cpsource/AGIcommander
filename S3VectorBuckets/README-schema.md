You're describing a classic word sense disambiguation problem! Here's a SQL schema that would work well for this:

## Core Schema Design

```sql
-- Main words table
CREATE TABLE words (
    word_id SERIAL PRIMARY KEY,
    word TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Word meanings/senses table
CREATE TABLE word_meanings (
    meaning_id SERIAL PRIMARY KEY,
    word_id INTEGER REFERENCES words(word_id),
    sense_key TEXT NOT NULL,  -- e.g., 'bank-financial', 'bank-river'
    definition TEXT,
    example_sentence TEXT,
    vector_embedding VECTOR(1536),  -- PostgreSQL with pgvector
    frequency_rank INTEGER,  -- how common this meaning is
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(word_id, sense_key)
);

-- Optional: Categories/domains for meanings
CREATE TABLE meaning_categories (
    category_id SERIAL PRIMARY KEY,
    category_name TEXT UNIQUE -- 'finance', 'geography', 'technology'
);

CREATE TABLE meaning_category_links (
    meaning_id INTEGER REFERENCES word_meanings(meaning_id),
    category_id INTEGER REFERENCES meaning_categories(category_id),
    PRIMARY KEY (meaning_id, category_id)
);
```

## Example Data for "Bank"

```sql
-- Insert the word
INSERT INTO words (word_id, word) VALUES (1, 'bank');

-- Insert the 9 meanings
INSERT INTO word_meanings (word_id, sense_key, definition, frequency_rank) VALUES
(1, 'bank-financial', 'A financial institution that accepts deposits', 1),
(1, 'bank-river', 'The land alongside a river or stream', 2),
(1, 'bank-storage', 'A storage facility (blood bank, data bank)', 3),
(1, 'bank-slope', 'A slope or inclined surface', 4),
(1, 'bank-aircraft', 'To tilt an aircraft during a turn', 5),
(1, 'bank-rely', 'To depend or rely on something', 6),
(1, 'bank-pool', 'A fund or pool of money in gambling', 7),
(1, 'bank-arrange', 'To arrange things in rows or tiers', 8),
(1, 'bank-fire', 'To cover a fire to make it burn slowly', 9);
```

## Query Examples

```sql
-- Get all meanings for 'bank'
SELECT w.word, wm.sense_key, wm.definition 
FROM words w 
JOIN word_meanings wm ON w.word_id = wm.word_id 
WHERE w.word = 'bank'
ORDER BY wm.frequency_rank;

-- Find the most common meaning
SELECT sense_key, definition 
FROM words w 
JOIN word_meanings wm ON w.word_id = wm.word_id 
WHERE w.word = 'bank' 
ORDER BY frequency_rank 
LIMIT 1;

-- Vector similarity search (find similar meanings)
SELECT sense_key, definition,
       vector_embedding <=> $1 AS similarity
FROM word_meanings wm
JOIN words w ON wm.word_id = w.word_id
WHERE w.word = 'bank'
ORDER BY similarity
LIMIT 3;
```

## Alternative: Simpler Flat Structure

If you want something simpler:

```sql
CREATE TABLE word_senses (
    sense_id SERIAL PRIMARY KEY,
    base_word TEXT NOT NULL,
    sense_key TEXT NOT NULL,  -- 'bank-financial'
    definition TEXT,
    vector_embedding VECTOR(1536),
    frequency_score FLOAT DEFAULT 1.0,
    
    INDEX(base_word),
    UNIQUE(base_word, sense_key)
);
```

## Benefits of This Schema

- **Scalable**: Works for words with any number of meanings
- **Flexible**: Easy to add new meanings or metadata
- **Query-friendly**: Simple joins to get all meanings
- **Vector-ready**: Supports semantic similarity searches
- **Ranked**: Can prioritize common meanings

Think of it like a dictionary where each word points to multiple definition entries, but each definition has its own vector representation for semantic search!

