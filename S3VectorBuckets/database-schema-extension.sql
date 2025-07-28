-- Extend your existing schema
CREATE TABLE relationships (
    relationship_id SERIAL PRIMARY KEY,
    relationship_type TEXT NOT NULL, -- 'located_in', 'is_a', 'owns', etc.
    relationship_vector VECTOR(32)
);

CREATE TABLE sentence_triplets (
    triplet_id SERIAL PRIMARY KEY,
    sentence_id INTEGER,
    subject_meaning_id INTEGER REFERENCES word_meanings(meaning_id),
    relationship_id INTEGER REFERENCES relationships(relationship_id),
    object_meaning_id INTEGER REFERENCES word_meanings(meaning_id),
    confidence_score FLOAT,
    triplet_vector VECTOR(96), -- Concatenated A+R+B vectors (32*3)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

