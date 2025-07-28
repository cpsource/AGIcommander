"""
Sentence Triplet Extraction System for Vector-Based Knowledge Graphs
This system extracts (Subject, Relationship, Object) triplets from sentences
and generates corresponding vector embeddings for semantic search.
"""

TRIPLET_EXTRACTION_PROMPT = """
System: You are an expert at extracting semantic relationships from sentences. Your task is to:

1. Parse the given sentence into triplets of the form (Subject, Relationship, Object)
2. Identify the specific word-meaning for ambiguous words (e.g., "bank-financial" vs "bank-river")
3. Normalize relationships to standard forms (e.g., "is located in", "owns", "is a type of")
4. Return results as structured JSON

Guidelines:
- Extract ALL meaningful relationships, not just the main one
- Use specific word-meanings (e.g., "apple-fruit" not just "apple")
- Standardize relationship types (provide a list of ~20 common relationships)
- Include confidence scores (0.0-1.0) for each triplet
- Handle implicit relationships (e.g., "John's car" implies "John owns car")

Standard Relationship Types:
- is_a, part_of, located_in, owns, works_at, born_in, married_to, 
- causes, prevents, enables, requires, before, after, during,
- likes, dislikes, fears, wants, believes, knows, creates, destroys

User: Extract triplets from this sentence: "{sentence}"

Expected JSON format:
{{
  "sentence": "{sentence}",
  "triplets": [
    {{
      "subject": "word-meaning",
      "relationship": "standard_relationship_type", 
      "object": "word-meaning",
      "confidence": 0.95,
      "explanation": "brief explanation of why this triplet was extracted"
    }}
  ]
}}
"""

def create_sentence_analysis_query(sentence):
    """Generate the complete prompt for sentence triplet extraction"""
    return TRIPLET_EXTRACTION_PROMPT.format(sentence=sentence)

# Example usage queries for the database
VECTOR_SIMILARITY_QUERIES = """
-- Find similar triplets by subject vector
SELECT 
    s1.sentence_id,
    wm_subj.sense_key as subject,
    r.relationship_type,
    wm_obj.sense_key as object,
    s1.subject_meaning_id <=> s2.subject_meaning_id as subject_similarity
FROM sentence_triplets s1
JOIN sentence_triplets s2 ON s1.triplet_id != s2.triplet_id
JOIN word_meanings wm_subj ON s1.subject_meaning_id = wm_subj.meaning_id
JOIN word_meanings wm_obj ON s1.object_meaning_id = wm_obj.meaning_id
JOIN relationships r ON s1.relationship_id = r.relationship_id
WHERE s1.subject_meaning_id IN (
    SELECT meaning_id FROM word_meanings 
    WHERE sense_key = 'bank-financial'
)
ORDER BY subject_similarity
LIMIT 10;

-- Find triplets with similar relationship vectors
SELECT 
    wm1.sense_key as subject,
    r1.relationship_type,
    wm2.sense_key as object,
    r1.relationship_vector <=> $1 as relationship_similarity
FROM sentence_triplets st
JOIN word_meanings wm1 ON st.subject_meaning_id = wm1.meaning_id
JOIN relationships r1 ON st.relationship_id = r1.relationship_id  
JOIN word_meanings wm2 ON st.object_meaning_id = wm2.meaning_id
ORDER BY relationship_similarity
LIMIT 20;

-- Complex query: Find analogous triplets (similar structure, different entities)
WITH target_triplet AS (
    SELECT 
        subject_meaning_id, 
        relationship_id, 
        object_meaning_id,
        triplet_vector
    FROM sentence_triplets 
    WHERE triplet_id = $1
)
SELECT 
    wm_subj.sense_key as analogous_subject,
    r.relationship_type as analogous_relationship,
    wm_obj.sense_key as analogous_object,
    st.triplet_vector <=> t.triplet_vector as structural_similarity,
    st.confidence_score
FROM sentence_triplets st
JOIN target_triplet t ON st.triplet_id != $1
JOIN word_meanings wm_subj ON st.subject_meaning_id = wm_subj.meaning_id
JOIN relationships r ON st.relationship_id = r.relationship_id
JOIN word_meanings wm_obj ON st.object_meaning_id = wm_obj.meaning_id
ORDER BY structural_similarity
LIMIT 15;
"""

# Python function to process a sentence
def process_sentence_to_triplets(sentence, llm_client, vector_client):
    """
    Complete pipeline: sentence -> triplets -> vectors -> database storage
    """
    
    # Step 1: Extract triplets using LLM
    prompt = create_sentence_analysis_query(sentence)
    response = llm_client.generate(prompt)
    triplets_json = json.loads(response)
    
    # Step 2: Generate vectors for each component
    processed_triplets = []
    
    for triplet in triplets_json['triplets']:
        # Generate 32-dim vectors for each component
        subject_vector = vector_client.embed(triplet['subject'])
        relation_vector = vector_client.embed(triplet['relationship']) 
        object_vector = vector_client.embed(triplet['object'])
        
        # Concatenate vectors for full triplet representation
        triplet_vector = np.concatenate([
            subject_vector, relation_vector, object_vector
        ])
        
        processed_triplets.append({
            'subject': triplet['subject'],
            'relationship': triplet['relationship'],
            'object': triplet['object'],
            'confidence': triplet['confidence'],
            'subject_vector': subject_vector,
            'relationship_vector': relation_vector,
            'object_vector': object_vector,
            'triplet_vector': triplet_vector
        })
    
    return processed_triplets

# Database insertion query
INSERT_TRIPLET_QUERY = """
WITH sentence_record AS (
    INSERT INTO sentences (sentence_text) 
    VALUES ($1) 
    RETURNING sentence_id
),
subject_meaning AS (
    INSERT INTO word_meanings (word_id, sense_key, vector_embedding)
    VALUES (
        (SELECT word_id FROM words WHERE word = split_part($2, '-', 1)),
        $2, $3
    ) 
    ON CONFLICT (word_id, sense_key) DO UPDATE SET vector_embedding = $3
    RETURNING meaning_id
),
relationship_record AS (
    INSERT INTO relationships (relationship_type, relationship_vector)
    VALUES ($4, $5)
    ON CONFLICT (relationship_type) DO UPDATE SET relationship_vector = $5
    RETURNING relationship_id
),
object_meaning AS (
    INSERT INTO word_meanings (word_id, sense_key, vector_embedding)
    VALUES (
        (SELECT word_id FROM words WHERE word = split_part($6, '-', 1)),
        $6, $7
    )
    ON CONFLICT (word_id, sense_key) DO UPDATE SET vector_embedding = $7
    RETURNING meaning_id
)
INSERT INTO sentence_triplets (
    sentence_id, subject_meaning_id, relationship_id, 
    object_meaning_id, confidence_score, triplet_vector
)
SELECT 
    s.sentence_id,
    sm.meaning_id,
    r.relationship_id,
    om.meaning_id,
    $8, -- confidence
    $9  -- triplet_vector
FROM sentence_record s, subject_meaning sm, relationship_record r, object_meaning om
RETURNING triplet_id;
"""

# Example usage
if __name__ == "__main__":
    # Test sentences
    test_sentences = [
        "The bank on the river processes financial transactions.",
        "Apple released a new iPhone model last year.",
        "John drives his red car to work every morning.",
        "The doctor prescribed medicine for the patient's headache."
    ]
    
    for sentence in test_sentences:
        print("="*50)
        print(f"Sentence: {sentence}")
        print("="*50)
        print(create_sentence_analysis_query(sentence))
        print()
