"""
Vector Generation System for Triplet Components
Generates semantic vectors for entities and relationships that include human qualities
"""

VECTOR_GENERATION_PROMPT = """
System: You are an expert at creating semantic vector representations for words, entities, and relationships. Your task is to generate a 32-dimensional vector that captures the semantic meaning of the given term, including human qualities and associations.

Guidelines:
- Each dimension represents a semantic quality or attribute
- Use values between -1.0 and 1.0
- Consider human qualities: good/bad, safe/dangerous, pleasant/unpleasant, etc.
- Include abstract concepts: power, emotion, time, physical properties, etc.
- Be consistent: similar concepts should have similar vectors
- Consider context when provided (e.g., "Apple-company" vs "apple-fruit")

Vector Dimensions (32 total):
1-4: Emotional valence (positive, negative, neutral, intensity)
5-8: Safety/Danger (safe, risky, harmful, beneficial)  
9-12: Physical properties (size, weight, hardness, temperature)
13-16: Temporal aspects (permanent, temporary, old, new)
17-20: Social aspects (public, private, collaborative, individual)
21-24: Cognitive load (simple, complex, familiar, abstract)
25-28: Economic value (expensive, cheap, valuable, worthless)
29-32: Power/Control (powerful, weak, controlling, submissive)

User: Generate a 32-dimensional semantic vector for: "{term}"
Context: {context}

Provide your reasoning for key dimensions, then output the vector as a Python list of 32 floats.

Format:
Reasoning: Brief explanation of key semantic qualities
Vector: [0.1, -0.3, 0.8, ...] (32 values)
"""

def create_vector_generation_query(term, context=""):
    """Generate prompt for creating semantic vectors"""
    return VECTOR_GENERATION_PROMPT.format(term=term, context=context)

def check_missing_vectors(triplets_data, cursor):
    """
    Check which triplet components need vectors generated
    Returns list of missing terms that need vectorization
    """
    missing_terms = []
    
    for triplet in triplets_data['triplets']:
        subject = triplet['subject']
        relationship = triplet['relationship'] 
        object_term = triplet['object']
        
        # Check if subject vector exists
        cursor.execute("""
            SELECT COUNT(*) FROM word_meanings 
            WHERE sense_key = %s AND vector_embedding IS NOT NULL
        """, (subject,))
        
        if cursor.fetchone()[0] == 0:
            missing_terms.append({
                'term': subject,
                'type': 'entity',
                'context': f"Subject in triplet: {subject} -> {relationship} -> {object_term}"
            })
        
        # Check if relationship vector exists
        cursor.execute("""
            SELECT COUNT(*) FROM relationships 
            WHERE relationship_type = %s AND relationship_vector IS NOT NULL
        """, (relationship,))
        
        if cursor.fetchone()[0] == 0:
            missing_terms.append({
                'term': relationship,
                'type': 'relationship',
                'context': f"Relationship connecting {subject} and {object_term}"
            })
        
        # Check if object vector exists
        cursor.execute("""
            SELECT COUNT(*) FROM word_meanings 
            WHERE sense_key = %s AND vector_embedding IS NOT NULL
        """, (object_term,))
        
        if cursor.fetchone()[0] == 0:
            missing_terms.append({
                'term': object_term,
                'type': 'entity', 
                'context': f"Object in triplet: {subject} -> {relationship} -> {object_term}"
            })
    
    return missing_terms

def generate_vectors_for_missing_terms(missing_terms, llm_client):
    """
    Generate vectors for all missing terms using LLM
    """
    generated_vectors = {}
    
    for term_info in missing_terms:
        term = term_info['term']
        context = term_info['context']
        
        print(f"Generating vector for: {term}")
        
        # Create the prompt
        prompt = create_vector_generation_query(term, context)
        
        # Get LLM response
        response = llm_client.generate(prompt)
        
        # Parse the vector from response (assuming it returns the format we specified)
        try:
            # Extract vector from response - you'll need to parse this based on your LLM's output
            vector_line = [line for line in response.split('\n') if line.startswith('Vector:')][0]
            vector_str = vector_line.replace('Vector:', '').strip()
            vector = eval(vector_str)  # Parse the Python list
            
            if len(vector) != 32:
                raise ValueError(f"Vector has {len(vector)} dimensions, expected 32")
                
            generated_vectors[term] = {
                'vector': vector,
                'type': term_info['type'],
                'context': context
            }
            
        except Exception as e:
            print(f"Error parsing vector for {term}: {e}")
            # Fallback: generate random vector (for testing)
            generated_vectors[term] = {
                'vector': [random.uniform(-1, 1) for _ in range(32)],
                'type': term_info['type'],
                'context': context
            }
    
    return generated_vectors

def store_generated_vectors(generated_vectors, cursor, connection):
    """
    Store the generated vectors in the database
    """
    for term, vector_info in generated_vectors.items():
        vector = vector_info['vector']
        term_type = vector_info['type']
        
        if term_type == 'entity':
            # Store in word_meanings table
            word_base = term.split('-')[0] if '-' in term else term
            
            # Ensure the base word exists
            cursor.execute("""
                INSERT INTO words (word) VALUES (%s) 
                ON CONFLICT (word) DO NOTHING
            """, (word_base,))
            
            # Store the meaning with vector
            cursor.execute("""
                INSERT INTO word_meanings (
                    word_id, sense_key, vector_embedding, definition
                ) VALUES (
                    (SELECT word_id FROM words WHERE word = %s),
                    %s, %s, %s
                ) ON CONFLICT (word_id, sense_key) 
                DO UPDATE SET vector_embedding = %s
            """, (word_base, term, vector, f"Generated meaning for {term}", vector))
            
        elif term_type == 'relationship':
            # Store in relationships table
            cursor.execute("""
                INSERT INTO relationships (relationship_type, relationship_vector)
                VALUES (%s, %s)
                ON CONFLICT (relationship_type) 
                DO UPDATE SET relationship_vector = %s
            """, (term, vector, vector))
    
    connection.commit()
    print(f"Stored {len(generated_vectors)} vectors in database")

def process_triplets_with_vector_generation(triplets_data, cursor, connection, llm_client):
    """
    Complete pipeline: check for missing vectors, generate them, store them, then create triplet vectors
    """
    print("Checking for missing vectors...")
    missing_terms = check_missing_vectors(triplets_data, cursor)
    
    if missing_terms:
        print(f"Found {len(missing_terms)} missing vectors")
        for term_info in missing_terms:
            print(f"  - {term_info['term']} ({term_info['type']})")
        
        print("Generating missing vectors...")
        generated_vectors = generate_vectors_for_missing_terms(missing_terms, llm_client)
        
        print("Storing vectors in database...")
        store_generated_vectors(generated_vectors, cursor, connection)
    else:
        print("All required vectors already exist!")
    
    # Now create the complete triplet vectors
    print("Creating triplet vectors...")
    triplet_vectors = create_triplet_vectors(triplets_data, cursor)
    
    return triplet_vectors

def create_triplet_vectors(triplets_data, cursor):
    """
    Retrieve vectors for all triplet components and concatenate them
    """
    complete_triplets = []
    
    for triplet in triplets_data['triplets']:
        subject = triplet['subject']
        relationship = triplet['relationship']
        object_term = triplet['object']
        
        # Get subject vector
        cursor.execute("""
            SELECT vector_embedding FROM word_meanings 
            WHERE sense_key = %s
        """, (subject,))
        subject_vector = cursor.fetchone()[0]
        
        # Get relationship vector  
        cursor.execute("""
            SELECT relationship_vector FROM relationships 
            WHERE relationship_type = %s
        """, (relationship,))
        relationship_vector = cursor.fetchone()[0]
        
        # Get object vector
        cursor.execute("""
            SELECT vector_embedding FROM word_meanings 
            WHERE sense_key = %s
        """, (object_term,))
        object_vector = cursor.fetchone()[0]
        
        # Concatenate into 96-dimensional triplet vector
        triplet_vector = subject_vector + relationship_vector + object_vector
        
        complete_triplets.append({
            'subject': subject,
            'relationship': relationship, 
            'object': object_term,
            'confidence': triplet['confidence'],
            'subject_vector': subject_vector,
            'relationship_vector': relationship_vector,
            'object_vector': object_vector,
            'triplet_vector': triplet_vector
        })
        
        print(f"Created 96-dim vector for: {subject} -> {relationship} -> {object_term}")
    
    return complete_triplets

# Example usage with your processed triplets
if __name__ == "__main__":
    import json
    import random
    
    # Your processed triplets from the JSON file
    triplets_data = {
        "sentence": "Apple released a new iPhone model last year.",
        "triplets": [
            {
                "subject": "Apple-company",
                "relationship": "creates",
                "object": "new_iPhone_model-product",
                "confidence": 0.98,
                "explanation": "The sentence explicitly states that 'Apple' (the company) 'released' (a form of 'creates' or makes available) a 'new iPhone model' (the product)."
            },
            {
                "subject": "new_iPhone_model-product",
                "relationship": "is_a",
                "object": "iPhone-product_line",
                "confidence": 0.92,
                "explanation": "The phrase 'new iPhone model' implies that this entity is a specific type ('is_a') of the broader 'iPhone' product line."
            },
            {
                "subject": "new_iPhone_model-product",
                "relationship": "released_during",
                "object": "last_year-time_period",
                "confidence": 0.95,
                "explanation": "The sentence specifies the timing of the release event, linking the product ('new iPhone model') to the time frame ('last year')."
            }
        ]
    }
    
    # Example of what the vector generation query would look like
    example_terms = ["Apple-company", "creates", "new_iPhone_model-product"]
    
    for term in example_terms:
        print("="*60)
        print(create_vector_generation_query(term, "Technology company context"))
        print("="*60)
        print()
