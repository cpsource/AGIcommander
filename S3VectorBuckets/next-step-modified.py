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

Output your response as JSON with the following structure:
{{
  "term": "{term}",
  "context": "{context}",
  "reasoning": "Brief explanation of key semantic qualities",
  "vector": [0.1, -0.3, 0.8, ...],
  "dimensions": {{
    "emotional_valence": {{
      "positive": {{"index": 0, "value": 0.8, "explanation": "Strong positive brand perception"}},
      "negative": {{"index": 1, "value": -0.3, "explanation": "Some controversy around pricing"}},
      "neutral": {{"index": 2, "value": 0.1, "explanation": "Limited neutral sentiment"}},
      "intensity": {{"index": 3, "value": 0.7, "explanation": "High emotional engagement"}}
    }},
    "safety_danger": {{
      "safe": {{"index": 4, "value": 0.9, "explanation": "Reputable, established company"}},
      "risky": {{"index": 5, "value": -0.1, "explanation": "Low investment risk"}},
      "harmful": {{"index": 6, "value": -0.1, "explanation": "Not considered harmful"}},
      "beneficial": {{"index": 7, "value": 0.8, "explanation": "Products provide utility"}}
    }},
    "physical_properties": {{
      "size": {{"index": 8, "value": 1.0, "explanation": "Massive global corporation"}},
      "weight": {{"index": 9, "value": 0.0, "explanation": "Not applicable to company"}},
      "hardness": {{"index": 10, "value": 0.0, "explanation": "Not applicable to company"}},
      "temperature": {{"index": 11, "value": 0.0, "explanation": "Not applicable to company"}}
    }},
    "temporal_aspects": {{
      "permanent": {{"index": 12, "value": 0.8, "explanation": "Long-established company"}},
      "temporary": {{"index": 13, "value": -0.9, "explanation": "Not temporary entity"}},
      "old": {{"index": 14, "value": 0.6, "explanation": "Founded in 1976, mature company"}},
      "new": {{"index": 15, "value": 0.9, "explanation": "Constantly innovating with new products"}}
    }},
    "social_aspects": {{
      "public": {{"index": 16, "value": 1.0, "explanation": "Major public corporation"}},
      "private": {{"index": 17, "value": -1.0, "explanation": "Not a private company"}},
      "collaborative": {{"index": 18, "value": 0.6, "explanation": "Some collaborative features in products"}},
      "individual": {{"index": 19, "value": 0.8, "explanation": "Focus on individual consumer experience"}}
    }},
    "cognitive_load": {{
      "simple": {{"index": 20, "value": 0.8, "explanation": "User-friendly product design"}},
      "complex": {{"index": 21, "value": 0.9, "explanation": "Complex internal operations and technology"}},
      "familiar": {{"index": 22, "value": 1.0, "explanation": "Globally recognized brand"}},
      "abstract": {{"index": 23, "value": 0.4, "explanation": "Concrete company with tangible products"}}
    }},
    "economic_value": {{
      "expensive": {{"index": 24, "value": 1.0, "explanation": "Premium pricing strategy"}},
      "cheap": {{"index": 25, "value": -1.0, "explanation": "Opposite of cheap"}},
      "valuable": {{"index": 26, "value": 1.0, "explanation": "One of world's most valuable companies"}},
      "worthless": {{"index": 27, "value": -1.0, "explanation": "Opposite of worthless"}}
    }},
    "power_control": {{
      "powerful": {{"index": 28, "value": 1.0, "explanation": "Immense market influence"}},
      "weak": {{"index": 29, "value": -1.0, "explanation": "Opposite of weak"}},
      "controlling": {{"index": 30, "value": 0.9, "explanation": "Controls ecosystem tightly"}},
      "submissive": {{"index": 31, "value": -1.0, "explanation": "Opposite of submissive"}}
    }}
  }}
}}
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

