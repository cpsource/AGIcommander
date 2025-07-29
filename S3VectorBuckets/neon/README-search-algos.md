I'll search for current information about Neon's vector search algorithms and the pgvector extension they use.Based on my research, Neon uses the **pgvector extension** which currently supports **3 main search algorithms** for vector similarity search. Let me break them down using analogies to help you understand:

## 1. **Sequential Scan (Exact Search)**
**The Library Analogy**: Like checking every single book in a massive library by hand

**How it works**:
- Performs a sequential scan on the database and calculates the distance between the query vector and all vectors in the table
- Examines every vector in your database one by one

**Pros**:
- Guarantees 100% recall (finds the true nearest neighbors)
- Perfect accuracy
- No setup required

**Cons**:
- Can be costly with large datasets
- Time complexity: O(N) where N = dataset size
- Gets exponentially slower as your data grows

**Best for**: Small datasets (under 10k vectors) where perfect accuracy is crucial

---

## 2. **IVFFlat Index (Inverted File with Flat Compression)**
**The Library Analogy**: Like organizing books into sections by topic, then only searching relevant sections

**How it works**:
- Divides the vectors in the table into multiple lists. The algorithm calculates a number of centroids and finds the clusters around those centroids
- Creates k-means centroids and partitions the dataset into clusters (also called lists) to optimize for vector search

**Key Parameters**:
- `lists`: Number of clusters (good starting point: rows/1000 for up to 1M rows and sqrt(rows) for over 1M rows)
- `probes`: How many clusters to search (A good starting point is sqrt(lists))

**Pros**:
- Faster build times and uses less memory than HNSW
- Time complexity: O(sqrt(rows)) typically
- Lower memory requirements

**Cons**:
- Lower query performance with respect to the speed-recall tradeoff
- Requires training step (needs data in table before building index)
- The closest vector to the search vector could correspond to a list that was not selected for probing, thus reducing the accuracy

**Best for**: Applications with memory constraints, medium-sized datasets

---

## 3. **HNSW Index (Hierarchical Navigable Small Worlds)**
**The Library Analogy**: Like a sophisticated GPS navigation system with multiple layers of maps

**How it works**:
- Constructs a multi-layered graph, where each layer is a subset of the previous one. During a vector similarity search, the algorithm navigates through the graph from the top layer to the bottom to quickly find the nearest neighbor
- In each of the HNSW index layers, the vectors are sorted according to the distance function

**Key Parameters**:
- `m`: Number of connections per node (affects graph connectivity)
- `ef_construction`: Size of candidate set during construction
- `ef_search`: Size of candidate set during search

**Pros**:
- Superior performance in terms of speed and accuracy
- Better query performance than an IVFFlat index (in terms of speed-recall tradeoff)
- Can be created without any data in the table since there isn't a training step
- Time complexity: O(log(rows))
- Supports dynamic updates

**Cons**:
- Slower build times and uses more memory
- Creating an HNSW index can be resource-intensive, especially for large datasets
- More complex parameter tuning

**Best for**: Large datasets, production applications requiring high performance

---

## **Performance Comparison Summary**

Think of it like transportation methods:

| Algorithm | Speed | Accuracy | Memory | Build Time | Best Use Case |
|-----------|-------|----------|---------|------------|---------------|
| Sequential | 🐌 Slowest | 🎯 Perfect | 💾 Minimal | ⚡ Instant | Small datasets |
| IVFFlat | 🚗 Good | 📊 Good | 💾 Moderate | ⚡ Fast | Memory-constrained |
| HNSW | 🚀 Fastest | 📊 Excellent | 💾 High | 🐌 Slow | Large production |

## **Real-World Recommendations**

**For learning/prototyping** (like your 100-vector example): Any method works, but include HNSW to learn production patterns.

**For production**:
- **< 10K vectors**: Sequential scan might be fine
- **10K - 100K vectors**: IVFFlat if memory-constrained, HNSW if performance-critical
- **> 100K vectors**: HNSW almost always

The key insight: HNSW indexes have better query performance than IVFFlat (in terms of speed-recall tradeoff) and can be created without any data in the table since there isn't a training step like there is for an IVFFlat index, making HNSW the preferred choice for most modern applications.

