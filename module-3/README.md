# Information Retrieval with Vector Databases

## Approximate Nearest Neighbors (ANN) Algorithms

### Limitations of Basic k-Nearest Neighbors (KNN)

- Vector search often starts with keyword or semantic search, but scaling creates performance challenges.
- **Naive vector search (KNN)**:
  - Compute embeddings for all documents and for the query.
  - Calculate the distance between the query vector and every document vector.
  - Sort results by distance and return the top *k* neighbors.
- **Scalability problem**:
  - KNN has **linear complexity (O(n))**.
  - Examples:
    - 1,000 documents → 1,000 distance calculations.
    - 1 billion documents → 1 billion calculations.
  - Performance slows drastically as the document corpus grows.
- **Conclusion**: KNN is simple but not suitable for large-scale retrieval.

### Approximate Nearest Neighbors (ANN)

- ANN algorithms significantly speed up vector search by allowing small trade-offs in accuracy.
- They do **not guarantee** the absolute nearest neighbors but return vectors that are sufficiently close.
- A common implementation approach is the **Navigable Small World (NSW)** graph.

---

### Navigable Small World (NSW) Algorithm

#### Building the Proximity Graph

- Create a graph where:
  - Each document is a node.
  - Edges connect each node to a few closest neighbors.
- Construction steps:
  1. Compute pairwise distances between all vectors.
  2. Add nodes for each document.
  3. Connect each node to its nearest neighbors.
- Result: A web-like proximity graph that supports fast traversal.

#### Querying the Graph

- Convert the prompt into a query vector.
- Start from a **random entry point** (candidate vector).
- Traversal procedure:
  - Examine neighbors of the current candidate.
  - Move to the neighbor that is closest to the query vector.
  - Repeat until no neighbor is closer.
- The final candidate is returned as the result.
- **Note**:
  - NSW is efficient because each step considers only a limited number of neighbors.
  - It finds locally optimal paths, not guaranteed global optima.

---

### Hierarchical Navigable Small World (HNSW)

#### Motivation

- NSW improves over KNN, but **HNSW** boosts performance further.
- HNSW delivers **logarithmic runtime**, enabling vector search at billion-scale with low latency.

#### Building the Hierarchical Graph

- Construct multiple proximity graph layers:
  - **Layer 1:** All vectors (e.g., 1,000).
  - **Layer 2:** Random subset (e.g., 100).
  - **Layer 3:** Further subset (e.g., 10).
- Each layer has its own proximity structure.

#### Querying with HNSW

1. Begin in the top layer with a random entry point.
2. Search locally to find the best candidate in that layer.
3. Drop down to the next layer using that candidate as the new starting point.
4. Repeat until reaching **Layer 1**, which includes all vectors.
5. The best candidate found in Layer 1 is the final search result.

#### Efficiency Benefits

- Upper layers allow large jumps to reach the approximate region of the solution.
- Lower layers refine the search locally.
- Overall runtime:
  - **HNSW ≈ O(log n)**  
  - **KNN = O(n)**

---

### Key Takeaways

- **KNN does not scale** because its runtime grows linearly with the number of documents.
- **ANN algorithms** (NSW, HNSW) enable fast vector search at scale:
  - Large performance gains.
  - Slight accuracy trade-offs.
- **Proximity graph construction**:
  - Computationally expensive.
  - Performed **offline** before serving queries.
- These algorithms make vector search feasible even for **billions of vectors** with low latency.
