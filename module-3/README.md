# Information Retrieval with Vector Databases

## Approximate Nearest Neighbors (ANN) Algorithms

### 1. Limitations of Basic k-Nearest Neighbors (KNN)

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

### 2. Approximate Nearest Neighbors (ANN)

- ANN algorithms significantly speed up vector search by allowing small trade-offs in accuracy.
- They do **not guarantee** the absolute nearest neighbors but return vectors that are sufficiently close.
- A common implementation approach is the **Navigable Small World (NSW)** graph.



### 3. Navigable Small World (NSW) Algorithm

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



### 4. Hierarchical Navigable Small World (HNSW)

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



### 5. Key Takeaways

- **KNN does not scale** because its runtime grows linearly with the number of documents.
- **ANN algorithms** (NSW, HNSW) enable fast vector search at scale:
  - Large performance gains.
  - Slight accuracy trade-offs.
- **Proximity graph construction**:
  - Computationally expensive.
  - Performed **offline** before serving queries.
- These algorithms make vector search feasible even for **billions of vectors** with low latency.


## Vector Databases

### 1. What Is a Vector Database?

- A database built specifically to store high-dimensional vectors and support vector-based algorithms (e.g., ANN, HNSW).
- Became popular due to the rise of large language models and embedding-based techniques (e.g., Semantic Search).
- Standard relational databases perform poorly for vector search, behaving similarly to slow KNN.
- Vector databases efficiently:
  - Build proximity graphs for ANN/HNSW.
  - Compute vector distances at scale.
  - Power high-throughput RAG applications.


### 2. Weaviate Overview

- Weaviate is the vector database used in the course.
- Open-source; can run locally or in the cloud.
- Other vector DBs offer similar capabilities—the concepts transfer across platforms.


### 3. Preparing a Vector Database for RAG

Common setup steps:

1. **Initialize the database** (create or connect to an instance).
2. **Create a collection** to store data:
   - Define fields (e.g., title, body).
   - Specify data types.
   - Choose an embedding model/vectorizer for semantic vectors.
3. **Load data**:
   - Insert documents using batch operations.
   - Track errors during ingestion.
4. **Create vectors**:
   - Sparse vectors for keyword search.
   - Dense embeddings for semantic search.
5. **Build ANN indexes** (e.g., HNSW) to enable fast retrieval.


### 4. Search Operations in Weaviate

- **Vector Search**:
  - Pass a text query; database returns closest vectors.
  - Metadata (e.g., distance) can be included.

- **Keyword Search**:
  - Uses an automatically created inverted index.
  - Supports BM25 queries for lexical matching.

- **Hybrid Search**:
  - Runs vector and keyword search in parallel.
  - Uses **alpha** to weight results:
    - `alpha = 0.25` → 25% vector search, 75% keyword search.
  - Common in production because it balances semantic and exact matching.

- **Filtering**:
  - Apply conditions on document fields.
  - Only documents whose fields match the filter criteria are returned.


### 5. End-to-End RAG Workflow with a Vector Database

1. Configure the database.
2. Load and index documents.
3. Generate sparse and dense vectors.
4. Build ANN indexes.
5. Execute retrieval queries (vector, keyword, hybrid) with optional filters.

This forms the backbone of search and retrieval in production RAG systems.

---

## Chunking

### 1. Why Chunking Matters
- Embedding models have input-length limits; long documents cannot be embedded as-is.
- Chunking improves search relevance by giving vectors more focused semantic meaning.
- Prevents overloading the LLM’s context window with unnecessarily large retrieved texts.

### 2. Problems Without Chunking
- A single vector per long document (e.g., a full book) compresses too much information:
  - Poor semantic specificity.
  - Irrelevant or overly broad retrieval.
- Retrieval would return entire documents, consuming LLM context capacity quickly.

### 3. Benefits of Chunking
- Converts large documents into many small, semantically meaningful chunks.
- Vector databases can scale to millions of chunks efficiently.
- Improves retrieval precision and LLM performance.

### 4. Choosing a Chunk Size
- **Too large (e.g., chapters):**
  - Loses nuance; still too broad for accurate retrieval.
  - Uses too much LLM context.
- **Too small (e.g., words or single sentences):**
  - Insufficient context → weak semantic vectors.
- **No universal best size:** need to balance context richness and specificity.

### 5. Fixed-Size Chunking
- Predetermined chunk length (e.g., 250 or 500 characters).
- Simple and consistent across documents.
- May split in unnatural locations (mid-word or mid-thought).

#### Overlapping Chunks
- Chunks overlap (e.g., 10%–20%) to preserve context across boundaries.
- Improves search relevance by ensuring edge words retain surrounding context.
- Increases database size with some redundancy.

### 6. Recursive / Structure-Aware Chunking
- Splits based on characters or structural markers:
  - Newlines → paragraphs
  - HTML tags → sections
  - Code → function definitions
- Produces variable chunk sizes aligned with natural document structure.
- Useful when handling diverse document formats.

### 7. Metadata Considerations
- Chunks should inherit metadata from source documents.
- May also include positional metadata (e.g., page, paragraph index).

### 8. Practical Starting Point
- Fixed-size chunks: ~500 characters.
- Overlap: 50–100 characters.
- Adjust based on document style, model performance, and retrieval quality.

### 9. Summary
- Chunking enhances vector retrieval accuracy and LLM efficiency.
- Simple fixed-size approaches work well; structure-aware techniques may provide further improvements depending on the dataset.

---

## Advanced Chunking Techniques

### 1. Limitations of Basic Chunking

- Fixed-size and recursive character splitting can **break context** in text.
  - Example: "That night she dreamed, as she did often, that she was finally an Olympic champion."
    - Poor splitting could imply she is already a champion instead of dreaming.
- These basic methods **ignore semantic meaning** and document structure.


### 2. Semantic Chunking

- **Goal:** Group sentences together if they have similar meaning.
- **Algorithm:**
  1. Move through the document sentence by sentence.
  2. Vectorize the current chunk and the next sentence.
  3. If vector distance < threshold → add sentence to the chunk.
  4. If distance ≥ threshold → create new chunk and restart.
- **Benefits:**
  - Creates variable-sized chunks that follow the author's train of thought.
  - Preserves related ideas across paragraphs.
- **Drawbacks:**
  - Computationally expensive (vectors for every sentence).


### 3. LLM-Based Chunking

- **Goal:** Use a language model to split text into meaningful chunks.
- **Method:**
  - Provide document + instructions (e.g., keep similar concepts together, split on new topics).
  - LLM generates chunks according to meaning.
- **Advantages:**
  - High-quality, context-aware chunks.
  - Can improve retrieval precision and relevance.
- **Considerations:**
  - Expensive in terms of computation.
  - Somewhat of a "black box" approach.


### 4. Context-Aware Chunking

- **Goal:** Add contextual summaries to chunks for better comprehension.
- **Example:**
  - Chunk contains a list of names at the end of a blog post.
  - LLM adds context to explain the chunk’s place in the overall document.
- **Benefits:**
  - Enhances vector representation for search.
  - Helps LLMs understand the chunk when retrieved.
- **Cost:** Pre-processing is computationally heavy.


### 5. Practical Recommendations

- **Default approaches:** Fixed-size or recursive character splitting — good for prototyping.
- **Higher-quality approaches:** Semantic or LLM-based chunking — improve search relevance but expensive.
- **Context-aware enhancement:** Can be applied on top of any strategy for better retrieval.
- **Design principle:** Choose chunking method based on:
  - Nature of your data.
  - Trade-offs between cost and relevance.
  - Maintainability and auditability of the system.


**Summary:** 

Chunking is essential in RAG systems to balance **search relevance** and **LLM context limits**. Start simple, then explore semantic or LLM-based strategies for higher performance where necessary.

---

## Query Parsing

### 1. Importance of Query Parsing

- Users interact with RAG systems in natural conversational language.
- Human-written prompts are often **unsuitable for retrieval**.
- Parsing and rewriting queries improves:
  - Retrieval relevance
  - Matching with knowledge base documents

---

### 2. Basic Query Rewriting

- Use an LLM to **rewrite user queries** before searching.
- Example workflow:
  1. Receive user prompt.
  2. Clarify ambiguous phrases.
  3. Add relevant terminology or synonyms.
  4. Remove unnecessary or distracting info.
- **Example:**
  - User prompt:  
    _"I was out walking my dog, a beautiful black lab named Poppy… shoulder is still numb and my fingers are all pins and needles. What's going on?"_
  - Rewritten query:  
    _"Experienced a sudden forceful pull on the shoulder resulting in persistent shoulder numbness and finger numbness for three days. What are the potential causes or diagnoses such as neuropathy or nerve impingement?"_
- Benefits: Clearer, more effective search queries.
- Costs: Extra LLM call per query.

---

### 3. Advanced Query Parsing Techniques

#### a) Named Entity Recognition (NER)

- Identifies **categories of information** in the query:
  - People, places, dates, books, characters, etc.
- Output can be used to:
  - Refine vector search
  - Apply metadata filtering
- Example: Using a model like **Gliner** to label entities in a query.
- Pros: Improves retrieval relevance.  
- Cons: Adds some latency.

#### b) Hypothetical Document Embeddings (HIDE)

- Generate a **hypothetical ideal document** for the query using an LLM.
- Steps:
  1. LLM generates a "perfect" document representing the intended result.
  2. Embed this hypothetical document into a vector.
  3. Use this vector for retrieval.
- Benefits: Matches more similar text, improving retrieval quality.  
- Costs: Extra latency and computational resources.

---

### 4. Practical Recommendations

- **Basic query rewriting** is usually sufficient and cost-effective.
- **Advanced techniques** (NER, HIDE) can improve results but are more complex and computationally expensive.
- Experimentation is key — choose techniques based on your data and performance needs.

---
**Summary:**  
Query parsing improves RAG system performance by transforming messy user prompts into clear, search-optimized queries. Start simple with LLM-based rewriting, and explore advanced techniques if needed.


## Cross-encoders and ColBERT

### 1. Bi-Encoder (Vanilla Semantic Search)

- **Architecture:**  
  - Documents and prompts embedded separately into single vectors.
  - Approximate nearest neighbor (ANN) search identifies similar documents.
- **Advantages:**  
  - Documents can be pre-embedded → fast retrieval.
  - Minimal vector storage.
- **Disadvantages:**  
  - May miss deeper contextual interactions between prompt and document.

---

### 2. Cross Encoder

- **Architecture:**  
  - Concatenates prompt and document, passes through a specialized model.
  - Outputs a relevancy score (0–1) directly.
- **Example:**  
  - Prompt: "Great places to eat in New York"  
  - Each document + prompt pair scored for relevance.
- **Advantages:**  
  - Captures deep contextual interactions → higher quality results.
- **Disadvantages:**  
  - Poor scalability: must score every prompt-document pair.
  - Cannot pre-compute embeddings.
  - Too slow for default search over millions/billions of documents.

---

### 3. ColBERT (Contextualized Late Interaction over BERT)

- **Architecture:**  
  - Document and prompt tokenized → each token embedded separately.
  - Document vectors precomputed; prompt vectors computed at query time.
  - Scoring: each prompt token finds most similar document token → max-sim scores summed.
- **Example:**  
  - Prompt tokens "New" + "York" matched to document tokens "New" + "York City".
- **Advantages:**  
  - Balances **speed** of bi-encoder with **rich interactions** of cross-encoder.
  - Suitable for near real-time search.
- **Disadvantages:**  
  - Large vector storage: one vector per token.
  - More computationally intensive than bi-encoder.

---

### 4. Summary of Trade-Offs

| Architecture    | Speed       | Quality         | Storage Requirements |
|-----------------|------------|----------------|--------------------|
| Bi-Encoder      | Fast       | Moderate       | Minimal            |
| Cross Encoder   | Slow       | High           | Minimal            |
| ColBERT         | Moderate   | High           | High               |

- **Default:** Bi-encoder → good speed, minimal storage.
- **Gold standard:** Cross-encoder → best quality, too slow for large-scale search.
- **Balanced approach:** ColBERT → nearly cross-encoder quality with faster search, but higher storage cost.
- Useful in domains where **precision and deep contextual understanding** matter, e.g., legal or medical fields.

## Reranking

### 1. What is Reranking?

- **Definition:** Post-retrieval process where initial search results are re-scored and re-ordered to improve relevance.
- **Purpose:** Ensures the most relevant documents are returned to the LLM after initial retrieval.
- **Key Idea:** Only a small subset of documents is re-ranked, allowing use of high-quality but expensive models.

---

### 2. How It Works

1. **Initial Retrieval:**  
   - Vector database retrieves a set of documents (e.g., 20–100) using hybrid search.
2. **Re-ranking:**  
   - Documents are re-scored using a more capable model (often a cross-encoder or LLM).
3. **Final Selection:**  
   - Return only top documents (e.g., 5–10) based on re-ranked scores.

- **Example:**  
  - Prompt: "What is the capital of Canada?"  
  - Initial results may include:
    - "Toronto is in Canada"
    - "The capital of France is Paris"
    - "Canada is the maple syrup capital of the world"
  - Re-ranker promotes only truly relevant results (e.g., Ottawa).

---

### 3. Reranking Architectures

#### a) Cross-Encoder Re-Ranking

- Concatenates prompt and document, outputs a relevance score.
- Provides high-quality ranking.
- **Cost:** Too slow for large-scale search, but feasible on small subsets of documents retrieved.

#### b) LLM-Based Re-Ranking

- Similar concept to cross-encoder but uses an LLM to assess relevance.
- Outputs numerical relevance scores for prompt-document pairs.
- **Cost:** Computationally expensive, used only post-vector search.

---

### 4. Practical Recommendations

- Overfetch 15–25 documents initially.
- Apply re-ranking to improve relevance with minimal latency.
- Easy to implement in many vector databases (often a single query parameter).
- **Benefit:** Simple technique with substantial improvement in retrieval quality.



