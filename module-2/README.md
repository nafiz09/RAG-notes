# Information Retriever and Search Foundations

A retriever finds useful documents from a knowledge base so the LLM can answer a question.

- This sounds simple, but it's actually hard because:

- Users talk in normal language, not clear search queries.

- The stored documents can be emails, memos, reports, or articles.

- These documents are written for humans, not computers.

- The retriever must search lots of messy text fast and return the most relevant parts within seconds.

Retrievers use different search techniques to match user questions with the right information.
Each method has strengths and weaknesses, and good retrievers often combine multiple methods for the best results.

## Retriever Architecture Overview
### 1. Overview
- The retriever processes the prompt to find the most relevant documents in the knowledge base.
- The knowledge base can be thought of as a collection of text files or documents in a database.
- Goal: Quickly identify and return documents for the LLM to generate accurate responses.

---

### 2. Search Techniques
1. **Keyword Search**
   - Looks for documents containing the exact words in the prompt.
   - A traditional, time-tested method used for decades in information retrieval systems.

2. **Semantic Search**
   - Finds documents that have a similar meaning to the prompt, even if they do not contain the exact words.
   - Provides flexibility to capture contextually relevant documents.

---

### 3. Document Filtering
- After both searches, lists are filtered based on **metadata**.
  - Example: Team-specific relevance (engineering vs HR).
- Metadata filters ensure only documents relevant to the user’s context move forward.

---

### 4. Hybrid Search
- Combines results from keyword and semantic searches into a **final ranking**.
- The top-ranked documents are returned to be included in the augmented prompt.
- **Benefits of hybrid search:**
  - Keyword search ensures sensitivity to exact terms.
  - Semantic search captures contextual meaning.
  - Metadata filtering allows exclusion based on rigid criteria.

---

### 5. Key Takeaway
- High-performing retrievers leverage multiple techniques and carefully tune the balance among them.
- Hybrid search provides flexibility, precision, and context-awareness for RAG systems.


## Metadata Filtering
### 1. Overview
- Metadata filtering narrows down documents based on **rigid criteria** in their metadata.
- Metadata can include document title, author, creation date, access privileges, section, region, etc.
- Filters are usually applied **before or alongside other retrieval techniques**.

---

### 2. Example
- Newspaper knowledge base with thousands of articles.
- Articles are tagged with metadata: title, publication date, author, section, region, access level.
- Queries can filter based on one or multiple metadata fields:
  - Articles by a particular author.
  - Articles in the opinion section published between June and July 2024.
- Only documents matching **all criteria** are returned.

---

### 3. How Metadata Filtering is Applied
- Filters often depend on **user attributes**, not just the prompt:
  - Exclude paid articles for free subscribers.
  - Show articles only from the user's region.
- Acts as a **refinement step** rather than the main retrieval method.

---

### 4. Advantages
- Conceptually simple and easy to debug.
- Fast, mature, and well-optimized.
- Provides **strict control** over what documents are returned.

---

### 5. Limitations
- Not a search technique—doesn't consider document content.
- Cannot rank documents by relevance.
- Overly rigid; must be paired with keyword or semantic search to provide value.

---

## TF-IDF
### 1. Overview
- Keyword search retrieves documents based on **shared words** with the prompt.
- Documents containing more words from the prompt are assumed to be more relevant.
- Both the prompt and documents are treated as a **bag of words**, ignoring order and focusing on word presence and frequency.

---

### 2. Term Document Matrix / Inverted Index
- Each document is converted into a **sparse vector** representing word counts.
- Sparse vectors are organized in a **term-document matrix**, with rows as words and columns as documents.
- Also called an **inverted index**: allows quick retrieval of documents containing a specific word.
- The inverted index can be **precomputed** before any search.

---

### 3. Scoring and Ranking Documents
- For each keyword in the prompt:
  - Find its row in the index.
  - Award points to documents containing that keyword.
- Documents are ranked by **total points**.
- Improvement: award multiple points if a keyword appears multiple times in a document.
- Normalization: divide scores by document length to avoid favoring longer documents.

---

### 4. Term Weighting with IDF
- Not all words are equally informative (e.g., "the" vs. "pizza").
- **Inverse Document Frequency (IDF)** gives higher weight to rare words:
  - IDF = total documents ÷ number of documents containing the word.
  - Take the log of the IDF to reduce exaggeration.
- Multiply the term frequencies by the IDF to produce a **TF-IDF matrix**.

---

### 5. TF-IDF Scoring
- For each keyword in the prompt, award documents the **TF-IDF score** for that keyword.
- High-scoring documents contain many keywords, especially **rare ones**.
- TF-IDF is a standard **baseline for keyword retrieval**.

---

## BM25

### 1. Overview
- BM25 (Best Matching 25) is the **standard keyword search algorithm** in most modern retrievers.
- It builds on TF-IDF with a few key improvements.
- Each keyword in a document gets a **relevance score**, summed across all keywords for a document's total score.

---

### 2. Improvements Over TF-IDF
1. **Term Frequency Saturation**
   - Documents get **diminishing returns** for repeated keywords.
   - Example: "pizza" appearing 20 times isn’t twice as relevant as appearing 10 times.
2. **Document Length Normalization**
   - Longer documents are penalized less aggressively than in TF-IDF.
   - Allows long documents with high keyword frequency to score well.
3. **Tunable Hyperparameters**
   - Control how quickly scores saturate with repeated keywords.
   - Control how document length affects scoring.
   - Tuned to fit the characteristics of your knowledge base.

---

### 3. Key Advantages
- Performs better than TF-IDF in retrieving relevant documents.
- Computationally similar to TF-IDF.
- Flexible via hyperparameter tuning.
- Matches **exact keywords**, making it useful for technical terms or product names.

---

### 4. How It Fits in Keyword Search
- Prompts and documents are converted to **sparse vectors** (word counts per vocabulary term).
- TF-IDF or BM25 processes these vectors to **score and rank** documents.
- Takes into account:
  - Keyword rarity (IDF)
  - Frequency of keyword in document
  - Document length

---

### 5. Strengths
- Simple and effective.
- Often sets a strong **baseline** in retriever pipelines.
- Ensures retrieved documents **contain prompt keywords**.

---

### 6. Limitations
- Relies on the prompt containing **exact matching keywords**.
- Cannot match documents that are relevant in **meaning but lack exact word overlap**.
- Leads to the need for **semantic search** to address meaning-based retrieval.


## **Semantic Search**

### 1. Overview
- Matches documents to prompts based on **shared meaning**, not just exact words.
- Captures nuances keyword search misses:
  - Example: "happy" ≈ "glad"  
  - Distinguishes homonyms: "Python" (language) vs. "Python" (snake)

---

### 2. How It Works
1. **Vector Representation**
   - Every document and prompt is mapped to a **vector**.
   - Vectors for semantically similar texts are **closer in space**.

2. **Embedding Models**
   - Special mathematical models generate vectors from text.
   - Map words, sentences, or documents to a **high-dimensional space**.
   - Example: "food" and "cuisine" are nearby; "trombone" and "cat" are far apart.
   - High-dimensional vectors allow nuanced relationships between concepts.

---

### 3. Measuring Similarity
- **Euclidean Distance**: straight-line distance between vectors.
- **Cosine Similarity**: measures similarity in **direction**, ignoring magnitude.
  - Range: -1 (opposite) to 1 (same direction)
- **Dot Product**: projection of one vector onto another.
  - Large value → similar vectors
  - Zero → orthogonal (unrelated) vectors
  - Negative → opposite vectors

---

### 4. Semantic Search Pipeline
1. Embed all documents into vector space using an **embedding model**.
2. Embed the prompt into its own vector.
3. Measure the distance between the prompt vector and each document vector.
4. **Rank documents** by similarity (closest = most relevant).
5. Return the top-ranked documents as the most semantically relevant results.

---

### 5. Key Points
- Embedding models encode meaning, not just words.
- Distance between vectors quantifies **semantic relevance**.
- Works for words, sentences, and entire documents.
- Understanding embedding models helps optimize semantic search performance.


## Embedding Models for Semantic Search

### 1. Goal of an Embedding Model
- Map **similar text** to vectors **close together**.
- Map **dissimilar text** to vectors **far apart**.
- Allows semantic search to find meaning-based matches.

---

### 2. Positive and Negative Pairs
- **Positive Pair:** Similar texts (e.g., "good morning" & "hello") → should be close.
- **Negative Pair:** Dissimilar texts (e.g., "good morning" & "that's a noisy trombone") → should be far.
- Each piece of text appears in many pairs to capture a wide range of relationships.

---

### 3. Training Process (Contrastive Training)
1. **Initialization:** Map all texts to random vectors.
2. **Evaluation:** Check distances between positive and negative pairs.
3. **Update:** Adjust vectors to:
   - Pull **positive pairs** closer.
   - Push **negative pairs** farther apart.
4. **Iteration:** Repeat many times across millions of pairs.
5. **Result:** Vectors capture semantic meaning; similar concepts cluster together in high-dimensional space.

---

### 4. Example
- Anchor text: "He could smell the roses"
  - Positive pair: "A field of fragrant flowers" → pull closer
  - Negative pair: "A lion roared majestically" → push away
- High-dimensional vectors allow nuanced placement for millions of texts simultaneously.

---

### 5. Key Takeaways
- Before training: vectors are random; after training: vectors encode semantic relationships.
- Clusters of similar concepts form, but exact locations are random.
- **Vectors from different embedding models are not comparable.**
- Understanding this helps reason about semantic search, even if you use off-the-shelf embeddings.


## Hybrid Search in a Retriever

Hybrid search combines **metadata filtering**, **keyword search**, and **semantic search** to leverage their complementary strengths.

---

### 1. Recap of Each Component
| Technique           | How it Works                                           | Strengths                                                   | Limitations                                             |
|--------------------|-------------------------------------------------------|-------------------------------------------------------------|--------------------------------------------------------|
| Metadata Filtering  | Filters documents using rigid metadata criteria      | Fast, simple, allows strict yes/no filtering              | Cannot rank or evaluate document relevance           |
| Keyword Search      | Scores and ranks documents based on matching keywords | Exact word matches, fast, effective for technical terms   | Fails when wording differs but meaning is the same   |
| Semantic Search     | Uses embeddings to score documents by meaning        | Flexible, finds conceptually similar documents            | Slower, more computationally intensive               |

---

### 2. Hybrid Search Pipeline
1. **Prompt received** by the retriever.
2. **Keyword search** and **semantic search** are performed separately.
   - Each produces a ranked list of documents.
3. **Metadata filtering** is applied to both lists.
   - Removes irrelevant documents based on strict criteria (e.g., user role, region, subscription status).
4. **Combine rankings** using **Reciprocal Rank Fusion (RRF)**:
   - Each document scores points based on its rank in each list.
   - Formula: `score = sum(1 / (k + rank_i))` where `k` is a hyperparameter.
   - Higher ranks contribute more; `k` controls the impact of top-ranked documents.
5. **Optional weighting** via `beta`:
   - Adjust relative importance of semantic vs keyword ranking.
   - Example: `beta = 0.7` → 70% weight to semantic, 30% to keyword.

---

### 3. Top-K Selection
- After fusion and weighting, retriever returns the **top K documents**.
- These documents are then passed to the LLM for augmentation.

---

### 4. Benefits of Hybrid Search
- **Keyword search:** Exact matches ensure relevant terms appear.
- **Semantic search:** Captures meaning even when words differ.
- **Metadata filtering:** Enforces strict constraints (e.g., access control, relevance by user type).
- Tunable hyperparameters allow system optimization:
  - BM25 parameters (for keyword search)
  - Beta (weighting between semantic & keyword search)
  - K (RRF scaling for top-ranked documents)

---

**Summary:**  
Hybrid search maximizes retriever effectiveness by combining fast, exact matching, meaning-based flexibility, and strict metadata constraints. Proper tuning ensures retrieval aligns with the needs of the dataset and application.

## Retriever Evaluation Metrics

To measure a retriever’s effectiveness, we focus on **search quality**: is it finding the relevant documents?  
This requires three ingredients:

1. **Prompt/query**  
2. **Retriever results** – ranked list of documents returned  
3. **Ground truth** – list of relevant documents for the prompt  

---

### 1. Precision and Recall

| Metric   | Formula | Intuition |
|----------|---------|-----------|
| **Precision** | Relevant retrieved / Total retrieved | How trustworthy the retrieved results are (penalizes irrelevant docs) |
| **Recall** | Relevant retrieved / Total relevant | How comprehensive the retriever is (penalizes missing relevant docs) |

**Example:**

- Knowledge base has 10 relevant documents  
- Retriever returns 12 documents, 8 relevant  
  
```  
Precision = 8 / 12 = 66% 
Recall = 8 / 10 = 80%
```


- Returning more documents may increase recall but reduce precision.  
- Trade-off is common: perfect retrieval requires returning only relevant documents at the top.

**Top-K Consideration:**

- Metrics are often measured at **top K** retrieved documents.  
- Example: Precision@5, Recall@10 – gives standardized comparison across prompts.

---

### 2. Mean Average Precision (MAP)

- Measures how well relevant documents are ranked at the top.
- **Average Precision at K**:
  1. Calculate precision at every rank where a relevant document appears.
  2. Average these precisions for the retrieved relevant documents.
- **Mean Average Precision**: Average the AP@K across multiple prompts.

**Example:**

- Top 6 documents: relevant at ranks 1, 4, 5  
- Precision at relevant ranks: 1.0, 0.5, 0.6  
- Average Precision = (1 + 0.5 + 0.6) / 3 = 0.7  

**MAP rewards**: ranking relevant documents higher; penalizes irrelevant documents appearing early.

---

### 3. Reciprocal Rank / Mean Reciprocal Rank (MRR)

- **Reciprocal Rank (RR)**: 1 / (rank of first relevant document)  
- **Mean Reciprocal Rank (MRR)**: Average RR across multiple prompts  

**Example:**

- First relevant document ranks: 1, 3, 6, 2  
- RR = 1, 1/3, 1/6, 1/2  
- MRR = (1 + 1/3 + 1/6 + 1/2) / 4 = 0.5  

**Emphasis**: measures how quickly a retriever surfaces at least one relevant document.

---

### 4. Using Metrics Together

- **Recall**: most fundamental; ensures retriever finds relevant documents  
- **Precision & MAP**: measure irrelevant documents and ranking effectiveness  
- **MRR**: specialized, focuses on top-ranked performance  

**Use case:**  
- Adjust hybrid search weights (semantic vs keyword) and evaluate effect on precision, recall, MAP, and MRR.  
- Metrics require ground truth – can be manual, but essential for monitoring performance.

---

**Summary:**  
Retriever evaluation balances **finding all relevant documents** (recall) with **ranking them highly and minimizing irrelevant results** (precision, MAP, MRR).  
Proper metrics guide tuning and ensure your retriever works effectively in production.







