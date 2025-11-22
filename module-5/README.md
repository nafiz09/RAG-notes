# RAG System in Production

## Production Challenges for RAG Systems

### Scaling & Performance
- More users increase throughput demands, request volume, and latency.
- Higher memory + compute usage → higher operational cost.
- Maintaining fast, stable performance at scale becomes difficult.

### Unpredictable User Prompts
- Real users generate diverse, unexpected prompts.
- System may fail on new request types despite strong pre-launch testing.

### Messy Real-World Data
- Data may be fragmented, unstructured, or missing metadata.
- Many sources (images, PDFs, slide decks) require preprocessing to be usable.
- Non-text formats need specialized extraction (OCR, parsing).

### Security & Privacy
- RAG often uses proprietary or sensitive data.
- Must ensure only authorized users access private knowledge base content.

### Real Business Impact
- Mistakes can cause financial or reputational damage.
- Examples:
  - Google AI suggested eating rocks due to misinterpreted joking forum posts.
  - Airline chatbots have offered invalid discounts.
  - Malicious users may try to exploit RAG for free products or secret data.

### Need for Observability & Monitoring
- Systems must anticipate issues, detect failures, and validate fixes.
- Observability is essential for reliable production RAG.

---

## Implementing RAG evaluation strategies

### Key Observability Components
- Track **software performance metrics**: latency, throughput, memory, compute usage.
- Track **quality metrics**: user satisfaction, retriever recall, response accuracy.
- Collect:
  - **Aggregate stats** for long-term trends and regression detection.
  - **Detailed logs** to trace individual prompt flows for debugging.
- Support **experimentation**: A/B tests, prompt changes, model swaps, retriever tweaks.

### Evaluation Dimensions
- **Scope**
  - *System-level*: overall performance + quality.
  - *Component-level*: retriever, LLM, indexing, etc., used for debugging source issues.
- **Evaluator Type**
  - *Code-based*: cheap, automatic, deterministic (latency, throughput, JSON validity).
  - *Human feedback*: thumbs up/down, written feedback, human-annotated datasets.
  - *LLM-as-judge*: flexible, cheaper than humans; grades relevance, quality, etc.

### Trade-offs in Evaluators
- Code-based evals → very cheap but limited.
- Human feedback → expensive but captures nuance others miss.
- LLM-as-judge → middle ground; needs careful rubric + bias control.

### Example Metrics to Track
- **Performance (code-based)**  
  - System + component latency  
  - Throughput  
  - Memory/compute usage  
  - Tokens/sec  
- **Quality (LLM/human-based)**  
  - User thumbs up/down for system-level quality  
  - Retriever precision & recall (requires human-annotated dataset)  
  - LLM relevance, citation quality, and grounding (e.g., via Ragas)

### Overall Approach
- Combine cheap system metrics with selective high-value human/LLM evals.
- Monitor both per-component and system-wide behavior.
- Build visibility to detect failures, debug issues, and validate improvements.

---


## Logging, monitoring, and observability

### Observability Platforms
- Many platforms exist for LLM/RAG monitoring: track metrics, logs, and experiments.
- They reduce engineering overhead and let you focus on improving system performance.

### Phoenix (Arise AI)
- Open-source observability + evaluation platform.
- Integrates easily with RAG pipelines and libraries like RAGAS.

### Traces
- Shows the full path of a prompt through the RAG system.
- Includes:
  - Original prompt
  - Retriever query
  - Returned chunks
  - Re-ranker output
  - Final LLM prompt + generated answer
  - Latency per step
- Useful for debugging poor responses and identifying failing components.

### Integrated Evaluations
- Phoenix supports:
  - Retriever relevance scoring
  - Citation accuracy
  - Other RAGAS-based metrics
- Enables quick experiments (e.g., try new prompts, add re-ranker, adjust settings).

### A/B Testing & Experiments
- Test system changes to compare response quality, latency, and accuracy.
- Helps validate whether a modification improves real performance.

### Aggregate Metrics
- Provides daily reports on:
  - Retriever accuracy
  - LLM hallucination rate
  - Overall system trends

### Complementary Tools
- Phoenix doesn’t cover everything (e.g., vector DB compute/memory usage).
- Use classical tools like **Datadog** and **Grafana** for infra-level monitoring.

### Improvement Flywheel
- Observability → find bugs → make changes → measure improvement.
- Over time, each system component becomes better aligned with real user behavior.

### Custom Datasets
- Collect real production prompts into datasets.
- Re-run them after system changes to measure impact on *actual* user queries.
---

## Custom Datasets

### Purpose of Custom Datasets
- Collect real prompts your system has processed.
- Understand past system behavior and test redesigns on real-world data.
- Enable both end-to-end and component-level evaluation.

### What to Store
- Minimum: user input prompt + final system response.
- For deeper analysis: store data from each component:
  - Retriever results + retrieved documents
  - Re-ranker inputs/outputs
  - Query rewrites
  - Router LLM decisions
  - Customer ID, metadata, chunk rankings, etc.
- Choice depends on **what you want to evaluate**.

### Benefits of Detailed Logs
- Enables component-level debugging.
- Example: detect certain topics (e.g., product delays) performing poorly → check logs → see retriever missing relevant docs → update knowledge base.
- Helps identify patterns across question types, user segments, or system components.

### Example Incident
- A RAG system produced low-quality diagrams.
- Logs revealed router LLM was incorrectly sending “draw a diagram” prompts to a text-to-image model.
- Fix: update router system prompt → correct routing → problem resolved.
- Robust logging made debugging straightforward.

### Visualization & Analysis
- With large datasets, visualize logs to spot trends.
- Cluster prompts by topic (refunds, product delays, troubleshooting, etc.).
- Run targeted evals on specific prompt clusters to find underperforming areas.

### Why Custom Datasets Matter
- They personalize evaluation to **real user behavior**.
- Enable consistent, repeatable testing on actual production prompts.
- Critical for reliably improving system performance and user satisfaction.

---

## Quantization

### What Quantization Is
- Compression technique for **LLMs** and **embedding vectors**.
- Converts high-precision values (e.g., 16-bit weights, 32-bit floats) to lower precision (8-bit, 4-bit, or even 1-bit).
- Results: **smaller models**, **cheaper compute**, **faster inference**, often with **minimal quality loss**.

### Analogy: Image Compression
- High-bit images look best but cost more storage.
- Lower-bit images save memory but degrade quality.
- Quantization applies the same trade-off to model weights and embedding vectors.

### Quantizing LLMs
- Original LLM parameters use 16-bit values.
- Quantized to 8-bit or 4-bit → drastically reduced GPU memory usage.
- Leads to faster inference, small performance drop on benchmarks.

### Quantizing Embeddings
- Typical embedding: 768 dimensions × 32-bit floats → ~3 KB per vector.
- At scale (millions/billions), memory + RAM cost becomes huge.
- Integer quantization:
  - Map each float to an 8-bit integer using min/max range + 256 buckets.
  - ~4× size reduction with only small accuracy loss.
- Very fast to store and compute.

### 1-bit (Binary) Quantization
- Compresses vectors by 32× (1 bit vs 32 bits per dimension).
- Encodes only sign: positive or negative.
- Fastest + smallest, but larger drop in retrieval performance.
- Often used for initial coarse retrieval, then re-ranked with full vectors.

### Matryoshka Embeddings
- Embeddings where dimensions are sorted by **information content** (variance).
- Enables dynamic use:
  - Use first 100 dims for fast retrieval.
  - Use full dims (e.g., 1000) for re-ranking.
- Useful in systems needing quick low-fidelity + slower high-fidelity stages.

### Practical Takeaway
- 8-bit and 4-bit quantized LLMs/embeddings are widely available.
- Major cost + memory savings with minimal quality reduction.
- Should be included in experiments for optimizing speed, cost, and performance.

---



## Cost vs Response Quality

### Major Cost Drivers
- **LLMs** (inference costs)
- **Vector databases** (memory + storage)

### Reducing LLM Costs
- **Use smaller models**:
  - Fewer parameters or quantized versions (8-bit, 4-bit).
  - Often perform well, especially for narrow tasks.
  - Small models + fine-tuning can be highly cost-effective.
- **Reduce token usage**:
  - Retrieve fewer documents (lower top-k).
  - Encourage concise responses via system prompts.
  - Enforce output token limits.
- **Host models on dedicated hardware**:
  - Renting GPUs (hourly) becomes cheaper at high request volumes.
  - Improves reliability since hardware serves only your traffic.

### Reducing Vector Database Costs
- Databases use 3 memory tiers:
  - **RAM** – fastest, most expensive.
  - **Disk** – medium speed, medium cost.
  - **Cloud object storage** – slowest, cheapest.
- Strategy:
  - Keep only performance-critical items (e.g., HNSW index) in RAM.
  - Move documents or rarely accessed items to disk or cloud storage.

### Multi-Tenancy for Dynamic Storage
- Split vectors by user/tenant.
- Load each tenant’s HNSW index into RAM only when needed.
- Example optimizations:
  - Load a customer’s vectors only when they log in.
  - Keep nighttime regions on slower storage.
- Enables efficient movement of data between RAM, disk, and object storage.

### Core Principle
- Understand **where costs come from** and ensure they are justified.
  - For LLMs → smaller models, shorter prompts.
  - For vector DBs → minimize RAM usage and use cheaper storage tiers.
- Use observability to measure effects and decide if cost–quality trade-offs are acceptable.

---

## Latency vs. Response Quality

### The Trade-Off
- Adding components like retrievers, re-rankers, or agentic systems improves response quality but increases latency.
- Importance of latency depends on context:
  - **E-commerce**: prioritize low latency, possibly sacrificing perfect recommendations.
  - **Medical diagnosis**: prioritize high-quality responses, tolerate higher latency.

### Primary Cause of Latency
- Most latency comes from transformer-based LLM calls.
- Retrieval adds minimal latency, especially with fast, scalable vector databases.

### Strategies to Reduce Latency

#### Optimize Core LLM
- Use smaller or quantized models for faster inference.
- Implement a **router LLM**:
  - Direct simple queries to smaller models.
  - Direct complex queries to larger, more powerful models.

#### Caching
- Cache responses for frequently seen prompts.
- Optionally personalize cached responses using a smaller LLM to adjust them.

#### Optimize Other Transformers
- Measure latency and quality contribution for query rewriters, re-rankers, and other LLM-based components.
- Remove components with low benefit-to-latency ratio.

#### Retriever Optimizations
- Use binary-quantized embeddings to speed up vector similarity calculations.
- Shard large databases to reduce search latency.

### Best Practices
- Determine acceptable latency thresholds.
- Iteratively optimize:
  1. Core LLM
  2. Other transformer components
  3. Retrieval/database components
- Use observability tools to measure the impact of changes.

### Key Principle
- Every RAG system has a **latency vs. quality trade-off**.
- Focus on the components that most affect latency first, balancing with required response quality.
---

## Security

### Importance of Security
- RAG systems often contain private or proprietary information not available on the open web.
- Protecting the knowledge base is critical to prevent unauthorized access or leakage.

### Potential Security Risks
1. **Direct Access via Prompts**
   - Users may craft prompts that extract information directly from retrieved chunks.
   - Even with safeguards, indirect access to knowledge base contents is possible.

2. **Multi-Tenant Access Issues**
   - Metadata filtering alone is not reliable for security.
   - Role-based access control (RBAC) with separate tenants is safer.

3. **Data Exposure to LLM Providers**
   - Sending augmented prompts with retrieved documents to external LLMs can leak sensitive information.
   - Running LLMs on-premises reduces this risk.

4. **Database Hacking**
   - Vector databases can be hacked like any traditional database.
   - Encryption of text chunks protects information, but dense vectors must remain unencrypted for ANN operations.

5. **Reconstruction from Dense Vectors**
   - Research shows that original text can sometimes be reconstructed from dense vectors.
   - Techniques like adding noise, transformations, or dimensionality reduction can help mitigate this risk but may impact performance.

### Recommended Security Measures
- **User Authentication**: Ensure only authorized users can access the RAG system.
- **Role-Based Access Control (RBAC)**: Use separate tenants for different access levels rather than relying solely on metadata filters.
- **On-Premises Deployment**: Host LLMs and vector databases locally to maintain full control over sensitive data.
- **Encryption**: Encrypt text chunks and decrypt only when needed to build augmented prompts.
- **Vector Security Awareness**: Understand the potential risk of dense vector reconstruction and explore mitigations carefully.

### Key Takeaways
- Knowledge base likely contains private information; access must be controlled.
- Combine RAG-specific security techniques with broader cybersecurity practices.
- Awareness of unique vector database vulnerabilities is important for robust security.


## Multimodal RAG

### Introduction
- Traditional RAG systems handle only text.
- Modern knowledge bases contain slides, PDFs, images, audio, and video.
- Multimodal RAG systems enable LLMs to process and retrieve multiple data types.

### Key Components

#### Multimodal Embedding Model
- Embeds multiple data formats into the same vector space.
- Text and image representations with similar meanings are placed close together.
- Enables vector-based retrieval of text and images in a unified space.

#### Multimodal Retrieval
- Prompts (text or image) are embedded using the multimodal model.
- Vector search returns the closest matching documents or images.
- Retrieved items are used to create the augmented prompt for the LLM.

#### Language Vision Model
- LLM capable of processing tokenized images alongside text.
- Images are split into patches, each treated as a token.
- Transformer processes both text and image tokens to produce text responses.

### Handling Complex Documents
- Slides and PDFs can be treated as images for multimodal RAG.
- Dense information on pages requires **chunking**:
  - Initially done with sophisticated layout detection.
  - PDF RAG approach: split pages into a grid of squares, each embedded separately.
- Vector search aggregates scores across all grid sections for retrieval.

### Advantages and Challenges
- **Advantages:**
  - Handles diverse file formats (images, slides, PDFs, etc.).
  - Flexible and performs well in retrieval tasks.
- **Challenges:**
  - Requires storing a large number of vectors in the database.
  - Multimodal embedding models are still experimental.
  - Complexity in tokenizing and processing images.

### Future Outlook
- Multimodal RAG is cutting-edge and rapidly evolving.
- Expect ongoing improvements in:
  - Multimodal embedding models
  - Language vision models
  - Vector database tools for multimodal retrieval

