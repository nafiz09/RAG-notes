# LLMs and Text Generation

## Transformers

### 1. Transformer Overview

- **Architecture:** Encoder-decoder (most LLMs use only the decoder for generation).  
- **Goal:** Understand input text deeply and generate meaningful outputs.

### 2. Token Processing

1. **Tokenization:** Split text into tokens.  
2. **Embedding:** Assign dense vectors + positional encoding.  
3. **Attention:** Each token attends to all others, determining importance (attention heads).  
4. **Feedforward Layers:** Refine token embeddings using context.  
5. **Repetition:** Process repeated across multiple layers (8–64 times) for richer representations.

### 3. Generation

- Tokens are predicted sequentially using probability distributions.  
- Each new token considers all prior tokens for context.  
- Process repeats until completion or end-of-sequence token.

### 4. Implications for RAG

- **Why it works:** Attention allows LLMs to interpret retrieved info contextually.  
- **Randomness:** Outputs may vary; grounding in retrieved content is key.  
- **Cost:** Token generation is computationally expensive and grows with prompt length.

---


## LLM sampling strategies

LLMs generate text by making **weighted random choices** for each token. The probability distribution of tokens controls how predictable or creative the output will be.

### 1. Greedy Decoding
- Always selects the **highest-probability token**.
- **Pros:** Deterministic, repeatable output.
- **Cons:** Can be generic or repetitive.
- Useful for **code completion** or debugging.

### 2. Temperature
- Controls randomness of token selection.
- **0:** Greedy decoding.  
- **1:** Default distribution.  
- **>1:** Flattens distribution, more variety and creativity.

### 3. Sampling Techniques
- **Top-k sampling:** Limit choices to the top `k` most probable tokens.
- **Top-p (nucleus) sampling:** Choose tokens until cumulative probability ≤ `p`.  
  - More dynamic than top-k: adapts to certainty of the model.

### 4. Controlling Repetition & Bias
- **Repetition penalty:** Reduces probability of repeated tokens.  
- **Logit biasing:** Adjust probability of specific tokens (e.g., block profanity, boost categories).

### 5. Guidelines
- **Factual/structured tasks:** lower temperature & top-p.  
- **Creative tasks:** higher temperature & top-p.  
- Experiment with parameters to match your **application needs**.


## Choosing an LLM

Selecting the right LLM affects **speed, quality, and cost** of your RAG application. Consider the following factors:

### 1. Quantifiable Metrics
- **Model size:** Number of parameters (small: 1–10B, large: 100–500B+). Larger models are generally more capable but more expensive.  
- **Cost:** Charged per million tokens (input/output).  
- **Context window:** Max tokens for prompt + completion.  
- **Speed:** Time to first token and tokens/sec.  
- **Knowledge cutoff:** Latest data included in training.

### 2. Model Quality
Quality includes reasoning, text generation, and domain-specific abilities. It’s harder to quantify but can be measured via benchmarks.

#### Benchmark Types
1. **Automated Benchmarks:** Machine-graded tasks like MMLU (math, STEM, law).  
2. **Human Evaluation:** Humans compare outputs of LLMs (e.g., LLM Arena).  
3. **LLM-as-a-Judge:** One LLM evaluates another using reference answers; cost-effective but may have bias.

#### Good Benchmark Criteria
- Relevant to your use case  
- Differentiates performance well  
- Reproducible and verifiable  
- Reflects real-world performance  

#### Caveats
- **Data contamination:** LLMs may have seen benchmark data during training.  
- **Saturation:** Many benchmarks are quickly maxed out as models improve. New benchmarks are frequently needed.  

### 3. Takeaways
- Quantitative factors (cost, latency) help narrow choices.  
- Quality metrics guide the best model for your application.  
- LLMs improve rapidly; plan to **swap in newer models** as they are released.

---

## Prompt Engineering

High-quality prompts are key to getting the most out of an LLM. **Prompt engineering** refers to techniques that improve LLM responses.

### 1. Building Prompts in Code
- Common format: **OpenAI messages JSON**.  
- Each message has a **role** (`system`, `user`, `assistant`) and **content**.  
- Multi-turn conversations are converted into this format, submitted as a single string.  
- Special tags indicate the start and end of each message, helping the LLM differentiate roles.

### 2. System Prompts
- Provide high-level instructions to the LLM.  
- Examples of content:
  - Tone and style of responses  
  - Step-by-step reasoning  
  - Knowledge cutoff and current date  
  - Safety guidelines and ethical constraints  
  - Use of retrieved documents only, relevance judgment, or citations  
- System prompts are added to **every prompt**, so refining them improves overall output.

### 3. Prompt Templates
- Define the structure of your prompts.  
- Typical components:
  1. **System prompt** (guidance and rules)  
  2. **Previous conversation messages** (for multi-turn context)  
  3. **Top retrieved chunks** from the retriever  
  4. **Most recent user prompt**  
- Templates make experimentation easy: you can adjust individual components to see their impact on output.

### 4. Takeaway
A well-constructed system prompt combined with a structured prompt template ensures your RAG system produces **relevant, context-aware, and high-quality responses**.


---

## Advanced Prompt Engineering

Once a basic prompt template is set, you can explore advanced techniques to improve LLM performance in your RAG system.

### 1. In-Context Learning
- Helps the LLM understand the kind of output desired by providing **examples in the prompt**.
- **Few-shot learning**: multiple examples.  
- **One-shot learning**: single example.  
- Examples can be **hard-coded** or **retrieved via RAG** from past interactions to improve response quality.

### 2. Reasoning and Step-by-Step Techniques
- Encourage LLMs to **think step-by-step** or use a **scratchpad** to organize thoughts before answering.  
- **Chain-of-thought prompting**: instructs the model to plan and reason incrementally.  
- Improves accuracy and makes errors easier to trace.

### 3. Reasoning Models
- Built to generate **reasoning tokens** before final responses.  
- More accurate but **slower and more expensive**.  
- Useful for assessing relevance and handling complex reasoning tasks.  
- Often perform better with **specific goals** and **explicit format instructions** rather than in-context learning examples.

### 4. Context Window Management
- Advanced techniques increase prompt and response length, consuming the LLM’s context window.  
- **Single-turn fixes**: remove unhelpful techniques if they don’t improve performance.  
- **Multi-turn fixes (context pruning)**:
  - Keep only recent messages.  
  - Summarize older messages to reduce size but retain key points.  
  - Drop reasoning tokens if using reasoning models.  
  - Only include retrieved chunks relevant to the most recent question.

### 5. Practical Advice
- Even with advanced techniques, sometimes a **simple prompt template** and well-written system prompt is sufficient.  
- Add advanced techniques **only when needed**.  
- Experimentation is key: try different prompts, structures, and strategies to optimize performance.

---

## Handling Hallucinations

Even well-designed RAG systems can hallucinate. Detecting, reducing, and ensuring accurate citations are key to building reliable systems.

### 1. Why LLMs Hallucinate
- LLMs generate **probable text sequences**, not necessarily factual content.  
- Hallucinations can sound plausible, making them hard to detect.  
- Over time, hallucinations can **erode user trust**.  

### 2. How RAG Helps
- Grounding responses in **retrieved documents** can reduce hallucinations.  
- Still, additional strategies are needed to further minimize them.

### 3. Types of Hallucinations
- Minor: factual detail slightly wrong (e.g., 5% discount vs 10%).  
- Moderate: denies a true fact.  
- Extreme: invents entirely false information.

### 4. Detection Strategies
- **Self-consistency**: generate multiple completions and check for consistency.  
- **Knowledge base grounding**: require the LLM to only make factual claims from retrieved documents.  
- **Source citation**: prompt the LLM to cite sources for each sentence or paragraph.

### 5. External Tools for Verification
- **ContextCite**: evaluates how well LLM responses are grounded in provided documents.  
  - Tags sentences with source documents or "no source."  
  - Can provide similarity scores between sentences and source material.  
- **ALCE Benchmark**: measures fluency, correctness, and citation quality of LLM outputs against a knowledge base.

### 6. Best Practices
- Design system prompts that enforce grounding in retrieved information.  
- Require citations to increase verifiability.  
- Test with hallucination-focused benchmarks to evaluate performance.  

By combining RAG, proper prompting, and evaluation tools, you can **significantly reduce hallucinations** and build more trustworthy LLM systems.

---

## Evaluating LLM Performance

To optimize your RAG system, it’s important to measure how well your LLM is performing, whether adjusting parameters, prompts, or models.

### 1. LLM’s Role in RAG
- **Retriever**: finds relevant information from the knowledge base.  
- **LLM**: uses retrieved information to construct accurate, high-quality responses.  
- Metrics should focus on the LLM’s performance, not the retriever, assuming the retriever works reasonably well.  

### 2. LLM-Specific Metrics
- Many LLM evaluation metrics rely on **other LLMs** to assess quality.  
- **Ragas library** provides several RAG-specific metrics:

#### 3. Response Relevancy
- Measures whether a response is relevant to the user prompt.  
- Uses semantic embeddings and cosine similarity between the actual prompt and prompts inferred from the response.  
- Focuses on relevance, not factual accuracy.  

#### 4. Faithfulness
- Checks if factual claims in the response are supported by retrieved documents.  
- Calculates the percentage of claims grounded in the knowledge base.  

#### 5. Other Metrics
- Sensitivity to irrelevant retrieved information.  
- Accuracy of citations.  
- Typically involve LLM calls and sometimes ground-truth examples.

### 6. System-Wide Metrics
- User feedback (e.g., thumbs up/down) can track overall satisfaction.  
- A/B testing changes to system prompts or LLM parameters helps attribute performance improvements.  

### 7. Best Practices
- Combine LLM-as-judge metrics and human feedback for more reliable evaluation.  
- Metrics should help you decide on **LLM adjustments** or whether to **swap in a new model**.  


## Agentic Workflows in RAG Systems

Agentic workflows enhance RAG system performance by using multiple specialized LLMs, each responsible for one step in the overall pipeline. This allows more control, higher accuracy, and better system behavior.

### What Is an Agentic Workflow?
- A workflow where **multiple LLMs** execute a sequence of decisions or tasks.
- Each LLM is specialized for **one specific step** (e.g., routing, evaluating, citing).
- LLMs may also access **external tools** such as:
  - Vector databases
  - Code interpreters
  - Web browsers

### Key Differences from a Standard LLM Call
- Tasks are broken into **multiple steps**, not a single prompt→response.
- LLMs operate with **tools** and **conditional logic**.
- The system behaves more like a **flow chart** than a single model call.

---

## Example Agentic RAG Workflow

### Step-by-step Flow
1. **User prompt enters the system.**
2. A **router LLM** evaluates whether retrieval is needed.
   - Outputs **yes** (retrieve) or **no** (skip retrieval).
3. If retrieval = **no**:
   - Prompt is sent directly to a **response-generation LLM**.
4. If retrieval = **yes**:
   - The system retrieves documents from the vector database.
5. An **evaluator LLM** judges whether retrieved docs are sufficient.
   - If insufficient → trigger additional retrieval.
6. Once sufficient context is collected:
   - Build an **augmented prompt**.
   - Send it to an LLM for **response generation**.
7. A **citation LLM** adds citations to the final answer.

### Key Takeaways
- Multiple LLMs handle **routing**, **retrieval validation**, **generation**, and **citation**.
- Lightweight, cheaper models can be used for simple tasks (routing, evaluation).
- Larger models can be reserved for heavy tasks (final answer generation).

---

## Patterns for Agentic Workflows

### Sequential Workflow
- Output moves linearly through multiple LLM components.
- Example sequence:  
  Query Parser → Query Rewriter → Generation → Citation Generator  
- Each LLM specializes in one step.

### Conditional Workflow
- An LLM decides which path the prompt should take.
- Examples:
  - Decide whether retrieval is required.
  - Select which model (e.g., summarizer vs. problem-solver) should answer.

### Iterative Workflow
- The system loops back to an earlier stage for improvement.
- Common for tasks like code generation or multi-step reasoning.
- An evaluator LLM judges drafts and requests improvements until acceptable.

### Parallel Workflow
- An orchestrator LLM breaks a task into multiple subtasks.
- Subtasks are executed by separate LLMs in parallel.
- A synthesizer LLM recombines responses.  
- Example:  
  Summarize two research papers independently → combine insights.

---

## Tooling and System Design Considerations

### Why Agentic Approaches Work
- LLMs become **modular components** rather than monolithic solutions.
- Smaller, faster LLMs can be placed in strategic positions.
- The workflow can be extended, customized, and optimized easily.

### When to Use Agentic Systems
- When your RAG system:
  - Needs consistent citation quality
  - Requires multiple refinement passes
  - Handles diverse task types
  - Needs better routing of user intents
  - Must scale efficiently with cost-aware LLM choices

### Tools and Frameworks
- As systems grow, agentic workflow platforms help manage complexity.
- Many libraries and frameworks support:
  - Routing
  - Tool use
  - Multi-step reasoning
  - Workflow orchestration

---

## Summary
- Agentic workflows enhance RAG systems by decomposing tasks into smaller, specialized steps.
- Each LLM has a **focused role** in the pipeline.
- Workflows can be sequential, conditional, iterative, or parallel.
- This modular design enables more flexibility, efficiency, and higher-quality outputs.

---

## Agentic RAG

Agentic RAG enhances retrieval-augmented generation by using multiple specialized LLMs arranged in structured workflows. Instead of relying on a single model call, the system breaks tasks into smaller steps handled by different LLMs and tools.

---

### What Is an Agentic Workflow?
- A system design where **multiple LLMs** each perform a specific step.
- Steps are connected through structured workflows (like a flow chart).
- LLMs may use external tools:
  - Vector databases
  - Code interpreters
  - Web browsers
- The approach increases flexibility, accuracy, and control.

#### How It Differs From Standard LLM Usage
- Tasks are decomposed into **sequential decisions and actions**.
- Multiple models collaborate instead of one model doing everything.
- External tools become integral parts of the workflow.

---

### Example Agentic RAG Workflow

#### Overall Flow
1. User submits a prompt.
2. **Router LLM** decides whether retrieval is needed.
   - Output: **yes** or **no**
3. If **no retrieval**:
   - Prompt → **Response Generator LLM**
4. If **retrieval needed**:
   - Retrieve documents from vector database.
5. **Evaluator LLM** checks if retrieved documents are sufficient.
   - If not: perform additional retrieval.
6. Build an **augmented prompt** with retrieved info.
7. Send to **Response LLM** to generate an answer.
8. **Citation LLM** adds citations to the final output.

#### Key Observations
- Each LLM has a **single, well-defined role**.
- Lightweight models can be used for routing/evaluation.
- Larger or specialized models handle generation and citation.

---

### Patterns of Agentic Workflows

#### Sequential Workflow
- Output flows in a linear pipeline:
  - Query Parser → Query Rewriter → Generator → Citation Model
- Each LLM specializes in one step.
- Good for predictable multi-step tasks.

#### Conditional Workflow
- An LLM decides between different branches.
- Examples:
  - Determine whether retrieval is required.
  - Select which domain-specialized LLM should answer.

#### Iterative Workflow
- The prompt loops through steps until satisfactory results are produced.
- Useful for:
  - Code generation
  - Multi-step problem solving
- An **Evaluator LLM** checks quality and requests revisions.

#### Parallel Workflow
- An orchestrator LLM splits a task into subtasks.
- Multiple LLMs solve subtasks in parallel.
- A synthesizer LLM recombines the outputs.
- Useful when comparing or synthesizing information (e.g., research papers).

---

### Designing Agentic Systems

#### Why Agentic RAG Works
- Models become **modular components**, not monolithic solutions.
- Smaller models can perform simpler tasks efficiently.
- The system becomes more controllable and extensible.
- Helps reduce hallucinations through structured evaluation steps.

#### When to Use Agentic Approaches
- When your RAG system needs:
  - Better routing of user intent
  - Multi-step reasoning or refinement
  - Reliable citation generation
  - Increased accuracy and grounded answers
  - Cost-efficient model usage at scale

#### Tools and Infrastructure
- Agentic systems often rely on orchestration frameworks providing:
  - Routing
  - Memory
  - Tool integration
  - Multi-model coordination

---

### Summary
- **Agentic RAG** builds on RAG by adding multi-step LLM workflows.
- Each LLM handles one narrow task for precision and specialization.
- Workflows can be sequential, conditional, iterative, or parallel.
- This design leads to more flexible, accurate, and powerful RAG systems.

---

## Fine-Tuning vs RAG

### What Fine-Tuning Is
- Retrains an existing LLM using domain-specific labeled data.
- Typically done via Supervised Fine-Tuning (SFT) with instruction–answer pairs.

### How It Works
- Model compares its output to the correct answer and updates parameters.
- Similar to pretraining but on a smaller, focused dataset.

### When It Helps
- Domain specialization (medical, legal, etc.).
- Small task-specific agent models (e.g., retrieval routing).

### Limitations
- May reduce performance outside the fine-tuned domain.
- Not ideal for adding new factual knowledge.
- Mostly changes style/behavior, not internal knowledge.

### RAG vs Fine-Tuning
- **RAG**: best for injecting new or updated knowledge.
- **Fine-tuning**: best for adapting model behavior to tasks/domains.

### Using Both
- Complementary methods.
- Fine-tuning can improve how a model uses retrieved context in a RAG system.
- Many pre-fine-tuned models are available for direct use.
