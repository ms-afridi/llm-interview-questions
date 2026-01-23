# LLM Fundamentals - Interview Questions & Answers

## Table of Contents
1. [Basic Concepts](#basic-concepts)
2. [Architecture & Mechanisms](#architecture--mechanisms)
3. [Training & Fine-tuning](#training--fine-tuning)
4. [Tokenization & Context](#tokenization--context)
5. [Model Parameters & Performance](#model-parameters--performance)

---

## Basic Concepts

### Q1: What is a Large Language Model (LLM)? How does it differ from traditional NLP models?

**Answer:**

A Large Language Model is a neural network trained on massive amounts of text data to understand and generate human-like text. The key differences from traditional NLP models are:

**LLMs:**
- Pre-trained on billions of words from the internet
- Can handle multiple tasks without task-specific training
- Use transformer architecture with self-attention
- Have billions of parameters (GPT-3 has 175B parameters)
- Can perform zero-shot and few-shot learning

**Traditional NLP Models:**
- Trained on specific tasks (sentiment analysis, NER, etc.)
- Require labeled data for each task
- Use architectures like RNNs, LSTMs, or CNNs
- Have fewer parameters (usually millions)
- Need retraining for new tasks

**Example:** A traditional model needs specific training to do sentiment analysis. An LLM can do sentiment analysis, translation, summarization, and coding just by providing appropriate prompts.

---

### Q2: Explain the Transformer architecture in simple terms. Why was it revolutionary?

**Answer:**

The Transformer is a neural network architecture introduced in the paper "Attention Is All You Need" (2017). Think of it as a smart reading system.

**Key Components:**

1. **Self-Attention Mechanism:** Allows the model to focus on different parts of the input text simultaneously. Like when you read "The animal didn't cross the street because it was too tired" - you know "it" refers to "animal", not "street".

2. **Encoder-Decoder Structure:**
   - **Encoder:** Processes and understands the input
   - **Decoder:** Generates the output

3. **Positional Encoding:** Tells the model the order of words (since it processes all words at once)

**Why Revolutionary:**
- **Parallel Processing:** Unlike RNNs that process word-by-word, Transformers process all words simultaneously (much faster)
- **Long-range Dependencies:** Can understand relationships between words far apart in text
- **Scalability:** Can be trained on massive datasets efficiently
- **Transfer Learning:** Pre-trained models can be fine-tuned for specific tasks

**Real Impact:** Before Transformers, training large models took months. Transformers made it possible to train models 100x faster.

---

### Q3: What is the difference between GPT, BERT, and T5 models?

**Answer:**

These are three different approaches to using the Transformer architecture:

**GPT (Generative Pre-trained Transformer):**
- **Type:** Decoder-only, autoregressive
- **Training:** Predicts the next word (left-to-right)
- **Best for:** Text generation, completion, creative writing
- **Example:** "The cat sat on the ___" → predicts "mat"
- **Models:** GPT-3, GPT-4, ChatGPT

**BERT (Bidirectional Encoder Representations from Transformers):**
- **Type:** Encoder-only, bidirectional
- **Training:** Masked Language Modeling (predicts missing words using context from both sides)
- **Best for:** Understanding tasks like classification, question answering, NER
- **Example:** "The cat [MASK] on the mat" → predicts "sat"
- **Models:** BERT, RoBERTa, DistilBERT

**T5 (Text-to-Text Transfer Transformer):**
- **Type:** Encoder-decoder (full transformer)
- **Training:** Converts all tasks to text-to-text format
- **Best for:** Versatile tasks - translation, summarization, Q&A
- **Example:** Input: "translate English to French: Hello" → Output: "Bonjour"
- **Models:** T5, FLAN-T5

**Quick Comparison:**
```
Task: Sentiment Analysis
GPT approach: "Review: Great movie! Sentiment:" → "Positive"
BERT approach: Classify text directly → [Positive/Negative]
T5 approach: "sentiment: Great movie!" → "positive"
```

---

## Architecture & Mechanisms

### Q4: What is Self-Attention? Explain with an example.

**Answer:**

Self-attention is a mechanism that allows the model to weigh the importance of different words in a sentence when processing each word.

**Simple Analogy:** When you read a sentence, you don't treat every word equally. Some words help you understand others better.

**How It Works:**

Consider: "The bank of the river was muddy"

When processing "bank":
- The model looks at all other words
- Notices "river" is nearby
- Assigns high attention to "river"
- Understands "bank" means riverbank, not financial bank

**Technical Process:**
1. **Query (Q):** What am I looking for?
2. **Key (K):** What do I contain?
3. **Value (V):** What do I actually represent?

For each word, calculate: `Attention = softmax(Q × K^T / √d) × V`

**Example with Numbers:**

Sentence: "Cat sat mat"
```
When processing "sat":
- Attention to "Cat": 0.6 (subject doing the action)
- Attention to "sat": 0.3 (the word itself)
- Attention to "mat": 0.1 (less relevant to understanding "sat")
```

**Why Multiple Heads?** 
Multi-head attention runs this process multiple times in parallel, letting the model focus on different aspects (grammar, meaning, relationships) simultaneously.

---

### Q5: What is the difference between encoder and decoder in transformers?

**Answer:**

**Encoder:**
- **Purpose:** Understands and processes the input
- **How it works:** Uses bidirectional self-attention (looks at all words at once)
- **Output:** Rich representations (embeddings) of the input
- **Used in:** BERT, understanding tasks

**Decoder:**
- **Purpose:** Generates the output
- **How it works:** Uses masked self-attention (only looks at previous words)
- **Output:** Predicted next token/word
- **Used in:** GPT, generation tasks

**Real Example - Translation:**

Input: "How are you?" → Output: "Comment allez-vous?"

```
ENCODER (processes "How are you?"):
- Reads all words simultaneously
- Builds understanding of the entire sentence
- Creates context vectors

DECODER (generates "Comment allez-vous?"):
- Step 1: Uses encoder output + generates "Comment"
- Step 2: Uses encoder output + "Comment" → generates "allez"
- Step 3: Uses encoder output + "Comment allez" → generates "vous"
- (Can only see previously generated words, not future ones)
```

**Key Difference:**
- **Encoder:** "I can see the whole picture at once"
- **Decoder:** "I build the picture one piece at a time, from left to right"

**Interview Tip:** Models like GPT are decoder-only because they only need to predict the next word. BERT is encoder-only because it only needs to understand text, not generate it.

---

### Q6: Explain positional encoding. Why is it necessary?

**Answer:**

Positional encoding adds information about the position of each word in a sequence.

**The Problem:**

Transformers process all words in parallel (simultaneously). Unlike RNNs that process sequentially, transformers have no inherent sense of word order.

Without positional encoding:
- "Dog bites man" and "Man bites dog" would look identical to the model!

**The Solution:**

Add a unique pattern (encoding) to each position that the model can learn to use.

**Two Main Approaches:**

1. **Sinusoidal Positional Encoding (Original Transformer):**
```
For position 'pos' and dimension 'i':
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Why this works:**
- Creates unique patterns for each position
- Allows model to learn relative positions
- Works for any sequence length

2. **Learned Positional Embeddings (BERT, GPT):**
- Treat positions as learnable parameters
- Train them like word embeddings
- Limited to maximum sequence length seen in training

**Visual Example:**
```
Sentence: "The cat sat"

Word Embeddings:        Positional Encoding:      Final Input:
The:  [0.2, 0.5, ...]   Pos 0: [0.1, 0.3, ...]   [0.3, 0.8, ...]
cat:  [0.7, 0.1, ...]   Pos 1: [0.4, 0.6, ...]   [1.1, 0.7, ...]
sat:  [0.3, 0.9, ...]   Pos 2: [0.7, 0.2, ...]   [1.0, 1.1, ...]
```

**Interview Insight:** Without positional encoding, the transformer would be like a "bag of words" model - it wouldn't know that "not good" is different from "good not".

---

### Q7: What is the attention mechanism? How does it improve model performance?

**Answer:**

Attention is a mechanism that allows models to focus on the most relevant parts of the input when making predictions.

**Before Attention (Seq2Seq with RNN):**

In translation, the entire input sentence was compressed into a single fixed-size vector. This created a bottleneck - long sentences lost information.

```
Input: "The quick brown fox jumps over the lazy dog"
↓ (compress everything)
[Single Vector] → Decoder
Problem: All information about 9 words squeezed into one vector!
```

**With Attention:**

The model can "attend to" different parts of the input at each step.

```
Translating to French:

Step 1 - Generating "Le":
  Attention: 0.8 on "The", 0.1 on "quick", 0.1 on others

Step 2 - Generating "rapide":
  Attention: 0.1 on "The", 0.7 on "quick", 0.2 on others

Step 3 - Generating "renard":
  Attention: 0.1 on "The", 0.1 on "quick", 0.8 on "fox"
```

**How It Improves Performance:**

1. **Better Long Sequences:** No information bottleneck
2. **Contextual Understanding:** Each word's meaning depends on context
3. **Interpretability:** Can visualize what the model focuses on
4. **Parallel Processing:** All attention scores computed at once

**Real Example - Question Answering:**
```
Question: "What color is the sky?"
Context: "The sky is blue. Trees are green. Grass is also green."

Attention weights when answering:
- "sky": 0.4
- "blue": 0.5
- "Trees", "green", "Grass": 0.1 total

Model focuses on relevant parts!
```

**Types of Attention:**
- **Self-Attention:** Words attend to other words in same sentence
- **Cross-Attention:** Decoder attends to encoder outputs (in translation)
- **Multi-Head Attention:** Multiple attention operations in parallel

---

## Training & Fine-tuning

### Q8: What is pre-training in LLMs? What are common pre-training objectives?

**Answer:**

Pre-training is the initial phase where an LLM learns general language understanding from massive amounts of unlabeled text data.

**Think of it like:** Learning to read and write before specializing in a career. The model learns grammar, facts, reasoning patterns, and world knowledge.

**Common Pre-training Objectives:**

1. **Causal Language Modeling (CLM) - Used in GPT:**
   - **Task:** Predict the next word
   - **Example:** "The cat sat on the ___" → predict "mat"
   - **Training:** Mask future words, predict left-to-right
   - **Good for:** Text generation

2. **Masked Language Modeling (MLM) - Used in BERT:**
   - **Task:** Predict randomly masked words
   - **Example:** "The cat [MASK] on the mat" → predict "sat"
   - **Training:** 15% of words masked randomly
   - **Good for:** Understanding context from both directions

3. **Next Sentence Prediction (NSP) - Used in BERT:**
   - **Task:** Predict if sentence B follows sentence A
   - **Example:**
     ```
     A: "I love pizza"
     B: "It's my favorite food" → IsNext (True)
     
     A: "I love pizza"
     B: "The sky is blue" → NotNext (False)
     ```

4. **Span Corruption (Used in T5):**
   - **Task:** Predict masked spans of text
   - **Example:** "Thank you for [MASK] me to your party" → "inviting"

**Pre-training Statistics:**

```
GPT-3 Pre-training:
- Data: 45TB of text (filtered to 570GB)
- Tokens: ~300 billion
- Cost: Estimated $4-12 million
- Time: Several months on thousands of GPUs
```

**Why Pre-training Matters:**
- Models learn language fundamentals once
- Transfer this knowledge to specific tasks
- Much cheaper than training from scratch for each task

**Interview Tip:** Pre-training is expensive and done once. Fine-tuning is cheaper and task-specific.

---

### Q9: What is fine-tuning? Explain different types of fine-tuning.

**Answer:**

Fine-tuning is the process of adapting a pre-trained model to a specific task or domain using a smaller, task-specific dataset.

**Analogy:** Pre-training = General education, Fine-tuning = Specialized training for a job

**Types of Fine-tuning:**

**1. Full Fine-tuning:**
- Update all model parameters
- **Pros:** Best performance, maximum adaptation
- **Cons:** Expensive, requires lots of data and compute
- **Use case:** When you have sufficient data and resources

```
Example: Fine-tuning GPT-3 for medical diagnosis
- Start: General GPT-3 (175B parameters)
- Update: All 175B parameters with medical data
- Result: Medical expert model
```

**2. Parameter-Efficient Fine-tuning (PEFT):**

**a) LoRA (Low-Rank Adaptation):**
- Add small trainable matrices to model layers
- Freeze original weights
- **Pros:** 10,000x fewer parameters to train
- **Example:** Instead of updating 175B parameters, update only 10M

```
Original Weight Matrix: W (frozen)
Trainable Addition: W + AB (where A and B are small matrices)
Storage: Only save A and B
```

**b) Prompt Tuning:**
- Keep model frozen
- Only train special tokens added to input
- **Example:** Add learnable "soft prompts" before your input

```
Input: [SOFT_PROMPT_TOKENS] + "Classify: This movie is great"
Only train: SOFT_PROMPT_TOKENS
Keep frozen: All model weights
```

**c) Adapter Layers:**
- Insert small trainable modules between frozen layers
- **Size:** Few million parameters vs billions

**3. Instruction Fine-tuning:**
- Train model to follow instructions
- Uses diverse tasks in instruction format
- **Example Data:**
```
Instruction: "Translate to French"
Input: "Hello"
Output: "Bonjour"

Instruction: "Summarize this article"
Input: [article text]
Output: [summary]
```

**4. RLHF (Reinforcement Learning from Human Feedback):**
- Fine-tune using human preferences
- **Process:**
  1. Generate multiple outputs
  2. Humans rank them
  3. Train reward model
  4. Optimize using RL
- **Used in:** ChatGPT, Claude

**Comparison Table:**
```
Method          Parameters Updated    Memory    Performance
Full            100%                  High      Best
LoRA            0.1-1%               Low       ~95% of full
Prompt Tuning   0.01%                Minimal   ~85% of full
```

**Interview Insight:** In production, LoRA is popular because you can have one base model and swap different LoRA adapters for different tasks.

---

### Q10: What is the difference between zero-shot, one-shot, and few-shot learning?

**Answer:**

These refer to how many examples you provide to the model for a task it hasn't been explicitly trained on.

**Zero-shot Learning:**
- **Definition:** No examples, just the instruction
- **How:** Rely on model's pre-trained knowledge
- **Example:**
```
Prompt: "Translate to Spanish: Hello"
Output: "Hola"
(No translation examples provided)
```

**One-shot Learning:**
- **Definition:** Provide exactly one example
- **How:** Model learns the pattern from one example
- **Example:**
```
Prompt: "Translate to Spanish:
English: Good morning
Spanish: Buenos días

English: Hello
Spanish:"
Output: "Hola"
```

**Few-shot Learning:**
- **Definition:** Provide 2-10 examples
- **How:** Model recognizes pattern from multiple examples
- **Example:**
```
Prompt: "Classify sentiment:
Review: 'Great product!' → Positive
Review: 'Terrible service' → Negative
Review: 'Okay, nothing special' → Neutral

Review: 'Amazing quality!' →"
Output: "Positive"
```

**Performance Comparison:**

For a classification task:
```
Zero-shot:  75% accuracy
One-shot:   82% accuracy
Few-shot:   88% accuracy
Fine-tuned: 95% accuracy
```

**When to Use What:**

**Zero-shot:**
- Quick experiments
- Well-known tasks (translation, summarization)
- Limited examples available

**One-shot:**
- Clarify format or style
- Novel task formats
- Show desired output structure

**Few-shot:**
- Complex patterns
- Consistent performance needed
- Edge cases to cover

**Real Interview Example:**

*Interviewer:* "How would you build a customer support classifier with limited labeled data?"

*Good Answer:*
"I'd start with few-shot prompting (5-10 examples per category) using GPT-4. This requires no training and can be deployed immediately. If accuracy isn't sufficient, I'd collect more data and fine-tune a smaller model like BERT, which would be cheaper to run at scale."

**Key Insight:** Few-shot learning is powerful because it works without any gradient updates - you just provide examples in the prompt!

---
q
