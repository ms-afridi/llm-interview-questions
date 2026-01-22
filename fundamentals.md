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
