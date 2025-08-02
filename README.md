# 🧠 Meddola – Intelligent AI Health Assistant

Meddola is a conversational AI health assistant built using **LangGraph**, **LLaMA 3.3-70B**, and **semantic memory search**. It acts like a friendly and intelligent medical companion, helping users with symptom analysis, medication info, daily routines, and more — all while remembering past conversations.

---

## 🔧 Features

- **🩺 Health Query Analysis**  
  Understands symptoms and provides helpful responses, remedies, or condition suggestions.

- **💊 Drug Information**  
  Retrieves drug-related details using the **FDA API** or fallback LLM-based summaries.

- **📋 Personalized Routine & Diet Plan**  
  Creates daily health routines or diet plans tailored to the user’s lifestyle and health condition.

- **🧠 Memory-Augmented Reasoning**  
  Uses **Sentence Transformers** to embed and recall past chats using vector search (ChromaDB).

- **🔁 Intent-Aware Flow with LangGraph**  
  Classifies user intent and routes to appropriate action nodes (like `create_routine`, `drug_query`, etc.)

- **🖥️ CLI Mode**  
  Chat with Meddola directly in the terminal.

---

## 📦 Tech Stack

- LLaMA 3.3-70B (via Groq API)  
- LangGraph (agentic workflows)  
- Sentence Transformers + Chroma (for vector memory)  
- FDA Drug API  
- Python


