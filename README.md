# Langchain v0.3 RAG Practice

## Purpose

This short project is from the LangChain documentation to build a simple question and answer retriever.
I changed the model from using OpenAI to a local Ollama model, Mistral 7b.

LangGraph continued to throw errors, therefore I removed the code as it was not needed for my purposes.

The project uses LangChain framework to:

1. Pull online article from a GitHub account
2. Chunk the text
3. Pass the chunks of text to the HuggingFace embedding model for tokenization
4. The code should add the vectors to ChromaDB but the LangChain documentation skips how to instantiate ChromaDB

Results - practice using LangChain syntax. No resuable code or true RAG built.
