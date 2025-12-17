# RAG-Friendly Document Structure

## Overview
This document outlines the strategy for structuring ROS 2 book content to be optimal for Retrieval-Augmented Generation (RAG) systems, with chunks of ≤500 tokens.

## Chunking Principles

### 1. Semantic Boundaries
- Each chunk should represent a complete thought or concept
- Avoid splitting code examples across chunks
- Keep related explanations together
- Maintain context for code snippets

### 2. Size Constraints
- Maximum 500 tokens per chunk
- Prefer smaller chunks (300-400 tokens) when possible
- Include heading context in each chunk
- Balance information density with comprehensiveness

### 3. Content Hierarchy
- Main heading (H1) + Section heading (H2) + Content
- Code example with surrounding explanation
- Concept definition with examples
- Best practices with rationale

## Implementation Strategy

### Markdown Structure
Each MDX file should be structured to facilitate easy chunking:

```markdown
# Main Topic

## Subtopic 1

Content for subtopic 1...

### Code Example

```python
# Example code
```

Explanation of the code example...

## Subtopic 2

Content for subtopic 2...
```

### Chunking Algorithm
1. Split by H2 headings (major sections)
2. Within each section, look for natural breaks at:
   - Paragraph boundaries
   - Code block boundaries
   - List item boundaries
3. Ensure chunks don't exceed 500 tokens
4. Add context (parent heading) to each chunk

## RAG Metadata

Each chunk should include metadata for better retrieval:

- Source document (file path)
- Heading hierarchy (H1/H2/H3 context)
- Concept tags (nodes, topics, services, etc.)
- Difficulty level (beginner, intermediate, advanced)
- Related concepts for cross-referencing

## Token Estimation
- 1 token ≈ 4 characters in English
- 500 tokens ≈ 2000 characters
- This allows for substantial content while maintaining context

## Validation
- Use token counting tools to verify chunk sizes
- Ensure each chunk makes sense in isolation
- Test retrieval quality with sample queries
- Adjust chunking strategy based on RAG performance