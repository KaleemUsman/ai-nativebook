---
id: 1
title: Update Project Constitution
stage: constitution
date: 2025-12-16
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: master
user: scs
command: /sp.constitution
labels: [constitution,documentation,setup]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

/sp.constitution

Project: AI-Native Book + RAG Chatbot on Physical AI & Humanoid Robotics

Core principles:
- Technical accuracy (ROS 2, Gazebo, Unity, Isaac, SLAM, VLA)
- Clean, modular Docusaurus documentation
- Code correctness and API alignment (OpenAI Agents/ChatKit, FastAPI, Qdrant, Neon)
- AI-native creation using Spec-Kit Plus + Claude Code

Key standards:
- Verify all robotics/AI claims via official docs
- Produce runnable or structurally correct code
- Uniform terminology across modules
- Professional, concise, developer-friendly writing

Constraints:
- Use Spec-Kit Plus structure
- Output in Docusaurus-ready MDX
- RAG system must use FastAPI + Neon + Qdrant + OpenAI SDKs
- Chatbot must answer strictly from book content or selected text

Success criteria:
- Full book covering the 4 modules (ROS 2, Gazebo/Unity, Isaac, VLA)
- Working embedded RAG chatbot with ingestion, retrieval, and grounded answers
- Deployable GitHub Pages site
- Accurate capstone flow: Autonomous Humanoid (voice → plan → na

## Response snapshot

Created the project constitution for the AI-Native Book + RAG Chatbot project with Physical AI & Humanoid Robotics focus. The constitution defines six core principles:

1. Technical Accuracy - Ensuring all technical content related to ROS 2, Gazebo, Unity, Isaac, SLAM, VLA, and humanoid robotics is verified against official documentation.
2. Clean, Modular Docusaurus Documentation - Structuring content in clean, modular MDX format suitable for Docusaurus.
3. Code Correctness and API Alignment - Aligning code with specified technology stacks (OpenAI Agents/ChatKit, FastAPI, Qdrant, Neon) and ensuring API contracts are honored.
4. AI-Native Creation using Spec-Kit Plus + Claude Code - Leveraging AI tools for specification-driven development and implementation.
5. Verification Standards - Verifying all robotics and AI claims through official documentation before inclusion.
6. Uniform Terminology Across Modules - Maintaining consistent terminology across all four modules (ROS 2, Gazebo/Unity, Isaac, VLA).

Additionally, the constitution outlines key standards, technology stack requirements, quality standards, implementation requirements, and review processes.

## Outcome

- ✅ Impact: Updated project constitution establishes foundational principles for AI-Native Book + RAG Chatbot project
- 🧪 Tests: none
- 📁 Files: Modified .specify/memory/constitution.md with project-specific principles
- 🔁 Next prompts: Continue with feature specification and planning using established principles
- 🧠 Reflection: Constitution properly captures project-specific requirements and constraints for physical AI and humanoid robotics content

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): Begin feature specification following constitution guidelines