<!-- SYNC IMPACT REPORT:
Version change: 1.0.0 → 1.1.0
Modified principles: [PRINCIPLE_1_NAME] → Technical Accuracy, [PRINCIPLE_2_NAME] → Clean Documentation, [PRINCIPLE_3_NAME] → Code Correctness, [PRINCIPLE_4_NAME] → AI-Native Creation, [PRINCIPLE_5_NAME] → Verification Standards, [PRINCIPLE_6_NAME] → Uniform Terminology
Added sections: Core Principles (completely replaced with project-specific content)
Removed sections: Template placeholders
Templates requiring updates: N/A (this is the first version)
Follow-up TODOs: None
-->
# AI-Native Book + RAG Chatbot on Physical AI & Humanoid Robotics Constitution

## Core Principles

### Technical Accuracy
All technical content related to ROS 2, Gazebo, Unity, Isaac, SLAM, VLA, and humanoid robotics must be verified against official documentation and technical specifications. Claims about robotics/AI concepts must be fact-checked through authoritative sources before inclusion. This ensures the highest technical integrity across all documentation and code examples.

### Clean, Modular Docusaurus Documentation
All content must be structured in clean, modular MDX format suitable for Docusaurus documentation. Each module should be self-contained yet connected to the broader narrative. The documentation structure must support easy navigation, consistent styling, and seamless integration with the RAG system.

### Code Correctness and API Alignment
All code implementations must align with specified technology stacks: OpenAI Agents/ChatKit, FastAPI, Qdrant, Neon, and related SDKs. Code must be functionally correct, properly tested, and follow established patterns for the respective frameworks. API contracts must be honored with proper error handling and response formatting.

### AI-Native Creation using Spec-Kit Plus + Claude Code
All content and implementation must leverage Spec-Kit Plus and Claude Code for AI-native creation. This includes using specification-driven development, automated documentation generation, and AI-assisted implementation. The development process should embrace AI tools as primary development assets.

### Verification Standards
All robotics and AI claims must be verified through official documentation, technical papers, or established resources. No unverified claims should be made about technical capabilities, performance characteristics, or functional properties. All code examples must be tested and proven runnable before inclusion.

### Uniform Terminology Across Modules
Consistent terminology must be maintained across all four modules (ROS 2, Gazebo/Unity, Isaac, VLA). Technical terms should be defined once and reused consistently. Acronyms and specialized vocabulary must be introduced clearly and applied uniformly throughout the documentation and codebase.

## Key Standards and Constraints

### Technology Stack Requirements
- Use Spec-Kit Plus structure for all project artifacts
- Output all documentation in Docusaurus-ready MDX format
- RAG system implementation must use FastAPI + Neon + Qdrant + OpenAI SDKs
- Chatbot functionality must strictly answer from book content or selected text
- All code must be deployable to GitHub Pages with associated RAG backend

### Quality Standards
- Verify all robotics/AI claims via official documentation
- Produce runnable or structurally correct code examples
- Maintain uniform terminology across all modules
- Ensure professional, concise, developer-friendly writing style
- Implement proper error handling and edge case management

## Development Workflow and Quality Gates

### Implementation Requirements
- Each module must be developed using specification-driven approach
- Code implementations must pass all functional and integration tests
- Documentation must be written in accessible, developer-friendly language
- All examples must be tested and validated for correctness
- RAG system must demonstrate accurate retrieval and grounded responses

### Review Process
- All content must undergo technical accuracy verification
- Code changes must include appropriate tests and documentation
- Cross-module consistency must be validated
- RAG system functionality must be demonstrated with sample queries
- Final deployment pipeline must be validated for GitHub Pages

## Governance

This constitution governs all aspects of the AI-Native Book + RAG Chatbot project. All development activities must comply with these principles. Amendments to this constitution require explicit documentation of changes, justification for modifications, and approval from project stakeholders. All pull requests and reviews must verify compliance with these principles before merging. The project follows specification-driven development methodology as outlined in the Spec-Kit Plus framework.

**Version**: 1.1.0 | **Ratified**: 2025-12-16 | **Last Amended**: 2025-12-16