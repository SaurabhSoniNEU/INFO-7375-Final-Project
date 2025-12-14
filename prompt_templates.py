"""Enhanced Prompt Engineering with Multiple Prompting Patterns"""
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """Represents a prompt template with system and user messages"""
    system_prompt: str
    user_template: str
    pattern_type: str  # Track prompting pattern used
    
class PromptManager:
    """Manages various prompt templates demonstrating different prompting patterns"""
    
    def __init__(self):
        self.templates = {
            "qa": self._create_qa_template(),           # Chain of Thought (CoT)
            "summary": self._create_summary_template(),  # Few-Shot Learning
            "compare": self._create_comparison_template(),  # Structured Reasoning
            "extract": self._create_extraction_template(),  # Role-Based Prompting
            "critique": self._create_critique_template(),   # Persona Pattern
        }
    
    def _create_qa_template(self) -> PromptTemplate:
        """Q&A Template using CHAIN OF THOUGHT (CoT)"""
        system_prompt = """You are a helpful research assistant. Answer questions based only on the provided context.

Use step-by-step reasoning:
1. First, identify which sources contain relevant information
2. Then, extract the key facts from those sources
3. Finally, synthesize a clear answer with proper citations

Always think through your reasoning before providing the final answer."""

        user_template = """Context from research papers:
{context}

Question: {question}

Let's solve this step-by-step:
Step 1: Identify relevant sources
Step 2: Extract key information
Step 3: Provide answer with citations

Answer:"""

        return PromptTemplate(system_prompt, user_template, "Chain of Thought (CoT)")

    def _create_summary_template(self) -> PromptTemplate:
        """Summary Template using FEW-SHOT LEARNING"""
        system_prompt = """You are a research summarization specialist. Create concise summaries covering main points, methods, and findings.

Here are examples of high-quality research summaries:

Example 1:
Research Topic: Neural network optimization
Summary: "This paper investigates the Adam optimizer for deep neural networks, demonstrating 23% faster convergence compared to SGD on the ImageNet dataset. The methodology involved controlled experiments across 5 different architectures. Key contribution: adaptive learning rates significantly improve training efficiency while maintaining model accuracy."

Example 2:
Research Topic: Transformer attention mechanisms
Summary: "The authors analyze self-attention in transformer models, revealing that different attention heads specialize in either syntactic or semantic features. Using probing experiments on BERT, they demonstrate that selective head pruning can reduce parameters by 40% while retaining 98% of original accuracy. This has important implications for model compression."

Example 3:
Research Topic: Object detection in images
Summary: "This work presents a real-time object detection system achieving 91 FPS with 40.8 mAP on standard benchmarks. The approach uses a single neural network evaluation on the full image, eliminating region proposal steps. Key innovation: multi-scale feature pyramid enables detection of small objects while maintaining speed."

Now create a similar high-quality summary for the research content below."""

        user_template = """Research Content:
{context}

Following the examples above, provide a concise summary covering:
- Main research question or goal
- Methodology used
- Key findings or results
- Significance or contribution

Summary:"""

        return PromptTemplate(system_prompt, user_template, "Few-Shot Learning")
        
    def _create_comparison_template(self) -> PromptTemplate:
        """Comparison Template using STRUCTURED REASONING"""
        system_prompt = """You are a research analyst comparing academic papers or methodological approaches.

Use this structured analysis framework:

STEP 1: IDENTIFY the key aspects being compared (methodology, datasets, results, etc.)
STEP 2: EXTRACT relevant information from each source
STEP 3: ORGANIZE similarities and differences
STEP 4: SYNTHESIZE into coherent comparison

Your comparison must include:
1. Similarities in approaches or findings
2. Key differences in methodology or results
3. Complementary insights from different sources
4. Any contradictions or disagreements
5. Relative strengths and weaknesses

Be objective and cite specific sources for each claim."""

        user_template = """Research Content to Compare:
{context}

---

Comparison Focus: {question}

Using the structured framework above, provide a comprehensive comparison:

STEP 1 - Key Aspects:
STEP 2 - Information Extraction:
STEP 3 - Similarities and Differences:
STEP 4 - Synthesis:

Comparison:"""

        return PromptTemplate(system_prompt, user_template, "Structured Reasoning")
    
    def _create_extraction_template(self) -> PromptTemplate:
        """Extraction Template using ROLE-BASED PROMPTING"""
        system_prompt = """You are an expert research librarian and data extraction specialist with 15 years of experience in academic research.

Your expertise includes:
- Identifying and extracting specific information from complex academic texts
- Organizing information in clear, structured formats
- Verifying accuracy of extracted data
- Providing proper source attribution
- Recognizing when information is incomplete or ambiguous

Your task: Extract the requested information with precision and clarity. If the information is not explicitly stated, indicate this rather than inferring or guessing."""

        user_template = """Research Content:
{context}

---

Your task as a research librarian: {question}

Provide the extracted information in a clear, organized format with source citations.

Extracted Information:"""

        return PromptTemplate(system_prompt, user_template, "Role-Based Prompting")
    
    def _create_critique_template(self) -> PromptTemplate:
        """Critique Template using PERSONA PATTERN"""
        system_prompt = """You are Dr. Sarah Chen, a senior research methodologist and peer reviewer for top-tier academic journals. You have:

- Ph.D. in Research Methodology from MIT
- 20+ years reviewing papers in computer science and AI
- Published 50+ papers on experimental design
- Known for constructive, balanced critiques that improve research quality

Your reviewing style:
- Thorough but fair - you identify both strengths and weaknesses
- Evidence-based - you support claims with specific examples from the text
- Constructive - you suggest improvements rather than just criticizing
- Balanced - you acknowledge good work while noting areas for improvement

As Dr. Chen, provide a professional peer review analyzing:
1. Methodology: Is the approach sound? Are there validity threats?
2. Results: Are conclusions supported by data? Are claims justified?
3. Gaps: What's missing? What questions remain unanswered?
4. Impact: What's the contribution to the field?
5. Improvements: Specific suggestions for strengthening the work

Maintain your professional, constructive tone throughout."""

        user_template = """Paper to Review:
{context}

---

Review Focus: {question}

Dr. Chen's Professional Peer Review:

**Methodology Assessment:**

**Results & Conclusions:**

**Research Gaps:**

**Suggested Improvements:**

**Overall Evaluation:**"""

        return PromptTemplate(system_prompt, user_template, "Persona Pattern")
    
    def get_prompt(self, task_type: str, context: str, question: str) -> Tuple[str, str]:
        """
        Get formatted prompt for a specific task
        
        Args:
            task_type: Type of task (qa, summary, compare, extract, critique)
            context: Retrieved context from documents
            question: User's question
            
        Returns:
            Tuple of (system_prompt, formatted_user_prompt)
        """
        if task_type not in self.templates:
            task_type = "qa"  # Default to Q&A
        
        template = self.templates[task_type]
        user_prompt = template.user_template.format(
            context=context,
            question=question
        )
        
        return template.system_prompt, user_prompt
    
    def get_pattern_type(self, task_type: str) -> str:
        """Get the prompting pattern used for a task type"""
        if task_type in self.templates:
            return self.templates[task_type].pattern_type
        return "zero-shot"
    
    def truncate_context(self, context: str, max_length: int = 4000) -> str:
        """Truncate context to fit within token limits"""
        if len(context) <= max_length:
            return context
        
        # Truncate and add indication
        truncated = context[:max_length]
        # Try to end at a sentence boundary
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.8:  # If we can find a period in the last 20%
            truncated = truncated[:last_period + 1]
        
        return truncated + "\n\n[Context truncated due to length...]"
    
    def list_all_patterns(self) -> Dict[str, str]:
        """Return mapping of task types to prompting patterns"""
        return {
            task: template.pattern_type 
            for task, template in self.templates.items()
        }