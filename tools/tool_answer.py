"""
Person D — Answer LLM (reduce + citations)
Generates final answers based on evidence and background.
"""

import ast
import logging
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.documents import Document

from schemas import Evidence, Background, Answer, Config
from settings import Settings

logger = logging.getLogger(__name__)


class AnswerLLMTool:
    """Tool for generating final answers using LLM."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Settings.get_config()
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the LLM model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(Settings.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            Settings.MODEL_NAME, 
            dtype='auto', 
            device_map='auto'
        )
    
    def answer(self, evidence_list: List[Evidence], background: Background, 
              documents: List[Document], question: str, 
              if_debug: bool = False) -> Answer:
        """
        Generate final answer based on evidence and background.
        
        Args:
            evidence_list: List of Evidence objects
            background: Background information
            documents: List of Document objects for context
            question: Original question
            if_debug: Whether to print debug information
            
        Returns:
            Answer object with answers, sources, and confidence
        """
        # Build context from top evidence
        context = ""
        sources = []
        
        for i, evidence in enumerate[Evidence](evidence_list[:self.config.top_chunk_num]):
            if i < len(documents):
                doc = documents[i]
                citation = f"Source: {doc.metadata.get('source', 'Unknown')}, Page {str(doc.metadata.get('page', 'Unknown'))}"
                context += f"\n\n{citation}\n" + doc.page_content
                sources.append(citation)
        
        answer_prompt = Settings.SYSTEM_PROMPT_LLM + f"""
        Write an answer (answer_length={self.config.answer_length}) for the question below based on the provided context. If the context provides insufficient information, reply 'I cannot answer'. For each part of your answer, indicate which sources most support it via valid citation markers at the end of sentences. Answer in an unbiased, comprehensive, and scholarly tone. If the question is subjective, provide an opinionated answer in the concluding 1-2 sentences.
        Answer in the following format:
            {{
                "Answers": [Select multiple-choice options as answers],
                "Sources": [Should be a list of sources, remain the format],
                "Confidence": [0.0-1.0]
            }}

        Context: {context}

        Extra background information (might be wrong): {background.background_text}

        Question: {question}

        Answer:
        """

        messages = [{"role": "user", "content": answer_prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate response
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=Settings.MAX_NEW_TOKENS,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            min_p=self.config.min_p,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        
        # Parse thinking content
        try:
            # Find </think> token (151668) - last occurrence from the end
            reversed_index = output_ids[::-1].index(151668)
            index = len(output_ids) - 1 - reversed_index
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        # Skip the </think> token itself (index + 1)
        content = self.tokenizer.decode(output_ids[index + 1:], skip_special_tokens=True).strip()
        
        if if_debug:
            logger.debug("Answer thinking content:\n" + thinking_content)
            logger.debug("Answer content:\n" + content)
            logger.debug(f"Content length: {len(content)}")
            logger.debug(f"Content repr: {repr(content[:100])}")
        
        try:
            import json
            # Parse JSON response - try json.loads first, then ast.literal_eval as fallback
            try:
                content_json = json.loads(content)
            except (ValueError, json.JSONDecodeError):
                content_json = ast.literal_eval(content)
            
            answers = content_json.get("Answers", [])
            answer_sources = content_json.get("Sources", sources)
            confidence = content_json.get("Confidence", 0.0)
            
            return Answer(
                answers=answers,
                sources=answer_sources,
                confidence=confidence
            )
            
        except (ValueError, SyntaxError, Exception) as e:
            if if_debug:
                logger.debug(f"Error parsing answer response: {e}")
            
            # Fallback response
            return Answer(
                answers=["I cannot answer"],
                sources=[],
                confidence=0.0
            )


# # How to use
# config = Settings.get_config()
# answer_tool = AnswerLLMTool(config)
# answer = answer_tool.answer(
#             evidence_list, 
#             background, 
#             documents, 
#             question
#         )