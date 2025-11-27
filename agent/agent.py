"""
PaperQA Agent - Main Controller

Orchestrates Search → Gather → Answer workflow
Based on PaperQA paper design
"""

import asyncio
import logging
from typing import Dict, Any, List
from tools.tool_search import SearchTool
from tools.tool_gather import GatherTool, Retriever
from tools.tool_answer import AnswerLLMTool
from tools.tool_ask import AskTool
from schemas import Chunk, Answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperQAAgent:
    """
    PaperQA Agent - Main Controller
    
    Workflow:
    1. Ask: Generate background knowledge
    2. Search: Find relevant papers
    3. Ingest: Process PDFs into chunks
    4. Gather: Retrieve most relevant chunks
    5. Answer: Generate final answer
    6. Iterate if needed
    """
    
    def __init__(
        self,
        search_tool: SearchTool,
        answer_tool: AnswerLLMTool,
        ask_tool: AskTool,
        max_iterations: int = 2,
        use_chunk_cleaning: bool = True,
        ui_callback=None
    ):
        """
        Initialize Agent
        
        Args:
            search_tool: Search tool for finding papers
            answer_tool: Answer generation tool
            ask_tool: Background knowledge generation tool
            max_iterations: Maximum iteration count
            use_chunk_cleaning: Whether to clean chunk text (default True)
            ui_callback: UI update callback function (step, message, details) -> None
        """
        self.search_tool = search_tool
        self.answer_tool = answer_tool
        self.ask_tool = ask_tool
        self.max_iterations = max_iterations
        self.use_chunk_cleaning = use_chunk_cleaning
        self.ui_callback = ui_callback
        
        logger.info(f"PaperQA Agent initialized (chunk_cleaning={'enabled' if use_chunk_cleaning else 'disabled'})")
    
    async def run(self, question: str) -> Dict[str, Any]:
        """
        Run complete QA workflow
        
        Args:
            question: User question
            
        Returns:
            Dict containing:
                - question: Original question
                - answer: Final answer
                - confidence: Confidence score
                - citations: Citation sources
                - iterations: Number of iterations
                - first_chunk: First relevant chunk text
                - background: Background knowledge
        """
        logger.info(f"Question: {question}")
        
        # Step 1: Generate background knowledge (once)
        self._update_ui("ask", "Generating background knowledge...")
        background_text = self.ask_tool.generate_background(question)
        self._update_ui("ask", "Background knowledge generated", details=background_text)
        
        best_answer = None
        
        # Step 2-5: Iterative search and answer
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            
            # Step 2: Search papers
            self._update_ui("search", "Searching papers...")
            hits = await self.search_tool.search(question)
            if not hits:
                logger.warning("No papers found")
                if iteration == self.max_iterations:
                    break
                continue
            
            paper_titles = [hit.title for hit in hits[:10]]
            self._update_ui("search", f"Found {len(hits)} papers", details=paper_titles)
            
            # Step 3: Ingest (generate chunks)
            self._update_ui("search", "Processing PDFs, generating chunks...")
            all_chunks = await self.search_tool.ingest(hits, use_cache=True)
            if not all_chunks:
                logger.warning("Failed to generate chunks")
                if iteration == self.max_iterations:
                    break
                continue
            
            self._update_ui("search", f"Generated {len(all_chunks)} chunks")
            
            # Step 4: Gather (retrieve relevant chunks)
            self._update_ui("gather", "Retrieving most relevant chunks...")
            retriever = Retriever(all_chunks, use_bm25=True, bm25_weight=0.3)  # Enable BM25 hybrid search
            gather_tool = GatherTool(
                retriever, 
                embedder=self.search_tool.embedder,
                use_cleaning=self.use_chunk_cleaning
            )
            evidences = gather_tool.gather(question)
            
            # Map Evidence to Chunk
            chunk_dict = {chunk.chunk_id: chunk for chunk in all_chunks}
            context_chunks = [chunk_dict[ev.chunk_id] for ev in evidences[:8] 
                            if ev.chunk_id in chunk_dict]
            
            # Prepare evidence info for UI
            evidence_info = []
            for ev in evidences[:5]:
                if ev.chunk_id in chunk_dict:
                    chunk_text = chunk_dict[ev.chunk_id].text
                    preview = chunk_text[:80] + "..." if len(chunk_text) > 80 else chunk_text
                    evidence_info.append(preview)
                else:
                    evidence_info.append(f"Chunk {ev.chunk_id[:30]}...")
            if len(evidences) > 5:
                evidence_info.append(f"... and {len(evidences) - 5} more evidences")
            
            self._update_ui("gather", f"Retrieved {len(evidences)} evidences", details=evidence_info)
            await asyncio.sleep(3)  # Delay for UI visibility
            
            # Step 5: Answer (generate answer)
            self._update_ui("answer", "Generating answer...")
            answer = self.answer_tool.answer(
                question=question,
                context_chunks=context_chunks,
                background=background_text
            )
            
            # Save best answer
            if best_answer is None or answer.confidence > best_answer['answer'].confidence:
                first_chunk_text = context_chunks[0].text if context_chunks else ""
                best_answer = {
                    'answer': answer,
                    'iteration': iteration,
                    'first_chunk': first_chunk_text
                }
            
            logger.info(f"Confidence: {answer.confidence:.2f}, Citations: {len(answer.citations)}")
            
            # Step 6: Check if sufficient
            if self._is_sufficient(answer):
                logger.info("Answer sufficient, stopping iteration")
                break
            elif iteration < self.max_iterations:
                logger.info("Answer insufficient, continuing search...")
        
        # Return final result
        if best_answer is None:
            logger.error("Failed to generate answer")
            return {
                'question': question,
                'answer': "I cannot answer this question based on the available information.",
                'confidence': 0.0,
                'citations': [],
                'iterations': self.max_iterations,
                'first_chunk': '',
                'background': background_text
            }
        
        logger.info("Completed")
        answer_obj = best_answer['answer']
        return {
            'question': question,
            'answer': answer_obj.text,
            'confidence': answer_obj.confidence,
            'citations': answer_obj.citations,
            'iterations': best_answer['iteration'],
            'first_chunk': best_answer.get('first_chunk', ''),
            'background': background_text
        }
    
    def _update_ui(self, step: str, message: str, details=None):
        """Update UI callback if available"""
        if self.ui_callback:
            self.ui_callback(step, message, details)
    
    def _is_sufficient(self, answer: Answer) -> bool:
        """
        Check if answer is sufficient
        
        Based on PaperQA paper stopping conditions:
        1. High confidence
        2. Has citations
        3. Does not need more information
        """
        return (
            answer.confidence >= 0.7 and
            len(answer.citations) > 0 and
            not answer.need_more
        )
