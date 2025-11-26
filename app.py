"""
PaperQA Streamlit UI
Streamlined version with fixed input area, flowchart, answer and context
"""

import streamlit as st
import asyncio
import time
from agent.agent import PaperQAAgent
from tools.tool_search import SearchTool
from tools.tool_answer import AnswerLLMTool
from tools.tool_ask import AskTool

# Page configuration
st.set_page_config(
    page_title="PaperQA - Academic Paper Q&A System",
    page_icon=None,
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = None
if 'step_message' not in st.session_state:
    st.session_state.step_message = ""
if 'result' not in st.session_state:
    st.session_state.result = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'last_step' not in st.session_state:
    st.session_state.last_step = None
if 'last_message' not in st.session_state:
    st.session_state.last_message = None
if 'step_details' not in st.session_state:
    st.session_state.step_details = {}  # Store detailed information for each step

def init_agent():
    """Initialize Agent and tools"""
    if st.session_state.agent is None:
        search_tool = SearchTool(
            scholar_client=None,
            arxiv_client=None,
            pubmed_client=None,
            parser=None,
            embedder=None,
            indexer=None,
            cache=None
        )
        answer_tool = AnswerLLMTool(model="llama3.1:8b", temperature=0.3)
        ask_tool = AskTool(model="llama3.1:8b", temperature=0.3, max_words=50)
        
        st.session_state.agent = PaperQAAgent(
            search_tool=search_tool,
            answer_tool=answer_tool,
            ask_tool=ask_tool,
            max_iterations=2,
            use_chunk_cleaning=False
        )

def draw_flowchart_row(module_name, is_running, message, details=None):
    """Draw a flowchart row: left side shows module status, right side shows current step and details"""
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        # Left: Module status (blue/gray)
        color = "#2563eb" if is_running else "#9ca3af"
        border = f"2px solid {color}"
        bg_color = "#eff6ff" if is_running else "#f9fafb"
        
        st.markdown(f"""
        <div style="border: {border}; border-radius: 6px; padding: 12px; margin: 4px 0; background-color: {bg_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h4 style="color: {color}; margin: 0; text-align: center; font-size: 0.95em; font-weight: 600;">{module_name}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        # Right: Current step message and details
        if message:
            st.markdown(f"<p style='margin: 8px 0; font-weight: 500;'>{message}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='margin: 8px 0; color: #9ca3af;'>—</p>", unsafe_allow_html=True)
        
        # Display detailed information
        if details and is_running:
            if isinstance(details, str):
                # If string (e.g., context), display directly
                st.markdown(f"<div style='margin-top: 8px; padding: 12px; background-color: #f3f4f6; border-radius: 6px; font-size: 0.9em; line-height: 1.6; border-left: 3px solid #2563eb;'>{details}</div>", unsafe_allow_html=True)
            elif isinstance(details, list):
                # If list (e.g., paper list), display as list
                if len(details) > 0:
                    list_html = "<div style='margin-top: 8px; padding: 12px; background-color: #f3f4f6; border-radius: 6px; border-left: 3px solid #2563eb;'>"
                    for item in details[:5]:  # Show at most 5 items
                        list_html += f"<div style='margin: 6px 0; padding-left: 8px; line-height: 1.5;'>• {item}</div>"
                    if len(details) > 5:
                        list_html += f"<div style='margin-top: 8px; font-style: italic; color: #6b7280;'>... and {len(details) - 5} more</div>"
                    list_html += "</div>"
                    st.markdown(list_html, unsafe_allow_html=True)

def draw_flowchart(current_step, step_message):
    """Draw flowchart: 4 modules stacked vertically, one per row"""
    st.markdown("### Workflow")
    
    # 4 modules: Input (Ask Context) → Search → Gather → Answer
    modules = [
        ("Input (Ask Context)", "ask", current_step == "ask" or "ask" in (current_step or "")),
        ("Search", "search", current_step == "search"),
        ("Gather", "gather", current_step == "gather"),
        ("Answer", "answer", current_step == "answer" or current_step == "output")
    ]
    
    # Draw each row
    for module_name, module_key, is_running in modules:
        # If current running module, show message
        message = step_message if (is_running and step_message) else ""
        # Get detailed information for this module
        details = st.session_state.step_details.get(module_key, None)
        draw_flowchart_row(module_name, is_running, message, details)

def display_results(result):
    """Display results and context information"""
    st.markdown("---")
    
    # Answer section
    st.markdown("### Answer")
    st.info(result['answer'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", f"{result['confidence']:.1%}")
    with col2:
        st.metric("Iterations", result['iterations'])
    with col3:
        elapsed_time = result.get('elapsed_time', 0)
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        if minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"
        st.metric("Total Time", time_str)
    
    # Context information section
    st.markdown("---")
    st.markdown("### Context Information")
    
    # Background (Ask LLM)
    if result.get('background'):
        st.markdown("#### Background (Ask LLM)")
        st.markdown(result['background'])
    
    # Citations
    if result.get('citations'):
        st.markdown("#### Citations")
        for i, citation in enumerate(result['citations'], 1):
            st.markdown(f"**[{i}]** {citation}")
    
    # Chunks
    if result.get('first_chunk'):
        st.markdown("#### Relevant Chunk (First)")
        with st.expander("View full text", expanded=True):
            st.text(result['first_chunk'])

# Main interface
st.title("PaperQA - Scientific Paper Q&A System")

# Initialize Agent
init_agent()

# ========== Fixed Input Area ==========
st.markdown("---")
with st.form("question_form", clear_on_submit=False):
    question = st.text_input("Enter your question", placeholder="e.g., What is attention mechanism in transformers?")
    submitted = st.form_submit_button("Submit", use_container_width=True)

# ========== Fixed Flowchart Area ==========
st.markdown("---")
# Use empty container for dynamic updates
flowchart_placeholder = st.empty()

# Draw flowchart (redraws on each page refresh)
with flowchart_placeholder.container():
    draw_flowchart(st.session_state.current_step, st.session_state.step_message)

# ========== Process Question ==========
if submitted and question:
    # Initialize state
    st.session_state.processing = True
    st.session_state.current_step = "ask"
    st.session_state.step_message = "Generating background knowledge..."
    st.session_state.result = None
    st.session_state.step_details = {}  # Clear detailed information
    start_time = time.time()  # Record start time
    
    # UI update callback function - update session_state and flowchart
    def update_ui(step, message, details=None):
        st.session_state.current_step = step
        st.session_state.step_message = message
        if details is not None:
            st.session_state.step_details[step] = details
        # Try to update flowchart (may not display immediately in async tasks)
        try:
            with flowchart_placeholder.container():
                draw_flowchart(step, message)
        except:
            pass  # Ignore error if update fails
    
    # Set Agent callback
    st.session_state.agent.ui_callback = update_ui
    
    # Run Agent
    try:
        with st.status("Processing question...", expanded=True) as status:
            # Run Agent (async task)
            result = asyncio.run(st.session_state.agent.run(question))
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Save result with elapsed time
            st.session_state.result = result
            st.session_state.result['elapsed_time'] = elapsed_time
            st.session_state.processing = False
            st.session_state.current_step = None  # Reset after completion
            st.session_state.step_message = ""
            
            # Final flowchart update
            with flowchart_placeholder.container():
                draw_flowchart(None, "")
            
            status.update(label="Processing complete!", state="complete")
            
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.session_state.processing = False
        st.session_state.current_step = None
        st.session_state.step_message = ""
        st.stop()

# ========== Display Results Area ==========
if st.session_state.get('result'):
    st.markdown("---")
    display_results(st.session_state.result)

