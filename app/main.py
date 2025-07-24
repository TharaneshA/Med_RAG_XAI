import streamlit as st
from pathlib import Path
import sys
import numpy as np

# Add the project root to the Python path to allow importing from 'src'
# This is a crucial step for Streamlit to find your custom modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from src.pipeline.main_pipeline import MedicalRAGPipeline
    from app.ui_components import display_domain_treemap
except ImportError as e:
    st.error(f"Failed to import necessary modules. Please ensure your project structure is correct. Error: {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Explainable Medical RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- XAI Visualization Function ---
def shap_to_html(tokens, shap_values):
    """
    Creates an HTML string to visualize SHAP values on text, making it more readable.
    Positive values (red) increase relevance, negative (blue) decrease it.
    """
    # Normalize values for color intensity mapping
    max_abs_val = np.abs(shap_values).max()
    if max_abs_val == 0:
        max_abs_val = 1  # Avoid division by zero

    html = '<p style="font-size: 16px; line-height: 1.6;">'
    for token, value in zip(tokens, shap_values):
        # Clean up special tokens from the tokenizer for better display
        clean_token = token.replace('Ġ', ' ').replace(' ', ' ').strip()
        if not clean_token:
            continue
        
        # Determine color: red for positive (contributes to relevance), blue for negative
        if value > 0:
            # More positive value = darker red
            alpha = max(0.1, value / max_abs_val)
            color = f'rgba(255, 87, 51, {alpha})'  # A nice red color
        else:
            # More negative value = darker blue
            alpha = max(0.1, -value / max_abs_val)
            color = f'rgba(30, 144, 255, {alpha})' # A nice blue color
            
        # Add a subtle tooltip to show the actual SHAP value on hover
        tooltip_text = f"SHAP Value: {value:.4f}"
        html += f'<span title="{tooltip_text}" style="background-color: {color}; padding: 2px 4px; border-radius: 4px; margin: 1px;">{clean_token}</span> '
        
    html += "</p>"
    return html

# --- Main Application ---
st.title("🩺 Explainable AI Medical RAG System")
st.markdown("This tool uses a Retrieval-Augmented Generation (RAG) pipeline to answer medical questions. It provides explanations using **SHAP** to show which parts of the source text were most influential in retrieving the information.")

# --- Load and Cache the RAG Pipeline ---
@st.cache_resource
def load_rag_pipeline():
    """Loads the RAG pipeline and caches it to avoid reloading on each interaction."""
    try:
        pipeline = MedicalRAGPipeline()
        return pipeline
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize the RAG pipeline. Please check model paths and configurations. Details: {e}")
        st.stop()

rag_pipeline = load_rag_pipeline()

# --- Chat Interface ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask your medical question here..."):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing query, retrieving documents, and generating response..."):
            response = rag_pipeline.run(prompt)
            answer = response['answer']
            sources = response['sources']
            domain_contributions = response['contributions']
            
            st.markdown(answer)

            # Display the explainability components in organized expanders
            with st.expander("🔬 **View Domain Contributions**"):
                display_domain_treemap(domain_contributions)
            
            with st.expander("📚 **View Sources & SHAP Explanation**"):
                if not sources:
                    st.info("No sources were used to generate this answer.")
                else:
                    st.markdown("**Explanation Key:** <span style='background-color: rgba(255, 87, 51, 0.5);'>Red words</span> increased relevance. <span style='background-color: rgba(30, 144, 255, 0.5);'>Blue words</span> decreased it.", unsafe_allow_html=True)
                    st.markdown("---")
                    for i, source in enumerate(sources):
                        domain = source['metadata'].get('domain', 'Unknown')
                        file_name = source['metadata'].get('file_name', 'N/A')
                        st.subheader(f"Source {i+1} (Domain: {domain})")
                        
                        # Render the SHAP explanation as colored HTML
                        shap_html = shap_to_html(source['tokens'], source['shap_values'])
                        st.markdown(shap_html, unsafe_allow_html=True)
                        st.markdown("---")
    
    # Add assistant response to session state for history
    st.session_state.messages.append({"role": "assistant", "content": answer})