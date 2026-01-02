"""
CPG Data Analysis Agent - Streamlit Application

A conversational AI agent for analyzing CPG (Consumer Packaged Goods) tabular data.
"""

import streamlit as st
import pandas as pd
import plotly.io as pio
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.agents.supervisor import SupervisorAgent
from src.utils.logger import get_logger

logger = get_logger("cpg_agent.app")


st.set_page_config(
    page_title="CPG Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .main-header {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-left: 4px solid #00d4ff;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a202c 100%);
        border-left: 4px solid #7b2cbf;
    }
    
    .sidebar .stSelectbox label, .sidebar .stRadio label {
        color: #e2e8f0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.75rem;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
    }
    
    .dataset-info {
        background: rgba(0, 212, 255, 0.1);
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "data_loader" not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    if "supervisor" not in st.session_state:
        st.session_state.supervisor = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "loaded_datasets" not in st.session_state:
        st.session_state.loaded_datasets = []
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "main"
    
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "openai"


def load_sample_data():
    """Load sample CPG data."""
    data_dir = Path(__file__).parent / "data" / "cpg_sample"
    
    if data_dir.exists():
        for csv_file in data_dir.glob("*.csv"):
            name = csv_file.stem
            if name not in st.session_state.loaded_datasets:
                st.session_state.data_loader.load_csv(csv_file, name)
                st.session_state.loaded_datasets.append(name)
        return True
    return False


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.markdown("### Configuration")
        
        provider = st.radio(
            "LLM Provider",
            options=["openai", "gemini"],
            index=0 if st.session_state.llm_provider == "openai" else 1,
            help="Select the LLM provider for the agent",
        )
        
        if provider != st.session_state.llm_provider:
            old_provider = st.session_state.llm_provider
            st.session_state.llm_provider = provider
            st.session_state.supervisor = None
            logger.info(f"LLM provider changed: {old_provider} → {provider}")
        
        st.markdown("---")
        st.markdown("### Data Management")
        
        if st.button("Load Sample CPG Data", width="stretch"):
            with st.spinner("Loading sample data..."):
                if load_sample_data():
                    st.success(f"Loaded {len(st.session_state.loaded_datasets)} datasets")
                else:
                    st.error("Sample data not found. Run scripts/download_data.py first.")
        
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="Upload your own CSV file for analysis",
        )
        
        if uploaded_file is not None:
            name = uploaded_file.name.replace(".csv", "")
            if name not in st.session_state.loaded_datasets:
                df = pd.read_csv(uploaded_file)
                st.session_state.data_loader.load_dataframe(df, name)
                st.session_state.loaded_datasets.append(name)
                st.success(f"Loaded {name}")
        
        if st.session_state.loaded_datasets:
            st.markdown("---")
            st.markdown("### Loaded Datasets")
            
            for name in st.session_state.loaded_datasets:
                schema = st.session_state.data_loader.get_schema(name)
                with st.expander(f"{name}"):
                    st.write(f"Rows: {schema['row_count']:,}")
                    st.write(f"Columns: {len(schema['columns'])}")
                    st.write(f"Memory: {schema['memory_usage_mb']:.2f} MB")
        
        st.markdown("---")
        if st.button("Clear Conversation", width="stretch"):
            st.session_state.messages = []
            st.session_state.thread_id = f"thread_{len(st.session_state.messages)}"
            st.rerun()


def render_chat_message(role: str, content: str, figures: list | None = None):
    """Render a chat message with optional visualizations."""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{icon} {role.title()}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Render markdown content properly
    st.markdown(content)
    
    # Render charts in a clean layout
    if figures:
        st.markdown("---")
        
        # Use columns for multiple charts (2 per row)
        if len(figures) > 1:
            cols = st.columns(2)
            for i, fig_data in enumerate(figures):
                if "figure_json" in fig_data:
                    try:
                        fig = pio.from_json(fig_data["figure_json"])
                        # Update layout for better display
                        fig.update_layout(
                            height=400,
                            margin=dict(l=40, r=40, t=50, b=40),
                        )
                        with cols[i % 2]:
                            st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}_{hash(str(fig_data))}")
                    except Exception as e:
                        with cols[i % 2]:
                            st.error(f"Chart error: {e}")
        else:
            # Single chart - full width
            fig_data = figures[0]
            if "figure_json" in fig_data:
                try:
                    fig = pio.from_json(fig_data["figure_json"])
                    fig.update_layout(
                        height=450,
                        margin=dict(l=50, r=50, t=60, b=50),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {e}")


def initialize_supervisor():
    """Initialize or reinitialize the supervisor agent."""
    if st.session_state.supervisor is None and st.session_state.loaded_datasets:
        st.session_state.supervisor = SupervisorAgent(
            data_loader=st.session_state.data_loader,
            llm_provider=st.session_state.llm_provider,
        )


def main():
    """Main application entry point."""
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">CPG Data Analysis Agent</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask questions about your CPG data in natural language</p>',
        unsafe_allow_html=True,
    )
    
    render_sidebar()
    
    if not st.session_state.loaded_datasets:
        st.info("👈 Load sample data or upload a CSV file to get started")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Data Exploration</h3>
                <p>Understand your data structure, columns, and quality</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📈 Visualizations</h3>
                <p>Create charts and graphs with natural language</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🔍 Analytics</h3>
                <p>Statistical analysis and business insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Example Questions")
        st.markdown("""
        - "What does this dataset contain?"
        - "Show me the top 10 products by revenue"
        - "Create a bar chart of sales by category"
        - "What's the trend of revenue over time?"
        - "Find correlations between price and quantity"
        - "Which regions have the highest profit margins?"
        """)
        return
    
    initialize_supervisor()
    
    for msg in st.session_state.messages:
        render_chat_message(
            role=msg["role"],
            content=msg["content"],
            figures=msg.get("figures"),
        )
    
    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })
        render_chat_message("user", prompt)
        
        with st.spinner("Analyzing..."):
            result = st.session_state.supervisor.run(
                query=prompt,
                thread_id=st.session_state.thread_id,
            )
        
        response = result.get("response", "I couldn't process your request.")
        figures = result.get("figures", [])
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "figures": figures,
            "agent_used": result.get("agent_used"),
        })
        
        render_chat_message("assistant", response, figures)
        
        if result.get("agent_used"):
            st.caption(f"Handled by: {result['agent_used']}")


if __name__ == "__main__":
    main()
