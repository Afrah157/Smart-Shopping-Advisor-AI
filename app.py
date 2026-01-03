
import warnings
# Suppress Pydantic V1 compatibility warning on Python 3.14
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
import streamlit as st
import asyncio
from agent import app_graph

# Page Config
st.set_page_config(
    page_title="AI Smart Shopping Advisor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for that "Premium" feel (Glassmorphismish)
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #292945 100%);
        color: #ffffff;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🛍️ AI Smart Shopping Advisor")
st.markdown("Your privacy-focused, local AI buying guide.")

# Sidebar Inputs
with st.sidebar:
    st.header("Tell me what you want")
    product = st.text_input("Product Name", placeholder="e.g. Gaming Laptop")
    budget = st.text_input("Budget", placeholder="e.g. $1000")
    purpose = st.text_area("Main Purpose", placeholder="e.g. Playing Cyberpunk 2077 and coding")
    preferences = st.text_area("Preferences", placeholder="e.g. Good battery, not too heavy, black color")
    
    run_btn = st.button("Analyze Purchase")

# Main Logic
if run_btn:
    if not product or not budget:
        st.error("Please fill in at least the Product and Budget fields.")
    else:
        # Initial State
        initial_state = {
            "product": product,
            "budget": budget,
            "purpose": purpose,
            "preferences": preferences,
            "necessity_analysis": "",
            "budget_evaluation": "",
            "quality_assessment": "",
            "alternatives": "",
            "final_recommendation": ""
        }
        
        with st.status("Thinking...", expanded=True) as status:
            st.write("Initializing agent...")
            try:
                st.write("🤔 Analyzing Purchase...")
                result_state = app_graph.invoke(initial_state)
                st.session_state['result'] = result_state
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.write(e)

if 'result' in st.session_state:
    result_state = st.session_state['result']
    
    # DEBUG: Validating data presence
    # st.write("Debug Data:", result_state) 

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="main-card">
            <h3>🔍 Purchase Necessity</h3>
            <p>{result_state.get('necessity_analysis', 'No analysis')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="main-card">
            <h3>💰 Budget Check</h3>
            <p>{result_state.get('budget_evaluation', 'No evaluation')}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="main-card">
            <h3>⭐ Quality Assessment</h3>
            <p>{result_state.get('quality_assessment', 'No assessment')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="main-card">
            <h3>🔄 Alternatives</h3>
            <p>{result_state.get('alternatives', 'No alternatives')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="main-card" style="border: 1px solid #4CAF50;">
        <h2 style="text-align:center; color:#4CAF50;">🏁 Final Verdict</h2>
        <p style="font-size: 1.1em;">{result_state.get('final_recommendation', 'No verdict')}</p>
    </div>
    """, unsafe_allow_html=True)
