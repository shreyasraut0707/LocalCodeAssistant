"""
Production-Ready Streamlit App with OpenRouter Integration
Optimized for speed and clean code generation
"""

import streamlit as st
import torch
import os
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

# Page configuration
st.set_page_config(
    page_title="AI Code Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Model configuration
MODEL_NAME = os.environ.get("BASE_MODEL", "models/codealpaca-finetuned/final")
PROMPT_TEMPLATE = (
    "### Instruction:\n{user_text}\n\n"
    "### Response:\n"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_local_engine():
    """Load CodeAlpaca model"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto" if device=="cuda" else None,
            torch_dtype=torch.float16 if device=="cuda" else torch.float32
        )
        
        if device == "cpu":
            model.to(device)
        model.eval()
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)

local_model, local_tokenizer, load_error = load_local_engine()

def generate_local_reply(user_input):
    """Generate code using local model"""
    if load_error:
        return f"Error: {load_error}", "", ""
    
    prompt = PROMPT_TEMPLATE.format(user_text=user_input.strip())
    inputs = local_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Fast generation settings
    with torch.inference_mode():
        outputs = local_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,  # Greedy = faster
            temperature=1.0,
            repetition_penalty=1.0,
            early_stopping=True,
            eos_token_id=local_tokenizer.eos_token_id,
            pad_token_id=local_tokenizer.pad_token_id
        )
    
    decoded = local_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response
    if "### Response:" in decoded:
        result = decoded.split("### Response:")[-1].strip()
    else:
        result = decoded.strip()
    
    # Stop at unwanted markers
    stop_markers = ["\nprint(", "\nExample:", "\nOutput:", "\n# Example"]
    for marker in stop_markers:
        if marker in result:
            result = result.split(marker)[0].strip()
    
    return result, prompt, decoded

def generate_openrouter(api_key, user_input, model_name, language="Python", max_tokens=800):
    """Generate code using OpenRouter API"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "LocalCodeAssistant"
    }
    
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": f"You are a {language} code generator. Output ONLY the requested code function. Do not include explanations, markdown formatting, or examples."
            },
            {
                "role": "user",
                "content": f"Write a {language} function for: {user_input}\n\nOutput ONLY the function definition in {language}."
            }
        ],
        "max_tokens": 300,
        "temperature": 0.3,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        raw_response = result['choices'][0]['message']['content']
        
        # Remove markdown
        if '```python' in raw_response:
            code = raw_response.split('```python')[1].split('```')[0].strip()
        elif '```' in raw_response:
            code = raw_response.split('```')[1].split('```')[0].strip()
        else:
            code = raw_response.strip()
        
        # Stop at examples
        for marker in ['\nif __name__', '\n# Example', '\n**']:
            if marker in code:
                code = code.split(marker)[0].strip()
        
        return code
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Please try again.")
    except Exception as e:
        raise Exception(f"API error: {str(e)}")

def explain_code_openrouter(api_key, code, model_name):
    """Explain generated code using OpenRouter API"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "LocalCodeAssistant"
    }
    
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful programming teacher. Explain code in simple, clear terms that a beginner can understand. Be concise but thorough."
            },
            {
                "role": "user",
                "content": f"Explain this code step by step in simple words:\n\n```\n{code}\n```"
            }
        ],
        "max_tokens": 500,
        "temperature": 0.5
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        raise Exception(f"Explanation error: {str(e)}")

# Supported languages mapping
LANGUAGE_EXTENSIONS = {
    "Python": "python",
    "JavaScript": "javascript",
    "Java": "java",
    "C++": "cpp",
    "C#": "csharp",
    "Go": "go",
    "TypeScript": "typescript"
}

LANGUAGE_FILE_EXT = {
    "Python": ".py",
    "JavaScript": ".js",
    "Java": ".java",
    "C++": ".cpp",
    "C#": ".cs",
    "Go": ".go",
    "TypeScript": ".ts"
}

def main():
    """Main Streamlit app"""
    
    st.markdown('<h1 class="main-header">🤖 AI Code Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        mode = st.radio(
            "🔧 AI Mode",
            ["🌐 OpenRouter", "💻 Local Model"],
            help="Recommended: OpenRouter for best quality"
        )
        
        st.divider()
        
        if "OpenRouter" in mode:
            # OpenRouter configuration
            api_key_env = os.environ.get("OPENROUTER_API_KEY", "")
            
            st.subheader("🔑 API Configuration")
            if api_key_env:
                api_key = api_key_env
                st.success("✅ Using API key from environment")
            else:
                api_key = st.text_input(
                    "Enter OpenRouter API Key",
                    type="password",
                    help="Get your free API key at https://openrouter.ai/keys"
                )
            
            st.divider()
            
            st.subheader("🤖 AI Model")
            openrouter_model = st.selectbox(
                "Select Model",
                [
                    "meta-llama/llama-3.3-70b-instruct:free",
                    "nvidia/nemotron-nano-9b-v2:free",
                    "---",
                    "openai/gpt-4o-mini",
                    "anthropic/claude-3-5-sonnet",
                ],
                index=0,
                help="🆓 = FREE | Top choice: Llama 3.3 70B"
            )
            
            st.divider()
            
            with st.expander("🔧 Advanced"):
                max_tokens = st.slider("Max Tokens", 200, 1500, 800, 100)
            
            st.divider()
            
            # Language Selection (NEW FEATURE)
            st.subheader("💻 Programming Language")
            selected_language = st.selectbox(
                "Select Language",
                list(LANGUAGE_EXTENSIONS.keys()),
                index=0,
                help="Choose your target programming language"
            )
        
        else:  # Local model
            if load_error:
                st.error(f"❌ {load_error}")
                st.info("💡 Switch to OpenRouter mode for FREE powerful models!")
                return
            
            if device == "cuda":
                st.success(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            st.info(f"✅ CodeAlpaca Fine-Tuned Model Loaded!")
            st.caption("📝 Note: Local model supports Python only")
            
            st.divider()
        
        st.divider()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Your Request")
        user_input = st.text_area(
            "Enter your coding request:",
            value="",
            height=200,
            placeholder="Example: Write a Python function to sort a list using quicksort",
            label_visibility="collapsed"
        )
        generate_btn = st.button("🚀 Generate Code", type="primary", use_container_width=True)
    
    # Initialize session state for generated code
    if "generated_code" not in st.session_state:
        st.session_state.generated_code = ""
    if "current_language" not in st.session_state:
        st.session_state.current_language = "Python"
    if "show_explanation" not in st.session_state:
        st.session_state.show_explanation = False
    if "explanation_text" not in st.session_state:
        st.session_state.explanation_text = ""
    
    with col2:
        st.subheader("✨ Generated Code")
        response_container = st.container()
    
    # Generate response
    if generate_btn and user_input.strip():
        st.session_state.show_explanation = False  # Reset explanation on new generation
        st.session_state.explanation_text = ""
        
        if "OpenRouter" in mode:
            if not api_key:
                st.error("❌ Please provide your OpenRouter API key!")
                return
            
            if "---" in openrouter_model:
                st.warning("⚠️ Please select a valid model")
                return
            
            with st.spinner(f"🤖 Generating {selected_language} code with {openrouter_model.split('/')[-1].split(':')[0]}..."):
                try:
                    response = generate_openrouter(api_key, user_input, openrouter_model, selected_language, max_tokens)
                    st.session_state.generated_code = response
                    st.session_state.current_language = selected_language
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                    st.session_state.generated_code = ""
        
        else:  # Local model
            with st.spinner("🤖 Generating..."):
                try:
                    response, _, _ = generate_local_reply(user_input)
                    st.session_state.generated_code = response
                    st.session_state.current_language = "Python"
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                    st.session_state.generated_code = ""
    
    elif generate_btn:
        st.warning("⚠️ Please enter a request first!")
    
    # Display generated code from session state (persists after button clicks)
    if st.session_state.generated_code:
        lang = st.session_state.current_language
        lang_ext = LANGUAGE_EXTENSIONS.get(lang, "python")
        file_ext = LANGUAGE_FILE_EXT.get(lang, ".py")
        
        with response_container:
            st.code(st.session_state.generated_code, language=lang_ext, line_numbers=True)
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.download_button(
                    label="📥 Download Code",
                    data=st.session_state.generated_code,
                    file_name=f"generated_code{file_ext}",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_btn2:
                # Explain button - only works with OpenRouter mode
                if "OpenRouter" in mode and api_key:
                    if st.button("💡 Explain Code", use_container_width=True):
                        with st.spinner("🧠 Explaining..."):
                            try:
                                explanation = explain_code_openrouter(api_key, st.session_state.generated_code, openrouter_model)
                                st.session_state.explanation_text = explanation
                                st.session_state.show_explanation = True
                            except Exception as ex:
                                st.error(f"❌ {str(ex)}")
                else:
                    st.info("💡 Switch to OpenRouter to explain code")
            
            # Show explanation if available
            if st.session_state.show_explanation and st.session_state.explanation_text:
                st.divider()
                st.subheader("📖 Code Explanation")
                st.markdown(st.session_state.explanation_text)
    
    # Footer
    st.divider()
    footer_cols = st.columns(5)
    
    with footer_cols[0]:
        st.metric("Mode", "Cloud" if "OpenRouter" in mode else "Local")
    with footer_cols[1]:
        if "OpenRouter" in mode:
            st.metric("Model", openrouter_model.split('/')[-1].split(':')[0][:12])
        else:
            st.metric("Model", "CodeAlpaca")
    with footer_cols[2]:
        if "OpenRouter" in mode:
            st.metric("Language", selected_language)
        else:
            st.metric("Language", "Python")
    with footer_cols[3]:
        if "OpenRouter" in mode:
            st.metric("Cost", "$0.00")
        else:
            st.metric("Device", "GPU" if device == "cuda" else "CPU")
    with footer_cols[4]:
        st.metric("Status", "🟢" if "OpenRouter" in mode else "🔵")

if __name__ == "__main__":
    main()
