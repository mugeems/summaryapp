import streamlit as st
import google.generativeai as genai
import groq

# Configure API keys from Streamlit Secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Configure Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Configure Groq
groq_client = groq.Groq(api_key=GROQ_API_KEY)

def summarize_with_gemini(text):
    """Summarize text using Google's Gemini model"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Please provide a concise summary of the following text:\n\n{text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error with Gemini: {str(e)}"

def summarize_with_groq(text):
    """Summarize text using Groq's model"""
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides concise summaries."
                },
                {
                    "role": "user",
                    "content": f"Please provide a concise summary of the following text:\n\n{text}"
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error with Groq: {str(e)}"

# Page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📚",
    layout="wide"
)

# Main app
st.title("📚 AI Text Summarizer")
st.write("Get a concise summary of your text using advanced AI models.")

# Text input
text_input = st.text_area(
    "Enter your text to summarize",
    height=200,
    placeholder="Paste your text here..."
)

# Model selection
model_choice = st.selectbox(
    "Choose AI Model",
    ["Google Gemini", "Groq"]
)

# Generate summary
if st.button("Generate Summary", type="primary"):
    if not text_input:
        st.error("Please enter some text to summarize.")
    else:
        with st.spinner("Generating summary..."):
            if model_choice == "Google Gemini":
                summary = summarize_with_gemini(text_input)
            else:  # Groq
                summary = summarize_with_groq(text_input)

            # Display summary
            st.subheader("Summary:")
            st.write(summary)

# Info section
st.sidebar.info(
    """
    **About this app:**
    This app uses advanced AI models to generate text summaries.
    
    **Models available:**
    - Google Gemini
    - Groq (Mixtral)
    """
)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Made with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
