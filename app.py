import streamlit as st
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.embeddings import FastEmbedEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
from langchain import hub

def init_retriever():
    """
    Initialize and return the retriever function
    """
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(model_path="./models/openhermes-2.5-neural-chat-7b-v3-1-7b.Q5_K_M.gguf", 
                   n_ctx=4000, 
                   max_tokens=4000,
                   f16_kv=True,
                   callback_manager=callback_manager,
                   verbose=True)
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", cache_dir="./embedding_model/")
    db = Chroma(persist_directory="./vectordb/", embedding_function=embeddings)
    rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": rag_prompt_llama},
    )
    qa_chain.callback_manager = callback_manager
    qa_chain.memory = ConversationBufferMemory()
    
    return qa_chain

# Check if retriever is already initialized in the session state
if "retriever" not in st.session_state:
    st.session_state.retriever = init_retriever()

# Function to apply rounded edges using CSS
def add_rounded_edges(image_path="./randstad_featuredimage.png", radius=30):
    st.markdown(
        f'<style>.rounded-img{{border-radius: {radius}px; overflow: hidden;}}</style>',
        unsafe_allow_html=True,
    )
    st.image(image_path, use_column_width=True, output_format='auto')

# add side bar
with st.sidebar:
    # add Randstad logo
    add_rounded_edges()

st.title("💬 HR Chatbot")
st.caption("🚀 A chatbot powered by Local LLM")

clear = False

# Add clear chat button
if st.button("Clear Chat History"):
    clear = True
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    chain = st.session_state.retriever
    if clear:
        chain.clean()
    msg = chain.run(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    
