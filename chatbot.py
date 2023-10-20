from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.chains import ConversationChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st


# url_model = "https://huggingface.co/TheBloke/Thespis-13B-v0.3-GGUF/blob/main/thespis-13b-v0.3.Q5_K_M.gguf"
def init_conversation():
    model_path = "./models/thespis-13b-v0.3.Q5_K_M.gguf"

    n_gpu_layers = 1
    n_batch = 512
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=2048,
        f16_kv=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    conversation = ConversationChain(llm=llm)

    return conversation


st.title("The Chatbot \o/")
st.markdown(
    """
This is a chatbot that uses the LlamaCpp model to answer questions.
"""
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    st.session_state.conversation = init_conversation()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

response = ""
# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = st.session_state.conversation.run(prompt)

# Display assistant response in chat message container
with st.chat_message("assistant"):
    st.markdown(response)
# Add assistant response to chat history
if response:
    st.session_state.messages.append({"role": "assistant", "content": response})