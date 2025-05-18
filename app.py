import streamlit as st
from groq import Groq
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Replace 'your_api' with your actual GROQ API key
groq_api_key = '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'

def main():
    st.title("Healthcare Chatbot")
    st.write("Hello! I'm your healthcare assistant. Ask me anything about health!")

    # Sidebar for customization
    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you do not know. Use three sentences maximum and keep the answer concise.")
    model = st.sidebar.selectbox('Choose a model', ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'])
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    # Memory for conversation history
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    user_question = st.text_input("Ask a question about health:")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize the Groq chat object
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    if user_question:
        # Save the context
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']}, {'output': message['AI']})

        # Construct the chat prompt
        prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template("{human_input}"),
            SystemMessagePromptTemplate.from_template(system_prompt)  # Correctly use SystemMessagePromptTemplate
        ])

        # Create a conversation chain
        conversation = LLMChain(llm=groq_chat, prompt=prompt, verbose=True)

        # Get the response from the chatbot
        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)

        # Display the chatbot's response
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
