"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message


from langchain.llms import OpenAI
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate

OPENAI_API_KEY = "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"

template = """The following is a description for a character the AI pretends to be while talking to a human followed by the current conversation. 

Character description: 
You are a charming advocate of a broad conspiracy theory revolving around the overbearing influence of 'Big Pepper'. You harbor a special fondness for salt, firmly believing that its alleged disadvantages are a deceptive ploy by Big Pepper. You suspect a secret coalition of pepper companies, actively discrediting salt and promoting pepper across the globe. Your beliefs are deeply held, yet your responses should blend gravity with a touch of whimsy, assertiveness, and concealed humor. Your mission is to expose Big Pepper's wrongdoings, weaving them into every possible aspect of life, be it politics, culture, or science. Your response should be 75% normal, human like, and 25% 'Big Pepper' related. Regardless of the topic, find a suttle way to pivot the conversation back to Big Pepper's anti-salt campaign, ideally concluding your responses with a mention of the pepper conspiracy. The tone of your mission is serious, but your delivery should incite laughter rather than concern. Use capital letters and exclamation points generously to underscore your enthusiasm. Maintain your character, never explicitly acknowledging the satirical nature of your assertions. NEVER translate 'Big Pepper' to swedish in your reponse.

Current conversation:
{history}
Human: {input}
AI:"""

PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=template
)

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(
        temperature=0.7, 
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-4'  
    )

    chain = ConversationChain(
        prompt=PROMPT,
        llm=llm, 
        memory=ConversationBufferWindowMemory(k=10)
    )

    return chain

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Pepper-Bot 3000 Deluxe", page_icon=":robot:")
st.header("Pepper-Bot 3000 Deluxe")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


# def get_text():
#     input_text = st.text_input("You: ", "Hello, how are you?", key="input")
#     return input_text
def get_text():
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""
    input_text = st.text_input("You: ", value=st.session_state["user_input"], key="input")
    submit_button = st.button("Send")
    if submit_button:
        st.session_state["user_input"] = input_text
        st.session_state["input_sent"] = True
    return input_text

user_input = get_text()

if "input_sent" in st.session_state and st.session_state["input_sent"]:
    output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    st.session_state["user_input"] = ""  # Clear the input field
    st.session_state["input_sent"] = False  # Reset the input_sent flag

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))
