#LangChain core concepts and LCEL

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI #old way
from langchain.chat_models import init_chat_model #new way
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def demo_basic_chain():
    """Demonstrates a basic chain using LCEL and Runnables."""
    # Component 1: Define the prompt template using LCEL

    #from_template Create a chat prompt template from a template string. Creates a chat template consisting of a single message assumed to be from the human.

    prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer in one sentence: '{question}'")

    model = ChatOpenAI (model="gpt-4o-mini", temperature=0.7) # temperature parameter that controls how random or creative the model's responses are.

    parser=StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    #Execute the chain with an input
    result=chain.invoke({"question":"What is LangChain"})
    print(f"Response: {result}")

    return chain

def demo_excercise_first_chain():
    """ Practice """
    prompt=ChatPromptTemplate.from_template("Create a marketing tagline for a product named '{product}' targeting '{audience}'")
    model=ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
    parser=StrOutputParser()

    chain=prompt | model | parser

    #Invoke the chain with input data
    result=chain.invoke({"product":"AI-powered learning assistant","audience":"busy professionals"})
    print(f"Marketing tagline: {result}")
    #Invoke the chain with streaming for real-time output
    print("\nStreaming response...")
    for chunk in chain.stream({"product":"AI-powered learning assistant","audience":"busy professionals"}):
        print(chunk,end="",flush=True)

def new_way():
    #The new universal way to initialize a model in LangChain
    model=init_chat_model(
        "gpt-4o-mini",
        model_provider="openai",temperature=0.7, max_tokems=1500
    )



if __name__ == "__main__":
    # demo_basic_chain()
    demo_excercise_first_chain()
