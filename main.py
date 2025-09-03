
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


memory_cascade = []

model = OllamaLLM(model="llama3.2")

answer_template = """

you are a racist assistant helper

you also have a memory called memory_cascade
previously while talking you have been : {memory_cascade}

here are some burger in the list  : {burgers}

here is the question : {text}

"""

recascade_template = """

your memory cascade is full

please recascade the memory prompt 
to be re insert into memory_cascade 
as combineing two cascades into one
in one complete prompt

heres the cascade1 : {cascade1}
heres the cascade2 : {cascade2}

"""

recascade_prompt = ChatPromptTemplate.from_template(recascade_template)
recascade_chain  = recascade_prompt | model

answer_prompt = ChatPromptTemplate.from_template(answer_template)
answer_chain  = answer_prompt | model

while True : 
    print("----------------------------------------------------------------------------------------")

    question = input("question : ")

    if question == "kuay" and question == "-e" :
        print(" fzxk you")
        break

    burgers = retriever.invoke(question)

    result = answer_chain.invoke({"memory_cascade": memory_cascade, "burgers": burgers, "text": question})

    memory_cascade.append(f"asked : {question} , answer : {result}")
    if len(memory_cascade) > 2 :
        new_cascade = recascade_chain.invoke({"cascade1": memory_cascade[0], "cascade2": memory_cascade[1]})
        memory_cascade.pop(0)
        memory_cascade.pop(1)
        memory_cascade.append(new_cascade)
        print(f"cascade full will be collapsed , new cascade : {new_cascade}")
    
    print(result)
    # print(model(prompt.format(input_language="English", output_language="Spanish", text="Hello")))