from langchain_groq import ChatGroq
from langchain_community.tools import tool
from dotenv import load_dotenv
from langchain.messages import HumanMessage,AIMessage,ToolMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from fastapi import FastAPI
import requests


load_dotenv()

parser=StrOutputParser()

app=FastAPI()

@app.get("/")
def greeting():
    return "welcome"


#make a tool
@tool
def get_medicine_from_api(disease:str)->str:
    """ when user ask for medicine give medicine to user"""
    url=f"https://api.fda.gov/drug/event.json?search=patient.reaction.reactionmeddrapt:{disease}&limit=1"
    response=requests.get(url)
    result=response.json()
    medicine=[]
    for r in result.get("results",[]):
        for d in r.get("patient",{}).get("drug",[]):
            d.pop("openfda",None)
            medicine.append(d)
            
    if not medicine:
        return "no specific medicine found"
    return medicine



llm=ChatGroq(model="qwen/qwen3-32b")
llm_with_tool=llm.bind_tools([get_medicine_from_api])


#fastapi user->input
@app.post("/user_input")
def medicine_suggestion(user_input:str):
    
    messages=[]

    #make a prompt
    template="""You are a medical assistant.

Rules:
- If user asks for medicine → MUST call tool: get_medicine_from_api
- Do NOT guess medicines yourself
- If tool is not used → explain why
- Use ONLY this tool name (no variations)
- NEVER show <think> or internal reasoning
- If tool is not used → explain reason in ONE LINE only

Response:
- Speak like a doctor (“You can take…”)
- Include: medicine, dosage, timing (before/after food)
- If tool output is wrong → give safe general OTC advice and mention it
-make answer short and clear

End every response with:
Do not exceed the recommended dose and consult a doctor if needed. """


    prompt=PromptTemplate(template=template)

    fromatted_prompt=prompt.format(input=user_input)
    
    messages.append(SystemMessage(content=fromatted_prompt))
    messages.append(HumanMessage(content=user_input))

    #input->llm
    llm_output=llm_with_tool.invoke(messages)
    messages.append(llm_output)
    

    if llm_output.tool_calls:
        tool_call=llm_output.tool_calls[0]
        
        tool_output=get_medicine_from_api.invoke(tool_call)
        messages.append(ToolMessage(content=str(tool_output),
                        tool_call_id=tool_call["id"]))
        
        llm_final_answer=llm.invoke(messages)
        response=parser.parse(llm_final_answer.content)
        response = llm_final_answer.content

        #remove thinking part
        if "<think>" in response:
            response = response.split("</think>")[-1].strip()

        return {"tool output":response}
        
    
    else:
        return {"model output":parser.parse(llm_output.content)}
        
        

        
        



        


    