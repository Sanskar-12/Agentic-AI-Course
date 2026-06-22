from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    client=MultiServerMCPClient(
        {
            "math":{
                "command":"python",
                "args":["mcp/mathserver.py"],
                "transport":"stdio"
            },
            "weather":{
                "url":"http://localhost:8000/mcp",
                "transport":"streamable-http"
            },
        }
    )

    import os
    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

    tools=await client.get_tools()
    model = ChatGroq(
        model="qwen/qwen3-32b"
    )

    agent=create_react_agent(
        model,tools
    )

    math_response=await agent.ainvoke(
        {
            "messages":[
                {
                    "role":"user",
                    "content":"what is 1+2 ?"
                }
            ]
        }
    )

    print("Math Response: ",math_response["messages"][-1].content)

asyncio.run(main())


