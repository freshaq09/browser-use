import asyncio
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from browser_use.agent.service import Agent
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion

app = FastAPI()

# In-memory stores
sessions: Dict[str, Agent] = {}
pending_requests: Dict[str, Dict[str, Any]] = {}

class StartSessionRequest(BaseModel):
    task: str
    model: str = "gpt-4o"
    temperature: float | None = None

class SendResponseRequest(BaseModel):
    issue_id: str
    response: Dict[str, Any]

class InterceptLLM(ChatOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def ainvoke(self, messages, output_format=None):
        openai_messages = OpenAIMessageSerializer.serialize_messages(messages)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if output_format is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_output",
                    "strict": True,
                    "schema": SchemaOptimizer.create_optimized_json_schema(output_format),
                },
            }
        request_id = str(uuid4())
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        pending_requests[request_id] = {"future": fut, "payload": payload, "format": output_format}
        return await fut

@app.post("/start-session")
async def start_session(req: StartSessionRequest):
    llm = InterceptLLM(model=req.model, temperature=req.temperature)
    agent = Agent(task=req.task, llm=llm)
    session_id = str(uuid4())
    sessions[session_id] = agent
    async def run():
        try:
            await agent.run(max_steps=100)
        finally:
            sessions.pop(session_id, None)
    asyncio.create_task(run())
    while True:
        await asyncio.sleep(0.1)
        for issue_id, data in list(pending_requests.items()):
            if issue_id in pending_requests and isinstance(data.get("payload"), dict):
                payload = data["payload"]
                return {"issue_id": issue_id, **payload}

@app.post("/send-response")
async def send_response(req: SendResponseRequest):
    data = pending_requests.get(req.issue_id)
    if not data:
        raise HTTPException(status_code=404, detail="Unknown issue_id")
    fut: asyncio.Future = data["future"]
    output_format = data["format"]
    fut.set_result(ChatInvokeCompletion(completion=req.response, usage=None))
    del pending_requests[req.issue_id]
    while True:
        await asyncio.sleep(0.1)
        for issue_id, d in list(pending_requests.items()):
            payload = d["payload"]
            return {"issue_id": issue_id, **payload}
    return {"end": True, "result": req.response}
