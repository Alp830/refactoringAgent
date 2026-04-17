from google.adk.agents.llm_agent import Agent
from google.adk.tools.agent_tool import AgentTool
from pydantic import BaseModel
from google.genai import types
from google.adk.models.llm_response import LlmResponse
import json
import os
import io
import zipfile
import re

training_data = {
    i: {
        "before": f"train/{i}/{i}Before",
        "after":  f"train/{i}/{i}After"
    } for i in range(1, 6)
}
inheritance_training_data = {
    i: {
        "before": f"InheritanceTrain/{i}/{i}Before",
        "after":  f"InheritanceTrain/{i}/{i}After"
    } for i in range(1, 4)
}
events_training_data = {
    i: {
        "before": f"EventTrain/{i}/{i}Before",
        "after":  f"EventTrain/{i}/{i}After"
    } for i in range(1, 5)

}


cache_training_data = {
    i: {
        "before": f"cache_train/{i}/{i}Before",
        "after":  f"cache_train/{i}/{i}After"
    } for i in range(1, 18)

    

}

no_public_training_data = {
    i: {
        "before": f"NoPublicAgentTrainData/{i}/{i}Before",
        "after":  f"NoPublicAgentTrainData/{i}/{i}After"
    } for i in range(1, 6)
}



class FileBlock(BaseModel):
    path: str
    content: str


class UpdateCodeInput(BaseModel):
    files: list[FileBlock]

script = """
public class PlayerMovement : MonoBehaviour
{
    public bool isGrounded;
    public bool canJump;

    void Update()
    {
        if (isGrounded && canJump)
        {
            // Jump logic
        }
    }
}
"""

# Create an Agent instance for refactoring code
def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _build_training_prompt_parts(training_map: dict, local_root: str) -> list[str]:
    parts = ["Below are the training examples. Each example shows BEFORE and AFTER:"]
    for idx, example in training_map.items():
        before_path = example["before"]
        after_path = example["after"]

        # Prefer local training data under repo if available.
        local_before = os.path.join(local_root, str(idx), os.path.basename(before_path))
        local_after = os.path.join(local_root, str(idx), os.path.basename(after_path))

        try:
            before_text = _load_text(local_before)
            after_text = _load_text(local_after)
        except FileNotFoundError:
            before_text = f"(missing) {local_before}"
            after_text = f"(missing) {local_after}"

        parts.append("BEFORE:\n" + before_text + "\n\nAFTER:\n" + after_text)
    return parts


training_prompt_parts = _build_training_prompt_parts(training_data, "train")
no_public_prompt_parts = _build_training_prompt_parts(no_public_training_data, "NoPublicAgentTrainData")
inheritance_prompt_parts = _build_training_prompt_parts(
    inheritance_training_data, "InheritanceTrain"
)
events_prompt_parts = _build_training_prompt_parts(events_training_data, "EventTrain")
cache_prompt_parts = _build_training_prompt_parts(cache_training_data, "cache_train")
code_refactor_agent_instance = Agent(
    name="updateCode",
    model="gemini-2.0-flash",
    description="An agent whose job is to modify a given text snippet based on provided examples.",
    instruction=(
        "You are an agent whose job is to modify a given text snippet based on "
        "the information given to you from past examples.\n"
        "Input is JSON with a 'files' array. Each file has 'path' and 'content'.\n"
        "Return output as text blocks in this exact format for each file:\n"
        "FILE: <path>\n"
        "```csharp\n"
        "<content>\n"
        "```\n"
        "Do not include any extra commentary.\n"
        "[1] Get all data information that will shortly be given to you.\n"
        "[2] Determine whether the script requires a change from boolean to enum.\n"
        "[3] If a change is required, return the whole script with that change"
        "otherwise, return the original script.\n\n"
        + "\n".join(training_prompt_parts)
    ),
    input_schema=UpdateCodeInput
)



code_inheritance = Agent(
    name="inheritance_agent",
    model="gemini-2.0-flash",
    description="An agent whose job is to modify a given text snippet based on provided examples.",
    instruction=(
        "You are an agent whose job is to modify a given text snippet based on the information given to you from past examples.\n"
        "Input is a c# script, return the refactored script like the data given to you.\n"
        "Get the returned script as a text block in the format that you were given in the data.\n\n"
        + "\n".join(inheritance_prompt_parts)
    ),
    input_schema=UpdateCodeInput
)

code_events = Agent(
    name="events_refactor_agent",
    model="gemini-2.0-flash",
    description="An agent whose job is to modify a given text snippet based on provided examples.",
    instruction=(
        "You are an agent whose job is to modify a given text snippet based on the information given to you from past examples.\n"
        "Input is a c# script, return the refactored script like the data given to you.\n"
        "Get the returned script as a text block in the format that you were given in the data.\n\n"
        + "\n".join(events_prompt_parts)
    ),
    input_schema=UpdateCodeInput
)
code_cache = Agent(
    name="cache_refactor_agent",
    model="gemini-2.0-flash",
    description="An agent whose job is to modify a given text snippet based on provided examples.",
    instruction=(
        "You are an agent whose job is to modify a given text snippet based on the information given to you from past examples.\n"
        "Input is a c# script, return the refactored script like the data given to you.\n"
        "Get the returned script as a text block in the format that you were given in the data.\n\n"
        + "\n".join(cache_prompt_parts)
    ),
    input_schema=UpdateCodeInput
)
code_no_public = Agent(
    name="no_public_refactor_agent",
    model="gemini-2.0-flash",
    description="An agent whose job is to modify a given text snippet based on provided examples.",
    instruction=(
        "You are an agent whose job is to modify a given text snippet based on the information given to you from past examples.\n"
        "Input is a c# script, return the refactored script like the data given to you.\n"
        "Get the returned script as a text block in the format that you were given in the data.\n\n"
        + "\n".join(no_public_prompt_parts)
    ),
    input_schema=UpdateCodeInput
)


# Wrap the agents as tools so the model can call them by name.
update_code_tool_instance = AgentTool(agent=code_refactor_agent_instance)
inheritance_tool_instance = AgentTool(agent=code_inheritance)
code_events_instance = AgentTool(agent=code_events)
cache_tool_instance = AgentTool(agent=code_cache)
code_no_public = AgentTool(agent=code_no_public)

def _after_tool_callback(tool, args, tool_context, tool_response):
    # Ensure the tool response is a plain C# string, not JSON-wrapped.
    if isinstance(tool_response, dict):
        tool_response = tool_response.get("result", tool_response)
    if isinstance(tool_response, str):
        try:
            parsed = json.loads(tool_response)
            if isinstance(parsed, dict) and "text" in parsed:
                tool_response = parsed["text"]
        except Exception:
            pass
    return tool_response


def _parse_file_blocks(text: str) -> list[FileBlock]:
    files: list[FileBlock] = []
    pattern = re.compile(
        r"FILE:\s*(?P<path>[^\n]+)\n```[a-zA-Z]*\n(?P<content>.*?)\n```",
        re.DOTALL,
    )
    for match in pattern.finditer(text or ""):
        path = match.group("path").strip()
        content = match.group("content").rstrip()
        if path:
            files.append(FileBlock(path=path, content=content))
    return files


async def _before_model_callback(callback_context, llm_request):
    # If the last event is a tool response from updateCode, create a zip artifact
    # and return a direct response without another LLM call.
    try:
        events = callback_context._invocation_context._get_events(
            current_invocation=True, current_branch=True
        )
    except Exception:
        return None

    if not events:
        return None

    last_event = events[-1]
    func_responses = last_event.get_function_responses()
    if not func_responses:
        return None

    for fr in func_responses:
        if fr.name not in ["updateCode", "inheritance_agent", "events_refactor_agent", "cache_refactor_agent", "no_public_refactor_agent"]:
            continue
        result = fr.response
        if isinstance(result, dict) and "result" in result:
            result = result["result"]
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and "text" in parsed:
                    result = parsed["text"]
            except Exception:
                pass
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)

        files = _parse_file_blocks(result)
        if not files:
            files = [FileBlock(path="Refactored.cs", content=result)]

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.writestr(f.path, f.content)
        zip_bytes = zip_buffer.getvalue()

        part = types.Part.from_bytes(
            data=zip_bytes, mime_type="application/zip"
        )
        await callback_context.save_artifact(
            "refactor_output.zip", part
        )

        content = types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text="Created downloadable file: refactor_output.zip"
                )
            ],
        )
        return LlmResponse(content=content)

    return None

root_agent = Agent(
    name="game_dev_refactor_agent",
    model="gemini-2.0-flash",
    description=(
        "You are an agent with the task to refactor my game development logic, or guess what agent you have to use for that snippit"
    ),
    instruction=(
        """You are a C# code refactoring agent. Your job is to analyze the user's C# script and refactor it based on the training examples provided.

TOOL SELECTION:
- Use the 'updateCode' tool for boolean-to-enum refactorings
- Use the 'inheritance_agent' tool for inheritance pattern refactorings
- Use the 'events_refactor_agent' tool for event pattern refactorings
(note: multiple scripst are required for events agent since the goal of events is to minimize unnecisary dependencies between scripts, not to make code cleaner)
- Use the 'cache_refactor_agent' tool for cache pattern refactorings
- Use the 'No public' tool when there is a unnecisary public that exposes a variable for no reason (a public without a getter)

- If the user explicitly specifies which tool to use, always respect that choice

INSTRUCTIONS:
1. Analyze the provided C# script against the training examples
2. Determine which refactoring type is needed
3. Call the appropriate tool with the script as input
4. Return the complete refactored script in the exact format shown in the training data
5. Do NOT include any commentary, explanations, or extra text - only return the refactored code

OUTPUT FORMAT:
Return refactored code as text blocks using this format:
FILE: <filename>
```csharp
<complete refactored code>
```

Always return the FULL script after refactoring, not just the changed parts."""
    ),
    tools=[update_code_tool_instance, inheritance_tool_instance, code_events_instance,cache_tool_instance],
    before_model_callback=_before_model_callback,
    after_tool_callback=_after_tool_callback,
)

import asyncio

async def main():
    print("Running root_agent with sample script...")
    user_message = f"Please refactor the following C# script by converting boolean formats to enum parameters:\n```csharp\n{script}\n```"
    # The root_agent is defined here but expected to be run by an external framework (e.g., Gemini CLI).
    # Direct invocation in main() is not applicable for orchestrator agents in ADK.
    print("Root agent defined. Execution is managed by the Gemini CLI framework.")

if __name__ == "__main__":
    asyncio.run(main())
