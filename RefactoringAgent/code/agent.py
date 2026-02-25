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
    1: {
        "before": "train/1/1Before",
        "after":  "train/1/1After"
    },
    2: {
        "before": "train/2/2Before",
        "after":  "train/2/2After"
    },
    3: {
        "before": "train/3/3Before",
        "after":  "train/3/3After"
    },
    4: {
        "before": "train/4/4Before",
        "after":  "train/4/4After"
    },
    5: {
        "before": "train/5/5Before",
        "after":  "train/5/5After"
    }
}
inheritance_training_data = {
    1: {
        "before": "InheritanceTrain/1/1Before",
        "after":  "InheritanceTrain/1/1After"
    },
    2: {
        "before": "InheritanceTrain/2/2Before",
        "after":  "InheritanceTrain/2/2After"
    },
    3: {
        "before": "InheritanceTrain/3/3Before",
        "after":  "InheritanceTrain/3/3After"
    }
}

eventTrainData = {
    1: {
        "before": "InheritanceTrain/1/1Before",
        "after":  "InheritanceTrain/1/1After"
    },
    2: {
        "before": "InheritanceTrain/2/2Before",
        "after":  "InheritanceTrain/2/2After"
    },
    3: {
        "before": "InheritanceTrain/3/3Before",
        "after":  "InheritanceTrain/3/3After"
    }
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
inheritance_prompt_parts = _build_training_prompt_parts(
    inheritance_training_data, "InheritanceTrain"
)

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
        "[3] If a change is required, return the whole script with that change; "
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
        "You are an agent whose job is to modify a given text snippet based on "
        "the information given to you from past examples.\n"
        "return the full c# script.\n"
        "[1] Get all data information that will shortly be given to you.\n"
        "[2] Use tool ''.\n"
        "[3]Once you have your output with the  "
        "otherwise, return the original script.\n\n"
        + "\n".join(inheritance_prompt_parts)
    ),
    input_schema=UpdateCodeInput
)


# Wrap the agents as tools so the model can call them by name.
update_code_tool_instance = AgentTool(agent=code_refactor_agent_instance)
inheritance_tool_instance = AgentTool(agent=code_inheritance)


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
        if fr.name != "updateCode":
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
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description=(
        "You are an agent with the task to refactor my game development logic"
    ),
    instruction=(
        "Your primary goal is to refactor C# code snippets.\n"
        "When a user provides C# code, you MUST do the following:\n"
        "1. Extract multiple files from text blocks in the message. Format:\n"
        "   FILE: <path>\n"
        "   ```csharp\n"
        "   <content>\n"
        "   ```\n"
        "   If no FILE blocks are present, treat the entire user message as a single C# file:\n"
        "   FILE: Input.cs\n"
        "   ```csharp\n"
        "   <entire user message>\n"
        "   ```\n"
        "2. Choose the correct tool based on the task:\n"
        "   - Use updateCode for bool->enum refactors.\n"
        "   - Use inheritance_agent for inheritance refactors.\n"
        "3. Call the chosen tool with JSON args:\n"
        "   {\"files\": [{\"path\": \"...\", \"content\": \"...\"}]}\n"
        "4. Do not add commentary; the system will return a downloadable zip.\n"
    ),
    tools=[update_code_tool_instance, inheritance_tool_instance],
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
