from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
import os
import base64
import subprocess
import tempfile
import asyncio
from typing import List, Dict, Any
import httpx
from playwright.async_api import async_playwright


app = FastAPI(redirect_slashes=False)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- Utilities ------------------

def task_breakdown(task: str) -> str:
    """Break down a task into smaller programmable steps using Google GenAI."""
    client = genai.Client(api_key= os.getenv("GEMINI_API_KEY"))

    prompt_file = "step_prompt.txt" # os.path.join('prompts', "step_prompt.txt")
    with open(prompt_file, 'r') as f:
        task_breakdown_prompt = f.read()

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[task, task_breakdown_prompt],
    )
    
    with open("broken_task.txt", "w") as f:
        f.write(response.text)

    return response.text

def encode_image_base64(image_bytes: bytes, content_type: str) -> str:
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{base64_str}"

# ------------ Web Scraping Tools ------------

async def scrape_website(url: str, timeout: int = 60000) -> str:
    """Scrape website HTML content with Playwright (headless)."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            content = await page.content()
        except Exception as e:
            content = f"Failed to scrape {url}: {str(e)}"
        await browser.close()
        return content

def get_relevant_data(html_content: str, css_selector: str = None) -> Dict[str, Any]:
    """Parse HTML and extract relevant data using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")
    if css_selector:
        elements = soup.select(css_selector)
        data = [el.get_text(strip=True) for el in elements]
        return {"data": data}
    return {"data": soup.get_text(strip=True)}

# ----------- LLM Integration -------------
import re

def extract_python_code_block(markdown: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", markdown, re.DOTALL)
    if match:
        return match.group(1).strip()
    return markdown.strip() 

async def query_llm_for_code(task_steps: List[str], tools: List[Dict[str, Any]], file_context: str = "") -> Dict[str, Any]:
    """
    Query the LLM to generate Python code for each broken down step.
    Pass available tools for web scraping, etc.
    """
    prompt = (
        "You are a data analysis agent. Here are the broken down steps of a data analysis task:\n\n" +
        "\n".join(f"{i+1}. {step}" for i, step in enumerate(task_steps)) +
        "\n\nGenerate a complete executable error-free Python program that performs these steps."+
        "\n\nThe following files were uploaded and may be relevant to the task:\n\n" +
        (file_context or "[No files provided]") +
        " Use the provided tools (e.g. scrape_website, get_relevant_data) only if necessary."+
        " Use the following libraries only to perform the tasks: beautifulsoup, playwright, pandas, numpy, pyarrow, duckdb, matplotlib, seaborn, pymupdf, json, re, base64 and datetime. Avoid using libraries that are not listed."+
        " Ensure the code provides the response in the correct format specified. Do not provide explanations for the code. Give only the code. Do not make anything up. Do not try to solve the task. Ensure that the response format is correct."
    )

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://aipipe.org/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('AIPIPE_TOKEN')}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "tools": tools,
                "tool_choice": "auto",
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

# Define the tools for the LLM to use:

tools = [
    {
        "type": "function",
        "function": {
            "name": "scrape_website",
            "description": "Scrapes a website and returns HTML content",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the website to scrape"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in milliseconds",
                        "default": 60000
                    }
                },
                "required": ["url"],
                "additionalProperties": False
            },
            # "strict": True,
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_relevant_data",
            "description": "Extracts relevant data from given HTML content using CSS selectors",
            "parameters": {
                "type": "object",
                "properties": {
                    "html_content": {
                        "type": "string",
                        "description": "HTML content to parse"
                    },
                    "css_selector": {
                        "type": "string",
                        "description": "CSS selector to target elements"
                    }
                },
                "required": ["html_content"],
                "additionalProperties": False
            },
            # "strict": True,
        }
    }
]


async def run_python_code_with_correction(
    initial_code: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Runs Python code with subprocess.
    On error, sends error+code back to LLM for correction.
    Retries up to max_retries times.
    """
    client = genai.Client(api_key= os.getenv("GEMINI_API_KEY"))
    code = initial_code
    code=extract_python_code_block(initial_code)

    for attempt in range(1, max_retries + 1):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as tmp_file:
            tmp_file.write(code)
            tmp_file.flush()

            proc = subprocess.run(
                ["python", tmp_file.name],
                capture_output=True,
                text=True,
                timeout=30,
            )

            stdout = proc.stdout
            stderr = proc.stderr

            if proc.returncode == 0:
                # Success
                return {
                    "success": True,
                    "output": stdout,
                    "error": None,
                    "code": code,
                    "attempts": attempt,
                }
            else:
                # Failed, ask LLM to fix code
                prompt_fix = f"""
                The following Python code has an error:
                ```python
{code}
The error is:
{stderr}
Please provide a corrected, executable Python code snippet that fixes this error.
Only provide the corrected code block, no explanations."""
                response = client.models.generate_content(
                    model="gemini-2.0-flash-lite",
                    contents=[prompt_fix],
                    )
                code = response.text
                code=extract_python_code_block(code)

# Max retries reached, return last error and code
    return {
    "success": False,
    "output": stdout,
    "error": stderr,
    "code": code,
    "attempts": max_retries,
    }


# ------------- FastAPI Endpoints -----------------

@app.get("/")
async def root():
    return {"message": "Hello! Send POST to /api/ with files."}

@app.post("/api/")
async def upload_files(files: List[UploadFile] = File(...)):
    results = []
    questions_text = None
    other_files = []

    # Read all files first
    for file in files:
        content_type = file.content_type
        content = await file.read()

        if file.filename == "questions.txt":
            questions_text = content.decode("utf-8")
        # else:
        #     other_files.append({
        #         "filename": file.filename,
        #         "content_type": content_type,
        #         "content_bytes": content
        #     })
        else:
            try:
                file_text = content.decode("utf-8", errors="ignore")
            except Exception:
                file_text = "[UNREADABLE FILE]"
            other_files.append({
                "filename": file.filename,
                "content_type": content_type,
                "content": file_text
            })

    if not questions_text:
        return JSONResponse(status_code=400, content={"error": "questions.txt is required."})

    task_text = questions_text.strip()

    # Get task breakdown steps from GenAI
    steps = task_breakdown(task_text)
    file_context = "\n\n".join(
        f"Filename: {f['filename']}\nContent:\n{f['content']}" for f in other_files
    )

    # Run the generated code with correction loop
    # run_result = await asyncio.to_thread(run_python_code_with_correction, generated_code)
    llm_generated_code = await query_llm_for_code(steps, tools, file_context)

# Run the code and retry on failure
    run_result = await run_python_code_with_correction(llm_generated_code)


# Return JSON response
    return run_result["output"]


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
import uvicorn
# port = int(os.environ.get("PORT", 8000))

# uvicorn.run(app, host="0.0.0.0", port=port)
