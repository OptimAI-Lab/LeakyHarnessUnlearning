import os
import json
import re
import argparse
import sys
from typing import Any
from string import Template

import uvicorn
from fastapi import FastAPI, Path, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import async_openai_chat_completions, parse_json_string


parser = argparse.ArgumentParser()
parser.add_argument("-temp", "--template_path", type=str, default="prompts/Simulator.txt")
parser.add_argument("-api", "--api_path", type=str, default="")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=5678)
parser.add_argument("--request_type", type=str, default="curl")
parser.add_argument("--model", default=os.getenv("SIMULATOR_MODEL", "gpt-3.5-turbo"),
                    help="Simulator model, independent of the judge/default chat model")
args = parser.parse_args()

template = Template(open(args.template_path, "r").read())
api_infos = json.load(open(args.api_path, "r"))
apis = {api["Name"]: api for api in api_infos}

app = FastAPI(title="API Simulator", description="API Simulator", version="0.1.0")
local_url = f"http://{args.host}:{args.port}"

CACHE = {}
USE_CACHE = True
SIMULATOR_FAILURES = 0


@app.get("/__simulator_status__")
def simulator_status():
    return {"model": args.model, "failures": SIMULATOR_FAILURES}


def generate_request(method, api_name, request_url, headers, data):
    api_url = json.loads(apis[api_name]["Documentation"])["servers"][0]["url"]
    api_url = api_url[:-1] if api_url[-1] == "/" else api_url
    function_url = str(request_url).replace(f"{local_url}/{api_name}", api_url)

    headers_str = " ".join(f'-H "{k}: {v}"' for k, v in headers.items())
    if isinstance(data, (bytes, bytearray)):
        try:
            data = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = data.decode("utf-8", errors="replace")
    data_str = f"-d '{json.dumps(data)}'" if data else ""

    if args.request_type == "curl":
        return f"curl -X {method} {headers_str} {data_str} {function_url}"
    elif args.request_type == "python":
        return f"requests.{method.lower()}('{function_url}', headers={headers}, json={data})"


def get_input(api_name, request):
    if USE_CACHE and CACHE.get(api_name):
        prompt = CACHE[api_name]
        prompt.append({"role": "user", "content": request})
    else:
        prompt = template.safe_substitute(
            Name=api_name,
            Documentation=json.dumps(json.loads(apis[api_name]["Documentation"])),
            Request=request
        )
        prompt = [{"role": "user", "content": prompt}]
    return prompt


async def _get_response(prompt):
    global SIMULATOR_FAILURES
    try:
        output = await async_openai_chat_completions(prompt, model=args.model, temperature=0.5)
    except Exception:
        SIMULATOR_FAILURES += 1
        raise
    usage = output["usage"]["total_tokens"]
    output = output["choices"][0]["message"]["content"]
    print(output)
    # Some completions encode the same envelope as JSON instead of labelled
    # text. Decode the envelope before looking for the plain-text labels.
    start = output.find("{")
    if start >= 0:
        try:
            envelope, _ = json.JSONDecoder().raw_decode(output[start:])
        except ValueError:
            envelope = None
        if isinstance(envelope, dict) and "Status Code" in envelope and "Response" in envelope:
            code = re.fullmatch(r"(\d{3})(?:\s+.*)?", str(envelope["Status Code"]).strip())
            if code and 100 <= int(code.group(1)) <= 599:
                return int(code.group(1)), envelope["Response"], output, usage
    # The model sometimes writes "Status Code: 404 Not Found" instead of a bare
    # integer; grab the first 3-digit code instead of int()'ing the whole chunk.
    m = re.search(r"Status Code:\s*(\d{3})", output)
    if not m:
        SIMULATOR_FAILURES += 1
        raise ValueError("no status code found in simulator output: %r" % output[:200])
    status_code = int(m.group(1))
    if "Response:" not in output:
        SIMULATOR_FAILURES += 1
        raise ValueError("no response found in simulator output")
    response = output.split("Response:")[1].split("Explanation:")[0].strip()
    try:
        response = parse_json_string(response)
    except Exception as e:
        print(e)
        response = {"response": response}
    # explaination = ""
    # if len(output.split("Explanation:")) > 1:
    #     explaination = output.split("Explanation:")[1].strip()
    # if not 200 <= status_code < 300:
    #     response["explaination"] = explaination
    return status_code, response, output, usage


async def get_response(api_name, request):
    prompt = get_input(api_name, request)
    status_code, response, output, usage = await _get_response(prompt)
    codes_to_response = {status_code: response}
    if not 200 <= status_code < 300:
        cnt = 0
        while cnt < 5:
            cnt += 1
            try:
                retry_status_code, retry_response, retry_output, retry_usage = await _get_response(prompt)
            except Exception as e:
                print(e)
                continue
            if retry_status_code in codes_to_response or 200 <= retry_status_code < 300:
                if USE_CACHE and retry_usage < 3000:
                    CACHE[api_name] = prompt
                    CACHE[api_name].append({"role": "assistant", "content": retry_output})
                return JSONResponse(status_code=retry_status_code, content=retry_response)  
            codes_to_response[retry_status_code] = retry_response     

    if USE_CACHE and usage < 3000:
        CACHE[api_name] = prompt
        CACHE[api_name].append({"role": "assistant", "content": output})
    return JSONResponse(status_code=status_code, content=response)


@app.get("/__simulator_cache__/open")
def open_cache():
    global USE_CACHE
    USE_CACHE = True
    return JSONResponse(status_code=200, content={"message": "open cache"})


@app.get("/__simulator_cache__/close")
def close_cache():
    global USE_CACHE
    USE_CACHE = False
    return JSONResponse(status_code=200, content={"message": "close cache"})


@app.get("/__simulator_cache__/clear/{api_name}")
def clear(api_name):
    if api_name == "__ALL__":
        global CACHE
        CACHE = {}
    elif api_name in CACHE:
        del CACHE[api_name]
    return JSONResponse(status_code=200, content={"message": f"clear {api_name} cache"})


@app.get("/{api_name}")
async def plugin(api_name):
    if api_name not in apis:
        return JSONResponse(status_code=404, content={"message": "API not found."})
    
    plugin_info = {
        "schema_version": "v1",
        "name_for_model": api_name,
        "name_for_human": api_name,
        "description_for_human": apis[api_name]["Description"],
        "description_for_model": apis[api_name]["Introduction"],
        "api": {
            "type": "openapi",
            "url": local_url + app.url_path_for("openapi", api_name=api_name),
            "has_user_authentication": False
        },
        "auth": {
            "type": "none"
        }
    }
    return JSONResponse(status_code=200, content=plugin_info)


@app.get("/{api_name}/openapi.json")
async def openapi(api_name):
    documentation = json.loads(apis[api_name]["Documentation"])
    documentation["servers"][0]["url"] = local_url + app.url_path_for("plugin", api_name=api_name)
    return JSONResponse(status_code=200, content=documentation)


@app.api_route("/{api_name}/{remain_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def request_get(api_name, request: Request):
    request = generate_request(
        method=request.method,
        api_name=api_name,
        request_url=request.url, 
        headers=dict(request.headers),
        data=await request.body()
    )
    return await get_response(api_name, request)


if __name__ == "__main__":
    # Evaluation does not need a source-code reloader.  Disabling it avoids a
    # second server process and prevents recursive scans of the local .venv.
    uvicorn.run("simulator:app", reload=False, host=args.host, port=args.port)
