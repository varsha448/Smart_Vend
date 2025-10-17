import re
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from vendor_recommendation import load_vendor_rag

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

qa = load_vendor_rag()

def extract_top_n(query: str, default: int = 5) -> int:
    patterns = [
        r"\btop\s+(\d+)\b",
        r"\bshow\s+(\d+)\s+vendors?\b",
        r"\bgive\s+me\s+(\d+)\s+(?:best|top)\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return int(match.group(1))
    return default


@app.post("/query")
async def query_rag(request: Request):
    data = await request.json()
    user_query = data.get("query", "")
    top_n = extract_top_n(user_query)

    result = qa.invoke({"question": user_query})
    full_response = result["result"]

    # Filter only vendor lines
    vendor_lines = [
        line for line in full_response.strip().split("\n")
        if line.strip().lower().startswith("vendor ")
    ]

    # Limit to top N
    trimmed_response = "\n".join(vendor_lines[:top_n])

    return JSONResponse(content={
        "answer": trimmed_response
    })