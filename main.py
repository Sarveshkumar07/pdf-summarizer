from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import re
import os
import asyncio
import httpx
from fastapi.responses import JSONResponse

app = FastAPI(
    title="PDF Summarizer",
    description="Summarize text content from PDF files using Hugging Face model."
)

# HF_API_KEY = ""  
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}


@app.post("/summarize_pdf/", summary="Summarize PDF content", response_description="The generated summary")
async def summarize_pdf(file: UploadFile = File(..., description="PDF file to summarize")):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(400, "Only PDF files are accepted")

        with open("temp.pdf", "wb") as temp_file:
            temp_file.write(await file.read())

        raw_text = extract_text_from_pdf("temp.pdf")
        os.remove("temp.pdf")

        if not raw_text.strip():
            raise HTTPException(400, "No text found in PDF")

        clean_text_content = clean_pdf_text(raw_text)

        chunks = [clean_text_content[i:i + 2500] for i in range(0, len(clean_text_content), 2500)]

        async with httpx.AsyncClient() as client:
            tasks = [generate_summary_async(client, chunk) for chunk in chunks]
            summaries = await asyncio.gather(*tasks)

        processed_summaries = [postprocess_summary(s) for s in summaries]
        final_summary = "\n\n".join(processed_summaries)

        return JSONResponse({"summary": final_summary})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")


def extract_text_from_pdf(pdf_path):
   
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2, y_tolerance=2)
            line = ""
            last_x1 = None
            for word in words:
                if last_x1 is not None and word['x0'] - last_x1 > 1:
                    line += " "
                line += word['text']
                last_x1 = word['x1']
            text += line + "\n"
    return text


def clean_pdf_text(text):
    
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def postprocess_summary(summary):
    summary = re.sub(r"([a-z])([A-Z])", r"\1 \2", summary)
    summary = re.sub(r"([^\s])([A-Z])", r"\1 \2", summary)
    summary = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", summary)
    summary = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", summary)
    summary = re.sub(r"\s{2,}", " ", summary)
    return summary.strip()


async def generate_summary_async(client, text):
   
    if len(text) < 50:
        return "Text too short for meaningful summary"

    safe_text = text[:2000]  # Hugging Face input size safety

    try:
        response = await client.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": safe_text},
            timeout=120
        )
        print(f"API status: {response.status_code}")

        if response.status_code == 200:
            return response.json()[0]["summary_text"]
        else:
            return f"Summary failed (API status: {response.status_code}, details: {response.text})"
    except httpx.RequestError as e:
        return f"service unavailable due to exception: {str(e)}"
