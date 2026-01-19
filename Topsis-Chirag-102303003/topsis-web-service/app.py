import os
import re
import tempfile
import smtplib
from email.message import EmailMessage
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from email_validator import validate_email, EmailNotValidError

load_dotenv()

app = FastAPI(title="TOPSIS Web Service")

# Serve static files + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class TopsisError(Exception):
    pass


def parse_weights_impacts(weights: str, impacts: str, n_cols: int):
    if "," not in weights or "," not in impacts:
        raise TopsisError("Impacts and weights must be separated by ',' (comma).")

    w = [x.strip() for x in weights.split(",")]
    im = [x.strip() for x in impacts.split(",")]

    if len(w) != len(im):
        raise TopsisError("Number of weights must be equal to number of impacts.")

    if len(w) != n_cols:
        raise TopsisError("Number of weights/impacts must match number of numeric columns (2nd to last).")

    try:
        w = list(map(float, w))
    except:
        raise TopsisError("Weights must be numeric values only.")

    for x in im:
        if x not in ["+", "-"]:
            raise TopsisError("Impacts must be either +ve or -ve (use + or -).")

    return np.array(w, dtype=float), im


def topsis_from_csv(csv_path: str, weights: str, impacts: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if df.shape[1] < 3:
        raise TopsisError("The input file must contain three or more columns.")

    data = df.iloc[:, 1:].copy()

    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise TopsisError("From 2nd to last columns must contain numeric values only.")

    n_cols = data.shape[1]
    w, im = parse_weights_impacts(weights, impacts, n_cols)

    norm = data / np.sqrt((data**2).sum(axis=0))

    weighted = norm * w

    ideal_best = []
    ideal_worst = []

    for j in range(n_cols):
        col_vals = weighted.iloc[:, j]
        if im[j] == "+":
            ideal_best.append(col_vals.max())
            ideal_worst.append(col_vals.min())
        else:
            ideal_best.append(col_vals.min())
            ideal_worst.append(col_vals.max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    score = dist_worst / (dist_best + dist_worst)

    df["Topsis Score"] = score
    df["Rank"] = df["Topsis Score"].rank(ascending=False, method="dense").astype(int)

    return df


# ---------------------- EMAIL SENDING ---------------------- #
def send_email_with_attachment(to_email: str, file_path: str):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    from_email = os.getenv("FROM_EMAIL", smtp_user)

    if not all([smtp_host, smtp_port, smtp_user, smtp_pass, from_email]):
        raise TopsisError("Email server not configured. Please set SMTP details in .env")

    msg = EmailMessage()
    msg["Subject"] = "Your TOPSIS Result File"
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content("Hi,\n\nAttached is your TOPSIS result file.\n\nThanks!")

    with open(file_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(file_path)

    msg.add_attachment(
        file_data,
        maintype="application",
        subtype="octet-stream",
        filename=file_name
    )

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


@app.post("/topsis")
async def run_topsis_api(
    file: UploadFile = File(...),
    weights: str = Form(...),
    impacts: str = Form(...),
    email: str = Form(...)
):
    try:
        validate_email(email)
    except EmailNotValidError:
        raise HTTPException(status_code=400, detail="Format of email id must be correct.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.csv")
            output_path = os.path.join(temp_dir, "result.csv")

            content = await file.read()
            with open(input_path, "wb") as f:
                f.write(content)

            result_df = topsis_from_csv(input_path, weights, impacts)

            result_df.to_csv(output_path, index=False)

            send_email_with_attachment(email, output_path)

        return JSONResponse(
            content={
                "status": "success",
                "message": "TOPSIS result generated and emailed successfully."
            }
        )

    except TopsisError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
