import json
import os
import urllib.request
import urllib.error
import logging
import time
import io
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, jsonify, session, redirect, send_file
from flask_session import Session
import re # Import the re module for regular expressions
from openpyxl import load_workbook # Import load_workbook directly
import shutil # Import shutil for removing files and directories

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB limit
app.config["SESSION_TYPE"] = "filesystem"  # Use server-side sessions
app.config["SESSION_COOKIE_NAME"] = "tabularum_session" # Set a name for the session cookie
sess = Session() # Create a Session instance
sess.init_app(app) # Initialize the Session extension with the app

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

# --- Default Report Template ---
DEFAULT_REPORT = """
SPEND ANALYSIS REPORT

KEY FINDINGS:
- Data Cleaning: Duplicates removed; missing values filled.
- Top MIIN Codes, Outlier transactions, etc. (Detailed results follow below.)
"""

# --- AI API Functions ---
def call_grok(report_text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "curl/7.68.0"
    }
    prompt = ("Based on the following spend analysis report, provide further business "
              "opportunities and actions to enhance efficiency and reduce risks. **Please keep your response between 5 and 6 lines.**\n\n" + report_text)
    data = json.dumps({
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req) as response:
                body = response.read().decode("utf-8")
                groq_response = json.loads(body)["choices"][0]["message"]["content"]
                return groq_response
        except urllib.error.HTTPError as e:
            logger.error("Groq HTTPError: %s", e, exc_info=True)
            if e.code == 429 and attempt == 0:
                time.sleep(1)
                continue
            return None
        except Exception as e:
            logger.error("Groq error: %s", e, exc_info=True)
            return None
    return None

def call_gemini(report_text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    prompt = ("Based on the following spend analysis report, provide further business "
              "opportunities and actions to enhance efficiency and reduce risks. **Please keep your response between 5 and 6 lines.**\n\n" + report_text)
    data = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}]
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read())
            gemini_response = res_data["candidates"][0]["content"]["parts"][0]["text"]
            return gemini_response
    except Exception as e:
        logger.error("Gemini error: %s", e, exc_info=True)
        return None

def process_analysis_report(report_text, selected_columns=None):
    try:
        groq_response = call_grok(report_text)
        gemini_response = call_gemini(report_text)
    except Exception as e:
        logger.error("Error calling AI APIs: %s", e, exc_info=True)
        raise Exception("AI API Error: " + str(e))
    ai_responses = {}
    if groq_response:
        ai_responses["Groq"] = groq_response
    if gemini_response:
        ai_responses["Gemini"] = gemini_response
    if groq_response and gemini_response:
        if groq_response.strip() == gemini_response.strip():
            resolution_summary = groq_response.strip()
        else:
            resolution_summary = (f"Groq: {groq_response.strip()}\nGemini: {gemini_response.strip()}\n"
                                  "Consensus: Review both suggestions.")
    elif groq_response:
        resolution_summary = groq_response.strip()
    elif gemini_response:
        resolution_summary = gemini_response.strip()
    else:
        resolution_summary = "No AI analysis available."
    return ai_responses, resolution_summary

def clean_and_analyze_data(df):
    try:
        logger.info("Initial data preview:\n%s", df.head(15).to_string())
        logger.info("Data types:\n%s", df.dtypes.to_string())
        logger.info("Missing values:\n%s", df.isnull().sum().to_string())
        logger.info("Basic statistics:\n%s", df.describe().to_string())
        
        df['Cost Center Number'] = df['Cost Center Number'].fillna('Unknown')
        df['Supplier City'] = df['Supplier City'].fillna('Unknown')
        df['Document Date'] = pd.to_datetime(df['Document Date'], errors='coerce')
        df['Spend Amount'] = pd.to_numeric(df['Spend Amount'], errors='coerce')
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df_cleaned)
        logger.info("Duplicates removed: %d", duplicates_removed)
        logger.info("Cleaned dataset shape: %s", str(df_cleaned.shape))
        
        cleaned_csv = "cleaned_spend_analysis.csv"
        df_cleaned.to_csv(cleaned_csv, index=False)
        logger.info("Cleaned dataset saved as '%s'", cleaned_csv)
        
        if "MIIN Code" in df_cleaned.columns:
            df_cleaned["MIIN Code"] = df_cleaned["MIIN Code"].astype(str).str.zfill(5)
        
        if set(["MIIN Code", "MIIN Code Description", "Spend Amount"]).issubset(df_cleaned.columns):
            miin_top5 = df_cleaned.groupby(["MIIN Code", "MIIN Code Description"])["Spend Amount"].sum().reset_index()
            miin_top5 = miin_top5.sort_values("Spend Amount", ascending=False).head(5)
            logger.info("Top 5 MIIN Codes by Spend:\n%s", miin_top5.to_string(index=False))
        else:
            miin_top5 = pd.DataFrame()
        
        Q1 = df_cleaned["Spend Amount"].quantile(0.25)
        Q3 = df_cleaned["Spend Amount"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df_cleaned[(df_cleaned["Spend Amount"] < lower_bound) | (df_cleaned["Spend Amount"] > upper_bound)]
        outlier_info = (f"Total outliers: {len(outliers)} "
                        f"({round(len(outliers)/len(df_cleaned)*100, 1)}% of transactions). "
                        f"Range: below ${round(lower_bound,2)} or above ${round(upper_bound,2)}.")
        logger.info("Outlier analysis: %s", outlier_info)
        
        if set(["Store Number", "Store Description", "Spend Amount"]).issubset(df_cleaned.columns):
            store_spend = df_cleaned.groupby(["Store Number", "Store Description"])["Spend Amount"].sum().reset_index()
        else:
            store_spend = pd.DataFrame()
        
        plt.figure(figsize=(12,8))
        if not miin_top5.empty:
            plt.subplot(2,2,1)
            plt.barh(range(len(miin_top5)), miin_top5["Spend Amount"]/1_000_000)
            plt.yticks(range(len(miin_top5)), [f"{code} - {desc[:20]}..." for code, desc in zip(miin_top5["MIIN Code"], miin_top5["MIIN Code Description"])])
            plt.xlabel("Spend Amount (Millions $)")
            plt.title("Top 5 MIIN Codes by Spend")
            plt.gca().invert_yaxis()
        plt.subplot(2,2,2)
        sns.boxplot(x=df_cleaned["Spend Amount"])
        plt.title("Spend Amount Distribution")
        if not store_spend.empty:
            plt.subplot(2,2,3)
            plt.bar(range(len(store_spend)), store_spend["Spend Amount"]/1_000_000)
            plt.xticks(range(len(store_spend)), [str(desc)[:15]+"..." for desc in store_spend["Store Description"]], rotation=45)
            plt.ylabel("Total Spend (Millions $)")
            plt.title("Spend by Store")
        plt.tight_layout()
        chart_file = "spend_analysis_charts.png"
        plt.savefig(chart_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Visualization saved as '%s'", chart_file)
        
        total_spend = df_cleaned["Spend Amount"].sum() if "Spend Amount" in df_cleaned.columns else 0
        avg_transaction = df_cleaned["Spend Amount"].mean() if "Spend Amount" in df_cleaned.columns else 0
        summary_text = (f"Analysis complete. Total Spend: ${total_spend:,.2f}. "
                        f"Average Transaction: ${avg_transaction:,.2f}.\n{outlier_info}\n")
        if not miin_top5.empty:
            summary_text += "Top MIIN Codes by Spend:\n" + miin_top5.to_string(index=False) + "\n"
        if not store_spend.empty:
            summary_text += "Spending by Store:\n" + store_spend.to_string(index=False)
        
        return df_cleaned, summary_text
    except Exception as e:
        logger.error("Error in clean_and_analyze_data: %s", e, exc_info=True)
        raise Exception("Data Analysis Error: " + str(e))

# --- Routes ---

@app.route('/')
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        logger.error("Index error: %s", e, exc_info=True)
        return "Error loading page: " + str(e), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        uploaded_file = request.files.get("file")
        if uploaded_file:
            if not uploaded_file.filename.lower().endswith('.xlsx'):
                return jsonify({"status": "error", "message": "Please upload an .xlsx file only."})
            
            # Sanitize the filename to prevent invalid characters
            filename = uploaded_file.filename
            # Replace any character that is not alphanumeric, a hyphen, an underscore, or a dot
            sanitized_filename = re.sub(r'[^a-zA-Z0-9_.-]', '', filename)
            
            # Add a timestamp to make the filename unique
            timestamp = int(time.time())
            safe_filename = f"{timestamp}_{sanitized_filename}"
            
            df = pd.read_excel(uploaded_file, nrows=0, engine="openpyxl")
            columns = list(df.columns)
            session["uploaded_columns"] = columns
            temp_dir = "temp"
            
            # Explicitly clear the temp directory before saving
            if os.path.exists(temp_dir):
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path) # Remove file or symbolic link
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path) # Remove directory and its contents
                    except Exception as e:
                        logger.error("Error removing item %s from temp directory: %s", item_path, e)

            os.makedirs(temp_dir, exist_ok=True)
            
            # Get the absolute path for the temporary directory
            abs_temp_dir = os.path.abspath(temp_dir)
            
            # Use the absolute path for the temporary file path
            temp_filepath = os.path.join(abs_temp_dir, safe_filename)
            
            logger.info("Attempting to save file to: %s", temp_filepath) # Log before saving
            try:
                # Manually read and write the file content
                with open(temp_filepath, 'wb') as f:
                    uploaded_file.stream.seek(0) # Ensure stream is at the beginning
                    f.write(uploaded_file.stream.read())
                logger.info("Temporary file successfully saved to: %s", temp_filepath) # Log on successful save
                session["uploaded_filepath"] = temp_filepath
                
                # --- Add verification step immediately after saving ---
                try:
                    logger.info(f"Attempting to verify saved file by opening: {temp_filepath}")
                    with open(temp_filepath, "rb") as f:
                        # Try reading a small part of the file
                        f.read(100) # Read first 100 bytes
                    logger.info(f"Verification of saved file successful: {temp_filepath}")
                except Exception as verify_exc:
                    logger.error(f"Verification of saved file failed {temp_filepath}: {verify_exc}", exc_info=True)
                    # If verification fails, return an error immediately
                    return jsonify({"status": "error", "message": f"Error verifying saved file immediately after saving: {verify_exc}. Path: {temp_filepath}."})
                # --- End verification step ---

                # Temporarily include filepath and existence check in success response
                file_exists_check = os.path.exists(temp_filepath)
                logger.info("Checking existence of saved file %s: %s", temp_filepath, file_exists_check)

                return jsonify({"status": "success", 
                                "columns": columns, 
                                "saved_filepath_check": temp_filepath, 
                                "file_exists_immediately_after_save": file_exists_check})
            except Exception as save_exc:
                logger.error("Error saving uploaded file to %s: %s", temp_filepath, save_exc, exc_info=True)
                return jsonify({"status": "error", "message": f"Error saving file: {save_exc}. Please check file permissions and path validity."})

        return jsonify({"status": "error", "message": "No file uploaded."})
    except Exception as e:
        logger.error("Upload error: %s", e, exc_info=True)
        # Return the specific upload error message
        return jsonify({"status": "error", "message": "Error during file upload: " + str(e)})

@app.route('/confirm_columns', methods=['POST'])
def confirm_columns():
    try:
        data = request.get_json()
        selected_columns = data.get("selected_columns", [])
        session["selected_columns"] = selected_columns
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error("Confirm columns error: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        filepath = session.get("uploaded_filepath")
        if not filepath or not os.path.exists(filepath):
            return jsonify({"status": "error", "message": "Uploaded file not found or does not exist."})

        # Read the Excel file by reading its content into BytesIO and passing to pd.ExcelFile
        try:
            logger.info(f"Attempting to read file content into BytesIO: {filepath}")
            # Explicitly open the file in binary read mode
            with open(filepath, "rb") as f:
                # Read the entire file content into bytes
                file_bytes = f.read()
            logger.info("Successfully read file content into bytes.")

            # Create a BytesIO object from the file content
            file_io = io.BytesIO(file_bytes)
            logger.info("Created BytesIO object from file content.")

            # Pass the BytesIO object to pd.ExcelFile
            logger.info("Attempting to create pd.ExcelFile from BytesIO.")
            xls = pd.ExcelFile(file_io, engine="openpyxl")
            logger.info("Successfully created pd.ExcelFile from BytesIO.")
            
            dataframes = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
            logger.info("Successfully parsed sheets into DataFrames.")

        except Exception as read_exc:
            logger.error("Error reading Excel file via BytesIO: %s", read_exc, exc_info=True)
            # Return the specific exception message for debugging, including the path
            return jsonify({"status": "error", "message": f"Error reading Excel file: {read_exc}. Path: {filepath}. Ensure it is a valid .xlsx file."})

        # --- Perform Analysis ---
        # Assuming the first sheet contains the main data for analysis
        main_df = dataframes.get(list(dataframes.keys())[0]) if dataframes else pd.DataFrame()

        if main_df.empty:
            return jsonify({"status": "error", "message": "No data found in the first sheet of the Excel file."})

        # Retrieve selected columns from session
        selected_columns = session.get("selected_columns")

        if not selected_columns:
             return jsonify({"status": "error", "message": "No columns were selected for analysis."})

        # Filter the DataFrame to include only selected columns
        # Drop columns that are not in selected_columns, handling potential KeyError if a selected column doesn't exist in the DataFrame
        cols_to_keep = [col for col in selected_columns if col in main_df.columns]
        if not cols_to_keep:
             return jsonify({"status": "error", "message": "None of the selected columns were found in the uploaded file."})

        filtered_df = main_df[cols_to_keep].copy()

        # Pass the filtered DataFrame to the analysis function
        cleaned_df, analysis_summary = clean_and_analyze_data(filtered_df)

        # Save cleaned data to a temporary file (e.g., CSV or Excel)
        # Using CSV for simplicity, can change to Excel if needed
        cleaned_filepath = os.path.join("temp", f"cleaned_{os.path.basename(filepath)}.csv")
        cleaned_df.to_csv(cleaned_filepath, index=False)
        session["cleaned_filepath"] = cleaned_filepath
        logger.info("Cleaned data saved to %s", cleaned_filepath)

        # Perform AI analysis
        aiResponses, resolutionSummary = process_analysis_report(analysis_summary)

        # Store results in session for /result route
        session["analysis_summary"] = analysis_summary
        session["aiResponses"] = aiResponses
        session["resolutionSummary"] = resolutionSummary

        # Return success status and redirect information for frontend
        # The frontend JavaScript should handle the redirect to /result on success
        return jsonify({"status": "success", "message": "Analysis complete. Redirecting..."})

    except Exception as e:
        logger.error("Analyze error: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": "An unexpected error occurred during analysis: " + str(e)})

@app.route('/result')
def result():
    try:
        aiResponses = session.get("aiResponses")
        resolutionSummary = session.get("resolutionSummary")
        analysis_summary = session.get("analysis_summary")
        if not aiResponses or not resolutionSummary or not analysis_summary:
            return redirect("/")
        logger.info("Rendering result.html with:")
        logger.info("aiResponses: %s", aiResponses)
        logger.info("resolutionSummary: %s", resolutionSummary)
        logger.info("analysis_summary: %s", analysis_summary)
        return render_template("result.html", aiResponses=aiResponses, resolutionSummary=resolutionSummary, analysisSummary=analysis_summary)
    except Exception as e:
        logger.error("Result page error: %s", e, exc_info=True)
        return "Error loading result page: " + str(e), 500

@app.route('/download/zip')
def download_zip():
    try:
        analysis_summary = session.get("analysis_summary", DEFAULT_REPORT)
        aiAnalysis = session.get("resolutionSummary", "")
        report_content = analysis_summary + "\n\nAI Analysis Results:\n" + aiAnalysis
        cleaned_filepath = session.get("cleaned_filepath")
        if not cleaned_filepath or not os.path.exists(cleaned_filepath):
            cleaned_content = b""
        else:
            with open(cleaned_filepath, "rb") as f:
                cleaned_content = f.read()
        chart_file = "spend_analysis_charts.png"
        if os.path.exists(chart_file):
            with open(chart_file, "rb") as f:
                chart_content = f.read()
        else:
            chart_content = b""
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, "w") as zf:
            zf.writestr("analysis_report.txt", report_content)
            zf.writestr("cleaned.xlsx", cleaned_content)
            if chart_content:
                zf.writestr("analysis_charts.png", chart_content)
        memory_file.seek(0)
        return send_file(memory_file, download_name="Processed_Files.zip", as_attachment=True)
    except Exception as e:
        logger.error("Download ZIP error: %s", e, exc_info=True)
        return "Error generating download: " + str(e), 500

if __name__ == "__main__":
    app.run(debug=True)
