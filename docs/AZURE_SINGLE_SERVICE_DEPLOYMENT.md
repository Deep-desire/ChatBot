# Single-Service Azure App Service Deployment Guide

This guide details how to deploy both the **React Frontend** and the **FastAPI Backend** together as a single unified service on **Azure App Service (Linux Web App)**, without requiring Azure Static Web Apps or Azure Function Apps.

---

## How It Works
1. **Unified Hosting**: The React application is built, and its compiled static files (`index.html`, JavaScript, CSS, images) are placed directly inside the FastAPI backend folder under a directory named `static/`.
2. **Static Mounting**: FastAPI is configured to serve these files from the root URL (`/`). 
3. **Same-Origin API**: Since the frontend and backend are served from the exact same domain, all API requests (`/api/...`) work out-of-the-box, resolving all CORS issues.
4. **Ingestion API**: Ingestion (file upload and indexing to Azure AI Search) is handled directly by the FastAPI endpoints (`/api/ingest/upload` and `/api/ingest/blob`), eliminating the need for a separate Azure Function App.

---

## 1. Prerequisites
Ensure you have the following installed on your machine:
* **Azure CLI**: [Install Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
* **Node.js 18+ and npm**
* **Python 3.11**

---

## 2. Step 1: Create the Deployment Package

Open a PowerShell terminal, navigate to the root directory of the project, and run the automated build script:
```powershell
# Navigate to the project root
cd "C:\Desire ChatBot\visit-to-lead"

# Execute the packaging script
.\build_deployment_package.ps1
```

### What this script does:
1. Installs React dependencies and compiles the frontend into static assets with `VITE_API_BASE_URL` set to `/` (enabling relative same-origin routing).
2. Deletes any old assets and copies the new build output (`dist/`) directly to `backend/static/`.
3. Creates a deployable zip archive named `deployment_package.zip` in your root folder, omitting unnecessary folders like local Python virtual environments (`venv`), local logs, and cache folders.

---

## 3. Step 2: Provision Azure App Service

Run the following commands using **Azure CLI** to log in and set up your App Service resources:

### 3.1 Authenticate with Azure CLI
```powershell
az login
az account set --subscription "Your-Subscription-Name-or-ID"
```

### 3.2 Create a Resource Group
```powershell
az group create --name rg-desire-chatbot --location eastus
```

### 3.3 Create a Linux App Service Plan
The `B1` tier is cost-effective and provides dedicated compute resources suitable for production workloads:
```powershell
az appservice plan create \
  --name plan-desire-chatbot \
  --resource-group rg-desire-chatbot \
  --sku B1 \
  --is-linux \
  --location eastus
```

### 3.4 Create the Web App
Create the web app running the Python 3.11 runtime:
```powershell
az webapp create \
  --name app-desire-chatbot \
  --resource-group rg-desire-chatbot \
  --plan plan-desire-chatbot \
  --runtime "PYTHON:3.11"
```
*(Note: Choose a globally unique name for `--name`. The app URL will be `https://<webapp-name>.azurewebsites.net`)*

---

## 4. Step 3: Configure Web App Settings

### 4.1 Set Environment Variables
Add the environment settings from your local `.env` file to your App Service. Do not wrap values in double quotes when running Azure CLI:
```powershell
az webapp config appsettings set \
  --name app-desire-chatbot \
  --resource-group rg-desire-chatbot \
  --settings \
    AZURE_OPENAI_ENDPOINT="https://your-resource.cognitiveservices.azure.com/" \
    AZURE_OPENAI_API_KEY="your_azure_openai_api_key_here" \
    AZURE_OPENAI_API_VERSION="2024-12-01-preview" \
    AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o" \
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small" \
    AZURE_OPENAI_TEMPERATURE="0.1" \
    LLM_MAX_OUTPUT_TOKENS="1200" \
    AZURE_OPENAI_MAX_COMPLETION_TOKENS="16384" \
    AZURE_SEARCH_ENDPOINT="https://your-search-service.search.windows.net" \
    AZURE_SEARCH_INDEX_NAME="chatbot-rag" \
    AZURE_SEARCH_API_KEY="your_azure_search_api_key" \
    AZURE_SEARCH_ID_FIELD="id" \
    AZURE_SEARCH_CONTENT_FIELD="content" \
    AZURE_SEARCH_VECTOR_FIELD="contentVector" \
    AZURE_SEARCH_TOP_K="5" \
    AZURE_SEARCH_SCORE_THRESHOLD="0.2" \
    GROQ_API_KEY="your_groq_api_key_here" \
    GROQ_TRANSCRIPTION_MODEL="whisper-large-v3" \
    EDGE_TTS_VOICE="en-US-AriaNeural" \
    CHAT_TRACE_ENABLED="true" \
    CHAT_TRACE_PRINT_CONSOLE="true" \
    CHAT_TRACE_INCLUDE_CONTEXT="true" \
    CHAT_TRACE_LOG_PATH="logs/chat_trace.jsonl" \
    CHAT_PROCESS_LOG_PATH="logs/chat_process_last10.json" \
    CHAT_PROCESS_LOG_LIMIT="10"
```

### 4.2 Configure the Startup Script
FastAPI requires Uvicorn to run. Configure Azure App Service to run the app using the Gunicorn Uvicorn worker:
```powershell
az webapp config set \
  --name app-desire-chatbot \
  --resource-group rg-desire-chatbot \
  --startup-file "gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app"
```

---

## 5. Step 4: Deploy the Package

Deploy the generated `deployment_package.zip` directly to the App Service instance:
```powershell
az webapp deployment source config-zip \
  --name app-desire-chatbot \
  --resource-group rg-desire-chatbot \
  --src .\deployment_package.zip
```
*(Wait 1-2 minutes for Azure to unpack, compile dependencies on the server, and restart the container).*

---

## 6. Step 5: Post-Deployment Verification

1. **Test Frontend & Chat**:
   Open a browser and navigate to `https://<your-webapp-name>.azurewebsites.net`. The React UI should load immediately. Complete the lead capture step and test a chat message.
   
2. **Health Check Endpoint**:
   Check `https://<your-webapp-name>.azurewebsites.net/health` to confirm the backend is up.

3. **Ingest Documents Directly via API**:
   Since the Azure Function is not deployed, you can upload PDFs and index them directly through the FastAPI endpoint:
   ```powershell
   curl -X POST "https://<your-webapp-name>.azurewebsites.net/api/ingest/upload" ^
     -H "X-Ingest-Key: your_ingest_key_if_configured" ^
     -F "file=@C:\path\to\your-document.pdf"
   ```
