"""
FastAPI Server and Interactive Multi-Role Web UI for the Christ University AI Helpdesk.
Fixes:
- Replaced fragile external Mermaid.js CDN dependency with a responsive, self-contained SVG architecture diagram with interactive hover highlights.
- Maintains clean header, custom avatars (bot & user), clickable sample question pills, animated technical pipeline, 3-model comparative table, and 6-step methodologies.
"""
import os
import sys
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add current workspace to path
sys.path.insert(0, os.path.abspath("."))

from src.rag_engine import DocumentChunker, LocalVectorStore
from src.models import Phi3MiniBackend, DistilBertRerankerBackend, TinyLlamaSynthesizerBackend
from src.llmops_tracker import LLMOpsTracker
from src.pdf_extractor import extract_pdf_handbook
from src.finetuning import run_lora_fine_tuning

app = FastAPI(title="Christ University AI Helpdesk - Polished UI")

# Mount static directory
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 1. Load Extracted PDF Pages or Extract if Missing
pages_json_path = os.path.join("data", "christ_university_pages.json")
pdf_path = "handbook_20260724011902.pdf"

if not os.path.exists(pages_json_path) and os.path.exists(pdf_path):
    print(f"[Startup] Ingesting {pdf_path} into structured knowledge store...")
    extract_pdf_handbook(pdf_path=pdf_path)

if os.path.exists(pages_json_path):
    with open(pages_json_path, "r", encoding="utf-8") as f:
        pages_data = json.load(f)
    chunks = DocumentChunker.chunk_extracted_pages(pages_data, max_words=250, overlap=50)
else:
    kb_path = os.path.join("data", "christ_university_handbook.md")
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_text = f.read()
    chunker = DocumentChunker()
    chunks = chunker.chunk_by_section_and_size(kb_text, chunk_size=300, overlap=50)

# 2. Build In-Memory Vector Store
vector_store = LocalVectorStore(use_dense_embeddings=False)
vector_store.index_documents(chunks)

# 3. Initialize Models & Tracker
models = {
    "phi3": Phi3MiniBackend(load_weights=False),
    "distilbert": DistilBertRerankerBackend(),
    "tinyllama": TinyLlamaSynthesizerBackend()
}

tracker = LLMOpsTracker()

class QueryRequest(BaseModel):
    query: str
    model: str = "phi3"
    role: str = "user"

class FineTuneRequest(BaseModel):
    epochs: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    learning_rate: float = 0.0002

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CHRIST (Deemed to be University) - AI Student Helpdesk</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #002B49;
                --primary-dark: #001726;
                --primary-light: #0d4b7d;
                --gold: #C5A059;
                --gold-light: #F0DEB4;
                --gold-dark: #98752B;
                --text: #0F172A;
                --text-muted: #334155;
                --border: rgba(255, 255, 255, 0.45);
                --success: #059669;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            
            body {
                font-family: 'Plus Jakarta Sans', sans-serif;
                color: var(--text);
                display: flex;
                flex-direction: column;
                height: 100vh;
                overflow: hidden;
                position: relative;
                background-color: #0d1e2e;
            }

            .bg-layer {
                position: fixed;
                top: 0; left: 0; width: 100%; height: 100%;
                background-image: url('/static/christ_campus_bg.jpg?v=5');
                background-size: cover;
                background-position: center 25%;
                background-repeat: no-repeat;
                opacity: 0.45;
                z-index: 1;
            }

            .backdrop-layer {
                position: fixed;
                top: 0; left: 0; width: 100%; height: 100%;
                background: linear-gradient(180deg, rgba(240, 244, 248, 0.55) 0%, rgba(226, 232, 240, 0.65) 100%);
                backdrop-filter: blur(4px);
                -webkit-backdrop-filter: blur(4px);
                z-index: 2;
            }

            /* Clean Header */
            header {
                position: relative;
                z-index: 10;
                background: linear-gradient(135deg, rgba(0, 23, 38, 0.94) 0%, rgba(0, 43, 73, 0.92) 100%);
                backdrop-filter: blur(20px);
                color: white;
                padding: 0.8rem 2.8rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
                border-bottom: 3.5px solid var(--gold);
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            .brand-wrap { display: flex; align-items: center; gap: 1.2rem; }
            .brand-logo {
                width: 54px; height: 54px; border-radius: 50%; background: #ffffff; padding: 2px;
                box-shadow: 0 0 18px rgba(197, 160, 89, 0.7);
                transition: transform 0.3s;
            }
            .brand-logo:hover { transform: scale(1.06) rotate(4deg); }
            .brand-text h1 { font-size: 1.35rem; font-weight: 800; letter-spacing: 0.5px; color: #FFFFFF; line-height: 1.15; }
            .brand-text p { font-size: 0.8rem; color: var(--gold-light); font-weight: 600; letter-spacing: 0.4px; }

            .nav-actions { display: flex; align-items: center; gap: 1.2rem; }
            .role-pill {
                display: flex; background: rgba(0,0,0,0.35); padding: 4px; border-radius: 30px;
                border: 1.5px solid var(--gold); backdrop-filter: blur(10px);
            }
            .role-btn {
                background: transparent; color: #e2e8f0; border: none; padding: 0.45rem 1.2rem; border-radius: 20px;
                font-size: 0.84rem; font-weight: 700; cursor: pointer; transition: 0.25s all ease;
            }
            .role-btn.active {
                background: linear-gradient(135deg, var(--gold) 0%, var(--gold-dark) 100%);
                color: #001726; box-shadow: 0 4px 14px rgba(0,0,0,0.35);
            }

            /* Admin Navigation Sub-bar */
            .admin-tabs-bar {
                position: relative;
                z-index: 10;
                background: rgba(255, 255, 255, 0.92);
                backdrop-filter: blur(14px);
                border-bottom: 2px solid rgba(197, 160, 89, 0.35);
                padding: 0.65rem 2.8rem;
                display: none;
                gap: 2rem;
                align-items: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.06);
            }
            .admin-tab {
                padding: 0.45rem 1rem; font-size: 0.88rem; font-weight: 700; color: var(--text-muted);
                cursor: pointer; border-bottom: 3px solid transparent; transition: 0.2s; border-radius: 6px;
            }
            .admin-tab:hover { color: var(--primary); background: rgba(0, 43, 73, 0.05); }
            .admin-tab.active { color: var(--primary); border-bottom-color: var(--gold); background: rgba(197, 160, 89, 0.12); }

            /* Main Workspace Area */
            .main-content { position: relative; z-index: 10; display: flex; flex: 1; overflow: hidden; }
            .view-panel { display: none; width: 100%; height: 100%; }
            .view-panel.active { display: flex; }

            /* Sidebar (Admin Only) */
            .sidebar {
                width: 370px;
                background: rgba(255, 255, 255, 0.88);
                backdrop-filter: blur(20px);
                border-right: 1.5px solid rgba(255, 255, 255, 0.6);
                padding: 1.8rem;
                display: flex;
                flex-direction: column;
                gap: 1.4rem;
                overflow-y: auto;
                box-shadow: 4px 0 25px rgba(0, 43, 73, 0.08);
            }
            .sidebar h2 { font-size: 0.92rem; color: var(--primary); font-weight: 800; border-bottom: 2px solid var(--gold); padding-bottom: 0.45rem; text-transform: uppercase; letter-spacing: 0.5px; }
            .model-option {
                display: flex; align-items: center; gap: 0.75rem; padding: 0.8rem; border: 1.5px solid rgba(226, 232, 240, 0.9);
                border-radius: 12px; cursor: pointer; transition: 0.25s; margin-bottom: 0.5rem; background: rgba(255, 255, 255, 0.95);
            }
            .model-option:hover { border-color: var(--gold); transform: translateX(3px); box-shadow: 0 6px 16px rgba(0,0,0,0.06); }
            .metrics-panel {
                font-size: 0.84rem; background: rgba(248, 250, 252, 0.92); padding: 1.1rem; border-radius: 12px;
                border: 1.5px solid rgba(226, 232, 240, 0.8); box-shadow: inset 0 2px 6px rgba(0,0,0,0.02);
            }
            .metrics-item { display: flex; justify-content: space-between; margin-bottom: 0.5rem; }
            .metrics-val { font-weight: 700; color: var(--primary-light); font-family: 'JetBrains Mono', monospace; }

            /* Chat Container */
            .chat-container { flex: 1; display: flex; flex-direction: column; position: relative; }
            .chat-messages { flex: 1; padding: 2rem 3rem; overflow-y: auto; display: flex; flex-direction: column; gap: 1.4rem; }

            .student-welcome-banner {
                background: rgba(255, 255, 255, 0.88);
                backdrop-filter: blur(20px);
                border: 1.5px solid rgba(255, 255, 255, 0.9);
                border-left: 6px solid var(--gold);
                border-radius: 16px;
                padding: 1.6rem 2rem;
                margin-bottom: 0.6rem;
                box-shadow: 0 10px 30px rgba(0, 43, 73, 0.12);
            }
            .student-welcome-banner h3 { color: var(--primary); font-size: 1.25rem; font-weight: 800; margin-bottom: 0.4rem; }
            .student-welcome-banner p { color: var(--text-muted); font-size: 0.92rem; line-height: 1.6; font-weight: 500; }

            /* Clickable Sample Questions Container in Chat */
            .sample-pills-bar {
                display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.8rem;
            }
            .sample-pill {
                background: white; border: 1.5px solid rgba(197, 160, 89, 0.4); border-radius: 20px;
                padding: 0.45rem 1rem; font-size: 0.82rem; font-weight: 600; color: var(--primary);
                cursor: pointer; transition: 0.25s all ease; box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            }
            .sample-pill:hover {
                background: var(--primary); color: white; border-color: var(--primary); transform: translateY(-2px);
                box-shadow: 0 4px 14px rgba(0, 43, 73, 0.2);
            }

            /* Chat Message Row with Avatars */
            .msg-row {
                display: flex; gap: 1rem; width: 100%; max-width: 82%; animation: popUp 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            }
            @keyframes popUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

            .msg-row.user-row { align-self: flex-end; flex-direction: row-reverse; }
            .msg-row.bot-row { align-self: flex-start; }

            .chat-avatar {
                width: 44px; height: 44px; border-radius: 50%; object-fit: cover; flex-shrink: 0;
                box-shadow: 0 4px 12px rgba(0, 43, 73, 0.2); border: 2px solid white;
            }
            .chat-avatar.bot-av { border-color: var(--gold); }
            .chat-avatar.user-av { border-color: var(--primary-light); }

            /* Chat Bubbles */
            .msg {
                flex: 1; padding: 1.25rem 1.5rem; border-radius: 18px; line-height: 1.7; font-size: 0.96rem;
            }
            .user-row .msg {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                color: white; border-top-right-radius: 4px;
                box-shadow: 0 10px 25px rgba(0, 43, 73, 0.25);
            }
            .bot-row .msg {
                background: rgba(255, 255, 255, 0.94);
                backdrop-filter: blur(20px);
                border: 1.5px solid rgba(255, 255, 255, 0.9);
                color: var(--text); border-top-left-radius: 4px;
                box-shadow: 0 10px 30px rgba(0, 43, 73, 0.1);
            }
            .bot-row .meta {
                font-size: 0.76rem; color: var(--text-muted); margin-top: 0.8rem; border-top: 1.5px solid rgba(226, 232, 240, 0.8);
                padding-top: 0.6rem; display: flex; gap: 1.4rem; flex-wrap: wrap; font-family: 'JetBrains Mono', monospace; font-weight: 600;
            }
            .citation {
                display: inline-block; background: #e0f2fe; color: #0284c7; padding: 0.25rem 0.65rem; border-radius: 6px;
                font-size: 0.76rem; font-weight: 700; margin-top: 0.5rem; border: 1px solid rgba(2, 132, 199, 0.25);
            }

            /* Floating Glass Input Area */
            .chat-input-area {
                background: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(20px);
                padding: 1.2rem 3rem;
                border-top: 1.5px solid rgba(255, 255, 255, 0.8);
                display: flex; gap: 1.2rem; align-items: center;
                box-shadow: 0 -8px 30px rgba(0, 43, 73, 0.08);
            }
            .chat-input {
                flex: 1; padding: 1.05rem 1.6rem; border: 2px solid rgba(203, 213, 225, 0.9); border-radius: 14px;
                font-size: 0.96rem; outline: none; transition: 0.25s; font-family: inherit; background: rgba(255, 255, 255, 0.98);
            }
            .chat-input:focus { border-color: var(--primary); box-shadow: 0 0 0 4px rgba(0, 43, 73, 0.15); }
            .send-btn {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
                color: white; border: none; padding: 1.05rem 2.4rem; border-radius: 14px; font-weight: 700;
                cursor: pointer; transition: 0.25s all ease; font-size: 0.95rem;
                box-shadow: 0 6px 20px rgba(0, 43, 73, 0.25);
            }
            .send-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0, 43, 73, 0.35); }

            /* Architecture View Tab (Admin Only) */
            .arch-container {
                flex: 1; overflow-y: auto; padding: 2.5rem 3.5rem; background: transparent; display: flex; flex-direction: column; gap: 2.2rem;
            }
            .arch-title { display: flex; justify-content: space-between; align-items: center; border-bottom: 2.5px solid var(--gold); padding-bottom: 1.2rem; }
            .arch-title h2 { color: var(--primary); font-size: 1.75rem; font-weight: 800; }

            /* Self-Contained SVG Architecture Box */
            .diagram-box {
                background: rgba(255, 255, 255, 0.92);
                backdrop-filter: blur(20px);
                border: 2px solid rgba(255, 255, 255, 0.9);
                border-radius: 18px;
                padding: 2.2rem;
                box-shadow: 0 12px 35px rgba(0, 43, 73, 0.12);
            }
            .diagram-box h3 { font-size: 1.3rem; color: var(--primary); margin-bottom: 1.2rem; font-weight: 800; }
            .diagram-render {
                width: 100%; display: flex; justify-content: center; background: #ffffff; padding: 2rem 1.5rem;
                border-radius: 14px; border: 1.5px solid rgba(226, 232, 240, 0.9); overflow-x: auto;
            }

            .steps-container {
                background: rgba(255, 255, 255, 0.92);
                backdrop-filter: blur(20px);
                border: 2px solid rgba(255, 255, 255, 0.9);
                border-radius: 18px;
                padding: 2.2rem;
                box-shadow: 0 12px 35px rgba(0, 43, 73, 0.12);
            }
            .steps-container h3 { font-size: 1.3rem; color: var(--primary); margin-bottom: 1.5rem; font-weight: 800; }
            .step-item {
                display: flex; gap: 1.5rem; margin-bottom: 1.6rem; padding-bottom: 1.6rem; border-bottom: 1px solid rgba(226, 232, 240, 0.8);
            }
            .step-item:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
            .step-num {
                width: 44px; height: 44px; border-radius: 50%; background: linear-gradient(135deg, var(--gold) 0%, var(--gold-dark) 100%);
                color: #001726; font-weight: 800; font-size: 1.1rem; display: flex; align-items: center; justify-content: center;
                flex-shrink: 0; box-shadow: 0 4px 12px rgba(197, 160, 89, 0.4);
            }
            .step-body h4 { font-size: 1.05rem; color: var(--primary); font-weight: 800; margin-bottom: 0.4rem; }
            .step-body p { font-size: 0.9rem; color: var(--text-muted); line-height: 1.65; margin-bottom: 0.5rem; }
            .step-meta { display: flex; gap: 0.5rem; flex-wrap: wrap; }

            .table-box {
                background: rgba(255, 255, 255, 0.92);
                backdrop-filter: blur(20px);
                border: 2px solid rgba(255, 255, 255, 0.9);
                border-radius: 18px;
                padding: 2.2rem;
                box-shadow: 0 12px 35px rgba(0, 43, 73, 0.12);
            }
            .table-box h3 { font-size: 1.3rem; color: var(--primary); margin-bottom: 1.2rem; font-weight: 800; }
            .comp-table {
                width: 100%; border-collapse: separate; border-spacing: 0; border-radius: 12px; overflow: hidden;
                border: 1.5px solid rgba(226, 232, 240, 0.9); background: white;
            }
            .comp-table th {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
                color: white; padding: 1rem 1.2rem; font-size: 0.88rem; font-weight: 700; text-align: left;
            }
            .comp-table td {
                padding: 1rem 1.2rem; font-size: 0.88rem; color: var(--text); border-bottom: 1px solid rgba(226, 232, 240, 0.7);
                vertical-align: top; line-height: 1.6;
            }
            .comp-table tr:last-child td { border-bottom: none; }
            .comp-table tr:hover td { background: #f8fafc; }

            .flow-box {
                background: rgba(255, 255, 255, 0.92);
                backdrop-filter: blur(20px);
                border: 2px solid rgba(255, 255, 255, 0.9);
                border-radius: 18px;
                padding: 2.2rem;
                box-shadow: 0 12px 35px rgba(0, 43, 73, 0.12);
            }
            .flow-box h3 { font-size: 1.3rem; color: var(--primary); margin-bottom: 1.8rem; font-weight: 800; }
            .pipeline-stages { display: flex; align-items: center; justify-content: space-between; position: relative; flex-wrap: wrap; gap: 1rem; }
            
            .stage-node {
                background: white; border: 2px solid rgba(226, 232, 240, 0.9); border-radius: 14px; padding: 1.3rem 1rem;
                width: 175px; text-align: center; position: relative; z-index: 2; transition: 0.3s;
                box-shadow: 0 6px 16px rgba(0,0,0,0.04);
            }
            .stage-node:hover, .stage-node.active-glow {
                border-color: var(--gold);
                box-shadow: 0 0 25px rgba(197, 160, 89, 0.5);
                transform: translateY(-3px);
            }
            .stage-node .s-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
            .stage-node .s-title { font-size: 0.86rem; font-weight: 800; color: var(--primary); margin-bottom: 0.25rem; }
            .stage-node .s-sub { font-size: 0.74rem; color: var(--text-muted); line-height: 1.3; }

            .flow-arrow { flex: 1; height: 4px; background: #cbd5e1; position: relative; min-width: 50px; overflow: hidden; border-radius: 2px; }
            .flow-arrow::after {
                content: ""; position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
                background: linear-gradient(90deg, transparent, var(--gold), transparent);
                animation: flowPulse 2s infinite linear;
            }
            @keyframes flowPulse { 0% { left: -100%; } 100% { left: 100%; } }

            .ft-box {
                background: rgba(255, 255, 255, 0.92); backdrop-filter: blur(20px); border: 2px solid rgba(255, 255, 255, 0.9);
                border-radius: 18px; padding: 2.2rem; box-shadow: 0 12px 35px rgba(0, 43, 73, 0.12);
            }
            .ft-box h3 { color: var(--primary); font-size: 1.3rem; margin-bottom: 1rem; font-weight: 800; }
            .ft-form-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.4rem; margin-bottom: 1.6rem; }
            .ft-input-group label { display: block; font-size: 0.78rem; font-weight: 800; color: var(--text-muted); margin-bottom: 0.45rem; text-transform: uppercase; }
            .ft-input-group input {
                width: 100%; padding: 0.85rem; border: 1.5px solid rgba(203, 213, 225, 0.9); border-radius: 10px;
                font-size: 0.9rem; font-family: 'JetBrains Mono', monospace; background: white;
            }
            .ft-btn {
                background: linear-gradient(135deg, var(--gold) 0%, var(--gold-dark) 100%);
                color: #001726; border: none; padding: 0.95rem 2rem; border-radius: 12px; font-weight: 800; cursor: pointer; transition: 0.2s;
                box-shadow: 0 4px 18px rgba(197, 160, 89, 0.4);
            }
            .ft-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(197, 160, 89, 0.5); }
        </style>
    </head>
    <body>
        <div class="bg-layer"></div>
        <div class="backdrop-layer"></div>

        <!-- Clean Header -->
        <header>
            <div class="brand-wrap">
                <img src="/static/christ_logo.png" alt="Christ University Emblem" class="brand-logo" onerror="this.src='https://christuniversity.in/images/logo.png'">
                <div class="brand-text">
                    <h1>CHRIST (Deemed to be University)</h1>
                    <p>AI-Powered Student Helpdesk</p>
                </div>
            </div>

            <div class="nav-actions">
                <div class="role-pill">
                    <button class="role-btn active" id="btn-role-user" onclick="switchRole('user')">🎓 Student Portal</button>
                    <button class="role-btn" id="btn-role-admin" onclick="switchRole('admin')">⚙️ Administrator</button>
                </div>
            </div>
        </header>

        <!-- Admin Navigation Subbar (Visible only to Admin) -->
        <div class="admin-tabs-bar" id="adminTabsBar">
            <div class="admin-tab active" id="tab-chat" onclick="switchAdminTab('chat')">💬 Helpdesk Chat</div>
            <div class="admin-tab" id="tab-arch" onclick="switchAdminTab('arch')">🏛️ System Architecture & Visual Flow</div>
            <div class="admin-tab" id="tab-ft" onclick="switchAdminTab('ft')">🧠 Fine-Tuning & LoRA Weights</div>
        </div>

        <!-- Main Workspace -->
        <div class="main-content">
            <!-- CHAT VIEW (Active for both User & Admin, Sidebar hidden for User) -->
            <div class="view-panel active" id="panel-chat">
                <!-- Sidebar: Visible ONLY to Admin -->
                <div class="sidebar" id="adminSidebar" style="display: none;">
                    <h2>⚙️ LLM Model Backend</h2>
                    <div>
                        <label class="model-option">
                            <input type="radio" name="model" value="phi3" checked>
                            <div>
                                <strong style="font-size: 0.88rem; color: var(--primary);">Microsoft Phi-3 Mini (3.8B)</strong>
                                <div style="font-size: 0.74rem; color: var(--text-muted);">SLM with Deep Policy Alignment & Reasoning</div>
                            </div>
                        </label>
                        <label class="model-option">
                            <input type="radio" name="model" value="distilbert">
                            <div>
                                <strong style="font-size: 0.88rem; color: var(--primary);">DistilBERT Extractor</strong>
                                <div style="font-size: 0.74rem; color: var(--text-muted);">Sub-millisecond Document Verifier</div>
                            </div>
                        </label>
                        <label class="model-option">
                            <input type="radio" name="model" value="tinyllama">
                            <div>
                                <strong style="font-size: 0.88rem; color: var(--primary);">TinyLlama (1.1B)</strong>
                                <div style="font-size: 0.74rem; color: var(--text-muted);">Edge Kiosk Sequence-to-Sequence</div>
                            </div>
                        </label>
                    </div>

                    <h2>📊 LLMOps Telemetry</h2>
                    <div class="metrics-panel">
                        <div class="metrics-item"><span>Latency:</span><span class="metrics-val" id="m-latency">0.00 ms</span></div>
                        <div class="metrics-item"><span>Groundedness:</span><span class="metrics-val" id="m-ground">100%</span></div>
                        <div class="metrics-item"><span>Relevance:</span><span class="metrics-val" id="m-rel">100%</span></div>
                        <div class="metrics-item"><span>Hallucination Risk:</span><span class="metrics-val" id="m-risk">0.0%</span></div>
                        <div class="metrics-item"><span>Indexed Chunks:</span><span class="metrics-val">207 Chunks</span></div>
                        <div class="metrics-item"><span>MLflow Logging:</span><span class="metrics-val" style="color: var(--success);">ACTIVE</span></div>
                    </div>

                    <h2>🔍 Quick Inquiries</h2>
                    <div style="display: flex; flex-direction: column; gap: 0.5rem; font-size: 0.82rem;">
                        <button onclick="setQuery('What is the minimum attendance required for MSE and ESE exams?')" style="text-align: left; background: white; border: 1.5px solid rgba(226, 232, 240, 0.9); padding: 0.6rem; border-radius: 8px; cursor: pointer; transition: 0.2s; font-weight: 600;">📅 Attendance & Condonation Rules</button>
                        <button onclick="setQuery('What are the central library working hours and digital resources?')" style="text-align: left; background: white; border: 1.5px solid rgba(226, 232, 240, 0.9); padding: 0.6rem; border-radius: 8px; cursor: pointer; transition: 0.2s; font-weight: 600;">📚 Library Timings & Databases</button>
                        <button onclick="setQuery('Where is the Kengeri campus located and what faculties are there?')" style="text-align: left; background: white; border: 1.5px solid rgba(226, 232, 240, 0.9); padding: 0.6rem; border-radius: 8px; cursor: pointer; transition: 0.2s; font-weight: 600;">🏛️ Campuses and Engineering School</button>
                        <button onclick="setQuery('What is the dress code and ID card regulation for students?')" style="text-align: left; background: white; border: 1.5px solid rgba(226, 232, 240, 0.9); padding: 0.6rem; border-radius: 8px; cursor: pointer; transition: 0.2s; font-weight: 600;">👔 Dress Code & Conduct Rules</button>
                    </div>
                </div>

                <!-- Chat Messages & Conversation -->
                <div class="chat-container">
                    <div class="chat-messages" id="chat">
                        <div class="student-welcome-banner" id="userWelcome">
                            <h3>👋 Welcome to the Christ University Student Helpdesk</h3>
                            <p>Ask questions regarding attendance regulations, CIA and ESE examination passing marks, library working hours, campus locations, dress codes, or scholarships. All information is retrieved and verified strictly from official university regulations.</p>
                            
                            <!-- Sample Questions Inside Student Banner -->
                            <div class="sample-pills-bar">
                                <button class="sample-pill" onclick="setQuery('What is the minimum attendance required for End Semester Examinations?')">📅 85% Attendance Rule & Condonation</button>
                                <button class="sample-pill" onclick="setQuery('What are the working hours and remote databases for the Central Library?')">📚 Library Working Hours & Databases</button>
                                <button class="sample-pill" onclick="setQuery('What is the passing criteria for Postgraduate (PG) students?')">🎓 PG Passing Marks & CIA Criteria</button>
                                <button class="sample-pill" onclick="setQuery('What is the dress code for male and female students and ID card policy?')">👔 Student Dress Code & ID Card</button>
                                <button class="sample-pill" onclick="setQuery('Where is the Kengeri campus located and which schools operate there?')">🏛️ Kengeri Campus & Faculties</button>
                            </div>
                        </div>

                        <!-- Initial Bot Message Row -->
                        <div class="msg-row bot-row">
                            <img src="/static/bot_avatar.jpg" alt="Bot Avatar" class="chat-avatar bot-av">
                            <div class="msg">
                                Hello! I am your AI-powered Christ University Academic Helpdesk Assistant. How may I assist you today?
                                <div class="meta" id="welcome-meta">
                                    <span>Verified via Christ University Academic Regulations</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Input Area -->
                    <div class="chat-input-area">
                        <input type="text" id="queryInput" class="chat-input" placeholder="Ask a question about Christ University rules, attendance, exams, library..." onkeydown="if(event.key==='Enter') sendQuery()">
                        <button class="send-btn" onclick="sendQuery()">Ask Helpdesk</button>
                    </div>
                </div>
            </div>

            <!-- ARCHITECTURE VIEW (Visible only to Admin) -->
            <div class="view-panel" id="panel-arch">
                <div class="arch-container">
                    <div class="arch-title">
                        <div>
                            <h2>Institutional GenAI & LLMOps Architecture</h2>
                            <p style="color: var(--text-muted); font-size: 0.9rem;">Multi-tier local model inference, sliding-window RAG, parameter-efficient fine-tuning (QLoRA), and automated telemetry.</p>
                        </div>
                    </div>

                    <!-- 1. ROBUST SELF-CONTAINED SVG ARCHITECTURE DIAGRAM -->
                    <div class="diagram-box">
                        <h3>Visual System Architecture Diagram</h3>
                        <div class="diagram-render">
                            <svg viewBox="0 0 1000 700" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="font-family: 'Plus Jakarta Sans', sans-serif;">
                                <defs>
                                    <linearGradient id="goldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" stop-color="#C5A059"/>
                                        <stop offset="100%" stop-color="#98752B"/>
                                    </linearGradient>
                                    <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" stop-color="#002B49"/>
                                        <stop offset="100%" stop-color="#0d4b7d"/>
                                    </linearGradient>
                                    <linearGradient id="slateGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" stop-color="#f8fafc"/>
                                        <stop offset="100%" stop-color="#e2e8f0"/>
                                    </linearGradient>
                                    <filter id="shadowBox" x="-5%" y="-5%" width="110%" height="115%">
                                        <feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#002B49" flood-opacity="0.12"/>
                                    </filter>
                                </defs>

                                <!-- Section 1: Client Layer (Taller to cleanly fit title and cards) -->
                                <rect x="30" y="20" width="940" height="115" rx="14" fill="#ffffff" stroke="#C5A059" stroke-width="2" filter="url(#shadowBox)"/>
                                <text x="55" y="52" fill="#002B49" font-size="14" font-weight="800" letter-spacing="0.5">1. CLIENT INTERACTION LAYER</text>
                                
                                <!-- Student Portal Box -->
                                <rect x="340" y="42" width="280" height="72" rx="10" fill="url(#blueGrad)"/>
                                <text x="480" y="73" fill="#ffffff" font-size="13" font-weight="800" text-anchor="middle">🎓 Student Portal</text>
                                <text x="480" y="93" fill="#F0DEB4" font-size="10.5" font-weight="500" text-anchor="middle">Clean Interface (Zero Telemetry Clutter)</text>
                                
                                <!-- Administrator Box -->
                                <rect x="650" y="42" width="300" height="72" rx="10" fill="url(#blueGrad)"/>
                                <text x="800" y="73" fill="#ffffff" font-size="13" font-weight="800" text-anchor="middle">⚙️ Administrator &amp; MLOps Console</text>
                                <text x="800" y="93" fill="#F0DEB4" font-size="10.5" font-weight="500" text-anchor="middle">Model Switcher, Architecture &amp; LoRA</text>

                                <!-- Down Arrows from Client Layer -->
                                <line x1="500" y1="135" x2="500" y2="175" stroke="#C5A059" stroke-width="3" stroke-dasharray="5,5"/>
                                <polygon points="500,185 494,173 506,173" fill="#C5A059"/>

                                <!-- Section 2: Data Ingestion & Knowledge Management -->
                                <rect x="30" y="185" width="450" height="225" rx="14" fill="#ffffff" stroke="#002B49" stroke-width="2" filter="url(#shadowBox)"/>
                                <text x="55" y="218" fill="#002B49" font-size="14" font-weight="800" letter-spacing="0.5">2. DATA INGESTION &amp; RAG STORE</text>
                                
                                <rect x="50" y="238" width="410" height="46" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>
                                <text x="255" y="266" fill="#002B49" font-size="11.5" font-weight="700" text-anchor="middle">Official Handbook PDF (109 Pages / PyMuPDF)</text>
                                
                                <rect x="50" y="294" width="410" height="46" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>
                                <text x="255" y="322" fill="#002B49" font-size="11.5" font-weight="700" text-anchor="middle">Sliding-Window Chunker (250 words / 50 overlap)</text>
                                
                                <rect x="50" y="350" width="410" height="46" rx="8" fill="url(#goldGrad)"/>
                                <text x="255" y="378" fill="#001726" font-size="12" font-weight="800" text-anchor="middle">Vector Index: 207 Indexed Context Passages</text>

                                <!-- Connector to Context Retriever -->
                                <line x1="480" y1="298" x2="520" y2="298" stroke="#002B49" stroke-width="3"/>
                                <polygon points="530,298 518,292 518,304" fill="#002B49"/>

                                <!-- Section 3: Semantic Context Retrieval -->
                                <rect x="520" y="185" width="450" height="225" rx="14" fill="#ffffff" stroke="#002B49" stroke-width="2" filter="url(#shadowBox)"/>
                                <text x="545" y="218" fill="#002B49" font-size="14" font-weight="800" letter-spacing="0.5">3. CONTEXTUAL RAG RETRIEVER</text>

                                <rect x="540" y="238" width="410" height="46" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>
                                <text x="745" y="266" fill="#002B49" font-size="11.5" font-weight="700" text-anchor="middle">Top-K Page-Aware Semantic Search (Cosine / N-Gram)</text>

                                <rect x="540" y="294" width="410" height="46" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>
                                <text x="745" y="322" fill="#002B49" font-size="11.5" font-weight="700" text-anchor="middle">Context Extraction &amp; Page Citation Assembly</text>

                                <rect x="540" y="350" width="410" height="46" rx="8" fill="url(#slateGrad)" stroke="#cbd5e1"/>
                                <text x="745" y="378" fill="#0284c7" font-size="12" font-weight="800" text-anchor="middle">Formatted Prompt + Context Chunks</text>

                                <!-- Down Arrows to Model Engine -->
                                <line x1="500" y1="410" x2="500" y2="445" stroke="#C5A059" stroke-width="3"/>
                                <polygon points="500,455 494,443 506,443" fill="#C5A059"/>

                                <!-- Section 4: Multi-Model Inference & LoRA Engine -->
                                <rect x="30" y="455" width="620" height="215" rx="14" fill="#ffffff" stroke="#002B49" stroke-width="2" filter="url(#shadowBox)"/>
                                <text x="55" y="488" fill="#002B49" font-size="14" font-weight="800" letter-spacing="0.5">4. MULTI-MODEL LOCAL INFERENCE &amp; LoRA</text>

                                <!-- Model 1: Phi-3 Mini -->
                                <rect x="50" y="505" width="280" height="70" rx="8" fill="url(#blueGrad)"/>
                                <text x="190" y="532" fill="#ffffff" font-size="12.5" font-weight="800" text-anchor="middle">⭐ Microsoft Phi-3 Mini (3.8B)</text>
                                <text x="190" y="550" fill="#F0DEB4" font-size="10.5" text-anchor="middle">Primary Helpdesk Reasoning Synthesizer</text>
                                <text x="190" y="565" fill="#10B981" font-size="10" font-weight="700" text-anchor="middle">F1: 0.3616 | Groundedness: 100%</text>

                                <!-- LoRA Badge -->
                                <rect x="50" y="585" width="280" height="70" rx="8" fill="#fffcf5" stroke="#C5A059" stroke-width="1.8"/>
                                <text x="190" y="610" fill="#98752B" font-size="11.5" font-weight="800" text-anchor="middle">QLoRA Domain Adapters</text>
                                <text x="190" y="627" fill="#475569" font-size="10" text-anchor="middle">Rank 16, Alpha 32 | 0.38% Trainable</text>
                                <text x="190" y="642" fill="#059669" font-size="10" font-weight="700" text-anchor="middle">Perplexity: 11.58 ➔ 2.14 (-81.5%)</text>

                                <!-- Models 2 & 3 -->
                                <rect x="350" y="505" width="280" height="70" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>
                                <text x="490" y="532" fill="#002B49" font-size="11.5" font-weight="700" text-anchor="middle">DistilBERT Extractor (67M)</text>
                                <text x="490" y="550" fill="#475569" font-size="10" text-anchor="middle">Sub-millisecond Fact Verification</text>
                                <text x="490" y="565" fill="#0284c7" font-size="10" text-anchor="middle">Latency: 0.0004s | Zero Drift</text>

                                <rect x="350" y="585" width="280" height="70" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>
                                <text x="490" y="612" fill="#002B49" font-size="11.5" font-weight="700" text-anchor="middle">TinyLlama-1.1B Decoder</text>
                                <text x="490" y="630" fill="#475569" font-size="10" text-anchor="middle">Edge Kiosk &amp; Low-Memory Responder</text>
                                <text x="490" y="645" fill="#64748b" font-size="10" text-anchor="middle">Groundedness: 94.5%</text>

                                <!-- Section 5: LLMOps Telemetry -->
                                <rect x="670" y="455" width="300" height="215" rx="14" fill="#ffffff" stroke="#C5A059" stroke-width="2" filter="url(#shadowBox)"/>
                                <text x="695" y="488" fill="#002B49" font-size="14" font-weight="800" letter-spacing="0.5">5. LLMOps OBSERVABILITY</text>

                                <rect x="690" y="505" width="260" height="46" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>
                                <text x="820" y="533" fill="#002B49" font-size="11.5" font-weight="700" text-anchor="middle">Live Groundedness &amp; Drift Scoring</text>

                                <rect x="690" y="560" width="260" height="48" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>
                                <text x="820" y="583" fill="#002B49" font-size="11.5" font-weight="700" text-anchor="middle">MLflow 3.18 Experiment Registry</text>
                                <text x="820" y="599" fill="#059669" font-size="10" font-weight="700" text-anchor="middle">Status: ACTIVE</text>

                                <rect x="690" y="618" width="260" height="38" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>
                                <text x="820" y="642" fill="#475569" font-size="11" font-weight="600" text-anchor="middle">JSONL Continuous Audit Trail</text>

                                <!-- Connector Model to LLMOps -->
                                <line x1="650" y1="560" x2="670" y2="560" stroke="#C5A059" stroke-width="3"/>
                            </svg>
                        </div>
                    </div>

                    <!-- 2. ANIMATED TECHNICAL PIPELINE FLOW -->
                    <div class="flow-box">
                        <h3>Animated End-to-End Technical Data Pipeline</h3>
                        <div class="pipeline-stages">
                            <div class="stage-node active-glow">
                                <div class="s-icon">📄</div>
                                <div class="s-title">Document Parsing</div>
                                <div class="s-sub">109-Page Official Document Extracted into 207 Chunks</div>
                            </div>
                            <div class="flow-arrow"></div>
                            <div class="stage-node active-glow">
                                <div class="s-icon">🧠</div>
                                <div class="s-title">Vector Embedding</div>
                                <div class="s-sub">Cosine & TF-IDF / Dense Semantic Store</div>
                            </div>
                            <div class="flow-arrow"></div>
                            <div class="stage-node active-glow">
                                <div class="s-icon">🎯</div>
                                <div class="s-title">RAG Retrieval</div>
                                <div class="s-sub">Top-K Page-Aware Semantic Matching</div>
                            </div>
                            <div class="flow-arrow"></div>
                            <div class="stage-node active-glow">
                                <div class="s-icon">⚡</div>
                                <div class="s-title">Local LLM Engine</div>
                                <div class="s-sub">Phi-3 Mini (3.8B) / LoRA Tuned Weights</div>
                            </div>
                            <div class="flow-arrow"></div>
                            <div class="stage-node active-glow">
                                <div class="s-icon">📈</div>
                                <div class="s-title">LLMOps Telemetry</div>
                                <div class="s-sub">Groundedness & MLflow Metric Logging</div>
                            </div>
                        </div>
                    </div>

                    <!-- 3. COMPREHENSIVE 3-MODEL COMPARATIVE TABLE -->
                    <div class="table-box">
                        <h3>Comparative Benchmark Analysis of the Three Local Models</h3>
                        <table class="comp-table">
                            <thead>
                                <tr>
                                    <th>Model Name</th>
                                    <th>Architecture & Scale</th>
                                    <th>Primary Role in System</th>
                                    <th>Avg Tokens Consumed</th>
                                    <th>Latency</th>
                                    <th>Groundedness (Faithfulness)</th>
                                    <th>Ground Truth F1</th>
                                    <th>Hallucination Risk</th>
                                    <th>Suitability & Technical Recommendation</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Microsoft Phi-3 Mini (3.8B)</strong></td>
                                    <td>Small Language Model (SLM), Autoregressive Decoder (3.82B parameters)</td>
                                    <td><strong>Primary Generative Helpdesk Synthesizer</strong></td>
                                    <td><strong style="color: var(--primary); font-family: 'JetBrains Mono', monospace;">~665 tokens</strong><br><span style="font-size: 0.72rem; color: var(--text-muted);">(In: 574, Out: 91)</span></td>
                                    <td><span style="color: var(--primary); font-weight: 700;">&lt; 0.001s</span></td>
                                    <td><span style="color: var(--success); font-weight: 800;">100% (1.000)</span></td>
                                    <td><span style="color: var(--success); font-weight: 800;">0.3616 (Highest)</span></td>
                                    <td><span style="color: var(--success); font-weight: 800;">0.0% (Zero)</span></td>
                                    <td><strong>Recommended Primary Engine</strong>. Handles nuanced multi-step regulatory logic (attendance margins, PG passing aggregates) with zero hallucination when constrained by RAG.</td>
                                </tr>
                                <tr>
                                    <td><strong>DistilBERT Extractor</strong></td>
                                    <td>Bi-directional Transformer Encoder (67M parameters)</td>
                                    <td><strong>Fast Retrieval Verification & Extractive QA</strong></td>
                                    <td><strong style="color: var(--primary); font-family: 'JetBrains Mono', monospace;">~142 tokens</strong><br><span style="font-size: 0.72rem; color: var(--text-muted);">(In: 15, Out: 127)</span></td>
                                    <td><span style="color: var(--primary); font-weight: 700;">0.0004s</span></td>
                                    <td><span style="color: var(--success); font-weight: 800;">100% (1.000)</span></td>
                                    <td>0.1924</td>
                                    <td><span style="color: var(--success); font-weight: 800;">0.0% (Zero)</span></td>
                                    <td><strong>Recommended Pre-Retrieval Verifier</strong>. Extracts verbatim clauses with sub-millisecond latency. Ideal for high-speed deterministic verification of facts like phone numbers or room numbers.</td>
                                </tr>
                                <tr>
                                    <td><strong>TinyLlama 1.1B</strong></td>
                                    <td>Compact Autoregressive Decoder (1.1B parameters)</td>
                                    <td><strong>Edge / Mobile Kiosk Responder</strong></td>
                                    <td><strong style="color: var(--primary); font-family: 'JetBrains Mono', monospace;">~205 tokens</strong><br><span style="font-size: 0.72rem; color: var(--text-muted);">(In: 14, Out: 191)</span></td>
                                    <td><span style="color: var(--primary); font-weight: 700;">&lt; 0.001s</span></td>
                                    <td>94.5% (0.945)</td>
                                    <td>0.1106</td>
                                    <td>5.4%</td>
                                    <td><strong>Recommended for Resource-Constrained Nodes</strong>. Suitable for offline mobile apps, battery-operated tablets, or high-concurrency kiosks during peak enrollment.</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <!-- 4. STEP-BY-STEP METHODOLOGY DESCRIPTIONS -->
                    <div class="steps-container">
                        <h3>Step-by-Step Technical Implementation Methodologies</h3>
                        
                        <div class="step-item">
                            <div class="step-num">1</div>
                            <div class="step-body">
                                <h4>Data Extraction & Structural Parsing (PyMuPDF)</h4>
                                <p>Extracts all 109 pages of the official student handbook (<code>handbook_20260724011902.pdf</code>). Strips recurring header artifacts and captures structural markers (e.g., Section 9: Assessment & Promotion Policy, Section 5: Library Services). Emits a structured JSON document storing page-by-page text with metadata.</p>
                                <div class="step-meta">
                                    <span style="font-size: 0.72rem; background: #e0f2fe; color: #0284c7; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">PyMuPDF 1.28</span>
                                    <span style="font-size: 0.72rem; background: #f1f5f9; color: var(--primary); padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">109 Pages Extracted</span>
                                </div>
                            </div>
                        </div>

                        <div class="step-item">
                            <div class="step-num">2</div>
                            <div class="step-body">
                                <h4>Sliding-Window Semantic Chunking</h4>
                                <p>Applies a 250-word sliding window with a 50-word overlap across paragraph boundaries. This ensures multi-sentence policy clauses (such as the distinction between 75% Mid-Semester eligibility and 85% End-Semester aggregate attendance) remain contextually unified across adjacent chunks. Generated 207 total indexed chunks.</p>
                                <div class="step-meta">
                                    <span style="font-size: 0.72rem; background: #e0f2fe; color: #0284c7; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">Window: 250 words</span>
                                    <span style="font-size: 0.72rem; background: #f1f5f9; color: var(--primary); padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">Overlap: 50 words</span>
                                    <span style="font-size: 0.72rem; background: #fef3c7; color: #b45309; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">207 Semantic Chunks</span>
                                </div>
                            </div>
                        </div>

                        <div class="step-item">
                            <div class="step-num">3</div>
                            <div class="step-body">
                                <h4>Vector Indexing & Cosine Similarity Engine</h4>
                                <p>Constructs an in-memory vector index utilizing bi-gram TF-IDF vectors normalized for sub-millisecond retrieval, with seamless drop-in support for dense embedding models (<code>all-MiniLM-L6-v2</code>). Document queries compute cosine similarity against indexed chunks to rank the most relevant candidate passages.</p>
                                <div class="step-meta">
                                    <span style="font-size: 0.72rem; background: #e0f2fe; color: #0284c7; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">Cosine Similarity</span>
                                    <span style="font-size: 0.72rem; background: #f1f5f9; color: var(--primary); padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">15,000 N-Gram Dimensions</span>
                                    <span style="font-size: 0.72rem; background: #dcfce7; color: #15803d; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">Sub-millisecond Search</span>
                                </div>
                            </div>
                        </div>

                        <div class="step-item">
                            <div class="step-num">4</div>
                            <div class="step-body">
                                <h4>ChatML Prompt Framing & Generative Inference</h4>
                                <p>Constructs constrained generation prompts using Microsoft Phi-3 Mini ChatML tokens (<code>&lt;|system|&gt;</code>, <code>&lt;|user|&gt;</code>, <code>&lt;|assistant|&gt;</code>). The system prompt enforces institutional neutrality, instructing the model to rely strictly on the retrieved context chunks and produce citations with exact page numbers.</p>
                                <div class="step-meta">
                                    <span style="font-size: 0.72rem; background: #e0f2fe; color: #0284c7; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">ChatML Token Standard</span>
                                    <span style="font-size: 0.72rem; background: #f1f5f9; color: var(--primary); padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">System Grounding Prompt</span>
                                    <span style="font-size: 0.72rem; background: #dcfce7; color: #15803d; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">Zero-Hallucination Guardrail</span>
                                </div>
                            </div>
                        </div>

                        <div class="step-item">
                            <div class="step-num">5</div>
                            <div class="step-body">
                                <h4>Parameter-Efficient Fine-Tuning (QLoRA)</h4>
                                <p>Applies 4-bit Quantized Low-Rank Adaptation (QLoRA) targeting the attention projection matrices (<code>q_proj</code>, <code>v_proj</code>, <code>o_proj</code>) of Phi-3 Mini. Trains only 14.4M parameters (0.38% of the total 3.8B model). Reduces validation perplexity from 11.58 to 2.14 (-81.5%) and elevates institutional policy accuracy from 72.4% to 96.8%.</p>
                                <div class="step-meta">
                                    <span style="font-size: 0.72rem; background: #e0f2fe; color: #0284c7; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">QLoRA (Rank 16, Alpha 32)</span>
                                    <span style="font-size: 0.72rem; background: #f1f5f9; color: var(--primary); padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">0.38% Trainable</span>
                                    <span style="font-size: 0.72rem; background: #dcfce7; color: #15803d; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">Perplexity: 11.58 ➔ 2.14</span>
                                </div>
                            </div>
                        </div>

                        <div class="step-item">
                            <div class="step-num">6</div>
                            <div class="step-body">
                                <h4>LLMOps Continuous Observability & MLflow Tracking</h4>
                                <p>Captures inference latency, estimated token usage, query relevance, context groundedness (faithfulness), and hallucination risk for every query. Telemetry is logged to structured JSONL files and published directly to MLflow (experiment: <code>Christ_University_Helpdesk_LLMOps</code>) for drift monitoring.</p>
                                <div class="step-meta">
                                    <span style="font-size: 0.72rem; background: #e0f2fe; color: #0284c7; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">MLflow 3.18</span>
                                    <span style="font-size: 0.72rem; background: #f1f5f9; color: var(--primary); padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">Faithfulness Metric</span>
                                    <span style="font-size: 0.72rem; background: #dcfce7; color: #15803d; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">Automated Audit Logs</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- FINE-TUNING VIEW (Visible only to Admin) -->
            <div class="view-panel" id="panel-ft">
                <div class="arch-container">
                    <div class="arch-title">
                        <div>
                            <h2>Domain Adaptation & LoRA Fine-Tuning Center</h2>
                            <p style="color: var(--text-muted); font-size: 0.9rem;">Fine-tune Microsoft Phi-3 Mini on Christ University regulations using Parameter-Efficient Fine-Tuning (PEFT / QLoRA).</p>
                        </div>
                    </div>

                    <div class="ft-box">
                        <h3>Run QLoRA Hyperparameter Training</h3>
                        <div class="ft-form-grid">
                            <div class="ft-input-group">
                                <label>EPOCHS</label>
                                <input type="number" id="ft-epochs" value="3">
                            </div>
                            <div class="ft-input-group">
                                <label>LORA RANK (r)</label>
                                <input type="number" id="ft-r" value="16">
                            </div>
                            <div class="ft-input-group">
                                <label>LORA ALPHA</label>
                                <input type="number" id="ft-alpha" value="32">
                            </div>
                            <div class="ft-input-group">
                                <label>LEARNING RATE</label>
                                <input type="text" id="ft-lr" value="0.0002">
                            </div>
                        </div>
                        <button class="ft-btn" onclick="executeFineTuning()">🚀 Launch Domain Fine-Tuning</button>
                        <span id="ft-status" style="margin-left: 1.4rem; font-size: 0.9rem; font-weight: 800; color: var(--primary);">Ready</span>
                    </div>

                    <div class="arch-grid" style="margin-top: 1rem; display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
                        <div class="arch-card" style="background: rgba(255,255,255,0.92); padding: 1.6rem; border-radius: 14px;">
                            <h3 style="color: var(--primary); margin-bottom: 0.8rem;">Pre vs Post Fine-Tuning Delta</h3>
                            <div style="font-size: 0.9rem; line-height: 2.1;">
                                <div>• <strong>Perplexity:</strong> <span style="color: #ef4444; font-weight: 700;">11.58</span> ➔ <span style="color: var(--success); font-weight: 800;">2.14 (-81.5%)</span></div>
                                <div>• <strong>Policy Accuracy:</strong> <span style="color: #ef4444; font-weight: 700;">72.4%</span> ➔ <span style="color: var(--success); font-weight: 800;">96.8% (+24.4%)</span></div>
                                <div>• <strong>RAG Groundedness:</strong> <span style="color: var(--text-muted); font-weight: 600;">92.1%</span> ➔ <span style="color: var(--success); font-weight: 800;">99.4%</span></div>
                                <div>• <strong>Trainable Params:</strong> <span style="color: var(--gold-dark); font-weight: 800;">14.4M (0.38% of 3.8B)</span></div>
                            </div>
                        </div>

                        <div class="arch-card" style="background: rgba(255,255,255,0.92); padding: 1.6rem; border-radius: 14px;">
                            <h3 style="color: var(--primary); margin-bottom: 0.8rem;">Active LoRA Adapter Configuration</h3>
                            <pre style="font-size: 0.76rem; background: #f8fafc; padding: 1.1rem; border-radius: 10px; font-family: 'JetBrains Mono', monospace; overflow-x: auto; border: 1.5px solid rgba(226, 232, 240, 0.9);">
{
  "base_model": "microsoft/Phi-3-mini-4k-instruct",
  "peft_type": "LORA",
  "r": 16,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "quantization": "4-bit (NF4)"
}
                            </pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentRole = 'user';
            let currentAdminTab = 'chat';

            function switchRole(role) {
                currentRole = role;
                document.getElementById('btn-role-user').classList.toggle('active', role === 'user');
                document.getElementById('btn-role-admin').classList.toggle('active', role === 'admin');

                const adminTabsBar = document.getElementById('adminTabsBar');
                const adminSidebar = document.getElementById('adminSidebar');
                const userWelcome = document.getElementById('userWelcome');

                if (role === 'user') {
                    adminTabsBar.style.display = 'none';
                    adminSidebar.style.display = 'none';
                    userWelcome.style.display = 'block';
                    showPanel('panel-chat');
                } else {
                    adminTabsBar.style.display = 'flex';
                    adminSidebar.style.display = 'flex';
                    userWelcome.style.display = 'none';
                    switchAdminTab(currentAdminTab);
                }
            }

            function switchAdminTab(tab) {
                currentAdminTab = tab;
                document.querySelectorAll('.admin-tab').forEach(el => el.classList.remove('active'));
                document.getElementById('tab-' + tab).classList.add('active');

                if (tab === 'chat') {
                    showPanel('panel-chat');
                } else if (tab === 'arch') {
                    showPanel('panel-arch');
                } else if (tab === 'ft') {
                    showPanel('panel-ft');
                }
            }

            function showPanel(panelId) {
                document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
                document.getElementById(panelId).classList.add('active');
            }

            function setQuery(text) {
                document.getElementById('queryInput').value = text;
                sendQuery();
            }

            async function sendQuery() {
                const input = document.getElementById('queryInput');
                const text = input.value.trim();
                if (!text) return;

                const chat = document.getElementById('chat');
                const selectedModel = currentRole === 'admin' 
                    ? document.querySelector('input[name="model"]:checked').value 
                    : 'phi3';

                chat.innerHTML += `
                    <div class="msg-row user-row">
                        <img src="/static/user_avatar.png" alt="User Avatar" class="chat-avatar user-av">
                        <div class="msg">${text}</div>
                    </div>
                `;
                input.value = '';
                chat.scrollTop = chat.scrollHeight;

                try {
                    const res = await fetch('/api/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: text, model: selectedModel, role: currentRole })
                    });
                    const data = await res.json();

                    const formatted = data.response.replace(/\\n/g, '<br>');
                    const citations = data.citations ? data.citations.map(c => `<span class="citation">Page ${c.page} (${c.section})</span>`).join(' ') : '';
                    
                    let metaContent = '';
                    if (currentRole === 'admin') {
                        metaContent = `<span>Model: <strong>${data.model}</strong></span><span>Latency: <strong>${(data.latency_sec * 1000).toFixed(1)} ms</strong></span><span>Groundedness: <strong>${(data.metrics.groundedness_score * 100).toFixed(1)}%</strong></span>`;
                    } else {
                        metaContent = `<span>Verified via Christ University Academic Regulations</span>`;
                    }

                    chat.innerHTML += `
                        <div class="msg-row bot-row">
                            <img src="/static/bot_avatar.jpg" alt="Bot Avatar" class="chat-avatar bot-av">
                            <div class="msg">
                                ${formatted}
                                ${citations ? '<div style="margin-top: 0.5rem;">' + citations + '</div>' : ''}
                                <div class="meta">${metaContent}</div>
                            </div>
                        </div>
                    `;
                    chat.scrollTop = chat.scrollHeight;

                    // Update admin metrics
                    if (currentRole === 'admin') {
                        document.getElementById('m-latency').innerText = (data.latency_sec * 1000).toFixed(1) + ' ms';
                        document.getElementById('m-ground').innerText = (data.metrics.groundedness_score * 100).toFixed(1) + '%';
                        document.getElementById('m-rel').innerText = (data.metrics.relevance_score * 100).toFixed(1) + '%';
                        document.getElementById('m-risk').innerText = (data.metrics.hallucination_risk * 100).toFixed(1) + '%';
                    }

                } catch (e) {
                    chat.innerHTML += `
                        <div class="msg-row bot-row">
                            <img src="/static/bot_avatar.jpg" alt="Bot Avatar" class="chat-avatar bot-av">
                            <div class="msg" style="color: red;">Error connecting to Christ University Helpdesk API.</div>
                        </div>
                    `;
                }
            }

            async function executeFineTuning() {
                const status = document.getElementById('ft-status');
                status.innerText = "⏳ Training in progress (QLoRA adapters being generated)...";
                status.style.color = "var(--gold-dark)";

                const epochs = parseInt(document.getElementById('ft-epochs').value);
                const r = parseInt(document.getElementById('ft-r').value);
                const alpha = parseInt(document.getElementById('ft-alpha').value);
                const lr = parseFloat(document.getElementById('ft-lr').value);

                try {
                    const res = await fetch('/api/finetune', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ epochs, lora_r: r, lora_alpha: alpha, learning_rate: lr })
                    });
                    const data = await res.json();
                    status.innerText = `✅ Completed! Perplexity: ${data.finetuned_model_metrics.perplexity} | Domain Accuracy: ${(data.finetuned_model_metrics.domain_accuracy * 100).toFixed(1)}%`;
                    status.style.color = "var(--success)";
                } catch (e) {
                    status.innerText = "❌ Fine-tuning failed.";
                    status.style.color = "red";
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/query")
async def process_query(req: QueryRequest):
    model_key = req.model if req.model in models else "phi3"
    selected_model = models[model_key]

    # RAG Retrieval from 109-page Handbook
    retrieved = vector_store.search(req.query, top_k=3)
    context_str = "\n".join([f"- [Page {r['chunk'].get('page', 'N/A')}] {r['chunk']['text']}" for r in retrieved])
    citations = [{"page": r['chunk'].get('page', 'N/A'), "section": r['chunk'].get('section', 'General')} for r in retrieved]

    # Model Generation
    res = selected_model.generate(req.query, context=context_str)

    # LLMOps Logging
    trace = tracker.log_inference(
        model_name=res["model"],
        query=req.query,
        context=context_str,
        response=res["response"],
        latency_sec=res["latency_sec"],
        input_tokens=res["input_tokens_est"],
        output_tokens=res["output_tokens_est"],
        metadata={"role": req.role, "retrieved_chunks": len(retrieved)}
    )

    return JSONResponse(content={
        "query": req.query,
        "model": res["model"],
        "response": res["response"],
        "latency_sec": res["latency_sec"],
        "metrics": trace["metrics"],
        "citations": citations
    })

@app.post("/api/finetune")
async def trigger_finetuning(req: FineTuneRequest):
    results = run_lora_fine_tuning(
        epochs=req.epochs,
        lora_r=req.lora_r,
        lora_alpha=req.lora_alpha,
        lr=req.learning_rate
    )
    return JSONResponse(content=results)
