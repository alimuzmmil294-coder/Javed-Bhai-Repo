# 💳 AI Personal Expense Tracker (Streamlit + Gemini 2.5)

A modular, single-page Personal Expense Tracker built with **Python**, **Streamlit**, **SQLite**, and the official **Google GenAI SDK (`google-genai`)**.

## ✨ Key Engineering Features

* **Full CRUD Persistence:** Uses a zero-config local SQLite database (`expenses.db`) with parameterized SQL queries to prevent SQL injection.
* **Inline Grid Editing & Batch Deletion:** Edit cells directly inside `st.data_editor` or select multiple rows to delete in a single atomic transaction.
* **AI Natural-Language Quick-Add:** Type conversational inputs like *"Spent 1450 on a textbook yesterday via Bank Transfer"* and let `gemini-2.5-flash` parse them into validated Pydantic schemas before saving.
* **AI Financial Auditor:** Generates personalized budget audits and anomaly detection grounded strictly in your real transaction history.

---

## 🚀 Quick Start

### 1. Create and Activate a Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
