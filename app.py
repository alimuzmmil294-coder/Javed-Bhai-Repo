import os
import sqlite3
from datetime import date, datetime
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables from .env if present
load_dotenv()

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
DB_PATH = "expenses.db"
MODEL_ID = "gemini-2.5-flash"
CATEGORIES = [
    "Food & Dining",
    "Transport",
    "Utilities & Bills",
    "Education",
    "Software & Tech",
    "Shopping",
    "Health & Fitness",
    "Entertainment",
    "Other",
]

st.set_page_config(
    page_title="AI Personal Expense Tracker",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# 2. PYDANTIC SCHEMA FOR GEMINI STRUCTURED OUTPUT
# ==========================================
class ParsedExpense(BaseModel):
    date: str = Field(description="Expense date in YYYY-MM-DD format.")
    category: str = Field(description=f"Must be one of: {', '.join(CATEGORIES)}")
    amount: float = Field(description="Numeric transaction amount.")
    description: str = Field(description="Concise summary of the expense.")
    payment_method: str = Field(description="One of: Cash, Card, Bank Transfer, Mobile Wallet")


# ==========================================
# 3. DATABASE LAYER (SQLite CRUD)
# ==========================================
def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expense_date TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount >= 0),
                payment_method TEXT NOT NULL,
                description TEXT
            )
            """
        )
        conn.commit()


def create_expense(
    expense_date: str, category: str, amount: float, payment_method: str, description: str
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO expenses (expense_date, category, amount, payment_method, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            (expense_date, category, round(float(amount), 2), payment_method, description.strip()),
        )
        conn.commit()


def read_expenses() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM expenses ORDER BY expense_date DESC, id DESC", conn
        )
    if not df.empty:
        df["expense_date"] = pd.to_datetime(df["expense_date"]).dt.date
    return df


def update_expense(
    expense_id: int,
    expense_date: str,
    category: str,
    amount: float,
    payment_method: str,
    description: str,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE expenses
            SET expense_date = ?, category = ?, amount = ?, payment_method = ?, description = ?
            WHERE id = ?
            """,
            (
                str(expense_date),
                category,
                round(float(amount), 2),
                payment_method,
                description.strip(),
                int(expense_id),
            ),
        )
        conn.commit()


def delete_expenses(expense_ids: list[int]) -> None:
    if not expense_ids:
        return
    placeholders = ",".join(["?"] * len(expense_ids))
    with get_connection() as conn:
        conn.execute(
            f"DELETE FROM expenses WHERE id IN ({placeholders})",
            [int(i) for i in expense_ids],
        )
        conn.commit()


# ==========================================
# 4. GEMINI AI SERVICE LAYER
# ==========================================
def get_gemini_client(api_key: str) -> Optional[genai.Client]:
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def ai_parse_natural_language_expense(client: genai.Client, raw_text: str) -> ParsedExpense:
    today_str = date.today().isoformat()
    prompt = f"""
    Today's date is {today_str}.
    Extract the expense details from the following user input: "{raw_text}".
    If the date is relative (e.g., 'yesterday' or 'today') or omitted, resolve it relative to {today_str}.
    Map the category strictly to one of: {', '.join(CATEGORIES)}.
    Map payment_method strictly to one of: Cash, Card, Bank Transfer, Mobile Wallet (default to Cash if unspecified).
    """
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ParsedExpense,
            temperature=0.1,
        ),
    )
    return ParsedExpense.model_validate_json(response.text)


def ai_generate_financial_insights(client: genai.Client, df: pd.DataFrame, user_query: str) -> str:
    if df.empty:
        return "No expense records available to analyze yet. Add some expenses first!"

    records_csv = df.to_csv(index=False)
    system_instruction = (
        "You are a senior financial analyst. Analyze the user's expense data concisely. "
        "Highlight spending anomalies, top burn categories, and concrete savings actions. "
        "Use clean Markdown formatting and exact numbers from the dataset."
    )
    prompt = f"Expense Dataset (CSV):\n{records_csv}\n\nUser Request: {user_query}"

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
        ),
    )
    return response.text


# ==========================================
# 5. UI PRESENTATION LAYER
# ==========================================
def main() -> None:
    init_db()

    # --- Sidebar: Configuration & Manual Add ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        env_key = os.getenv("GEMINI_API_KEY", "")
        api_key = st.text_input(
            "Gemini API Key",
            value=env_key,
            type="password",
            help="Get your key from Google AI Studio. Stored in session only.",
        )
        client = get_gemini_client(api_key)

        st.divider()
        st.subheader("➕ Add New Expense")
        with st.form("add_expense_form", clear_on_submit=True):
            exp_date = st.date_input("Date", value=date.today())
            category = st.selectbox("Category", options=CATEGORIES)
            amount = st.number_input("Amount", min_value=0.01, step=10.0, format="%.2f")
            payment_method = st.selectbox(
                "Payment Method",
                options=["Cash", "Card", "Bank Transfer", "Mobile Wallet"],
            )
            description = st.text_input(
                "Description", placeholder="e.g., Afternoon coffee & snacks"
            )
            submitted = st.form_submit_button("Save Expense", use_container_width=True, type="primary")

            if submitted:
                if not description.strip():
                    st.error("Please enter a short description.")
                else:
                    create_expense(
                        exp_date.isoformat(),
                        category,
                        amount,
                        payment_method,
                        description,
                    )
                    st.toast("Expense added!", icon="✅")
                    st.rerun()

    # --- Main Header ---
    st.title("💳 Personal Expense Tracker & AI Auditor")
    st.caption("Full CRUD persistence with SQLite, inline editing, and Gemini 2.5 Flash intelligence.")

    # --- AI Quick-Add Bar ---
    with st.expander("✨ AI Quick-Add (Log expenses in plain English)", expanded=False):
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            nl_input = st.text_input(
                "Describe your expense",
                placeholder="e.g., Spent 1450 on a textbook yesterday via Bank Transfer",
                label_visibility="collapsed",
            )
        with col_btn:
            ai_add_clicked = st.button("Parse & Add", use_container_width=True)

        if ai_add_clicked:
            if not client:
                st.warning("Please provide your Gemini API Key in the sidebar first.")
            elif not nl_input.strip():
                st.warning("Type an expense sentence to parse.")
            else:
                with st.spinner("Extracting structured expense via Gemini..."):
                    try:
                        parsed = ai_parse_natural_language_expense(client, nl_input)
                        create_expense(
                            parsed.date,
                            parsed.category if parsed.category in CATEGORIES else "Other",
                            parsed.amount,
                            parsed.payment_method,
                            parsed.description,
                        )
                        st.success(
                            f"Added: **{parsed.description}** ({parsed.category}) — **{parsed.amount:.2f}** on {parsed.date}"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"AI Parsing Error: {e}")

    # Load Data
    df = read_expenses()

    # --- KPI Metrics Row ---
    if not df.empty:
        total_spent = df["amount"].sum()
        tx_count = len(df)
        top_cat = df.groupby("category")["amount"].sum().idxmax()
        avg_tx = df["amount"].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Spend", f"{total_spent:,.2f}")
        m2.metric("Transactions", f"{tx_count}")
        m3.metric("Average / Tx", f"{avg_tx:,.2f}")
        m4.metric("Top Category", top_cat)
    else:
        st.info("No expenses logged yet. Use the sidebar form or AI Quick-Add above to add your first record.")

    st.divider()

    # --- Workspace Tabs ---
    tab_crud, tab_analytics, tab_ai = st.tabs(
        ["📋 Manage Expenses (Edit / Delete)", "📊 Spending Analytics", "🤖 Gemini AI Financial Advisor"]
    )

    # ==========================================
    # TAB 1: INLINE EDIT, FILTER & DELETE
    # ==========================================
    with tab_crud:
        if not df.empty:
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                selected_cats = st.multiselect(
                    "Filter by Category", options=CATEGORIES, default=CATEGORIES
                )
            with f_col2:
                search_query = st.text_input("Search Description", placeholder="Filter by keyword...")

            filtered_df = df[df["category"].isin(selected_cats)].copy()
            if search_query.strip():
                filtered_df = filtered_df[
                    filtered_df["description"].str.contains(search_query.strip(), case=False, na=False)
                ]

            st.markdown(
                "**Inline Editor:** Double-click any cell to **Edit** its value, or check the `Delete?` box to **Remove** rows, then click **Apply Changes**."
            )

            # Add temporary Delete selection column
            editor_df = filtered_df.copy()
            editor_df["Delete?"] = False

            edited_df = st.data_editor(
                editor_df,
                key="expense_editor",
                use_container_width=True,
                hide_index=True,
                disabled=["id"],
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "expense_date": st.column_config.DateColumn("Date", required=True),
                    "category": st.column_config.SelectboxColumn(
                        "Category", options=CATEGORIES, required=True
                    ),
                    "amount": st.column_config.NumberColumn(
                        "Amount", min_value=0.01, format="%.2f", required=True
                    ),
                    "payment_method": st.column_config.SelectboxColumn(
                        "Payment Method",
                        options=["Cash", "Card", "Bank Transfer", "Mobile Wallet"],
                        required=True,
                    ),
                    "description": st.column_config.TextColumn("Description", required=True),
                    "Delete?": st.column_config.CheckboxColumn("Delete?", default=False),
                },
            )

            b_col1, b_col2 = st.columns([1, 5])
            with b_col1:
                if st.button("💾 Apply Changes", type="primary", use_container_width=True):
                    # 1. Process Deletions
                    ids_to_delete = edited_df.loc[edited_df["Delete?"] == True, "id"].tolist()
                    if ids_to_delete:
                        delete_expenses(ids_to_delete)

                    # 2. Process Edits on surviving rows
                    surviving_df = edited_df[edited_df["Delete?"] == False]
                    for _, row in surviving_df.iterrows():
                        orig_row = filtered_df[filtered_df["id"] == row["id"]].iloc[0]
                        has_changed = (
                            str(row["expense_date"]) != str(orig_row["expense_date"])
                            or row["category"] != orig_row["category"]
                            or float(row["amount"]) != float(orig_row["amount"])
                            or row["payment_method"] != orig_row["payment_method"]
                            or str(row["description"]) != str(orig_row["description"])
                        )
                        if has_changed:
                            update_expense(
                                expense_id=int(row["id"]),
                                expense_date=str(row["expense_date"]),
                                category=row["category"],
                                amount=float(row["amount"]),
                                payment_method=row["payment_method"],
                                description=str(row["description"]),
                            )

                    st.toast("Database updated!", icon="💾")
                    st.rerun()

            with b_col2:
                csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Export Filtered CSV",
                    data=csv_bytes,
                    file_name=f"expenses_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )

    # ==========================================
    # TAB 2: ANALYTICS
    # ==========================================
    with tab_analytics:
        if not df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Spend by Category")
                cat_summary = df.groupby("category", as_index=True)["amount"].sum().sort_values(ascending=False)
                st.bar_chart(cat_summary)
            with c2:
                st.subheader("Daily Spending Trend")
                daily_summary = df.groupby("expense_date", as_index=True)["amount"].sum()
                st.line_chart(daily_summary)

    # ==========================================
    # TAB 3: GEMINI AI FINANCIAL ADVISOR
    # ==========================================
    with tab_ai:
        st.subheader("🤖 Ask Gemini About Your Finances")
        user_prompt = st.text_area(
            "Question or Audit Request",
            value="Analyze my spending habits, identify my top 2 expense categories, and give me 3 actionable ways to reduce my monthly burn.",
            height=100,
        )
        if st.button("Generate AI Audit", type="primary"):
            if not client:
                st.error("Please enter your Gemini API Key in the sidebar to use AI features.")
            else:
                with st.spinner("Gemini is auditing your ledger..."):
                    try:
                        insight = ai_generate_financial_insights(client, df, user_prompt)
                        st.markdown(insight)
                    except Exception as e:
                        st.error(f"Gemini API Error: {e}")


if __name__ == "__main__":
    main()
