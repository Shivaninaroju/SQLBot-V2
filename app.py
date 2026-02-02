import streamlit as st
import sqlite3
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from typing import Optional, Tuple, Dict, Any

from langchain_groq import ChatGroq

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="NL → SQL Chatbot",
    page_icon="🧠",
    layout="wide"
)

load_dotenv()

# ==================================================
# INITIALIZE SESSION STATE
# ==================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "db_path" not in st.session_state:
    st.session_state.db_path = None

if "pending_delete_confirmation" not in st.session_state:
    st.session_state.pending_delete_confirmation = None

if "delete_confirmation_query" not in st.session_state:
    st.session_state.delete_confirmation_query = None

# ==================================================
# SIDEBAR - CONFIGURATION
# ==================================================
st.sidebar.header("🔑 Configuration")

# API Key is managed via environment variables only
if not os.getenv("GROQ_API_KEY"):
    st.sidebar.warning("⚠️ GROQ_API_KEY not found in environment.")

uploaded_db = st.sidebar.file_uploader(
    "Upload SQLite Database (.db)",
    type=["db", "sqlite", "sqlite3"],
    help="Upload a SQLite database file to query"
)

# ==================================================
# DATABASE LOADING (PERSISTENT + SAFE)
# ==================================================
def load_database(uploaded_file) -> Path:
    """Load database file once per session."""
    if st.session_state.db_path is None or not st.session_state.db_path.exists():
        db_path = Path("user_uploaded.db")
        with open(db_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.db_path = db_path
    return st.session_state.db_path

def get_connection() -> sqlite3.Connection:
    """Get a new database connection with WAL mode enabled."""
    if st.session_state.db_path is None:
        raise ValueError("No database loaded")
    
    conn = sqlite3.connect(
        str(st.session_state.db_path),
        check_same_thread=False,
        timeout=30.0
    )
    conn.execute("PRAGMA journal_mode=DELETE;")
    conn.execute("PRAGMA synchronous=FULL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def close_connection(conn: sqlite3.Connection):
    """Safely close database connection."""
    try:
        conn.close()
    except:
        pass

# Load database if uploaded
db_loaded = False
if uploaded_db:
    try:
        load_database(uploaded_db)
        db_loaded = True
    except Exception as e:
        st.sidebar.error(f"Error loading database: {e}")

if st.session_state.db_path and st.session_state.db_path.exists():
    db_loaded = True

# ==================================================
# SCHEMA EXTRACTION (ENHANCED)
# ==================================================
def get_schema() -> str:
    """Extract comprehensive database schema."""
    if not db_loaded:
        return ""
    
    conn = get_connection()
    try:
        schema_parts = []
        
        # Get all tables
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        ).fetchall()
        
        for (table_name,) in tables:
            # Get column information
            cols = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
            
            col_details = []
            for col in cols:
                col_id, col_name, col_type, not_null, default_val, pk = col
                col_info = f"{col_name} {col_type}"
                if pk:
                    col_info += " PRIMARY KEY"
                if not_null and not pk:
                    col_info += " NOT NULL"
                if default_val:
                    col_info += f" DEFAULT {default_val}"
                col_details.append(col_info)
            
            schema_parts.append(f"Table: {table_name}")
            schema_parts.append(f"Columns: {', '.join(col_details)}")
            
            # Get sample data count
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0]
                schema_parts.append(f"Rows: {count}")
            except:
                pass
            
            schema_parts.append("")
        
        return "\n".join(schema_parts)
    finally:
        close_connection(conn)

def get_table_preview(table_name: str, limit: int = 5) -> Optional[pd.DataFrame]:
    """Get preview of table data."""
    if not db_loaded:
        return None
    
    conn = get_connection()
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit};", conn)
        return df
    except Exception as e:
        return None
    finally:
        close_connection(conn)

# ==================================================
# LLM INITIALIZATION
# ==================================================
@st.cache_resource
def get_llm():
    """Initialize LLM with caching."""
    if not os.getenv("GROQ_API_KEY"):
        return None
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=2048
    )

llm = get_llm()

# ==================================================
# QUERY CLASSIFIER (LLM-BASED)
# ==================================================
def classify_query(query: str, db_loaded: bool) -> str:
    """
    Classify query into SQL_READ, SQL_WRITE, or NON_SQL.
    Uses LLM for intelligent classification.
    """
    if not db_loaded:
        # Allow CREATE/DATABASE commands even if no DB is loaded to support new DB creation
        q_lower = query.lower().strip()
        if any(kw in q_lower for kw in ["create", "database", "table"]):
             return "SQL_WRITE"
        
        # If no database, only allow non-SQL queries
        sql_keywords = ["select", "insert", "update", "delete", "drop", 
                       "query", "show", "list", "count", 
                       "add", "remove", "modify", "alter"]
        if any(kw in q_lower for kw in sql_keywords):
            return "NON_SQL"  # Treat as general question about SQL
        return "NON_SQL"
    
    if not llm:
        # Fallback to keyword-based classification
        q_lower = query.lower().strip()
        
        write_keywords = ["insert", "add", "create", "update", "modify", "change", 
                         "set", "delete", "remove", "drop", "alter"]
        read_keywords = ["select", "show", "list", "get", "find", "count", 
                        "how many", "what", "who", "which", "display", "fetch"]
        
        if any(kw in q_lower for kw in write_keywords):
            return "SQL_WRITE"
        if any(kw in q_lower for kw in read_keywords):
            return "SQL_READ"
        return "NON_SQL"
    
    # Use LLM for intelligent classification
    classification_prompt = f"""Classify the following user query into exactly one of these categories:
- SQL_READ: Query that reads/retrieves data from database (SELECT, COUNT, SHOW, LIST, etc.)
- SQL_WRITE: Query that modifies database (INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, etc.)
- NON_SQL: General conversation, greetings, help requests, explanations, or questions not related to database operations

User query: "{query}"

Respond with ONLY one word: SQL_READ, SQL_WRITE, or NON_SQL. No explanations."""
    
    try:
        response = llm.invoke(classification_prompt)
        classification = response.content.strip().upper()
        
        if "SQL_READ" in classification:
            return "SQL_READ"
        elif "SQL_WRITE" in classification:
            return "SQL_WRITE"
        else:
            return "NON_SQL"
    except:
        # Fallback
        q_lower = query.lower().strip()
        if any(kw in q_lower for kw in ["insert", "add", "create", "update", "delete", "drop"]):
            return "SQL_WRITE"
        if any(kw in q_lower for kw in ["select", "show", "list", "count", "get", "find"]):
            return "SQL_READ"
        return "NON_SQL"

# ==================================================
# SQL GENERATION (STRICT COMPILER MODE)
# ==================================================
def generate_sql(user_query: str, schema: str) -> str:
    """Generate SQL using strict compiler-mode LLM prompt."""
    if not llm:
        raise ValueError("LLM not initialized. Please provide Groq API key.")
    
    prompt = f"""You are a SQLite SQL compiler. Your ONLY job is to output valid SQLite SQL code.

Database Schema:
{schema}

CRITICAL RULES:
1. Output ONLY raw SQLite SQL code
2. NO explanations before or after
3. NO markdown formatting (no ```sql blocks)
4. NO comments
5. NO Python code
6. NO refusal messages
7. Use ONLY tables and columns that exist in the schema above
8. If the query is ambiguous, make reasonable assumptions based on schema

User Request: {user_query}

Output ONLY the SQL statement:"""

    try:
        response = llm.invoke(prompt)
        sql = response.content.strip()
        
        # Clean up markdown if present
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        sql = sql.strip()
        
        # Remove leading/trailing quotes if present
        if sql.startswith('"') and sql.endswith('"'):
            sql = sql[1:-1]
        if sql.startswith("'") and sql.endswith("'"):
            sql = sql[1:-1]
        
        # Validate SQL starts with a valid keyword
        sql_upper = sql.upper().strip()
        valid_starts = ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER")
        if not any(sql_upper.startswith(start) for start in valid_starts):
            raise ValueError(f"Generated SQL does not start with a valid keyword. Got: {sql[:50]}")
        
        return sql
    except Exception as e:
        raise ValueError(f"SQL generation failed: {str(e)}")

# ==================================================
# WHERE CLAUSE NORMALIZATION (CRITICAL FOR DATA MATCHING)
# ==================================================
def normalize_where_clause(where_clause: str) -> str:
    """
    Normalize WHERE clause for case-insensitive matching and NULL/empty/whitespace handling.
    
    Rules:
    1. String equality: LOWER(TRIM(column)) = LOWER(TRIM(value))
    2. NULL/empty/whitespace: (column IS NULL OR TRIM(column) = '')
    3. Preserves AND/OR logic
    
    Returns normalized WHERE clause that can be reused for validation and execution.
    """
    if not where_clause or not where_clause.strip():
        return where_clause
    
    normalized = where_clause.strip()
    
    # Pattern 1: Handle string equality comparisons (column = 'value' or column = "value")
    def normalize_equality(match):
        full_match = match.group(0)
        column = match.group(1).strip()
        value_part = match.group(2).strip() if match.lastindex >= 2 else ""
        
        # Handle quoted values
        value = value_part
        if value_part:
            if (value_part.startswith("'") and value_part.endswith("'")) or \
               (value_part.startswith('"') and value_part.endswith('"')):
                value = value_part[1:-1]
        
        # Check for NULL/empty/whitespace keywords
        value_lower = value.lower().strip()
        null_keywords = ['null', 'empty', 'blank', 'no value', 'missing']
        
        # Handle empty string literals
        if value == '' or value == "''" or value == '""' or value_lower in null_keywords:
            # Handle as NULL or empty
            return f"({column} IS NULL OR TRIM({column}) = '')"
        else:
            # Case-insensitive string comparison - escape single quotes in value
            escaped_value = value.replace("'", "''")
            return f"LOWER(TRIM({column})) = LOWER(TRIM('{escaped_value}'))"
    
    # Pattern 2: Handle IS NULL checks
    def normalize_is_null(match):
        column = match.group(1).strip()
        return f"({column} IS NULL OR TRIM({column}) = '')"
    
    # Pattern 3: Handle != or <> comparisons
    def normalize_inequality(match):
        column = match.group(1).strip()
        value_part = match.group(3).strip() if match.lastindex >= 3 else ""
        
        value = value_part
        if value_part:
            if (value_part.startswith("'") and value_part.endswith("'")) or \
               (value_part.startswith('"') and value_part.endswith('"')):
                value = value_part[1:-1]
        
        value_lower = value.lower().strip()
        null_keywords = ['null', 'empty', 'blank', 'no value', 'missing']
        
        if value == '' or value == "''" or value == '""' or value_lower in null_keywords:
            return f"({column} IS NOT NULL AND TRIM({column}) != '')"
        else:
            escaped_value = value.replace("'", "''")
            return f"LOWER(TRIM({column})) != LOWER(TRIM('{escaped_value}'))"
    
    # Apply normalization patterns
    # Pattern: column = 'value' or column = "value" (more robust regex)
    # Match column names (including bracketed) followed by = and quoted/unquoted values
    normalized = re.sub(
        r'(\[[^\]]+\]|[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*((?:["\'](?:[^"\'\\]|\\.)*["\'])|[^\s\)]+)',
        normalize_equality,
        normalized,
        flags=re.IGNORECASE
    )
    
    # Pattern: column IS NULL
    normalized = re.sub(
        r'(\[[^\]]+\]|[a-zA-Z_][a-zA-Z0-9_]*)\s+IS\s+NULL',
        normalize_is_null,
        normalized,
        flags=re.IGNORECASE
    )
    
    # Pattern: column != 'value' or column <> 'value'
    normalized = re.sub(
        r'(\[[^\]]+\]|[a-zA-Z_][a-zA-Z0-9_]*)\s*(!=|<>)\s*((?:["\'](?:[^"\'\\]|\\.)*["\'])|[^\s\)]+)',
        normalize_inequality,
        normalized,
        flags=re.IGNORECASE
    )
    
    return normalized

def extract_and_normalize_where_clause(sql: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract table name and WHERE clause from SQL, then normalize WHERE clause.
    Returns (table_name, normalized_where_clause).
    """
    sql_upper = sql.strip().upper()
    table_name = None
    where_clause = None
    
    # Extract table name
    if sql_upper.startswith("UPDATE"):
        match = re.search(r"UPDATE\s+([^\s\[\]]+|\[[^\]]+\])", sql, re.IGNORECASE)
        if match:
            table_name = match.group(1).replace("[", "").replace("]", "")
    elif sql_upper.startswith("DELETE"):
        match = re.search(r"DELETE\s+FROM\s+([^\s\[\]]+|\[[^\]]+\])", sql, re.IGNORECASE)
        if match:
            table_name = match.group(1).replace("[", "").replace("]", "")
    
    # Extract WHERE clause
    where_match = re.search(r"WHERE\s+(.*?)(?:\s+ORDER\s+BY|\s+LIMIT|\s+GROUP\s+BY|$)", sql, re.IGNORECASE | re.DOTALL)
    if not where_match:
        where_match = re.search(r"WHERE\s+(.*)", sql, re.IGNORECASE | re.DOTALL)
    
    if where_match:
        where_clause = where_match.group(1).strip()
        # Normalize the WHERE clause
        where_clause = normalize_where_clause(where_clause)
    
    return table_name, where_clause

# ==================================================
# DATA VALIDATION
# ==================================================
def validate_modification(sql: str, conn: sqlite3.Connection) -> Tuple[bool, str]:
    """
    Validate UPDATE/DELETE operations by checking if data exists.
    Uses normalized WHERE clause for accurate matching.
    Returns (is_valid, message).
    """
    sql_upper = sql.strip().upper()
    
    if not (sql_upper.startswith("UPDATE") or sql_upper.startswith("DELETE")):
        return True, ""
        
    try:
        # Extract and normalize WHERE clause
        table_name, normalized_where = extract_and_normalize_where_clause(sql)
        
        if not table_name:
            return True, ""  # Can't validate without table name
        
        # Use normalized WHERE clause for validation
        check_sql = f"SELECT COUNT(*) FROM [{table_name}]"
        if normalized_where:
            check_sql += f" WHERE {normalized_where}"
        
        cursor = conn.cursor()
        count = cursor.execute(check_sql).fetchone()[0]
        
        if count == 0:
            return False, "Requested data not found in the database. No changes were made."
                
    except Exception as e:
        # Fallback to True to avoid blocking valid queries due to parsing errors
        # Log the error for debugging but don't block execution
        pass

    return True, ""

# ==================================================
# SQL EXECUTION
# ==================================================
def execute_sql(sql: str) -> Tuple[bool, Any, Optional[pd.DataFrame]]:
    """
    Execute SQL and return (success, message, dataframe).
    For SELECT queries, returns dataframe. For others, returns success message.
    CRITICAL: For UPDATE/DELETE, uses normalized WHERE clause identical to validation.
    """
    # Auto-initialize database if not loaded (for CREATE operations)
    if not st.session_state.db_path:
        st.session_state.db_path = Path("user_uploaded.db")
        # Ensure file exists
        if not st.session_state.db_path.exists():
            with sqlite3.connect(st.session_state.db_path) as temp_conn:
                pass # Create empty file

    conn = get_connection()
    try:
        sql_upper = sql.strip().upper()
        cursor = conn.cursor()
        
        # For UPDATE/DELETE, normalize WHERE clause to match validation
        if sql_upper.startswith("UPDATE") or sql_upper.startswith("DELETE"):
            table_name, normalized_where = extract_and_normalize_where_clause(sql)
            
            if table_name and normalized_where:
                # Rebuild SQL with normalized WHERE clause
                if sql_upper.startswith("UPDATE"):
                    # Extract SET clause - more robust pattern
                    # Match everything between SET and WHERE
                    set_match = re.search(r"SET\s+(.*?)\s+WHERE", sql, re.IGNORECASE | re.DOTALL)
                    if set_match:
                        set_clause = set_match.group(1).strip()
                        # Remove trailing semicolon if present
                        set_clause = set_clause.rstrip(';')
                        normalized_sql = f"UPDATE [{table_name}] SET {set_clause} WHERE {normalized_where}"
                        cursor.execute(normalized_sql)
                    else:
                        # Fallback to original SQL if SET extraction fails
                        cursor.execute(sql)
                elif sql_upper.startswith("DELETE"):
                    normalized_sql = f"DELETE FROM [{table_name}] WHERE {normalized_where}"
                    cursor.execute(normalized_sql)
                else:
                    cursor.execute(sql)
            elif table_name and not normalized_where:
                # UPDATE/DELETE without WHERE clause - execute as-is (rare but possible)
                cursor.execute(sql)
            else:
                # Couldn't extract table name - execute as-is
                cursor.execute(sql)
        else:
            # For SELECT, INSERT, CREATE, etc., execute as-is
            cursor.execute(sql)
        
        if sql_upper.startswith("SELECT"):
            # Fetch results
            rows = cursor.fetchall()
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(rows, columns=columns)
                return True, f"Query returned {len(df)} rows", df
            else:
                return True, "Query executed successfully", None
        else:
            # Write operation - commit
            conn.commit()
            rows_affected = cursor.rowcount
            return True, f"✅ Operation completed successfully. Rows affected: {rows_affected}", None
    
    except sqlite3.Error as e:
        return False, f"SQL Error: {str(e)}", None
    except Exception as e:
        return False, f"Error: {str(e)}", None
    finally:
        close_connection(conn)

# ==================================================
# DELETE/DROP SAFETY CHECK
# ==================================================
def requires_confirmation(sql: str) -> bool:
    """Check if SQL requires confirmation."""
    sql_upper = sql.strip().upper()
    return sql_upper.startswith(("DELETE", "DROP", "UPDATE"))

# ==================================================
# NON-SQL QUERY HANDLING
# ==================================================
def handle_non_sql_query(query: str) -> str:
    """Handle non-SQL queries with normal chatbot behavior."""
    if not llm:
        return "I'm a SQL chatbot assistant. Please provide a Groq API key to enable full functionality."
    
    # Context-aware prompt
    context = ""
    if db_loaded:
        schema = get_schema()
        context = f"\n\nNote: A database is currently loaded with the following schema:\n{schema}\nYou can help the user understand the database structure, but do NOT generate SQL unless explicitly asked."
    
    prompt = f"""You are a helpful AI assistant for a SQL database chatbot application. 
Answer the user's question in a friendly and informative way.

{context}

User question: {query}

Provide a helpful response:"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again."

# ==================================================
# SIDEBAR - DATABASE INFO
# ==================================================
st.sidebar.markdown("---")

if db_loaded:
    st.sidebar.success("✅ Database Loaded")
    
    try:
        conn = get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        ).fetchall()
        close_connection(conn)
        
        table_names = [t[0] for t in tables]
        st.sidebar.write(f"**📂 Tables ({len(table_names)}):**")
        
        # Table selector for preview
        selected_table = st.sidebar.selectbox(
            "Preview Table:",
            ["None"] + table_names,
            help="Select a table to preview its structure and sample data"
        )
        
        if selected_table and selected_table != "None":
            preview_df = get_table_preview(selected_table)
            if preview_df is not None:
                st.sidebar.dataframe(preview_df, width="stretch", height=200)
        
        # Download button
        st.sidebar.markdown("---")
        if st.session_state.db_path and st.session_state.db_path.exists():
            with open(st.session_state.db_path, "rb") as f:
                st.sidebar.download_button(
                    label="⬇️ Download Updated Database",
                    data=f.read(),
                    file_name="updated_database.db",
                    mime="application/octet-stream",
                    help="Download the current state of the database with all modifications"
                )
    except Exception as e:
        st.sidebar.error(f"Error accessing database: {e}")
else:
    st.sidebar.info("👈 Upload a SQLite database to begin")

# ==================================================
# MAIN UI
# ==================================================
st.title("🧠 Natural Language → SQL Chatbot")
st.markdown("**Ask questions in natural language. The system will intelligently convert database queries to SQL or answer general questions.**")

# Display chat history
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "sql:" in message["content"].lower():
            # Display SQL in code block
            parts = message["content"].split("\n", 1)
            if len(parts) > 1:
                st.code(parts[1], language="sql")
                if "Results:" in message.get("results", ""):
                    st.write(message["results"])
        else:
            st.write(message["content"])
            if "dataframe" in message:
                st.dataframe(message["dataframe"], width="stretch")

# Handle delete confirmation
if st.session_state.pending_delete_confirmation:
    st.warning("⚠️ **Dangerous Operation Detected**")
    st.write(f"**Query:** {st.session_state.delete_confirmation_query}")
    st.write(f"**Generated SQL:**")
    st.code(st.session_state.pending_delete_confirmation, language="sql")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ YES, Execute", type="primary"):
            # Execute the SQL
            sql_to_execute = st.session_state.pending_delete_confirmation
            success, msg, df = execute_sql(sql_to_execute)
            
            # Add to history
            history_entry = {
                "role": "assistant",
                "content": f"SQL:\n{sql_to_execute}",
                "results": msg
            }
            st.session_state.history.append(history_entry)
            
            # Clear confirmation state
            st.session_state.pending_delete_confirmation = None
            st.session_state.delete_confirmation_query = None
            st.rerun()
    
    with col2:
        if st.button("❌ Cancel"):
            st.session_state.history.append({
                "role": "assistant",
                "content": "❌ Operation cancelled by user."
            })
            st.session_state.pending_delete_confirmation = None
            st.session_state.delete_confirmation_query = None
            st.rerun()

# User input
user_query = st.chat_input("Ask in natural language...")

if user_query:
    # Add user message to history
    st.session_state.history.append({
        "role": "user",
        "content": user_query
    })
    
    with st.chat_message("user"):
        st.write(user_query)
    
    # Check for API key
    if not llm:
        with st.chat_message("assistant"):
            error_msg = "⚠️ Please provide a Groq API key in the sidebar to use this application."
            st.error(error_msg)
            st.session_state.history.append({
                "role": "assistant",
                "content": error_msg
            })
    else:
        # Classify query
        query_type = classify_query(user_query, db_loaded)
        
        with st.chat_message("assistant"):
            if query_type == "NON_SQL":
                # Handle as normal chatbot
                response = handle_non_sql_query(user_query)
                st.write(response)
                st.session_state.history.append({
                    "role": "assistant",
                    "content": response
                })
            
            elif query_type in ["SQL_READ", "SQL_WRITE"]:
                    try:
                        # Get schema and generate SQL
                        schema = get_schema()
                        sql = generate_sql(user_query, schema)
                        
                        # MANDATORY: Validation BEFORE confirmation
                        if any(sql.upper().startswith(kw) for kw in ["UPDATE", "DELETE"]):
                            conn = get_connection()
                            try:
                                is_valid, v_msg = validate_modification(sql, conn)
                                if not is_valid:
                                    st.error(v_msg)
                                    st.session_state.history.append({
                                        "role": "assistant",
                                        "content": v_msg
                                    })
                                    # Skip the rest to avoid showing SQL or confirmation
                                    st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
                            finally:
                                close_connection(conn)
                        
                        # If validation passed or wasn't needed, check if confirmation needed
                        if requires_confirmation(sql):
                            st.session_state.pending_delete_confirmation = sql
                            st.session_state.delete_confirmation_query = user_query
                            st.warning("⚠️ This operation will modify or delete data. Please confirm below.")
                            st.rerun()
                        
                        # Display SQL (only if no confirmation was needed)
                        st.code(sql, language="sql")
                        
                        # Execute SQL
                        success, msg, df = execute_sql(sql)
                        
                        if success:
                            if df is not None:
                                # Display results
                                st.dataframe(df, width="stretch")
                                st.session_state.history.append({
                                    "role": "assistant",
                                    "content": f"SQL:\n{sql}",
                                    "results": f"Results:\n{msg}",
                                    "dataframe": df
                                })
                            else:
                                st.success(msg)
                                st.session_state.history.append({
                                    "role": "assistant",
                                    "content": f"SQL:\n{sql}",
                                    "results": msg
                                })
                                st.rerun()
                        else:
                            st.error(msg)
                            st.session_state.history.append({
                                "role": "assistant",
                                "content": f"SQL:\n{sql}\n\nError: {msg}"
                            })
                    
                    except ValueError as e:
                        error_msg = f"❌ {str(e)}"
                        st.error(error_msg)
                        st.session_state.history.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                    except Exception as e:
                        error_msg = f"❌ Unexpected error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.history.append({
                            "role": "assistant",
                            "content": error_msg
                        })

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption("⚡ Professional Natural Language → SQL System | Intelligent Query Classification | Persistent Database Changes | Download Enabled")

