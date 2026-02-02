import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sqlite3
import pandas as pd
import os
import re
import threading
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import Optional, Tuple, Dict, Any

# =========================================================================================
# THEME CONFIGURATION (Modern Dark Mode)
# =========================================================================================
THEME = {
    "bg_main": "#1e1e1e",       # VS Code Dark background
    "bg_sidebar": "#252526",    # Sidebar background
    "bg_input": "#3c3c3c",      # Input field background
    "fg_text": "#cccccc",       # Primary text
    "fg_dim": "#858585",        # Secondary text
    "accent": "#007acc",        # Blue accent (VS Code blue)
    "accent_hover": "#005a9e",  # Darker blue
    "success": "#89d185",       # Green
    "error": "#f48771",         # Red
    "warning": "#cca700",       # Yellow
    "bubble_user": "#264f78",   # User message bubbles (Dark Blue)
    "bubble_ai": "#303031",     # AI message bubbles (Dark Gray)
    "font_main": ("Segoe UI", 10),
    "font_mono": ("Consolas", 10),
    "font_heading": ("Segoe UI", 12, "bold")
}

# Load environment variables
load_dotenv()

# =========================================================================================
# BACKEND LOGIC (PRESERVED & UNCHANGED)
# =========================================================================================
class BackendLogic:
    def __init__(self):
        self.db_path = None
        
    def get_llm(self):
        if not os.getenv("GROQ_API_KEY"):
            return None
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=2048
        )
        
    def set_db_path(self, path):
        self.db_path = Path(path)
        
    def get_connection(self):
        if not self.db_path:
            raise ValueError("No database loaded")
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=DELETE;")
        conn.execute("PRAGMA synchronous=FULL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def get_schema(self) -> str:
        if not self.db_path: return ""
        try:
            conn = self.get_connection()
            schema_parts = []
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;").fetchall()
            for (table_name,) in tables:
                cols = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
                col_details = []
                for col in cols:
                    col_id, col_name, col_type, not_null, default_val, pk = col
                    col_info = f"{col_name} {col_type}"
                    if pk: col_info += " PRIMARY KEY"
                    col_details.append(col_info)
                schema_parts.append(f"Table: {table_name}")
                schema_parts.append(f"Columns: {', '.join(col_details)}")
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0]
                    schema_parts.append(f"Rows: {count}")
                except: pass
                schema_parts.append("")
            conn.close()
            return "\n".join(schema_parts)
        except: return ""

    def normalize_where_clause(self, where_clause: str) -> str:
        if not where_clause or not where_clause.strip(): return where_clause
        normalized = where_clause.strip()
        
        def normalize_equality(match):
            column = match.group(1).strip()
            value_part = match.group(2).strip() if match.lastindex >= 2 else ""
            value = value_part
            if value_part:
                if (value_part.startswith("'") and value_part.endswith("'")) or \
                   (value_part.startswith('"') and value_part.endswith('"')):
                    value = value_part[1:-1]
            value_lower = value.lower().strip()
            null_keywords = ['null', 'empty', 'blank', 'no value', 'missing']
            if value == '' or value == "''" or value == '""' or value_lower in null_keywords:
                return f"({column} IS NULL OR TRIM({column}) = '')"
            else:
                escaped_value = value.replace("'", "''")
                return f"LOWER(TRIM({column})) = LOWER(TRIM('{escaped_value}'))"
        
        def normalize_is_null(match):
            column = match.group(1).strip()
            return f"({column} IS NULL OR TRIM({column}) = '')"
        
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
        
        normalized = re.sub(r'(\[[^\]]+\]|[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*((?:["\'](?:[^"\'\\]|\\.)*["\'])|[^\s\)]+)', normalize_equality, normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'(\[[^\]]+\]|[a-zA-Z_][a-zA-Z0-9_]*)\s+IS\s+NULL', normalize_is_null, normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'(\[[^\]]+\]|[a-zA-Z_][a-zA-Z0-9_]*)\s*(!=|<>)\s*((?:["\'](?:[^"\'\\]|\\.)*["\'])|[^\s\)]+)', normalize_inequality, normalized, flags=re.IGNORECASE)
        return normalized

    def extract_and_normalize_where_clause(self, sql: str) -> Tuple[Optional[str], Optional[str]]:
        sql_upper = sql.strip().upper()
        table_name = None
        where_match = None
        
        if sql_upper.startswith("UPDATE"):
            match = re.search(r"UPDATE\s+([^\s\[\]]+|\[[^\]]+\])", sql, re.IGNORECASE)
            if match: table_name = match.group(1).replace("[", "").replace("]", "")
        elif sql_upper.startswith("DELETE"):
            match = re.search(r"DELETE\s+FROM\s+([^\s\[\]]+|\[[^\]]+\])", sql, re.IGNORECASE)
            if match: table_name = match.group(1).replace("[", "").replace("]", "")
        
        where_match = re.search(r"WHERE\s+(.*?)(?:\s+ORDER\s+BY|\s+LIMIT|\s+GROUP\s+BY|$)", sql, re.IGNORECASE | re.DOTALL)
        if not where_match:
            where_match = re.search(r"WHERE\s+(.*)", sql, re.IGNORECASE | re.DOTALL)
        
        where_clause = None
        if where_match:
            where_clause = self.normalize_where_clause(where_match.group(1).strip())
        return table_name, where_clause

    def validate_modification(self, sql: str, conn: sqlite3.Connection) -> Tuple[bool, str]:
        sql_upper = sql.strip().upper()
        if not (sql_upper.startswith("UPDATE") or sql_upper.startswith("DELETE")):
            return True, ""
        try:
            table_name, normalized_where = self.extract_and_normalize_where_clause(sql)
            if not table_name: return True, ""
            check_sql = f"SELECT COUNT(*) FROM [{table_name}]"
            if normalized_where: check_sql += f" WHERE {normalized_where}"
            cursor = conn.cursor()
            count = cursor.execute(check_sql).fetchone()[0]
            if count == 0: return False, "Requested data not found in the database. No changes were made."
        except: pass
        return True, ""

# =========================================================================================
# UI COMPONENTS
# =========================================================================================
class ChatMessage(tk.Frame):
    def __init__(self, parent, role, text, **kwargs):
        super().__init__(parent, bg=THEME["bg_main"], **kwargs)
        self.role = role
        self.text = text
        
        # Container for bubble to allow alignment
        bubble_container = tk.Frame(self, bg=THEME["bg_main"])
        bubble_container.pack(fill=tk.X, padx=10, pady=5)
        
        # Style based on role
        if role == "User":
            bubble_bg = THEME["bubble_user"]
            align = tk.RIGHT
            pad = (50, 0) # Left margin for user
        else:
            bubble_bg = THEME["bubble_ai"]
            align = tk.LEFT
            pad = (0, 50) # Right margin for AI
            
        self.bubble = tk.Label(
            bubble_container, 
            text=text, 
            bg=bubble_bg, 
            fg=THEME["fg_text"],
            font=THEME["font_main"],
            wraplength=600,
            justify=tk.LEFT,
            padx=15, 
            pady=10,
            relief=tk.FLAT
        )
        self.bubble.pack(side=align, padx=pad)

class ResultTable(tk.Frame):
    def __init__(self, parent, dataframe, **kwargs):
        super().__init__(parent, bg=THEME["bg_main"], **kwargs)
        
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure Table Style
        style.configure("Treeview", 
            background=THEME["bg_input"],
            foreground=THEME["fg_text"],
            rowheight=25,
            fieldbackground=THEME["bg_input"],
            bordercolor=THEME["bg_sidebar"],
            borderwidth=0
        )
        style.map("Treeview", background=[("selected", THEME["accent"])])
        
        style.configure("Treeview.Heading",
            background=THEME["bg_sidebar"],
            foreground=THEME["fg_text"],
            relief="flat",
            font=THEME["font_heading"]
        )
        style.map("Treeview.Heading", background=[("active", THEME["accent_hover"])])

        # Scrollbars
        scroll_y = tk.Scrollbar(self, orient="vertical", bg=THEME["bg_sidebar"])
        scroll_x = tk.Scrollbar(self, orient="horizontal", bg=THEME["bg_sidebar"])
        
        self.tree = ttk.Treeview(self, columns=list(dataframe.columns), show="headings", 
                                 yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        scroll_y.config(command=self.tree.yview)
        scroll_x.config(command=self.tree.xview)
        
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Setup Columns
        for col in dataframe.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
            
        # Add Data
        for _, row in dataframe.iterrows():
            self.tree.insert("", tk.END, values=list(row))

# =========================================================================================
# MAIN APP
# =========================================================================================
class ModernSQLChatbot(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("SQL Intelligence - Desktop")
        self.geometry("1100x800")
        self.configure(bg=THEME["bg_main"])
        
        self.backend = BackendLogic()
        self.llm = self.backend.get_llm()
        
        self.setup_ui()
        
        if not self.llm:
            self.add_system_message("⚠️ Groq API key missing. Please check your environment.")

    def setup_ui(self):
        # 1. Sidebar (Left)
        sidebar = tk.Frame(self, bg=THEME["bg_sidebar"], width=250)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False) # Fixed width
        
        # App Title in Sidebar
        tk.Label(sidebar, text="SQL Intelligence", bg=THEME["bg_sidebar"], fg=THEME["fg_text"], 
                 font=("Segoe UI", 16, "bold"), pady=20).pack()
        
        # Database Section
        db_frame = tk.LabelFrame(sidebar, text="DATABASE", bg=THEME["bg_sidebar"], fg=THEME["fg_dim"], 
                                 font=("Segoe UI", 8, "bold"), relief=tk.FLAT, padx=10, pady=10)
        db_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.lbl_db_status = tk.Label(db_frame, text="No DB Selected", bg=THEME["bg_sidebar"], fg=THEME["error"], anchor="w")
        self.lbl_db_status.pack(fill=tk.X, pady=(0, 5))
        
        btn_browse = tk.Button(db_frame, text="📂 Open Database", bg=THEME["accent"], fg="white", 
                               font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=10, pady=5, cursor="hand2",
                               command=self.select_db)
        btn_browse.pack(fill=tk.X)
        
        # Actions Info
        tk.Label(sidebar, text="• Secure Local Mode\n• Changes Persisted\n• Strict Validation", 
                 bg=THEME["bg_sidebar"], fg=THEME["fg_dim"], justify=tk.LEFT).pack(side=tk.BOTTOM, pady=20)

        # 2. Main Content (Right)
        main_content = tk.Frame(self, bg=THEME["bg_main"])
        main_content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Chat History container with Scrollbar
        self.canvas = tk.Canvas(main_content, bg=THEME["bg_main"], highlightthickness=0)
        self.scrollbar = tk.Scrollbar(main_content, orient="vertical", command=self.canvas.yview, bg=THEME["bg_main"])
        self.scrollable_frame = tk.Frame(self.canvas, bg=THEME["bg_main"])
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=800) # Initial width attempt
        
        # Dynamic resizing of canvas content
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas.create_window((0,0), window=self.scrollable_frame, anchor="nw"), width=e.width))
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 3. Input Area (Bottom)
        input_container = tk.Frame(main_content, bg=THEME["bg_input"], pady=15, padx=20)
        input_container.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.entry_query = tk.Entry(input_container, bg="#4a4a4a", fg="white", 
                                    font=("Segoe UI", 11), insertbackground="white", relief=tk.FLAT)
        self.entry_query.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        self.entry_query.bind("<Return>", lambda e: self.on_send())
        self.entry_query.focus()
        
        btn_send = tk.Button(input_container, text="➤", bg=THEME["accent"], fg="white", 
                             font=("Segoe UI", 12, "bold"), relief=tk.FLAT, width=4, cursor="hand2",
                             command=self.on_send)
        btn_send.pack(side=tk.RIGHT)
        
        self.add_system_message("Welcome to SQL Intelligence. Please select a database to begin.")

    def select_db(self):
        path = filedialog.askopenfilename(filetypes=[("SQLite Files", "*.db *.sqlite")])
        if path:
            self.backend.set_db_path(path)
            self.lbl_db_status.config(text=f"✔ {Path(path).name}", fg=THEME["success"])
            self.add_system_message(f"Database loaded successfully: {Path(path).name}")

    def add_user_message(self, text):
        msg = ChatMessage(self.scrollable_frame, "User", text)
        msg.pack(fill=tk.X, pady=5, padx=10)
        self.scroll_to_bottom()

    def add_ai_message(self, text):
        msg = ChatMessage(self.scrollable_frame, "AI", text)
        msg.pack(fill=tk.X, pady=5, padx=10)
        self.scroll_to_bottom()
        
    def add_system_message(self, text):
        lbl = tk.Label(self.scrollable_frame, text=text, bg=THEME["bg_main"], fg=THEME["fg_dim"], font=("Segoe UI", 9, "italic"))
        lbl.pack(pady=10)
        self.scroll_to_bottom()

    def add_table_result(self, df):
        if df.empty:
            self.add_system_message("Query executed successfully. No rows returned.")
            return
        
        # Container for table
        container = tk.Frame(self.scrollable_frame, bg=THEME["bg_main"], pady=10)
        container.pack(fill=tk.X, padx=20)
        
        lbl = tk.Label(container, text=f"Result ({len(df)} rows):", bg=THEME["bg_main"], fg=THEME["success"], font=("Segoe UI", 9, "bold"), anchor="w")
        lbl.pack(fill=tk.X)
        
        # Table
        table = ResultTable(container, df, height=200) # limitation: fixed height for now
        table.pack(fill=tk.X, pady=5)
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def on_send(self):
        query = self.entry_query.get().strip()
        if not query: return
        
        self.entry_query.delete(0, tk.END)
        self.add_user_message(query)
        
        # Run in thread to prevent UI freezing
        threading.Thread(target=self.process_query_thread, args=(query,), daemon=True).start()

    def process_query_thread(self, query):
        if not self.backend.db_path:
            # Classification check for 'create database' first
            q_lower = query.lower()
            if any(k in q_lower for k in ["create", "database"]) and not self.backend.db_path:
                 self.after(0, self.add_ai_message, "⚠️ Please create a new empty text file, name it 'mydb.db', and Open it using the sidebar first.")
                 return
            self.after(0, self.add_system_message, "⚠️ No database selected. Use the sidebar to open one.")
            return

        # Classification
        q_lower = query.lower()
        is_sql = any(k in q_lower for k in ["select", "insert", "update", "delete", "create", "drop", "table", "show"])
        
        if is_sql:
            try:
                schema = self.backend.get_schema()
                prompt = f"You are a SQLite SQL compiler. Schema:\n{schema}\nUser Request: {query}\nOutput ONLY raw SQLite SQL."
                response = self.llm.invoke(prompt)
                sql = response.content.strip().replace("```sql", "").replace("```", "").strip()
                
                # Validation
                conn = self.backend.get_connection()
                try:
                    is_valid, msg = self.backend.validate_modification(sql, conn)
                    if not is_valid:
                        self.after(0, self.add_ai_message, f"❌ {msg}")
                        return
                    
                    # Confirmation
                    sql_upper = sql.upper()
                    if sql_upper.startswith(("DELETE", "DROP", "UPDATE")):
                        # Must run confirmation on main thread
                        self.after(0, self.ask_confirmation, sql, conn)
                        return # Execution continues in callback
                    
                    # Immediate Execution
                    self.execute_and_show(sql, conn)
                    
                except Exception as e:
                    self.after(0, self.add_ai_message, f"runtime Error: {e}")
                finally:
                    conn.close()
            except Exception as e:
                 self.after(0, self.add_ai_message, f"Error: {e}")
        else:
            # General Chat
            try:
                msg = self.llm.invoke(f"User Request: {query}\nAnswer helpfuly.").content
                self.after(0, self.add_ai_message, msg)
            except Exception as e:
                self.after(0, self.add_ai_message, f"Error: {e}")

    def ask_confirmation(self, sql, conn):
        # Re-open conn because thread closed it? Actually we passed conn but it's risky across threads if closed.
        # Let's open fresh conn
        conn = self.backend.get_connection() 
        confirm = messagebox.askyesno("Confirm SQL Action", f"Execute this SQL?\n\n{sql}")
        if confirm:
            self.execute_and_show(sql, conn)
        else:
            self.add_system_message("Action cancelled by user.")
        conn.close()

    def execute_and_show(self, sql, conn):
        try:
            cursor = conn.cursor()
            
            # Normalization execution Logic
            sql_upper = sql.strip().upper()
            if sql_upper.startswith("UPDATE") or sql_upper.startswith("DELETE"):
                table_name, normalized_where = self.backend.extract_and_normalize_where_clause(sql)
                if table_name and normalized_where:
                   if sql_upper.startswith("DELETE"):
                       sql = f"DELETE FROM [{table_name}] WHERE {normalized_where}"
                   elif sql_upper.startswith("UPDATE"):
                       set_match = re.search(r"SET\s+(.*?)\s+WHERE", sql, re.IGNORECASE | re.DOTALL)
                       if set_match:
                           set_clause = set_match.group(1).strip().rstrip(';')
                           sql = f"UPDATE [{table_name}] SET {set_clause} WHERE {normalized_where}"

            cursor.execute(sql)
            
            if sql.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                if cursor.description:
                    cols = [d[0] for d in cursor.description]
                    df = pd.DataFrame(rows, columns=cols)
                    self.after(0, self.add_table_result, df)
                else:
                    self.after(0, self.add_ai_message, "Done.")
            else:
                conn.commit()
                msg = f"✅ Success.\n\n• Action: {sql.split()[0]}\n• Rows Affected: {cursor.rowcount}"
                self.after(0, self.add_ai_message, msg)
        except Exception as e:
            self.after(0, self.add_ai_message, f"SQL Error: {e}")

if __name__ == "__main__":
    app = ModernSQLChatbot()
    app.mainloop()
