
import os
import re
import time

from google import genai
from google.genai import errors

from sqlite_reader import SQLiteReader
from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# MODELS_TO_TRY = [
#     # "gemini-3.8-flash",
#     # "gemini-3.7-flash",
#     "gemini-3.6-flash"
# ]


MODELS_TO_TRY = [
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
    "gemini-4-argon",
]


MAX_ROWS = 100


# --------------------------------------------------
# GEMINI CONNECTION WITH RETRIES AND FALLBACKS
# --------------------------------------------------

def call_gemini(client, prompt):
    """Try Gemini models in order, retrying temporary server errors."""

    last_error = None

    for model in MODELS_TO_TRY:
        print(f"\nTrying Gemini model: {model}")

        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )

                if not response.text:
                    raise RuntimeError(
                        f"Gemini model {model} returned an empty response."
                    )

                print(f"Using model: {model}")
                return response.text

            except errors.APIError as exc:
                last_error = exc
                code = getattr(exc, "code", None)

                # print(
                #     f"Gemini API error (model={model}, "
                #     f"attempt={attempt + 1}, code={code}): {exc}"
                # )

                # Retry temporary errors once. For other errors,
                # move on to the next model.
                if code in (429, 500, 502, 503, 504) and attempt == 0:
                    time.sleep(2)
                    continue

                break

            except Exception as exc:
                last_error = exc
                print(f"Unexpected Gemini error: {exc}")
                break

    raise RuntimeError(
        "All configured Gemini models failed. "
        "Check your API key, model access, quota, and internet connection. "
        f"Last error: {last_error}"
    )


# --------------------------------------------------
# DATABASE SCHEMA
# --------------------------------------------------

def build_schema_description(reader):
    """Read the database schema so Gemini knows which tables exist."""

    schema = reader.get_schema()

    if isinstance(schema, str):
        return schema

    return repr(schema)


# --------------------------------------------------
# GENERATE SQL
# --------------------------------------------------

def generate_sql(client, reader, question):
    """Ask Gemini to translate a natural-language question into SQL."""

    schema_description = build_schema_description(reader)

    prompt = f"""
You are a SQLite query generator for a Warhammer 40,000 unit database.

Convert the user's question into ONE SQLite SELECT query.

DATABASE SCHEMA:
{schema_description}

RULES:
1. Use only tables and columns present in the schema.
2. Generate only a SELECT statement or a WITH ... SELECT statement.
3. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
   ATTACH, DETACH, PRAGMA, or any other modifying statement.
4. Do not generate multiple SQL statements.
5. Use appropriate JOINs when information is stored across tables.
6. For unit names, use case-insensitive matching where appropriate.
7. Include faction information when the user asks for it.
8. Return only the SQL query, with no Markdown fences or explanation.
9. Limit results to at most {MAX_ROWS} rows.
10. Do not assume a faction ID or unit ID; retrieve it from the database.
11. When filtering or displaying costs, exclude empty values using
    NULLIF(TRIM(cost), '') IS NOT NULL. 
12. When comparing costs numerically, use CAST(NULLIF(TRIM(cost), '')
    AS INTEGER), and filter out empty values first.
13. Select only the columns needed to answer the user's question.
    Never use SELECT * unless the user explicitly asks for all fields.
14. For questions asking to find or list units by name, return the
    unit name and, when useful, its ID and faction ID.
15. Use clear column aliases, such as "unit_name", "unit_id",
    and "faction_id".
16. When the user asks for a list of names, do not return long
    descriptions, loadouts, or URLs unless requested.

USER QUESTION:
{question}

SQL QUERY:
"""

    raw_sql = call_gemini(client, prompt).strip()

    # Remove Markdown code fences if the model adds them.
    raw_sql = re.sub(r"^```(?:sql)?\s*", "", raw_sql, flags=re.IGNORECASE)
    raw_sql = re.sub(r"\s*```$", "", raw_sql).strip()

    # Remove trailing semicolons because the reader wraps the query.
    raw_sql = raw_sql.rstrip(";").strip()

    if not raw_sql:
        raise ValueError("Gemini returned an empty SQL query.")

    print("\nGenerated SQL:")
    print(raw_sql)

    return raw_sql


# --------------------------------------------------
# BASIC SQL SAFETY CHECK
# --------------------------------------------------

def validate_sql(sql):
    """
    Basic guard before execution.

    The SQLiteReader's read-only executor must also enforce
    read-only access. This check is an additional safeguard,
    not a complete SQL security parser.
    """

    cleaned = sql.strip()

    # Only allow SELECT or a CTE beginning with WITH.
    if not re.match(r"^(SELECT|WITH)\b", cleaned, flags=re.IGNORECASE):
        raise ValueError(
            "Blocked: Gemini did not generate a SELECT query or CTE."
        )

    # Reject multiple statements.
    if ";" in cleaned:
        raise ValueError("Blocked: multiple SQL statements are not allowed.")

    # Reject SQL comments to reduce opportunities to bypass simple checks.
    if "--" in cleaned or "/*" in cleaned or "*/" in cleaned:
        raise ValueError("Blocked: SQL comments are not allowed.")

    return cleaned


# --------------------------------------------------
# RUN A QUESTION
# --------------------------------------------------

def answer_question(client, reader, question):
    """Generate SQL, run it through the read-only reader, and print results."""

    sql = generate_sql(client, reader, question)
    sql = validate_sql(sql)

  #  print("\nRunning read-only database query...")

    results = reader.execute_readonly_query(
        sql,
        params=(),
        max_rows=MAX_ROWS,
        timeout_seconds=3.0,
    )

    print("\nResults:")

    if not results:
        print("No matching records found.")
        return

    # sqlite3.Row objects and dictionaries can both be displayed this way.
    for index, row in enumerate(results, start=1):
        try:
            row_data = dict(row)
        except (TypeError, ValueError):
            row_data = row

        print(f"{index}. {row_data}")

    print(f"\nReturned {len(results)} row(s).")


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():
    # print("=" * 60)
    # print("SQLite + Gemini Text-to-SQL")
    # print("=" * 60)

    # The Google Gen AI SDK reads GEMINI_API_KEY from the environment.
    if not os.environ.get("GEMINI_API_KEY"):
        print("\nERROR: GEMINI_API_KEY is not set.")
        print("In PowerShell, run:")
        print('$env:GEMINI_API_KEY = "YOUR_API_KEY_HERE"')
        return

    try:
        client = genai.Client()

        
        db_path = (
            Path(__file__).resolve().parent
            / "data"
            / "wahadb (1).sqlite"
        )

        reader = SQLiteReader(db_path)

        # # TEMPORARY TEST: inspect the cost data
        # rows = reader.execute_readonly_query(
        #     """
        #     SELECT DatasheetId, line, description, cost,
        #         typeof(cost) AS cost_type
        #     FROM datasheets_models_cost
        #     LIMIT 30
        #     """,
        #     max_rows=30,
        # )

        # # for row in rows:
        # #     print(dict(row))


        print("\nDatabase reader initialized.")
        print("Enter a question about the database.")
        print("Type 'exit' to quit.")

        while True:
            try:
                question = input("\nYour question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if question.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break

            if not question:
                continue

            try:
                answer_question(client, reader, question)
            except Exception as exc:
                print(f"\nERROR: {exc}")
                print(
                    "Check the error above. You can try the question again "
                    "or share the complete traceback for troubleshooting."
                )

    except Exception as exc:
        print(f"\nCould not initialize the program: {exc}")


if __name__ == "__main__":
    main()
