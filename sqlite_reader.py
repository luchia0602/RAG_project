
from pathlib import Path
import sqlite3
import time

# Locate the project and database
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def find_database() -> Path:
    """Find the SQLite database inside the project's data folder."""
    candidates = [
        *DATA_DIR.glob("*.sqlite"),
        *DATA_DIR.glob("*.sqlite3"),
        *DATA_DIR.glob("*.db"),
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No SQLite database found in {DATA_DIR}"
        )

    if len(candidates) > 1:
        names = "\n".join(str(path.name) for path in candidates)
        raise RuntimeError(
            "Multiple SQLite databases found. "
            f"Please select one explicitly:\n{names}"
        )

    return candidates[0]


class SQLiteReader:
    """Read-only reader for the Warhammer SQLite database."""

    def __init__(self, db_path: Path):
        self.db_path = db_path.resolve()

        if not self.db_path.is_file():
            raise FileNotFoundError(
                f"Database not found: {self.db_path}"
            )

    def connect(self) -> sqlite3.Connection:
        """Open a read-only database connection."""
        uri = self.db_path.as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def get_tables(self) -> list[str]:
        """Return the names of all user-created tables."""
        with self.connect() as connection:
            rows = connection.execute("""
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """).fetchall()

        return [row["name"] for row in rows]

    def get_schema(self) -> dict[str, list[str]]:
        """Return each table and its column names."""
        schema = {}

        with self.connect() as connection:
            for table in self.get_tables():
                safe_table = table.replace('"', '""')

                rows = connection.execute(
                    f'PRAGMA table_info("{safe_table}")'
                ).fetchall()

                schema[table] = [
                    row["name"] for row in rows
                ]

        return schema


    

    def search_units_by_name(
        self,
        unit_name: str,
        faction_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search datasheets by unit name and optional faction."""

        query = """
            SELECT Id, name, factionId, legend, loadout, link
            FROM datasheets
            WHERE name LIKE ?
        """
        params = [f"%{unit_name}%"]

        if faction_id:
            query += " AND factionId = ?"
            params.append(faction_id)

        query += " ORDER BY name LIMIT ?"
        params.append(limit)

        with self.connect() as connection:
            rows = connection.execute(
                query, params
            ).fetchall()

        return [dict(row) for row in rows]


    def get_unit_by_id(self, unit_id: str) -> dict | None:
        """Retrieve one unit's datasheet by its ID."""
        query = """
            SELECT *
            FROM datasheets
            WHERE id = ?
        """

        with self.connect() as connection:
            row = connection.execute(
                query, (unit_id,)
            ).fetchone()

        return dict(row) if row else None


    def get_unit_details(self, unit_id: str) -> dict | None:
        """Retrieve a unit and its related datasheet records."""

        queries = {
            "unit": """
                SELECT *
                FROM datasheets
                WHERE Id = ?
            """,
            "abilities": """
                SELECT
                    da.line,
                    da.abilityId,
                    da.name AS datasheet_ability_name,
                    da.description AS datasheet_ability_description,
                    da.type,
                    da.parameter,
                    a.name AS ability_name,
                    a.description AS ability_description
                FROM datasheets_abilities AS da
                LEFT JOIN abilities AS a
                    ON a.Id = da.abilityId
                WHERE da.DatasheetId = ?
                ORDER BY da.line
            """,
            "weapons": """
                SELECT *
                FROM datasheets_wargear
                WHERE DatasheetId = ?
                ORDER BY line, lineInWargear
            """,
            "models": """
                SELECT *
                FROM datasheets_models
                WHERE DatasheetId = ?
                ORDER BY line
            """,
            "keywords": """
                SELECT *
                FROM datasheets_keywords
                WHERE DatasheetId = ?
            """,
            "costs": """
                SELECT *
                FROM datasheets_models_cost
                WHERE DatasheetId = ?
                ORDER BY line
            """,
            "options": """
                SELECT *
                FROM datasheets_options
                WHERE DatasheetId = ?
                ORDER BY line
            """,
        }

        with self.connect() as connection:
            unit = connection.execute(
                queries["unit"], (unit_id,)
            ).fetchone()

            if unit is None:
                return None

            result = {"unit": dict(unit)}

            for key in (
                "abilities",
                "weapons",
                "models",
                "keywords",
                "costs",
                "options",
            ):
                rows = connection.execute(
                    queries[key], (unit_id,)
                ).fetchall()

                result[key] = [dict(row) for row in rows]

        return result


    



    def execute_readonly_query(
        self,
        sql: str,
        params: tuple = (),
        max_rows: int = 100,
        timeout_seconds: float = 3.0,
    ) -> list[dict]:
        """Execute a restricted, read-only SQL query safely."""

        if not sql or not sql.strip():
            raise ValueError("SQL query cannot be empty.")

        if not 1 <= max_rows <= 1000:
            raise ValueError("max_rows must be between 1 and 1000.")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        statement = sql.strip().rstrip(";").strip()

        # Only allow queries beginning with SELECT or WITH.
        first_word = statement.split(None, 1)[0].upper()

        if first_word not in {"SELECT", "WITH"}:
            raise ValueError("Only SELECT queries and CTEs are allowed.")

        deadline = time.monotonic() + timeout_seconds

        with self.connect() as connection:
            # Extra SQLite-level restrictions.
            connection.execute("PRAGMA query_only = ON")

            # Deny database writes and other potentially dangerous actions.
            denied_actions = {
                sqlite3.SQLITE_INSERT,
                sqlite3.SQLITE_UPDATE,
                sqlite3.SQLITE_DELETE,
                sqlite3.SQLITE_CREATE_INDEX,
                sqlite3.SQLITE_CREATE_TABLE,
                sqlite3.SQLITE_CREATE_TEMP_INDEX,
                sqlite3.SQLITE_CREATE_TEMP_TABLE,
                sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
                sqlite3.SQLITE_CREATE_TEMP_VIEW,
                sqlite3.SQLITE_CREATE_TRIGGER,
                sqlite3.SQLITE_CREATE_VIEW,
                sqlite3.SQLITE_DROP_INDEX,
                sqlite3.SQLITE_DROP_TABLE,
                sqlite3.SQLITE_DROP_TEMP_INDEX,
                sqlite3.SQLITE_DROP_TEMP_TABLE,
                sqlite3.SQLITE_DROP_TEMP_TRIGGER,
                sqlite3.SQLITE_DROP_TEMP_VIEW,
                sqlite3.SQLITE_DROP_TRIGGER,
                sqlite3.SQLITE_DROP_VIEW,
                sqlite3.SQLITE_ALTER_TABLE,
                sqlite3.SQLITE_ATTACH,
                sqlite3.SQLITE_DETACH,
                sqlite3.SQLITE_PRAGMA,
            }

            def authorize(action, arg1, arg2, db_name, source):
                if action in denied_actions:
                    return sqlite3.SQLITE_DENY

                if (
                    action == sqlite3.SQLITE_FUNCTION
                    and (arg2 or arg1 or "").lower() == "load_extension"
                ):
                    return sqlite3.SQLITE_DENY

                return sqlite3.SQLITE_OK

            connection.set_authorizer(authorize)

            def check_timeout():
                return int(time.monotonic() > deadline)

            connection.set_progress_handler(check_timeout, 1000)

            # SQLite's execute() rejects multiple SQL statements.
            # Wrap the query to enforce a maximum result count.
            wrapped_query = (
                f"SELECT * FROM ({statement}) AS rag_result LIMIT ?"
            )

            try:
                cursor = connection.execute(
                    wrapped_query,
                    (*params, max_rows),
                )
                return [dict(row) for row in cursor.fetchall()]
            except sqlite3.OperationalError as exc:
                if "interrupted" in str(exc).lower():
                    raise TimeoutError(
                        "SQL query exceeded its execution time limit."
                    ) from exc
                raise




# def main() -> None:
#     reader = SQLiteReader(find_database())

#     print("=== UNIT SEARCH TEST ===")

#     results = reader.search_units_by_name(
#         "Hive Tyrant",
#         limit=5,
#     )

#     if not results:
#         print("No matching units found.")
#         return

#     for unit in results:
#         print(
#             f"Name: {unit['name']} | "
#             f"ID: {unit['Id']} | "
#             f"Faction: {unit['factionId']}"
#         )

#     # Test retrieval using the first matching unit's ID.
#     unit_id = results[0]["Id"]
#     details = reader.get_unit_details(unit_id)

#     print("\n=== UNIT DETAILS TEST ===")
#     print(f"Unit: {details['unit']['name']}")

#     for section in (
#         "abilities",
#         "weapons",
#         "models",
#         "keywords",
#         "costs",
#         "options",
#     ):
#         print(f"\n{section.upper()}: {len(details[section])} records")

#         for item in details[section][:3]:
#             print(item)


# if __name__ == "__main__":
#     main()



def main() -> None:
    reader = SQLiteReader(find_database())

    print("=== READ-ONLY SQL TEST ===")

    query = """
        SELECT name, factionId, Id
        FROM datasheets
        WHERE name LIKE ?
        ORDER BY name
    """

    results = reader.execute_readonly_query(
        query,
        params=("%Hive Tyrant%",),
        max_rows=10,
    )

    for row in results:
        print(row)

    print(f"\nReturned {len(results)} rows.")

    print("\n=== WRITE BLOCKING TEST ===")

    try:
        reader.execute_readonly_query(
            "DELETE FROM datasheets"
        )
    except (ValueError, sqlite3.DatabaseError) as exc:
        print(f"Correctly rejected: {exc}")


if __name__ == "__main__":
    main()
