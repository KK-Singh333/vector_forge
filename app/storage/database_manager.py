import sqlite3 as sq
import numpy as np
from pathlib import Path
class DataBaseStoreManager:
    def __init__(self,db_path,logger=None):
        self.vector_manager=None
        self.conn = sq.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sq.Row
        self._init_tables()
        self.logger=logger
    def _init_tables(self):
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
                          user_id INTEGER PRIMARY KEY,
                          current_chunk_count INTEGER NOT NULL DEFAULT 0
                          );
        ''')
        self.conn.commit()
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS pdf_table (
                          pdf_id INTEGER PRIMARY KEY,
                          user_id INTEGER NOT NULL
                          );
        ''')
        self.conn.commit()
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS chunk_table (
                          chunk_id INTEGER PRIMARY KEY,
                          pdf_id INTEGER NOT NULL,
                          user_id INTEGER NOT NULL,
                          page_no INTEGER NOT NULL,
                          chunk_text TEXT
                          );
        ''')
        self.conn.commit()
    def register_vector_store_manager(self,vector_store):
        self.vector_manager=vector_store

    def get_user_chunk_count(self,user_id):
        cursor = self.conn.execute('SELECT current_chunk_count FROM users WHERE user_id = ?', (user_id,))
        row=cursor.fetchone()
        if row:
            return row[0]
        else:
            return 0
    def set_user_chunk_count(self,user_id,chunk_count):
        self.conn.execute('UPDATE users SET current_chunk_count = ? WHERE user_id = ?',
                          (chunk_count,user_id))
        self.conn.commit()
    def get_user_id_by_pdf_id(self,pdf_id):
        cursor = self.conn.execute('SELECT user_id FROM pdf_table WHERE pdf_id = ?', (pdf_id,))
        row = cursor.fetchone()
        if row:
            return row[0]
        else:
            return None
    def get_pdf_ids_by_user_id(self,user_id):
        cursor = self.conn.execute('SELECT pdf_id FROM pdf_table WHERE user_id = ?', (user_id,))
        rows = cursor.fetchall()
        pdf_ids = [row[0] for row in rows]
        return pdf_ids
    def get_chunk_metadata_by_chunk_id(self,chunk_id):
        if isinstance(chunk_id,list):
            rows=[]
            for chunk in chunk_id:
                cursor=self.conn.execute('SELECT * FROM chunk_table WHERE chunk_id = ?',
                                 (chunk,))
                row=cursor.fetchone()
                rows.append(row)
            return rows
        else:
            cursor=self.conn.execute('SELECT * FROM chunk_table WHERE chunk_id = ?',
                                    (chunk_id,))
            row=cursor.fetchone()
            if not row:
                return None
            return row
    def allocate_vectors(self, embeddings):
        vectors, chunks = embeddings

        if not chunks:
            return []

        if len(vectors) != len(chunks):
            raise ValueError("Mismatch between vectors and chunk metadata length")

        user_id = chunks[0].user_id
        count = len(chunks)

        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")

        try:
            row = cursor.execute(
                "SELECT current_chunk_count FROM users WHERE user_id = ?",
                (user_id,)
            ).fetchone()

            if row is None:
                raise ValueError(f"User {user_id} not found")

            current = row["current_chunk_count"]
            new_value = current + count

            cursor.execute(
                "UPDATE users SET current_chunk_count = ? WHERE user_id = ?",
                (new_value, user_id)
            )

            
            global_ids = [
                (user_id << 32) | (current + i + 1)
                for i in range(count)
            ]

           
            rows = [
                (
                    global_ids[i],
                    chunks[i].user_id,
                    chunks[i].pdf_id,
                    chunks[i].page_no,
                    chunks[i].chunk_text
                )
                for i in range(count)
            ]

            
            cursor.executemany("""
                INSERT INTO chunk_table
                (chunk_id, user_id, pdf_id, page_no, chunk_text)
                VALUES (?, ?, ?, ?, ?)
            """, rows)

           
            success=self.vector_manager.add_vectors(vectors,global_ids)
            if not success:
                self.conn.rollback()
                return []

            self.conn.commit()
            self.logger.info('[DATABASE] UPDATED SUCCESS')
            return global_ids

        except Exception:
            self.logger.exception('[DATABASE] ERROR')
            self.conn.rollback()
            raise

