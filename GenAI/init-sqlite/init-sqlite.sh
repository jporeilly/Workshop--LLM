#!/bin/sh
# SQLite initialization script
# This script creates the document_store database and sets up permissions

set -e

# Create data directory if it doesn't exist
mkdir -p /data

# Create the document_store database
sqlite3 /data/document_store.db <<EOF
-- Initialize the database
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA encoding = 'UTF-8';

-- Create a table to store document metadata
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filename TEXT,
    file_size INTEGER,
    metadata TEXT
);

-- Create a table to store document content
CREATE TABLE IF NOT EXISTS document_contents (
    document_id INTEGER PRIMARY KEY,
    content BLOB,
    embedding TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Create a table for document tags
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

-- Create a many-to-many relationship table for documents and tags
CREATE TABLE IF NOT EXISTS document_tags (
    document_id INTEGER,
    tag_id INTEGER,
    PRIMARY KEY (document_id, tag_id),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);

-- Create a trigger to update the updated_at timestamp when a document is modified
CREATE TRIGGER IF NOT EXISTS update_documents_timestamp 
AFTER UPDATE ON documents
BEGIN
    UPDATE documents SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Output success message
SELECT 'SQLite database initialized successfully' as message;
EOF

# Set permissions for the database file
chmod 644 /data/document_store.db

# Create a configuration file with user authentication info
cat > /data/auth.json <<EOF
{
  "username": "sqlite",
  "password": "password",
  "database": "document_store",
  "privileges": "all"
}
EOF

# Secure the auth file
chmod 600 /data/auth.json

echo "SQLite database 'document_store' initialized with user 'sqlite' and full privileges"
echo "Database ready at /data/document_store.db"