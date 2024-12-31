# QA and RAG

- Question Answering (QA) and Retrieval Augmented Generation (RAG)

## Table of Contents

- [QA and RAG](#qa-and-rag)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Create The Datasets](#create-the-datasets)

## Setup

### Create The Datasets

- Chinook Database
- Run the following commands to create the SQLite database.

```sh
# Install SQLite3 (Optional if you have it already)
sudo apt-get install sqlite3

# Create the database
sqlite3 path/to/save_db.db < path/to/source_file.sql

# e.g.
sqlite3 data/chinook.db < data/sql/chinook_sqlite.sql

# OR (Interactive shell)
sqlite3 data/chinook.db    
.read ./data/sql/chinook_sqlite.sql
```
