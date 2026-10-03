# LocalRAG

LocalRAG answers questions over a person's own documents, offline by default.
This glossary fixes the words used for what it ingests, stores, and retrieves.

## Ingestion

**Source**:
One ingested document, identified by its resolved file path.
_Avoid_: file, document (when meaning the stored identity)

**Source provenance**:
The facts about a source at the moment it was ingested: its file type, content hash, modification time, git commit, ingestion time, and the group it belongs to.
_Avoid_: file metadata

**Group**:
A set of people sharing one deployment whose queries are answered from the sources they are allowed to see, such as the members of a household.
_Avoid_: tenant, team, organization

**Collection**:
A named store of chunks built with one embedding model.
_Avoid_: index, database

## Chunks

**Chunking strategy**:
The rule that splits a source's text into chunk drafts: `fixed`, `structural`, or `recursive`.
_Avoid_: chunker mode, splitter

**Chunk draft**:
A piece of text a chunking strategy produced, before it has a position or identity in its source.
_Avoid_: piece, split, raw chunk

**Chunk**:
A chunk draft placed in its source: it has a source, a zero-based index, the strategy that made it, and a chunk ID.
_Avoid_: node, passage, segment

**Chunk ID**:
The deterministic identity of a chunk, derived from the chunk contract version, source, strategy, index, and text.
_Avoid_: document ID, vector ID

**Heading path**:
The chain of section headings that encloses a chunk in its source; chunks sharing one form a section.
_Avoid_: breadcrumb, parent

**Chunk metadata**:
The fixed set of fields stored alongside a chunk's text and embedding: the chunk's own fields plus its source provenance.
_Avoid_: payload, attributes, extras

## Retrieval

**Retrieval context**:
A chunk returned for a question, carrying its text, chunk metadata, and ranking score.
_Avoid_: hit, result, match

**Citation**:
A pointer from an answer to the exact span of a source that supports it, not merely to the source or section around it.
_Avoid_: reference, source link
