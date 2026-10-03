# Research: how production RAG stacks model the per-chunk record

Date: 2026-10-03.
Purpose: inform one typed chunk record in LocalRAG whose stored Chroma format must stay byte-compatible.
Method: library source on GitHub and official docs, plus local probes against chromadb 1.5.9 and pydantic 2.13.4 (Python 3.13.15).
Caveat: GitHub sources below were read from the default branch (`main` or `master`) on the date above, not from a pinned release tag.
Caveat: page text was extracted through a summarising fetcher, so quoted snippets are as returned by it; re-check line-level claims before citing them in an ADR.

## 1. LlamaIndex

Source: https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/schema.py
`BaseNode` carries `id_` (defaults to a uuid4), `embedding`, `metadata` (a flat dict), `excluded_embed_metadata_keys`, `excluded_llm_metadata_keys`, `relationships`, and metadata template/separator fields.
The two `excluded_*` lists are key names, not separate dicts, so one metadata bag serves three audiences: the vector store filter, the embedder, and the LLM prompt.
`NodeRelationship` has five members: `SOURCE`, `PREVIOUS`, `NEXT`, `PARENT`, `CHILD`.
`TextNode` adds `text`, `mimetype`, `start_char_idx`, `end_char_idx`, and `text_template`.
`TextNode.hash` is SHA-256 over `str(text) + str(metadata)`, so it is a content fingerprint used for change detection, not the node id.
`ref_doc_id` is a property that returns `source_node.node_id` from the SOURCE relationship, and it is marked deprecated in favour of reading `source_node`.
`NodeWithScore` is a separate wrapper with two fields, `node: BaseNode` and `score: float | None`, plus pass-through accessors such as `node_id` and `metadata`.
So LlamaIndex separates the stored/retrievable node from the retrieved hit by wrapping rather than by sharing a base class with a score field.

Source: https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/vector_stores/utils.py
`node_to_metadata_dict` serialises the whole node to JSON under the key `_node_content` and records the class name under `_node_type`.
It also writes `document_id`, `doc_id`, and `ref_doc_id`, all set to `node.ref_doc_id or "None"`, so the same value is duplicated under three names for compatibility across vector stores.
With `remove_text=True` the text is blanked inside `_node_content` because the store keeps the text in its own document column.
With `flat_metadata=True` it validates that user metadata contains only str, int, float, or None values.
`metadata_dict_to_node` requires `_node_content` and `_node_type`, and routes to `Node`, `IndexNode`, `ImageNode`, or `TextNode` by the type string.
Lesson: LlamaIndex stores an opaque JSON blob plus a few duplicated flat keys, which makes round-tripping lossless but makes the stored metadata unfilterable beyond the flat keys.

Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/vector_stores/llama-index-vector-stores-chroma/llama_index/vector_stores/chroma/base.py
The Chroma integration passes `node.node_id` as the Chroma id and the text (with `MetadataMode.NONE`) as the Chroma document.
It builds metadata with `node_to_metadata_dict(node, remove_text=True, flat_metadata=...)` and replaces null values with empty strings before calling Chroma.
On read it calls `metadata_dict_to_node(metadata, text=text)` first and, if that fails, falls back to `legacy_metadata_dict_to_node`, which rebuilds a `TextNode` from the older flat layout (`start`, `end`, relationships).
Lesson: a production store keeps a tolerant legacy read path next to the current codec, and it never rewrites old rows during a read.

## 2. LangChain

Source: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/documents/base.py
`BaseMedia` defines `id: str | None` (coerced from numbers, "ideally unique across the document collection") and `metadata: dict` (arbitrary, default empty).
`Document` adds `page_content: str` and `type: Literal["Document"]`.
Metadata is an open bag with no schema, and the `id` is optional.
There is no score on `Document`; scores come back as `(Document, float)` tuples from `similarity_search_with_score`, so the hit is a tuple rather than a type.

Source: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/indexing/api.py
`_get_document_with_hash` hashes content and metadata separately and then hashes the concatenation: `content_hash = H(page_content)`, `metadata_hash = H(json.dumps(metadata, sort_keys=True))`, `id = H(content_hash + metadata_hash)`.
Metadata that is not JSON-serialisable raises `ValueError` rather than being skipped.
Supported algorithms are `sha1`, `sha256`, `sha512`, and `blake2b`.
SHA-1 is the default and the module emits a one-time warning that it is not collision resistant.
The docstring states that "when changing the key encoder, you must change the index as well to avoid duplicated documents", which is the library's only versioning rule for derived ids: changing the hash recipe orphans old ids.
`NAMESPACE_UUID = uuid.UUID(int=1984)` and `_hash_string_to_uuid` returns `uuid5(NAMESPACE_UUID, sha1_hex)`, so ids are UUIDs derived from a digest.
The `index()` function takes a `RecordManager` (a separate table of key, group id, and update timestamp) and a `cleanup` mode.
Cleanup modes: `incremental` deletes stale docs for each seen source id as it goes, `scoped_full` deletes stale docs for seen source ids at the end, `full` deletes everything the loader did not return, and `None` deletes nothing.
`source_id_key` is required for `incremental` and `scoped_full`, because the manager needs a group key to know which old chunks belong to a re-indexed source.
Lesson: LangChain keeps provenance (source id) and change detection (hash) in a side table, not in a versioned chunk schema.

## 3. Haystack 2.x

Source: https://github.com/deepset-ai/haystack/blob/main/haystack/dataclasses/document.py
`Document` is a dataclass with `id`, `content`, `blob`, `meta`, `score`, `embedding`, and `sparse_embedding`.
In `__post_init__`, an empty id is replaced by a SHA-256 over the concatenation `"{text}{dataframe}{blob!r}{mime_type}{meta}{embedding}{sparse_embedding}"`, so the embedding and the metadata are part of the id.
`meta` must be JSON-serialisable and is sorted by key before hashing so insertion order does not change the id.
`score` is documented as "usually assigned by retrievers", so the retrieved hit and the stored document are the same class, with score as an optional field.
`to_dict(flatten=True)` merges `meta` keys into the top level unless they collide with field names, and `from_dict` accepts both nested and flattened forms.
A metaclass `_RemoveLegacyFields` strips deprecated constructor arguments (`content_type`, `id_hash_keys`, `dataframe`) so old serialised documents still load.
Lesson: Haystack content-addresses over everything including metadata and embedding, so any change to either silently produces a new id; it accepts this because ids are only a default.

## 4. Chroma 1.x (the byte-compatibility constraint)

Local source: `.venv/lib/python3.13/site-packages/chromadb/api/types.py` (chromadb 1.5.9), functions `validate_metadata`, `_validate_metadata_list_value`, `validate_ids`, `validate_where`, `validate_include`.
Upstream: https://github.com/chroma-core/chroma/blob/main/chromadb/api/types.py.
Docs: https://docs.trychroma.com/docs/collections/add-data and https://docs.trychroma.com/docs/querying-collections/metadata-filtering.

Allowed value types per the validator: `str`, `int`, `float`, `bool`, `SparseVector`, and lists of those.
Lists must be non-empty and homogeneous (`_validate_metadata_list_value`); the docs say the same.
Nested dicts raise `ValueError`, and non-string keys raise `TypeError`.
The key `chroma:document` (`META_KEY_CHROMA_DOCUMENT`) is reserved.
An empty metadata dict raises `ValueError` ("non-empty dict"), so every record needs at least one key.

None handling is inconsistent between layers and between write calls; both were verified locally against chromadb 1.5.9.
The Python validator lists `type(None)` as accepted, but `collection.add(... metadatas=[{"k": None}])` fails in the Rust binding with `TypeError: ... Cannot convert Python object to MetadataValue`.
`collection.upsert(...)` with the same metadata succeeds and silently drops the key: `{"n": None, "s": "k"}` is stored as `{"s": "k"}`.
So a `None` field must be omitted from the stored dict (or encoded as a sentinel), never written as `None`: depending on the call it either fails or vanishes.
`where={"k": None}` is rejected by `validate_where` ("Expected where value to be a str, int, float, or operator expression").

Type round-trip, verified locally: a stored `3` comes back as `int` and `3.0` comes back as `float`, so the writer's Python type is preserved.
`where={"n": 1.0}` matched a stored int `1`, so equality filters compare numerically.
Booleans are checked before ints in the validator because `isinstance(True, int)` is true, so a bool stays a bool.

Missing keys, verified locally: `{"s": "x"}` excludes records that lack `s`, while `{"s": {"$ne": "x"}}` and `{"s": {"$nin": ["x"]}}` include them.
The docs confirm the `$nin` case ("or the attribute's key is not present").
Consequence: a legacy row that lacks a newly added key will be silently included by negative filters and silently excluded by positive ones.

Ids and write semantics, verified locally: `add` with an existing id keeps the old document and metadata and ignores the new ones, while `upsert` replaces both.
The docs state the same for `add`.
`validate_ids` requires a non-empty list of unique strings, so duplicates inside one batch raise.

`include` contract, verified locally: `get` returns `ids`, `documents`, `metadatas` by default (embeddings only when requested), and `query` adds `distances`.
`validate_include` rejects unknown names and, for `get`, disallows `distances`.

## 5. Other vector stores

Qdrant: https://qdrant.tech/documentation/manage-data/payload/ says payloads are anything representable as JSON, with typed values (integer, float, bool, keyword, geo, datetime, uuid) and an explicit payload index per field only for filtering.
Qdrant is therefore schemaless for storage and typed only at the index; the docs do not state null behaviour.
Qdrant point ids must be unsigned 64-bit integers or UUIDs, and upserting an existing id overwrites it (https://qdrant.tech/documentation/concepts/points/).

Weaviate: https://docs.weaviate.io/weaviate/config-refs/collections describes a typed property schema per collection.
Properties can be added but not modified afterwards, and auto-schema (infer types from data) is on by default but "not recommended for production".
Weaviate's Python client offers `generate_uuid5(identifier, namespace="")`, implemented as `uuid5(NAMESPACE_DNS, str(namespace) + str(identifier))`, to make object ids deterministic (https://github.com/weaviate/weaviate-python-client/blob/main/weaviate/util.py).

Pinecone: https://docs.pinecone.io/guides/index-data/indexing-overview says metadata is flat key-value pairs (no nested objects) of string, integer, float, boolean, or list of strings, with up to 40 KB per record.
I did not find a null rule on that page.

Pattern: every store accepts a flat dict of scalars, the strict ones (Weaviate) are typed per collection, and none validates a record against an application-level chunk type; that is left to the client library.

## 6. Cross-cutting patterns

Closed typed schema versus open bag: LlamaIndex, LangChain, and Haystack all expose an open `metadata`/`meta` dict, and the typed part is limited to text, id, embedding, and score.
LlamaIndex narrows the open bag only at the vector-store boundary with `flat_metadata` validation.
Weaviate is the only store here that enforces a closed schema, and it forbids modifying existing properties, so schema evolution there is additive-only.

Versioning of stored schemas: none of the three libraries stores a `schema_version` field on the chunk.
They evolve by tolerant reads instead: LlamaIndex's `legacy_metadata_dict_to_node` fallback, and Haystack's `_RemoveLegacyFields` metaclass.
LangChain versions only by rule of thumb, "change the key encoder, change the index", i.e. a hash-recipe change is treated as a re-index event.

Deterministic ids: LangChain hashes content plus sorted-JSON metadata into a UUID5, Haystack hashes content plus metadata plus embedding into a SHA-256 hex digest, and LlamaIndex uses random uuid4 ids with a separate `hash` for change detection.
Content-addressed ids that include metadata make any metadata edit a new id, which is why LangChain needs a RecordManager and a cleanup mode to delete the superseded rows.
Position-derived ids (source path plus chunk index) avoid that churn but need explicit deletion of trailing chunks when a document shrinks.

Three-stage separation: LlamaIndex splits the pipeline into node (stored), serialised metadata dict (store boundary), and `NodeWithScore` (hit).
Haystack and LangChain collapse stage one and stage three into one class (Haystack with an optional `score`, LangChain with a tuple).
The only library that gives the store-boundary form its own named functions is LlamaIndex (`node_to_metadata_dict` and `metadata_dict_to_node`), and it is also the one with the most legacy-compat code.

## 7. Pydantic v2 cost on small flat dicts

Docs: https://pydantic.dev/docs/validation/latest/concepts/performance/ recommends reusing a `TypeAdapter`, preferring concrete types (`list`, `dict`) over abstract ones, and using `TypedDict` over nested models (the page cites roughly 2.5x for that case).
The page gives no per-call figures for flat models, so these are my own local numbers.

Local microbenchmark (pydantic 2.13.4, Python 3.13.15, 12-field flat frozen model with str, int, float, bool, and optional fields, best of 5 runs of 10,000 iterations):

| Operation | Time per call |
| --- | --- |
| `Model.model_validate(dict)` | 1.55 microseconds |
| `Model(**dict)` | 1.62 microseconds |
| `model.model_dump()` | 1.53 microseconds |
| `model_validate` then `model_dump` | 3.15 microseconds |
| frozen dataclass constructor `D(**dict)` | 1.25 microseconds |
| `dataclasses.asdict(D(...))` | 4.06 microseconds |
| plain `dict(raw)` copy | 0.10 microseconds |

Order of magnitude: a validated round trip costs about 3 microseconds, so 100,000 chunks cost about 0.3 seconds.
That is negligible beside embedding and Chroma I/O, and `dataclasses.asdict` is slower than `model_dump` because it deep-copies recursively.
The script is not committed; it is a 20-line timeit loop that can be recreated from the table's description.

## Implications for LocalRAG

(a) Pydantic frozen model versus dataclass for the stored-metadata codec.
Pydantic validation costs about 1.5 microseconds per record, within 25 percent of a dataclass constructor, so performance does not decide this (section 7).
Pydantic is the better fit because the codec's job is to reject or coerce what Chroma would reject, and the Chroma validators are strict about types and keys (section 4).
The repo rules already prefer snake_case fields with `alias_generator`, which lets a model map to existing stored key names without a hand-written mapping.

(b) Tolerant legacy reads with defaults.
Chroma never rewrites old rows and negative `where` filters include rows lacking a key (section 4), so old rows will exist indefinitely.
LlamaIndex keeps an explicit legacy fallback in the read path (section 1) and Haystack strips legacy fields on load (section 3), so the precedent is: strict on write, tolerant on read with defaults for absent keys.
Missing-key defaults must match what the old writer implied, otherwise filters and the decoded record will disagree.

(c) Splitter draft, stored chunk, and retrieved hit as separate types.
LlamaIndex separates stored node from `NodeWithScore`, and its boundary functions are the only place where serialisation lives (sections 1 and 6).
Haystack's single class with optional `score` shows the cost of merging: a field that is meaningless for two of the three stages.
A draft has no id or score, a stored chunk has an id and metadata but no score, and a hit has a score and distance; three small types express those invariants without optional fields.

(d) Closed schema versus extras bag.
All three libraries use an open bag, but LocalRAG's stored format is fixed and Chroma rejects nested values and rejects or drops `None` (section 4).
A closed model with a single explicit extras escape hatch is feasible, but the extras must be validated as flat scalars at the boundary, as LlamaIndex's `flat_metadata` does.
If other code or users already write arbitrary keys into collections, a closed model with `extra="ignore"` on read would drop them silently, so decide read behaviour for unknown keys explicitly.

(e) Explicit ids passed to the vector store.
Every store here takes caller-supplied ids (LlamaIndex passes `node_id`, Qdrant and Chroma accept them, Weaviate offers `generate_uuid5`).
Chroma `add` silently keeps the old record on a duplicate id while `upsert` replaces it (section 4), so deterministic ids only help if the write path is an upsert, which `vector_store.py` already uses via `_upsert_batched`.
Whatever goes into the id hash is a compatibility contract: LangChain's rule that changing the encoder orphans old ids applies equally here, so this PR must keep the existing derivation untouched.

(f) Schema-version field.
No surveyed library stores a per-chunk schema version; they rely on tolerant reads and re-indexing (section 6).
Adding a version key would change stored metadata and would be a new key that negative filters treat as absent on old rows, so it is out of scope for this PR.
It becomes warranted only when a change cannot be expressed as a default for an absent key, for example redefining the meaning of an existing key; until then, absence of a key is itself a usable version signal.
If it is added later, record it with an ADR and keep the id recipe independent of it so a schema bump does not orphan existing chunks.
