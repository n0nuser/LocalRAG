# Foundations pre-grill: #224 #225 #226 #227 #228 #229, plus #206 #207 #213 #195

Prepared 2026-10-03 while #223 was being implemented; #223 landed as commit 86684a8 during the work.
Every file and line cite below was re-checked against 86684a8, which is the current HEAD, and sections 1.2 and 12 describe the record as landed.
Terms follow CONTEXT.md: source, source provenance, group, collection, chunking strategy, chunk draft, chunk, chunk ID, heading path, chunk metadata, retrieval context, citation.
"SETTLED (technical)" means I decided it with a one-line rationale and the grilling round should only confirm it.
"OPEN (product)" means the user must own it.
Rule references (R1.8, R4.1 and so on) are rows of `.cursor/rules/python-review-checklist.mdc`.

## Product decisions (settled 2026-10-03)

The owner accepted every recommended answer to the open product questions below; the "OPEN (product)" entries in later sections are now settled as stated here.

1. **Access assignment (#225, #195):** each source carries a list of groups that may see it, not a single owner.
   Ingest folders get access rules in config, and new files under a folder inherit its groups.
   Uploads belong to the uploader's groups, with an optional explicit share list.
   One admin principal, set in config, alone can rebuild, delete collections, and reassign access.
2. **Unlabelled data when groups are enabled (#225):** fail closed.
   Chunks with no access list are invisible to every restricted principal until an admin assigns them; startup warns and an audit command lists unlabelled sources.
3. **Identical uploads from two members (#195):** one stored source whose groups are the union of both uploaders' groups, revoked per uploader.
4. **Freshness after an edit (#206):** unchanged chunks keep their original ingestion time; recency means when the text first entered the index.
5. **Citation precision (#213):** three tiers, documented.
   Text, Markdown, and code cite an exact character range; PDF cites a page plus a quoted snippet; Office and other converted formats cite a quoted snippet.
   A citation into a file that changed since indexing shows the stored quote with its surrounding text and is marked stale.
6. **Uploaded originals (#213, #201):** kept by default, within the existing upload quota, with an opt-out.

## 0. Verified facts that the rest of this document rests on

I ran small probes against the installed packages and a scratch persist path under the scratchpad directory.
Nothing in the repo was modified.

F1. The API process holds two parallel sets of runtime objects, and they are not the same instances.
`api/dependencies.py:29-100` and `application/container.py:23-76` plus `application/runtime.py:13-48` each have their own `lru_cache` factories.
`_BuiltinPlugin.create` (`plugins/retriever.py:69-90`) pulls the vector store, embedder, BM25 and reranker from `application.runtime`.
`api/dependencies.get_ingestion_service` (`:75-90`) uses its own `get_vector_store` and `get_bm25_index` from the same file.
I constructed both inside one process: the API ingestion service's BM25 index is not the API retriever's BM25 index, the two vector stores differ, and the two embedders differ.
Consequence: after an API ingest, `IngestionService` refreshes its own BM25 (`ingestion/service.py:198`) while the retriever keeps scoring a BM25 snapshot nobody refreshed.
The ingest routers only call `query_cache.clear()` (`api/routers/ingest.py:79,91,103`), and `rebuild` does the same (`api/routers/collections.py:57`).
I did not run a live ingest-then-query; this is a probe of object identity plus a read of the call sites.

F2. Chroma 1.5.9 behaviour I probed (scratch persist path, three-document collection).
Array metadata works with `where={"field": {"$contains": "alice"}}`, and `$or` over several `$contains` clauses works.
`where={"source": {"$in": [...]}}` works.
`collection.update(ids=[...], metadatas=[{"acl": [...]}])` with a partial dict merges: other keys (`source`, `n`) were kept and the list value was replaced.
`update` with metadata only left the stored text and embedding untouched.
The research note `docs/research-chunk-record.md` §4 adds: lists must be non-empty and homogeneous, `None` values are dropped or rejected, and a positive `where` excludes rows that lack the key.

F3. There are three implementations of "metadata filter", not two.
Dense search passes the client dict straight to Chroma `where` (`rag/retriever.py:271`).
BM25 post-filters with `_matches_filter` after top-k (`rag/retriever.py:38-41`, applied at `:256`; `bm25_index.py:74-103` takes no filter).
Parent expansion reimplements it a third time: it builds an `$and` for Chroma and then re-checks equality in Python (`storage/vector_store.py:344` and `:358-363`).
The public client surface is already equality-only on strings: `dict[str, str]` in `api/schemas.py:61` and `application/dto.py:12`.

F4. The SSE endpoint never runs adaptive retrieval.
`api/routers/query.py:59-61` passes `[]` as contexts when `adaptive_enabled`.
`iter_query_sse_events` then calls `engine.stream_chat_from_contexts(contexts=[])` (`application/service.py:608-612`), not `stream_answer`.
So with adaptive on, the streamed answer is generated over an empty context (or refused only if `rag_min_context_score > 0`).
This is read from code, not run.

F5. "Tokens used" means three different things.
Adaptive JSON path counts characters: `application/service.py:476-478`.
Retrieve-then-stream JSON path and SSE path count streamed chunks: `:535` and `:623`.
Ollama's `LLMResponse.tokens_used` is `len(answer.split())` (`llm/providers/ollama.py:131`).
The stream path drops provider usage entirely: `OllamaChatStreamChunk` (`ollama/schemas.py:87`) does not parse it, and every `stream_from_prompt` ends with `{"type": "final", "sources": []}`.
Ollama's final `done: true` chunk carries `prompt_eval_count` and `eval_count` (https://docs.ollama.com/api/chat).
OpenAI streams usage only with `stream_options={"include_usage": true}`, in a last chunk with `choices: []`, and an interrupted stream may never send it (https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events).
Anthropic non-stream responses already expose `resp.usage` (`anthropic_provider.py:57,96`).

F6. Uploads are deleted right after ingest by default.
`upload_retention_seconds` defaults to `0.0` (`settings_groups.py:96`), and `ingest_upload` unlinks the file when it is `<= 0` (`application/service.py:319-321`).
Uploads are stored as `<sha256>.<ext>` (`:298`), so the original filename is not in the source identity, and identical bytes dedupe into one source (`:299-303`).
`rebuild_collection` treats any source whose file is gone as missing and calls `delete_by_source` (`ingestion/service.py:122`).
So with defaults, an uploaded source cannot be rebuilt, and any citation that points into the original file points at nothing.

F7. Chunkers do not track source positions, and several chunk texts are not substrings of the source.
`chunks/record.py:48-50` says offsets are absent because strategies strip whitespace and structural chunking repacks blocks.
`chunks/recursive.py:28` joins pieces with a single space and prepends an overlap slice, so a recursive chunk is not a contiguous slice.
`chunks/fixed.py:9,22` strips the whole text and each window.
`parsers/markdown.py` strips frontmatter and surrounding whitespace, and PDF and Office files are converted to Markdown (`parsers/pdf.py`, `parsers/anydoc.py`), so even a perfect offset into parsed text is not an offset into the original file.

F8. The prompt block that `[n]` refers to is not the chunk the user sees as a source.
Parent expansion is on by default (`settings_groups.py:295`), and it replaces a retrieval context's text with the whole section: `"\n\n".join(...)` at `rag/retriever.py:474`, stored as `expanded_text`.
`build_prompt` numbers `compression.contexts` (`rag/engine.py:185,192-195`), which are extractive sentences from those sections.
The returned `sources` come from `claim_filter.contexts` (`:206`), which is a different list and is deduplicated by `(source, chunk_index)` (`:215`).

F9. `last_hyde` is mutable state on a shared object.
`Retriever.last_hyde` is assigned during `retrieve` (`rag/retriever.py:68,127,201-203`) and read later by the engine through `getattr` (`rag/engine.py:108,129,209`).
The retriever is a process singleton, so two concurrent queries can report each other's HyDE trace (R11.2).

F10. Request-scoped collections are expensive and leak.
`for_collection` exists four times: `storage/vector_store.py:140`, `rag/retriever.py:70`, `rag/engine.py:31`, `plugins/retriever.py:199`.
`Retriever.for_collection` rebuilds BM25 with a full-collection scan on every request that names a collection (`rag/retriever.py:77`).
`ManagedRetriever.for_collection` calls `registry.track` on every request (`plugins/retriever.py:204`), so tracked instances grow without bound (R12.1).

F11. The plugin version check is stricter than ADR 032 says.
ADR 032 describes `MAJOR.MINOR` where minor compatibility is declared by the plugin.
The code rejects any plugin whose `contract_version` is not equal to the host's (`plugins/retriever.py:103`), so a host minor bump rejects every existing plugin.
`_corpus_revision` returns `"0"` for any non-built-in retriever (`application/service.py:387-392`), so cached answers for plugin retrievers never see a revision change.

F12. Import-graph scan of `localrag/` (AST, absolute imports).
Ticket edges confirmed: embedding to ingestion (`embedding/factory.py:6`), llm to rag (`llm/providers/{anthropic_provider.py:14, ollama.py:21, openai_provider.py:14}`), ingestion to rag (`ingestion/service.py:22`), plugins to application (`plugins/retriever.py:16`).
Edges the tickets miss: `settings_groups.py:18` imports `rag.prompt` (the settings foundation depends on rag), `storage/vector_store.py:17` imports `embedding.base`, and `application/container.py` imports `plugins.retriever` while `plugins.retriever` imports `application.runtime` (a cycle waiting to happen when runtime merges into the container).
`parsers/docx.py` is imported by no production code: `loader.parse_file` routes `.docx` through `anydoc` (`ingestion/loader.py:137-160`), and only `tests/test_parsers_and_loader.py:10` imports it.
`HTTPStatus` appears in domain code only in `rag/retriever.py:6,181,275`.

## 1. Order, dependencies, and conflicts

### 1.1 Recommended order

Step 0. #223 has landed (86684a8).
Section 12 lists the gaps in the landed record that later tickets need closed.
Step 1. #205 as a standalone slice of #225: mask BM25 candidates before top-k with the existing equality semantics.
It is a bug fix and does not need the Scope type.
Step 2. #224 (composition root).
Everything below needs one place to wire things.
Step 3. #229a in parallel with step 2 or right after it: the pure moves and deletions (section 7).
Step 4. #226 (collection handle).
It is small, it removes the reach-through #225 needs for the cache key, and it gives ACL re-stamping its BM25 invalidation.
Step 5. #225 (Scope) with its ADR.
Step 6. #227 (answer contract).
It can land before #228 because it only needs the #223 `RetrievalContext` type, not the #228 retrieved-chunk type.
Step 7. #228 (retrieval module), which then replaces the dict-shaped contexts in the answer path.
Step 8. #206, #207 and the span fields in one "chunk contract v2" window so users rebuild once (section 1.3).
Step 9. #213 then #195.

This differs from the tracking issue only by putting #226 before #225 and splitting #229.
The tracking issue's step 3 (`#225 then #194 -> #195 -> #196`) is unchanged; #194 identity wiring sits between step 5 and #195.

### 1.2 Conflicts between tickets and with the #223 design

C1. The landed record has no offsets (`Chunk` docstring, `chunks/record.py:48-50`), and the user's citation requirement needs them.
`ChunkDraft` (`chunks/record.py:31-41`) and `ChunkMetadata` need an optional covered source range (start and end in parsed-text coordinates), written only when present because Chroma cannot store `None`.
`ChunkMetadata.to_stored` already omits `None` fields (`chunks/record.py:147-148`), so adding an optional field leaves old rows byte-identical.
The chunkers must compute the range while splitting (F7); it cannot be recovered afterwards from stripped and repacked text.

C2. The access field must be representable in `ChunkMetadata`, and today it cannot be.
`ChunkMetadata` declares `extra="ignore"` and a closed field list (`chunks/record.py:103-117`), so an `access_groups` key written by a metadata `update` is dropped by any read and rewrite that goes through the codec.
`StoredValue` is `str | int | float | bool` (`chunks/record.py:20`), so a list-of-strings field needs the type widened.
`ChunkField` has two members, `SOURCE` and `HEADING_PATH` (`chunks/record.py:23-27`), so the client-filterable set in section 4 is a new, explicit subset.

C3. The landed record has no stored chunk-contract version, but #206 needs one.
`stable_chunk_id` hashes `CHUNK_CONTRACT_VERSION` (`chunks/record.py:18,62-78`), yet no stored key records which version produced a row.
`rebuild_collection` skips a source whose `content_hash` is unchanged (`ingestion/service.py:126`), so after a contract bump a rebuild would silently keep every old-version ID.
`ChunkMetadata` needs a `chunk_contract_version` (omitted on legacy rows, read as version 1), and rebuild must treat a version mismatch as "changed".

C4. The CONTEXT.md glossary and the landed `SourceProvenance` conflict with #195.
"Source provenance" includes "the group it belongs to" (singular, an ingest-time fact), and the code carries it as `tenant_id` (`chunks/record.py:90`).
#195 needs an allowed-groups list that changes after ingest without re-embedding, which is not provenance.
Recommendation: remove "group" from Source provenance, add a separate stored "access" field, and add glossary terms Principal, Scope and Access.
The product answer on how access is assigned (Q1 below) decides the final wording.

C5. "Chunk ID" in the glossary is defined with the index; #206 removes the index from the recipe.
The single place that derives it is `Chunk.place` (`chunks/record.py:62-78`), so #206 changes one function plus the glossary and `CHUNK_CONTRACT_VERSION`.

C6. #228 says "typed hits (the #223 record)", but #223 landed only a `RetrievalContext` TypedDict, a `retrieval_context()` builder, `Chunk` and `ChunkMetadata` (`chunks/record.py:151-197`).
There is no internal retrieved-chunk dataclass yet; #228 must define one (section 6).

C7. #229 says `prompt` moves to generation and domain errors use the application error enum.
Both break layering as written (section 7): moving `prompt` to generation keeps the llm to generation to llm cycle, and `rag` importing `application` inverts ADR 038.

C8. #228 acceptance says "Generation is the only caller of retrieval", which contradicts ADR 038 and the product surface.
MCP `search_documents` (`mcp/server.py` tool `search_documents`) and `POST /query/contexts` (`api/routers/query.py:30-49`) are retrieval-only callers.
Reword to "nothing reaches past the retrieval seam".

C9. #224 says it "implements ADR 038 as written", but merging `runtime.py` into `container.py` creates an import cycle through `plugins.retriever` (F12).
The builtin retriever must be injected into plugin discovery (section 2), which touches the plugin factory but not the public `RetrieverPlugin.create(settings)` contract.

C10. #229 moves `bm25_index` to indexing while #226 creates the indexing handle that owns BM25.
Do the pure rename once in #229a, before #226, so #226's diff is only the handle.

C11. #225 and #226 both want to own the cache key's corpus revision.
Today it is read by `_corpus_revision` reaching through `engine.retriever.vector_store.collection.metadata` (`application/service.py:387-392`).
After #226 the handle exposes `revision`, and #225 builds the key from Scope plus `handle.revision`.

### 1.3 One rebuild for the whole chunk contract

These changes all alter what is stored or how it is derived: #206 (chunk ID recipe), span fields (#213), #207 (embedded and lexical text), and the access field (#195).
Only #206 and #207 force a re-embed; the span and access fields can be added as metadata-only updates.
Recommendation: ship #206 and the span fields together as `CHUNK_CONTRACT_VERSION = 2`, ship #207 behind an opt-in with its format recorded in the collection's embedding identity, and let #195 add its field without a version bump (old rows read as unlabelled).
That gives users one forced rebuild instead of three.

## 2. #224 composition root

### 2.1 Design tree

- Goal: one constructor for every runtime object, one invalidation path, one test override point.
  - Branch A, what holds the objects.
    - Today: module-level `lru_cache` functions in two files (F1).
      `lru_cache` does not lock the call, so two threads can both build on first use.
    - Choice: an instance of `Container` built from `Settings`, not module globals (R4.1 is CRITICAL).
  - Branch B, the cycle with plugins.
    - `container.py` imports `plugins.retriever`; `plugins.retriever` imports `application.runtime` (`plugins/retriever.py:16`).
    - Choice: `discover_retriever_plugins` receives the builtin retriever factory from the container instead of importing it.
  - Branch C, who closes and invalidates.
    - Today: `api/main.py:12` imports `get_embedder` and `get_retriever` for lifespan teardown; `invalidate_retrieval_caches` (`api/dependencies.py:128-137`) hand-clears four caches and the router calls it only on collection delete (`api/routers/collections.py:45-46`).
    - Choice: `Container.close()` and `Container.invalidate()`.
  - Branch D, what stays in `api/dependencies.py`.
    - `require_api_key`, `Security`/`Depends` glue, and thin getters that read the container from `request.app.state`.
  - Branch E, identity wiring for #194/#225.
    - The container is the place where adapters will ask for a `Principal`; nothing is added now (R1.8).

### 2.2 SETTLED (technical)

S224.1. Replace the two `lru_cache` sets with one `Container` class built by `Container.build(settings)`, owning vector store, embedder, reranker, BM25, retriever, engine, query cache, ingestion service and job registry.
Rationale: R4.1 forbids module-level mutable singletons, R4.2 wants a `@classmethod` factory, and an instance gives tests one override point.
S224.2. Guard lazy construction with one `threading.RLock` and double-checked access.
Rationale: sync FastAPI routes run in a thread pool, `functools.cached_property` has no lock since 3.12, and `lru_cache` does not serialise concurrent first calls, which is exactly how two vector stores could be created again (R11.2).
S224.3. Keep construction lazy, not eager.
Rationale: `mcp/app.py:36-46` documents that `tools/list` and `initialize` must stay cheap, and the CLI builds a container per command.
S224.4. API: build the container in the FastAPI lifespan, store it on `app.state.container`, and expose `get_container(request)` plus thin `Depends` getters.
Rationale: this is the FastAPI-documented place for process-lifetime resources (R12.2) and tests override `get_container` once.
S224.5. CLI builds a container per invocation; MCP stdio builds one in `main()`; MCP HTTP builds one in its lifespan and closes it.
Rationale: matches how `cli/commands/*` and `mcp/__main__.py` call the container functions today.
S224.6. `RetrieverPluginRegistry` / `discover_retriever_plugins` take a `builtin_factory: Callable[[Settings], RetrieverContract]` supplied by the container; `plugins/retriever.py` stops importing `application` and `rag.retriever`.
Rationale: removes the cycle, keeps the public `RetrieverPlugin.create(settings)` signature unchanged for third parties, so no contract bump in #224.
S224.7. Delete `application/runtime.py`, `settings_for_runtime`, `get_api_settings` and the `clear_runtime_caches` helper.
Rationale: R1.12 pass-throughs; settings become `container.settings`.
S224.8. One `Container.invalidate()` closes the retriever and rebuilds engine, retriever and BM25, and clears the query cache; ingest, rebuild and delete all go through it.
Rationale: F1 shows the current mix leaves a stale BM25; #226 later narrows this to the collection handle.
S224.9. Leave `ChromaCollectionRepository` and the `cast(...)` calls in `container.py` for #226 and #228.
Rationale: they are removed by deleting the thing they paper over (`ManagedRetriever.__getattr__`, `plugins/retriever.py:221`), not by editing them here; the casts must not be copied into the new class.
S224.10. Tests are interface-level: assert shared identity (`container.ingestion_service.bm25_index is container.retriever` collaborators), and a behavioural test with a fake store: ingest through the API app, then retrieve through the MCP tool function in the same container, and see the new chunk.
Rationale: the acceptance criterion names exactly this, and R10.2 forbids isolated `__init__` tests.
S224.11. Amend ADR 038 with one status line ("the container is the sole composition root; adapters depend on it") and update `docs/architecture.md`, `docs/agent-navigation.md` and the AGENTS.md package map.
No new ADR.
Rationale: ADR 038 already decides the layering; this removes the duplicate that violated it.
S224.12. Document the cross-process limit instead of fixing it: a separate MCP HTTP process keeps its own snapshot because ADR 035 allows one writer per persist path.
Rationale: out of scope and already an accepted boundary.

OPEN (product): none.
Count: 12 settled, 0 open.

## 3. #226 collection handle

### 3.1 Design tree

- Goal: one object that binds a collection's store, BM25 snapshot, embedding identity and revision, and owns invalidation.
  - Branch A, where the `localrag:*` keys live.
    - Today `VectorStore` writes and reads `localrag:embedding_provider/model/dimension` and `localrag:corpus_revision` (`storage/vector_store.py:243-312`) and `application/service.py:387-392` reads the revision from outside.
    - Choice: a small codec in the new `localrag/indexing/` package owns the key names; `VectorStore` keeps only generic collection-metadata read/write primitives.
      That also removes `storage` to `embedding` (F12).
  - Branch B, the four `for_collection` copies (F10).
    - Choice: one `open(name)` on a registry of handles with a bounded LRU.
  - Branch C, probes.
    - Choice: delete the `getattr` probes and the legacy fakes (listed in S226.5).
  - Branch D, alias swap (#209).
    - Choice: not built here; only one function resolves a logical name to a physical collection.

### 3.2 SETTLED (technical)

S226.1. New package `localrag/indexing/` with `CollectionHandle` (store, BM25, `EmbeddingIdentity`, `revision`, `invalidate()`, `commit()`), and a `CollectionRegistry` that opens handles.
Rationale: the tracking issue's target map names an indexing module and nothing else owns this state.
S226.2. `EmbeddingIdentity` is a frozen value (provider, model, dimension or None) with `from_provider()` and a compatibility check, stored under the existing `localrag:*` keys so existing collections open unchanged.
Rationale: same codec pattern as the settled #223 `ChunkMetadata`, and ADR 019/035 stored bytes stay compatible.
S226.3. `VectorStore` loses `ensure_embedding_compatibility`, `record_embedding_compatibility` and `_bump_revision`; it exposes plain collection-metadata get/set.
The handle does compatibility and revision.
Rationale: acceptance says no module outside indexing reads `localrag:*`, and `storage` stops importing `embedding`.
S226.4. The handle verifies compatibility once per (handle, embedder identity) and caches the verdict until `invalidate()`.
Rationale: today it runs once per HyDE or expansion variant (`rag/retriever.py:164,168`), and for legacy collections without a recorded dimension each call does a `collection.get(include=["embeddings"], limit=1)` (the legacy branch of `ensure_embedding_compatibility`, `storage/vector_store.py:243-286`).
S226.5. Delete these probes, and rewrite their fakes to implement the real interface: `ingestion/service.py:310,325,338`, `rag/retriever.py:162,461`, `storage/vector_store.py:229,311,313,402` (#223 already removed the `replace_source` probe).
The `embed`/`embed_text` fallback at `rag/retriever.py:188-192` and the `embed_texts` fallback at `ingestion/service.py:333-337` go with it, the latter here and the former in #228.
Rationale: R5.5 and the acceptance checklist; `get_max_batch_size` and `modify` exist on chromadb 1.5.9.
S226.6. Ingestion receives the handle and calls `handle.replace_source(...)` and one `handle.commit()` per batch, which bumps the revision and refreshes BM25 once (keeping the O(total chunks) once-per-batch behaviour of `ingestion/service.py:198`).
Rationale: "ingest/rebuild/delete invalidate through the handle only".
S226.7. BM25 is built lazily on first lexical use and rebuilt after `commit()` or `invalidate()`, not eagerly on construction.
Rationale: removes the per-request full scan at `rag/retriever.py:76`.
S226.8. The registry keeps a bounded LRU of handles and evicts a name on delete or rebuild.
`ManagedRetriever.for_collection` and `registry.track` per request go away.
Rationale: R12.1 and F10.
S226.9. `application/repository.py` is deleted; list/delete become registry methods, and `check_readiness` and the collection use cases call them.
Rationale: R1.12, it is an 18-line forward.
S226.10. The query cache key takes `handle.revision` for every retriever type, including plugins.
Rationale: F11, plugins currently get `"0"`.
S226.11. Revision stays an integer in collection metadata bumped under the store's write lock; it is not made cross-process.
Rationale: ADR 035 already limits writers to one process via `ingest_lock`.
S226.12. Do not introduce an alias registry yet.
`CollectionRegistry.open(name)` is the only place a name becomes a Chroma collection, which is the seam #209 needs.
Rationale: R1.8.
S226.13. Integration tests against real Chroma at the handle's interface: embedding-identity rejection for provider, model, dimension, malformed dimension, and a legacy collection with no recorded dimension; revision bump on add, replace, delete-by-source and rebuild; LRU eviction after `delete_collection`.
Delete the unit tests of the absorbed pieces rather than keeping both.
Rationale: acceptance, plus `AGENTS.md` ("anything that can only fail against real Chroma needs an integration test").
S226.14. Amend ADR 035 (handle owns the snapshot and revision; atomic replacement unchanged).
#209 gets its own ADR later.
Update the three architecture docs.

OPEN (product): none.
The upload-retention default that makes rebuild impossible for uploads (F6) is a product decision recorded as Q6 under #213.
Count: 14 settled, 0 open.

## 4. #225 Scope seam

### 4.1 Design tree

- Goal: one filter meaning, applied before ranking on dense and lexical paths, with identity coming from the server side.
  - Branch A, what a Scope is.
    - `Principal`: id plus group names, produced by an adapter from authentication.
      With no identity configured the adapter yields an unrestricted scope.
    - `Scope`: frozen value with `visible_to` (a frozenset of groups, or `None` for unrestricted) and a typed client narrowing.
  - Branch B, one filter language.
    - Client narrowing is an AND of equality clauses on a closed set of string-typed `ChunkField` keys.
    - One compile function produces the Chroma `where`; one produces a Python predicate over chunk metadata.
  - Branch C, lexical path.
    - Mask the BM25 snapshot before top-k (`bm25_index.py:74-103`).
  - Branch D, cache key.
    - Derived from `visible_to`, the narrowing, and `handle.revision`.
  - Branch E, other read paths: parent expansion, `/query/contexts`, MCP search, agent, plugins, audit.

### 4.2 SETTLED (technical)

S225.1. Types live in `localrag/retrieval/scope.py` (the `retrieval` package is created here, and #228 moves the retriever in later).
Rationale: the tracking issue puts Scope inside retrieval, and creating the package with one file avoids a big-bang rename in this PR.
S225.2. `Scope` is frozen, has no setters, and the only constructors are `Scope.unrestricted()` and `Scope.for_principal(principal)`; `scope.narrowed(client_filter)` returns a new Scope with the narrowing ANDed in.
Rationale: "a client filter can narrow a Scope but never widen it" is then structural, not a review comment.
(Contrast Haystack's `FilterPolicy.REPLACE`, which lets a runtime filter replace the init filter: https://docs.haystack.deepset.ai/reference/retrievers-api.)
S225.3. The request DTO carries no group field.
Client filter keys are validated against an explicit set of string-typed stored keys (`file_type`, `chunk_type`, `chunking_strategy`, `source`, `heading_path`, `git_commit`, `content_hash`; `ChunkField` today has only the first and fourth of those, `chunks/record.py:23-27`), excluding the access field; an unknown key is a 400, not a silent empty result.
Rationale: today `{"nonsense": "x"}` quietly returns nothing, and `chunk_index` or `source_mtime` can never equal a string anyway (F3).
S225.4. The access predicate is "chunk's `access_groups` intersects `visible_to`".
Chroma form: `$or` of `$contains` clauses, verified on chromadb 1.5.9 (F2).
Python form: set intersection.
Rationale: household group counts are small, so the `$or` stays short.
S225.5. Compile once from a single normalised clause list into both targets.
A contract suite, parametrised over {Chroma, BM25 predicate}, runs the same corpus and asserts identical id sets, including a corpus where the excluded chunks would outrank the allowed ones.
Rationale: that case is what distinguishes pre-ranking from post-filtering; the missing-key rule (a positive clause never matches a row lacking the key) is pinned by the same suite.
S225.6. `Bm25Index.query` takes the predicate and masks the snapshot before scoring and before the `[:top_k]` slice.
The O(N) substring-boost loop (`bm25_index.py:86-88`) runs only over unmasked rows.
Rationale: this is the #205 bug fix.
S225.7. BM25 IDF and average document length stay corpus-wide.
Rationale: scores are never exposed across groups, only ranks of visible chunks, and the product model is a trusted household.
If untrusted tenancy ever returns, build a per-scope index.
S225.8. Cache key = sorted `visible_to` + normalised narrowing + `handle.revision` + the existing question/model/mode/collection/provider fields.
Two people with the same visible set share entries.
Rationale: the cached answer is a function of the visible chunks only, and keying by person would cut hit rate for no safety gain.
The key function takes a `Scope`, not a dict (`rag/query_cache.py:13-37`).
S225.9. The same compiled scope is passed to parent expansion; `VectorStore.get_chunks_by_headings` loses its own equality re-check (`vector_store.py:344-363`).
Rationale: third copy of the semantics (F3).
S225.10. Apply the scope in `/query/contexts`, MCP `search_documents`, `answer_question`, the agent's `engine.answer`, adaptive retrieval, and request-scoped collections.
`list_distinct_sources`, rebuild and collection delete are administrative and require an admin principal (an assumption confirmed through Q1).
Rationale: destructive and collection-wide operations have no per-group meaning.
S225.11. Plugins: the seam post-checks every plugin result against the scope predicate, drops violations and logs a warning.
Passing Scope into the plugin signature waits for the single contract bump in #228.
Rationale: a foreign index cannot be filtered before ranking, but a cheap predicate on returned metadata is defence in depth.
S225.12. `adaptive.py:221` (`FILTERED_NO_RESULTS if metadata_filter`) becomes `scope.is_narrowed`.
Rationale: with groups, "no result" must not claim a client filter exists when the narrowing came from identity.
S225.13. Audit records gain `principal_id`.
Rationale: a shared deployment needs to know who asked; the log stays metadata-only when configured so.
S225.14. `TENANT_ID` (`settings.py:441-444,507`, `ingestion/service.py:347`) is deprecated: still read, no longer a documented security-adjacent filter.
`docs/rag-retrieval.md` "Tenant tagging" is rewritten around groups.
Rationale: the user chose "group" over tenant, and that section already says it is not a security boundary.
S225.15. Single-user behaviour is unchanged: `Scope.unrestricted()` with no narrowing compiles to `where=None`, and a test asserts the dense call arguments are identical to today's.
S225.16. Land in three slices: (1) #205, BM25 masking with today's equality filter; (2) Scope type, compile functions, contract suite; (3) cache key, plugin post-check, ADR.
Rationale: the first slice is a bug fix that does not depend on any design question.
S225.17. One new ADR, "Groups and Scope" (next free number is 042), covering the trust boundary; #195 amends it.
It touches ADR 038 (adapter to use-case identity handoff) and ADR 040 (request-scoped collection selection).
S225.18. Identity is a `Principal(id, groups)` whatever authenticates it.
How members authenticate is #194's decision; the recommendation for #194 is a per-member key map in the config file, because OIDC does not fit an offline-first single-container deployment.

### 4.3 OPEN (product)

1. ❓ Access assignment: how does a source get the groups that may see it, given that folders change in place and uploads arrive from people?
   - Options: (a) per ingest-root rules in config (everything under `/data/family` is visible to group `family`; new files inherit) plus uploads owned by the uploader's groups with an optional explicit share list; (b) a per-file ACL set through the API for every document; (c) one owning group per source only.
   - ➡️ Recommended: (a), plus one admin principal set in config who alone can rebuild, delete collections and re-stamp access.
     Each source carries a list of groups, not one, because a household has overlapping sets such as "everyone", "parents", and one person's private files.
     Per-file ACLs (b) cannot work for a changing folder where new files appear without anyone calling an API.
     This also resolves the glossary conflict C4 (access is a list, separate from provenance).
2. ❓ Existing and unlabelled data when groups are switched on: who can see chunks that were ingested before any access labels existed?
   - ➡️ Recommended: fail closed.
     Unlabelled chunks are invisible to every restricted principal until an admin assigns them (the positive `$contains` already behaves this way), and a startup warning plus `localrag access audit` lists unlabelled sources.
     The alternative of stamping everything to an "everyone" group makes enabling groups silently publish a private corpus.

Count: 18 settled, 2 open.

## 5. #195 source-level access labels (lighter depth)

### 5.1 Design tree

- Representation: a non-empty list-of-strings chunk metadata key; sentinel `"*"` means everyone, because Chroma rejects empty lists (F2).
- Source of the ACL at ingest: explicit value (upload or CLI flag), else the existing stamp on the source's current chunks, else the root rule, else the default.
- Changing an ACL: metadata-only update through the handle, then revision bump and BM25 refresh.
- Read paths: everything in section 4 plus parent expansion, compression, rerank, HyDE variants, answer cache.

### 5.2 SETTLED (technical)

S195.1. Store `access_groups: list[str]` on every chunk as a `ChunkMetadata` field (C2) and a `ChunkField` member; `"*"` is the public sentinel.
Rationale: Chroma lists must be non-empty and homogeneous, and `$contains` on them works (F2).
S195.2. Re-stamping uses `collection.update(ids, metadatas=[{access_groups: [...]}])`, which merges keys and leaves text and embedding alone (F2).
It runs inside the handle's write lock, bumps the revision, and refreshes the BM25 snapshot, because the snapshot copies metadata.
Rationale: acceptance "re-stamps without re-embedding".
S195.3. Revocation semantics are stated explicitly: after `restamp` returns, any query that starts later sees the new ACL; a query already running may still return pre-revocation chunks (it holds the old BM25 snapshot, ADR 035).
Rationale: R11.1 to R11.4 require the race to be named, and a millisecond window is acceptable for a household.
S195.4. Re-ingesting a changed file must not lose its ACL: `replace_source` deletes the old chunks, so ingestion reads the existing stamp from the source's current chunks before replacing, unless an explicit value or rule applies.
No new source table.
Rationale: R1.8; `rebuild_collection` calls `_ingest_paths_locked` with no ACL input today.
S195.5. Precedence at ingest: explicit value, then existing stamp, then longest-prefix root rule, then the configured default group.
Rationale: deterministic and testable; a re-ingest of an upload keeps its owner's groups.
S195.6. A caller may only assign groups it belongs to; the admin principal (Q1) may assign any.
Rationale: otherwise a member can publish another person's data to a group they are not in.
S195.7. Parent expansion fetches siblings with the compiled scope (S225.9).
Compression, claim filter and rerank only see already-scoped candidates and never re-fetch.
HyDE and expansion variants are query text only and run through the same scoped dense and lexical calls.
The answer cache is keyed by scope (S225.8) and the revision moves on restamp.
Rationale: this is the list in the acceptance criterion; each item has exactly one enforcement point.
S195.8. The embedding cache stores vectors only, with no text or metadata (`docs/architecture.md`, ADR 024), so it is not a leak path.
Audit and trace attributes list only visible sources.
S195.9. Tests use canary text unique to one group and assert absence from contexts, answers, cache entries, audit lines and traces for every retrieval mode; the full matrix is #196, the first cases land here.
S195.10. Add `localrag access set <path> --group ...` and the application use case behind it; no new HTTP route in this ticket beyond the ingest/upload field.
Rationale: keep the public surface small until the product answer in Q1 is confirmed.
S195.11. Amend the #225 ADR rather than writing a second one.

### 5.3 OPEN (product)

3. ❓ Two members upload identical bytes.
   Today identical bytes dedupe into one source (`application/service.py:298-303`).
   Should the second upload be shared with the first uploader's groups, or kept as a separate copy per group?
   - ➡️ Recommended: one stored source whose groups are the union of both uploaders' groups, with revocation per uploader (removing Bob's access removes only Bob's group contribution).
     The second person already holds the bytes, so merging discloses nothing new; per-group copies double embeddings and make a later "who owns this" question ambiguous.

Count: 11 settled, 1 open.

## 6. #227 answer contract and #228 retrieval module

### 6.1 #227 design tree

- Goal: one result type, one recorder, one definition of low confidence.
  - Branch A, entry points.
    There are five, not three: JSON and MCP (`application/service.py:415`, MCP without a cache), SSE (`:593`), CLI `engine.stream_answer` (`cli/commands/query.py`), agent `engine.answer` (`agent/service.py:119`).
    The CLI and agent paths record no metrics, audit or cost today.
  - Branch B, result shape.
    `AnswerResult`: answer, citations, outcome, usage, trace.
  - Branch C, streaming.
    A prepare phase (retrieval) before the stream so errors map to HTTP before SSE starts, which is why the router retrieves first today.
  - Branch D, recorder.
    Called once by the application layer with the final result, including failure and cancellation.
  - Branch E, usage.
    From provider usage, estimated only as a labelled fallback.

### 6.2 #227 SETTLED (technical)

S227.1. `AnswerResult` is a frozen dataclass: `answer`, `citations` (a tuple of a typed `Citation`), `outcome` (an enum: answered, or refused with a reason), `usage` (a typed value), `trace`, `model`, `provider`.
`low_confidence` is a derived property: the outcome is a refusal.
Rationale: today three definitions exist (`application/service.py:466` adaptive uses `not bool(raw_sources)`, the stream path uses the engine flag, and the post-generation `final` always says `False`, `rag/engine.py:207-211`).
S227.2. Streaming yields typed events (`TokenEvent`, `FinalEvent(AnswerResult)`), not `{"type": ...}` dicts.
Sync `answer()` is a fold over the stream, so there is one implementation.
Rationale: R2.4/R2.5, and it deletes the second fork inside `RAGEngine.answer` (`engine.py:41-78`).
S227.3. Two phases: `prepare(question, scope, ...)` runs retrieval (and adaptive retrieval) and may raise; `PreparedAnswer.stream()` generates.
The SSE route calls `prepare` before returning the `EventSourceResponse`.
Rationale: keeps "errors map to HTTP before SSE starts" and fixes F4, because `prepare` always runs the configured retrieval policy.
S227.4. `Citation` carries chunk ID, source, chunk index, heading path, chunk type, score, `cited: bool | None` (None until #213) and an optional span.
The existing `sources` key and `SourceRef` fields stay; new fields are additive and nullable.
Rationale: no breaking API change, and #213 fills the same type.
S227.5. `Usage(input_tokens, output_tokens, origin)` where origin is provider-reported or estimated.
Providers return it on the final event: Ollama from `prompt_eval_count`/`eval_count` (parse them in `OllamaChatStreamChunk`), OpenAI by requesting `include_usage` and tolerating the `choices: []` chunk, Anthropic from the final message's usage.
If a stream ends without usage, fall back to `count_tokens` and mark it estimated.
Rationale: F5; a metric named tokens must count tokens.
(How the OpenAI SDK's `chat.completions.stream` helper surfaces usage should be checked against the pinned SDK before coding; I did not verify that.)
S227.6. The metric name `tokens_used_total` stays and counts provider tokens; add a `kind` label (input, output) rather than renaming.
Rationale: splitting input from output is what #217 needs for cost.
The provisioned Grafana dashboard `infra/grafana/provisioning/dashboards/localrag.json` references the metric, so its panels must be changed to `sum by (provider)` in the same PR, and the new label must be documented in `docs/observability.md`.
S227.7. One `QueryRecorder` in an operations module takes an `AnswerResult` plus transport and request metadata and writes metrics, audit and cost once.
The application layer calls it exactly once per query, in a `finally`, with outcome answered, refused, failed or cancelled.
Rationale: removes the three copies of the audit block (`:479-492`, `:552-565`, `:631-644`) and records client disconnects, which no path does today.
S227.8. The CLI and agent route through the same application use case, so they are recorded too.
This is a visible change only when `AUDIT_LOG_PATH` is set.
Rationale: ADR 038 says the CLI uses the application container.
S227.9. Replace the three blanket `except Exception` (`application/service.py:455,524,647`) with a domain `GenerationError` raised by the provider layer, chained with `from exc` instead of `from None` (`:527`); the HTTP detail text stays "LLM provider request failed." Rationale: R7.1 is CRITICAL and R7.2 requires chaining.
S227.10. The query cache stays in the application layer, stores a serialised `AnswerResult`, and keeps today's coverage (JSON and MCP yes, SSE no).
Making SSE cache-aware is a follow-up.
Rationale: limits the blast radius of this PR.
S227.11. Keep the public `trace` JSON shape; internally the trace is typed (`AnswerTrace` with optional HyDE, claim-filter and adaptive observations) and serialised by `to_dict()`.
Rationale: `QueryResponse.trace` is public.
S227.12. #227 may land before #228: it consumes the #223 `RetrievalContext` TypedDict, and #228 later swaps in typed retrieved chunks and removes `last_hyde`.
S227.13. Interface test: the same input and fake provider through JSON and SSE produce identical audit records, metric increments and confidence.
`query_json` loses both complexity `noqa`s because the three bodies collapse into one.
S227.14. New ADR "Answer contract" (next free number after the Scope ADR); it touches ADR 038.
Update the three architecture docs.

OPEN (product): none.
Count: 14 settled, 0 open.

### 6.3 #228 design tree

- Goal: `retrieve(question, scope, n) -> RetrievalResult(chunks, trace)` as the only way into retrieval.
  - Branch A, internal structure of the 490-line `retriever.py`: planning, dense, fusion, freshness, parents.
  - Branch B, the retrieved-chunk type.
  - Branch C, the trace.
  - Branch D, the plugin contract.
  - Branch E, adaptive policy and request-scoped collections.

### 6.4 #228 SETTLED (technical)

S228.1. Structure: `retrieval/planning.py` (query plan, rewrite, HyDE, expansion), `retrieval/fusion.py` (RRF, recency list, freshness), `retrieval/retriever.py` (dense and lexical orchestration, rerank, parent expansion).
Three modules, not five.
Rationale: R1.5 asks for cohesive units, R1.8 warns against generalising early, and fusion and freshness share the timestamp logic.
S228.2. Introduce `RetrievedChunk` (frozen): the stored chunk (text plus typed `ChunkMetadata`), `score`, optional `distance`, optional `freshness_factor`, optional expanded section.
`ingested_at` is parsed to a `datetime` once, in the metadata codec's tolerant read.
Rationale: that deletes the four `_parse_ingested_at` calls (`rag/retriever.py:350,372,391,427`); the NodeWithScore wrapper pattern is the same separation LlamaIndex uses (research note §1).
S228.3. `RetrievalContext` stays the plugin and wire TypedDict (settled in #223); `RetrievedChunk.to_context()` produces it at the boundary.
This fills the gap in C6.
S228.4. Fusion and adaptive de-duplication key on `chunk_id`, not `(source, chunk_index)` (`rag/retriever.py:442-446`, `adaptive.py:99-100`).
Rationale: after #206 the index is ordering metadata only.
S228.5. `RetrievalTrace` is returned, never stored on the retriever: per-stage candidate IDs, scores and durations (bounded by `_HARD_MAX_CANDIDATES = 100`), plus rewrite, HyDE and expansion observations.
This removes `last_hyde` (F9, R11.2).
The API response keeps today's `trace` content; the full trace goes to the recorder and OpenTelemetry only.
Rationale: stage candidate lists would bloat every response.
S228.6. The seam rule is reworded as "nothing reaches past the retrieval seam" (C8): application use cases that are retrieval-only (MCP search, `/query/contexts`) call `retrieval.retrieve`, answers call it through generation.
S228.7. Adaptive policy moves under retrieval as a strategy selected inside `retrieve`; the engine's two adaptive branches (`rag/engine.py:48-68,93-112`) are deleted with #227.
S228.8. Retrieval is stateless per call: it receives a `CollectionHandle` (#226) per call.
`Retriever.for_collection`, `RAGEngine.for_collection` and the per-request `ManagedRetriever` disappear.
A request-scoped collection with a third-party plugin returns an explicit `BAD_REQUEST` instead of today's `TypeError`.
S228.9. Plugin contract version 2.0: `retrieve(question, scope, n_results)`, `RetrievalContext` unchanged.
Validation is fixed to honour `compatible_contract_versions` rather than exact equality (F11).
The example plugin and `docs/plugin-author-guide.md` are updated.
No 1.x shim.
Rationale: plugins are trusted, pinned and few; a shim would carry the metadata-filter dict forever.
S228.10. `plugins/retriever.py` imports neither `application` nor `rag.retriever` (S224.6 supplies the builtin factory).
S228.11. `ManagedRetriever.__getattr__` (`plugins/retriever.py:210`) and the `isinstance(engine.retriever, Retriever)` reach-through (`application/service.py:389`) are deleted.
Callers get what they need from the handle or the trace.
S228.12. `RetrievalError` raise sites change to the kind enum (S229.5); the legacy `embed_text` fallback and `getattr(embedder, "provider_name")` logging probes at `rag/retriever.py:176-177,188-192` are removed (the embedding base class already defines these attributes).
S228.13. Behaviour-preserving gate: rank order on the bundled eval fixtures is identical before and after (run the existing `task benchmark` smoke or the deterministic retrieval metrics), because fusion and freshness are being moved, not changed.
S228.14. Tests that reach into `_fuse_rank_lists`, `_QueryPlan`, `_bm25_hits`, `_expand_to_parent_section` or `apply_freshness` (`tests/test_retriever_hybrid.py`, `tests/test_freshness_decay.py`) are rewritten as interface tests and the old ones deleted.

OPEN (product): none.
Count: 14 settled, 0 open.

## 7. #229 shallow modules and import cycles

### 7.1 Design tree

- Layering rule: embedding and llm are lower layers than ingestion and rag; storage and indexing are lower than retrieval; application and adapters are top.
- Fix the three cycles in the ticket, the extra edges from F12, add a ratchet test, and delete dead code.

### 7.2 SETTLED (technical)

S229.1. Split into #229a (do now: moves, deletions, error enum, layering test) and nothing deferred, since the remaining ticket text is covered by #226 and #228.
Rationale: it is pure noise removal and should not wait on design questions.
S229.2. `OllamaEmbedder` moves to `embedding/ollama.py`, next to `embedding/sentence_transformers.py`, not into indexing.
Rationale: embedding is the lower layer, and `embedding/factory.py:6` is the only reason it sits in ingestion.
S229.3. `prompt.py` does not move to generation.
`build_prompt` and `MAX_SECTION_CHARS` go to `llm/prompt.py`, and `DEFAULT_SYSTEM_PROMPT` goes to a dependency-free `localrag/system_prompt.py` (the canonical name in R1.3/R1.10).
Rationale: the providers' own `stream`/`generate` build the context-list prompt (`llm/providers/ollama.py:87`), HyDE, query rewrite and adaptive call `provider.generate(..., context=[])` (`rag/hyde.py:72`, `rag/query_rewrite.py:88,127`, `rag/adaptive.py:185,290`), and generation imports `llm`, so a move to generation keeps the llm to generation to llm cycle.
`settings_groups.py:18` must not import a package that imports settings, which `llm` does (`llm/factory.py`), so the system prompt needs a leaf module.
S229.4. Do not remove the provider context-list methods in this ticket.
Doing so would change the prompt HyDE and rewrite send (`generate(context=[])` wraps the prompt in the provider template, and `rag/hyde.py:64` already overrides the system prompt via `settings.model_copy(update=...)` to compensate).
That is an eval-affecting change; file it as a separate issue with a benchmark run.
Rationale: this ticket must stay behaviour-preserving.
(Whether the QA-template wrapping of HyDE prompts hurts quality is unmeasured.)
S229.5. `RetrievalError` keeps the domain enum `RetrievalFailureKind` (`rag/exceptions.py:6-8`) and drops `status_code`; the application maps `kind` to `ApplicationErrorKind` (`application/service.py:572-575`).
It does not import `ApplicationErrorKind` into `rag`.
Rationale: the ticket's wording would make `rag` import `application`, inverting ADR 038; both enums already share the same string values.
S229.6. `bm25_index.py` moves to `indexing/bm25.py` as a pure rename here, before #226 (C10).
Rationale: one move, and it removes `ingestion` to `rag`.
S229.7. Merge `code.py`, `markdown.py`, `text.py` and `anydoc.py` into one `parsers/simple.py`; keep `pdf.py` (OCR, 164 lines) separate.
Delete `parsers/docx.py` and its test, and drop `python-docx` from `pyproject.toml` if nothing else imports it.
Loader routing is unchanged.
Rationale: F12 shows `docx.py` is dead in production, and the ticket's "parsers in one module" cannot sensibly include the OCR parser.
S229.8. Add a layering test in the style of `tests/test_application_boundary.py`: an AST scan with a ratchet allowlist of known violations that may only shrink.
Start with the edges this PR fixes forbidden outright.
Rationale: no new dependency (import-linter would be a new dev dependency for 30 lines of test), and it keeps the fix from regressing.
S229.9. Edges left to their tickets and recorded in the allowlist: `storage` to `embedding` (#226), `plugins` to `application` (#224), `rag` to `llm` (legitimate: HyDE, rewrite, claim filter and adaptive call providers).
S229.10. Update `docs/architecture.md`, `docs/agent-navigation.md`, the AGENTS.md package map, and `docs/document-formats.md` (parser module names only; routing stays true).
Update test imports.

OPEN (product): none.
Count: 10 settled, 0 open.

## 8. #206 content-addressed chunk IDs (lighter depth)

### 8.1 Design tree

- Identity recipe: source, strategy, text, occurrence counter, contract version.
- Diff at ingest: fetch existing IDs and metadata for the source, embed only new IDs, update metadata for changed ones, delete removed ones.
- Migration: forced by a stored contract version (C3).

### 8.2 SETTLED (technical)

S206.1. Hash the exact chunk text, not a normalised form.
The ticket says "normalized text"; I recommend against it.
Rationale: the optimisation "same ID, skip re-embedding" is only safe if equal ID implies equal text; chunkers already strip whitespace (`chunks/fixed.py:9,22`).
S206.2. The occurrence counter is the number of earlier chunks in the same source with identical text.
Inserting an earlier duplicate shifts later duplicates' IDs; document that as accepted.
S206.3. `chunk_index` stays as ordering metadata.
When an insertion renumbers later chunks their IDs are unchanged, so the diff must also compare metadata: ids with changed metadata get a metadata-only `update`, new IDs get an `upsert` with embeddings, removed IDs are deleted.
S206.4. The diff happens before embedding.
Today `_ingest_one_stages` embeds every chunk (`ingestion/service.py:303-337`) and the embedding cache is off by default, so "upsert only changed" is not real until embeddings are skipped for unchanged IDs.
The three-way write is still one atomic source replacement with rollback (ADR 035): the captured old state in `replace_source` already covers update, add and delete.
S206.5. Stability expectations are set honestly.
`structural` should be stable outside the edited section, because the Markdown path flushes a section at each heading (read from `chunks/structural.py`, to be confirmed by the insertion test).
`fixed` windows and overlapped `recursive` windows shift on any upstream insertion, so most of their text changes and most of their IDs change anyway.
Tests use the `structural` strategy for insertion, deletion and duplicate-paragraph cases; `fixed` and `recursive` get a test that documents their instability.
S206.6. Bump `CHUNK_CONTRACT_VERSION` to 2 and store it per chunk (C3).
Rebuild treats a stored version below current as changed, even when `content_hash` matches.
The ADR 021 amendment documents the rebuild.
CONTEXT.md "Chunk ID" is rewritten.
S206.7. Because `replace_source` deletes IDs that are not in the new set, a re-ingested source loses its version-1 IDs automatically; only sources never re-ingested keep them.
S206.8. Bundle with the span fields (section 1.3) so there is one forced rebuild.
S206.9. The occurrence counter needs the earlier chunks of the same source, so `chunk_source` (`chunks/strategies.py`) passes it into `Chunk.place`; `place` stays the only place an ID is derived.

OPEN (product):
4. ❓ After an edit, should the unchanged paragraphs of a file count as "recent" for freshness ranking?
   `ingested_at` (`ingestion/service.py:343`) drives freshness decay and the recency list in fusion.
   - ➡️ Recommended: no.
     Unchanged chunks keep their original `ingested_at` (only new or changed text gets a new one), so recency means "when this text first entered the index".
     File modification time is a poor signal for synced folders that rewrite files on sync.

Count: 9 settled, 1 open.

## 9. #207 contextual enrichment (lighter depth)

### 9.1 Design tree

- Where the embedded text and the lexical text come from; where the title comes from; how the format is versioned; how the default is chosen.

### 9.2 SETTLED (technical)

S207.1. Introduce one `contextual_text(chunk)` function in the chunks package that builds the enriched string from the typed metadata (title, heading path, text) and is used both at ingest for embedding (`ingestion/service.py:303`, currently the same list is embedded and stored) and when building BM25 (`indexing/bm25.py` refresh, which today uses the stored text).
Rationale: the prefix is derivable from metadata, so no second copy of the text is stored.
S207.2. The stored text stays unprefixed (it is what citations and the prompt show).
S207.3. `document_title` is derived in the chunking stage: first H1, else the filename stem.
Frontmatter titles are not available because `parsers/markdown.py` strips the frontmatter; that belongs to #202.
The prefix is capped at `MAX_SECTION_CHARS` (120, `rag/prompt.py:33`) so it cannot crowd the chunk body.
S207.4. The enrichment format is a versioned string recorded in the collection's embedding identity (the #226 handle), so an enriched and a non-enriched collection cannot be mixed.
The embedding cache key already includes the exact input text (`embedding/cache.py:61`), so enriched and plain vectors cannot collide; the ticket's "part of the cache key" is satisfied by construction and needs one test.
S207.5. Add an `enrichment` dimension to `SUPPORTED_DIMENSIONS` in `evals/matrix.py:27`.
The default stays off unless benchmarks show a clear recall@k gain with no MRR loss; the bundled datasets are small, so a marginal difference is not evidence.
Flipping the default is a behaviour-changing default and needs an ADR per AGENTS.md.
S207.6. Queries are not prefixed.
The cross-encoder reranker keeps seeing stored text.
S207.7. Depends on #223 and the #226 identity; schedule after #206 so the forced rebuild is shared (section 1.3).

OPEN (product): none.
Count: 7 settled, 0 open.

## 10. #213 verifiable citations (lighter depth)

### 10.1 Design tree

- Marker parsing and validation.
- Mapping `[n]` to the right list (F8).
- Span derivation (the user's requirement: exact span, not file or section).
- Delivery over JSON, SSE and MCP; eval metric.

### 10.2 SETTLED (technical)

S213.1. Number markers against the prompt list (the post-compression contexts), and build the citation list from that same list, not from `claim_filter.contexts` (F8).
The mapping `index -> chunk_id` is created at the point `build_prompt` is called.
S213.2. Parser accepts `[3]`, `[1][2]` and `[1, 2]`; anything out of range lands in `invalid_markers`.
Ranges such as `[1-3]` are not accepted.
Rationale: the ticket asks for short prompt wording for small models, and the parser should be strict about what it counts as a valid citation.
S213.3. Prompt instruction is one added sentence in the system prompt module ("Cite the context blocks you use as [n].").
A user-overridden `rag_system_prompt` is still validated; it simply yields more uncited answers.
S213.4. Response fields (additive): per source `chunk_id`, `score`, `cited`, and an optional `span`; top level `invalid_markers` and a `citation_status` of cited, uncited or invalid.
Delivered on JSON, in the SSE `final` event, and in the MCP `answer_question` dict.
Refusals carry no citations.
S213.5. Span is derived, not requested from the model.
The prompt text of each block is extractive (`rag/compressor.py`, "extractive-v1", always on), so the sentences the model saw are exact substrings of the section text.
With parent expansion on by default (F8) the section is `"\n\n".join` of sibling chunks, so parent expansion must return its segments `(chunk_id, text)` and the evidence sentences map to the right chunk by position in that deterministic join.
Small local models cannot be trusted to quote.
S213.6. The citation anchor follows the W3C Web Annotation pattern: an exact quote with prefix and suffix, plus a position hint (start and end in parsed text) and the source's `content_hash` (https://www.w3.org/TR/annotation-model/#text-quote-selector). Position alone is not enough because of F7: chunk text is not always a slice of the source, and parsed text is not the original file.
Resolution searches the quote inside the chunk's covered range; if the stored `content_hash` no longer matches the file, mark the citation as stale and fall back to quote search.
Anthropic's own Citations API reports `char_location`, `page_location` and `content_block_location` with `cited_text`, the same three-way split by source type (https://platform.claude.com/docs/en/build-with-claude/citations).
S213.7. Chunkers record the covered source range while splitting (C1).
`parse_file` must also keep the number of leading characters it strips, so parsed-text offsets can be mapped to file offsets for plain text, Markdown and code.
This is a follow-up inside this ticket, not part of #229.
S213.8. PDF files get quote plus page, not a file offset: `parse_pdf` already holds per-page Markdown before it joins the pages with newlines (`parsers/pdf.py:67,78`), so recording each page's start offset in the parsed text is a parser change, not a new capability.
Office files go through `anydoc`, which returns Markdown only, so they get quote only unless a page map is shown to exist (not verified).
S213.9. Eval: add `citation_validity_rate` (valid markers divided by all markers) to the deterministic metrics in `evals/metrics.py`; the existing annotation-backed `score_citation_accuracy` (`evals/metrics.py:101`) is a different metric and keeps its name.
S213.10. A bad marker never changes the answer text: the API returns the model's text untouched, sets `citation_status` to invalid, and lists the marker in `invalid_markers`.
It does not refuse or rewrite the answer.
Rationale: the repository has no UI, so rendering is a client decision, and silently editing model output would hide the failure the ticket wants visible.
S213.11. Dependencies: #227 (the `Citation` type and recorder), #223 (chunk IDs, optional span fields), #228 (parent segments), and soft on #206 (stable IDs keep stored references valid across edits).

### 10.3 OPEN (product)

5. ❓ What precision does a citation promise per file type?
   - Plain text, Markdown and code can point at an exact character range in the file.
   - PDF files can point at a quoted snippet plus a page.
   - Office files and anything else converted by `anydoc` can only point at a quoted snippet inside the extracted text.
   - ➡️ Recommended: accept these three tiers and say so in the docs.
   - When the file has changed since indexing, show the stored quote with its surrounding text and mark the citation stale.
   - Promising an exact range for every format would require storing page and layout maps for all of them, which is a large parser project for little household value.
6. ❓ Should uploaded originals be kept by default?
   - Today they are deleted after ingest (`UPLOAD_RETENTION_SECONDS=0`, F6), so citations cannot open the original and `rebuild` deletes the source as "missing".
   - ➡️ Recommended: keep originals by default, with the existing quota and an opt-out.
   - Rebuild, citation resolution and incremental sync (#201) all need the original, and a household server has the disk.
   - The privacy cost (files kept on the server) is the thing the user must accept.

Count: 11 settled, 2 open.

## 11. Summary table

| Ticket | Settled | Open | Notes |
|---|---|---|---|
| #224 | 12 | 0 | Fixes a live stale-BM25 bug (F1) |
| #225 | 18 | 2 (Q1, Q2) | Q1 also decides the glossary conflict C4 |
| #226 | 14 | 0 | Upload retention (Q6) lives under #213 |
| #227 | 14 | 0 | Fixes SSE with adaptive (F4) |
| #228 | 14 | 0 | Plugin contract goes to 2.0 |
| #229 | 10 | 0 | Two deviations from the ticket text (S229.3, S229.5) |
| #206 | 9 | 1 (Q4) | Normalisation deviation (S206.1) |
| #207 | 7 | 0 | Default decided by benchmark |
| #213 | 11 | 2 (Q5, Q6) | Needs offsets from #223 (C1) |
| #195 | 11 | 1 (Q3) | Depends on Q1 and Q2 |

## 12. Gaps in the landed #223 record (86684a8)

#223 landed while this document was being written, so these are follow-up changes rather than review comments on an open branch.
G1. Add an optional covered source range to `ChunkDraft`, `Chunk` and `ChunkMetadata`, omitted from the stored row when absent (C1).
G2. Widen `StoredValue` and the codec for optional list-of-string fields, and add an `access_groups` field so a metadata update is not lost by a read and rewrite through `ChunkMetadata` (C2).
G3. Add a stored `chunk_contract_version` (absent on legacy rows, read as 1), and make rebuild treat a lower version as changed (C3).
G4. Decide who owns the internal retrieved-chunk type: the record, or #228 (C6).
G5. Move the group out of `SourceProvenance` once Q1 is answered (C4).
G6. `tests/test_stored_chunk_format.py` pins the stored bytes per strategy; any new stored key must be optional so that file keeps passing for legacy rows.
