# Agent Friction Notes

This is a short record of repository traps discovered while resolving and
verifying changes. Keep entries concrete and delete them only when the
underlying workflow or contract changes.

## Merge Conflicts

PR #169 conflicted with the progress-output work because both branches changed
the same CLI ingest section, ingest entry point, and ingestion-service tests.
The correct resolution was additive: keep progress callbacks and wrap the whole
write path in the cross-process ingest lock. Rebase the PR branch onto current
`main` before reconstructing tests; resolving only the documentation conflict
leaves the test file structurally broken.

## Docker Drift

An already-running Compose stack serves the old image even when the source tree
has changed. Rebuild before integration tests. The image now carries
`LOCALRAG_BUILD_SHA`; `task docker-up` stamps the current revision and
`task docker-check` compares it through the authenticated `/build-info`
endpoint.

## Chroma Lifecycle

Deleting a collection removes Chroma metadata but can leave its persisted HNSW
directory. Resolve the vector segment ID before deletion, remove only that
directory after successful metadata deletion, and invalidate cached retrieval
objects. Otherwise the next query in the same long-running API can use the
deleted collection UUID and return a misleading 503.

## Integration Test Assumptions

Integration tests must use the stack's configured default LLM model rather than
hardcoding a model tag: local Compose setup and CI intentionally provision
different models. Run the full integration suite only after rebuilding the
image and starting the stack, using `LOCALRAG_TEST_API_KEY` for protected
endpoints.
