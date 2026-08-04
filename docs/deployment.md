# Kubernetes Deployment

The checked-in manifests describe a single-node deployment. Apply `pvc.yaml`,
`configmap.yaml`, `secret.yaml`, `deployment.yaml`, and `service.yaml` in that
order. Do not apply `hpa.yaml`: LocalRAG has an in-process job registry and an
embedded, file-backed Chroma store, so more than one API replica would split
jobs and diverge from the active vector-store state.

## Persistence contract

`localrag-api` stores Chroma under `/app/data/chroma` on the `localrag-data`
PVC. The claim is `ReadWriteOnce` and the deployment has exactly one replica.
This is durable across pod replacement on a node, but is not multi-node HA,
multi-replica storage, or a backup policy. Back up the PVC before upgrades and
keep the Chroma data format compatible with the application image.

## Dependency endpoints and readiness

The deployment expects Ollama to be provided separately at
`http://ollama.default.svc.cluster.local:11434`, configured by
`OLLAMA_BASE_URL`. Its service must expose port `11434` and its readiness
should represent a responsive Ollama API plus any model-pull initialization.
The API readiness probe checks Ollama `GET /api/tags`.

Chroma is embedded in the API process; there is no Chroma Kubernetes Service in
this repository. The API readiness check opens the local Chroma store through
the application repository, so the PVC must be mounted and writable before the
pod becomes ready. If Chroma is later moved to an HTTP service, add that
endpoint to configuration and make its health check an explicit readiness
dependency before enabling multiple replicas.

`/health` is liveness only and returns `200` while the process can serve HTTP.
`/ready` returns `200` only when required dependencies are available and `503`
otherwise. Both responses are deliberately minimal and do not expose paths,
collection names, URLs, or exception details.

## Image, security, and resources

Deployment images must use a release tag that CI treats as immutable, or a
registry digest. Never use `latest`; update the pinned image as part of a
reviewed release. The example uses `localrag-api:0.1.0` as the release-tag
shape and must be replaced with the registry-qualified release image.

The pod runs as non-root with the RuntimeDefault seccomp profile, drops Linux
capabilities, disallows privilege escalation, and uses a read-only root
filesystem. Writable application data is limited to the PVC and `/tmp`.
Requests and limits in `deployment.yaml` are intentionally conservative
starting values; tune them from observed Ollama/model memory and CPU usage,
without removing requests or limits.
