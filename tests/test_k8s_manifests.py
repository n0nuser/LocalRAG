from __future__ import annotations

from pathlib import Path

import yaml

K8S = Path(__file__).parents[1] / "k8s"


def _read(name: str) -> dict[str, object]:
    value = yaml.safe_load((K8S / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_deployment_uses_pvc_security_and_split_probes() -> None:
    deployment = _read("deployment.yaml")
    spec = deployment["spec"]["template"]["spec"]
    container = spec["containers"][0]
    assert container["image"] == "ghcr.io/n0nuser/localrag-api:0.1.0"
    assert ":latest" not in container["image"]
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert spec["volumes"][0]["persistentVolumeClaim"]["claimName"] == "localrag-data"
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert spec["securityContext"]["runAsNonRoot"] is True


def test_pvc_is_single_writer() -> None:
    pvc = _read("pvc.yaml")
    assert pvc["spec"]["accessModes"] == ["ReadWriteOnce"]


def test_hpa_is_constrained_to_one_replica() -> None:
    hpa = _read("hpa.yaml")
    assert hpa["spec"]["minReplicas"] == 1
    assert hpa["spec"]["maxReplicas"] == 1


def test_ollama_service_matches_configured_dependency_endpoint() -> None:
    service = _read("ollama-service.yaml")
    port = service["spec"]["ports"][0]
    assert service["metadata"]["name"] == "ollama"
    assert port["port"] == 11434
    assert port["targetPort"] == 11434
