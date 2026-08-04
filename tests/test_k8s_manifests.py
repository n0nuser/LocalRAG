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
    assert container["image"] == "localrag-api:0.1.0"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert spec["volumes"][0]["persistentVolumeClaim"]["claimName"] == "localrag-data"
    assert container["securityContext"]["allowPrivilegeEscalation"] is False


def test_pvc_is_single_writer() -> None:
    pvc = _read("pvc.yaml")
    assert pvc["spec"]["accessModes"] == ["ReadWriteOnce"]


def test_hpa_is_constrained_to_one_replica() -> None:
    hpa = _read("hpa.yaml")
    assert hpa["spec"]["minReplicas"] == 1
    assert hpa["spec"]["maxReplicas"] == 1
