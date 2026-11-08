"""Contract fuzz of api/index.py from its own OpenAPI schema (Schemathesis).

Every generated request to /health, /ready and /predict must get a response
that matches the schema, and none may be a 5xx except the documented 503s.
Runs against the tiny stand-in model, so it needs no release asset.
"""

from __future__ import annotations

import pytest

schemathesis = pytest.importorskip("schemathesis")
pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, settings  # noqa: E402
from schemathesis.specs.openapi import checks as oas_checks  # noqa: E402

from conftest import _load_api_module, write_standin_model  # noqa: E402


@pytest.fixture(scope="module")
def contract_app(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("VEMO_MODEL_DIR", str(write_standin_model(tmp_path_factory.mktemp("contract_model"))))
    mp.setenv("RATE_LIMIT_PER_MIN", "100000")  # fuzzing is not abuse; 429 is tested in test_api
    for var in ("MODEL_SHA256", "VERCEL", "ALLOWED_ORIGINS"):
        mp.delenv(var, raising=False)
    module = _load_api_module()
    module.reset_model_cache()
    yield module.create_app()
    module.reset_model_cache()
    mp.undo()


schema = schemathesis.pytest.from_fixture("contract_schema")


@pytest.fixture(scope="module")
def contract_schema(contract_app):
    return schemathesis.openapi.from_asgi("/openapi.json", contract_app)


def test_schema_covers_both_endpoints(contract_app):
    from fastapi.testclient import TestClient

    paths = TestClient(contract_app).get("/openapi.json").json()["paths"]
    assert {"/predict", "/health", "/ready"} <= set(paths)
    body = paths["/predict"]["post"]["requestBody"]["content"]
    assert "application/octet-stream" in body


def not_a_server_error_except_503(ctx, response, case):
    if response.status_code >= 500 and response.status_code != 503:
        raise AssertionError(f"{case.method} {case.path} returned {response.status_code}")
    if response.status_code == 503 and case.path not in ("/ready", "/predict"):
        raise AssertionError(f"unexpected 503 from {case.path}")


@schema.parametrize()
@settings(max_examples=40, deadline=None, suppress_health_check=list(HealthCheck))
def test_api_honours_its_schema(case):
    # Arbitrary bytes are valid per the schema (format: binary) but the body must be
    # exactly 192,000 bytes, so "positive data rejected" is expected and excluded.
    case.call_and_validate(
        excluded_checks=[oas_checks.positive_data_acceptance],
        additional_checks=[not_a_server_error_except_503],
    )
