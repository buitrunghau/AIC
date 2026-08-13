#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-.}"

MODULES=(
  "phase1_pipeline"
  "phase2_metadata"
  "phase3_indexing"
  "phase4_retrieval"
  "phase5_qa_ui"
  "shared_contracts"
)

for module in "${MODULES[@]}"; do
  mkdir -p "${PROJECT_ROOT}/${module}"
  touch "${PROJECT_ROOT}/${module}/__init__.py"
done

cat > "${PROJECT_ROOT}/phase1_pipeline/transnet_segmentation.py" <<'PY'
"""Phase 1: Video shot detection placeholders."""
PY

cat > "${PROJECT_ROOT}/phase2_metadata/multimodal_extractor.py" <<'PY'
"""Phase 2: Multimodal metadata extraction placeholders."""
PY

cat > "${PROJECT_ROOT}/phase3_indexing/milvus_indexer.py" <<'PY'
"""Phase 3: Dense/sparse indexing placeholders."""
PY

cat > "${PROJECT_ROOT}/phase4_retrieval/hybrid_retriever.py" <<'PY'
"""Phase 4: Hybrid retrieval placeholders."""
PY

cat > "${PROJECT_ROOT}/phase5_qa_ui/streamlit_app.py" <<'PY'
"""Phase 5: QA/UI placeholders."""
PY

cat > "${PROJECT_ROOT}/shared_contracts/contracts.py" <<'PY'
"""Shared dataclasses and typing contracts placeholders."""
PY

mkdir -p "${PROJECT_ROOT}/tests"

cat > "${PROJECT_ROOT}/tests/test_phase1_pipeline.py" <<'PY'
"""Placeholder test module for phase 1."""


def test_phase1_placeholder() -> None:
    assert True
PY

cat > "${PROJECT_ROOT}/tests/test_phase2_metadata.py" <<'PY'
"""Placeholder test module for phase 2."""


def test_phase2_placeholder() -> None:
    assert True
PY

cat > "${PROJECT_ROOT}/tests/test_phase3_indexing.py" <<'PY'
"""Placeholder test module for phase 3."""


def test_phase3_placeholder() -> None:
    assert True
PY

cat > "${PROJECT_ROOT}/tests/test_phase4_retrieval.py" <<'PY'
"""Placeholder test module for phase 4."""


def test_phase4_placeholder() -> None:
    assert True
PY

cat > "${PROJECT_ROOT}/tests/test_phase5_qa_ui.py" <<'PY'
"""Placeholder test module for phase 5."""


def test_phase5_placeholder() -> None:
    assert True
PY

echo "Project scaffolding created at ${PROJECT_ROOT}"
