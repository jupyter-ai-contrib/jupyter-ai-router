"""
Tests for extension handlers (currently none).

This extension provides server-side functionality only
and does not expose HTTP endpoints.
"""

import json

async def test_get_health(jp_fetch):
    # When
    response = await jp_fetch("jupyter-ai-router", "health")

    # Then
    assert response.code == 200
    payload = json.loads(response.body)
    assert payload == {
        "data": "JupyterLab extension @jupyter-ai/router is activated!"
    }