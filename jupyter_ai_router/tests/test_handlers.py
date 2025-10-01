"""
Tests for extension handlers (currently none).

This extension provides server-side functionality only
and does not expose HTTP endpoints.
"""

def test_no_handlers():
    """Test that extension loads without HTTP handlers."""
    from jupyter_ai_router.extension import RouterExtension
    
    ext = RouterExtension()
    assert ext.handlers == []