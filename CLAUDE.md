# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `jupyter_ai_router`, a core message routing layer for Jupyter AI. It's a JupyterLab extension that provides foundational message routing functionality, automatically detecting new chat sessions and routing messages to registered callbacks based on message type (slash commands vs regular messages).

## Architecture

### Core Components

- **MessageRouter** (`jupyter_ai_router/router.py`): Central routing class that manages chat connections and message callbacks
- **RouterExtension** (`jupyter_ai_router/extension.py`): Jupyter server extension that initializes the router and listens for chat events
- **Frontend Plugin** (`src/index.ts`): JupyterLab frontend extension for client-side integration
- **API Handler** (`jupyter_ai_router/handlers.py`): Simple health check endpoint

### Key Dependencies

- `jupyterlab-chat>=0.17.0`: Core chat functionality and YChat document handling
- `jupyter-collaboration>=4.0.0`: Real-time collaboration features and event system
- `pycrdt`: CRDT (Conflict-free Replicated Data Type) for message handling

### Message Flow

1. RouterExtension listens for `jupyter_collaboration` chat room initialization events
2. When new chat detected, retrieves YChat document and connects to MessageRouter
3. MessageRouter observes YChat message streams and routes to registered callbacks
4. Messages are classified as slash commands (starting with `/`) or regular messages
5. Appropriate callbacks are invoked with `(room_id, message)` parameters

## Development Commands

### Initial Setup
```bash
# Install development dependencies
jlpm

# Install Python package in development mode
pip install -e ".[test]"

# Link extension to JupyterLab
jupyter labextension develop . --overwrite

# Enable server extension
jupyter server extension enable jupyter_ai_router
```

### Building
```bash
# Build TypeScript and JupyterLab extension (development)
jlpm build

# Production build
jlpm build:prod

# Watch mode (rebuilds on changes)
jlpm watch
```

### Testing
```bash
# Run Python tests with coverage
pytest -vv -r ap --cov jupyter_ai_router

# Run JavaScript/TypeScript tests
jlpm test

# Run integration tests (Playwright)
cd ui-tests && jlpm test
```

### Code Quality
```bash
# Run all linting and formatting
jlpm lint

# Check code style without fixing
jlpm lint:check

# Individual tools
jlpm eslint          # Fix TypeScript/JavaScript issues
jlpm prettier       # Fix formatting
jlpm stylelint       # Fix CSS issues
```

### Development Workflow
```bash
# Clean build artifacts
jlpm clean:all

# Full development setup from scratch
jlpm dev:install

# Uninstall development setup
jlpm dev:uninstall
```

## Code Style

### TypeScript/JavaScript
- Single quotes for strings
- No trailing commas
- Arrow functions preferred
- Interface names must start with `I` and be PascalCase
- 2-space indentation

### Python
- Follow standard Python conventions
- Use type hints where appropriate
- Inherit from `LoggingConfigurable` for components that need logging

## Extension Integration

Other Jupyter extensions can access the router via settings:

```python
router = self.serverapp.web_app.settings.get("jupyter-ai", {}).get("router")

# Register callbacks
def on_new_chat(room_id: str, ychat: YChat):
    print(f"New chat: {room_id}")

def on_slash_command(room_id: str, message: Message):
    print(f"Slash command: {message.body}")

router.observe_chat_init(on_new_chat)
router.observe_slash_cmd_msg(room_id, on_slash_command)
router.observe_chat_msg(room_id, on_regular_message)
```

## Version Compatibility

- Python 3.9-3.13
- JupyterLab 4.x
- Handles both jupyter-collaboration v3+ and v2.x compatibility through version detection

## Testing Strategy

- **Unit tests**: Jest for TypeScript, pytest for Python
- **Integration tests**: Playwright via Galata for full JupyterLab integration
- **Coverage**: Aim for comprehensive test coverage, especially for routing logic