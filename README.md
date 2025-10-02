# jupyter_ai_router

[![Github Actions Status](https://github.com/jupyter-ai-contrib/jupyter-ai-router/workflows/Build/badge.svg)](https://github.com/jupyter-ai-contrib/jupyter-ai-router/actions/workflows/build.yml)

Core message routing layer for Jupyter AI

This extension provides the foundational message routing functionality for Jupyter AI. It automatically detects new chat sessions and routes messages to registered callbacks based on message type (slash commands vs regular messages). Extensions can register callbacks to handle specific chat events without needing to manage chat lifecycle directly.

## Usage

### Basic MessageRouter Setup

```python
# The router is available in other extensions via settings
router = self.serverapp.web_app.settings.get("jupyter-ai", {}).get("router")

# Register callbacks for different event types
def on_new_chat(room_id: str, ychat: YChat):
    print(f"New chat connected: {room_id}")

def on_slash_command(room_id: str, message: Message):
    print(f"Slash command in {room_id}: {message.body}")

def on_regular_message(room_id: str, message: Message):`
    print(f"Regular message in {room_id}: {message.body}")

# Register the callbacks
router.observe_chat_init(on_new_chat)
router.observe_slash_cmd_msg("room-id", on_slash_command)
router.observe_chat_msg("room-id", on_regular_message)
```

### Message Flow

1. **Router detects new chats** - Automatically listens for chat room initialization events
2. **Router connects chats** - Establishes observers on YChat message streams
3. **Router routes messages** - Calls appropriate callbacks based on message type (slash vs regular)
4. **Extensions respond** - Your callbacks receive room_id and message data

### Available Methods

- `observe_chat_init(callback)` - Called when new chat sessions are initialized with `(room_id, ychat)`
- `observe_slash_cmd_msg(room_id, callback)` - Called for messages starting with `/` in a specific room
- `observe_chat_msg(room_id, callback)` - Called for regular (non-slash) messages in a specific room

## Install

To install the extension, execute:

```bash
pip install jupyter_ai_router
```

## Uninstall

To remove the extension, execute:

```bash
pip uninstall jupyter_ai_router
```

## Troubleshoot

If you are seeing the frontend extension, but it is not working, check
that the server extension is enabled:

```bash
jupyter server extension list
```

If the server extension is installed and enabled, but you are not seeing
the frontend extension, check the frontend extension is installed:

```bash
jupyter labextension list
```

## Contributing

### Development install

Note: You will need NodeJS to build the extension package.

The `jlpm` command is JupyterLab's pinned version of
[yarn](https://yarnpkg.com/) that is installed with JupyterLab. You may use
`yarn` or `npm` in lieu of `jlpm` below.

```bash
# Clone the repo to your local environment
# Change directory to the jupyter_ai_router directory
# Install package in development mode
pip install -e ".[test]"
# Link your development version of the extension with JupyterLab
jupyter labextension develop . --overwrite
# Server extension must be manually installed in develop mode
jupyter server extension enable jupyter_ai_router
# Rebuild extension Typescript source after making changes
jlpm build
```

You can watch the source directory and run JupyterLab at the same time in different terminals to watch for changes in the extension's source and automatically rebuild the extension.

```bash
# Watch the source directory in one terminal, automatically rebuilding when needed
jlpm watch
# Run JupyterLab in another terminal
jupyter lab
```

With the watch command running, every saved change will immediately be built locally and available in your running JupyterLab. Refresh JupyterLab to load the change in your browser (you may need to wait several seconds for the extension to be rebuilt).

By default, the `jlpm build` command generates the source maps for this extension to make it easier to debug using the browser dev tools. To also generate source maps for the JupyterLab core extensions, you can run the following command:

```bash
jupyter lab build --minimize=False
```

### Development uninstall

```bash
# Server extension must be manually disabled in develop mode
jupyter server extension disable jupyter_ai_router
pip uninstall jupyter_ai_router
```

In development mode, you will also need to remove the symlink created by `jupyter labextension develop`
command. To find its location, you can run `jupyter labextension list` to figure out where the `labextensions`
folder is located. Then you can remove the symlink named `@jupyter-ai/router` within that folder.

### Testing the extension

#### Server tests

This extension is using [Pytest](https://docs.pytest.org/) for Python code testing.

Install test dependencies (needed only once):

```sh
pip install -e ".[test]"
# Each time you install the Python package, you need to restore the front-end extension link
jupyter labextension develop . --overwrite
```

To execute them, run:

```sh
pytest -vv -r ap --cov jupyter_ai_router
```

#### Frontend tests

This extension is using [Jest](https://jestjs.io/) for JavaScript code testing.

To execute them, execute:

```sh
jlpm
jlpm test
```

#### Integration tests

This extension uses [Playwright](https://playwright.dev/docs/intro) for the integration tests (aka user level tests).
More precisely, the JupyterLab helper [Galata](https://github.com/jupyterlab/jupyterlab/tree/master/galata) is used to handle testing the extension in JupyterLab.

More information are provided within the [ui-tests](./ui-tests/README.md) README.

### Packaging the extension

See [RELEASE](RELEASE.md)
