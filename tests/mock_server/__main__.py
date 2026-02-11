"""Entry point for `python -m tests.mock_server`."""

import argparse

from tests.mock_server.server import MockLLMServer


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible mock LLM server")
    parser.add_argument("--port", type=int, default=9999, help="Port to listen on (default: 9999)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    server = MockLLMServer(host=args.host, port=args.port)
    print(f"Mock LLM server running at http://{args.host}:{args.port}/v1")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
