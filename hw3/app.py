"""
HW3 - Custom LangChain IT Utility Agent

This application demonstrates a custom LangChain agent that combines a
Python REPL tool with custom networking and web diagnostic tools.

The agent can:
- Perform Python calculations using PythonREPL.
- Resolve DNS names to IP addresses.
- Check HTTP/HTTPS websites and inspect response information.

API credentials are loaded through environment variables and are never
hard-coded into the source code.
"""

import os
import socket

import httpx

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain_google_genai import ChatGoogleGenerativeAI


@tool
def dns_lookup(hostname: str) -> str:
    """
    Resolve a hostname to its IP addresses.

    Use this tool when the user asks for the IP address of a domain,
    wants to verify DNS resolution, or is troubleshooting hostname
    resolution.

    Args:
        hostname: A hostname such as example.com.

    Returns:
        A string containing the IP addresses associated with the hostname.
    """
    try:
        results = socket.getaddrinfo(hostname, None)

        addresses = sorted(
            {
                result[4][0]
                for result in results
            }
        )

        if not addresses:
            return f"No IP addresses were found for {hostname}."

        return f"{hostname} resolves to: {', '.join(addresses)}"

    except socket.gaierror as exc:
        return f"DNS lookup failed for {hostname}: {exc}"


@tool
def http_check(url: str) -> str:
    """
    Check whether a website responds over HTTP or HTTPS.

    Use this tool when the user asks whether a website is reachable,
    requests an HTTP status code, or wants basic response information
    from a web server.

    Args:
        url: A complete URL such as https://example.com.

    Returns:
        The HTTP status code, final URL, server header, and content type.
    """
    try:
        response = httpx.get(
            url,
            timeout=10.0,
            follow_redirects=True,
        )

        server = response.headers.get("server", "Not provided")
        content_type = response.headers.get(
            "content-type",
            "Not provided",
        )

        return (
            f"HTTP check for {url}\n"
            f"Status code: {response.status_code}\n"
            f"Final URL: {response.url}\n"
            f"Server: {server}\n"
            f"Content-Type: {content_type}"
        )

    except httpx.RequestError as exc:
        return f"HTTP request failed for {url}: {exc}"


def build_agent():
    """
    Build and return the LangChain IT utility agent.

    The agent combines the Python REPL tool from the course labs with
    custom DNS and HTTP diagnostic tools.
    """
    model_name = os.getenv("GOOGLE_MODEL")

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not configured."
        )

    if not model_name:
        raise RuntimeError(
            "GOOGLE_MODEL environment variable is not configured."
        )

    llm = ChatGoogleGenerativeAI(
        model=model_name,
    )

    python_repl = PythonREPLTool()

    tools = [
        python_repl,
        dns_lookup,
        http_check,
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are an IT troubleshooting assistant. "
            "Use the available tools when they are useful. "
            "Use dns_lookup for DNS and hostname resolution questions. "
            "Use http_check for website connectivity and HTTP response questions. "
            "Use the Python REPL for calculations and Python-based analysis. "
            "Explain tool results clearly and concisely."
        ),
    )

    return agent


def main() -> None:
    """
    Run the interactive command-line interface for the agent.
    """
    try:
        agent = build_agent()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}")
        return

    print("=" * 60)
    print("HW3 - LangChain IT Utility Agent")
    print("=" * 60)

    print(
        "\nAvailable capabilities:\n"
        "  - Python calculations\n"
        "  - DNS lookups\n"
        "  - HTTP website checks\n"
    )

    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You> ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not user_input:
            continue

        try:
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ]
                }
            )

            final_message = result["messages"][-1]

            print("\nAgent>")
            print(final_message.content)
            print()

        except Exception as exc:
            print(f"\nAgent error: {exc}\n")


if __name__ == "__main__":
    main()