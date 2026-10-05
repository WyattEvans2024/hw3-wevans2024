# Title: LangChain IT Utility Agent
# Description: A simple LangChain agent that can run Python calculations,
# perform DNS lookups, and check website connectivity using custom tools.

import os
import socket

import httpx
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain_google_genai import ChatGoogleGenerativeAI


@tool
def dns_lookup(hostname: str) -> str:
	"""Resolve a hostname and return its unique IP addresses."""
	try:
		# DNS results may repeat addresses, so collect and sort them once.
		results = socket.getaddrinfo(hostname, None)
		addresses = sorted({result[4][0] for result in results})

		if not addresses:
			return f"No IP addresses were found for {hostname}."

		return f"{hostname} resolves to: {', '.join(addresses)}"
	except socket.gaierror as exc:
		return f"DNS lookup failed for {hostname}: {exc}"


@tool
def http_check(url: str) -> str:
	"""Request a website and report its HTTP response details."""
	try:
		# Follow redirects to report the final destination URL.
		response = httpx.get(url, timeout=10.0, follow_redirects=True)
		server = response.headers.get("server", "Not provided")
		content_type = response.headers.get("content-type", "Not provided")

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
	"""Configure Gemini, the available tools, and the agent instructions."""
	# Keep credentials and model selection in environment variables.
	required_settings = ("GOOGLE_API_KEY", "GOOGLE_MODEL")
	missing_settings = [name for name in required_settings if not os.getenv(name)]
	if missing_settings:
		names = ", ".join(missing_settings)
		raise RuntimeError(f"Missing required environment variable(s): {names}.")

	llm = ChatGoogleGenerativeAI(model=os.environ["GOOGLE_MODEL"])

	# Combine the Python REPL with the custom network diagnostic tools.
	tools = [PythonREPLTool(), dns_lookup, http_check]
	system_prompt = (
		"You are an IT troubleshooting assistant. "
		"Use dns_lookup for hostname and DNS questions. "
		"Use http_check for website connectivity and HTTP questions. "
		"Use PythonREPLTool for calculations and Python-based analysis. "
		"Explain tool results clearly and concisely."
	)

	return create_agent(model=llm, tools=tools, system_prompt=system_prompt)


def main() -> None:
	"""Run the interactive command-line agent until the user exits."""
	try:
		agent = build_agent()
	except RuntimeError as exc:
		print(f"Configuration error: {exc}")
		return

	print("LangChain IT Utility Agent")
	print("Ask for a calculation, DNS lookup, or website check. Type 'exit' to stop.\n")

	while True:
		try:
			user_input = input("You> ").strip()
		except (EOFError, KeyboardInterrupt):
			# Treat end-of-input and Ctrl+C as a normal request to stop.
			print("\nGoodbye.")
			break

		if user_input.lower() in {"exit", "quit"}:
			print("Goodbye.")
			break
		if not user_input:
			continue

		try:
			# Send each prompt as a user message and display the final reply.
			result = agent.invoke(
				{"messages": [{"role": "user", "content": user_input}]}
			)
			final_message = result["messages"][-1]
			response_content = final_message.content
			if isinstance(response_content, list):
				# Gemini may return content blocks; display only their readable text.
				text_parts = []
				for block in response_content:
					if isinstance(block, str):
						text_parts.append(block)
					elif isinstance(block, dict) and block.get("type") == "text":
						block_text = block.get("text")
						if isinstance(block_text, str):
							text_parts.append(block_text)
				response_content = "\n".join(text_parts)

			print(f"\nAgent> {response_content}\n")
		except Exception as exc:
			# Report model or agent errors and keep the session available.
			print(f"\nAgent error: {exc}\n")


if __name__ == "__main__":
	main()
