# polyglot-llm
# Purpose
This is an open source LLM wrapper library in Go. 
It provides a common interface for interacting with different LLM providers, such as OpenAI, Anthropic, and Azure. The goal is to make it easy to switch between different LLM providers without having to change your code.

There is a a library call LangchainGo.  The library is old and not being actively maintained, and it also has a little different design philosophy than what we want to achieve with this library.
But its a good reference for how to implement the LLM interface and the generator interface, as well as how to handle tools and logging.
We plan on providing a wrapper for it as well.

# License
This is licensed under Apache 2.0, so feel free to use it in your projects and contribute to it as well.
