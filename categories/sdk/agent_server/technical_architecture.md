# Agent Server Technical Architecture

The diagram illustrates the layered architecture of the Agent Server, a local-first AI orchestration backend running on port 7701.

<div class="mxgraph"
     data-mxgraph='{
        "url": "https://logus2k.com/docbro/categories/sdk/agent_server/diagrams/architecture.drawio",
        "lightbox": false,
        "nav": true,
        "resize": false,
        "auto-fit": true,
        "auto-crop": true
     }'>
</div>

Clients connect through two entry points. Real-time clients — browsers, mobile apps, and IoT devices — communicate over WebSocket using the Socket.IO protocol, optionally through the provided agentClient.js SDK. OpenAI-compatible clients — such as curl, the Python and JavaScript OpenAI SDKs, or LangChain — connect via standard HTTP using the REST API.

Inside the Agent Server, the system is organized in five layers:

* The API Layer exposes two interfaces side by side. The Socket.IO interface handles real-time events: Chat and Interrupt for text interaction, JoinSTT/LeaveSTT for speech-to-text subscriptions, and JoinTTS/LeaveTTS for text-to-speech output. The OpenAI REST API provides POST /v1/chat/completions (with both streaming and non-streaming modes) and GET /v1/models, with optional Bearer token authentication.

* The Orchestration layer manages three concerns. Session State tracks per-client tasks, cancellation events, and locks. The Router Dispatcher runs the router agent preset as a fire-and-forget task that classifies user intent into structured JSON (emitted back to the client as a RouterResult event). Agent Presets hold the loaded configurations for all eight agents — general, router, topic, ml, robot, docbro, floorplan, and succint — each with its own system prompt, sampling parameters, and memory policy.

* The Processing layer contains the Worker Pool and Memory Registry. The Worker Pool manages an async queue of N LLM engine instances, acquired and released via a context manager to bound concurrent inference. Both the Socket.IO and REST API paths share this pool. The Memory Registry provides pluggable conversation history strategies; the current implementation, ThreadWindowMemory, maintains a rolling window of messages per thread_id and generates a context preamble that is injected into the LLM prompt.

* The Inference layer is the LLM Engine, a wrapper around llama-cpp-python. It handles streaming token generation, automatic chat template detection, three-tier sampling configuration (engine defaults, agent overrides, per-request overrides), system prompt injection, memory preamble insertion, and cooperative cancellation.

* The Voice Integration layer contains two managers that bridge the Agent Server to external speech services. The STT Manager maintains multiplexed Socket.IO connections to STT servers — one connection per URL, supporting many client rooms on the same link. When a transcript arrives, it is forwarded to the client and then routed through the normal agent processing pipeline. The TTS Manager streams generated text chunks to a TTS server for voice synthesis, with support for per-client voice and speed configuration and optional JSON field extraction for agents that produce structured output.

At the bottom of the diagram, external services and data stores sit outside the server boundary. The STT Server (port 2700) and TTS Server (port 7700) are separate microservices that communicate with the Agent Server over Socket.IO. All three services run on a shared Docker network. The GGUF model files (Qwen, Phi-4, Gemma, EuroLLM, SmolLM, among others) are loaded by the LLM Engine at startup. Configuration data — agent_config.json, the *.agent.json preset files, and system prompt text files — are read at startup and define the server's runtime behavior.
