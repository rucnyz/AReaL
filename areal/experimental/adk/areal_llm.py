"""
AReaL-compatible LLM for Google ADK (Agent Development Kit).

This module provides ArealLlm, a Google ADK BaseLlm implementation that wraps
AReaL's ArealOpenAI client. This allows Google ADK agents (like those in AIgiSE)
to use AReaL's inference engine while automatically tracking token log probabilities
and rewards for RL training.

Architecture:
    Google ADK Agent
         │
         ▼
    ArealLlm (this class)
         │  - Converts LlmRequest → OpenAI messages
         │  - Calls ArealOpenAI.chat.completions.create
         │  - Converts response → LlmResponse
         ▼
    ArealOpenAI
         │  - Tracks token log probabilities
         │  - Manages reward assignment
         ▼
    AReaL InferenceEngine (SGLang backend)

Usage:
    from areal.experimental.adk import ArealLlm
    from areal.experimental.openai import ArealOpenAI

    # Create ArealOpenAI client
    client = ArealOpenAI(engine=engine, tokenizer=tokenizer, ...)

    # Create ADK-compatible model
    model = ArealLlm(openai_client=client)

    # Use with Google ADK agent
    agent = LlmAgent(model=model, ...)

    # After agent run, set reward and export
    client.set_last_reward(reward)
    client.apply_reward_discount(turn_discount=0.9)
    interactions = client.export_interactions(style="individual")
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generator

from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai.types import Content, FunctionCall, Part

if TYPE_CHECKING:
    from areal.experimental.openai import ArealOpenAI

logger = logging.getLogger(__name__)


class ArealLlm(BaseLlm):
    """Google ADK-compatible LLM that wraps AReaL's ArealOpenAI client.

    This class implements Google ADK's BaseLlm interface, allowing ADK agents
    to use AReaL's inference engine. All LLM calls are routed through ArealOpenAI,
    which automatically tracks token log probabilities and supports reward
    assignment for reinforcement learning.

    Attributes:
        model: Model identifier string (default: "areal")
        openai_client: The ArealOpenAI client instance

    Example:
        >>> from areal.experimental.openai import ArealOpenAI
        >>> from areal.experimental.adk import ArealLlm
        >>>
        >>> client = ArealOpenAI(engine=engine, tokenizer=tokenizer, ...)
        >>> model = ArealLlm(openai_client=client)
        >>>
        >>> # Use with ADK agent
        >>> agent = LlmAgent(model=model, tools=[...])
        >>> result = await agent.run(prompt)
        >>>
        >>> # Set reward after agent completes
        >>> client.set_last_reward(1.0)
    """

    model: str = "areal"

    def __init__(
        self,
        openai_client: "ArealOpenAI",
        model_name: str = "areal",
        **kwargs: Any,
    ):
        """Initialize ArealLlm.

        Args:
            openai_client: ArealOpenAI client instance for LLM inference.
                This client handles token log probability tracking and
                reward management.
            model_name: Model identifier string (default: "areal")
            **kwargs: Additional arguments passed to BaseLlm
        """
        super().__init__(**kwargs)
        self.model = model_name
        self._client = openai_client

    @staticmethod
    def supported_models() -> list[str]:
        """Return list of supported model identifiers."""
        return ["areal"]

    def _convert_contents_to_messages(
        self, contents: list[Content]
    ) -> list[dict[str, Any]]:
        """Convert Google ADK Content list to OpenAI messages format.

        Args:
            contents: List of Content objects from LlmRequest

        Returns:
            List of message dicts in OpenAI chat format
        """
        messages = []

        for content in contents:
            role = content.role
            if role == "model":
                role = "assistant"

            # Process parts
            text_parts = []
            tool_calls = []
            tool_results = []

            if content.parts:
                for part in content.parts:
                    # Handle text content
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)

                    # Handle function calls (tool calls from assistant)
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        tool_calls.append(
                            {
                                "id": f"call_{fc.name}_{len(tool_calls)}",
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": json.dumps(fc.args or {}),
                                },
                            }
                        )

                    # Handle function responses (tool results)
                    if hasattr(part, "function_response") and part.function_response:
                        fr = part.function_response
                        tool_results.append(
                            {
                                "tool_call_id": f"call_{fr.name}_{len(tool_results)}",
                                "role": "tool",
                                "content": json.dumps(fr.response)
                                if isinstance(fr.response, dict)
                                else str(fr.response),
                            }
                        )

            # Build message(s) based on content type
            if tool_results:
                # Tool results become separate tool messages
                for result in tool_results:
                    messages.append(result)
            elif tool_calls:
                # Assistant message with tool calls
                msg = {"role": "assistant", "tool_calls": tool_calls}
                if text_parts:
                    msg["content"] = "\n".join(text_parts)
                messages.append(msg)
            elif text_parts:
                # Regular text message
                messages.append({"role": role, "content": "\n".join(text_parts)})

        return messages

    def _convert_tools_to_openai(
        self, tools_dict: dict[str, Any] | None
    ) -> list[dict[str, Any]] | None:
        """Convert Google ADK tools to OpenAI tools format.

        Args:
            tools_dict: Dictionary of tool name to tool object from LlmRequest

        Returns:
            List of tool definitions in OpenAI format, or None if no tools
        """
        if not tools_dict:
            return None

        openai_tools = []
        for tool_name, tool in tools_dict.items():
            try:
                # Get tool declaration from ADK tool
                declaration = tool._get_declaration()
                if declaration:
                    openai_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": declaration.name or tool_name,
                                "description": declaration.description or "",
                                "parameters": self._convert_schema(
                                    declaration.parameters
                                ),
                            },
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to convert tool {tool_name}: {e}")
                continue

        return openai_tools if openai_tools else None

    def _convert_schema(self, schema: Any) -> dict[str, Any]:
        """Convert ADK schema to OpenAI JSON schema format.

        Args:
            schema: Schema object from tool declaration

        Returns:
            JSON schema dict for OpenAI tools
        """
        if schema is None:
            return {"type": "object", "properties": {}}

        if isinstance(schema, dict):
            return schema

        # Handle google.genai.types.Schema
        result: dict[str, Any] = {"type": "object", "properties": {}}

        if hasattr(schema, "properties") and schema.properties:
            for prop_name, prop_schema in schema.properties.items():
                prop_dict: dict[str, Any] = {}
                if hasattr(prop_schema, "type"):
                    type_val = prop_schema.type
                    # Convert enum to string if needed
                    if hasattr(type_val, "value"):
                        type_val = type_val.value
                    if hasattr(type_val, "name"):
                        type_val = type_val.name.lower()
                    prop_dict["type"] = str(type_val).lower()
                if hasattr(prop_schema, "description") and prop_schema.description:
                    prop_dict["description"] = prop_schema.description
                result["properties"][prop_name] = prop_dict

        if hasattr(schema, "required") and schema.required:
            result["required"] = list(schema.required)

        return result

    def _convert_response_to_llm_response(
        self, response: Any, response_id: str
    ) -> LlmResponse:
        """Convert OpenAI ChatCompletion to Google ADK LlmResponse.

        Args:
            response: ChatCompletion response from ArealOpenAI
            response_id: Response ID for tracking

        Returns:
            LlmResponse in Google ADK format
        """
        choice = response.choices[0]
        message = choice.message
        parts = []

        # Add text content
        if message.content:
            parts.append(Part.from_text(text=message.content))

        # Add tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.type == "function":
                    func = tool_call.function
                    try:
                        args = json.loads(func.arguments) if func.arguments else {}
                    except json.JSONDecodeError:
                        args = {"raw": func.arguments}

                    parts.append(
                        Part.from_function_call(
                            name=func.name,
                            args=args,
                        )
                    )

        content = Content(role="model", parts=parts)
        return LlmResponse(content=content)

    def generate_content(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> Generator[LlmResponse, None, None]:
        """Synchronous content generation (not recommended for AReaL).

        This method is provided for interface compatibility but is not
        recommended. Use generate_content_async for proper async operation.

        Args:
            llm_request: LlmRequest from Google ADK
            stream: Whether to stream responses (not supported)

        Yields:
            LlmResponse objects

        Raises:
            NotImplementedError: Always raised, use async version instead
        """
        raise NotImplementedError(
            "ArealLlm only supports async generation. "
            "Use generate_content_async instead."
        )

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generate content asynchronously using ArealOpenAI.

        This method:
        1. Converts LlmRequest to OpenAI messages format
        2. Calls ArealOpenAI.chat.completions.create
        3. Converts the response back to LlmResponse format

        All token log probabilities are automatically tracked by ArealOpenAI
        for use in RL training.

        Args:
            llm_request: LlmRequest from Google ADK containing:
                - contents: Conversation history
                - tools_dict: Available tools
                - config: Generation configuration
            stream: Whether to stream responses (not currently supported)

        Yields:
            LlmResponse containing the model's response

        Note:
            After the agent completes, call client.set_last_reward(reward)
            to assign the reward for training.
        """
        # Convert contents to OpenAI messages
        messages = self._convert_contents_to_messages(llm_request.contents or [])

        # Convert tools to OpenAI format
        tools = self._convert_tools_to_openai(llm_request.tools_dict)

        # Extract generation config
        config = llm_request.config
        kwargs: dict[str, Any] = {}

        if config:
            if hasattr(config, "temperature") and config.temperature is not None:
                kwargs["temperature"] = config.temperature
            if hasattr(config, "top_p") and config.top_p is not None:
                kwargs["top_p"] = config.top_p
            if hasattr(config, "max_output_tokens") and config.max_output_tokens:
                kwargs["max_tokens"] = config.max_output_tokens

        # Add tools if present
        if tools:
            kwargs["tools"] = tools

        try:
            # Call ArealOpenAI
            response = await self._client.chat.completions.create(
                messages=messages,
                **kwargs,
            )

            # Convert and yield response
            llm_response = self._convert_response_to_llm_response(
                response, response.id
            )
            yield llm_response

        except Exception as e:
            logger.error(f"ArealLlm generation error: {e}")
            # Return error response
            error_content = Content(
                role="model",
                parts=[Part.from_text(text=f"Error: {str(e)}")],
            )
            yield LlmResponse(content=error_content)
