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
import os
import re
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.genai.types import Content, Part


def _debug_log(msg: str) -> None:
    """Write debug message to a file for post-run analysis."""
    log_path = os.environ.get("AREAL_DEBUG_LOG", "/tmp/areal_llm_debug.log")
    try:
        with open(log_path, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass

def _try_parse_python_style_tool_calls(
    text: str,
    tool_names: set[str],
) -> list[tuple[str, str]] | None:
    """Try to parse Python-style function calls like: tool_name({json_args}).

    Some models (e.g. Qwen3) output tool calls as Python-style text instead
    of <tool_call> XML tags. This fallback detects patterns like:
        set_model_response({"vulnerabilities": []})

    Returns list of (name, arguments_json) tuples, or None if no match.
    """
    if not tool_names:
        return None

    results = []
    for name in tool_names:
        pattern = re.escape(name) + r"\s*\(\s*(\{.*\})\s*\)\s*$"
        match = re.search(pattern, text.strip(), re.DOTALL)
        if match:
            args_str = match.group(1).strip()
            try:
                args = json.loads(args_str)
                if isinstance(args, dict):
                    results.append((name, json.dumps(args, ensure_ascii=False)))
                    logger.info("Parsed Python-style tool call: %s (fallback)", name)
            except json.JSONDecodeError:
                pass

    return results if results else None


# Mapping of OpenAI finish_reason to ADK FinishReason
_FINISH_REASON_MAPPING = {
    "length": types.FinishReason.MAX_TOKENS,
    "stop": types.FinishReason.STOP,
    "tool_calls": types.FinishReason.STOP,  # Tool calls are normal completion
    "function_call": types.FinishReason.STOP,  # Legacy function call
    "content_filter": types.FinishReason.SAFETY,
}

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
        openai_client: ArealOpenAI,
        model_name: str = "areal",
        default_max_tokens: int | None = None,
        on_generate: Callable[[str, str], None] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize ArealLlm.

        Args:
            openai_client: ArealOpenAI client instance for LLM inference.
                This client handles token log probability tracking and
                reward management.
            model_name: Model identifier string (default: "areal")
            default_max_tokens: Default max tokens per generation. Used when
                LlmRequest.config.max_output_tokens is not set.
            on_generate: Optional callback function called after each generation
                with (input_text, output_text) as arguments. Useful for logging
                or debugging.
            generation_kwargs: Default kwargs merged into every
                ``ArealOpenAI.chat.completions.create()`` call. Accepts any
                parameter the API supports (``extra_body``, ``temperature``,
                etc.). Per-call values from LlmRequest.config take precedence.
                Example::

                    generation_kwargs={
                        "extra_body": {
                            "chat_template_kwargs": {"enable_thinking": True}
                        }
                    }
            **kwargs: Additional arguments passed to BaseLlm
        """
        super().__init__(**kwargs)
        self.model = model_name
        self._client = openai_client
        self._default_max_tokens = default_max_tokens
        self._on_generate = on_generate
        self._generation_kwargs = generation_kwargs or {}

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
                        # Use the id from function_call if available, otherwise generate one
                        call_id = (
                            getattr(fc, "id", None)
                            or f"call_{fc.name}_{len(tool_calls)}"
                        )
                        tool_calls.append(
                            {
                                "id": call_id,
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
                        # Use the id from function_response if available
                        tool_call_id = (
                            getattr(fr, "id", None)
                            or f"call_{fr.name}_{len(tool_results)}"
                        )
                        tool_results.append(
                            {
                                "tool_call_id": tool_call_id,
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
                # Always include content key (even if empty) to match what
                # ArealOpenAI stores in the interaction cache, ensuring
                # correct prefix matching for conversation tree building.
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else "",
                    "tool_calls": tool_calls,
                })
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
                    converted_params = self._convert_schema(
                        declaration.parameters
                    )
                    tool_def = {
                        "type": "function",
                        "function": {
                            "name": declaration.name or tool_name,
                            "description": declaration.description or "",
                            "parameters": converted_params,
                        },
                    }
                    openai_tools.append(tool_def)
                    # Log set_model_response tool def once
                    if tool_name == "set_model_response" and not getattr(
                        self, "_logged_smr_def", False
                    ):
                        self._logged_smr_def = True
                        detail = json.dumps(tool_def, indent=2, default=str)
                        logger.warning(
                            "[DEBUG] set_model_response tool def: %s", detail,
                        )
                        _debug_log(f"set_model_response tool def:\n{detail}")
                else:
                    logger.warning(
                        "[DEBUG] Tool %s: _get_declaration() returned None", tool_name
                    )
                    _debug_log(f"Tool {tool_name}: _get_declaration() returned None")
            except Exception as e:
                logger.warning(f"Failed to convert tool {tool_name}: {e}")
                import traceback
                tb = traceback.format_exc()
                logger.warning("[DEBUG] Traceback: %s", tb)
                _debug_log(f"Failed to convert tool {tool_name}: {e}\n{tb}")
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

    def _convert_function_declarations_to_openai(
        self, config: Any
    ) -> list[dict[str, Any]] | None:
        """Convert function declarations from config.tools to OpenAI format.

        This is an alternative to _convert_tools_to_openai that works with
        the config.tools[0].function_declarations pattern used by LiteLLM.

        Args:
            config: Generation config from LlmRequest

        Returns:
            List of tool definitions in OpenAI format, or None if no tools
        """
        if not config:
            return None

        if not hasattr(config, "tools") or not config.tools:
            return None

        # config.tools is a list, and function_declarations are in the first element
        if not config.tools[0] or not hasattr(config.tools[0], "function_declarations"):
            return None

        function_declarations = config.tools[0].function_declarations
        if not function_declarations:
            return None

        openai_tools = []
        for func_decl in function_declarations:
            if not func_decl.name:
                continue

            parameters = {"type": "object", "properties": {}}
            if func_decl.parameters and hasattr(func_decl.parameters, "properties"):
                parameters = self._convert_schema(func_decl.parameters)
            elif (
                hasattr(func_decl, "parameters_json_schema")
                and func_decl.parameters_json_schema
            ):
                parameters = func_decl.parameters_json_schema

            tool_def = {
                "type": "function",
                "function": {
                    "name": func_decl.name,
                    "description": func_decl.description or "",
                    "parameters": parameters,
                },
            }

            # Add required fields if present
            if func_decl.parameters and hasattr(func_decl.parameters, "required"):
                if func_decl.parameters.required:
                    tool_def["function"]["parameters"]["required"] = list(
                        func_decl.parameters.required
                    )

            openai_tools.append(tool_def)

        return openai_tools if openai_tools else None

    def _convert_response_to_llm_response(
        self,
        response: Any,
        response_id: str,
        tool_names: set[str] | None = None,
    ) -> LlmResponse:
        """Convert OpenAI ChatCompletion to Google ADK LlmResponse.

        Args:
            response: ChatCompletion response from ArealOpenAI
            response_id: Response ID for tracking
            tool_names: Available tool names for Python-style fallback parsing

        Returns:
            LlmResponse in Google ADK format
        """
        choice = response.choices[0]
        message = choice.message
        parts = []

        has_tool_calls = hasattr(message, "tool_calls") and message.tool_calls
        content_text = message.content or ""

        # Add text content
        if content_text:
            parts.append(Part.from_text(text=content_text))

        # Add tool calls (from upstream parser)
        if has_tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.type == "function":
                    func = tool_call.function
                    try:
                        args = json.loads(func.arguments) if func.arguments else {}
                    except json.JSONDecodeError:
                        args = {"raw": func.arguments}

                    part = Part.from_function_call(
                        name=func.name,
                        args=args,
                    )
                    if hasattr(part, "function_call") and part.function_call:
                        part.function_call.id = tool_call.id
                    parts.append(part)

        # Fallback: detect Python-style tool calls like tool_name({json})
        # Some models (e.g. Qwen3) output these instead of <tool_call> XML
        elif content_text and tool_names:
            parsed = _try_parse_python_style_tool_calls(content_text, tool_names)
            if parsed:
                # Replace text part with function_call parts
                parts = []
                for name, arguments in parsed:
                    args = json.loads(arguments)
                    parts.append(Part.from_function_call(name=name, args=args))

        content = Content(role="model", parts=parts)
        llm_response = LlmResponse(content=content)

        # Set finish_reason
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason:
            finish_reason_str = str(finish_reason).lower()
            llm_response.finish_reason = _FINISH_REASON_MAPPING.get(
                finish_reason_str, types.FinishReason.OTHER
            )

        # Set usage_metadata if available
        usage = getattr(response, "usage", None)
        if usage:
            llm_response.usage_metadata = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=getattr(usage, "prompt_tokens", 0),
                candidates_token_count=getattr(usage, "completion_tokens", 0),
                total_token_count=getattr(usage, "total_tokens", 0),
            )

        return llm_response

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

        # Extract generation config
        config = llm_request.config

        # Add system instruction if present
        if (
            config
            and hasattr(config, "system_instruction")
            and config.system_instruction
        ):
            messages.insert(0, {"role": "system", "content": config.system_instruction})

        # Convert tools to OpenAI format (try tools_dict first, then config.tools)
        tools = self._convert_tools_to_openai(llm_request.tools_dict)
        if not tools:
            tools = self._convert_function_declarations_to_openai(config)

        # Collect tool names for Python-style fallback parsing
        tool_name_set: set[str] = set()
        if tools:
            tool_name_set = {t["function"]["name"] for t in tools}

        kwargs: dict[str, Any] = {}

        if config:
            if hasattr(config, "temperature") and config.temperature is not None:
                kwargs["temperature"] = config.temperature
            if hasattr(config, "top_p") and config.top_p is not None:
                kwargs["top_p"] = config.top_p
            if hasattr(config, "max_output_tokens") and config.max_output_tokens:
                kwargs["max_tokens"] = config.max_output_tokens

        # Use default_max_tokens if max_tokens not set from config
        if "max_tokens" not in kwargs and self._default_max_tokens is not None:
            kwargs["max_tokens"] = self._default_max_tokens

        # Add tools if present
        if tools:
            kwargs["tools"] = tools

        # Merge generation_kwargs (config/per-call values take precedence)
        for k, v in self._generation_kwargs.items():
            if k not in kwargs:
                kwargs[k] = v

        try:
            # Call ArealOpenAI
            response = await self._client.chat.completions.create(
                messages=messages,
                **kwargs,
            )
            # Call on_generate callback if provided
            if self._on_generate:
                interaction = self._client.get_interaction(response.id)
                if interaction and interaction.model_response:
                    input_text = self._client.tokenizer.decode(
                        interaction.model_response.input_tokens
                    )
                    output_text = self._client.tokenizer.decode(
                        interaction.model_response.output_tokens_without_stop
                    )
                    self._on_generate(input_text, output_text)

            # Convert and yield response
            llm_response = self._convert_response_to_llm_response(
                response, response.id, tool_names=tool_name_set
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
