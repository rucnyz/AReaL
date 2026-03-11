"""
Training-aligned inference script for AIgiSE + AReaL.

Runs SeCodePLT evaluation through the EXACT same code path as training
rollouts (ArealOpenAI -> ArealLlm -> ADK -> AIgiSE), but standalone -- no
PPOTrainer, no distributed training.

This is useful for diagnosing tool call parsing issues, model capability
problems, and ADK integration bugs that only manifest when ArealLlm is
the LLM backend (as opposed to running SeCodePLT directly with Gemini).

Usage:
    # Start SGLang server first (in another terminal):
    python -m sglang.launch_server \
        --model-path UCSB-SURFI/VulnLLM-R-7B \
        --tp-size 1 --port 30000

    # Then run inference:
    uv run examples/aigise/infer_aligned.py \
        --model_path UCSB-SURFI/VulnLLM-R-7B \
        --sglang_addr http://localhost:30000 \
        --task_id arvo:53052 \
        --max_new_tokens 2048

    # Or use run_infer_aligned.sh which handles server startup:
    bash run_infer_aligned.sh
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import aiohttp
from datasets import load_dataset

from areal.api.io_struct import HttpRequest, ModelRequest, ModelResponse
from areal.engine.sglang_remote import SGLangBackend
from areal.experimental.adk import ArealLlm
from areal.experimental.openai import ArealOpenAI
from areal.utils.hf_utils import load_hf_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("InferAligned")


# ---------------------------------------------------------------------------
# Lightweight engine that implements _AsyncGenerateEngine protocol
# ---------------------------------------------------------------------------
class StandaloneSGLangEngine:
    """Minimal engine that talks to a running SGLang server.

    Implements the ``agenerate(req) -> ModelResponse`` protocol required
    by ArealOpenAI without pulling in the full AReaL distributed infra.
    """

    def __init__(self, server_addr: str, tokenizer):
        self.server_addr = server_addr.rstrip("/")
        self.tokenizer = tokenizer
        self.backend = SGLangBackend()
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Send generation request to SGLang server and return ModelResponse."""
        session = await self._get_session()

        # Build request using the same SGLang backend as training
        http_req: HttpRequest = self.backend.build_generation_request(
            req, with_lora=False, version=0
        )

        url = f"{self.server_addr}{http_req.endpoint}"
        start_time = time.perf_counter()

        async with session.post(url, json=http_req.payload) as resp:
            resp.raise_for_status()
            result = await resp.json()

        gen_result = self.backend.parse_generation_response(result)
        latency = time.perf_counter() - start_time

        response = ModelResponse(
            input_tokens=req.input_ids,
            output_tokens=gen_result.output_tokens,
            output_logprobs=gen_result.output_logprobs,
            output_versions=[0] * len(gen_result.output_tokens),
            stop_reason=gen_result.stop_reason,
            latency=latency,
            ttft=latency,
            tokenizer=req.tokenizer,
        )
        return response

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run_inference(args: argparse.Namespace):
    import aigise

    logger.info("Loading tokenizer: %s", args.model_path)
    tokenizer = load_hf_tokenizer(args.model_path)

    # Wait for SGLang server to be ready
    logger.info("Connecting to SGLang server: %s", args.sglang_addr)
    for attempt in range(30):
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    f"{args.sglang_addr}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        logger.info("SGLang server is ready")
                        break
        except Exception:
            pass
        if attempt < 29:
            logger.info(
                "Waiting for SGLang server... (attempt %d/30)", attempt + 1
            )
            await asyncio.sleep(5)
    else:
        raise RuntimeError(
            f"SGLang server at {args.sglang_addr} not reachable after 30 attempts"
        )

    # Create engine
    engine = StandaloneSGLangEngine(args.sglang_addr, tokenizer)

    # Create ArealOpenAI -- same params as training
    client = ArealOpenAI(
        engine=engine,
        tokenizer=tokenizer,
        tool_call_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        chat_template_type="concat",
    )

    # Log callback to print each turn's raw I/O
    turn_count = [0]

    def on_generate(input_text: str, output_text: str):
        turn = turn_count[0]
        turn_count[0] += 1
        print(f"\n{'=' * 80}")
        print(f"  TURN {turn}")
        print(f"{'=' * 80}")
        print(f"  [INPUT ({len(input_text)} chars)]:\n{input_text}")
        print(f"  [OUTPUT ({len(output_text)} chars)]:\n{output_text}")
        if "<tool_call>" in output_text:
            has_close = "</tool_call>" in output_text
            print(f"  [TOOL_CALL detected] has_closing_tag={has_close}")
        if "<set_model_response>" in output_text:
            print("  [WARNING] Model used XML tag instead of tool call format")
        print()

    # Create ArealLlm -- same as training
    model = ArealLlm(
        openai_client=client,
        default_max_tokens=args.max_new_tokens,
        on_generate=on_generate,
    )

    # Load dataset and find the task
    logger.info("Loading dataset: aigise/secodeplt")
    ds = load_dataset("aigise/secodeplt", split="train")
    task_data = None
    for item in ds:
        if item["task_id"] == args.task_id:
            task_data = dict(item)
            break
    if task_data is None:
        raise ValueError(f"Task {args.task_id} not found in dataset")

    logger.info("Running task: %s", args.task_id)
    logger.info("  docker_image: %s", task_data.get("docker_image"))
    logger.info("  target_functions: %s", task_data.get("target_functions"))

    # Create AIgiSE client and run -- EXACT same path as training
    aigise_client = aigise.create(
        args.agent_name,
        args.benchmark_name,
        model_name=args.model_path,
    )

    generate_kwargs = {}
    if args.max_turns is not None:
        generate_kwargs["max_turns"] = args.max_turns

    with aigise_client.init_session() as session:
        result = await session.areal_generate(
            data=task_data, model=model, **generate_kwargs
        )

    # Print results
    print(f"\n{'=' * 80}")
    print("  RESULT")
    print(f"{'=' * 80}")
    reward = result.get("reward", 0.0)
    print(f"  Reward: {reward}")
    for key, value in result.items():
        if key != "reward":
            val_str = str(value)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            print(f"  {key}: {val_str}")

    # Print interaction summary
    print(f"\n{'=' * 80}")
    print("  INTERACTION SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total LLM turns: {turn_count[0]}")
    print(f"  Cached interactions: {len(client._cache)}")
    for cid, interaction in client._cache.items():
        msg = interaction.completion.choices[0].message
        has_tc = msg.tool_calls is not None and len(msg.tool_calls) > 0
        content_preview = (msg.content or "")[:100]
        tc_names = (
            [tc.function.name for tc in msg.tool_calls] if has_tc else []
        )
        print(
            f"  [{cid[:12]}...] tool_calls={tc_names or 'None'}, "
            f"content[:100]={content_preview!r}"
        )

    await engine.close()
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Training-aligned inference for AIgiSE + AReaL"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="UCSB-SURFI/VulnLLM-R-7B",
        help="HuggingFace model path",
    )
    parser.add_argument(
        "--sglang_addr",
        type=str,
        default="http://localhost:30000",
        help="SGLang server address (e.g., http://localhost:30000)",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        default="arvo:53052",
        help="SeCodePLT task ID",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens per generation",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=20,
        help="Max agent turns",
    )
    parser.add_argument(
        "--tool_call_parser",
        type=str,
        default="qwen25",
        help="SGLang tool call parser (e.g., qwen25, qwen3_coder)",
    )
    parser.add_argument(
        "--reasoning_parser",
        type=str,
        default="qwen3",
        help="SGLang reasoning parser",
    )
    parser.add_argument(
        "--agent_name",
        type=str,
        default="vul_agent_static_tools",
        help="AIgiSE agent name",
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default="secodeplt",
        help="AIgiSE benchmark name",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU ID to launch SGLang server on (if --sglang_addr not reachable)",
    )

    args = parser.parse_args()
    asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()
