from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Optional
from typing import Sequence
from typing import Union

from .compaction import CompactionResult
from .compaction import create_default_compactor
from .compaction import should_auto_compact
from .compaction import uses_host_auto_compaction
from .context import InteractionContext
from .default_environment import DefaultEnvironment
from .display import render_interaction_items
from .environment import Environment
from .experimental_tools import create_inject_user_message_tool
from .items import CompactionMetadata
from .items import Init
from .items import Instructions
from .items import Message
from .items import ModelFailure
from .items import ModelSampleBoundary
from .items import SampleMetadata
from .items import ToolCall
from .items import ToolResult
from .items import TurnSummary
from .items import UserInteractionBoundary
from .items import summarize_turn_usage
from .media import parse_user_prompt
from .model import Model
from .model import ModelError
from .model import SampleParams
from .model_config import DEFAULT_SAVE_PATH as DEFAULT_SAVE_PATH
from .model_config import build_model
from .model_config import build_parser
from .model_config import initial_model_name
from .model_config import resolve_save_path
from .model_config import frontend_catalog, prepare_namespace, render_model_catalog
from .model_catalog import ModelBinding
from ._model_binding_debug import debug_model_binding_path
from ._model_binding_debug import save_debug_model_bindings
from .runtime_config import InteractionConfig
from .runtime_config import InteractionConfigSnapshot
from .save import load_interaction_save
from .save import save_interaction_save
from .user import UserInteraction


_SYSTEM_MESSAGE = (
    "You are a repository analyst. Inspect the repository with tools as "
    "needed. Do not modify files when the user only asks for analysis."
)

DEFAULT_PROMPT = "Summarize the repository in the current working directory."
# DEFAULT_PROMPT = "Here is the log for a recent run of the pythia/interaction demo. Let's investigate why there appears to be no interleaved assistant reasoning/response text, but only tool calls."
# DEFAULT_PROMPT = "Here is the log (demo.log.1) for a recent run of the pythia/interaction demo. Let's review the investigation of why there appears to be no interleaved assistant reasoning/response text but only tool calls, and implement a narrow fix for reasoning (no system message change). (Note that we should support both \"reasoning_content\" and \"reasoning\", as the former is still returned by some inference engines (although we might prioritize the latter if it exists and is non-null/non-empty)."
# DEFAULT_PROMPT = "Here is the log (demo.log.1) for a recent run of the pythia/interaction demo. Notice that the blocks (`[user] ...`, `[assistant] ...`, etc.) are not left-indented/delimited like in the existing autopythia/contradex implementation. Let's investigate and plan to re-add the same left-indentation/decoration to pythia.interaction as part of the display item impl."
# DEFAULT_PROMPT = "In pythia.interaction is the model context (list of interaction items) sufficient state for saving/resuming sessions? (Pending/interrupted tool calls/results might pose an issue, but we can ignore those for now so long as those interrupted calls can be swept over on resume.) Assuming sufficiency, let's implement initial support for saving the current session (in interaction.jsonl), and optionally resuming from it by passing --resume to the demo (let's also keep the working tree changes to the demo)."

EXPERIMENTAL_USER_MESSAGE_PROMPT = (
    "Run the synthetic-user-message integration test. Call "
    "`experimental_inject_user_message` exactly once with `{}` before answering. "
    "Do not use other tools. After the host appends the synthetic user message, "
    "reply with exactly `received: ` followed by that message's text, then stop. "
    "Do not call the tool again."
)


def run(
    model: Model,
    environment: Environment,
    *,
    prompt: Optional[str] = DEFAULT_PROMPT,
    instructions: Optional[Union[str, Instructions]] = None,
    max_samples: Optional[int] = None,
    sample_params: Optional[SampleParams] = None,
    save_path: Optional[Union[str, Path]] = None,
    resume: bool = False,
    enable_auto_compaction: bool = True,
    enable_media: bool = False,
    enable_workspace: bool = True,
    cwd: Path = Path("."),
    auto_compact_tokens: Optional[int] = None,
    max_context_tokens: Optional[int] = None,
    debug_save_model_binding: bool = False,
) -> str:
    if not hasattr(model, "sample") or not callable(model.sample):
        raise TypeError("model must provide sample(...)")
    if not isinstance(environment, Environment):
        raise TypeError("environment must be Environment")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool")
    if not isinstance(enable_auto_compaction, bool):
        raise TypeError("enable_auto_compaction must be a bool")
    if not isinstance(enable_media, bool):
        raise TypeError("enable_media must be a bool")
    if not isinstance(enable_workspace, bool):
        raise TypeError("enable_workspace must be a bool")
    if not isinstance(debug_save_model_binding, bool):
        raise TypeError("debug_save_model_binding must be a bool")
    if debug_save_model_binding and save_path is None:
        raise ValueError("debug_save_model_binding requires save_path")
    for field_name, value in (
        ("auto_compact_tokens", auto_compact_tokens),
        ("max_context_tokens", max_context_tokens),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{field_name} must be a positive integer or None")
    if prompt is not None and (
        not isinstance(prompt, str) or not prompt.strip()
    ):
        raise ValueError("prompt must be a non-empty string or None")
    if instructions is not None and not isinstance(
        instructions, (str, Instructions)
    ):
        raise TypeError("instructions must be a string, Instructions, or None")
    # Empty/whitespace-only strings are supported; only absence (None)
    # means "no instructions". Later items override earlier ones.
    instructions_item: Optional[Instructions] = None
    if isinstance(instructions, str):
        instructions_item = Instructions(text=instructions)
    elif isinstance(instructions, Instructions):
        instructions_item = instructions
    if resume and save_path is None:
        raise ValueError("resume requires save_path")
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples <= 0
    ):
        raise ValueError("max_samples must be a positive integer or None")
    if sample_params is not None and not isinstance(sample_params, SampleParams):
        raise TypeError("sample_params must be SampleParams or None")
    base_params = sample_params or SampleParams()
    if (
        auto_compact_tokens is not None
        and base_params.auto_compact_tokens is not None
        and auto_compact_tokens != base_params.auto_compact_tokens
    ):
        raise ValueError("Conflicting auto_compact_tokens keyword and sample param")
    # Bind config-owned fields once; the projection keeps the caller's other
    # per-call preferences, including a per-call ``extra`` override.
    inputs = InteractionConfigSnapshot(
        enable_workspace=enable_workspace,
        max_samples=max_samples,
        max_output_tokens=base_params.max_output_tokens,
        enable_auto_compaction=(
            enable_auto_compaction and base_params.enable_auto_compaction is not False
        ),
        auto_compact_tokens=(
            auto_compact_tokens if auto_compact_tokens is not None
            else base_params.auto_compact_tokens
        ),
        max_context_tokens=max_context_tokens,
    )
    turn_config = InteractionConfig.from_model(model, inputs).snapshot()
    sample_params = turn_config.sample_params(base_params)
    binding = getattr(model, "binding", None) if debug_save_model_binding else None
    if not isinstance(binding, ModelBinding):
        binding = None

    resumed_existing_save = False
    if resume and Path(save_path).exists():
        context = load_interaction_save(save_path)
        resumed_existing_save = True

        # Restore the human-visible transcript as well as the model state.
        for display_item in render_interaction_items(context.items):
            print(display_item)
    else:
        if resume:
            print(
                f"Warning: no existing {Path(save_path).name} was found; "
                "a fresh one was created.",
                file=sys.stderr,
            )
            if prompt is None:
                prompt = DEFAULT_PROMPT
        if prompt is None:
            raise ValueError("prompt must not be None without resume")
        initial: list = [Init(model=initial_model_name(model))]
        if instructions_item is not None:
            initial.append(instructions_item)
        context = InteractionContext(tuple(initial))

    if save_path is not None:
        save_interaction_save(save_path, context)
        if binding is not None:
            warning = save_debug_model_bindings(
                debug_model_binding_path(save_path), {"main": binding},
            )
            if warning is not None:
                print(warning, file=sys.stderr)
        elif debug_save_model_binding:
            print(
                "Warning: model does not expose a resolved model binding; "
                "debug snapshot was not written.",
                file=sys.stderr,
            )

    def _persist() -> None:
        if save_path is not None:
            save_interaction_save(save_path, context)

    pending_calls = context.pending_tool_calls()
    if pending_calls:
        # A process interruption may leave the durable context after model
        # output but before its tool results.  Complete that batch before
        # asking the model for a new sample.
        result = environment.execute_tool_calls(pending_calls)
        context.extend(result.context_items())
        _persist()
        for display_item in result.display_items(source_calls=pending_calls):
            print(display_item)
    elif resumed_existing_save and prompt is None and instructions_item is None:
        final_text = _final_assistant_text(context)
        if final_text is None or not final_text.strip():
            raise RuntimeError("resumed save has no final assistant text")
        return final_text

    if instructions_item is not None and resumed_existing_save:
        # Append override after pending batch is valid again. Strict
        # tool-sequence validation requires no pending calls here.
        context.extend((instructions_item,))
        _persist()
        for display_item in render_interaction_items((instructions_item,)):
            print(display_item)

    if prompt is not None:
        # For a resumed save, add the follow-up only after any pending
        # tool batch has been made valid again.
        message = parse_user_prompt(
            prompt,
            cwd=Path(cwd).expanduser().resolve(),
            enabled=enable_media,
            enable_workspace=enable_workspace,
        )
        user_interaction = UserInteraction(
            items=(message,),
        )
        context.extend(user_interaction.context_items())
        _persist()
        for display_item in user_interaction.display_items():
            print(display_item)

    turn_started = perf_counter()
    sample_count = 0
    while turn_config.max_samples is None or sample_count < turn_config.max_samples:
        threshold = turn_config.auto_compact_tokens
        if (
            turn_config.enable_auto_compaction
            and uses_host_auto_compaction(model)
            and threshold is not None
            and should_auto_compact(context, threshold)
        ):
            compaction = create_default_compactor(model).compact(
                context.copy(),
                tools=environment.tool_specs,
            )
            if not isinstance(compaction, CompactionResult):
                raise TypeError(
                    "compactor must return CompactionResult, got "
                    f"{type(compaction).__name__}"
                )
            context.extend(compaction.context_items())
            _persist()
            for display_item in compaction.display_items():
                print(display_item)
        sample_count += 1
        try:
            sample = model.sample(
                context,
                tools=environment.tool_specs,
                sample_params=sample_params,
            )
        except ModelError as exc:
            contribution = (
                *exc.completed_items,
                *((exc.failure,) if exc.failure is not None else ()),
            )
            if contribution:
                context.extend((*contribution, ModelSampleBoundary()))
                _persist()
                for display_item in render_interaction_items(contribution):
                    print(display_item)
                recovered_calls = tuple(
                    item for item in exc.completed_items
                    if isinstance(item, ToolCall)
                )
                if recovered_calls:
                    results = tuple(
                        ToolResult(
                            call_id=call.call_id,
                            output=(
                                "Not executed because the model response did "
                                "not complete."
                            ),
                            success=False,
                        )
                        for call in recovered_calls
                    )
                    context.extend(results)
                    _persist()
                    for display_item in render_interaction_items(
                        results,
                        source_calls=recovered_calls,
                    ):
                        print(display_item)
            raise
        context.extend(sample.context_items())
        _persist()
        for display_item in sample.display_items():
            print(display_item)
        if sample.stop_reason == "compaction":
            # A paused Messages server compaction contains a durable
            # compaction block but no final assistant text. Replay it
            # immediately so the provider can continue the turn.
            continue
        if not sample.tool_calls:
            final_text = sample.last_assistant_text
            if final_text is None or not final_text.strip():
                raise RuntimeError("model returned no final assistant text")
            # Derive the cumulative end-of-turn usage and make it visible.
            # ``summarize_turn_usage`` skips existing ``TurnSummary`` items
            # so re-entering this path never double-counts, and
            # ``TurnSummary`` is encoder-transparent and durable.
            turn_summary = summarize_turn_usage(
                context.items,
                elapsed_seconds=perf_counter() - turn_started,
            )
            context.extend((turn_summary,))
            _persist()
            for display_item in render_interaction_items((turn_summary,)):
                print(display_item)
            return final_text

        result = environment.execute_tool_calls(sample.tool_calls)
        context.extend(result.context_items())
        _persist()
        for display_item in result.display_items(source_calls=sample.tool_calls):
            print(display_item)

    raise RuntimeError(
        f"model did not produce a final answer within {turn_config.max_samples} samples"
    )


# Preserve the original public demo helper name for existing callers.
run_repository_summary = run


def _final_assistant_text(context: InteractionContext) -> Optional[str]:
    """Return final assistant text if the effective context ends with it."""
    for item in reversed(context.model_items()):
        if isinstance(
            item,
            (
                ModelSampleBoundary,
                SampleMetadata,
                CompactionMetadata,
                TurnSummary,
                UserInteractionBoundary,
            ),
        ):
            continue
        if isinstance(item, ModelFailure):
            return None
        if isinstance(item, Message) and item.role == "assistant":
            return item.content_text
        return None
    return None


def _build_model(args: argparse.Namespace) -> Model:
    return build_model(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "Ask a Chat Completions, Messages, or Codex Responses model to "
        "summarize a repository using Pythia's default local tools."
    )
    parser.add_argument(
        "--experimental-user-message-injection",
        action="store_true",
        help=(
            "include the experimental synthetic-user-message tool and use its "
            "test prompt for fresh sessions unless --prompt is supplied"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        catalog = frontend_catalog(args)
        if args.list_models:
            print(render_model_catalog(catalog))
            return 0
        args = prepare_namespace(args, catalog)
        cwd = Path(args.cwd).expanduser().resolve()
        prompt = args.prompt
        print(
            "Warning: exec_command runs without a sandbox; use only with a "
            "trusted model and workspace.",
            file=sys.stderr,
        )
        if not args.enable_workspace:
            print(
                "Warning: workspace path restrictions are disabled; "
                "exec_command workdir and apply_patch paths may resolve "
                "outside --cwd.",
                file=sys.stderr,
            )
        config = InteractionConfig.from_namespace(args).snapshot()
        save_path = resolve_save_path(args.save_path)
        if prompt is None and (not args.resume or not save_path.exists()):
            # A missing resume file is also a fresh session. Existing saves
            # must not receive the seed prompt again just to enable the tool.
            prompt = (
                EXPERIMENTAL_USER_MESSAGE_PROMPT
                if args.experimental_user_message_injection
                else DEFAULT_PROMPT
            )
        model = _build_model(args)
        extra_tools = (
            (create_inject_user_message_tool(),)
            if args.experimental_user_message_injection
            else ()
        )
        with DefaultEnvironment(
            cwd=cwd,
            enable_workspace=args.enable_workspace,
            extra_tools=extra_tools,
        ) as environment:
            run(
                model,
                environment,
                prompt=prompt,
                instructions=args.instructions,
                max_samples=config.max_samples,
                sample_params=config.sample_params(),
                save_path=save_path,
                resume=args.resume,
                enable_auto_compaction=config.enable_auto_compaction,
                enable_media=args.enable_experimental_media,
                enable_workspace=config.enable_workspace,
                cwd=cwd,
                auto_compact_tokens=config.auto_compact_tokens,
                max_context_tokens=config.max_context_tokens,
                debug_save_model_binding=args.debug_save_model_binding,
            )
    except Exception as exc:
        print(f"demo failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
