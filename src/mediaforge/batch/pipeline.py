"""
Pipeline system.
Run multiple operations as a chained sequence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import PipelineError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineStep:
    """Represents a single step in the pipeline."""
    name: str
    func: Callable
    kwargs: dict[str, Any] = field(default_factory=dict)
    condition: Callable[..., bool] | None = None  # Conditional execution
    on_error: str = "stop"  # 'stop', 'skip', 'retry'
    max_retries: int = 3

    def __repr__(self) -> str:
        return f"PipelineStep(name={self.name!r})"


@dataclass
class PipelineResult:
    """Pipeline run result."""
    success: bool = True
    steps_completed: int = 0
    total_steps: int = 0
    step_results: list[dict[str, Any]] = field(default_factory=list)
    output_path: Path | None = None
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class Pipeline:
    """
    Processing pipeline.
    Runs multiple media operations in sequence.

    Usage:
        pipeline = Pipeline("Image Processing")
        pipeline.add_step("resize", image_processor.resize, width=800)
        pipeline.add_step("watermark", watermark_engine.add_text_watermark, text="(c)")
        pipeline.add_step("convert", converter.convert, target_format="webp")
        result = pipeline.execute("input.jpg", "output/")
    """

    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.steps: list[PipelineStep] = []
        self.logger = get_logger(f"Pipeline:{name}")

    def add_step(
        self,
        name: str,
        func: Callable,
        condition: Callable[..., bool] | None = None,
        on_error: str = "stop",
        **kwargs,
    ) -> Pipeline:
        """
        Adds a step to the pipeline. Supports method chaining.

        Args:
            name: Step name
            func: Function to run
            condition: Optional condition function for conditional execution
            on_error: Behavior on error ('stop', 'skip', 'retry')
        """
        step = PipelineStep(
            name=name, func=func, kwargs=kwargs,
            condition=condition, on_error=on_error,
        )
        self.steps.append(step)
        return self

    def remove_step(self, name: str) -> Pipeline:
        """Removes a step by name."""
        self.steps = [s for s in self.steps if s.name != name]
        return self

    def execute(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        keep_intermediates: bool = False,
    ) -> PipelineResult:
        """
        Runs the pipeline.

        Args:
            input_path: Input file
            output_dir: Output directory
            keep_intermediates: Keep intermediate files
        """
        start = time.time()
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = PipelineResult(total_steps=len(self.steps))
        current_input = input_path
        intermediate_files: list[Path] = []

        self.logger.info(f"Pipeline started: {self.name} ({len(self.steps)} steps)")

        for i, step in enumerate(self.steps):
            step_start = time.time()

            if step.condition and not step.condition(current_input):
                self.logger.info(f"  Step {i+1}/{len(self.steps)}: {step.name} - SKIPPED (condition not met)")
                result.step_results.append({
                    "step": step.name, "status": "skipped", "reason": "condition_not_met",
                })
                result.steps_completed += 1
                continue

            is_last = i == len(self.steps) - 1
            if is_last:
                step_output = output_dir / f"{input_path.stem}_final{input_path.suffix}"
            else:
                step_output = output_dir / f"_step_{i}_{step.name}{current_input.suffix}"
                intermediate_files.append(step_output)

            try:
                self.logger.info(f"  Step {i+1}/{len(self.steps)}: {step.name}")
                step_result = step.func(
                    current_input, step_output, **step.kwargs
                )

                result.step_results.append({
                    "step": step.name,
                    "status": "success",
                    "duration": round(time.time() - step_start, 2),
                    "output": str(step_output),
                })
                result.steps_completed += 1
                current_input = step_output

            except Exception as e:
                error_msg = f"Step '{step.name}' error: {e}"
                self.logger.error(error_msg)

                if step.on_error == "stop":
                    result.success = False
                    result.errors.append(error_msg)
                    break
                elif step.on_error == "skip":
                    result.step_results.append({
                        "step": step.name, "status": "skipped", "error": str(e),
                    })
                    result.steps_completed += 1
                elif step.on_error == "retry":
                    retry_success = False
                    for attempt in range(step.max_retries):
                        try:
                            step_result = step.func(current_input, step_output, **step.kwargs)
                            retry_success = True
                            result.steps_completed += 1
                            current_input = step_output
                            break
                        except Exception:
                            continue

                    if not retry_success:
                        result.success = False
                        result.errors.append(f"{error_msg} ({step.max_retries} retries failed)")
                        break

        if not keep_intermediates:
            for f in intermediate_files:
                if f.exists():
                    f.unlink()

        result.output_path = current_input if result.success else None
        result.duration_seconds = time.time() - start

        status = "SUCCESS" if result.success else "FAILED"
        self.logger.info(
            f"Pipeline {status}: {result.steps_completed}/{result.total_steps} steps "
            f"({result.duration_seconds:.1f}s)"
        )

        return result

    def dry_run(self) -> list[dict[str, str]]:
        """Lists pipeline steps without running (dry run)."""
        return [
            {"step": i + 1, "name": s.name, "on_error": s.on_error, "has_condition": s.condition is not None}
            for i, s in enumerate(self.steps)
        ]

    def __repr__(self) -> str:
        steps = " -> ".join(s.name for s in self.steps)
        return f"Pipeline({self.name!r}: {steps})"

    def __len__(self) -> int:
        return len(self.steps)
