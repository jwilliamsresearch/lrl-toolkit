"""``lrl`` command-line interface.

Examples::

    lrl init welsh                     # scaffold projects/welsh.yaml
    lrl sources list welsh             # show the source catalog
    lrl run -c projects/welsh.yaml     # run the full pipeline (resumable)
    lrl ingest -c projects/welsh.yaml  # run a single stage
    lrl languages                      # list available language profiles
    lrl dashboard                      # launch the web UI
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import STAGE_ORDER, load_project
from .pipeline import run_pipeline, run_single_stage
from .registry import list_compute, list_languages, list_models, load_language

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="lrl-toolkit: build custom LLMs for low-resource languages.",
)
sources_app = typer.Typer(no_args_is_help=True, help="Inspect corpus sources for a language.")
app.add_typer(sources_app, name="sources")

console = Console()

CONFIG_OPT = typer.Option(..., "--config", "-c", help="Path to the project YAML.")


@app.command()
def version() -> None:
    """Print the lrl-toolkit version."""
    console.print(f"lrl-toolkit {__version__}")


@app.command()
def languages() -> None:
    """List available language, model, and compute profiles."""
    console.print("[bold]Languages:[/bold] " + ", ".join(list_languages() or ["(none)"]))
    console.print("[bold]Base models:[/bold] " + ", ".join(list_models() or ["(none)"]))
    console.print("[bold]Compute profiles:[/bold] " + ", ".join(list_compute() or ["(none)"]))


@app.command()
def init(
    language: str = typer.Argument(..., help="Language profile slug, e.g. 'welsh'."),
    base_model: str = typer.Option("qwen2.5-1.5b", "--base-model", "-m"),
    compute: str = typer.Option("consumer_gpu", "--compute"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output path for the project YAML."),
) -> None:
    """Scaffold a new project YAML for a language."""
    try:
        load_language(language)
    except FileNotFoundError:
        console.print(
            f"[yellow]Warning:[/yellow] no language profile '{language}' found. "
            f"Available: {', '.join(list_languages()) or '(none)'}. "
            "Writing the project anyway; add the profile before running."
        )

    out_path = out or Path("projects") / f"{language}.yaml"
    if out_path.exists():
        console.print(f"[red]Refusing to overwrite existing file:[/red] {out_path}")
        raise typer.Exit(code=1)

    content = _scaffold_yaml(language, base_model, compute)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    console.print(f"[green]Created[/green] {out_path}")
    console.print(f"Next: [bold]lrl run -c {out_path}[/bold]")


@sources_app.command("list")
def sources_list(language: str = typer.Argument(..., help="Language profile slug.")) -> None:
    """List the corpus sources catalogued for a language."""
    profile = load_language(language)
    table = Table(title=f"Sources for {profile.display_name} ({profile.iso639_3})")
    table.add_column("Connector", style="cyan")
    table.add_column("License")
    table.add_column("Notes")
    if not profile.sources:
        console.print("[yellow]No sources catalogued for this language yet.[/yellow]")
        return
    for hint in profile.sources:
        table.add_row(
            hint.connector,
            str(hint.params.get("license", "?")),
            hint.notes or "",
        )
    console.print(table)


@app.command()
def run(
    config: Path = CONFIG_OPT,
    force: bool = typer.Option(False, "--force", help="Re-run every selected stage."),
    force_from: str | None = typer.Option(None, "--from", help="Re-run from this stage onward."),
    only: list[str] = typer.Option(None, "--only", help="Run only these stage(s)."),
) -> None:
    """Run the full pipeline (or a subset), resumable via the manifest."""
    project = load_project(config)
    if force_from and force_from not in STAGE_ORDER:
        console.print(f"[red]Unknown stage:[/red] {force_from}")
        raise typer.Exit(code=1)
    outcomes = run_pipeline(
        project, stages=list(only) if only else None, force=force, force_from=force_from
    )
    _print_outcomes(project.name, outcomes)


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port"),
) -> None:
    """Launch the local web dashboard (requires the [dashboard] extra)."""
    import shutil
    import subprocess

    app_path = Path(__file__).resolve().parent.parent.parent / "dashboard" / "app.py"
    if shutil.which("streamlit") is None:
        console.print(
            "[red]Streamlit not found.[/red] Install with: pip install 'lrl-toolkit[dashboard]'"
        )
        raise typer.Exit(code=1)
    subprocess.run(["streamlit", "run", str(app_path), "--server.port", str(port)], check=False)


def _make_stage_command(stage_name: str):
    def _cmd(
        config: Path = CONFIG_OPT,
        force: bool = typer.Option(False, "--force", help="Re-run even on a manifest hit."),
    ) -> None:
        project = load_project(config)
        outcome = run_single_stage(project, stage_name, force=force)
        _print_outcomes(project.name, [outcome])

    _cmd.__doc__ = f"Run the '{stage_name}' stage."
    return _cmd


# Register one command per pipeline stage: `lrl ingest`, `lrl clean`, ...
# `evaluate` is exposed as `eval` for brevity, matching the plan.
for _stage in STAGE_ORDER:
    _name = "eval" if _stage == "evaluate" else _stage
    app.command(name=_name)(_make_stage_command(_stage))


def _print_outcomes(project_name: str, outcomes) -> None:
    table = Table(title=f"Pipeline: {project_name}")
    table.add_column("Stage", style="cyan")
    table.add_column("Status")
    table.add_column("Fingerprint", style="dim")
    for o in outcomes:
        color = "green" if o.status == "ran" else "yellow"
        table.add_row(o.stage, f"[{color}]{o.status}[/{color}]", o.fingerprint)
    console.print(table)


def _scaffold_yaml(language: str, base_model: str, compute: str) -> str:
    return f"""# lrl-toolkit project: {language}
name: {language}
language: {language}
base_model: {base_model}
compute: {compute}
seed: 42

ingest:
  sources: []          # empty = use every source in the language profile
  max_gb: 5

clean:
  lang_id: glotlid
  dedup: minhash
  min_quality: 0.6

tokenizer:
  strategy: extend
  added_tokens: 8000

pretrain:
  method: qlora
  epochs: 1
  seq_len: 2048

convdata:
  translate: [dolly]
  translate_limit: 500
  provider: ollama          # local teacher LLM (ollama/local/mock); no proprietary APIs
  synth:
    provider: ollama
    n: 2000
  review: true

finetune:
  method: qlora
  dpo: false

evaluate:
  benchmarks: [perplexity, flores, belebele]

export:
  quantize: [gguf_q4_k_m]
  push_to_hub: false
"""


if __name__ == "__main__":  # pragma: no cover
    app()
