"""
Command-line interface for Mozo.

Provides simple commands for starting the server and checking version.
"""

import sys
import time

import click
import uvicorn


@click.group()
def cli():
    """Mozo - Universal CV Model Server"""
    pass


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload on code changes')
@click.option('--workers', default=1, type=int, help='Number of worker processes')
def start(host, port, reload, workers):
    """Start the Mozo model server"""
    click.echo(f"Starting Mozo server on {host}:{port}...")
    if reload:
        click.echo("Auto-reload enabled (development mode)")
    if workers > 1:
        click.echo(f"Running with {workers} worker processes")

    uvicorn.run(
        "mozo.server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1  # reload only works with 1 worker
    )


@cli.command()
@click.argument("workflow", type=click.Path(exists=True, dir_okay=False))
@click.option("--file", "media",
              help="Run on this image or video instead of the one saved in the workflow")
@click.option("--set", "settings", multiple=True, metavar="NAME=VALUE",
              help="Override a parameter. Repeatable.")
@click.option("--test", is_flag=True,
              help="Run one item and describe every node, instead of the whole source")
def run(workflow, media, settings, test):
    """Run a workflow from a JSON file, with no browser and no server.

    **The whole source, not one item of it.** A workflow whose input is a folder of ten thousand
    photographs processes ten thousand; one whose input is a two-hour video processes every frame.
    This used to take one item and stop -- which for a folder wrote a single file and reported
    success, and is the same thing the editor's button still does.

    ``--test`` is that older behaviour, kept because it is worth having: one item through the
    graph, with every node describing what it produced. An image says its size, detections say how
    many and what they were, a depth map says its range -- because a terminal is not a place to put
    a photograph, and the thing worth seeing is whether the graph did its job before pointing it at
    ten thousand files.

    A full run prints progress instead, because ten thousand items is ten thousand lines nobody
    reads. Nodes that write files have already written them.
    """
    from mozo.workflow import Workflow

    built = Workflow.load(workflow)

    overrides = {}
    for setting in settings:
        name, separator, value = setting.partition("=")
        if not separator:
            raise click.BadParameter(f"expected NAME=VALUE, got {setting!r}")
        overrides[name] = _typed(built, name, value)
    if media:
        # Asked of the workflow rather than named here. This line used to say "image", which was
        # one node's parameter written down in a second place -- so renaming that parameter left
        # the command line raising KeyError while the HTTP endpoint went on working.
        try:
            overrides[built.file_parameter] = media
        except ValueError as error:
            raise click.BadParameter(str(error)) from error
    if test or built.source is None:
        # No source means the caller has to bring the items, and this command does not: one pass
        # over what is wired is the only thing it could mean.
        for event in built.stream(**overrides):
            if event.status == "failed":
                raise click.ClickException(event.error)
            if event.status == "completed":
                ends = " (an end)" if event.node in built.terminals else ""
                click.echo(f"  {event.node}: {_describe(event.output)}{ends}")
        return

    _process(built, overrides)


def _process(built, overrides: dict) -> None:
    """One pass over the whole source, reporting how far it has got.

    Throttled on :data:`~mozo.workflow.wire.PREVIEW_EVERY`, which is where that trade-off is
    stated. Written over itself on a terminal and one line per update elsewhere, so a log file does
    not become a progress bar nobody can read.
    """
    from mozo.workflow.wire import PREVIEW_EVERY

    live = sys.stdout.isatty()
    began = last = time.monotonic()
    done = 0
    try:
        for _item, _results in built.process(**overrides):
            done += 1
            now = time.monotonic()
            if now - last >= PREVIEW_EVERY:
                last = now
                rate = done / (now - began)
                click.echo(f"\r  {_items(done)} ({rate:,.0f}/s)", nl=not live)
    except RuntimeError as error:
        # The engine names the item and the node; wrapping it in a traceback would bury both.
        raise click.ClickException(str(error)) from error
    finally:
        if live and done:
            click.echo()

    # Floored rather than branched on: a coarse clock can report zero for a fast run, and a
    # division by it is the only thing that cares.
    took = max(time.monotonic() - began, 1e-9)
    click.echo(f"  {_items(done)} in {took:.1f}s ({done / took:,.0f}/s)")


def _items(count: int) -> str:
    """``1 item`` rather than ``1 items``, which is the sort of thing a person reads as a bug."""
    return f"{count:,} item{'' if count == 1 else 's'}"


def _typed(built, name: str, text: str):
    """Read *text* as whatever the parameter it is setting was declared to be.

    A command line hands over strings. Passed through, ``--set threshold=0.6`` reaches a model as
    the string "0.6", and ``--set show_names=false`` reaches a node as a non-empty string, which is
    true -- the opposite of what was asked, with nothing raised. The catalogue already says what
    each parameter is, so it says what to read.
    """
    kinds = {parameter.name: parameter.kind
             for step in built.steps.values() for parameter in step.spec.parameters}
    kind = kinds.get(name)
    if kind is None:
        raise click.BadParameter(f"no parameter {name!r} in this workflow. It has: {sorted(kinds)}")

    try:
        if kind == "int":
            return int(text)
        if kind == "float":
            return float(text)
        if kind == "bool":
            if text.lower() not in ("true", "false"):
                raise ValueError("expected true or false")
            return text.lower() == "true"
    except ValueError as error:
        raise click.BadParameter(f"{name} is a {kind}: {error}") from error
    return text


def _describe(value) -> str:
    """What a node produced, in one line."""
    import numpy as np

    if value is None:
        return "nothing"
    if isinstance(value, tuple):
        return ", ".join(_describe(part) for part in value)
    if isinstance(value, list):
        return f"{len(value)} x [{_describe(value[0])}]" if value else "an empty batch"
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            return f"a {value.shape[1]}x{value.shape[0]} map, {value.min():.3g} to {value.max():.3g}"
        return f"a {value.shape[1]}x{value.shape[0]} image"
    named = getattr(value, "to_dict", None)
    if named:
        rows = named()
        seen = {row.get("class_name") for row in rows if isinstance(row, dict)}
        return f"{len(rows)} x {sorted(name for name in seen if name)}" if rows else "nothing found"
    return type(value).__name__


@cli.command()
def version():
    """Show Mozo version"""
    from mozo import __version__
    click.echo(f"Mozo version {__version__}")


if __name__ == '__main__':
    cli()
