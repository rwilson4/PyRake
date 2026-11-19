"""Regenerate the figures in ``population_mean.md`` from its own code.

Run from the repository root:

    uv run python docs/user_guides/population_mean_figures.py

The guide's ``{testsetup}`` and ``{doctest}`` blocks run in order, in one
namespace, exactly as ``make doctest`` runs them. Each figure a block leaves open
is saved to the next name in ``FIGURES``, so the pictures cannot drift from the
code beside them. CI computes with ``OPENBLAS_NUM_THREADS=1`` and
``OPENBLAS_CORETYPE=Haswell``; set both to reproduce its numbers.

"""

import doctest
import pathlib
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
GUIDE = HERE / "population_mean.md"
FIGURES = ["population_mean_balance.png", "population_mean_frontier.png"]
BLOCK = re.compile(r"^```\{(testsetup|doctest)\}\n(.*?)^```", re.M | re.S)


def main() -> None:
    """Run the guide's code blocks and save the figures they draw."""
    namespace: dict[str, object] = {}
    names = iter(FIGURES)
    parser = doctest.DocTestParser()
    for kind, body in BLOCK.findall(GUIDE.read_text()):
        if kind == "testsetup":
            exec(body, namespace)
        else:
            for example in parser.get_examples(body):
                exec(compile(example.source, str(GUIDE), "single"), namespace)
        for num in plt.get_fignums():
            path = HERE / "images" / next(names)
            plt.figure(num).savefig(path, dpi=150, bbox_inches="tight")
            print(f"wrote {path}")
        plt.close("all")
    leftover = list(names)
    if leftover:
        raise RuntimeError(f"No figure was drawn for {leftover}")


if __name__ == "__main__":
    main()
