# Structure

The lab consists of three parts, each with two group tutorial sessions given by the TAs and one individual homework for students to solve.

- Part 1: Federated & Distributed Learning
    - [Tutorial 1A: Horizontal FL](tutorial_1a/)
    - [Tutorial 1B: Data & Model Parallelism](tutorial_1b/)
    - [Homework 1](homework-1.ipynb)
- Part 2: (Vertical) Federated Learning on Generative Models
    - Tutorial 2A: Generative FL
    - Tutorial 2B: Vertical FL
    - Homework 2
- Part 3: Attacks & Defenses in Generative Models
    - Tutorial 3A: Attacks
    - Tutorial 3B: Defenses
    - Homework 3

# Setup

All the tutorial and homework content is within the current directory of the repository. Homework submission happens via ILIAS, as a Jupyter Notebook answering the assignment questions and containing the code needed to produce the answers.

Before working on any lab task, please follow the following steps to get up and running:
- Ensure [Python 3](https://www.python.org/downloads/) is installed (preferably a version from 3.9 to 3.13, but older ones should work by lowering some dependency versions in the requirements file).
- Have a [VS Code](https://code.visualstudio.com/Download) installation with the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extension packs (other IDEs like [PyCharm](https://www.jetbrains.com/pycharm/download/) also work, but this guide does not contain specific configuration files/steps for them).
- From the root directory of this repository, execute `python3 -m venv lab/.venv`.
- Open the repository in VS Code, and [create a new terminal](https://code.visualstudio.com/docs/terminal/basics) (checking that the beginning of the line starts with `(.venv)` to signal the correct Python environment got activated).
- Execute `pip install -r lab/requirements.txt` in the terminal.
