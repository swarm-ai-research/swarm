---
description: "SWARM command-line operating procedure written in ASD-STE100 Simplified Technical English. Install the software, list the scenarios, run a scenario, and save the results."
---

# SWARM Operating Procedure

## About this document

This document is written in Simplified Technical English (STE). STE is the
controlled language of the specification ASD-STE100. This document obeys the
STE writing rules:

- Each instruction is a short sentence.
- Each instruction uses the active voice.
- Each instruction starts with a command verb.
- One sentence gives one instruction.
- The words keep the same meaning in all sentences.

The technical names and the technical verbs for this document are in the
section [Terms](#terms). Read the terms before you do the procedures.

!!! note "Scope"
    This document tells you how to operate the SWARM command-line program.
    It does not tell you how to write the Python code. For the code interface,
    read the [Framework Overview](framework-overview.md).

## Terms

The table shows the technical names in this document. Use only these names.

| Term | Definition |
|---|---|
| SWARM | The simulation program in this repository. |
| scenario | A YAML file that sets the agents and the governance rules. |
| scenario file | The file that holds one scenario. |
| agent | A software object that does interactions in the simulation. |
| epoch | One full cycle of steps in the simulation. |
| step | One interaction cycle in an epoch. |
| seed | The number that starts the random generator. |
| run | One execution of one scenario. |
| result | The data that a run makes. |
| export | To write the result to a file. |
| plot | An image that shows the result data. |

The table shows the technical verbs in this document.

| Verb | Meaning |
|---|---|
| install | To put the program on the computer. |
| list | To show the names in a set. |
| run | To make the program do a scenario. |
| export | To write data to a file or a directory. |
| override | To replace a value in the scenario file with a new value. |

## Before you start

You must obey these conditions before you operate SWARM:

1. Make sure that the computer has Python 3.10 or a later version.
2. Make sure that you have the SWARM repository on the computer.
3. Open a terminal.
4. Go to the top directory of the repository.

## Procedure 1 — Install the software

Do these steps to install SWARM:

1. Type this command:

   ```bash
   python -m pip install -e ".[dev,runtime]"
   ```

2. Push the **Enter** key.
3. Wait for the installation to stop.
4. Make sure that the terminal shows no error message.

!!! warning
    Do not use a different Python version after the installation. A different
    version can cause an import error.

## Procedure 2 — List the scenarios

Do these steps to see the available scenarios:

1. Type this command:

   ```bash
   python -m swarm list
   ```

2. Push the **Enter** key.
3. Read the list of scenario files.

The program shows each scenario file in the `scenarios/` directory. Each line
shows one scenario file.

!!! note
    To list the scenarios in a different directory, add the directory name.
    For example, type `python -m swarm list --dir my_scenarios`.

## Procedure 3 — Run a scenario

Do these steps to run a scenario:

1. Select a scenario file from the list.
2. Type this command. Replace `<scenario>` with the scenario file:

   ```bash
   python -m swarm run <scenario>
   ```

   For example, type:

   ```bash
   python -m swarm run scenarios/baseline.yaml
   ```

3. Push the **Enter** key.
4. Wait for the run to stop.
5. Read the result in the terminal.

### Override the scenario values

You can replace three values in the scenario file. Use the options in the
table. Each option needs a number.

| Option | Function |
|---|---|
| `--seed` | Set the seed. |
| `--epochs` | Set the number of epochs. |
| `--steps` | Set the number of steps in each epoch. |

To override the values, add the options to the command. For example, type:

```bash
python -m swarm run scenarios/baseline.yaml --seed 42 --epochs 20 --steps 10
```

!!! warning
    Record the seed for each run. If you do not use the same seed, the program
    makes a different result. The same scenario, the same seed, and the same
    version always make the same result.

### Stop the terminal output

To stop the progress output, add the `--quiet` option or the `-q` option. For
example, type:

```bash
python -m swarm run scenarios/baseline.yaml --quiet
```

## Procedure 4 — Save the results

You can export the result to a file. Use the options in the table.

| Option | Function |
|---|---|
| `--export-json <PATH>` | Write the result to one JSON file. |
| `--export-csv <DIR>` | Write the result to CSV files in a directory. |

Do these steps to save the result as a JSON file:

1. Type this command. Replace `<PATH>` with the file name:

   ```bash
   python -m swarm run scenarios/baseline.yaml --export-json <PATH>
   ```

   For example, type:

   ```bash
   python -m swarm run scenarios/baseline.yaml --export-json results.json
   ```

2. Push the **Enter** key.
3. Wait for the run to stop.
4. Make sure that the file is in the location.

!!! note
    You can use `--export-json` and `--export-csv` together. The program then
    makes a JSON file and CSV files in one run.

## Procedure 5 — Make the plots

The program can make the standard plot bundle. The bundle shows the toxicity
data, the welfare data, and the selection geometry.

Do these steps to make the plots:

1. Type this command. Replace `<DIR>` with the directory name:

   ```bash
   python -m swarm run scenarios/baseline.yaml --plots <DIR>
   ```

2. Push the **Enter** key.
3. Wait for the run to stop.
4. Open the directory `<DIR>/plots/`.
5. Look at the image files.

!!! note
    If you give no directory to the `--plots` option, the program selects a
    default directory. The program uses the export path or the path
    `runs/<scenario>_seed<seed>`.

## Fault isolation

The table shows the usual faults and the corrective actions.

| Fault | Cause | Corrective action |
|---|---|---|
| The terminal shows `scenario file '...' not found`. | The scenario path is wrong. | Do Procedure 2. Then use a correct path. |
| The terminal shows `... is not readable`. | The file permissions are wrong. | Change the file permissions. Then run the scenario again. |
| The program makes a different result each run. | The seed is not constant. | Add the `--seed` option with the same number. |
| The program shows an import error. | The Python version is wrong. | Do Procedure 1 with Python 3.10 or a later version. |
| The plots directory is empty. | The run stopped before the end. | Read the terminal for an error. Then run the scenario again. |

## Related documents

- [Installation](../getting-started/installation.md)
- [Quick Start](../getting-started/quickstart.md)
- [Writing Scenarios](scenarios.md)
- [Reproducibility](../getting-started/reproducibility.md)
