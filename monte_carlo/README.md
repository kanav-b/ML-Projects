# Assignment: Population Genetics Simulation and Analysis

## Description

You are a population geneticist studying the evolutionary history of a species that experienced a severe population bottleneck followed by recovery. You will use both forward-time (Wright-Fisher) and backward-time (coalescent) simulations to model this evolutionary scenario and analyze how demographic events affect genetic diversity.

This assignment consists of two separate implementations:
- **Part 1**: Forward simulation using Wright-Fisher model with selection
- **Part 2**: Backward simulation using coalescent theory (neutral evolution)

## Evolutionary Scenario

Your study species experienced this demographic history:
- **Ancestral population**: Large, stable population (N = 10,000) for 1000 generations
- **Bottleneck event**: Population crashed to N = 100 for 50 generations  
- **Recovery phase**: Population expanded back to N = 10,000
- **Present day**: You have genetic samples from the recovered population

## Assignment Tasks

### Part 1: Wright-Fisher Simulation (50 points)

**Script name**: `firstname_lastname_assignment7_part1.py`

Implement a Wright-Fisher simulator that models demographic changes with selection.

**Command Line Interface:**
```bash
python firstname_lastname_assignment7_part1.py wright_fisher_config.csv wright_fisher_demography.csv --seed 12345 --output forward_results.tsv
```

**Configuration File (`wright_fisher_config.csv`):**
```
parameter,value
mutation_rate,1e-8
sequence_length,1000
total_generations,1500
selection_coefficient,0.1
beneficial_mutation_time,1025
```

**Demographics File (`demography.csv`):**
```
generation,population_size
0,10000
1000,100
1050,10000
```

**Key Requirements:**
- Simulate population size changes over time
- Add new mutations each generation using Poisson process
- Model beneficial mutation arising during bottleneck with fitness advantage
- Track genetic diversity (π) and Watterson's θ over time
- Generate time-series data showing diversity changes

**Output Format (`forward_results.tsv`):**
```
generation	population_size	num_mutations	nucleotide_diversity	theta_watterson	beneficial_freq
0	10000	0	0.000	0.000	0.000
1	10000	3	0.001	0.002	0.000
1025	100	45	0.023	0.034	0.010
...
```

### Part 2: Coalescent Simulation (50 points)

**Script name**: `firstname_lastname_assignment7_part2.py`

Implement a coalescent simulator that reconstructs the same demographic scenario without selection.

**Command Line Interface:**
```bash
python firstname_lastname_assignment7_part2.py coalescent_config.csv coalescent_demography.csv --seed 12345 --output coalescent_results.tsv
```

**Configuration File (`coalescent_config.csv`):**
```
parameter,value
mutation_rate,1e-8
sequence_length,1000
sample_size,20
replicates,100
```

**Demographics File (`coalescent_demography.csv`):**
```
time_ago,population_size
0,10000
50,100
100,10000
```

**Key Requirements:**
- Generate coalescent trees under variable population size model
- Place mutations on branches using Poisson process proportional to branch length
- Calculate tree-based estimates of genetic diversity
- Account for demographic history in coalescent rates
- Run multiple replicates and summarize results

**Output Format (`coalescent_results.tsv`):**
```
replicate	total_tree_length	time_to_mrca	num_mutations	theta_estimate	nucleotide_diversity
1	2.456	0.234	45	0.0123	0.0234
2	2.678	0.267	52	0.0143	0.0267
...
```

## Technical Requirements

### Required Environment

Both scripts must run in the provided conda environment: `monte_carlo.yaml`

You may use the following Python libraries (all included in the environment):
- **numpy** - for numerical operations and random number generation
- **pandas** - for data manipulation and file output
- **networkx** - for graph data structure and algorithms
- Standard library modules (csv, argparse, etc.)
- **DO NOT** use specialized population genetics libraries (tskit, msprime, etc.)

## Submission Requirements

Submit both Python scripts:
1. `firstname_lastname_monte_carlo_part1.py` - Wright-Fisher simulation
2. `firstname_lastname_monte_carlo_part2.py` - Coalescent simulation

Both scripts must:
- Run in the provided `monte_carlo.yaml` conda environment
- Accept the config file and required command line arguments
- Generate the specified output TSV files
- Use the provided random seed for reproducible results