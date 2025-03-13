In the genetic algorithm implementation for the battle environment, mutations happen at multiple levels throughout the evolutionary process. Let me clarify how mutations work in this context:

## Mutation Frequency

### Per Individual

Each individual (decision tree) undergoes mutation with a certain probability after it's selected for the next generation. The typical settings are:

- **Mutation rate**: Typically set to around 0.2 (20%)
- This means each node in the tree has a 20% chance of being mutated
- For a tree with ~10-15 nodes, you might see 2-3 nodes mutated per individual

### Per Generation

In each generation:
- The elite individuals (usually top 5) are preserved without mutation
- The remaining individuals (typically 45 in a population of 50) undergo possible mutation
- This means roughly 45 individuals × 2-3 mutations per individual = ~90-135 mutation events per generation

### Over the Full Training

For a typical run with 100 generations:
- Total mutation events: ~9,000-13,500 across the entire training process
- But remember many of these mutations may be reverted or further modified in subsequent generations

## Types of Mutations

The implementation includes three main types of mutations:

1. **Feature mutations** (changing which game state aspect is considered)
2. **Threshold mutations** (adjusting the decision boundaries)
3. **Subtree replacement** (completely replacing parts of the decision tree)

The relative frequency is roughly:
- 40% feature/threshold changes in decision nodes
- 20% action value changes in leaf nodes
- 5% complete subtree replacements (more dramatic changes)

## Mutation vs. Episodes

It's important to clarify the terminology:

- **Episodes**: Individual battle simulations used to evaluate fitness (not directly related to mutations)
- **Generations**: Complete cycles of evaluation, selection, crossover, and mutation

Mutations happen between generations, not episodes. Each individual might be tested in ~5 episodes to calculate its fitness, but mutations only happen during reproduction between generations.

## Practical Impact

The mutation rate (0.2) and other parameters were likely chosen through experimentation:
- Too low (e.g., 0.05) → slow exploration of possible strategies
- Too high (e.g., 0.5) → too much randomness, disrupting good solutions

The mutation system is designed to balance exploration (finding new strategies) with exploitation (refining good strategies). Over 100-200 generations, this approach allows the algorithm to discover increasingly effective decision trees for the battle environment.
