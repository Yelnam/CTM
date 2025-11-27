# CTM Visualizer

An interactive visualizer for understanding **Continuous Thought Machines** (CTMs), a novel neural network architecture from Sakana AI

🔗 **[Live Demo](https://yelnam.github.io/CTM/)** 

## What is it?

This is an educational tool that lets you watch a CTM "think." You can:

- See how the network's 12 neurons evolve over 6 computational ticks
- Click neurons to inspect their private internal networks (NLMs)
- Click connections to see how sync pairs contribute to the final prediction
- Step through ticks to watch confidence build (or waver)

The model is trained on simple arithmetic sequences: given [2, 4, 6], predict 8.

## What's a CTM?

Continuous Thought Machines are a new architecture introduced in [Sakana AI's 2025 paper](https://arxiv.org/abs/2505.05522). Two key innovations make them different from standard neural networks:

1. **Neuron-Level Models (NLMs)**: Each neuron has its own private neural network that processes its recent history. This creates diverse, complex dynamics across the population

2. **Synchronization-based representation**: Instead of reading individual neuron activations, CTMs derive their output from *correlations between neuron pairs over time*. The prediction emerges from relationships, not individuals

The result is a network that "thinks" through multiple iterations, with adaptive computation emerging naturally — easy inputs settle quickly, hard inputs keep the dynamics churning

## This is a toy model

Our visualizer uses a tiny 12-neuron, 6-tick model with ~8,000 parameters. The original paper uses 256–4,096 neurons and 50–100 ticks. We've simplified things to make the architecture graspable, not to demonstrate full CTM capabilities

The model has essentially memorized its training sequences. It doesn't truly generalize. But it faithfully implements the CTM architecture, making it useful for understanding *how* CTMs work

## Running locally

```bash
# Clone the repo
git clone https://github.com/yelnam/CTM.git
cd ctm-visualizer

# Serve the files (any static server works)
python -m http.server 8000

# Open http://localhost:8000 in your browser
```

## Training your own model

The `training/` folder contains everything needed to train a CTM from scratch:

```bash
# Generate training data
python training/generate_sequence_data.py

# Train the model
python training/train_ctm_sequence.py

# Export weights for the browser
python training/export_model.py
```

### Key files

| File | Description |
|------|-------------|
| `training/tiny_ctm.py` | The CTM model implementation in PyTorch |
| `training/generate_sequence_data.py` | Creates arithmetic sequence datasets |
| `training/train_ctm_sequence.py` | Training loop with logging |
| `training/export_model.py` | Converts PyTorch weights to JSON for browser |

### Model architecture defaults (adjustable during training)

```
Neurons:        12
Ticks:          6
Memory length:  4 (pre-activation history)
NLM hidden:     6 (per-neuron MLP width)
Sync pairs:     78 (all unique pairs from 12 neurons)
Total params:   ~8,000
```

## How the visualizer works

The frontend is a single HTML file with vanilla JavaScript — no build step, no dependencies (except KaTeX for equation rendering). The model runs entirely in the browser:

1. `model_weights.json` contains all learned parameters
2. On load, weights are parsed into typed arrays
3. Inference runs in JavaScript, computing all 6 ticks
4. Canvas renders the 3D visualization with manual depth sorting

### The one-shot version

We've included `index_oneshot.html` in the repo — this is the very first version of the visualizer, generated in a single prompt to Claude. Included here in order to demonstrate how much can be achieved "one shot"

Comparing it to `index.html` shows how the project evolved through iteration: adding the contribution chart, refining explanations, improving the UI, and fixing edge cases. We left it in as a record of where we started

## Credits

- **CTM architecture**: [Continuous Thought Machines](https://arxiv.org/abs/2505.05522) by Luke Darlow, Ciaran Regan, Sebastian Risi, Jeffrey Seely, and Llion Jones at Sakana AI
- **Interactive demos**: See [Sakana AI's project page](https://pub.sakana.ai/ctm/) for their official demos

This visualizer was built as a learning project to understand the CTM architecture by implementing it from scratch. It's not affiliated with Sakana AI

## License

MIT — do what you like with it