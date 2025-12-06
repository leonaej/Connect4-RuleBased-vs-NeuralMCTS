# Connect4-RuleBased-vs-NeuralMCTS
A comparative study of two Connect Four AI agents: a rule-based heuristic bot using minimax with alpha-beta pruning, and an AlphaZero-style neural MCTS agent trained through self-play.




## Quick Start

### Prerequisites

Install the requirements mentioned in the requirement.txt file.



## Playing Games

### 1. Human vs Heuristic
Play against the minimax heuristic agent:

```bash
python main/play_human_vs_heuristic.py
```




### 2. Human vs AlphaZero
Play against the trained AlphaZero agent:

```bash
python main/play_human_vs_alphazero.py
```





### 3. Agent vs Agent Matches

#### AlphaZero vs Heuristic
```bash
python main/play_alphazero_vs_heuristic.py
```

Evaluate AlphaZero's performance against the heuristic agent.

---

#### Heuristic vs Random (Benchmark)
```bash
python main/play_heuristic_vs_random.py
```
Expected Result: Heuristic should wins 100% of 400 games

---

#### Network1 vs Network2 Comparison
```bash
python main/play_net1_vs_net2.py
```

Compare the two network architectures:



**Requirements:** Both `logs/` and `logs2/` must contain trained models

---

#### Network2 vs Network2 Boosted
```bash
python main/play_net2_vs_net2_boosted.py
```

Compare standard Network2 vs heuristic-boosted Network2.

---

#### AlphaZero Self-Play
```bash
python main/play_alphazero_selfplay.py
```

Watch AlphaZero play against itself.

---

#### AlphaZero vs AlphaZero (Different Iterations)
```bash
python main/play_alphazero_vs_alphazero.py
```

Pit different training iterations against each other.

---



## License

This project is open source and available under the MIT License.

---



## Contact

For questions or issues, please open a GitHub issue.

---
