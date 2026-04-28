import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from connect4_env.connect4_env import Connect4Env
from agents.alphazero.az_network2 import AZNetwork2
from agents.alphazero.alphazero_agent import AlphaZeroAgent
from agents.heuristic_agent import HeuristicAgent
from training.replay_buffer import ReplayBuffer
from training.play_one_game import play_one_game_with_mcts, encode_board
from training.train_one_iter import train_one_iteration


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GAMES_PER_ITER   = 60       # 30 as P1, 30 as P2
BATCH_SIZE       = 64
TRAIN_STEPS      = 100
NUM_SIMULATIONS  = 200
C_PUCT           = 1.5
BUFFER_CAPACITY  = 30000
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR          = "logs3"

# Phase-advance thresholds
SELFPLAY_BALANCE_TOL = 0.10   # |p1_rate - p2_rate| must be < this
HEURISTIC_WIN_THRESH = 0.60   # az win rate must reach this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def play_game_vs_heuristic(env, az_agent, heuristic_agent, buffer,
                            az_player, temperature=1.0):
    env.reset()
    game_states = []

    # If AlphaZero goes second, let heuristic make the first move
    if az_player == -1:
        action = heuristic_agent.select_action(env)
        if action is None or action not in env.get_valid_actions():
            action = env.get_valid_actions()[0]
        env.step(int(action))

    move_count = 0
    winner = 0

    while move_count < 42:
        board = env.board.copy()
        current_player = env.current_player

        if current_player == az_player:
            action, pi = az_agent.policy_from_root(board, current_player, temperature)
            if hasattr(action, 'item'):
                action = action.item()
            action = int(action)
            game_states.append((encode_board(board, current_player), pi, current_player))
        else:
            action = heuristic_agent.select_action(env)
            if action is None or action not in env.get_valid_actions():
                valid = env.get_valid_actions()
                if not valid:
                    winner = 0
                    break
                action = valid[0]
            action = int(action)

        player_who_moved = current_player
        _, reward, done = env.step(action)
        move_count += 1

        if env.check_win(player_who_moved):
            winner = player_who_moved
            break
        if env.is_draw():
            winner = 0
            break

    for state, pi, player in game_states:
        value = 0.0 if winner == 0 else (1.0 if winner == az_player else -1.0)
        buffer.push(state, pi, value)

    return winner


def run_self_play_iteration(env, agent, buffer, temperature):
    p1_wins = p2_wins = draws = 0
    half = GAMES_PER_ITER // 2

    for _ in range(half):
        # AlphaZero as P1
        env.reset()
        winner = play_one_game_with_mcts(env, agent, buffer, temperature)
        if winner == 1:
            p1_wins += 1
        elif winner == -1:
            p2_wins += 1
        else:
            draws += 1

    for _ in range(half):
        # AlphaZero as P2 — same model, just track who wins
        env.reset()
        winner = play_one_game_with_mcts(env, agent, buffer, temperature)
        # In self-play the "first mover" is always player 1 internally;
        # we accumulate separately to measure balance
        if winner == 1:
            p1_wins += 1
        elif winner == -1:
            p2_wins += 1
        else:
            draws += 1

    total = p1_wins + p2_wins + draws
    p1_rate = p1_wins / total
    p2_rate = p2_wins / total
    return p1_wins, p2_wins, draws, p1_rate, p2_rate


def run_heuristic_iteration(env, az_agent, heuristic_agent, buffer, temperature):
    az_wins = heuristic_wins = draws = 0
    half = GAMES_PER_ITER // 2

    for _ in range(half):
        winner = play_game_vs_heuristic(env, az_agent, heuristic_agent,
                                        buffer, az_player=1,
                                        temperature=temperature)
        if winner == 1:
            az_wins += 1
        elif winner == -1:
            heuristic_wins += 1
        else:
            draws += 1

    for _ in range(half):
        winner = play_game_vs_heuristic(env, az_agent, heuristic_agent,
                                        buffer, az_player=-1,
                                        temperature=temperature)
        if winner == -1:
            az_wins += 1
        elif winner == 1:
            heuristic_wins += 1
        else:
            draws += 1

    az_rate = az_wins / GAMES_PER_ITER
    return az_wins, heuristic_wins, draws, az_rate


def should_advance_phase(phase, p1_rate=None, p2_rate=None, az_rate=None):
    if phase == 1:
        return abs(p1_rate - p2_rate) < SELFPLAY_BALANCE_TOL
    elif phase in (2, 3):
        return az_rate >= HEURISTIC_WIN_THRESH
    return False  # phase 4 runs indefinitely


def temperature_for(iter_in_phase):
    return 1.0 if iter_in_phase < 10 else 0.3


def save_logs(log_data):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "curriculum_logs.json"), "w") as f:
        json.dump(log_data, f, indent=4)


def plot_progress(log_data):
    entries = log_data["iterations"]
    if len(entries) < 2:
        return

    phases  = [e["phase"] for e in entries]
    iters   = list(range(1, len(entries) + 1))
    losses  = [e["loss"] for e in entries]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Loss
    axes[0].plot(iters, losses, 'b-', linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Global Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # Self-play balance (phase 1)
    sp = [e for e in entries if e["phase"] == 1]
    if sp:
        sp_x = [entries.index(e) + 1 for e in sp]
        axes[1].plot(sp_x, [e["p1_rate"] for e in sp], label="P1 win rate", linewidth=2)
        axes[1].plot(sp_x, [e["p2_rate"] for e in sp], label="P2 win rate", linewidth=2)
        axes[1].set_title("Phase 1 — Self-play Balance")
        axes[1].set_xlabel("Global Iteration")
        axes[1].set_ylabel("Win Rate")
        axes[1].legend()
        axes[1].grid(True)

    # Heuristic win rate (phases 2-4)
    hr = [e for e in entries if e["phase"] > 1]
    if hr:
        hr_x = [entries.index(e) + 1 for e in hr]
        axes[2].plot(hr_x, [e["az_rate"] for e in hr], 'g-', linewidth=2)
        axes[2].axhline(y=HEURISTIC_WIN_THRESH, color='r', linestyle='--',
                        alpha=0.5, label=f'{int(HEURISTIC_WIN_THRESH*100)}% threshold')
        axes[2].set_title("AlphaZero Win Rate vs Heuristic")
        axes[2].set_xlabel("Global Iteration")
        axes[2].set_ylabel("Win Rate")
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "curriculum_curves.png"), dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Curriculum Training — Connect4 AlphaZero (AZNetwork2)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Phases:")
    print(f"  1. Self-play       → advance when |P1_rate - P2_rate| < {SELFPLAY_BALANCE_TOL}")
    print(f"  2. Heuristic d=1   → advance when AZ win rate >= {HEURISTIC_WIN_THRESH}")
    print(f"  3. Heuristic d=2   → advance when AZ win rate >= {HEURISTIC_WIN_THRESH}")
    print(f"  4. Heuristic d=3   → train until stopped")
    print("=" * 70)

    os.makedirs(LOG_DIR, exist_ok=True)

    env    = Connect4Env()
    net    = AZNetwork2()
    agent  = AlphaZeroAgent(env=env, network=net,
                             num_simulations=NUM_SIMULATIONS,
                             c_puct=C_PUCT, device=DEVICE)
    buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

    # Heuristic agents for each phase (created lazily when needed)
    heuristic_agents = {
        2: HeuristicAgent(depth=1),
        3: HeuristicAgent(depth=2),
        4: HeuristicAgent(depth=3),
    }

    log_data = {"iterations": []}

    phase          = 1
    global_iter    = 0
    iter_in_phase  = 0

    phase_labels = {1: "Self-play", 2: "Heuristic d=1",
                    3: "Heuristic d=2", 4: "Heuristic d=3"}

    try:
        while phase <= 4:
            global_iter   += 1
            iter_in_phase += 1
            temp = temperature_for(iter_in_phase)

            print(f"\n{'='*70}")
            print(f"Global Iter {global_iter} | Phase {phase}: {phase_labels[phase]} "
                  f"| Iter in phase: {iter_in_phase} | Temp: {temp:.1f}")
            print(f"{'='*70}")

            p1_rate = p2_rate = az_rate = None

            if phase == 1:
                print(f"Playing {GAMES_PER_ITER} self-play games...")
                p1_wins, p2_wins, draws, p1_rate, p2_rate = run_self_play_iteration(
                    env, agent, buffer, temp)
                print(f"  P1 wins: {p1_wins}  P2 wins: {p2_wins}  Draws: {draws}")
                print(f"  P1 rate: {p1_rate:.3f}  P2 rate: {p2_rate:.3f}  "
                      f"Gap: {abs(p1_rate - p2_rate):.3f}")
            else:
                h_agent = heuristic_agents[phase]
                print(f"Playing {GAMES_PER_ITER} games vs {phase_labels[phase]}...")
                az_wins, h_wins, draws, az_rate = run_heuristic_iteration(
                    env, agent, h_agent, buffer, temp)
                print(f"  AZ wins: {az_wins}  Heuristic wins: {h_wins}  Draws: {draws}")
                print(f"  AZ win rate: {az_rate:.3f}  (threshold: {HEURISTIC_WIN_THRESH})")

            # Train
            loss = train_one_iteration(agent, buffer,
                                       batch_size=BATCH_SIZE, steps=TRAIN_STEPS)
            print(f"  Training loss: {loss:.4f}  Buffer: {len(buffer)}")

            # Save checkpoint
            ckpt_path = os.path.join(LOG_DIR, f"model_phase{phase}_iter{global_iter}.pt")
            torch.save(agent.network.state_dict(), ckpt_path)
            print(f"  Saved: {ckpt_path}")

            # Log
            entry = {
                "global_iter":   global_iter,
                "phase":         phase,
                "iter_in_phase": iter_in_phase,
                "loss":          loss,
                "p1_rate":       p1_rate,
                "p2_rate":       p2_rate,
                "az_rate":       az_rate,
            }
            log_data["iterations"].append(entry)
            save_logs(log_data)
            plot_progress(log_data)

            # Check phase advance
            if should_advance_phase(phase, p1_rate=p1_rate,
                                    p2_rate=p2_rate, az_rate=az_rate):
                print(f"\n>>> Phase {phase} complete! Advancing to Phase {phase + 1}...")
                phase         += 1
                iter_in_phase  = 0
                if phase <= 4:
                    print(f">>> Now: {phase_labels[phase]}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    print("\n" + "=" * 70)
    print("Training finished.")
    print(f"Logs:  {LOG_DIR}/curriculum_logs.json")
    print(f"Plot:  {LOG_DIR}/curriculum_curves.png")
    print(f"Models saved as {LOG_DIR}/model_phase{{N}}_iter{{M}}.pt")
    print("=" * 70)
