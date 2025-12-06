
import numpy as np
import torch
from connect4_env.connect4_env import Connect4Env
from agents.alphazero.az_network2 import AZNetwork2
from agents.alphazero.alphazero_agent import AlphaZeroAgent


def print_test(name, passed):
    status = " PASS" if passed else " FAIL"
    print(f"{status}: {name}")


def test_1_obvious_win():
    """Test: MCTS should play winning move when one exists"""
    print("\n" + "="*70)
    print("TEST 1: Does MCTS Choose Obvious Winning Move?")
    print("="*70)
    
    env = Connect4Env()
    network = AZNetwork2()
    agent = AlphaZeroAgent(env, network, num_simulations=50, device="cpu")
    
    # Setup board with obvious win in column 3
    env.board = np.array([
        [0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  1,  0,  0,  0],
        [0,  0,  0,  1,  0,  0,  0],
        [0,  0,  0,  1,  0,  0,  0],
    ])
    env.current_player = 1
    
    print("\nBoard (Player 1's turn - can win in column 3):")
    print_board(env.board)
    
    action, policy = agent.policy_from_root(env.board, 1, temperature=0.0)
    
    print(f"\nMCTS chose column: {action}")
    print(f"Policy distribution: {policy}")
    print(f"Highest probability column: {np.argmax(policy)}")
    
    passed = (action == 3)
    print_test("MCTS plays winning move", passed)
    
    return passed


def test_2_block_opponent():
    """Test: MCTS should block opponent's winning move"""
    print("\n" + "="*70)
    print("TEST 2: Does MCTS Block Opponent's Win?")
    print("="*70)
    
    env = Connect4Env()
    network = AZNetwork2()
    agent = AlphaZeroAgent(env, network, num_simulations=50, device="cpu")
    
    # Setup board where opponent (Player -1) threatens win in column 3
    env.board = np.array([
        [0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0, -1,  0,  0,  0],
        [0,  0,  0, -1,  0,  0,  0],
        [0,  0,  0, -1,  0,  0,  0],
    ])
    env.current_player = 1  # Player 1's turn
    
    print("\nBoard (Player 1's turn - must block column 3):")
    print_board(env.board)
    
    action, policy = agent.policy_from_root(env.board, 1, temperature=0.0)
    
    print(f"\nMCTS chose column: {action}")
    print(f"Policy distribution: {policy}")
    
    passed = (action == 3)
    print_test("MCTS blocks opponent's winning move", passed)
    
    return passed


def test_3_policy_sums_to_one():
    """Test: Policy distribution should sum to 1.0"""
    print("\n" + "="*70)
    print("TEST 3: Does Policy Sum to 1.0?")
    print("="*70)
    
    env = Connect4Env()
    network = AZNetwork2()
    agent = AlphaZeroAgent(env, network, num_simulations=50, device="cpu")
    
    env.reset()
    action, policy = agent.policy_from_root(env.board, 1, temperature=1.0)
    
    policy_sum = np.sum(policy)
    print(f"\nPolicy sum: {policy_sum}")
    print(f"Policy: {policy}")
    
    passed = abs(policy_sum - 1.0) < 0.01
    print_test("Policy sums to 1.0", passed)
    
    return passed


def test_4_only_valid_moves():
    """Test: MCTS should only suggest valid moves"""
    print("\n" + "="*70)
    print("TEST 4: Does MCTS Only Consider Valid Moves?")
    print("="*70)
    
    env = Connect4Env()
    network = AZNetwork2()
    agent = AlphaZeroAgent(env, network, num_simulations=50, device="cpu")
    
    # Fill column 3 completely
    env.board = np.array([
        [0,  0,  0,  1,  0,  0,  0],
        [0,  0,  0, -1,  0,  0,  0],
        [0,  0,  0,  1,  0,  0,  0],
        [0,  0,  0, -1,  0,  0,  0],
        [0,  0,  0,  1,  0,  0,  0],
        [0,  0,  0, -1,  0,  0,  0],
    ])
    env.current_player = 1
    
    print("\nBoard (Column 3 is full - invalid move):")
    print_board(env.board)
    
    action, policy = agent.policy_from_root(env.board, 1, temperature=0.0)
    
    print(f"\nMCTS chose column: {action}")
    print(f"Policy distribution: {policy}")
    print(f"Policy for invalid column 3: {policy[3]}")
    
    # MCTS should never choose column 3, and ideally policy[3] should be 0
    passed = (action != 3)
    print_test("MCTS avoids invalid moves", passed)
    
    if policy[3] > 0.01:
        print(f"  WARNING: Policy gives {policy[3]:.3f} probability to invalid move")
    
    return passed


def test_5_value_range():
    """Test: Value predictions should be in [-1, 1]"""
    print("\n" + "="*70)
    print("TEST 5: Are Value Predictions in Valid Range?")
    print("="*70)
    
    network = AZNetwork2()
    
    # Random board state
    state = torch.randn(1, 2, 6, 7)
    
    with torch.no_grad():
        policy_logits, value = network(state)
    
    value = value.item()
    print(f"\nValue prediction: {value}")
    
    passed = (-1.0 <= value <= 1.0)
    print_test("Value in range [-1, 1]", passed)
    
    return passed


def test_6_visit_counts_increase():
    """Test: More simulations = more confident policy"""
    print("\n" + "="*70)
    print("TEST 6: Do More Simulations Improve Confidence?")
    print("="*70)
    
    env = Connect4Env()
    network = AZNetwork2()
    
    env.reset()
    
    # Few simulations
    agent_weak = AlphaZeroAgent(env, network, num_simulations=10, device="cpu")
    _, policy_weak = agent_weak.policy_from_root(env.board, 1, temperature=0.0)
    
    # Many simulations
    agent_strong = AlphaZeroAgent(env, network, num_simulations=100, device="cpu")
    _, policy_strong = agent_strong.policy_from_root(env.board, 1, temperature=0.0)
    
    # More simulations should concentrate probability more
    entropy_weak = -np.sum(policy_weak * np.log(policy_weak + 1e-10))
    entropy_strong = -np.sum(policy_strong * np.log(policy_strong + 1e-10))
    
    print(f"\n10 simulations - Policy entropy: {entropy_weak:.3f}")
    print(f"100 simulations - Policy entropy: {entropy_strong:.3f}")
    
    passed = (entropy_strong < entropy_weak)
    print_test("More simulations = more confident", passed)
    
    return passed


def print_board(board):
    """Pretty print board"""
    print("\n  0 1 2 3 4 5 6")
    print(" ┌─────────────┐")
    for row in board:
        print(" │", end="")
        for cell in row:
            if cell == 1:
                print(" X", end="")
            elif cell == -1:
                print(" O", end="")
            else:
                print(" .", end="")
        print(" │")
    print(" └─────────────┘")


if __name__ == "__main__":
    print("="*70)
    print("MCTS CORRECTNESS TEST SUITE")
    print("="*70)
    print("\nTesting MCTS implementation for basic correctness...")
    print("Note: Some tests may fail with untrained network, but logic should work")
    
    results = []
    
    # Run all tests
    results.append(("Obvious Win", test_1_obvious_win()))
    results.append(("Block Opponent", test_2_block_opponent()))
    results.append(("Policy Sums to 1", test_3_policy_sums_to_one()))
    results.append(("Only Valid Moves", test_4_only_valid_moves()))
    results.append(("Value Range", test_5_value_range()))
    results.append(("Simulation Count", test_6_visit_counts_increase()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "passed" if passed else "not passed"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n ALL TESTS PASSED! MCTS implementation looks correct!")
    elif passed_count >= 4:
        print("\n  Most tests passed. MCTS logic is mostly correct.")
        print("   Failures might be due to untrained network (expected)")
    else:
        print("\n Multiple test failures. MCTS implementation may have issues.")
    
    print("\nNote: Tests 1-2 may fail with random network (not a bug!)")
    print("      Tests 3-6 should always pass (core MCTS logic)")