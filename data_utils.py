import torch

def get_stationary_distribution(P: torch.Tensor, n_order: int, num_symbols: int, steps: int = 50) -> torch.Tensor:
    """
    計算轉移矩陣的平穩分佈。
    如果是 N-gram (n_order > 1)，會先進行矩陣擴張。
    """
    num_states = num_symbols ** n_order
    if n_order == 1:
        T = P
    else:
        # 建立大轉移矩陣 T [S^N, S^N]
        T = torch.zeros((num_states, num_states), device=P.device)
        for i in range(num_states):
            # 根據當前歷史 i，計算下一個 N-gram 可能的 base index
            base_idx = (i % (num_symbols ** (n_order - 1))) * num_symbols
            for j in range(num_symbols):
                T[i, base_idx + j] = P[i, j]
                
    # 冪次法逼近
    T_n = torch.linalg.matrix_power(T, steps)
    return T_n[0]

def get_index_from_history(history: torch.Tensor, powers: torch.Tensor) -> int:
    """利用進位制權重極速將序列轉為索引"""
    return (history * powers).sum().item()
