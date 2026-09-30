"""Implementação do Baseline e Thompson Sampling"""

import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class BaselinePolicy:
    """Política determinística: oferece sempre o melhor braço histórico"""
    
    def __init__(self, num_arms: int = 3):
        """
        Inicializa Baseline
        
        Args:
            num_arms: número de ofertas/braços
        """
        self.num_arms = num_arms
        self.arm_rewards = np.zeros(num_arms)
        self.arm_counts = np.zeros(num_arms)
        self.best_arm = 0
        
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Treina policy com dados históricos
        
        Args:
            y_true: conversões reais (0 ou 1)
            y_pred: braço/oferta escolhida para cada cliente
        """
        for arm in range(self.num_arms):
            mask = y_pred == arm
            if mask.sum() > 0:
                self.arm_counts[arm] = mask.sum()
                self.arm_rewards[arm] = y_true[mask].mean()
        
        # Melhor braço é aquele com maior taxa de conversão
        self.best_arm = np.argmax(self.arm_rewards)
        
        logger.info(f"Baseline treinado:")
        for arm in range(self.num_arms):
            logger.info(f"  Braço {arm}: {self.arm_counts[arm]:.0f} exposições, "
                       f"{self.arm_rewards[arm]:.2%} conversão")
        logger.info(f"  Melhor braço: {self.best_arm}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna o melhor braço para cada cliente
        
        Args:
            X: features dos clientes
            
        Returns:
            Array com braço recomendado para cada cliente
        """
        return np.full(X.shape[0], self.best_arm)
    
    def get_conversion_rate(self) -> float:
        """Retorna taxa de conversão do baseline"""
        return self.arm_rewards[self.best_arm]


class ThompsonSampling:
    """Thompson Sampling: exploração Bayesiana com distribuições Beta"""
    
    def __init__(self, num_arms: int = 3, alpha_prior: float = 1.0, 
                 beta_prior: float = 1.0, random_state: int = 42):
        """
        Inicializa Thompson Sampling
        
        Args:
            num_arms: número de ofertas/braços
            alpha_prior: parâmetro alpha da distribuição Beta (prévio)
            beta_prior: parâmetro beta da distribuição Beta (prévio)
            random_state: seed para reprodutibilidade
        """
        self.num_arms = num_arms
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.random_state = random_state
        
        # Estado de cada braço
        self.alpha = np.full(num_arms, alpha_prior, dtype=float)
        self.beta = np.full(num_arms, beta_prior, dtype=float)
        self.arm_counts = np.zeros(num_arms)
        self.arm_rewards = np.zeros(num_arms)
        
        np.random.seed(random_state)
        
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Treina policy com dados históricos
        
        Args:
            y_true: conversões reais (0 ou 1)
            y_pred: braço/oferta escolhida para cada cliente
        """
        for arm in range(self.num_arms):
            mask = y_pred == arm
            if mask.sum() > 0:
                conversions = y_true[mask].sum()
                rejections = mask.sum() - conversions
                
                # Atualizar parâmetros Beta
                self.alpha[arm] += conversions
                self.beta[arm] += rejections
                self.arm_counts[arm] = mask.sum()
                self.arm_rewards[arm] = y_true[mask].mean()
        
        logger.info(f"Thompson Sampling treinado:")
        for arm in range(self.num_arms):
            logger.info(f"  Braço {arm}: α={self.alpha[arm]:.0f}, β={self.beta[arm]:.0f}, "
                       f"conversão={self.arm_rewards[arm]:.2%}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Recomenda braço para cada cliente usando Thompson Sampling
        
        Args:
            X: features dos clientes
            
        Returns:
            Array com braço recomendado para cada cliente
        """
        n_samples = X.shape[0]
        recommendations = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Amostra da distribuição Beta para cada braço
            samples = np.random.beta(self.alpha, self.beta)
            # Escolhe braço com maior amostra
            recommendations[i] = np.argmax(samples)
        
        return recommendations
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna probabilidade esperada de conversão para cada braço
        
        Args:
            X: features dos clientes
            
        Returns:
            Array (n_samples, num_arms) com probabilidades
        """
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.num_arms))
        
        for arm in range(self.num_arms):
            # E[X] da Beta(α, β) = α/(α+β)
            probas[:, arm] = self.alpha[arm] / (self.alpha[arm] + self.beta[arm])
        
        return probas
    
    def update(self, arm: int, reward: int):
        """
        Atualiza parâmetros Beta de um braço com nova observação
        
        Args:
            arm: índice do braço
            reward: 1 se conversão, 0 caso contrário
        """
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        
        self.arm_counts[arm] += 1
        
        # Atualizar taxa de conversão média
        old_reward = self.arm_rewards[arm]
        self.arm_rewards[arm] = (
            (old_reward * (self.arm_counts[arm] - 1) + reward) / self.arm_counts[arm]
        )
    
    def get_conversion_rate(self) -> float:
        """Retorna taxa de conversão esperada (média dos braços)"""
        expected_rates = self.alpha / (self.alpha + self.beta)
        return np.mean(expected_rates)


def evaluate_strategy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de avaliação da estratégia
    
    Args:
        y_true: conversões reais
        y_pred: braços/ofertas recomendadas
        
    Returns:
        Dict com métricas calculadas
    """
    n_samples = len(y_true)
    n_unique_arms = len(np.unique(y_pred))
    
    # Taxa de conversão geral
    conversion_rate = y_true.mean()
    
    # Recompensa acumulada
    total_reward = y_true.sum()
    
    # Taxa de exploração (não escolhendo o melhor braço)
    arm_rates = {
        int(arm): y_true[y_pred == arm].mean()
        for arm in np.unique(y_pred)
    }
    best_arm = max(arm_rates, key=arm_rates.get)
    exploration_rate = (y_pred != best_arm).mean()
    
    metrics = {
        'conversion_rate': conversion_rate,
        'total_reward': total_reward,
        'exploration_rate': exploration_rate,
        'n_samples': n_samples,
        'n_unique_arms': n_unique_arms
    }
    
    return metrics


def simulate_bandit(X_train: np.ndarray, y_train: np.ndarray, 
                    policy, batch_size: int = 100) -> Dict:
    """
    Simula execução do bandit em mini-batches
    
    Args:
        X_train: features de treino
        y_train: conversões de treino
        policy: política (Thompson Sampling ou Baseline)
        batch_size: tamanho dos batches para simulação
        
    Returns:
        Dict com histórico de execução
    """
    n_batches = (len(X_train) + batch_size - 1) // batch_size
    history = {
        'cumulative_reward': [],
        'cumulative_regret': [],
        'exploration_rate': []
    }
    
    cumulative_reward = 0
    cumulative_regret = 0
    best_policy_reward = 0  # Baseline para regret
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(X_train))
        
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        
        # Recomendação
        y_pred = policy.predict(X_batch)
        
        # Recompensa obtida
        rewards_obtained = y_batch
        cumulative_reward += rewards_obtained.sum()
        
        # Regret: diferença com a melhor política possível
        best_arm = np.argmax(policy.arm_rewards) if hasattr(policy, 'arm_rewards') else 0
        best_possible_reward = (y_batch == 1).sum()  # Se todos fossem conversões
        cumulative_regret += best_possible_reward - rewards_obtained.sum()
        
        # Taxa de exploração
        exploration = (y_pred != np.argmax(policy.arm_rewards)).mean() \
                     if hasattr(policy, 'arm_rewards') else 0
        
        history['cumulative_reward'].append(cumulative_reward)
        history['cumulative_regret'].append(cumulative_regret)
        history['exploration_rate'].append(exploration)
    
    return history
