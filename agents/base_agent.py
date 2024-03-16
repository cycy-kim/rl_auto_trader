from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def cuda(self):
        pass

    @abstractmethod
    def save(self, filename):
        """
        에이전트의 모델 저장
        """
        pass

    @abstractmethod
    def load(self, filename):
        """
        저장된 모델 로드
        """
        pass
