from abc import ABC, abstractmethod


class DirectoryService(ABC):

    @abstractmethod
    def account_status(
        self,
        user_id: str,
    ) -> dict:
        pass

    @abstractmethod
    def check_access(
        self,
        user_id: str,
        resource: str,
    ) -> dict:
        pass

    @abstractmethod
    def unlock_user(
        self,
        user_id: str,
    ) -> dict:
        pass

    @abstractmethod
    def reset_password(
        self,
        user_id: str,
    ) -> dict:
        pass