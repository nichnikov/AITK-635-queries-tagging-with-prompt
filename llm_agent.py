# бесчеловечный код:
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from datetime import datetime

class AIClient(ABC):
    """
    Абстрактный класс для взаимодействия с AI API.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Выполняет валидацию запроса через AI API.

        :param prompt: Строка запроса для валидации.
        :param kwargs: Дополнительные параметры для запроса.
        :return: Ответ модели в виде строки.
        """
        pass


class OpenAIClient(AIClient):
    """
    Реализация AI-клиента для работы с OpenAI API.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.vsegpt.ru/v1"):
        """
        Инициализирует клиента OpenAI.

        :param api_key: API ключ для доступа к OpenAI.
        :param base_url: Базовый URL API.
        """
        from openai import OpenAI  # Локальный импорт для уменьшения зависимости
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Выполняет валидацию запроса через OpenAI API.

        :param prompt: Строка запроса для валидации.
        :param kwargs: Дополнительные параметры для запроса.
        :return: Ответ модели в виде строки.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")


class AIAgent:
    """
    Класс для валидации ответов с использованием AI клиента.
    """

    def __init__(self, ai_client: AIClient, logger: logging.Logger = None):
        """
        Инициализирует агент с указанным AI клиентом.

        :param ai_client: Экземпляр AI клиента.
        :param logger: Логгер для записи событий.
        """
        self.ai_client = ai_client
        self.logger = logger or logging.getLogger(__name__)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Выполняет валидацию ответа через AI клиента.

        :param prompt: Строка запроса для валидации.
        :param kwargs: Дополнительные параметры для клиента AI.
        :return: Ответ модели в виде строки.
        """
        try:
            return self.ai_client.generate(prompt, **kwargs)
        except RuntimeError as e:
            self.logger.error(f"Validation failed: {e}")
            return "An error occurred during validation."

    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Позволяет использовать объект класса как функцию для валидации.

        :param prompt: Строка запроса для валидации.
        :param kwargs: Дополнительные параметры для клиента AI.
        :return: Ответ модели в виде строки.
        """
        return self.generate(prompt, **kwargs)


# Пример использования
if __name__ == "__main__":
    import os
    import logging
    from dotenv import load_dotenv

    dotenv_path = Path('/home/an/Data/github/AITK611-expert-bot-service-with-baai-openai/data/.env')
    load_dotenv(dotenv_path=dotenv_path)

    # Настройка логгера
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AIAgent")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API ключ не найден. Убедитесь, что файл .env содержит переменную OPENAI_API_KEY.")


    # Инициализация клиента и агента
    openai_client = OpenAIClient(api_key=api_key)
    ai_agent = AIAgent(ai_client=openai_client, logger=logger)

    start_time = datetime.now() 
    # Использование агента
    prompt = """
    запросы: {}
    
    Посмотри на запросы выше и определи, к какой группе они относятся.

    Группа 1

    Нормативный правовой акт — это письменный официальный документ, принятый (изданный) в определенной форме правотворческим органом в пределах его компетенции и направленный на установление, изменение или отмену правовых норм.

    Группа 2

    Форма, шаблон образец документа — это документ, содержащий пробелы (также называемые полями или заполнителями), в которых можно что-то написать или выбрать вариант.

    Группа 3

    Ответ на вопрос - разъяснения по изменениям в работе.
    """

    response = ai_agent(prompt,
                        model="openai/gpt-4o-mini",
                        temperature=0.7,
                        max_tokens=3000)
    end_time = datetime.now() 
    
    interval = end_time - start_time
    
    print(response)
    print(f"interval: {interval}")