import logging
import os
import sys

import click
import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

from app.agent import FraudAnalysisAgent
from app.agent_executor import CurrencyAgentExecutor


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10000)
def main(host, port):
    """Starts the Currency Agent server."""
    try:
        if os.getenv('model_source', 'google') == 'google':
            if not os.getenv('GOOGLE_API_KEY'):
                raise MissingAPIKeyError(
                    'GOOGLE_API_KEY environment variable not set.'
                )
        else:
            if not os.getenv('TOOL_LLM_URL'):
                raise MissingAPIKeyError(
                    'TOOL_LLM_URL environment variable not set.'
                )
            if not os.getenv('TOOL_LLM_NAME'):
                raise MissingAPIKeyError(
                    'TOOL_LLM_NAME environment not variable not set.'
                )

        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skill = AgentSkill(
            id='fraud_analysis',
            name='Blockchain Fraud Analysis Tool',
            description='Analyzes wallet addresses for fraud risk and suspicious activity',
            tags=['fraud analysis', 'blockchain', 'risk assessment'],
            examples=['Analyze the fraud risk for wallet address 0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6'],
        )
        agent_card = AgentCard(
            name='Fraud Analysis Agent',
            description='Analyzes blockchain wallet addresses for fraud risk and suspicious activity patterns',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=FraudAnalysisAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=FraudAnalysisAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # --8<-- [start:DefaultRequestHandler]
        httpx_client = httpx.AsyncClient()
        request_handler = DefaultRequestHandler(
            agent_executor=CurrencyAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(httpx_client),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)
        # --8<-- [end:DefaultRequestHandler]

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
