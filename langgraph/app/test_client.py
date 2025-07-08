import logging

from typing import Any
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)


PUBLIC_AGENT_CARD_PATH = '/.well-known/agent.json'
EXTENDED_AGENT_CARD_PATH = '/agent/authenticatedExtendedCard'


async def main() -> None:
    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)  # Get a logger instance

    # --8<-- [start:A2ACardResolver]

    base_url = 'http://localhost:10000'

    async with httpx.AsyncClient(timeout=120.0) as httpx_client:
        # Initialize A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
            # agent_card_path uses default, extended_agent_card_path also uses default
        )
        # --8<-- [end:A2ACardResolver]

        # Fetch Public Agent Card and Initialize Client
        final_agent_card_to_use: AgentCard | None = None

        try:
            logger.info(
                f'Attempting to fetch public agent card from: {base_url}{PUBLIC_AGENT_CARD_PATH}'
            )
            _public_card = (
                await resolver.get_agent_card()
            )  # Fetches from default public path
            logger.info('Successfully fetched public agent card:')
            logger.info(
                _public_card.model_dump_json(indent=2, exclude_none=True)
            )
            final_agent_card_to_use = _public_card
            logger.info(
                '\nUsing PUBLIC agent card for client initialization (default).'
            )

            if _public_card.supportsAuthenticatedExtendedCard:
                try:
                    logger.info(
                        '\nPublic card supports authenticated extended card. '
                        'Attempting to fetch from: '
                        f'{base_url}{EXTENDED_AGENT_CARD_PATH}'
                    )
                    auth_headers_dict = {
                        'Authorization': 'Bearer dummy-token-for-extended-card'
                    }
                    _extended_card = await resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs={'headers': auth_headers_dict},
                    )
                    logger.info(
                        'Successfully fetched authenticated extended agent card:'
                    )
                    logger.info(
                        _extended_card.model_dump_json(
                            indent=2, exclude_none=True
                        )
                    )
                    final_agent_card_to_use = (
                        _extended_card  # Update to use the extended card
                    )
                    logger.info(
                        '\nUsing AUTHENTICATED EXTENDED agent card for client '
                        'initialization.'
                    )
                except Exception as e_extended:
                    logger.warning(
                        f'Failed to fetch extended agent card: {e_extended}. '
                        'Will proceed with public card.',
                        exc_info=True,
                    )
            elif (
                _public_card
            ):  # supportsAuthenticatedExtendedCard is False or None
                logger.info(
                    '\nPublic card does not indicate support for an extended card. Using public card.'
                )

        except Exception as e:
            logger.error(
                f'Critical error fetching public agent card: {e}', exc_info=True
            )
            raise RuntimeError(
                'Failed to fetch the public agent card. Cannot continue.'
            ) from e

        # --8<-- [start:send_message]
        client = A2AClient(
            httpx_client=httpx_client, agent_card=final_agent_card_to_use
        )
        logger.info('A2AClient initialized.')

        send_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': 'Analyze the fraud risk for wallet address 0x1f438428db57621a2b3c4d5e6f7a8b9c0d1e2f3'}
                ],
                'messageId': uuid4().hex,
            },
        }
        request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**send_message_payload)
        )

        response = await client.send_message(request)
        print(response.model_dump(mode='json', exclude_none=True))
        
        # Extract and display the agent's response
        if hasattr(response, 'root') and hasattr(response.root, 'result'):
            result = response.root.result
            if hasattr(result, 'status') and hasattr(result.status, 'message'):
                agent_message = result.status.message
                if hasattr(agent_message, 'parts') and agent_message.parts:
                    print(f"\n=== AGENT RESPONSE ===")
                    for part in agent_message.parts:
                        if hasattr(part, 'text'):
                            print(f"Agent: {part.text}")
                    print(f"=====================\n")
        # --8<-- [end:send_message]

        # --8<-- [start:Multiturn]
        send_message_payload_multiturn: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {
                        'kind': 'text',
                        'text': 'What is the risk score for wallet 0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6?',
                    }
                ],
                'messageId': uuid4().hex,
            },
        }
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**send_message_payload_multiturn),
        )

        response = await client.send_message(request)
        print(response.model_dump(mode='json', exclude_none=True))
        
        # Extract and display the agent's response for multiturn
        if hasattr(response, 'root') and hasattr(response.root, 'result'):
            result = response.root.result
            if hasattr(result, 'status') and hasattr(result.status, 'message'):
                agent_message = result.status.message
                if hasattr(agent_message, 'parts') and agent_message.parts:
                    print(f"\n=== AGENT RESPONSE (MULTITURN) ===")
                    for part in agent_message.parts:
                        if hasattr(part, 'text'):
                            print(f"Agent: {part.text}")
                    print(f"==================================\n")

        # Check if response has error
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error in response: {response.error}")
            return

        # Only proceed if we have a successful response
        if hasattr(response, 'root') and hasattr(response.root, 'result'):
            task_id = response.root.result.id
            contextId = response.root.result.contextId
        else:
            logger.error("Response does not contain expected result structure")
            return

        second_send_message_payload_multiturn: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': 'Is this wallet suspicious?'}],
                'messageId': uuid4().hex,
                'taskId': task_id,
                'contextId': contextId,
            },
        }

        second_request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**second_send_message_payload_multiturn),
        )
        second_response = await client.send_message(second_request)
        print(second_response.model_dump(mode='json', exclude_none=True))
        
        # Extract and display the agent's response for second message
        if hasattr(second_response, 'root') and hasattr(second_response.root, 'result'):
            result = second_response.root.result
            if hasattr(result, 'status') and hasattr(result.status, 'message'):
                agent_message = result.status.message
                if hasattr(agent_message, 'parts') and agent_message.parts:
                    print(f"\n=== AGENT RESPONSE (FOLLOW-UP) ===")
                    for part in agent_message.parts:
                        if hasattr(part, 'text'):
                            print(f"Agent: {part.text}")
                    print(f"====================================\n")
        # --8<-- [end:Multiturn]

        # --8<-- [start:send_message_streaming]

        streaming_request = SendStreamingMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**send_message_payload)
        )

        stream_response = client.send_message_streaming(streaming_request)
        
        print(f"\n=== STREAMING RESPONSE ===")
        async for chunk in stream_response:
            print(chunk.model_dump(mode='json', exclude_none=True))
        print(f"==========================\n")
        # --8<-- [end:send_message_streaming]


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
