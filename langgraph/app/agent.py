import os

from collections.abc import AsyncIterable
from typing import Any, Literal

import httpx

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from typing import Dict
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Set, Optional
from difflib import SequenceMatcher
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import re
import json
import pandas as pd
import os
import argparse
import sys
import time
from datetime import datetime

# from a2a.client import A2AClient
memory = MemorySaver()
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

wallet_df = pd.read_csv("mock_blockchain_wallet_data.csv")
print(f"[DEBUG] Global wallet data loaded: {len(wallet_df)} records")

@tool
def fetch_wallet_data(wallet_address: str) -> Dict:
    """
    Fetch wallet data from the database
    """
    print(f"[DEBUG] fetch_wallet_data: Searching for wallet address: {wallet_address}")
    
    # Search for the wallet address (case-insensitive)
    wallet_row = wallet_df[wallet_df['wallet_address'].str.lower() == wallet_address.lower()]
    
    if wallet_row.empty:
        return {
            "error": f"Wallet address {wallet_address} not found in database",
            "wallet_address": wallet_address,
            "found": False
        }
    
    # Convert the row to a dictionary
    wallet_data = wallet_row.iloc[0].to_dict()
    
    result = {
        "wallet_address": wallet_data['wallet_address'],
        "first_seen": wallet_data['first_seen'],
        "last_seen": wallet_data['last_seen'],
        "tx_count": int(wallet_data['tx_count']),
        "distinct_contracts": int(wallet_data['distinct_contracts']),
        "token_outflows": int(wallet_data['token_outflows']),
        "unique_outflow_addresses": int(wallet_data['unique_outflow_addresses']),
        "found": True
    }
    
    print(f"[DEBUG] fetch_wallet_data: Found wallet with {result['tx_count']} transactions")
    return result

@tool
def compute_risk_score(wallet_data: Dict) -> float:
    """
    Compute the risk score for a wallet based on various factors
    """
    print(f"[DEBUG] compute_risk_score: Calculating risk for wallet: {wallet_data.get('wallet_address', 'Unknown')}")
    
    if not wallet_data.get("found", False):
        return 0.0
    
    risk_score = 0.0
    
    # Factor 1: Transaction count (higher = more suspicious)
    tx_count = wallet_data.get("tx_count", 0)
    if tx_count > 40:
        risk_score += 0.3
    elif tx_count > 20:
        risk_score += 0.2
    elif tx_count > 10:
        risk_score += 0.1
    
    # Factor 2: Distinct contracts (higher = more suspicious)
    distinct_contracts = wallet_data.get("distinct_contracts", 0)
    if distinct_contracts > 8:
        risk_score += 0.25
    elif distinct_contracts > 5:
        risk_score += 0.15
    elif distinct_contracts > 2:
        risk_score += 0.1
    
    # Factor 3: Token outflows (higher = more suspicious)
    token_outflows = wallet_data.get("token_outflows", 0)
    if token_outflows > 8:
        risk_score += 0.25
    elif token_outflows > 5:
        risk_score += 0.15
    elif token_outflows > 2:
        risk_score += 0.1
    
    # Factor 4: Unique outflow addresses (higher = more suspicious)
    unique_outflow_addresses = wallet_data.get("unique_outflow_addresses", 0)
    if unique_outflow_addresses > 5:
        risk_score += 0.2
    elif unique_outflow_addresses > 3:
        risk_score += 0.15
    elif unique_outflow_addresses > 1:
        risk_score += 0.1
    
    # Factor 5: Account age (newer accounts are more suspicious)
    try:
        first_seen = datetime.fromisoformat(wallet_data.get("first_seen", "").replace("Z", "+00:00"))
        last_seen = datetime.fromisoformat(wallet_data.get("last_seen", "").replace("Z", "+00:00"))
        account_age_days = (last_seen - first_seen).days
        
        if account_age_days < 7:
            risk_score += 0.2
        elif account_age_days < 30:
            risk_score += 0.1
    except Exception as e:
        # If we can't parse dates, assume older account
        pass
    
    # Normalize risk score to 0-1 range
    risk_score = min(risk_score, 1.0)
    print(f"[DEBUG] compute_risk_score: Final risk score: {risk_score}")
    
    return round(risk_score, 3)

@tool
def generate_report(wallet_data: Dict, risk_score: float = None) -> str:
    """
    Generate a risk analysis report for a wallet using GPT
    """
    print(f"[DEBUG] generate_report: Generating report for wallet: {wallet_data.get('wallet_address', 'Unknown')}")
    
    # Handle case where agent might pass parameters differently
    if isinstance(wallet_data, str):
        # If wallet_data is a string, it might be the risk_score
        risk_score = wallet_data
        wallet_data = None
        print("WARNING: generate_report received string instead of wallet_data")
        return "Error: generate_report requires wallet_data dictionary as first parameter"
    
    if not wallet_data or not wallet_data.get("found", False):
        print("WALLET DATA NOT FOUND")
        return f"Error: Wallet address {wallet_data.get('wallet_address', 'Unknown') if wallet_data else 'None'} not found in database."
    
    
    # If risk_score is not provided, compute it
    if risk_score is None:
        risk_score = compute_risk_score(wallet_data)

    # Prepare the prompt for GPT
    prompt = f"""
    You are a blockchain fraud analyst. Analyze the following wallet data and generate a comprehensive risk analysis report.
    
    Wallet Address: {wallet_data.get('wallet_address', 'Unknown')}
    Risk Score: {risk_score:.3f} (0.0 = Low Risk, 1.0 = High Risk)
    
    Wallet Data:
    - First Seen: {wallet_data.get('first_seen', 'Unknown')}
    - Last Seen: {wallet_data.get('last_seen', 'Unknown')}
    - Transaction Count: {wallet_data.get('tx_count', 0)}
    - Distinct Contracts Interacted: {wallet_data.get('distinct_contracts', 0)}
    - Token Outflows: {wallet_data.get('token_outflows', 0)}
    - Unique Outflow Addresses: {wallet_data.get('unique_outflow_addresses', 0)}
    
    Please generate a detailed risk analysis report that includes:
    1. Overall risk assessment
    2. Key risk factors identified
    3. Behavioral analysis
    4. Recommendations for further investigation
    5. Confidence level in the assessment
    
    Format the report professionally and be specific about what makes this wallet suspicious or safe.
    """

    try:
        # Initialize the chat model
        model = init_chat_model(model="gpt-4o-mini")
        
        # Generate the report
        response = model.invoke(prompt)

        print("Generated report: ", response.content)
        
        print(f"[DEBUG] generate_report: Report generated successfully ({len(response.content)} chars)")
        return response.content
    except Exception as e:
        print("Error generating report: ", e)
        return f"Error generating report: {str(e)}"
    
@tool
def get_exchange_rate(
    currency_from: str = 'USD',
    currency_to: str = 'EUR',
    currency_date: str = 'latest',
):
    """Use this to get current exchange rate.

    Args:
        currency_from: The currency to convert from (e.g., "USD").
        currency_to: The currency to convert to (e.g., "EUR").
        currency_date: The date for the exchange rate or "latest". Defaults to
            "latest".

    Returns:
        A dictionary containing the exchange rate data, or an error message if
        the request fails.
    """
    try:
        response = httpx.get(
            f'https://api.frankfurter.app/{currency_date}',
            params={'from': currency_from, 'to': currency_to},
        )
        response.raise_for_status()

        data = response.json()
        if 'rates' not in data:
            return {'error': 'Invalid API response format.'}
        return data
    except httpx.HTTPError as e:
        return {'error': f'API request failed: {e}'}
    except ValueError:
        return {'error': 'Invalid JSON response from API.'}


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class FraudAnalysisAgent:
    """FraudAnalysisAgent - a specialized assistant for fraud analysis."""

    SYSTEM_INSTRUCTION = (
        'You are a fraud analyst. Your task is to analyze the wallet address: {wallet_address}\n\n'
        'Workflow:\n'
        '1. Call fetch_wallet_data with "{wallet_address}" to get wallet data\n'
        '2. Call compute_risk_score with the wallet data to get risk score\n'
        '3. Call generate_report with the wallet data and risk score to get final report\n'
        '4. Return the final report\n\n'
        'Important:\n'
        '- Use the exact wallet address: {wallet_address}\n'
        '- Call each tool only once\n'
        '- Pass the output from one tool as input to the next tool\n'
        '- For generate_report, pass both wallet_data and risk_score as separate parameters\n'
        '- The final report should be your final answer'
    )
    
    FORMAT_INSTRUCTION = (
        'Set response status to input_required if the user needs to provide more information to complete the request.'
        'Set response status to error if there is an error while processing the request.'
        'Set response status to completed if the request is complete.'
    )

    def __init__(self):
        self.model = ChatOpenAI(
            model=os.getenv('TOOL_LLM_NAME'),
            openai_api_key=os.getenv('API_KEY', 'EMPTY'),
            openai_api_base=os.getenv('TOOL_LLM_URL'),
            temperature=0,
        )
        self.tools = [fetch_wallet_data, compute_risk_score, generate_report]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Looking up the exchange rates...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the exchange rates..',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
