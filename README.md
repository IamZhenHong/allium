# Fraud Analysis Agent

A blockchain fraud analysis agent built with LangGraph and the A2A Protocol. This agent analyzes wallet addresses for fraud risk and suspicious activity patterns.

## Features

- **Wallet Analysis**: Analyzes blockchain wallet addresses for fraud risk
- **Risk Scoring**: Computes risk scores based on transaction patterns, contract interactions, and account age
- **Detailed Reports**: Generates comprehensive fraud analysis reports
- **Real-time Processing**: Streams analysis results as they're generated
- **A2A Protocol**: Implements the Agent-to-Agent (A2A) protocol for standardized agent communication

## Prerequisites

- Python 3.12+
- OpenAI API key or Google API key
- `uv` package manager

## Setup

1. **Clone and navigate to the project:**
   ```bash
   cd langgraph
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Configure environment variables:**
   Create a `.env` file in the `langgraph` directory with:
   ```
   # For OpenAI
   TOOL_LLM_NAME=gpt-4o-mini
   TOOL_LLM_URL=https://api.openai.com/v1
   API_KEY=your-openai-api-key
   
   # For Google (alternative)
   GOOGLE_API_KEY=your-google-api-key
   ```

### Start the Server

```bash
uv run app
```

The server will start on `http://localhost:10000`

### Test the Agent

```bash
uv run python app/test_client.py
```

This will:
- Connect to the fraud analysis agent
- Send test wallet addresses for analysis
- Display the agent's responses

### API Endpoints

- **GET** `/.well-known/agent.json` - Agent card (public information)
- **POST** `/` - Send messages to the agent

### Example Requests

```bash
# Analyze a wallet for fraud risk
curl -X POST http://localhost:10000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "test",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Analyze the fraud risk for wallet address 0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6"}],
        "messageId": "test"
      }
    }
  }'
```

## Agent Capabilities

The fraud analysis agent can:

1. **Fetch Wallet Data**: Retrieve transaction history and activity patterns
2. **Compute Risk Scores**: Calculate risk based on multiple factors:
   - Transaction count
   - Distinct contracts interacted with
   - Token outflows
   - Unique outflow addresses
   - Account age
3. **Generate Reports**: Create detailed fraud analysis reports using GPT

## Project Structure

```
langgraph/
├── app/
│   ├── __main__.py          # Server entry point
│   ├── agent.py             # Fraud analysis agent implementation
│   ├── agent_executor.py    # Agent executor
│   └── test_client.py       # Test client
├── mock_blockchain_wallet_data.csv  # Sample wallet data
├── pyproject.toml           # Project configuration
└── README.md               # This file
```

## Troubleshooting

- **Timeout errors**: Increase the timeout in the test client or check your API key
- **Missing dependencies**: Run `uv sync` to install all required packages
- **Server not starting**: Check that all environment variables are set correctly
