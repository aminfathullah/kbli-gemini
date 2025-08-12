# Gemini AI REST API

A full-featured REST API that serves as an AI endpoint using Google Gemini, built with FastAPI and the Gemini-API library. Features persistent chat sessions, intelligent queue management, and comprehensive conversation handling.

## Features

- 🚀 **FastAPI-based REST API** with automatic OpenAPI documentation
- 🧠 **Single Persistent Chat Session** with metadata persistence
- 📋 **Advanced Queue Management** for handling concurrent requests
- 🔄 **Conversation Continuity** across multiple API calls
- 📁 **File Upload Support** for images and documents
- 🎯 **Model Selection** (Gemini 2.5 Flash, Gemini 2.5 Pro, etc.)
- 🎨 **Image Generation** with Imagen4
- 🔧 **Gemini Extensions** support (Gmail, YouTube, etc.)
- 📊 **Request Tracking** and response caching
- ⚡ **Async Processing** for optimal performance

## Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env` file:
   ```env
   GEMINI_SECURE_1PSID=your_secure_1psid_cookie
   GEMINI_SECURE_1PSIDTS=your_secure_1psidts_cookie
   API_HOST=0.0.0.0
   API_PORT=8000
   MAX_QUEUE_SIZE=100
   REQUEST_TIMEOUT=300
   ```

4. Get Gemini cookies:
   - Go to https://gemini.google.com and login
   - Press F12, go to Network tab, refresh page
   - Copy `__Secure-1PSID` and `__Secure-1PSIDTS` cookie values

## Usage

### Start the API Server

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Documentation

Once running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

### Example Requests

#### Simple Chat
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

#### Chat with Model Selection
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum computing",
    "model": "gemini-2.5-pro"
  }'
```

#### Upload File and Chat
```bash
curl -X POST "http://localhost:8000/chat/file" \
  -F "message=Analyze this image" \
  -F "file=@image.jpg"
```

## API Endpoints

- `POST /chat` - Send a message to the AI
- `POST /chat/file` - Send a message with file upload
- `GET /health` - Check API health status
- `GET /models` - List available models
- `GET /queue/status` - Check queue status
- `DELETE /conversation/reset` - Reset the conversation

## Project Structure

```
gemini-ala2/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── gemini_service.py    # Gemini API service
│   ├── queue_manager.py     # Request queue management
│   └── config.py            # Configuration settings
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
├── main.py                  # Application entry point
└── README.md               # This file
```

## Configuration

The API can be configured through environment variables in the `.env` file:

- `GEMINI_SECURE_1PSID`: Your Gemini __Secure-1PSID cookie
- `GEMINI_SECURE_1PSIDTS`: Your Gemini __Secure-1PSIDTS cookie
- `API_HOST`: Host address (default: 0.0.0.0)
- `API_PORT`: Port number (default: 8000)
- `MAX_QUEUE_SIZE`: Maximum requests in queue (default: 100)
- `REQUEST_TIMEOUT`: Request timeout in seconds (default: 300)

## Queue Management

The API implements intelligent queue management to handle concurrent requests efficiently:

- **FIFO Queue**: Requests are processed in first-in-first-out order
- **Queue Limits**: Configurable maximum queue size to prevent overload
- **Timeout Handling**: Automatic cleanup of expired requests
- **Status Monitoring**: Real-time queue status via `/queue/status` endpoint

## License

This project is licensed under the AGPL-3.0 License - see the LICENSE file for details.
