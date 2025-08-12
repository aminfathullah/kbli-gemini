<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Gemini AI REST API Project Instructions

This is a FastAPI-based REST API that serves as an AI endpoint using the Google Gemini AI model through the Gemini-API library.

## Project Architecture

- **FastAPI**: Main web framework for REST API
- **Gemini-API**: Library for interacting with Google Gemini AI
- **Queue Management**: Advanced request queuing for handling concurrent requests
- **Persistent Chat**: Single chat session with metadata persistence across requests

## Key Components

1. **app/main.py**: FastAPI application with all endpoints
2. **app/gemini_service.py**: Service layer for Gemini API interactions
3. **app/queue_manager.py**: Advanced queue management for request handling
4. **app/models.py**: Pydantic models for request/response validation
5. **app/config.py**: Configuration management with environment variables

## Development Guidelines

- Use async/await for all I/O operations
- Implement proper error handling with structured error responses
- Follow RESTful API conventions
- Use type hints for all function parameters and return values
- Log important events and errors for debugging
- Maintain conversation persistence using metadata storage
- Handle file uploads properly with temporary file management

## API Features

- Single persistent chat session for all requests
- Queue management for handling concurrent requests
- File upload support (images, documents)
- Multiple Gemini model selection
- Conversation reset functionality
- Health checks and status monitoring
- Request timeout handling

## Error Handling

- Use HTTPException for API errors
- Implement global exception handler
- Return structured error responses with ErrorResponse model
- Log errors appropriately for debugging

## Security Considerations

- Store sensitive credentials in environment variables
- Validate all input data using Pydantic models
- Implement proper CORS configuration
- Handle file uploads securely with temporary storage
