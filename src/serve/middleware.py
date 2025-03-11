"""
Middleware components for the Biblical AI API.
Provides:
- Request/response logging
- Performance monitoring
- Request metadata handling
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import setup_logger

# Set up logging
logger = setup_logger(__name__)


async def add_request_metadata(request: Request, call_next: Callable) -> Response:
    """
    Middleware to add metadata to each request.
    Adds:
    - request_id: Unique identifier for tracking
    - timestamp: Time of request
    """
    # Generate a unique ID for this request
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.timestamp = time.time()
    
    # Process the request and get the response
    response = await call_next(request)
    
    # Add request ID to response headers for tracking
    response.headers["X-Request-ID"] = request_id
    
    return response


async def log_request_response(request: Request, call_next: Callable) -> Response:
    """
    Middleware to log all requests and responses for monitoring.
    Logs:
    - Request method and path
    - Response status code
    - Processing time
    """
    start_time = time.time()
    
    # Attempt to get the request body for logging
    # This has to be done carefully as reading the body consumes it
    try:
        request_body = await request.body()
        # Store body for later use by route handlers
        request.state.body = request_body
    except Exception:
        request_body = b""
    
    # Log the request
    logger.info(
        f"Request {request.state.request_id}: {request.method} {request.url.path} "
        f"- Client: {request.client.host if request.client else 'unknown'}"
    )
    
    # Try to get path parameters as well
    path_params = getattr(request, "path_params", {})
    if path_params:
        logger.debug(f"Path params: {path_params}")
    
    # Query parameters
    query_params = dict(request.query_params)
    if query_params:
        # Filter out sensitive information if needed
        logger.debug(f"Query params: {query_params}")
    
    try:
        # Process the request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log the response
        logger.info(
            f"Response {request.state.request_id}: Status {response.status_code} "
            f"- Processed in {process_time:.4f}s"
        )
        
        # Add processing time to response headers
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    except Exception as e:
        # Log any exceptions that occur during processing
        logger.error(f"Error processing request {request.state.request_id}: {str(e)}")
        raise


class BiblicalContentFilter(BaseHTTPMiddleware):
    """
    Middleware to filter and monitor content for theological accuracy and sensitivity.
    This is an example of a more complex middleware that could be implemented.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Process the request
        response = await call_next(request)
        
        # Content filtering could be implemented here
        # This would inspect response content and apply theological filters
        
        return responsepublic class Main {
    public static void main(String[] args){
    //start coding
    }
}
