"""
OpenClaw: Conversational agent framework for photo management.
"""
import inspect
from typing import Any

class Agent:
    """
    Conversational agent that can call tools to help with photo management.
    """
    def __init__(self, tools=None, system_prompt=None):
        self.tools = tools or []
        self.system_prompt = system_prompt
        self._tool_map = {self._get_tool_name(t): t for t in self.tools}
    
    def _get_tool_name(self, func):
        """Extract a readable tool name from a function"""
        name = func.__name__
        # Convert snake_case to readable form: search_tool -> search
        return name.replace('_tool', '').replace('_', ' ').lower()
    
    def chat(self, message: str, history=None) -> str:
        """
        Process a message and return an agent response.
        Routes to appropriate tool based on message content.
        """
        message_lower = message.lower()
        history = history or []
        
        # Intent detection based on keywords
        if any(word in message_lower for word in ['search', 'find', 'show', 'look for', 'show me']):
            return self._call_search_tool(message)
        
        elif any(word in message_lower for word in ['duplicate', 'duplicates', 'similar', 'copies']):
            return self._call_duplicates_tool(message)
        
        elif any(word in message_lower for word in ['tag', 'tags', 'label', 'categorize']):
            return self._call_tag_tool(message)
        
        else:
            # Default: try search if message looks like a query
            return self._call_search_tool(message)
    
    def _call_search_tool(self, message: str) -> str:
        """Call the search tool"""
        if not self.tools:
            return "No tools available"
        
        search_tool = self.tools[0]  # First tool is typically search
        try:
            # Extract the query from the message
            # Remove common prefixes
            query = message
            for prefix in ['find ', 'search ', 'show ', 'show me ', 'look for ', 'find all ']:
                if query.lower().startswith(prefix):
                    query = query[len(prefix):]
                    break
            
            result = search_tool(query=query, top_k=10)
            return self._format_response(result, "Search Results")
        except Exception as e:
            return f"Error in search: {str(e)}"
    
    def _call_duplicates_tool(self, message: str) -> str:
        """Call the duplicates tool"""
        if len(self.tools) < 2:
            return "Duplicates tool not available"
        
        duplicates_tool = self.tools[1]  # Second tool is typically duplicates
        try:
            result = duplicates_tool(threshold=0.97)
            return self._format_response(result, "Duplicate Detection Results")
        except Exception as e:
            return f"Error in duplicate detection: {str(e)}"
    
    def _call_tag_tool(self, message: str) -> str:
        """Call the tag tool"""
        if len(self.tools) < 3:
            return "Tag tool not available"
        
        tag_tool = self.tools[2]  # Third tool is typically tag
        try:
            # Try to extract image path from message
            import re
            path_match = re.search(r'(?:at|from|in)\s+([^\s]+)', message)
            if path_match:
                path = path_match.group(1)
                result = tag_tool(image_path=path)
                return self._format_response(result, f"Tags for {path}")
            else:
                return "Please specify the image path (e.g., 'tag the photo at /path/to/photo.jpg')"
        except Exception as e:
            return f"Error in tagging: {str(e)}"
    
    def _format_response(self, result: Any, title: str = "Result") -> str:
        """Format tool result into readable response"""
        if isinstance(result, dict):
            # If it's a dict with results, format nicely
            if 'results' in result:
                items = result['results']
                if not items:
                    return f"No {title.lower()} found."
                
                response = f"{title}:\n"
                if isinstance(items, list):
                    for i, item in enumerate(items, 1):
                        if isinstance(item, dict):
                            # Check if item has an image path
                            path = item.get('path')
                            filename = item.get('filename', 'Photo')
                            score = item.get('score', 'N/A')
                            tags = item.get('tags', [])
                            
                            # Format as IMAGE_MARKER for frontend to recognize
                            response += f"\n[IMAGE_MARKER]{path}[/IMAGE_MARKER]\n"
                            response += f"{filename}\n"
                            response += f"Score: {score}\n"
                            if tags:
                                response += f"Tags: {', '.join(tags)}\n"
                        else:
                            response += f"{i}. {str(item)}\n"
                return response
            else:
                return str(result)
        elif isinstance(result, list):
            if not result:
                return f"No {title.lower()} found."
            response = f"{title}:\n"
            for i, item in enumerate(result, 1):
                if isinstance(item, dict):
                    # Check if item has an image path
                    path = item.get('path')
                    filename = item.get('filename', 'Photo')
                    score = item.get('score', 'N/A')
                    tags = item.get('tags', [])
                    
                    # Format as IMAGE_MARKER for frontend to recognize
                    response += f"\n[IMAGE_MARKER]{path}[/IMAGE_MARKER]\n"
                    response += f"{filename}\n"
                    response += f"Score: {score}\n"
                    if tags:
                        response += f"Tags: {', '.join(tags)}\n"
                else:
                    response += f"{i}. {str(item)}\n"
            return response
        else:
            return str(result)


def tool(description=None):
    """
    Decorator to mark a function as a callable tool for the agent.
    """
    def decorator(func):
        func.is_tool = True
        func.tool_description = description
        return func
    return decorator
