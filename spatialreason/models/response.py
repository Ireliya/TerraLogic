"""
Response class for handling model responses with tool calls.
"""

class Response(dict):
    """
    A dict that supports resp.text, resp.content and resp.tool_calls,
    but under the hood is a plain dict (so msgpack can serialize it).
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        self[name] = value
