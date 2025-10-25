# paperqa/agent/registry.py
class ToolRegistry:
    def __init__(self, search_tool, gather_tool, ask_tool, answer_tool):
        self.search = search_tool
        self.gather = gather_tool
        self.ask = ask_tool
        self.answer = answer_tool