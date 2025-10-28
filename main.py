import argparse
from agent.agent import PaperQAgent
from agent.registry import ToolRegistry
from tools.tool_search import SearchTool
from tools.tool_gather import GatherTool
from tools.tool_ask import AskTool
from tools.tool_answer import AnswerTool
from evaluation import run_evaluation

def main():
    pipeline = PaperQAPipeline()
    run_evaluation(pipeline, dataset_size=10, debug=True, format_type="text")

if __name__ == "__main__":
    main()
