import argparse
from agent.agent import PaperQAgent
from agent.registry import ToolRegistry
from tools.tool_search import SearchTool
from tools.tool_gather import GatherTool, SimpleRetriever
from tools.tool_ask import AskTool
from tools.tool_answer import AnswerTool

def build_registry():
    search = SearchTool()
    retriever = SimpleRetriever(chunks=search.get_chunks())
    gather = GatherTool(retriever=retriever)
    ask = AskTool()
    answer = AnswerTool(min_evidence=3, top_k=6, min_avg_score=6.0)  # looser for the demo
    return ToolRegistry(search, gather, ask, answer)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, help="Question")
    args = parser.parse_args()
    agent = PaperQAgent(build_registry())
    ans = agent.run(args.q)
    print(ans)

if __name__ == "__main__":
    main()
