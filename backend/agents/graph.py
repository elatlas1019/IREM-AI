from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agents.state import AgentState
from agents.orchestrator import orchestrator_node
from agents.specialized import goal_node, plan_node, feedback_node, motivation_node, health_node, teach_node

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("GOAL", goal_node)
workflow.add_node("PLAN", plan_node)
workflow.add_node("FEEDBACK", feedback_node)
workflow.add_node("MOTIVATION", motivation_node)
workflow.add_node("HEALTH", health_node)
workflow.add_node("TEACH", teach_node)

# Set entry point
workflow.set_entry_point("orchestrator")

# Add conditional edges from orchestrator
workflow.add_conditional_edges(
    "orchestrator",
    lambda x: x["next_agent"],
    {
        "GOAL": "GOAL",
        "PLAN": "PLAN",
        "FEEDBACK": "FEEDBACK",
        "MOTIVATION": "MOTIVATION",
        "HEALTH": "HEALTH",
        "TEACH": "TEACH"
    }
)

# Add edges from specialists to END
workflow.add_edge("GOAL", END)
workflow.add_edge("PLAN", END)
workflow.add_edge("FEEDBACK", END)
workflow.add_edge("MOTIVATION", END)
workflow.add_edge("HEALTH", END)
workflow.add_edge("TEACH", END)

# Initialize memory checkpointer
memory = MemorySaver()

# Compile the graph
coaching_agent_app = workflow.compile(checkpointer=memory)
