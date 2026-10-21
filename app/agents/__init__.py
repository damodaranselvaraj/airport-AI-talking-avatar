"""Multi-agent LangGraph kiosk workflow.

Nodes wired together by ``MultiAgentKioskWorkflow`` in ``graph.py``:

    input guardrail -> supervisor -> greeting | deny | RAG sub-graph
                                       -> answer agent -> validation agent -> finalize | fallback

The data contracts (``KioskRequest``, ``AnswerResult``, ``TurnOutcome`` ...), the router, the
retriever, the evidence assessor, the answer generator and the validator are all sourced from
``app/workflow`` and ``app/retrieval``.
"""
