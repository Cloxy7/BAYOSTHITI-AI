"""Function-calling loop. The LLM never sees the dataset -- only the results of the three tools.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python -m agent.chat "I'm working with tolfenamic acid. What should I pair it with?"

Swap the three provider lines for any other function-calling API; nothing else changes.
"""
import json
import os
import sys

from .tools import SYSTEM, TOOLS, dispatch

MODEL = os.environ.get('COUNTERION_MODEL', 'claude-sonnet-4-5')
MAX_TURNS = 6


def ask(question, verbose=True):
    import anthropic
    client = anthropic.Anthropic()
    messages = [dict(role='user', content=question)]
    for _ in range(MAX_TURNS):
        r = client.messages.create(model=MODEL, max_tokens=1500, system=SYSTEM, tools=TOOLS, messages=messages)
        messages.append(dict(role='assistant', content=r.content))
        calls = [b for b in r.content if b.type == 'tool_use']
        if not calls:
            return ''.join(b.text for b in r.content if b.type == 'text')
        results = []
        for c in calls:
            if verbose:
                print(f'  -> {c.name}({json.dumps(c.input)})', file=sys.stderr)
            results.append(dict(type='tool_result', tool_use_id=c.id,
                                content=json.dumps(dispatch(c.name, c.input), default=str)))
        messages.append(dict(role='user', content=results))
    return 'stopped: too many tool rounds'


if __name__ == '__main__':
    q = ' '.join(sys.argv[1:]) or "I'm working with tolfenamic acid. What counter-ion should I try, and has anyone made it?"
    print(ask(q))
