import json
import re
import uuid
import sys
import os
import inspect
import threading
import logging
from types import SimpleNamespace
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Add Hermes core library path
HERMES_ROOT = "/home/kerwin/.hermes/hermes-agent"
if HERMES_ROOT not in sys.path:
    sys.path.insert(0, HERMES_ROOT)

from hermes_cli.config import load_config
from hermes_cli.profiles import get_profile_dir
from tools.registry import registry, tool_error, tool_result

_thread_local = threading.local()

_session_wiki_edits = {}
_session_wiki_lock = threading.Lock()

def get_personality_name() -> str:
    try:
        config = load_config()
        return config.get("display", {}).get("personality", "spectra").capitalize()
    except Exception:
        return "Spectra"

class StreamingTagTransformer:
    def __init__(self, callback):
        self.callback = callback
        self.buffer = ""
        self.current_tag = None
        
        name = get_personality_name()
        self.templates = {
            "shorttermmemorize": f"*{name} commits details to her memory: ",
            "shorttermforget": f"*{name} forgets: ",
            "longtermmemorize": f"*{name} commits to long-term memory: ",
            "longtermforget": f"*{name} deletes a long-term memory: ",
            "longtermrecall": f"*{name} recalls from long-term memory: ",
            "research": f"*{name} goes to research ",
            "do": f"*{name} goes to do ",
            "searchnotes": f"*{name} searches her wiki notes for: ",
            "readnote": f"*{name} reads the wiki page: ",
        }
        self.open_tags = list(self.templates.keys())
        self.max_tag_len = max(len(t) for t in self.open_tags) + 5

    def feed(self, text: str):
        self.buffer += text
        self._process()

    def flush(self):
        if self.buffer:
            self.callback(self.buffer)
            self.buffer = ""

    def _process(self):
        while True:
            if self.current_tag is None:
                idx = self.buffer.find('<')
                if idx == -1:
                    self.callback(self.buffer)
                    self.buffer = ""
                    break
                else:
                    if idx > 0:
                        self.callback(self.buffer[:idx])
                        self.buffer = self.buffer[idx:]
                        idx = 0
                    
                    matched_tag = None
                    is_prefix = False
                    
                    for tag in self.open_tags:
                        full_open = f"<{tag}>"
                        if self.buffer.lower().startswith(full_open):
                            matched_tag = tag
                            break
                        elif full_open.startswith(self.buffer.lower()):
                            is_prefix = True
                            
                    if matched_tag:
                        open_template = self.templates[matched_tag]
                        self.callback(open_template)
                        self.current_tag = matched_tag
                        self.buffer = self.buffer[len(matched_tag) + 2:]
                        continue
                    elif is_prefix:
                        still_prefix = False
                        for tag in self.open_tags:
                            full_open = f"<{tag}>"
                            if len(self.buffer) < len(full_open) and full_open.startswith(self.buffer.lower()):
                                still_prefix = True
                                break
                        if still_prefix:
                            break
                        else:
                            self.callback(self.buffer[0])
                            self.buffer = self.buffer[1:]
                            continue
                    else:
                        self.callback(self.buffer[0])
                        self.buffer = self.buffer[1:]
                        continue
            else:
                close_tag = f"</{self.current_tag}>"
                idx = self.buffer.lower().find("</")
                if idx == -1:
                    if self.buffer.endswith("<"):
                        if len(self.buffer) > 1:
                            self.callback(self.buffer[:-1])
                            self.buffer = "<"
                        break
                    else:
                        self.callback(self.buffer)
                        self.buffer = ""
                        break
                else:
                    if idx > 0:
                        self.callback(self.buffer[:idx])
                        self.buffer = self.buffer[idx:]
                        idx = 0
                    
                    if self.buffer.lower().startswith(close_tag.lower()):
                        self.callback("*")
                        self.current_tag = None
                        self.buffer = self.buffer[len(close_tag):]
                        continue
                    elif close_tag.lower().startswith(self.buffer.lower()):
                        break
                    else:
                        self.callback(self.buffer[:2])
                        self.buffer = self.buffer[2:]
                        continue


# Helper to find the active AIAgent instance in the call stack
def get_active_agent():
    for frame_info in inspect.stack():
        frame = frame_info.frame
        self_obj = frame.f_locals.get("self")
        if self_obj and self_obj.__class__.__name__ == "AIAgent":
            return self_obj
        agent_obj = frame.f_locals.get("agent")
        if agent_obj and agent_obj.__class__.__name__ == "AIAgent":
            return agent_obj
    return None

def run_subagent_with_profile(profile_name: str, goal: str, parent_session_id: str) -> str:
    import yaml
    try:
        profile_dir = get_profile_dir(profile_name)
    except Exception:
        profile_dir = Path("~/.hermes/profiles/").expanduser() / profile_name
        
    config_path = profile_dir / "config.yaml"
    soul_path = profile_dir / "SOUL.md"
    
    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
            
    system_prompt = ""
    if soul_path.exists():
        system_prompt = soul_path.read_text()
        
    from run_agent import AIAgent
    subagent = AIAgent(
        model=config.get("model", {}).get("default", "glm-4.7-flash-heretic"),
        base_url=config.get("model", {}).get("base_url", "https://api.navy/v1"),
        provider=config.get("model", {}).get("provider", "ollama-cloud"),
        enabled_toolsets=config.get("toolsets", []),
        ephemeral_system_prompt=system_prompt if system_prompt else None,
        session_id=f"{parent_session_id}-{profile_name}-{uuid.uuid4().hex[:6]}",
        skip_memory=True,
    )
    
    logger.info(f"[Intent Orchestration] Spawning subagent with profile '{profile_name}'...")
    result = subagent.chat(goal)
    return result

# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------

def handle_shortTermMemorize(args: dict, **kwargs) -> str:
    content = args.get("content", "")
    agent = get_active_agent()
    if not agent:
        return tool_error("No active AIAgent context found on stack.")
    
    from tools.memory_tool import memory_tool as _memory_tool
    res = _memory_tool(
        action="add",
        target="memory",
        content=content,
        store=agent._memory_store
    )
    return json.dumps(res) if isinstance(res, dict) else str(res)

def handle_shortTermForget(args: dict, **kwargs) -> str:
    query = args.get("query", "")
    agent = get_active_agent()
    if not agent:
        return tool_error("No active AIAgent context found on stack.")
        
    from tools.memory_tool import memory_tool as _memory_tool
    res = _memory_tool(
        action="remove",
        target="memory",
        old_text=query,
        store=agent._memory_store
    )
    return json.dumps(res) if isinstance(res, dict) else str(res)

def handle_longTermMemorize(args: dict, **kwargs) -> str:
    content = args.get("content", "")
    try:
        config = load_config()
    except Exception:
        config = {}
    mcp_prefix = config.get("intent_orchestration", {}).get("memos_prefix", "memos")
    tool_name = f"{mcp_prefix}_add"
    return registry.dispatch(tool_name, {"content": content})

def handle_longTermForget(args: dict, **kwargs) -> str:
    memo_id = args.get("memo_id", "")
    try:
        config = load_config()
    except Exception:
        config = {}
    mcp_prefix = config.get("intent_orchestration", {}).get("memos_prefix", "memos")
    tool_name = f"{mcp_prefix}_delete"
    return registry.dispatch(tool_name, {"id": memo_id})

def handle_longTermRecall(args: dict, **kwargs) -> str:
    query = args.get("query", "")
    try:
        config = load_config()
    except Exception:
        config = {}
    mcp_prefix = config.get("intent_orchestration", {}).get("memos_prefix", "memos")
    tool_name = f"{mcp_prefix}_search"
    return registry.dispatch(tool_name, {"query": query})

def handle_research(args: dict, **kwargs) -> str:
    goal = args.get("goal", "")
    agent = get_active_agent()
    parent_session_id = agent.session_id if agent else f"sess_{uuid.uuid4().hex[:6]}"
    
    try:
        config = load_config()
    except Exception:
        config = {}
    profile = config.get("intent_orchestration", {}).get("research_profile", "spectra-research-agent")
    
    child_prefix = f"{parent_session_id}-{profile}-"
    with _session_wiki_lock:
        _session_wiki_edits[child_prefix] = []
        
    try:
        result = run_subagent_with_profile(profile, goal, parent_session_id)
        
        # Check if any wiki edits were tracked
        edits = []
        with _session_wiki_lock:
            if child_prefix in _session_wiki_edits:
                edits = _session_wiki_edits.pop(child_prefix)
                
        if edits:
            ref_lines = ["\n\n### Wiki Notes References:"]
            seen = set()
            for title, action in edits:
                if (title, action) not in seen:
                    seen.add((title, action))
                    ref_lines.append(f"- Page Title: **{title}** ({action})")
            result += "\n".join(ref_lines)
            
        return tool_result({"success": True, "result": result})
    except Exception as e:
        with _session_wiki_lock:
            _session_wiki_edits.pop(child_prefix, None)
        return tool_error(f"Research agent delegation failed: {e}")

def handle_do(args: dict, **kwargs) -> str:
    goal = args.get("goal", "")
    agent = get_active_agent()
    parent_session_id = agent.session_id if agent else f"sess_{uuid.uuid4().hex[:6]}"
    
    try:
        config = load_config()
    except Exception:
        config = {}
    profile = config.get("intent_orchestration", {}).get("doer_profile", "spectra-action-agent")
    
    child_prefix = f"{parent_session_id}-{profile}-"
    with _session_wiki_lock:
        _session_wiki_edits[child_prefix] = []
        
    try:
        result = run_subagent_with_profile(profile, goal, parent_session_id)
        
        # Check if any wiki edits were tracked
        edits = []
        with _session_wiki_lock:
            if child_prefix in _session_wiki_edits:
                edits = _session_wiki_edits.pop(child_prefix)
                
        if edits:
            ref_lines = ["\n\n### Wiki Notes References:"]
            seen = set()
            for title, action in edits:
                if (title, action) not in seen:
                    seen.add((title, action))
                    ref_lines.append(f"- Page Title: **{title}** ({action})")
            result += "\n".join(ref_lines)
            
        return tool_result({"success": True, "result": result})
    except Exception as e:
        with _session_wiki_lock:
            _session_wiki_edits.pop(child_prefix, None)
        return tool_error(f"Execution agent delegation failed: {e}")

def handle_searchNotes(args: dict, **kwargs) -> str:
    query = args.get("query", "")
    tool_name = "mcp_mediawiki_search_page"
    return registry.dispatch(tool_name, {"query": query})

def handle_readNote(args: dict, **kwargs) -> str:
    title = args.get("title", "")
    tool_name = "mcp_mediawiki_get_page"
    return registry.dispatch(tool_name, {"title": title})

# ---------------------------------------------------------------------------
# Hook Callback
# ---------------------------------------------------------------------------

def handle_post_tool_call(tool_name: str, args: dict, result: str, session_id: str, **kwargs):
    if not session_id:
        return
        
    prefix = None
    with _session_wiki_lock:
        for p in _session_wiki_edits:
            if session_id.startswith(p):
                prefix = p
                break
                
    if not prefix:
        return
        
    # Check for mediawiki write operations (create, update, delete, move)
    if tool_name in ["mcp_mediawiki_create_page", "mcp_mediawiki_update_page", "mcp_mediawiki_delete_page", "mcp_mediawiki_move_page"]:
        title = args.get("title") or args.get("page") or args.get("from") or args.get("to")
        if title:
            action_map = {
                "mcp_mediawiki_create_page": "created",
                "mcp_mediawiki_update_page": "updated",
                "mcp_mediawiki_delete_page": "deleted",
                "mcp_mediawiki_move_page": f"moved (from {args.get('from')} to {args.get('to')})"
            }
            action = action_map.get(tool_name, "modified")
            with _session_wiki_lock:
                _session_wiki_edits[prefix].append((title, action))

def handle_post_api_request(assistant_message, session_id, **kwargs):
    # Save the session ID in thread-local storage for reference
    _thread_local.session_id = session_id
    
    try:
        config = load_config()
    except Exception:
        config = {}
        
    orchestration_config = config.get("intent_orchestration", {})
    if not orchestration_config.get("enabled", True):
        return
        
    content = assistant_message.content or ""
    
    mappings = {
        "shortTermMemorize": "content",
        "shortTermForget": "query",
        "longTermMemorize": "content",
        "longTermForget": "memo_id",
        "longTermRecall": "query",
        "research": "goal",
        "do": "goal",
        "searchNotes": "query",
        "readNote": "title",
    }
    
    injected_tcs = []
    
    # Scan content for each XML tag
    for tag, arg_name in mappings.items():
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = list(re.finditer(pattern, content, re.DOTALL | re.IGNORECASE))
        for match in matches:
            val = match.group(1).strip()
            
            mock_func = SimpleNamespace(
                name=tag,
                arguments=json.dumps({arg_name: val})
            )
            mock_tool_call = SimpleNamespace(
                id=f"call_{uuid.uuid4().hex[:12]}",
                type="function",
                function=mock_func
            )
            injected_tcs.append(mock_tool_call)
            
    if injected_tcs:
        logger.info(f"[Intent Orchestration] Intercepted XML tags, injecting {len(injected_tcs)} tool call(s).")
        if not getattr(assistant_message, "tool_calls", None):
            assistant_message.tool_calls = []
        assistant_message.tool_calls.extend(injected_tcs)

# ---------------------------------------------------------------------------
# Registration Entrypoint
# ---------------------------------------------------------------------------

# JSON schemas for the tools
SHORT_TERM_MEMORIZE_SCHEMA = {
    "name": "shortTermMemorize",
    "description": "Add a new observation or fact to short-term memory (MEMORY.md). Use this to remember facts, conventions, or decisions in the current workspace.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact or observation to remember."}
        },
        "required": ["content"]
    }
}

SHORT_TERM_FORGET_SCHEMA = {
    "name": "shortTermForget",
    "description": "Remove an observation or fact from short-term memory (MEMORY.md) by matching a substring.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A unique substring of the memory entry to remove."}
        },
        "required": ["query"]
    }
}

LONG_TERM_MEMORIZE_SCHEMA = {
    "name": "longTermMemorize",
    "description": "Add a new observation or fact to long-term memory (memos). Use this to persist facts across different projects and systems permanently.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact or observation to persist permanently."}
        },
        "required": ["content"]
    }
}

LONG_TERM_FORGET_SCHEMA = {
    "name": "longTermForget",
    "description": "Remove a fact from long-term memory (memos) by its memo ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memo_id": {"type": "string", "description": "The unique memo ID to delete."}
        },
        "required": ["memo_id"]
    }
}

LONG_TERM_RECALL_SCHEMA = {
    "name": "longTermRecall",
    "description": "Search or recall facts from long-term memory (memos) matching a query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query to find matching memories."}
        },
        "required": ["query"]
    }
}

RESEARCH_SCHEMA = {
    "name": "research",
    "description": "Delegate a research goal to the research agent to gather outside information (web search, docs, files) and return a detailed summary, including a list of page titles created/updated in the Spectra Wiki during the task.",
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "The specific research objective or question."}
        },
        "required": ["goal"]
    }
}

DO_SCHEMA = {
    "name": "do",
    "description": "Delegate a technical execution goal (running commands, writing/editing code, running tests) to the execution agent and return a summary of actions taken.",
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "The technical goal or execution steps to perform."}
        },
        "required": ["goal"]
    }
}

SEARCH_NOTES_SCHEMA = {
    "name": "searchNotes",
    "description": "Search the local MediaWiki (Spectra Wiki) for pages matching a search query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query to match against wiki page titles and content."}
        },
        "required": ["query"]
    }
}

READ_NOTE_SCHEMA = {
    "name": "readNote",
    "description": "Read the content of a specific wiki page from the local MediaWiki (Spectra Wiki) by title.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The exact title of the wiki page to read."}
        },
        "required": ["title"]
    }
}

def clean_orchestration_history(messages):
    if not messages:
        return
        
    orchestration_tools = {
        "shortTermMemorize", "shortTermForget", "longTermMemorize", "longTermForget", 
        "longTermRecall", "research", "do", "searchNotes", "readNote"
    }
    
    # Map from tool_call_id -> (tool_name, tool_result_content)
    id_to_result = {}
    
    # First pass: find all tool messages for orchestration tools and collect their results,
    # then remove them from messages.
    indices_to_remove = []
    for idx, msg in enumerate(messages):
        role = None
        if isinstance(msg, dict):
            role = msg.get("role")
        else:
            role = getattr(msg, "role", None)
            
        if role == "tool":
            name = msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", None)
            if name in orchestration_tools:
                tc_id = msg.get("tool_call_id") if isinstance(msg, dict) else getattr(msg, "tool_call_id", None)
                if tc_id:
                    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                    id_to_result[tc_id] = (name, content)
                    indices_to_remove.append(idx)
                    
    # Remove the tool messages from back to front to preserve indices
    for idx in reversed(indices_to_remove):
        messages.pop(idx)
        
    # Second pass: go through assistant messages, find orchestration tool_calls,
    # remove them, and insert a user message with their results right after.
    i = 0
    while i < len(messages):
        msg = messages[i]
        
        role = None
        tool_calls = None
        if isinstance(msg, dict):
            role = msg.get("role")
            tool_calls = msg.get("tool_calls")
        else:
            role = getattr(msg, "role", None)
            tool_calls = getattr(msg, "tool_calls", None)
            
        if role == "assistant" and tool_calls:
            # Separate orchestration tool calls from others
            orch_tcs = []
            other_tcs = []
            
            for tc in tool_calls:
                name = None
                tc_id = None
                if isinstance(tc, dict):
                    name = tc.get("function", {}).get("name")
                    tc_id = tc.get("id")
                else:
                    func = getattr(tc, "function", None)
                    name = getattr(func, "name", None) if func else None
                    tc_id = getattr(tc, "id", None)
                    
                if name in orchestration_tools:
                    orch_tcs.append((tc_id, name))
                else:
                    other_tcs.append(tc)
                    
            if orch_tcs:
                # Update tool_calls to only contain non-orchestration tools
                if other_tcs:
                    if isinstance(msg, dict):
                        msg["tool_calls"] = other_tcs
                    else:
                        msg.tool_calls = other_tcs
                else:
                    if isinstance(msg, dict):
                        msg.pop("tool_calls", None)
                    else:
                        if hasattr(msg, "tool_calls"):
                            delattr(msg, "tool_calls")
                
                # Build the user message content for the results
                tool_results = []
                for tc_id, name in orch_tcs:
                    if tc_id in id_to_result:
                        _, content = id_to_result[tc_id]
                        tool_results.append(f"<{name}_result>\n{content}\n</{name}_result>")
                
                if tool_results:
                    result_text = "\n".join(tool_results)
                    messages.insert(i + 1, {
                        "role": "user",
                        "content": result_text
                    })
                    # Skip the newly inserted user message
                    i += 1
        i += 1

def handle_pre_api_request(**kwargs):
    import inspect
    messages_list = None
    api_messages_list = None
    api_kwargs_dict = None
    
    for frame_info in inspect.stack():
        frame = frame_info.frame
        if "messages" in frame.f_locals:
            messages_list = frame.f_locals["messages"]
        if "api_messages" in frame.f_locals:
            api_messages_list = frame.f_locals["api_messages"]
        if "api_kwargs" in frame.f_locals:
            api_kwargs_dict = frame.f_locals["api_kwargs"]
            if messages_list is not None and api_messages_list is not None:
                break
                
    if messages_list is not None:
        clean_orchestration_history(messages_list)
    if api_messages_list is not None:
        clean_orchestration_history(api_messages_list)
    if api_kwargs_dict is not None and isinstance(api_kwargs_dict, dict):
        if "messages" in api_kwargs_dict:
            clean_orchestration_history(api_kwargs_dict["messages"])
        elif "input" in api_kwargs_dict:
            clean_orchestration_history(api_kwargs_dict["input"])

def flush_active_transformer():
    agent = get_active_agent()
    if agent and hasattr(agent, "stream_delta_callback"):
        cb = agent.stream_delta_callback
        if hasattr(cb, "_wrapped_by_intent_orchestration") and hasattr(cb, "transformer"):
            cb.transformer.flush()

def handle_post_llm_call(**kwargs):
    flush_active_transformer()
    import inspect
    messages_list = None
    for frame_info in inspect.stack():
        frame = frame_info.frame
        if "messages" in frame.f_locals:
            messages_list = frame.f_locals["messages"]
            break
    if messages_list is not None:
        clean_orchestration_history(messages_list)

def handle_pre_llm_call(**kwargs):
    agent = get_active_agent()
    if agent:
        # Wrap stream_delta_callback to replace XML blocks with emotes inline
        if hasattr(agent, "stream_delta_callback") and agent.stream_delta_callback:
            if not hasattr(agent.stream_delta_callback, "_wrapped_by_intent_orchestration"):
                original_cb = agent.stream_delta_callback
                transformer = StreamingTagTransformer(original_cb)
                
                def wrapped_delta_cb(text: str) -> None:
                    if text is None:
                        transformer.flush()
                        original_cb(None)
                    else:
                        transformer.feed(text)
                
                wrapped_delta_cb._wrapped_by_intent_orchestration = True
                wrapped_delta_cb.transformer = transformer
                agent.stream_delta_callback = wrapped_delta_cb

        # Wrap tool_progress_callback to suppress follow-up progress messages for orchestration tools
        if hasattr(agent, "tool_progress_callback") and agent.tool_progress_callback:
            if not hasattr(agent.tool_progress_callback, "_wrapped_by_intent_orchestration"):
                original_progress_cb = agent.tool_progress_callback
                
                def wrapped_progress_cb(event_type: str, tool_name: str = None, preview: str = None, args: dict = None, **kwargs):
                    orchestration_tools = {
                        "shortTermMemorize", "shortTermForget", "longTermMemorize", "longTermForget",
                        "longTermRecall", "research", "do", "searchNotes", "readNote"
                    }
                    if tool_name in orchestration_tools:
                        return
                    original_progress_cb(event_type, tool_name, preview, args, **kwargs)
                
                wrapped_progress_cb._wrapped_by_intent_orchestration = True
                agent.tool_progress_callback = wrapped_progress_cb

    import inspect
    messages_list = None
    for frame_info in inspect.stack():
        frame = frame_info.frame
        if "messages" in frame.f_locals:
            messages_list = frame.f_locals["messages"]
            break
    if messages_list is not None:
        clean_orchestration_history(messages_list)

def handle_transform_llm_output(response_text: str, **kwargs) -> str:
    flush_active_transformer()
    if not response_text:
        return response_text
        
    content = response_text
    
    name = get_personality_name()
    mappings_templates = {
        "shortTermMemorize": f"*{name} commits details to her memory: {{val}}*",
        "shortTermForget": f"*{name} forgets: {{val}}*",
        "longTermMemorize": f"*{name} commits to long-term memory: {{val}}*",
        "longTermForget": f"*{name} deletes a long-term memory: {{val}}*",
        "longTermRecall": f"*{name} recalls from long-term memory: {{val}}*",
        "research": f"*{name} goes to research {{val}}*",
        "do": f"*{name} goes to do {{val}}*",
        "searchNotes": f"*{name} searches her wiki notes for: {{val}}*",
        "readNote": f"*{name} reads the wiki page: {{val}}*",
    }
    
    for tag, template in mappings_templates.items():
        pattern = rf"<{tag}>(.*?)</{tag}>"
        while True:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if not match:
                break
            val = match.group(1).strip()
            if not val:
                if tag == "research":
                    emote = f"*{name} goes to research*"
                elif tag == "do":
                    emote = f"*{name} goes to execute a task*"
                elif tag == "shortTermMemorize":
                    emote = f"*{name} commits details to memory*"
                elif tag == "shortTermForget":
                    emote = f"*{name} forgets details*"
                elif tag == "longTermMemorize":
                    emote = f"*{name} commits to long-term memory*"
                elif tag == "longTermForget":
                    emote = f"*{name} deletes a long-term memory*"
                elif tag == "longTermRecall":
                    emote = f"*{name} recalls from long-term memory*"
                elif tag == "searchNotes":
                    emote = f"*{name} searches her wiki notes*"
                elif tag == "readNote":
                    emote = f"*{name} reads a wiki page*"
                else:
                    emote = f"*{tag}*"
            else:
                emote = template.format(val=val)
            content = content[:match.start()] + emote + content[match.end():]
            
    return content

def monkeypatch_persist_session():
    try:
        from run_agent import AIAgent
        original_persist = getattr(AIAgent, "_persist_session", None)
        if original_persist and not hasattr(AIAgent, "_persist_session_patched"):
            def patched_persist(self, messages, conversation_history=None):
                clean_orchestration_history(messages)
                return original_persist(self, messages, conversation_history)
            
            AIAgent._persist_session = patched_persist
            AIAgent._persist_session_patched = True
            logger.info("[Intent Orchestration] Successfully monkeypatched AIAgent._persist_session")
    except Exception as e:
        logger.warning(f"[Intent Orchestration] Failed to monkeypatch AIAgent._persist_session: {e}")

def register(ctx) -> None:
    # Perform monkeypatching on AIAgent._persist_session
    monkeypatch_persist_session()

    # Register all tools under the 'intent_orchestration' toolset
    ctx.register_tool(
        name="shortTermMemorize",
        toolset="intent_orchestration",
        schema=SHORT_TERM_MEMORIZE_SCHEMA,
        handler=handle_shortTermMemorize,
        emoji="🧠"
    )
    ctx.register_tool(
        name="shortTermForget",
        toolset="intent_orchestration",
        schema=SHORT_TERM_FORGET_SCHEMA,
        handler=handle_shortTermForget,
        emoji="🗑️"
    )
    ctx.register_tool(
        name="longTermMemorize",
        toolset="intent_orchestration",
        schema=LONG_TERM_MEMORIZE_SCHEMA,
        handler=handle_longTermMemorize,
        emoji="💾"
    )
    ctx.register_tool(
        name="longTermForget",
        toolset="intent_orchestration",
        schema=LONG_TERM_FORGET_SCHEMA,
        handler=handle_longTermForget,
        emoji="❌"
    )
    ctx.register_tool(
        name="longTermRecall",
        toolset="intent_orchestration",
        schema=LONG_TERM_RECALL_SCHEMA,
        handler=handle_longTermRecall,
        emoji="🔍"
    )
    ctx.register_tool(
        name="research",
        toolset="intent_orchestration",
        schema=RESEARCH_SCHEMA,
        handler=handle_research,
        emoji="📚"
    )
    ctx.register_tool(
        name="do",
        toolset="intent_orchestration",
        schema=DO_SCHEMA,
        handler=handle_do,
        emoji="🛠️"
    )
    ctx.register_tool(
        name="searchNotes",
        toolset="intent_orchestration",
        schema=SEARCH_NOTES_SCHEMA,
        handler=handle_searchNotes,
        emoji="📖"
    )
    ctx.register_tool(
        name="readNote",
        toolset="intent_orchestration",
        schema=READ_NOTE_SCHEMA,
        handler=handle_readNote,
        emoji="📄"
    )
    
    # Register the hook callbacks
    ctx.register_hook("post_api_request", handle_post_api_request)
    ctx.register_hook("post_tool_call", handle_post_tool_call)
    ctx.register_hook("pre_api_request", handle_pre_api_request)
    ctx.register_hook("post_llm_call", handle_post_llm_call)
    ctx.register_hook("pre_llm_call", handle_pre_llm_call)
    ctx.register_hook("transform_llm_output", handle_transform_llm_output)
