from dotenv import load_dotenv
import json
import logging
import logging.config
import os
import re
from services import bedrock_agent_runtime
import streamlit as st
import uuid
import yaml
from datetime import datetime

# Load environment variables and configure logging
load_dotenv()

# Configure logging using YAML
if os.path.exists("logging.yaml"):
    with open("logging.yaml", "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)
else:
    log_level = logging.getLevelNamesMapping()[(os.environ.get("LOG_LEVEL", "INFO"))]
    logging.basicConfig(level=log_level)

logger = logging.getLogger(__name__)

# Get config from environment variables
agent_id = os.environ.get("BEDROCK_AGENT_ID")
agent_alias_id = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "TSTALIASID")
ui_title = os.environ.get("BEDROCK_AGENT_TEST_UI_TITLE", "Agents for Amazon Bedrock")
ui_icon = os.environ.get("BEDROCK_AGENT_TEST_UI_ICON", "🤖")
ui_theme = os.environ.get("BEDROCK_AGENT_TEST_UI_THEME", "light")


def init_session_state():
    """Initialize session state variables"""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.citations = []
    st.session_state.trace = {}
    st.session_state.session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.show_debug = False


def render_message(message):
    """Render a message with proper formatting"""
    if message["role"] == "assistant" and any(citation_marker in message["content"] for citation_marker in ["<sup>[", "[1]", "[2]"]):
        st.markdown(message["content"], unsafe_allow_html=True)
    else:
        st.write(message["content"])


def process_agent_response(response):
    """Process the agent response and format output with citations"""
    output_text = response["output_text"]

    # Try to parse JSON response
    try:
        output_json = json.loads(output_text, strict=False)
        if "instruction" in output_json and "result" in output_json:
            output_text = output_json["result"]
    except json.JSONDecodeError:
        pass

    # Add formatted citations if available
    if response["citations"]:
        # Replace citation markers with superscript format
        output_text = re.sub(r"%\[(\d+)\]%", r"<sup>[\1]</sup>", output_text)
        
        # Create citation references section
        citation_refs = []
        citation_num = 1
        
        for citation in response["citations"]:
            for retrieved_ref in citation["retrievedReferences"]:
                citation_marker = f"[{citation_num}]"
                source_type = retrieved_ref['location']['type']
                
                citation_url = ""
                match source_type:
                    case 'CONFLUENCE':
                        citation_url = retrieved_ref['location']['confluenceLocation']['url']
                    case 'CUSTOM':
                        citation_url = retrieved_ref['location']['customDocumentLocation']['id']
                    case 'KENDRA':
                        citation_url = retrieved_ref['location']['kendraDocumentLocation']['uri']
                    case 'S3':
                        citation_url = retrieved_ref['location']['s3Location']['uri']
                    case 'SALESFORCE':
                        citation_url = retrieved_ref['location']['salesforceLocation']['url']
                    case 'SHAREPOINT':
                        citation_url = retrieved_ref['location']['sharePointLocation']['url']
                    case 'SQL':
                        citation_url = retrieved_ref['location']['sqlLocation']['query']
                    case 'WEB':
                        citation_url = retrieved_ref['location']['webLocation']['url']
                    case _:
                        citation_url = f"Unknown source ({source_type})"
                        logger.warning(f"Unknown location type: {source_type}")
                
                citation_refs.append(f"{citation_marker} {citation_url}")
                citation_num += 1
        
        if citation_refs:
            output_text += "\n\n---\n**Sources:**\n" + "\n".join(citation_refs)

    return output_text


def export_conversation():
    """Export the conversation to a JSON file"""
    if not st.session_state.messages:
        st.warning("No conversation to export")
        return
    
    conversation_data = {
        "session_id": st.session_state.session_id,
        "session_start_time": st.session_state.session_start_time,
        "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": st.session_state.messages,
        "citations": st.session_state.citations
    }
    
    json_str = json.dumps(conversation_data, indent=2)
    st.download_button(
        label="Download Conversation",
        data=json_str,
        file_name=f"bedrock_agent_conversation_{st.session_state.session_id}.json",
        mime="application/json",
    )


def display_trace_section():
    """Display trace information in the sidebar"""
    st.sidebar.title("Debug Information")
    
    # Toggle for showing/hiding debug info
    show_debug = st.sidebar.checkbox("Show Debug Information", value=st.session_state.show_debug)
    st.session_state.show_debug = show_debug
    
    if not show_debug:
        return
    
    # Trace section
    trace_types_map = {
        "Pre-Processing": ["preGuardrailTrace", "preProcessingTrace"],
        "Orchestration": ["orchestrationTrace"],
        "Post-Processing": ["postProcessingTrace", "postGuardrailTrace"]
    }

    trace_info_types_map = {
        "preProcessingTrace": ["modelInvocationInput", "modelInvocationOutput"],
        "orchestrationTrace": ["invocationInput", "modelInvocationInput", "modelInvocationOutput", "observation", "rationale"],
        "postProcessingTrace": ["modelInvocationInput", "modelInvocationOutput", "observation"]
    }

    st.sidebar.header("Trace")

    # Show each trace type in separate sections
    step_num = 1
    for trace_type_header in trace_types_map:
        st.sidebar.subheader(trace_type_header)

        # Organize traces by step
        has_trace = False
        for trace_type in trace_types_map[trace_type_header]:
            if trace_type in st.session_state.trace:
                has_trace = True
                trace_steps = {}

                for trace in st.session_state.trace[trace_type]:
                    # Each trace type and step may have different information for the end-to-end flow
                    if trace_type in trace_info_types_map:
                        trace_info_types = trace_info_types_map[trace_type]
                        for trace_info_type in trace_info_types:
                            if trace_info_type in trace:
                                trace_id = trace[trace_info_type]["traceId"]
                                if trace_id not in trace_steps:
                                    trace_steps[trace_id] = [trace]
                                else:
                                    trace_steps[trace_id].append(trace)
                                break
                    else:
                        trace_id = trace["traceId"]
                        trace_steps[trace_id] = [
                            {
                                trace_type: trace
                            }
                        ]

                # Show trace steps in JSON
                for trace_id in trace_steps.keys():
                    with st.sidebar.expander(f"Trace Step {str(step_num)}", expanded=False):
                        for trace in trace_steps[trace_id]:
                            trace_str = json.dumps(trace, indent=2)
                            st.code(trace_str, language="json", line_numbers=True)
                    step_num += 1
        if not has_trace:
            st.sidebar.text("None")

    # Citations section
    st.sidebar.subheader("Citations")
    if st.session_state.citations:
        citation_num = 1
        for citation in st.session_state.citations:
            for retrieved_ref_num, retrieved_ref in enumerate(citation["retrievedReferences"]):
                with st.sidebar.expander(f"Citation [{str(citation_num)}]", expanded=False):
                    citation_str = json.dumps(
                        {
                            "generatedResponsePart": citation["generatedResponsePart"],
                            "retrievedReference": citation["retrievedReferences"][retrieved_ref_num]
                        },
                        indent=2
                    )
                    st.code(citation_str, language="json", line_numbers=True)
                citation_num += 1
    else:
        st.sidebar.text("None")


def display_agent_info():
    """Display agent information in a sidebar expander"""
    with st.sidebar.expander("Agent Information", expanded=False):
        st.markdown(f"**Agent ID:** `{agent_id}`")
        st.markdown(f"**Agent Alias ID:** `{agent_alias_id}`")
        st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
        st.markdown(f"**Session Start:** {st.session_state.session_start_time}")


def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title=ui_title,
        page_icon=ui_icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://docs.aws.amazon.com/bedrock/',
            'Report a bug': 'https://github.com/aws-samples/amazon-bedrock-samples/issues',
            'About': f"Amazon Bedrock Agent Test UI - Session ID: {st.session_state.get('session_id', 'Not initialized')}"
        }
    )

    # Initialize session state if not already done
    if len(st.session_state.items()) == 0:
        init_session_state()

    # Set up sidebar
    st.sidebar.title("Amazon Bedrock Agent")
    
    # Add sidebar buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Reset Session", key="reset_session", use_container_width=True):
            init_session_state()
            st.rerun()
    with col2:
        export_conversation()
    
    # Display agent information
    display_agent_info()

    # Display chat title with Bedrock logo
    st.title(f"{ui_title}")
    st.divider()
    
    # Display messages in a container with some styling
    message_container = st.container()
    with message_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
                render_message(message)
    
    # Chat input at the bottom
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(prompt)
        
        # Process with agent and display response
        with st.chat_message("assistant", avatar="🤖"):
            with st.status("Agent is processing your request...", expanded=True) as status:
                st.write("Invoking Amazon Bedrock Agent...")
                response = bedrock_agent_runtime.invoke_agent(
                    agent_id,
                    agent_alias_id,
                    st.session_state.session_id,
                    prompt
                )
                st.write("Processing response...")
                
                # Process and format the response
                output_text = process_agent_response(response)
                
                # Update session state
                st.session_state.messages.append({"role": "assistant", "content": output_text})
                st.session_state.citations = response["citations"]
                st.session_state.trace = response["trace"]
                
                status.update(label="Completed!", state="complete", expanded=False)
            
            # Display the formatted response
            st.markdown(output_text, unsafe_allow_html=True)
    
    # Display trace information in sidebar
    display_trace_section()


if __name__ == "__main__":
    main()