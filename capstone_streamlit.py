"""
capstone_streamlit.py
ShopEase FAQ Bot — Streamlit UI

"""

import uuid
import streamlit as st


# Page Config  (must be first Streamlit call)

st.set_page_config(
    page_title="ShopEase Assistant",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Cache expensive resources — loaded ONCE per server session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_agent():
    """Import agent module — triggers KB build + graph compile once."""
    from agent import app as _app, CapstoneState as _State, _INITIAL_STATE_TEMPLATE
    return _app, _State, _INITIAL_STATE_TEMPLATE


agent_app, CapstoneState, INITIAL_STATE = load_agent()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _reset_session():
    st.session_state.chat_messages  = []   # UI display messages
    st.session_state.agent_messages = []   # internal agent memory
    st.session_state.thread_id      = str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "chat_messages" not in st.session_state:
    _reset_session()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 ShopEase Assistant")
    st.markdown("*Your 24 / 7 customer support bot*")
    st.divider()

    st.markdown("### 💬 Topics I can help with")
    TOPICS = [
        ("📦", "Returns & Exchanges"),
        ("🚚", "Shipping & Delivery"),
        ("💳", "Payment Methods"),
        ("🔍", "Order Tracking"),
        ("❌", "Order Cancellation"),
        ("🛡️", "Warranty & Repairs"),
        ("🏷️", "Coupons & Discounts"),
        ("🛍️", "Product Categories"),
        ("👤", "Account & Login"),
        ("🌍", "Delivery Coverage"),
        ("📞", "Customer Support"),
        ("🔎", "Live Web Search"),
    ]
    for icon, label in TOPICS:
        st.markdown(f"&nbsp;&nbsp;{icon} {label}")

    st.divider()
    st.markdown(f"**Session:** `{st.session_state.thread_id[:8]}...`")

    if st.button("🔄 New Conversation", use_container_width=True, type="primary"):
        _reset_session()
        st.rerun()

    st.divider()
    st.markdown("📞 **Helpline:** 1800-123-4567")
    st.markdown("📧 **Email:** support@shopease.in")
    st.markdown("🕘 9 AM – 9 PM IST, Mon–Sun")


# ─────────────────────────────────────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛒 ShopEase Customer Assistant")
st.caption(
    "Ask me about orders, returns, shipping, payments, warranty, coupons, and more. "
    "For emergencies or complex issues, call **1800-123-4567**."
)

# Render conversation history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────────────────────────────────────
# Chat Input
# ─────────────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask your question here…"):

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("ShopEase Assistant is thinking…"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            invoke_state = {
                **INITIAL_STATE,
                "question": prompt,
                "messages": st.session_state.agent_messages,
            }

            try:
                result       = agent_app.invoke(invoke_state, config=config)
                answer       = result.get("answer", "")
                route        = result.get("route", "")
                sources      = result.get("sources", [])
                faithfulness = result.get("faithfulness", None)
                # Persist agent messages for next turn
                st.session_state.agent_messages = result.get("messages", [])

            except Exception as exc:
                answer       = (
                    f"⚠️ Something went wrong: {exc}\n\n"
                    "Please try again or contact us at **1800-123-4567**."
                )
                route, sources, faithfulness = "error", [], None

        st.markdown(answer)

        # Collapsible debug / metadata panel
        with st.expander("🔍 Response details", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("Route", route or "—")
            col2.metric("Sources", ", ".join(sources) if sources else "—")
            if faithfulness is not None:
                badge = "🟢" if faithfulness >= 0.7 else "🔴"
                col3.metric("Faithfulness", f"{badge} {faithfulness:.2f}")
            else:
                col3.metric("Faithfulness", "N/A")

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
