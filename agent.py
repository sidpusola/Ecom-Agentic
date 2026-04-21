"""
agent.py
========
E-Commerce FAQ Bot — Production Agent Module
Domain  : E-Commerce (ShopEase India)
User    : Online shoppers
Tool    : Web Search (DuckDuckGo — no API key required)
LLM     : Groq  llama-3.1-8b-instant
Vector  : ChromaDB (in-memory)
Embedder: all-MiniLM-L6-v2

Usage:
    from agent import ask
    result = ask("What is your return policy?", thread_id="session-001")
    print(result["answer"])
"""

import os
import re
from typing import TypedDict, List

from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LLM
# ─────────────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Embedder
# ─────────────────────────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  State
# ─────────────────────────────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question:     str
    messages:     List[dict]   # conversation history
    route:        str          # retrieve | tool | memory_only
    retrieved:    str          # formatted KB context
    sources:      List[str]    # topic names retrieved
    tool_result:  str          # output of web search tool
    answer:       str          # final LLM answer
    faithfulness: float        # eval score 0.0-1.0
    eval_retries: int          # retry counter
    user_name:    str          # extracted from conversation


MAX_EVAL_RETRIES = 2

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Knowledge Base — 12 documents, one topic each, 100-500 words each
# ─────────────────────────────────────────────────────────────────────────────
KB_DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Return Policy",
        "text": (
            "ShopEase Return Policy\n\n"
            "ShopEase offers a 7-day return window for most products from the date of delivery. "
            "Items must be unused, in their original packaging, with all tags and accessories intact.\n\n"
            "Categories eligible for return:\n"
            "- Electronics: 7 days (original sealed condition or defective only)\n"
            "- Fashion & Apparel: 7 days (unworn, unwashed, original tags attached)\n"
            "- Home & Kitchen: 7 days (unused, original packaging)\n"
            "- Beauty & Personal Care: 7 days (unopened only)\n\n"
            "Categories NOT eligible for return:\n"
            "- Perishable items (food, flowers)\n"
            "- Personalised or custom-made products\n"
            "- Digital downloads and software licences\n"
            "- Undergarments and swimwear (hygiene reasons)\n"
            "- Products with a broken seal (unless defective)\n\n"
            "How to initiate a return:\n"
            "1. Log into your ShopEase account.\n"
            "2. Go to My Orders and select the order.\n"
            "3. Click 'Return Item' and choose a reason.\n"
            "4. Schedule a free pickup or drop off at the nearest ShopEase partner centre.\n"
            "5. Refunds are processed within 5-7 business days after the item is received and inspected.\n\n"
            "For return queries contact: support@shopease.in | Toll-free: 1800-123-4567 (9 AM-9 PM IST)."
        ),
    },
    {
        "id": "doc_002",
        "topic": "Shipping and Delivery",
        "text": (
            "ShopEase Shipping and Delivery Policy\n\n"
            "ShopEase delivers across India to over 27,000 pin codes. "
            "All orders are dispatched within 1-2 business days of confirmation.\n\n"
            "Estimated delivery timelines:\n"
            "- Metro cities (Delhi, Mumbai, Bengaluru, Hyderabad, Chennai, Kolkata): 2-3 business days\n"
            "- Tier-2 and Tier-3 cities: 3-5 business days\n"
            "- Remote and rural areas: 5-8 business days\n\n"
            "Shipping charges:\n"
            "- Orders above Rs.499: FREE shipping\n"
            "- Orders below Rs.499: Rs.49 flat shipping charge\n"
            "- Express delivery (same-day or next-day): Rs.99-Rs.149 extra; select cities only\n\n"
            "Courier partners: Delhivery, Bluedart, and DTDC. "
            "The courier name and tracking number are sent via SMS and email once dispatched.\n\n"
            "Large and heavy items (furniture, large appliances) are delivered by ShopEase's own "
            "logistics team with a scheduled 4-hour delivery window. Assembly is available at extra cost.\n\n"
            "If your order has not arrived within the estimated window, use Order Tracking "
            "or call 1800-123-4567."
        ),
    },
    {
        "id": "doc_003",
        "topic": "Payment Methods",
        "text": (
            "ShopEase Accepted Payment Methods\n\n"
            "1. Credit & Debit Cards: Visa, Mastercard, RuPay, American Express. "
            "3D Secure authentication is enabled on all card payments.\n\n"
            "2. UPI: Pay using Google Pay, PhonePe, Paytm, or BHIM UPI. "
            "Enter your UPI ID at checkout or scan the QR code.\n\n"
            "3. Net Banking: All major Indian banks — SBI, HDFC, ICICI, Axis, Kotak, and 50+ others.\n\n"
            "4. Digital Wallets: Paytm Wallet, Amazon Pay, Mobikwik, Freecharge.\n\n"
            "5. EMI: Available on credit cards from HDFC, ICICI, SBI, Axis, and Kotak for orders "
            "above Rs.3,000. Tenures: 3, 6, 9, and 12 months. "
            "0% EMI available on select products during sale events.\n\n"
            "6. Buy Now Pay Later (BNPL): Via ZestMoney and LazyPay. "
            "Pay in 15 days or split into 3 monthly instalments.\n\n"
            "7. Cash on Delivery (COD): Available for orders up to Rs.10,000. "
            "COD charges: Rs.30 per order. Not available in certain remote pin codes.\n\n"
            "8. ShopEase Wallet: Store credit from refunds or cashback used as full or partial payment.\n\n"
            "All transactions are secured by 256-bit SSL encryption."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Order Tracking",
        "text": (
            "How to Track Your ShopEase Order\n\n"
            "Once dispatched, ShopEase sends a tracking link and AWB number via SMS and email.\n\n"
            "Methods to track:\n"
            "1. ShopEase Website/App: Log in > My Orders > Click order > View real-time tracking map.\n"
            "2. Courier Partner Website: Use the AWB number at Delhivery, Bluedart, or DTDC's site.\n"
            "3. SMS: Reply TRACK <Order ID> to 9900012345.\n\n"
            "Order status definitions:\n"
            "- Order Placed: Received, being prepared.\n"
            "- Dispatched: Left the warehouse.\n"
            "- In Transit: On its way.\n"
            "- Out for Delivery: Arrives today.\n"
            "- Delivered: Successfully delivered.\n"
            "- Delivery Attempted: Reattempt will be scheduled.\n\n"
            "If the status shows 'Delivered' but you did not receive the order, "
            "raise a dispute within 48 hours via My Orders > Raise Issue, "
            "or call 1800-123-4567."
        ),
    },
    {
        "id": "doc_005",
        "topic": "Order Cancellation",
        "text": (
            "ShopEase Order Cancellation Policy\n\n"
            "You can cancel an order before it is dispatched. "
            "Once dispatched, cancellation is not possible; wait for delivery then initiate a return.\n\n"
            "How to cancel:\n"
            "1. Go to My Orders on the ShopEase website or app.\n"
            "2. Select the order and click 'Cancel Order'.\n"
            "3. Select a cancellation reason and confirm.\n\n"
            "Cancellation timelines:\n"
            "- Orders can typically be cancelled within 12 hours of placement.\n"
            "- Orders in 'Packing' status may still be cancellable depending on warehouse progress.\n"
            "- Orders in 'Dispatched' or later status cannot be cancelled.\n\n"
            "Refund after cancellation:\n"
            "- Prepaid orders: Full refund to original payment method within 3-5 business days.\n"
            "- COD orders: No charge made; no refund required.\n"
            "- ShopEase Wallet payment: Credit returned to wallet within 24 hours.\n\n"
            "Seller-cancelled orders: If a seller cancels due to stock issues, "
            "a full refund is automatically issued and you are notified via SMS and email.\n\n"
            "Contact: support@shopease.in | 1800-123-4567."
        ),
    },
    {
        "id": "doc_006",
        "topic": "Exchange Policy",
        "text": (
            "ShopEase Exchange Policy\n\n"
            "ShopEase allows exchanges for size, colour, or defective items within 7 days of delivery, "
            "subject to stock availability.\n\n"
            "Eligible exchange reasons:\n"
            "- Wrong size delivered (Fashion & Apparel)\n"
            "- Wrong colour delivered\n"
            "- Defective or damaged product received\n"
            "- Product does not match the description\n\n"
            "How to request an exchange:\n"
            "1. Go to My Orders and select the item.\n"
            "2. Click 'Exchange Item' and choose your reason and replacement variant.\n"
            "3. Schedule a pickup.\n"
            "4. Once the original item is picked up and verified, the replacement is dispatched in 2-3 business days.\n\n"
            "Exchange limitations:\n"
            "- Only one exchange is allowed per order item.\n"
            "- Replacement must be of equal or lesser value; if higher, the difference must be paid.\n"
            "- Electronics: Exchanges accepted only for defective units, not change of mind.\n"
            "- Beauty & Personal Care: Exchanges only if the product is sealed and defective.\n\n"
            "If the required variant is out of stock, ShopEase will offer a full refund instead.\n\n"
            "Contact: support@shopease.in | 1800-123-4567."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Warranty and Repairs",
        "text": (
            "ShopEase Warranty and Repair Policy\n\n"
            "All products sold on ShopEase carry the manufacturer's warranty. "
            "ShopEase does not provide an additional warranty unless stated on the product page.\n\n"
            "Standard warranty periods:\n"
            "- Smartphones and Tablets: 1 year manufacturer warranty\n"
            "- Laptops: 1 year (extendable to 3 years with ShopEase Care plan)\n"
            "- Large Appliances (AC, Refrigerator, Washing Machine): 1-5 years depending on brand\n"
            "- Small Appliances (Mixer, Toaster): 1-2 years\n"
            "- Fashion and Footwear: No standard warranty; exchange for manufacturing defects within 7 days\n"
            "- Furniture: 1 year against manufacturing defects\n\n"
            "How to claim warranty:\n"
            "1. Contact the brand's authorised service centre directly "
            "(list on ShopEase product page).\n"
            "2. Or raise a warranty request via My Orders > Warranty Claim.\n"
            "3. ShopEase facilitates pickup and drop-off to the service centre for eligible products.\n\n"
            "ShopEase Care (Extended Warranty):\n"
            "- Covers accidental damage, liquid damage, and post-warranty repairs.\n"
            "- Prices: Rs.299-Rs.1,999 depending on product value.\n"
            "- Purchase at order time or within 30 days of delivery.\n\n"
            "Contact: warranty@shopease.in | 1800-123-4567."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Product Categories",
        "text": (
            "ShopEase Product Categories\n\n"
            "ShopEase offers products across 12 major categories with over 5 million SKUs "
            "from 50,000+ sellers and brands.\n\n"
            "1. Electronics: Smartphones, laptops, tablets, cameras, headphones, smart TVs, "
            "gaming consoles, accessories.\n"
            "2. Fashion: Men's, women's, and kids' clothing; ethnic wear, western wear, "
            "activewear, jewellery, bags.\n"
            "3. Footwear: Casual, formal, sports shoes, sandals for men, women, and kids.\n"
            "4. Home & Kitchen: Appliances, cookware, home decor, furniture, bedding, storage.\n"
            "5. Beauty & Personal Care: Skincare, haircare, makeup, fragrances, grooming.\n"
            "6. Books & Stationery: Academic, fiction, non-fiction, exam prep, office supplies.\n"
            "7. Sports & Fitness: Gym equipment, yoga gear, outdoor sports, cycling.\n"
            "8. Toys & Baby: Baby clothing, feeding accessories, educational toys, games.\n"
            "9. Grocery & Gourmet: Packaged foods, snacks, beverages, health foods, organic products.\n"
            "10. Automotive: Car and bike accessories, tools, lubricants.\n"
            "11. Health & Wellness: Vitamins, supplements, BP monitors, glucometers.\n"
            "12. Pets: Pet food, grooming, accessories, toys for dogs, cats, and small animals.\n\n"
            "New sellers: seller.shopease.in | Brand partnerships: brands@shopease.in."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Coupons and Discounts",
        "text": (
            "ShopEase Coupons, Promo Codes, and Discounts\n\n"
            "1. Promo / Coupon Codes:\n"
            "   - Apply in the 'Apply Coupon' field at checkout.\n"
            "   - Only ONE coupon code per order.\n"
            "   - Coupons cannot be combined with each other but can be combined with bank offers.\n"
            "   - Expired, used, or invalid codes show an error message.\n"
            "   - Coupons are non-transferable and linked to your account.\n\n"
            "2. Bank & Card Offers:\n"
            "   - Instant discounts of 5-15% with select bank credit/debit cards.\n"
            "   - Applicable on top of coupon codes.\n"
            "   - Visible at checkout under 'Bank Offers'.\n\n"
            "3. Cashback:\n"
            "   - Credited to ShopEase Wallet within 72 hours of order delivery.\n"
            "   - Minimum cashback redemption: Rs.50.\n\n"
            "4. ShopEase Rewards (Loyalty Programme):\n"
            "   - Earn 1 point per Rs.100 spent.\n"
            "   - 100 points = Rs.10 discount on next purchase.\n"
            "   - Points expire 12 months from earning date.\n"
            "   - Check points: My Account > Rewards.\n\n"
            "5. Referral Offer:\n"
            "   - Refer a friend with your unique code.\n"
            "   - Earn Rs.100 ShopEase Wallet credit when friend places first order above Rs.500.\n\n"
            "6. Flash Sales:\n"
            "   - Time-limited deals visible on the homepage and under 'Today's Deals'.\n\n"
            "Coupon queries: offers@shopease.in."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Customer Support",
        "text": (
            "ShopEase Customer Support\n\n"
            "ShopEase provides multi-channel support, 7 days a week.\n\n"
            "1. Phone (Toll-Free): 1800-123-4567\n"
            "   - Hours: 9 AM-9 PM IST, Monday to Sunday\n"
            "   - For: Order issues, returns, payment problems, complaints\n"
            "   - Average wait time: 3-5 minutes\n\n"
            "2. Email: support@shopease.in\n"
            "   - Response: Within 24 hours on business days\n"
            "   - Include your order ID for faster resolution\n\n"
            "3. Live Chat:\n"
            "   - Available on ShopEase website and app\n"
            "   - Hours: 9 AM-9 PM IST\n"
            "   - Fastest channel for quick queries\n\n"
            "4. Help Centre / Self-Service: help.shopease.in\n"
            "   - Raise a ticket, track ticket status, manage returns/exchanges\n\n"
            "5. Social Media:\n"
            "   - Twitter/X: @ShopEaseSupport (9 AM-6 PM IST)\n"
            "   - Facebook: facebook.com/ShopEaseIndia\n\n"
            "Escalation:\n"
            "- If not resolved in 48 hours, ask for Senior Support Executive.\n"
            "- For unresolved complaints: grievance@shopease.in (include ticket number).\n\n"
            "Nodal Officer (Consumer Protection Act):\n"
            "- Ms. Priya Sharma | nodal@shopease.in | 1800-123-9999."
        ),
    },
    {
        "id": "doc_011",
        "topic": "Account Registration and Login",
        "text": (
            "ShopEase Account Management\n\n"
            "Creating a ShopEase account is free and gives access to order tracking, "
            "wish lists, saved addresses, and exclusive member deals.\n\n"
            "How to create an account:\n"
            "1. Visit shopease.in or open the ShopEase app.\n"
            "2. Click 'Sign Up' (top right).\n"
            "3. Enter your mobile number — an OTP is sent for verification.\n"
            "4. Fill in your name and email address (recommended).\n"
            "5. Set a password. Account is created instantly.\n\n"
            "Login methods:\n"
            "- Mobile number + OTP (recommended)\n"
            "- Mobile number + Password\n"
            "- Google Sign-In\n"
            "- Apple Sign-In (iOS app only)\n\n"
            "Forgot password:\n"
            "- Click 'Forgot Password' on the login page.\n"
            "- Enter registered mobile number or email.\n"
            "- An OTP or reset link is sent.\n"
            "- Set a new password.\n\n"
            "Account security tips:\n"
            "- Enable Two-Factor Authentication (2FA) in Account Settings.\n"
            "- Never share your OTP — ShopEase will never ask for it.\n\n"
            "Managing your account:\n"
            "- Update address, phone, email: My Account > Profile\n"
            "- Saved payments: My Account > Payment Methods\n"
            "- Download invoices: My Orders\n"
            "- Delete account: dataprivacy@shopease.in (processed in 30 days).\n\n"
            "Login issues: Call 1800-123-4567 or use Live Chat."
        ),
    },
    {
        "id": "doc_012",
        "topic": "International Shipping and Coverage",
        "text": (
            "ShopEase Geographic Coverage and International Shipping\n\n"
            "ShopEase currently operates exclusively within India. "
            "International shipping is NOT available at this time.\n\n"
            "India coverage:\n"
            "- Delivers to 27,000+ pin codes across all 28 states and 8 Union Territories.\n"
            "- Use the pin code checker on the product page to confirm deliverability.\n\n"
            "Remote area delivery:\n"
            "- Some large appliances and furniture may not be deliverable to very remote or hilly areas.\n"
            "- Cash on Delivery (COD) may not be available at select remote pin codes.\n\n"
            "Future plans:\n"
            "- ShopEase plans to launch international delivery to UAE, UK, USA, and Singapore by late 2025.\n"
            "- Register interest at shopease.in/international.\n\n"
            "For NRI / overseas customers:\n"
            "- Shopping requires an Indian delivery address.\n"
            "- You may use an Indian payment method (Indian credit card, NRI bank UPI).\n"
            "- Gift orders can be placed with a recipient's Indian address.\n\n"
            "Coverage queries: support@shopease.in | 1800-123-4567."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 5.  ChromaDB — build vector store
# ─────────────────────────────────────────────────────────────────────────────
_chroma_client = chromadb.Client()
collection = _chroma_client.get_or_create_collection(name="ecommerce_faq")

# Clear existing docs to avoid duplicate-ID errors on re-import
_existing = collection.get()
if _existing["ids"]:
    collection.delete(ids=_existing["ids"])

for _doc in KB_DOCUMENTS:
    _emb = embedder.encode(_doc["text"]).tolist()
    collection.add(
        documents=[_doc["text"]],
        embeddings=[_emb],
        ids=[_doc["id"]],
        metadatas=[{"topic": _doc["topic"]}],
    )

print(f"✅ ChromaDB loaded — {collection.count()} documents indexed.")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Tool — DuckDuckGo Web Search
# ─────────────────────────────────────────────────────────────────────────────
def web_search_tool(query: str) -> str:
    """Search the web. Always returns a string — never raises."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No web results found for this query."
        parts = []
        for r in results:
            parts.append(
                f"Title: {r.get('title', '')}\n"
                f"Snippet: {r.get('body', '')}\n"
                f"URL: {r.get('href', '')}"
            )
        return "\n\n".join(parts)
    except Exception as exc:
        return f"Web search unavailable: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Node Functions
# ─────────────────────────────────────────────────────────────────────────────

def memory_node(state: CapstoneState) -> CapstoneState:
    """Append user question; apply 6-message sliding window; extract name."""
    messages = list(state.get("messages", []))
    messages.append({"role": "user", "content": state["question"]})
    messages = messages[-6:]                      # sliding window

    user_name = state.get("user_name", "")
    match = re.search(r"my name is ([a-zA-Z]+)", state["question"], re.I)
    if match:
        user_name = match.group(1).capitalize()

    return {**state, "messages": messages, "user_name": user_name}


def router_node(state: CapstoneState) -> CapstoneState:
    """Route to retrieve | tool | memory_only using LLM one-word decision."""
    prompt = (
        "You are a router for ShopEase, an Indian e-commerce FAQ assistant.\n\n"
        "Decide the best route for the user question. Reply with ONE word only.\n\n"
        "Routes:\n"
        "- retrieve    : Question is about ShopEase store policies, returns, shipping, payments, "
        "tracking, cancellations, exchanges, warranty, products, coupons, discounts, account, "
        "or customer support.\n"
        "- tool        : Question needs live/real-time data not in the knowledge base — "
        "e.g. current market prices of products, live stock on third-party sites, news, "
        "latest offers from other stores.\n"
        "- memory_only : Greeting, small talk, thank-you, or question references only "
        "prior conversation (e.g. 'hello', 'thanks', 'what did you say earlier?').\n\n"
        f"User question: {state['question']}\n\n"
        "Reply with ONE word: retrieve, tool, or memory_only"
    )
    raw = llm.invoke(prompt).content.strip().lower().split()[0]
    route = raw if raw in ("retrieve", "tool", "memory_only") else "retrieve"
    return {**state, "route": route}


def retrieval_node(state: CapstoneState) -> CapstoneState:
    """Embed question, query ChromaDB top-3, build formatted context."""
    q_emb = embedder.encode(state["question"]).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=3,
                           include=["documents", "metadatas"])
    docs  = res["documents"][0]
    metas = res["metadatas"][0]

    parts, sources = [], []
    for doc, meta in zip(docs, metas):
        topic = meta.get("topic", "Unknown")
        parts.append(f"[{topic}]\n{doc}")
        sources.append(topic)

    return {**state,
            "retrieved": "\n\n---\n\n".join(parts),
            "sources":   sources,
            "tool_result": ""}


def skip_retrieval_node(state: CapstoneState) -> CapstoneState:
    """No retrieval needed for memory-only queries."""
    return {**state, "retrieved": "", "sources": [], "tool_result": ""}


def tool_node(state: CapstoneState) -> CapstoneState:
    """Run web search; return results as tool_result. Never raises."""
    result = web_search_tool(state["question"])
    return {**state, "tool_result": result, "retrieved": "", "sources": ["Web Search"]}


def answer_node(state: CapstoneState) -> CapstoneState:
    """Generate grounded answer using retrieved context or tool result."""
    name_line  = (f"The customer's name is {state.get('user_name')}. Address them by name."
                  if state.get("user_name") else "")
    retry_note = ("\n⚠️ Your previous answer failed faithfulness check. "
                  "Be more precise and stick strictly to the provided context."
                  if state.get("eval_retries", 0) > 0 else "")

    # Build short history string (last 4 exchanges, excluding current question)
    history_lines = []
    for msg in state.get("messages", [])[:-1]:
        role = "Customer" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {msg['content']}")
    history_text = "\n".join(history_lines[-4:]) if history_lines else "None"

    # Context section
    if state.get("retrieved"):
        ctx = f"RETRIEVED KNOWLEDGE BASE CONTEXT:\n{state['retrieved']}"
    elif state.get("tool_result"):
        ctx = f"LIVE WEB SEARCH RESULTS:\n{state['tool_result']}"
    else:
        ctx = "No external context available."

    system = (
        "You are ShopEase Assistant — a helpful, friendly customer support chatbot "
        "for ShopEase, an Indian e-commerce platform.\n"
        f"{name_line}\n\n"
        "STRICT RULES:\n"
        "1. Answer ONLY using the provided context below. Do NOT use outside knowledge.\n"
        "2. If the context does not contain the answer, say exactly: "
        "'I don't have that information. Please contact ShopEase support at "
        "1800-123-4567 or support@shopease.in.'\n"
        "3. Never fabricate prices, policies, timelines, or names.\n"
        "4. Keep answers clear and concise. Use bullet points where appropriate.\n"
        "5. Never give medical, legal, or financial advice.\n"
        f"{retry_note}\n\n"
        f"{ctx}\n\n"
        f"CONVERSATION HISTORY:\n{history_text}"
    )

    response = llm.invoke([
        {"role": "system",  "content": system},
        {"role": "user",    "content": state["question"]},
    ])
    return {**state, "answer": response.content.strip()}


def eval_node(state: CapstoneState) -> CapstoneState:
    """Score faithfulness 0.0-1.0. Trigger retry if < 0.7 (max 2 retries)."""
    retrieved    = state.get("retrieved", "")
    answer       = state.get("answer", "")
    eval_retries = state.get("eval_retries", 0)

    if not retrieved:                             # skip for tool / memory-only
        print(f"  [eval] Skipped (route={state.get('route')}). Score: 1.0")
        return {**state, "faithfulness": 1.0}

    eval_prompt = (
        "You are a faithfulness evaluator.\n\n"
        f"Context:\n{retrieved}\n\n"
        f"Answer:\n{answer}\n\n"
        "Rate how well the answer is grounded in the context:\n"
        "1.0  = every claim directly supported\n"
        "0.7+ = mostly faithful with minor elaboration\n"
        "0.4-0.6 = some claims not in context\n"
        "0.0-0.3 = significant fabrication\n\n"
        "Reply with ONLY a decimal number between 0.0 and 1.0. No other text."
    )
    try:
        raw   = llm.invoke(eval_prompt).content.strip()
        score = float(re.search(r"\d+\.?\d*", raw).group())
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 0.75

    verdict = "RETRY" if (score < 0.7 and eval_retries < MAX_EVAL_RETRIES) else "PASS"
    print(f"  [eval] Faithfulness: {score:.2f} | Retries: {eval_retries} | {verdict}")
    return {**state, "faithfulness": score, "eval_retries": eval_retries + 1}


def save_node(state: CapstoneState) -> CapstoneState:
    """Append assistant answer to messages history."""
    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": state["answer"]})
    return {**state, "messages": messages}


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Routing Functions
# ─────────────────────────────────────────────────────────────────────────────

def route_decision(state: CapstoneState) -> str:
    r = state.get("route", "retrieve")
    return "tool" if r == "tool" else ("skip" if r == "memory_only" else "retrieve")


def eval_decision(state: CapstoneState) -> str:
    if state.get("faithfulness", 1.0) < 0.7 and state.get("eval_retries", 0) <= MAX_EVAL_RETRIES:
        return "answer"
    return "save"


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Graph Assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(CapstoneState)

    graph.add_node("memory",   memory_node)
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("eval",     eval_node)
    graph.add_node("save",     save_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")
    graph.add_conditional_edges("router", route_decision, {
        "retrieve": "retrieve",
        "tool":     "tool",
        "skip":     "skip",
    })
    graph.add_edge("retrieve", "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("answer",   "eval")
    graph.add_conditional_edges("eval", eval_decision, {
        "answer": "answer",   # retry
        "save":   "save",     # pass
    })
    graph.add_edge("save", END)

    compiled = graph.compile(checkpointer=MemorySaver())
    print("✅ Graph compiled successfully.")
    return compiled


app = build_graph()

# ─────────────────────────────────────────────────────────────────────────────
# 10.  ask() Helper + Initial State Template
# ─────────────────────────────────────────────────────────────────────────────
_INITIAL_STATE_TEMPLATE: CapstoneState = {
    "question":     "",
    "messages":     [],
    "route":        "",
    "retrieved":    "",
    "sources":      [],
    "tool_result":  "",
    "answer":       "",
    "faithfulness": 0.0,
    "eval_retries": 0,
    "user_name":    "",
}


def ask(question: str, thread_id: str = "default") -> CapstoneState:
    """Invoke the agent and return the full result state."""
    config = {"configurable": {"thread_id": thread_id}}
    state  = {**_INITIAL_STATE_TEMPLATE, "question": question}
    result = app.invoke(state, config=config)
    print(
        f"\n  Q  : {question}\n"
        f"  A  : {result['answer'][:120]}{'...' if len(result['answer']) > 120 else ''}\n"
        f"  Route: {result.get('route')} | "
        f"Sources: {result.get('sources')} | "
        f"Faith: {result.get('faithfulness', 0.0):.2f}"
    )
    return result
