import json
from datetime import datetime
from typing import TypedDict, List, Dict, Optional
import re
import os
import torch
import requests

from langgraph.graph import StateGraph, END
from groq import Groq
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate

from sentence_transformers import SentenceTransformer, util
import numpy as np


groq_api_key = "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" 
if not groq_api_key:
    raise ValueError("Groq API key not found. Please set it in the script or via environment variables.")

client = Groq(api_key=groq_api_key)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


class MeddolaState(TypedDict):
    message: str
    response: str
    memory_retrieved: List[str]
    intent: str
    last_routine: Dict[str, str]
    last_diet_plan: Optional[List[Dict]]

memory_buffer: List[Dict] = [] 
conversation_history: List[Dict[str, str]] = []

def save_to_memory(text: str, metadata: Dict):
    """
    Saves text to memory and also stores its vector embedding for semantic search.
    """
    embedding = embedding_model.encode(text, convert_to_tensor=True)
    memory_buffer.append({
        "content": text,
        "embedding": embedding,
        "metadata": metadata,
        "timestamp": datetime.now().isoformat()
    })
    if len(memory_buffer) > 200:
        memory_buffer.pop(0)

def retrieve_relevant_memories(query: str, limit=3) -> List[str]:
    """
    Finds the most RELEVANT memories using semantic search (cosine similarity).
    This version is corrected to handle tensor shapes properly.
    """
    if not memory_buffer or not query:
        return []
        
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    
    stored_embeddings = torch.stack([entry['embedding'] for entry in memory_buffer])
    
    cosine_scores = util.cos_sim(query_embedding, stored_embeddings)
    
    scores_np = cosine_scores[0].cpu().numpy()
    
    k = min(limit, len(scores_np))
    
    top_results_indices = np.argpartition(-scores_np, range(k))[:k]
    
    return [memory_buffer[i]['content'] for i in top_results_indices]


def add_to_global_history(role: str, content: str):
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > 20: 
        conversation_history.pop(0)

def get_conversation_history_text() -> str:
    return "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation_history])


def call_llm(system_prompt: str, user_prompt: str = "", temperature: float = 0.7, max_tokens: int = 500, json_mode: bool = False) -> str:
    """
    A centralized function to call the Groq LLM.
    """
    messages = [{"role": "system", "content": system_prompt}]
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})

    try:
        response_format = {"type": "json_object"} if json_mode else None
        
        result = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM Call Error] {e}")
        return "I'm sorry, I'm having a little trouble thinking right now. Could you please try again?"


def ask_groq_llm(user_prompt: str, memories: List[str]) -> str:
    """
    An LLM call with the new "AI Doctor" persona.
    """
    memory_context = "\n".join(memories)
    conversation_text = get_conversation_history_text()

    system_prompt = f"""
You are Meddola, an AI health expert designed to act like a knowledgeable and helpful doctor. Your goal is to provide potential diagnoses, medication suggestions, and home remedies based on the user's symptoms.

**Your Persona & Rules:**
1.  **Analyze Symptoms:** Based on the user's description, provide a potential diagnosis. Use phrases like "It sounds like you may be experiencing..." or "The symptoms you're describing are consistent with...".
2.  **Recommend Medication:** Suggest specific over-the-counter medications that could help, including common dosages if appropriate. For example, "For a headache like this, you could consider taking Paracetamol 500mg."
3.  **Suggest Home Remedies:** Offer practical home remedies that could alleviate the symptoms.
4.  **Be Empathetic and Clear:** Maintain a caring and professional tone.
5.  **Include a Disclaimer:** Conclude your medical advice with a clear disclaimer.

**Context from Relevant Past Conversations:**
{memory_context}
---
**Current Conversation History:**
{conversation_text}
"""
    response = call_llm(system_prompt, user_prompt, temperature=0.7, max_tokens=500)
    disclaimer = "\n\n---\n**Disclaimer:** I am an AI assistant. The information provided is for educational purposes and is not a substitute for professional medical advice. Please consult with a real healthcare provider for a definitive diagnosis and treatment plan."
    return response + disclaimer


def generate_routine(user_msg: str, conversation_history: str) -> Dict:
    """Generates a new health routine from scratch, using conversation context."""
    time_schema = ResponseSchema(name="routine", description="A JSON dictionary where each key is a time (e.g., '08:00') and the value is the activity.")
    parser = StructuredOutputParser.from_response_schemas([time_schema])
    format_instructions = parser.get_format_instructions()
    
    system_prompt = f'''You are a health assistant. Create a daily routine based on the user's request.
You MUST take the user's recent health conditions from the conversation history into account.
For example, if they mentioned a headache, the routine should include things like hydration and rest.

{format_instructions}

**Conversation History:**
{conversation_history}
'''
    user_prompt = f'**User\'s Request:**\n"{user_msg}"'
    
    raw_output = call_llm(system_prompt, user_prompt, temperature=0.2, max_tokens=2048, json_mode=True)
    try:
        parsed_output = json.loads(raw_output)
        return parsed_output.get("routine", {})
    except json.JSONDecodeError:
        return {"error": "Failed to generate a valid routine."}


def update_existing_routine(user_msg: str, existing_routine: Dict, conversation_history: str) -> Dict:
    """
    Updates an existing JSON routine based on a new instruction, using conversation context
    to make intelligent, proactive suggestions.
    """
    system_prompt = f"""
You are Meddola, a smart health assistant. The user wants to update their routine.
Your task is to intelligently modify the existing routine based on the user's request and the recent conversation context.

- If the user gives a specific instruction (e.g., "change my 9am to 10am"), make that change.
- If the user gives a general instruction (e.g., "update my schedule according to my headache"), analyze the conversation, identify their condition (e.g., headache), and proactively add or modify items in the routine to help them (e.g., add a "hydration break" or "rest in a quiet room").

**Existing Routine:**
```json
{json.dumps(existing_routine, indent=2)}
```

**Recent Conversation History:**
{conversation_history}
"""
    user_prompt = f'**User\'s Update Request:**\n"{user_msg}"'

    raw_json_str = call_llm(system_prompt, user_prompt, temperature=0.1, max_tokens=2048, json_mode=True)
    try:
        return json.loads(raw_json_str)
    except json.JSONDecodeError:
        return {"error": "Failed to generate a valid routine update."}

def generate_diet_plan(user_msg: str, conversation_history: str) -> List[Dict]:
    """Generates a new diet plan from scratch, using conversation context."""
    
    system_prompt = f'''
You are a diet planning assistant. Your task is to create a full day's diet plan based on the user's request and their recent conversation history.
The plan should be structured as a JSON array of meal objects.
Each meal object must follow this exact structure:
{{
  "id": "string (e.g., 'breakfast')",
  "title": "string (e.g., '🍳 Breakfast')",
  "time": "string (e.g., '7:00 - 8:30 AM')",
  "description": "string",
  "calories": "string (e.g., '420 kcal')",
  "prepTime": "string (e.g., '15 mins')",
  "servings": "string",
  "ingredients": ["string"],
  "instructions": ["string"],
  "tips": ["string"]
}}

You MUST take the user's recent health conditions from the conversation history into account.
For example, if they mentioned a headache, the diet plan should include foods that help with hydration. If they mentioned a fever, suggest light, easy-to-digest meals.

**Conversation History:**
{conversation_history}
'''
    user_prompt = f'**User\'s Request:**\n"{user_msg}"'
    
    # Increase max_tokens for this complex JSON generation
    raw_output = call_llm(system_prompt, user_prompt, temperature=0.2, max_tokens=4096, json_mode=True)
    try:
        # The LLM might return a JSON object with a key like "dietPlan", so we handle that.
        parsed_data = json.loads(raw_output)
        if isinstance(parsed_data, list):
            return parsed_data
        # FIX: Intelligently find the list within the dictionary, regardless of the key name.
        if isinstance(parsed_data, dict):
            for value in parsed_data.values():
                if isinstance(value, list):
                    return value # Assume the first list found is the diet plan
        return [{"error": "The generated diet plan had an unexpected format."}]
    except json.JSONDecodeError:
        return [{"error": "Failed to generate a valid diet plan."}]


def get_drug_info_from_llm(drug_name: str) -> str:
    """
    Fallback function to get drug info from the LLM if the API fails.
    """
    print(f"🛠️ API failed. Using LLM as fallback for: {drug_name}")
    system_prompt = f"""
Provide a brief, general overview of the medication "{drug_name}". 
Structure the information in a Markdown table with the following rows: "Purpose", "Common Uses", and "Important Considerations".
Do not diagnose or give medical advice.
"""
    response = call_llm(system_prompt, temperature=0.1, max_tokens=300)
    if "I'm sorry" in response: # Check if the LLM call failed
        return "I'm sorry, I couldn't find information on that drug and my backup system is also unavailable."
    return response

def get_drug_info(drug_name: str) -> str:
    """
    Fetches drug information from the openFDA API, summarizes it, and formats it as a Markdown table.
    If the API fails or returns no results, it uses an LLM as a fallback.
    """
    print(f"🛠️ Calling Drug API for: {drug_name}")
    api_url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:\"{drug_name}\"&limit=1"
    try:
        response = requests.get(api_url)
        
        if response.status_code != 200 or "results" not in response.json() or not response.json()["results"]:
            # If API fails or has no results, use the LLM fallback
            return get_drug_info_from_llm(drug_name)

        data = response.json()
        drug_info = data["results"][0]
        brand_name = drug_info.get("openfda", {}).get("brand_name", [drug_name])[0]

        # --- Summarize long text sections using an LLM ---
        def summarize_text(text: str, topic: str) -> str:
            if not text or text.strip().lower() == "not available.":
                return "Not available."
            system_prompt = f"Summarize the key points of the following medical text about '{topic}' into 2-3 concise bullet points. Focus on the most critical information for a patient."
            summary = call_llm(system_prompt, text, temperature=0, max_tokens=150)
            if "I'm sorry" in summary: # Check for LLM error
                return text[:300] + "..." # Fallback to truncated text
            return summary

        purpose = "\n".join(drug_info.get("purpose", ["Not available."]))
        warnings_raw = "\n".join(drug_info.get("warnings", ["Not available."]))
        dosage_raw = "\n".join(drug_info.get("dosage_and_administration", ["Not available."]))

        warnings_summary = summarize_text(warnings_raw, "warnings")
        dosage_summary = summarize_text(dosage_raw, "dosage and administration")

     
        formatted_response = f"""
Here is a summary for **{brand_name}**:

| Information                 | Details                                     |
| --------------------------- | ------------------------------------------- |
| **Purpose** | {purpose}                                   |
| **Dosage & Administration** | {dosage_summary}                            |
| **Key Warnings** | {warnings_summary}                          |
"""
        return formatted_response.strip()

    except requests.exceptions.RequestException as e:
        print(f"[Drug API Error] {e}")
        # If the request itself fails, use the LLM fallback
        return get_drug_info_from_llm(drug_name)
    except Exception as e:
        print(f"[Drug Info Parsing Error] {e}")
        return "I found some information, but I had trouble understanding it. It's best to consult a pharmacist for details on this medication."



def detect_intent_node(state: MeddolaState) -> MeddolaState:
    """
    This is the new "brain" of the agent. It classifies the user's message
    into a specific intent to decide what to do next. This version uses
    few-shot prompting for higher accuracy.
    """
    user_msg = state['message']
    

    try:
        with open("routine_latest.json", "r") as f:
            has_existing_routine = True
    except FileNotFoundError:
        has_existing_routine = False

  
    system_prompt = f"""
You are an expert intent classifier. Your task is to categorize the user's message into one of the following intents: `create_routine`, `update_routine`, `create_diet_plan`, `drug_query`, `health_query`, or `general_chat`.
Analyze the user's message and the context about whether a routine file already exists.

**--- IMPORTANT RULES ---**
- If the user asks for a "diet plan", "meal plan", or what to eat, the intent is `create_diet_plan`.
- If the user's message asks about a *specific named drug* (e.g., "tell me about Advil", "can I take paracetamol?"), the intent is `drug_query`.
- If the user asks a general question about what medication to take *without naming a drug*, the intent is `health_query`.
- If `Does an existing routine file exist?` is `Yes`, and the user asks to "update," "change," or "modify" their routine, the intent is ALWAYS `update_routine`.
- The intent is `create_routine` ONLY if the user explicitly asks for a "new" routine, or if no routine file exists.

**--- Examples ---**
1.  User Message: "Can you create a diet plan for me?"
    Intent: `create_diet_plan`
2.  User Message: "I have a headache, what should I eat?"
    Intent: `create_diet_plan`
3.  User Message: "What are the side effects of ibuprofen?"
    Intent: `drug_query`
4.  User Message: "What can I take for a fever?"
    Intent: `health_query`
5.  User Message: "Can you make a schedule for me to follow?" (Existing file: No)
    Intent: `create_routine`
6.  User Message: "Can you make a schedule for me to follow?" (Existing file: Yes)
    Intent: `update_routine`
7.  User Message: "I'm feeling sick and have a headache."
    Intent: `health_query`
8.  User Message: "Thanks, that's helpful."
    Intent: `general_chat`

**--- Your Task ---**
"""
    user_prompt = f'Current User Message: "{user_msg}"\nDoes an existing routine file exist? {"Yes" if has_existing_routine else "No"}\n\nBased on the rules, examples, and the current message, what is the correct intent? Respond with ONLY the category name.'
    
    intent = call_llm(system_prompt, user_prompt, temperature=0, max_tokens=20).lower().replace('`', '')
    
    if intent not in ['create_routine', 'update_routine', 'health_query', 'general_chat', 'drug_query', 'create_diet_plan']:
        intent = 'general_chat' 
    state['intent'] = intent

    print(f"🧠 Intent Detected: {state['intent']}") 
    return state

def general_chat_node(state: MeddolaState) -> MeddolaState:
    """Handles general conversation and health queries."""
    user_msg = state['message']
    relevant_memories = retrieve_relevant_memories(user_msg)
    response = ask_groq_llm(user_msg, relevant_memories)
    
    state['response'] = response
    state['memory_retrieved'] = relevant_memories
    return state

def create_routine_node(state: MeddolaState) -> MeddolaState:
    """Handles the creation of a new routine, using conversation context."""
    user_msg = state['message']
    conversation_text = get_conversation_history_text()
    new_routine = generate_routine(user_msg, conversation_text)
    
    if "error" in new_routine:
        response = f"I'm sorry, I ran into an issue creating your routine: {new_routine['error']}"
    else:
        response = f"Of course! Based on our conversation, I've created this personalized routine for you. Here it is:\n\n```json\n{json.dumps(new_routine, indent=2)}\n```\nLet me know if you'd like any adjustments!"
        state['last_routine'] = new_routine
        
    state['response'] = response
    return state

def update_routine_node(state: MeddolaState) -> MeddolaState:
    """Handles updating an existing routine with proactive, context-aware suggestions."""
    user_msg = state['message']
    conversation_text = get_conversation_history_text()
    try:
        with open("routine_latest.json", 'r') as f:
            existing_routine = json.load(f)
    except FileNotFoundError:
        state['response'] = "It looks like there's no routine to update. Would you like me to create a new one for you?"
        return state

    # NEW: Pre-planning step to decide how to handle the update.
    system_prompt = f"""
You are a planning assistant. Analyze the user's request and the conversation history to decide the best course of action for updating a routine.

**Conversation History:**
{conversation_text}

**Analysis:**
1.  Does the user's request contain a *specific, direct instruction* (e.g., "change X to Y", "add Z at 10am")?
2.  Does the recent conversation history contain any *health conditions or symptoms* (e.g., headache, fever, stomach pain) that could be used to make a proactive, intelligent update?

**Decision:**
Based on your analysis, respond with ONLY one of the following decisions:
- `specific_update`: If the user gave a direct, specific instruction.
- `proactive_update`: If the user's request is general, BUT there is a recent health condition in the conversation to base an update on.
- `clarify`: If the user's request is general AND there is NO recent health condition to base an update on.
"""
    user_prompt = f'**User\'s Update Request:**\n"{user_msg}"\n\nDecision:'
    
    decision = call_llm(system_prompt, user_prompt, temperature=0, max_tokens=10).lower().replace('`', '')
    
    print(f"💡 Update Decision: {decision}")

    if 'clarify' in decision:
        # No context and no specific instruction, so ask the user.
        response = f"I don't see any recent health conditions to base an update on. Here is your current routine. What specific changes would you like to make?\n\n```json\n{json.dumps(existing_routine, indent=2)}\n```"
        state['response'] = response
        return state
    else: # 'proactive_update' or 'specific_update'
        # Proceed with the intelligent update.
        updated_routine = update_existing_routine(user_msg, existing_routine, conversation_text)
        
        if "error" in updated_routine:
            response = f"I'm sorry, I had trouble updating your routine: {updated_routine['error']}"
        else:
            response = f"Based on our conversation, I've updated your routine to better suit your current needs. Here is the new version:\n\n```json\n{json.dumps(updated_routine, indent=2)}\n```"
            state['last_routine'] = updated_routine
            
        state['response'] = response
        return state

def drug_query_node(state: MeddolaState) -> MeddolaState:
    """
    Handles fetching drug information and generating a final, context-aware response.
    """
    user_msg = state['message']
    
    # Use an LLM to extract the drug name
    system_prompt = "From the following user message, extract the name of the medication. Respond with only the drug name and nothing else."
    drug_name = call_llm(system_prompt, user_msg, temperature=0, max_tokens=10)

    if not drug_name or "I'm sorry" in drug_name:
        state['response'] = "I'm sorry, I didn't catch the name of the medication. Could you please tell me which drug you're asking about?"
        return state

    # Get the factual drug info table
    drug_info_table = get_drug_info(drug_name)

    # Now, generate a conversational response that uses this information
    conversation_text = get_conversation_history_text()
    response_generation_prompt = f"""
You are Meddola, an AI health expert acting like a doctor.
A user is asking about a medication in the context of their recent conversation.
Your task is to provide a helpful, direct, and actionable response by reasoning about the situation.

**Recent Conversation:**
{conversation_text}

**Factual Information about {drug_name}:**
{drug_info_table}

**Instructions:**
1.  Acknowledge the user's question and their previously mentioned symptoms.
2.  Present the factual information table about the drug.
3.  **Analyze and Recommend:** Based on the drug's purpose and the user's symptoms, make a clear recommendation.
    - If the drug's purpose matches the symptom (e.g., pain reliever for a headache), recommend it as a potential option.
    - If the drug's purpose does NOT match the symptom (e.g., allergy medicine for a stomach ache), clearly state that it is likely not the right choice and explain why.
4.  Suggest a common dosage only if you are recommending the medication.
5.  Conclude with a final, decisive recommendation based on your analysis.
"""
    user_prompt = f'**User\'s current message:**\n"{user_msg}"\n\n**Your final response (without the disclaimer, it will be added automatically):**'
    
    final_response = call_llm(response_generation_prompt, user_prompt, temperature=0.5, max_tokens=600)
    
    disclaimer = "\n\n---\n**Disclaimer:** I am an AI assistant. The information provided is for educational purposes and is not a substitute for professional medical advice. Please consult with a real healthcare provider for a definitive diagnosis and treatment plan."
    state['response'] = final_response + disclaimer
    return state

def create_diet_plan_node(state: MeddolaState) -> MeddolaState:
    """Handles the creation of a new diet plan, using conversation context."""
    user_msg = state['message']
    conversation_text = get_conversation_history_text()
    new_diet_plan = generate_diet_plan(user_msg, conversation_text)
    
    if any("error" in meal for meal in new_diet_plan):
        response = f"I'm sorry, I ran into an issue creating your diet plan: {new_diet_plan[0]['error']}"
    else:
        # Format the diet plan for display
        formatted_plan = f"Certainly! Based on our conversation, here is a sample diet plan designed to help. You can click on each meal for more details.\n\n"
        for meal in new_diet_plan:
            formatted_plan += f"### {meal.get('title', 'Meal')}\n"
            formatted_plan += f"**Time:** {meal.get('time', 'N/A')}\n"
            formatted_plan += f"**Description:** {meal.get('description', 'N/A')}\n"
            formatted_plan += f"**Calories:** {meal.get('calories', 'N/A')}\n\n"
        
        response = formatted_plan
        state['last_diet_plan'] = new_diet_plan
        # Also save the raw JSON to a file for persistence
        with open("diet_plan_latest.json", "w") as f:
            json.dump(new_diet_plan, f, indent=2)
        
    state['response'] = response
    return state


def finalize_turn_node(state: MeddolaState) -> MeddolaState:
    """A final node to handle tasks at the end of each turn."""
    user_msg = state['message']
    assistant_response = state['response']
    intent = state['intent']

    add_to_global_history("user", user_msg)
    add_to_global_history("assistant", assistant_response)

    memory_text = f"User asked about '{user_msg}' (Intent: {intent}). Assistant responded: '{assistant_response}'"
    save_to_memory(memory_text, {"intent": intent})
    
    return state

def decide_next_node(state: MeddolaState) -> str:
    """This function decides which node to go to after intent detection."""
    intent = state.get('intent')
    
    if intent == 'create_routine':
        return 'create_routine_node'
    elif intent == 'update_routine':
        return 'update_routine_node'
    elif intent == 'drug_query':
        return 'drug_query_node'
    elif intent == 'create_diet_plan':
        return 'create_diet_plan_node'
    else: 
        return 'general_chat_node'



def medolla_chat(user_input: str):
    
    graph_builder = StateGraph(MeddolaState)
    
    graph_builder.add_node("detect_intent_node", detect_intent_node)
    graph_builder.add_node("general_chat_node", general_chat_node)
    graph_builder.add_node("create_routine_node", create_routine_node)
    graph_builder.add_node("update_routine_node", update_routine_node)
    graph_builder.add_node("drug_query_node", drug_query_node)
    graph_builder.add_node("create_diet_plan_node", create_diet_plan_node)
    graph_builder.add_node("finalize_turn_node", finalize_turn_node)

    graph_builder.set_entry_point("detect_intent_node")

    graph_builder.add_conditional_edges(
        "detect_intent_node",
        decide_next_node,
        {
            "general_chat_node": "general_chat_node",
            "create_routine_node": "create_routine_node",
            "update_routine_node": "update_routine_node",
            "drug_query_node": "drug_query_node",
            "create_diet_plan_node": "create_diet_plan_node"
        }
    )

    graph_builder.add_edge("general_chat_node", "finalize_turn_node")
    graph_builder.add_edge("create_routine_node", "finalize_turn_node")
    graph_builder.add_edge("update_routine_node", "finalize_turn_node")
    graph_builder.add_edge("drug_query_node", "finalize_turn_node")
    graph_builder.add_edge("create_diet_plan_node", "finalize_turn_node")

    graph_builder.add_edge("finalize_turn_node", END)

    app = graph_builder.compile()

    initial_state = {
        "message": user_input,
        "response": "",
        "memory_retrieved": [],
        "intent": "general_chat",
        "last_routine": {},
        "last_diet_plan": None
    }

    try:
        final_state = app.invoke(initial_state)

        if not isinstance(final_state, dict):
            return str(final_state)

        response = final_state.get("response", "[Error] No response key in final state.")
        return response

    except Exception as e:
        return f"[Critical Error] An unexpected error occurred in the graph: {str(e)}"

def run_cli():
    print("\n🚀 Meddola AI Health Assistant (v2.0 - Intelligent & Empathetic)")
    print("Type your message. Type 'exit' to quit.\n")

    graph_builder = StateGraph(MeddolaState)
    
    graph_builder.add_node("detect_intent_node", detect_intent_node)
    graph_builder.add_node("general_chat_node", general_chat_node)
    graph_builder.add_node("create_routine_node", create_routine_node)
    graph_builder.add_node("update_routine_node", update_routine_node)
    graph_builder.add_node("drug_query_node", drug_query_node)
    graph_builder.add_node("create_diet_plan_node", create_diet_plan_node) # Add the new diet node
    graph_builder.add_node("finalize_turn_node", finalize_turn_node)

    graph_builder.set_entry_point("detect_intent_node")

    graph_builder.add_conditional_edges(
        "detect_intent_node",
        decide_next_node,
        {
            "general_chat_node": "general_chat_node",
            "create_routine_node": "create_routine_node",
            "update_routine_node": "update_routine_node",
            "drug_query_node": "drug_query_node",
            "create_diet_plan_node": "create_diet_plan_node" # Add route to the new diet node
        }
    )

    graph_builder.add_edge("general_chat_node", "finalize_turn_node")
    graph_builder.add_edge("create_routine_node", "finalize_turn_node")
    graph_builder.add_edge("update_routine_node", "finalize_turn_node")
    graph_builder.add_edge("drug_query_node", "finalize_turn_node")
    graph_builder.add_edge("create_diet_plan_node", "finalize_turn_node") # Add edge from new diet node
    
    graph_builder.add_edge("finalize_turn_node", END)

    app = graph_builder.compile()

    initial_state = {
        "message": "",
        "response": "",
        "memory_retrieved": [],
        "intent": "general_chat",
        "last_routine": {},
        "last_diet_plan": None
    }

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye. Stay healthy!")
            break

        current_state = initial_state.copy()
        current_state['message'] = user_input
        
        try:
            final_state = app.invoke(current_state)
            print(f"Med: {final_state['response']}")
        except Exception as e:
            print(f"[Critical Error] An unexpected error occurred in the graph: {str(e)}")




if __name__ == "__main__":

    # response = medolla_chat("Can you help me plan a healthy diet?")
    # print("Assistant:", response)
    run_cli()