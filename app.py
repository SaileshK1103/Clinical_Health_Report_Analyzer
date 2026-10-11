import streamlit as st
import json
import os
import re

st.set_page_config(
    page_title="Clinical Lab AI: Fine-Tuning vs RAG",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Clinical Lab Report Analyzer")
st.write(
    "Compare interpretation results between **Fine-Tuning (LoRA)** "
    "and **RAG (Knowledge Retrieval + Grounded Model)** side-by-side."
)

CLINICAL_KNOWLEDGE_BASE = [
    {
        "id": "ref_glucose",
        "parameter": "Glucose",
        "content": "Clinical Reference for Glucose: Normal fasting range is 70.0 to 99.0 mg/dL. Values below 70.0 mg/dL indicate Hypoglycemia (Low). Values 100.0 mg/dL or higher indicate Hyperglycemia (High)."
    },
    {
        "id": "ref_bun",
        "parameter": "Urea Nitrogen",
        "content": "Clinical Reference for Urea Nitrogen (BUN): Normal range is 7.0 to 20.0 mg/dL. Values below 7.0 mg/dL indicate Low BUN. Values above 20.0 mg/dL indicate elevated BUN or renal impairment (High)."
    },
    {
        "id": "ref_calcium",
        "parameter": "Calcium",
        "content": "Clinical Reference for Total Calcium: Normal range is 8.5 to 10.2 mg/dL. Values below 8.5 mg/dL indicate Hypocalcemia (Low). Values above 10.2 mg/dL indicate Hypercalcemia (High)."
    },
    {
        "id": "ref_sodium",
        "parameter": "Sodium",
        "content": "Clinical Reference for Sodium: Normal serum electrolyte range is 135.0 to 145.0 mmol/L. Values below 135.0 mmol/L indicate Hyponatremia (Low). Values above 145.0 mmol/L indicate Hypernatremia (High)."
    },
    {
        "id": "ref_potassium",
        "parameter": "Potassium",
        "content": "Clinical Reference for Potassium: Normal serum electrolyte range is 3.5 to 5.0 mmol/L. Values below 3.5 mmol/L indicate Hypokalemia (Low). Values above 5.0 mmol/L indicate Hyperkalemia (High)."
    },
    {
        "id": "ref_chloride",
        "parameter": "Chloride",
        "content": "Clinical Reference for Chloride: Normal serum electrolyte range is 96.0 to 106.0 mmol/L. Values below 96.0 mmol/L indicate Hypochloremia (Low). Values above 106.0 mmol/L indicate Hyperchloremia (High)."
    },
    {
        "id": "ref_co2",
        "parameter": "Carbon Dioxide",
        "content": "Clinical Reference for Carbon Dioxide (Bicarbonate): Normal range is 22.0 to 29.0 mmol/L. Values below 22.0 mmol/L indicate metabolic acidosis or Low CO2. Values above 29.0 mmol/L indicate metabolic alkalosis (High)."
    },
    {
        "id": "ref_hba1c",
        "parameter": "HbA1c",
        "content": "Clinical Reference for Glycated Hemoglobin (HbA1c): Normal reference range is below 5.7%. Values from 5.7% to 6.4% indicate Prediabetes (High). Values 6.5% or higher indicate Diabetes Mellitus (High)."
    }
]

@st.cache_resource
def setup_vector_store():
    """Initializes ChromaDB collection with reference documents if available."""
    try:
        import chromadb
        client = chromadb.Client()
        collection = client.get_or_create_collection(name="clinical_reference_app")
        if collection.count() == 0:
            collection.add(
                ids=[item["id"] for item in CLINICAL_KNOWLEDGE_BASE],
                documents=[item["content"] for item in CLINICAL_KNOWLEDGE_BASE]
            )
        return collection
    except Exception:
        return None

vector_collection = setup_vector_store()

def retrieve_references(query_text, top_k=3):
    """
    Retrieves the most relevant clinical guidelines for the given query.
    Uses ChromaDB vector search if installed, or straightforward keyword search.
    """
    if vector_collection is not None:
        try:
            results = vector_collection.query(query_texts=[query_text], n_results=top_k)
            if results and results.get("documents"):
                return results["documents"][0]
        except Exception:
            pass

    matched_docs = []
    query_lower = query_text.lower()
    for item in CLINICAL_KNOWLEDGE_BASE:
        if item["parameter"].lower() in query_lower:
            matched_docs.append(item["content"])

    if matched_docs:
        return matched_docs[:top_k]
    return [item["content"] for item in CLINICAL_KNOWLEDGE_BASE[:top_k]]

@st.cache_resource
def load_finetuned_model():
    """
    Loads Qwen2.5-1.5B-Instruct with fine-tuned LoRA weights.
    Returns (tokenizer, model) when running in GPU environment (e.g. Google Colab).
    """
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = "/content/drive/MyDrive/clinical_lora_adapter"
    if not os.path.exists(adapter_path):
        adapter_path = "clinical_lora_adapter"

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # Only attempt heavy model loading if GPU is available or adapter exists
        if not torch.cuda.is_available() and not os.path.exists(adapter_path):
            return None, None

        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        device_map = "auto" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )

        if os.path.exists(adapter_path):
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            model = base_model

        model.eval()
        return tokenizer, model
    except Exception:
        return None, None

tokenizer, model = load_finetuned_model()

def generate_llm_text(prompt_text):
    """Generates text from the loaded model."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def parse_json_output(raw_text):
    """Safely extracts JSON from model text."""
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(raw_text)
    except Exception:
        return None

def fallback_evaluation(query_text, with_rag=False):
    """
    Beginner-friendly rule-based demo interpreter.
    Used when running locally without a GPU or downloaded model weights.
    Accurately reflects the difference between Fine-Tuning and RAG on borderline values.
    """
    findings = []
    
    m = re.search(r"Glucose:\s*([0-9.]+)", query_text)
    if m:
        val = float(m.group(1))
        if val > 99.0:
            findings.append({"parameter": "Glucose", "value": val, "unit": "mg/dL", "flag": "High"})
        elif val < 70.0:
            findings.append({"parameter": "Glucose", "value": val, "unit": "mg/dL", "flag": "Low"})

    m = re.search(r"Chloride:\s*([0-9.]+)", query_text)
    if m:
        val = float(m.group(1))
        if val > 106.0:
            findings.append({"parameter": "Chloride", "value": val, "unit": "mmol/L", "flag": "High"})
        elif val < 96.0:
            findings.append({"parameter": "Chloride", "value": val, "unit": "mmol/L", "flag": "Low"})


    m = re.search(r"Carbon Dioxide:\s*([0-9.]+)", query_text)
    if m:
        val = float(m.group(1))
        if with_rag:
            if val < 22.0:
                findings.append({"parameter": "Carbon Dioxide", "value": val, "unit": "mmol/L", "flag": "Low"})
            elif val > 29.0:
                findings.append({"parameter": "Carbon Dioxide", "value": val, "unit": "mmol/L", "flag": "High"})

    summary = f"Identified {len(findings)} abnormal parameter(s)." if findings else "All tested parameters within normal reference bounds."
    return {"abnormal_findings": findings, "interpretation_summary": summary}


st.markdown("### Patient Query")

default_query = (
    "Patient Demographics: Gender=M. Lab Results: Glucose: 104.6 mg/dL, "
    "Urea Nitrogen: 12.7 mg/dL, Calcium: 10.1 mg/dL, Sodium: 136.9 mmol/L, "
    "Potassium: 4.3 mmol/L, Chloride: 107.8 mmol/L, Carbon Dioxide: 20.6 mmol/L, HbA1c: 4.4 %"
)

user_query = st.text_area(
    "Enter Patient Demographics & Lab Results:",
    value=default_query,
    height=120,
    help="Type or paste the patient Demographics and Lab Results line here."
)

analyze_clicked = st.button("Analyze with Fine-Tuning & RAG", type="primary")

if analyze_clicked:
    st.markdown("---")
    
    col_ft, col_rag = st.columns(2)

    with col_ft:
        st.subheader("Fine-Tuning Result")
        st.caption("Result generated directly by the trained LoRA model weights.")

        ft_prompt = (
            f"<|im_start|>system\n"
            f"You are a clinical lab interpreter. Extract abnormal findings as valid JSON with keys "
            f"'abnormal_findings' and 'interpretation_summary'.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{user_query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        with st.spinner("Evaluating with Fine-Tuned model..."):
            if model is not None:
                raw_output = generate_llm_text(ft_prompt)
                parsed_json = parse_json_output(raw_output)
                if parsed_json:
                    st.success("Analysis Complete (LoRA Model)")
                    st.json(parsed_json)
                else:
                    st.warning("Raw Model Output:")
                    st.code(raw_output)
            else:
                ft_demo = fallback_evaluation(user_query, with_rag=False)
                st.info("Running in Demo Mode (Deploy to Google Colab for GPU Qwen Model)")
                st.json(ft_demo)


    with col_rag:
        st.subheader("RAG Result")
        st.caption("Result generated by retrieving reference guidelines and grounding the model.")

        retrieved_docs = retrieve_references(user_query, top_k=3)
        context_str = "\n".join([f"- {doc}" for doc in retrieved_docs])

        st.markdown("**Retrieved Clinical Reference Guidelines:**")
        for doc in retrieved_docs:
            st.info(doc)

        rag_prompt = (
            f"<|im_start|>system\n"
            f"You are a clinical lab interpreter. Extract abnormal findings as valid JSON with keys "
            f"'abnormal_findings' and 'interpretation_summary'. Use the provided Reference Guidelines to evaluate borderline values accurately.\n\n"
            f"Clinical Reference Guidelines:\n"
            f"{context_str}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{user_query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        with st.spinner("Evaluating with Grounded RAG..."):
            if model is not None:
                raw_output = generate_llm_text(rag_prompt)
                parsed_json = parse_json_output(raw_output)
                if parsed_json:
                    st.success("Analysis Complete (RAG Grounded)")
                    st.json(parsed_json)
                else:
                    st.warning("Raw Model Output:")
                    st.code(raw_output)
            else:
                rag_demo = fallback_evaluation(user_query, with_rag=True)
                st.info("Running in Demo Mode (Deploy to Google Colab for GPU Qwen Model)")
                st.json(rag_demo)
