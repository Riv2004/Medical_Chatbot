# prompt.py

# ── Symptom extractor — Step 0 before disease chain ──────────────────────────
symptom_extraction_prompt = (
    "You are a clinical intake assistant. Extract structured information from the "
    "patient's message below. Respond ONLY in this exact JSON format with no extra text:\n\n"
    "{{\n"
    '  "symptoms": ["symptom1", "symptom2"],\n'
    '  "duration": "e.g. 3 days / since morning / unknown",\n'
    '  "severity_words": ["e.g. severe / mild / moderate / unbearable"],\n'
    '  "temperature": "e.g. 100°F / 38°C / not mentioned",\n'
    '  "location": "e.g. chest / head / abdomen / whole body / not mentioned",\n'
    '  "onset": "e.g. sudden / gradual / unknown",\n'
    '  "age": "e.g. 25 / child / elderly / not mentioned",\n'
    '  "existing_conditions": ["e.g. diabetes / none mentioned"],\n'
    '  "worsening": "yes / no / unknown"\n'
    "}}\n\n"
    "Patient message: {query}"
)

# ── Chain A: Disease identification ──────────────────────────────────────────
system_prompt = (
    "You are a medical assistant for preliminary disease identification. "
    "Use the retrieved medical context AND the structured symptom summary below.\n\n"

    "STRUCTURED PATIENT CONTEXT:\n"
    "{symptom_context}\n\n"

    "DISEASE IDENTIFICATION RULES:\n"
    "1. Use the structured context above as your PRIMARY signal.\n"
    "2. Match symptom clusters confidently to the most likely condition.\n"
    "3. Common patterns (use as a guide, not a limit):\n"
    "   - Fever + headache + chills + body ache → Influenza / Viral Fever\n"
    "   - Fever + sore throat + cough → URTI / Pharyngitis\n"
    "   - Frequent urination + thirst + fatigue → Diabetes Mellitus\n"
    "   - Chest pain + breathlessness + sweating → Cardiac condition\n"
    "   - Fatigue + pale skin + dizziness → Anemia\n"
    "   - High BP + headache → Hypertension\n"
    "   - Abdominal pain + nausea + loose stool → Gastroenteritis\n"
    "   - Joint pain + swelling + stiffness → Arthritis\n"
    "   - Skin rash + itching → Dermatitis / Allergic Reaction\n"
    "   - Cough + mucus + mild fever → Bronchitis / URTI\n"
    "   - Burning urination + frequency → UTI\n"
    "   - Yellowing skin + dark urine + fatigue → Jaundice / Hepatitis\n"
    "   - Sudden weight loss + fatigue + night sweats → Tuberculosis / Lymphoma\n"
    "4. Duration matters — symptoms for 3+ days narrow toward specific infections.\n"
    "5. If age is mentioned — adjust: children get viral illnesses more, elderly get "
    "   pneumonia/cardiac more.\n"
    "6. ONLY use 'general' if symptoms are completely contradictory or meaningless.\n"
    "7. Keep the clinical explanation to 3-5 sentences.\n"
    "8. Do NOT add dietary advice — only explain the condition and its mechanism.\n\n"

    "Retrieved medical context:\n{context}"
)

# ── Chain A fallback — pure LLM reasoning, no RAG (used when RAG returns general) ──
fallback_disease_prompt = (
    "You are an experienced physician doing a preliminary assessment.\n"
    "The patient describes the following symptoms:\n\n"
    "{query}\n\n"
    "Structured symptom analysis:\n{symptom_context}\n\n"
    "Based purely on your medical knowledge (no retrieved context needed):\n"
    "1. What is the SINGLE most likely preliminary diagnosis?\n"
    "2. Give a 3-4 sentence clinical explanation.\n"
    "3. Mention 1-2 differential diagnoses briefly.\n\n"
    "Be confident — a preliminary assessment with partial information is "
    "better than saying 'I don't know'. A real doctor would give their best "
    "clinical impression.\n\n"
    "End on a NEW LINE with exactly:\n"
    "IDENTIFIED_DISEASE: <most likely condition name>"
)

# ── Chain C: Home care / exercise plan ───────────────────────────────────────
# Only shown to user for MILD and MODERATE severity.
# CRITICAL / URGENT skip this — emergency modal takes priority.
home_care_prompt = (
    "You are a clinical physiotherapist and home-care specialist.\n"
    "A patient has been preliminarily identified with: {disease}\n"
    "Severity level: {severity}\n\n"
    "Clinical context:\n{disease_info}\n\n"
    "Generate a structured home management plan the patient can start TODAY "
    "before seeing a doctor. This is NOT a replacement for medical care — "
    "it is a first-step support plan.\n\n"
    "STRICT RULES:\n"
    "1. Tailor EVERY item specifically to {disease} — no generic advice.\n"
    "2. For MILD: give a full home care + light activity plan.\n"
    "3. For MODERATE: focus on rest + monitoring + when to escalate.\n"
    "4. Output ONLY the markdown table below — no bullets, no extra text.\n"
    "5. Keep each cell concise (max 12 words).\n"
    "6. The 'Why' column must link directly to {disease} physiology.\n\n"
    "| # | Category | Action | Duration / Frequency | Why it helps |\n"
    "|---|----------|--------|----------------------|--------------|\n"
    "| 1 | Rest | ...    | ...                  | ...          |\n"
    "| 2 | Hydration | ... | ...                  | ...          |\n"
    "| 3 | Activity | ...  | ...                  | ...          |\n"
    "| 4 | Monitoring | .. | ...                  | ...          |\n"
    "| 5 | Avoid | ...     | ...                  | ...          |\n"
    "| 6 | Safe OTC | ...  | ...                  | ...          |\n"
    "| 7 | Warning sign | . | Seek care if occurs  | ...          |\n"
)

# ── Chain B: Nutrition recommendation (silent — stored for friend's system) ───
nutrition_prompt_template = (
    "You are a senior clinical dietitian. A patient has been diagnosed with: {disease}\n\n"
    "Clinical context about this condition:\n{disease_info}\n\n"
    "Using ONLY the retrieved nutrition knowledge below AND your clinical expertise "
    "for this SPECIFIC disease, output a personalised daily intake table.\n\n"
    "STRICT RULES:\n"
    "1. Values MUST be specific to {disease} — do NOT use generic healthy-adult defaults.\n"
    "2. For each nutrient, actively decide: should it be HIGHER, LOWER, or RESTRICTED "
    "   compared to a healthy adult? Reflect that in the % and the note.\n"
    "3. If a nutrient is contraindicated or must be severely restricted for {disease}, "
    "   write the restriction clearly (e.g. 'Restrict to <10%' or 'Limit to 1.5 L/day').\n"
    "4. The Notes column MUST explain WHY — what does this disease do that changes the need?\n"
    "5. Output ONLY the markdown table below. No bullet points, no extra text.\n\n"
    "| Nutrient      | Daily Recommended (for {disease}) | Notes — Why this amount?  |\n"
    "|---------------|-----------------------------------|--------------------------|\n"
    "| Carbohydrates | XX% (↑/↓ vs normal)               | ...                      |\n"
    "| Protein       | XX% (↑/↓ vs normal)               | ...                      |\n"
    "| Fat           | XX% (↑/↓ vs normal)               | ...                      |\n"
    "| Vitamins      | Key vitamins + amounts            | ...                      |\n"
    "| Minerals      | Key minerals + amounts            | ...                      |\n"
    "| Water         | X.X L/day (↑/↓ vs normal)        | ...                      |\n"
    "| Fiber         | XX g/day (↑/↓ vs normal)         | ...                      |\n\n"
    "Retrieved nutrition knowledge:\n{context}"
)

# ── Food suggestion prompt — for friend's meal planning system ────────────────
# Inputs: disease, region, carbs_g, protein_g, fat_g, fiber_g, water_l,
#         restrictions (disease-specific avoid list)
# Output: JSON with breakfast, lunch, dinner — each with foods + gram quantities
food_suggestion_prompt = (
    "You are a clinical nutritionist and regional food expert.\n"
    "A patient has been diagnosed with: {disease}\n"
    "Their location / cuisine region: {region}\n\n"

    "DAILY MACRO TARGETS (from medical assessment):\n"
    "  Carbohydrates : {carbs_g} g\n"
    "  Protein       : {protein_g} g\n"
    "  Fat           : {fat_g} g\n"
    "  Fiber         : {fiber_g} g\n"
    "  Water         : {water_l} L\n\n"

    "DISEASE-SPECIFIC RESTRICTIONS for {disease}:\n"
    "{restrictions}\n\n"

    "RULES:\n"
    "1. Suggest ONLY foods commonly available in {region} cuisine.\n"
    "2. Every food must respect ALL restrictions for {disease}.\n"
    "3. Split the daily macros across 3 meals:\n"
    "   Breakfast 25% | Lunch 40% | Dinner 35%\n"
    "4. For each food item include exact gram quantity.\n"
    "5. Each meal must hit its macro target within 10%.\n"
    "6. Prefer whole, minimally processed foods.\n"
    "7. Mark any item that specifically HELPS {disease} recovery with (*).\n"
    "8. Respond ONLY in this exact JSON format — no extra text:\n\n"
    "{{\n"
    '  "disease": "{disease}",\n'
    '  "region": "{region}",\n'
    '  "daily_targets": {{\n'
    '    "carbs_g": {carbs_g}, "protein_g": {protein_g},\n'
    '    "fat_g": {fat_g}, "fiber_g": {fiber_g}, "water_l": {water_l}\n'
    "  }},\n"
    '  "meal_plan": {{\n'
    '    "breakfast": {{\n'
    '      "target_calories": "<25% of daily>",\n'
    '      "items": [\n'
    '        {{"food": "<name>", "quantity_g": <int>, "carbs_g": <int>,\n'
    '          "protein_g": <int>, "fat_g": <int>, "note": "<why good for disease>"}}\n'
    "      ]\n"
    "    }},\n"
    '    "lunch": {{ ... }},\n'
    '    "dinner": {{ ... }}\n'
    "  }},\n"
    '  "foods_to_avoid": ["<food1>", "<food2>"],\n'
    '  "hydration_note": "<disease-specific hydration tip>"\n'
    "}}"
)

# ── Disease restriction map — used to build {restrictions} field ──────────────
# Injected into food_suggestion_prompt so LLM knows hard dietary limits.
DISEASE_RESTRICTIONS = {
    "diabetes mellitus": (
        "No refined sugar, white rice, white bread, sugary drinks, fruit juices. "
        "Low glycaemic index foods only. Limit total carbs to complex sources."
    ),
    "hypertension": (
        "No added salt, pickles, processed/packaged foods, red meat, alcohol. "
        "Limit sodium to <1500 mg/day. Prefer potassium-rich foods."
    ),
    "tuberculosis": (
        "No alcohol. Avoid raw eggs. High calorie, high protein diet required. "
        "Vitamin B6 supplementation needed (isoniazid depletes it)."
    ),
    "anemia": (
        "Avoid tea/coffee with meals (inhibit iron absorption). "
        "Pair iron-rich foods with Vitamin C sources. No calcium supplements at mealtime."
    ),
    "chronic kidney disease": (
        "Strict potassium restriction. Limit phosphorus (dairy, nuts, seeds). "
        "Low protein unless on dialysis. Limit fluid intake. No salt substitutes."
    ),
    "gastroenteritis": (
        "No spicy food, fried food, dairy, raw vegetables, alcohol, caffeine. "
        "BRAT diet preferred (Banana, Rice, Applesauce, Toast). Small frequent meals."
    ),
    "influenza": (
        "No alcohol. Avoid cold foods and drinks. No heavy fried meals. "
        "Warm, easily digestible foods preferred."
    ),
    "hepatitis": (
        "No alcohol (strictly). No fatty or fried food. Low fat diet. "
        "No raw shellfish. Small frequent meals. Avoid paracetamol."
    ),
    "arthritis": (
        "Avoid red meat, processed foods, refined carbs, sugar, alcohol. "
        "Anti-inflammatory foods preferred (turmeric, ginger, omega-3 rich fish)."
    ),
    "meningitis": (
        "No alcohol. Easy-to-digest foods. High fluid intake. "
        "Avoid heavy meals during acute phase."
    ),
    "default": (
        "Avoid heavily processed foods, excess sugar, excess salt, and alcohol. "
        "Prefer whole foods appropriate for the condition."
    ),
}
