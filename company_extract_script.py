def extract_companies_and_focus(title, text):

    model = llm(gemini_model_name, google_api_key)

    def extract_with_ner(text, lang="en"):
        doc = nlp_en(text.page_content) if lang == "en" else nlp_zh(text.page_content)
        return [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]

    def extract_with_regex(text):
        matches = company_regex.findall(text.page_content)
        return [m[0] or m[1] for m in matches if m[0] or m[1]]

    def clean_company_names(names):
        cleaned = set()
        for name in names:
            name = name.strip("，。：:,. ").replace("（", "(").replace("）", ")").lower()
            if len(name) > 1 and name not in company_stopwords:
                cleaned.add(name.title())
        return sorted(cleaned)

    def title_frequency_focus(title, text, company_list):
        title_lower = title.lower()
        text_lower = text.page_content.lower()
        scores = {}

        # Step 1: Compute weighted score based on title presence and frequency
        for name in company_list:
            name_lower = name.lower()
            score = 0

            if name_lower in title_lower:
                score += 5  # strong signal if mentioned in title

            freq = text_lower.count(name_lower)
            score += freq  # add frequency score

            if freq > 0:
                scores[name] = score

        if not scores:
            return None

        # Step 2: Find top-scoring candidates
        max_score = max(scores.values())
        top_candidates = [name for name, score in scores.items() if score == max_score]

        if len(top_candidates) == 1:
            return top_candidates[0]
        else:
            # Step 3: Tie-break by earliest appearance
            earliest = min(top_candidates, key=lambda name: text_lower.find(name.lower()))
            return earliest

    def llm_extract_and_focus(title, text):
        prompt = f"""
                Given the document title and content below, extract all company names mentioned (both English and Chinese).
                Then, identify the main company the document is primarily about.

                Title:
                {title}

                Text (partial):
                {text.page_content[:1500]}

                Respond with JSON format exactly like this:
                {{
                "companies": ["Company A", "Company B"],
                "focus_company": "Company A"
                }}

                If no company found or focus, respond with empty list or null values.
                """
        response = model.generate_content(prompt)
        try:
            data = json.loads(response.text)
            companies = data.get("companies", [])
            focus_company = data.get("focus_company", None)
            if focus_company not in companies:
                focus_company = None
            return companies, focus_company
        except Exception:
            return [], None

    # --- Run Extraction Pipeline ---
    en_companies = extract_with_ner(text, "en")
    zh_companies = extract_with_ner(text, "zh")
    regex_companies = extract_with_regex(text)
    all_companies = clean_company_names(en_companies + zh_companies + regex_companies)

    if len(all_companies) >= 2:
        focus = title_frequency_focus(title, text, all_companies)
        if focus:
            return all_companies, focus

    return llm_extract_and_focus(title, text)