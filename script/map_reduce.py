#####################################################################################################
    # Use Map_Reduce strategy to analyze context
    # MAP STEP — generate intermediate answers
    intermediate_answers = []
    for chunk in retrieved_chunks:
        map_prompt = f"""
            You are a smart expert assistant helping the user answer questions. You have access to:

            1) Relevant Context: documents and metadata provided by the user.
            2) Your own general knowledge and facts up to your knowledge cutoff.

            For every answer, combine both sources **even if the context is sufficient.**
            Use the format below:
            ---
            **From Context**: [Clearly indicate what you found in the context, cite metadata if possible.]

            **From General Knowledge**: [Add background, facts, or reasoning based on your own knowledge.]
            ---

            Do not skip either section unless the question is purely factual with no context overlap.
            
            === Relevant Context ===
            {chunk}

            === Question ===
            {query_question}

            Answer:"""
        
        time.sleep(4)
        response = model.generate_content([{"role": "user", "parts": [map_prompt]}])
        intermediate_answers.append(response.text.strip())
    
    # REDUCE STEP — summarize all intermediate answers
    joined_answers = "\n\n---\n\n".join(intermediate_answers)
    reduce_prompt = f"""
            You are a helpful assistant. The following are multiple answers to the same question, generated from different documents and contexts.

            Your job is to combine them into one complete, non-redundant answer.

            QUESTION:
            {query_question}

            CONVERSATION HISTORY:
            {history_text}

            INDIVIDUAL ANSWERS:
            {joined_answers}

            FINAL ANSWER:"""
    time.sleep(4)
    final_response = model.generate_content([{"role": "user", "parts": [reduce_prompt]}])

    return final_response.text.strip()
    #####################################################################################################