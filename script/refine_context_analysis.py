#####################################################################################
    # Query with Refine strategy 
    # First chunk
    existing_answer = ""
    for idx, chunk in enumerate(retrieved_chunks):
        if idx == 0:
            # Initial generation (include conversation history here)
            prompt = f"""
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

                    === Conversation History ===
                    {history_text}

                    Answer:"""
            time.sleep(4)
            response = model.generate_content([{"role": "user", "parts": [prompt]}])
            existing_answer = response.text.strip()
        else:
            # Include conversation history only on the final chunk
            if idx == len(retrieved_chunks) - 1:
                refine_prompt = f"""
                            You are refining an answer based on new context and previous reasoning. Improve the previous answer if new context adds value.

                            ---
                            PREVIOUS ANSWER:
                            {existing_answer}

                            NEW CONTEXT:
                            {chunk}

                            CONVERSATION HISTORY:
                            {history_text}

                            QUESTION:
                            {query_question}

                            Only modify if new context provides additional information, corrections, or clarifications.
                            ---

                            Refined Answer:"""
            else:
                refine_prompt = f"""
                            You are refining an answer based on new context and previous reasoning. Improve the previous answer if new context adds value.

                            ---
                            PREVIOUS ANSWER:
                            {existing_answer}

                            NEW CONTEXT:
                            {chunk}

                            QUESTION:
                            {query_question}

                            Only modify if new context provides additional information, corrections, or clarifications.
                            ---

                            Refined Answer:"""
            
            time.sleep(4)
            response = model.generate_content([{"role": "user", "parts": [refine_prompt]}])
            existing_answer = response.text.strip()

    return existing_answer
    #####################################################################################