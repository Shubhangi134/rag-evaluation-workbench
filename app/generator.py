
import os
import re
import ollama
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = (
    "You answer questions about documents using only the provided "
    "retrieved context. Do not use your general knowledge to fill gaps. "
    "Do not invent facts. If the context is insufficient, say so clearly. "
    "Cite the supporting chunk IDs in your answer."
)


def format_answer_citations(answer, retrieved_chunks):
    """Convert valid chunk references into PDF page citations."""

    sources = {
        str(chunk.chunk_index): (
            f"[{chunk.document_id}, Page {chunk.page_number}]"
        )
        for chunk in retrieved_chunks
        if chunk.page_number is not None
    }

    def replace_citation(match):
        chunk_id = match.group(1)
        return sources.get(chunk_id, "[Source unavailable]")

    # Format 1: 【33】
    answer = re.sub(
        r"【(\d+)】",
        replace_citation,
        answer,
    )

    # Format 2: 【Source: filename, Page 6, Chunk 33】
    answer = re.sub(
        r"【Source:[^】]*Chunk\s+(\d+)】",
        replace_citation,
        answer,
    )

    return answer


def generate_answer(
    question: str,
    retrieved_chunks: list,
    conversation_history: list | None = None,
) -> str:
    """Generate an answer using the configured LLM provider."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        return "I couldn't find relevant information in the retrieved document."

    
    context = "\n\n".join(
        (
            f"[Source: {chunk.document_id}, "
            f"Page {chunk.page_number}, "
            f"Chunk {chunk.chunk_index}]\n"
            f"{chunk.text}"
        )
        for chunk in retrieved_chunks
    )


    history_text = ""
    if conversation_history:
        history_text = "\n".join(
            f"{item['role'].capitalize()}: {item['content']}"
            for item in conversation_history
            if item.get("role") in {"user", "assistant"}
            and item.get("content")
        )

    user_prompt = (
        f"Previous conversation:\n{history_text or 'No previous conversation.'}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Current question: {question}"
    )

    primary_provider = os.getenv(
        "PRIMARY_LLM_PROVIDER", "ollama"
    ).strip().lower()

    fallback_provider = os.getenv(
        "FALLBACK_LLM_PROVIDER", ""
    ).strip().lower()

    providers = [primary_provider]

    if fallback_provider and fallback_provider != primary_provider:
        providers.append(fallback_provider)

    errors = []

    for provider in providers:
        try:
            if provider == "ollama":
                api_key = os.getenv("OLLAMA_API_KEY", "").strip()

                if not api_key:
                    raise ValueError(
                        "OLLAMA_API_KEY is not configured."
                    )

                client = ollama.Client(
                
                )

                response = client.chat(
                    model=os.getenv(
                        "OLLAMA_MODEL", "gpt-oss:120b-cloud"
                    ),
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                )

                answer = response["message"]["content"]

                return format_answer_citations(
                    answer,
                    retrieved_chunks,
                )

            elif provider == "openai":
                api_key = os.getenv(
                    "OPENAI_API_KEY", ""
                ).strip()

                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY is not configured."
                    )

                client = OpenAI(api_key=api_key)

                response = client.chat.completions.create(
                    model=os.getenv(
                        "OPENAI_MODEL", "gpt-4.1"
                    ),
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                )

                answer = response.choices[0].message.content

                if not answer:
                    raise RuntimeError(
                        "The provider returned an empty answer."
                    )

                return answer

            else:
                raise ValueError(
                    f"Unsupported LLM provider: {provider}"
                )

        except Exception as exc:
            # Keep trying the next configured provider.
            # Do not include API keys in error messages.
            errors.append(
                f"{provider}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "All configured LLM providers failed. "
        + " | ".join(errors)
    )

def rewrite_question(
    question: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """Rewrite a follow-up question using previous chat history."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # No history means the question can be used directly.
    if not conversation_history:
        return question.strip()

    # Use only the last 10 messages.
    recent_history = conversation_history[-10:]

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in recent_history
        if message.get("role") in ("user", "assistant")
        and message.get("content")
    )

    if not history_text:
        return question.strip()

    system_prompt = (
        "You rewrite questions for a PDF search system. "
        "Use the previous conversation to understand references "
        "such as 'it', 'its', 'this method', or 'that paper'. "
        "Rewrite the current question as a standalone question. "
        "If the question is already clear or starts a new topic, "
        "keep its meaning unchanged. "
        "Do not answer the question. "
        "Do not invent information. "
        "Return only the rewritten question."
    )

    user_prompt = (
        f"Previous conversation:\n{history_text}\n\n"
        f"Current question: {question}\n\n"
        "Standalone question:"
    )

    try:
        client = ollama.Client()

        response = client.chat(
            model=os.getenv(
                "OLLAMA_MODEL",
                "gpt-oss:120b-cloud",
            ),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        rewritten = response["message"]["content"].strip()

        # Use the original question if the model returns nothing.
        return rewritten or question.strip()

    except Exception as error:
        print(
            f"Question rewriting unavailable: "
            f"{type(error).__name__}"
        )

        # Keep the chatbot usable if rewriting fails.
        return question.strip()
