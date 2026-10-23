
import os

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
        f"[Chunk {chunk.document_id}:{chunk.chunk_index}]\n{chunk.text}"
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

                return response["message"]["content"]

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
