# python3 analyse.py
# Purpose: Run FinBERT sentiment analysis on a block of text.
#           Splits text into sentences, scores each, returns aggregated result.

from transformers import BertTokenizer, BertForSequenceClassification, pipeline

def load_finbert():
    """
    Loads and returns a HuggingFace sentiment-analysis pipeline using FinBERT.
    Call this once and reuse the result — loading a model is expensive.
    """
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer, top_k=3)

    return nlp


def chunk_text(text: str, max_sentences: int = 10) -> list[str]:
    """
    Splits a long transcript section into manageable chunks.
    FinBERT has a 512-token limit — passing the full transcript at once will fail.

    Returns a list of chunks where each chunk is a few sentences combined together 
    Each chunk is set as 10 sentences (max_sentences) and can be decreased/increased depending on whether or not the 512 tokens limit is hit 
    """
    sentence_list = [sentence for sentence in text.split(". ")]
    chunks = []

    for i in range(0, len(sentence_list), max_sentences):   # range(start, stop - 1, step) 
            sentence_chunk = sentence_list[i : max_sentences + i]
            joined_chunk = "".join(sentence_chunk)
            chunks.append(joined_chunk)

    return chunks 


def analyse_section(text: str, pipe) -> dict:
    """
    Runs FinBERT on each chunk of the section.
    Aggregates scores across all chunks.
    Returns:
        {
            "positive": float,   # average probability
            "neutral": float,
            "negative": float,
            "dominant": str      # "positive", "neutral", or "negative"
        }
    """
    chunks = chunk_text(text)
    positive = []
    negative = []
    neutral = []

    for chunk in chunks:
        results = pipe(chunk)   # [[{'label': 'Positive', 'score': 0.9999997615814209}, {'label': 'Negative', 'score': 1.957796769147535e-07}, {'label': 'Neutral', 'score': 3.637225631791807e-08}]]
        for item in results[0]:
            if item["label"] == "Positive":
                positive.append(item["score"])
            elif item["label"] == "Negative":
                negative.append(item["score"])
            else:
                neutral.append(item["score"])

    scores = {
        "positive": sum(positive) / len(positive) if positive else 0.0,
        "neutral": sum(neutral) / len(neutral) if neutral else 0.0,
        "negative": sum(negative) / len(negative) if negative else 0.0,
    }

    dominant = max(scores, key=scores.get)  # key= here tells max() to compare based on values for scores.get("positive"), scores.get("neutral"), scores.get("negative")
    scores.update({"dominant": dominant})

    return scores
