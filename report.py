# python3 report.py
# Purpose: Compute credibility score from two sentiment distributions,
#          format and print the final CLI report.

import math

def kl_divergence(p: dict, q: dict) -> float:
    """
    Computes KL Divergence between two probability distributions p and q.
    p = prepared remarks sentiment dict {"positive": float, "neutral": float, "negative": float}
    q = Q&A sentiment dict (same structure)

    KL divergence measures how different q is from p.
    Returns a float >= 0. The closer to 0, the more similar the distributions.

    KL(P || Q) = sum over all labels: p(x) * log(p(x) / q(x))
    Add a small epsilon (1e-9) to all values before dividing to avoid log(0).
    """

    keys = ["positive", "neutral", "negative"]
    epsilon = 1e-9
    kl_value = 0

    for key in keys:
        p_value = p[key] + epsilon  # Add epsilon to ensure that the fraction wont be 0 since log(0) is undefined
        q_value = q[key] + epsilon
        fraction = p_value / q_value 
        kl_value += p_value * math.log(fraction)

    return float(kl_value)


def credibility_score(kl: float, max_kl: float = 2.0) -> float:
    """
    Converts a KL divergence value into a 0-100 credibility score.
    Higher score = Q&A tone is consistent with prepared remarks.
    Lower score = significant tone shift detected.

    credibility = [(1 - (min(kl, max_kl) / max_kl)] * 100
    max_kl is the divergence value you treat as "maximum possible shift" —
    2.0 is a reasonable ceiling for 3-class distributions.
    """
    return round(min((1 - min(kl, max_kl) / max_kl) * 100, 100.0), 2)

def interpret_credibility(score: float) -> str:
    """
    Returns a plain-English interpretation of the credibility score.
    Also append a caveat: a low score reflects tone divergence, not confirmed dishonesty.
    """
    credibility_statement = "Q&A tone strongly consistent with prepared remarks, good sign 🟩" if score >= 85 else "Minor tone shift detected, worth monitoring 🟨" if 70 <= score < 85 else "Moderate tone shift detected, be cautious 🟧" if 40 <= score < 70 else "Large tone shift, be wary of inconsistency between sections 🟥"

    return credibility_statement


def print_report(company_name: str, ticker: str, year: int, quarter: int,
                 remarks_result: dict, qa_result: dict):
    """
    Prints the full CLI report:

    ═══════════════════════════════════════
    AAPL — Apple Inc.  |  Q3 2024
    ═══════════════════════════════════════

    PREPARED REMARKS
    Sentiment : Positive (81%)

    Q&A SECTION
    Sentiment : Negative (35%)

    CREDIBILITY SCORE: 34 / 100
    Large tone shift, be wary of inconsistency between sections
    Note: score reflects tone divergence, not confirmed dishonesty.

    """

    remarks_sentiment = remarks_result["dominant"]
    remarks_sentiment_percentage = remarks_result[remarks_sentiment]

    qna_sentiment = qa_result["dominant"]
    qna_sentiment_percentage = qa_result[qna_sentiment]

    kl_value = kl_divergence(remarks_result, qa_result)
    credibility = credibility_score(kl_value)
    credibility_statement = interpret_credibility(credibility)

    print()
    print(f"═══════════════════════════════════════\n{ticker}: {company_name} | Q{quarter} {year}\n═══════════════════════════════════════")
    print()
    print(f"PREPARED REMARKS\n Sentiment: {remarks_sentiment} ({remarks_sentiment_percentage:.0%})")
    print()
    print(f"Q&A SECTION\n Sentiment: {qna_sentiment} ({qna_sentiment_percentage:.0%})")
    print()
    # print(f"remarks: {remarks_result}")
    # print(f"qa: {qa_result}")
    # print(f"kl raw: {kl_value}")
    print(f"CREDIBILITY SCORE: {credibility} / 100")
    print(credibility_statement)
    print("Note: score reflects tone divergence, not confirmed dishonesty.")
    print()
