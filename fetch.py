# python3 fetch.py
# Purpose: Fetch earnings call transcript for a given ticker and quarter,
#           return prepared remarks and Q&A as separate strings.

from earningscall import get_company

def fetch_transcript(ticker: str, year: int, quarter: int) -> dict:
    """
    Returns a dict with keys:
        "prepared_remarks" -> str
        "questions_and_answers" -> str
        "company_name" -> str
    Raises ValueError if transcript not found.
    """
    company = get_company(ticker)
    transcript = company.get_transcript(year=year, quarter=quarter, level=4)    # lvl 4 gives access to prepared remarks and the q&a sections

    if not transcript:  # If no transcipts are available for the company in that quarter and year 
        raise ValueError(f"No transcript found for {ticker} Q{quarter} {year}")

    prepared = transcript.prepared_remarks
    qna = transcript.questions_and_answers
    
    return {
        "prepared_remarks": str(prepared),
        "questions_and_answers": str(qna),
        "company_name": str(company)
    }
    