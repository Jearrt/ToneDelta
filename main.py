# python3 main.py --ticker AAPL --year 2025 --quarter 1
# python3 main.py --ticker MSFT --year 2025 --quarter 1
# Purpose: CLI entry point — parse args, orchestrate fetch -> analyse -> report.

import argparse
from fetch import fetch_transcript
from analyse import load_finbert, analyse_section
from report import print_report

def parse_args():
    # Create a parser object 
    parser = argparse.ArgumentParser(description="EarningsLens — Earnings Call Sentiment Analyser") 

    # Add arguments that the parser should expect
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker e.g. AAPL")
    parser.add_argument("--year", type=int, required=True, help="Year e.g. 2024")
    parser.add_argument("--quarter", type=int, required=True, help="Quarter 1-4")

    # Parse and return 
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading FinBERT model...")
    pipe = load_finbert()

    print(f"Fetching transcript for {args.ticker} Q{args.quarter} {args.year}...")
    transcript = fetch_transcript(args.ticker, args.year, args.quarter)

    print("Analysing Prepared Remarks...")
    remarks_result = analyse_section(transcript["prepared_remarks"], pipe)

    print("Analysing Q&A...")
    qa_result = analyse_section(transcript["questions_and_answers"], pipe)

    print_report(
        company_name=transcript["company_name"],
        ticker=args.ticker,
        year=args.year,
        quarter=args.quarter,
        remarks_result=remarks_result,
        qa_result=qa_result
    )

if __name__ == "__main__":
    main()
