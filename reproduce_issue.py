from sentiment_engine import run_pipeline
import sys

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

test_text = """OMG!!! 🔥 The sound quality on this ALEXA is literally AMAZING!!! 😍😍😍 but, the cable conection is super sketchhh and it keeps disconecting... not good. Maybe a bad unit??"""

print(f"Original Text: {test_text}\n")

steps = run_pipeline(test_text)

for step in steps:
    print(f"STEP {step['step_no']}: {step['step_name']}")
    print(f"  Description: {step['description']}")
    print(f"  Result: {step['result_text']}")
    print("-" * 20)
