import argparse
import sys
from studio_core.data_io import load_retail_demo
from studio_core.example_prompt import RETAIL_PROMPTS, RETAIL_CODE, RETAIL_SOURCE
from studio_core.studio import PromptNeedsAnswer, PromptRewriteOffered, analyze
from studio_core.app import launch, reference_forecast


def main():
    parser = argparse.ArgumentParser(description="Macroeconomic forecasting and cross-sectional or monthly DML")
    commands = parser.add_subparsers(dest="command")
    app = commands.add_parser("app", help="Open the local prompt page")
    app.add_argument("--port", type=int, default=0)
    app.add_argument("--no-browser", action="store_true")
    demo = commands.add_parser("demo", help="Run the fixed offline teaching example")
    ask = commands.add_parser("ask", help="Interpret a new prompt with local Ollama")
    ask.add_argument("prompt")
    ask.add_argument("--data", help="CSV of independent units or monthly observations (monthly data require a date column)")
    ask.add_argument("--model", default="llama3.1:8b")
    for subparser in (demo, ask):
        subparser.add_argument("--out", help="New directory for CSV results and audit.json")
    args = parser.parse_args()
    try:
        if args.command is None or args.command == "app":
            launch(getattr(args, "port", 0), not getattr(args, "no_browser", False))
            return
        def log(e):
            print(f"Round {e['round']} / {e['role']}: {e['message']}", flush=True)
            if e.get("suggested_prompt"):
                print("Suggested wording:", e["suggested_prompt"])
            if "code" in e:
                print(e["code"])
        if args.command == "demo":
            res = reference_forecast(load_retail_demo(), prompt=RETAIL_PROMPTS[0], code=RETAIL_CODE, source=RETAIL_SOURCE)
        else:
            exchanges = []
            wording = None
            while True:
                try:
                    res = analyze(args.prompt, args.data, args.model, on_event=log, clarifications=exchanges,
                                  pause_for_rewrite=sys.stdin.isatty(), wording=wording)
                    break
                except PromptRewriteOffered as exc:
                    try:
                        answer = input("Use the suggestion [y], enter your rewrite, or press Enter to keep the original: ").strip()
                    except EOFError:
                        answer = ""
                    action = "keep" if not answer else "use" if answer.lower()=="y" else "edit"
                    wording = {"action": action, "suggestion": str(exc), "edited": answer if action=="edit" else ""}
                except PromptNeedsAnswer as exc:
                    if not sys.stdin.isatty() or len(exchanges) == 6:
                        raise
                    try:
                        answer = input(f"{exc}\nYour answer: ").strip()
                    except EOFError:
                        answer = ""
                    if not answer:
                        raise ValueError("No clarification supplied.") from exc
                    exchanges.append({"question": str(exc), "answer": answer})
        if args.command == "demo":
            print(res.audit["original_prompt"])
        print(res.metrics.round(4).to_string())
        if "next_month" in res.audit:
            print("Weights:", res.audit["weights"])
            print("Next forecast:", res.audit["next_month"])
        if "interpretation" in res.audit:
            print(res.audit["interpretation"])
        if args.out:
            res.save(args.out)
            print("Results written to", args.out)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Stopped: {exc}\n")
