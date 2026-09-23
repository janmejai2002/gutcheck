"""`gutcheck train` and `gutcheck eval`."""
from __future__ import annotations

import json


def cmd_train(a):
    from .data import Task, format_report, read_rows
    from .head import finetune

    task = Task.load(a.task)
    rows = read_rows(a.data)
    ev = read_rows(a.eval) if a.eval else None
    m = finetune(task, rows, a.name or task.name, base=a.model, eval_rows=ev, epochs=a.epochs, lr=a.lr,
                 device=a.device, depth=a.depth)
    if m.get("eval"):
        print("\nbefore (zero-shot %s):\n%s" % (a.model, format_report(m["eval"]["before"])))
        print("\nafter (%s):\n%s" % (m["name"], format_report(m["eval"]["after"])))
    print("\nuse it:  gutcheck ask -m %s \"<text>\"   (answers the task's questions by default)" % m["name"])


def cmd_eval(a):
    from ..engine import Decider
    from .data import Task, evaluate_decider, format_report, read_rows

    task = Task.load(a.task)
    d = Decider(a.model, device=a.device, verbose=False)
    rep = evaluate_decider(d, task, read_rows(a.data))
    print(json.dumps(rep, indent=2) if a.json else format_report(rep))


def register(sub, common):
    sp = sub.add_parser("train", help="fine-tune on your labelled data (minutes, on a laptop)")
    sp.add_argument("--task", required=True, help="task spec JSON (questions + label columns)")
    sp.add_argument("--data", required=True, help=".jsonl / .csv / .json rows")
    sp.add_argument("--eval", help="held-out rows to report before/after metrics")
    sp.add_argument("--name", help="name for the trained model (default: task name)")
    sp.add_argument("--epochs", type=int, default=4)
    sp.add_argument("--lr", type=float, default=3e-4)
    sp.add_argument("--depth", type=int, default=4,
                    help="also train the top N encoder layers (default 4: ~25 min on a laptop CPU, much more accurate; "
                         "0 = decision head only: ~10 min, mostly improves calibration)")
    common(sp)
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("eval", help="score a model on labelled data (accuracy, macro-F1, ECE, Brier)")
    sp.add_argument("--task", required=True)
    sp.add_argument("--data", required=True)
    sp.add_argument("--json", action="store_true")
    common(sp)
    sp.set_defaults(fn=cmd_eval)
