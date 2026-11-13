"""Reproducible synthetic helpdesk data, with a held-out wording split.

IMPORTANT: These are simulated, labelled examples, not real support tickets.
They demonstrate a complete ML workflow; field performance is unvalidated.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .tokenizer import WordTokenizer

type Ticket = dict[str, str]

INTENTS = (
    "access_reset",
    "billing_issue",
    "bug_report",
    "feature_request",
    "service_outage",
    "account_cancellation",
)
PRIORITIES = ("low", "normal", "high")

# Wording templates 0-9 are for train/val, 10-12 for test, 13 for challenge.
# Several keywords are deliberately common across classes, to require contextual learning.
TEMPLATES = {
    "access_reset": [
        "I cannot sign in to {product}; my password reset link fails.",
        "Locked out of {product} after changing my password.",
        "Login to {product} says my account credentials are invalid.",
        "Please help me recover access to {product}.",
        "Two-factor code does not arrive when I try to enter {product}.",
        "I forgot my password for {product} and need to log in.",
        "Sign-in keeps redirecting me to the authentication page on {product}.",
        "My login is blocked after too many attempts on {product}.",
        "I cannot get past the verification step to open {product}.",
        "The account recovery email for {product} never arrived.",
        "Can't access {product} because the one-time code is rejected.",
        "Authentication to {product} is denying my valid credentials.",
        "I was signed out of {product} and recovery isn't working.",
        "Getting into {product} fails at the identity check.",
    ],
    "billing_issue": [
        "The invoice for {product} shows an incorrect amount.",
        "I was charged twice for {product} this month.",
        "Our credit card payment for {product} was declined.",
        "Can you explain the new fee on our {product} bill?",
        "We need a corrected receipt for a {product} payment.",
        "My {product} subscription renewed at the wrong price.",
        "The payment screen for {product} fails to accept my card.",
        "I need a refund for an extra {product} invoice.",
        "Tax on the {product} invoice seems incorrect.",
        "We have a question about charges for {product} seats.",
        "Our {product} statement includes a duplicate transaction.",
        "The amount billed by {product} doesn't match our plan.",
        "I need help reconciling the latest {product} receipt.",
        "The billing total in {product} looks unexpectedly high.",
    ],
    "bug_report": [
        "The export button in {product} crashes when I click it.",
        "Saving a dashboard in {product} returns a server error.",
        "The {product} filters produce incorrect search results.",
        "A report in {product} is missing some records.",
        "I found a UI bug with the {product} settings menu.",
        "The {product} API returns 500 for a valid request.",
        "After refreshing {product}, the chart displays wrong values.",
        "The notifications panel in {product} freezes intermittently.",
        "Uploading a small file to {product} throws an exception.",
        "A newly created table in {product} disappears after save.",
        "The {product} editor refuses to save valid changes.",
        "Something is broken in the {product} sorting controls.",
        "The {product} application shows stale data after reload.",
        "A normal action in {product} triggers an unexpected error.",
    ],
    "feature_request": [
        "Could you add a dark mode option to {product}?",
        "Please support scheduled report exports in {product}.",
        "I'd like a bulk editing feature in {product}.",
        "Can {product} offer a CSV import wizard?",
        "It would be helpful to customize alerts in {product}.",
        "I have a suggestion for a new {product} dashboard view.",
        "Please consider adding team templates to {product}.",
        "We would like more keyboard shortcuts in {product}.",
        "Is there a plan to support recurring tasks in {product}?",
        "A new calendar integration for {product} would help us.",
        "I'd love an option to save searches in {product}.",
        "Could the {product} interface allow custom layouts?",
        "Requesting a new notification preference in {product}.",
        "A batch update capability would improve our {product} workflow.",
    ],
    "service_outage": [
        "{product} is completely unavailable to our entire team.",
        "None of us can reach the {product} service right now.",
        "The {product} status page shows systems are down.",
        "All requests to {product} time out across our office.",
        "{product} is returning 503 for every user in our company.",
        "Our organization cannot use {product} since the outage began.",
        "The whole {product} site has been offline for an hour.",
        "Multiple regions report {product} is not responding.",
        "Everyone on our team sees a service unavailable message in {product}.",
        "The {product} platform is inaccessible from multiple networks.",
        "The {product} service appears down for all employees.",
        "Company-wide requests to {product} have stopped working.",
        "A widespread availability incident is affecting {product}.",
        "No one in our group can open the {product} application.",
    ],
    "account_cancellation": [
        "I want to cancel my {product} subscription at the end of this month.",
        "Please close our {product} account and stop future renewals.",
        "How do I terminate my organization's {product} plan?",
        "I no longer need {product}; please end the membership.",
        "Can you disable auto-renewal and cancel {product}?",
        "We are leaving {product} and would like to close the account.",
        "Please process a cancellation request for {product}.",
        "What is the procedure to discontinue our {product} service?",
        "I'd like to end the {product} contract before next renewal.",
        "Our team wants to deactivate its paid {product} plan.",
        "We decided to stop subscribing to {product} next cycle.",
        "Please end future billing by cancelling our {product} plan.",
        "I would like to formally shut down our {product} subscription.",
        "We need to terminate the {product} account permanently.",
    ],
}

PRODUCTS = [
    "AtlasDesk",
    "NovaBoard",
    "CloudLedger",
    "OrbitCRM",
    "PulseReports",
    "ZenFlow",
    "DataHarbor",
    "TeamSphere",
]
CONTEXTS = [
    "This started today.",
    "It happened yesterday.",
    "We use the web app.",
    "I am writing from our mobile browser.",
    "The issue occurs in our staging workspace.",
    "This is for our production workspace.",
    "My colleague asked me to contact support.",
    "I tried it again after lunch.",
    "Our account is on the standard plan.",
    "The same request was fine last week.",
]
# Urgency is deliberately separate from the ticket intent.
URGENCY = {
    "high": [
        "This blocks an important business deadline and we need help immediately.",
        "Our team is blocked from an essential task today.",
        "Please prioritize this: a customer deliverable is due within an hour.",
        "We cannot finish critical work until this is resolved.",
        "This is time-sensitive and affecting an important launch.",
    ],
    "normal": [
        "Please help when you can during normal business hours.",
        "This matters to our weekly workflow but there is no immediate deadline.",
        "I would appreciate a response in the next few days.",
        "This can be handled within your standard support window.",
        "There is no immediate emergency, but I'd like guidance.",
    ],
    "low": [
        "There is no rush; next week is fine.",
        "This is a minor question for whenever you have time.",
        "It is not urgent and can wait until later.",
        "I'm planning ahead and don't need an immediate answer.",
        "This is only for future reference, not an active blocker.",
    ],
}


def _make_record(
    rng: random.Random,
    intent: str,
    priority: str,
    template_index: int,
    ticket_id: str,
    templates: dict[str, list[str]] = TEMPLATES,
) -> Ticket:
    product = rng.choice(PRODUCTS)
    base = templates[intent][template_index].format(product=product)
    urgency = rng.choice(URGENCY[priority])
    context = rng.choice(CONTEXTS)
    pieces = [base, urgency, context]

    if rng.random() < 0.5:
        pieces = [context, base, urgency]
    elif rng.random() < 0.25:
        pieces = [urgency, base, context]

    return {
        "id": ticket_id,
        "text": " ".join(pieces),
        "intent": intent,
        "priority": priority,
        "template_id": f"{intent}:{template_index}",
    }


def generate_dataset(
    output_dir: str | Path,
    seed: int = 2026,
    n_train: int = 3600,
    n_val: int = 612,
    n_test: int = 612,
    n_challenge: int = 180,
    version: str = "ticket-v1",
) -> dict[str, Path]:
    """Generate balanced, unique synthetic tickets; test uses unseen wording templates."""
    if version not in {"ticket-v1", "ticket-v2"}:
        raise ValueError("Unknown dataset version")

    sizes = {"train": n_train, "val": n_val, "test": n_test, "challenge": n_challenge}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    seen = set()
    paths = {}

    for split, n in sizes.items():
        if n < 18 or n % (len(INTENTS) * len(PRIORITIES)):
            raise ValueError("split size must be divisible by 18 for balancing")

        rows: list[Ticket] = []
        per_group = n // (len(INTENTS) * len(PRIORITIES))

        for intent in INTENTS:
            for priority in PRIORITIES:
                for _ in range(per_group):
                    while True:
                        templates = (
                            TEMPLATES
                            if version == "ticket-v1" or split == "train"
                            else V2_TEMPLATES[split]
                        )
                        choices = (
                            (
                                range(0, 10)
                                if split in ("train", "val")
                                else range(10, 13)
                                if split == "test"
                                else [13]
                            )
                            if version == "ticket-v1"
                            else range(
                                10 if split == "train" else len(templates[intent])
                            )
                        )
                        template_index = rng.choice(choices)
                        record = _make_record(
                            rng,
                            intent,
                            priority,
                            template_index,
                            f"{split}-{len(rows):05d}",
                            templates,
                        )

                        if version == "ticket-v2":
                            record["template_id"] = (
                                f"ticket-v2:{split}:{intent}:{template_index}"
                            )

                        if record["text"] not in seen:
                            seen.add(record["text"])
                            rows.append(record)
                            break

        rng.shuffle(rows)
        path = output_dir / f"{split}.csv"

        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["id", "text", "intent", "priority", "template_id"]
            )
            w.writeheader()
            w.writerows(rows)

        paths[split] = path

    return paths


def read_rows(path: str | Path) -> list[Ticket]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# Authored before ticket-v2 training. Each split uses different wording families;
# the original ticket-v1 generator above remains byte-compatible.
V2_TEMPLATES = {
    "val": {
        "access_reset": [
            "My {product} password has expired and the replacement is not accepted.",
            "I cannot authenticate my identity to sign into {product}.",
            "The {product} recovery process sends me around in a login loop.",
            "I need assistance restoring my personal sign-in to {product}.",
        ],
        "billing_issue": [
            "There is an unexplained extra charge on our {product} receipt.",
            "Our {product} payment was collected twice for one subscription.",
            "Please investigate a discrepancy in the {product} invoice amount.",
            "We paid for {product} but the bill still says payment is overdue.",
        ],
        "bug_report": [
            "The {product} search bar stops responding after entering a query.",
            "A software defect in {product} duplicates saved records.",
            "The {product} report download finishes with a corrupted file.",
            "I can reproduce an exception in the {product} editing screen.",
        ],
        "feature_request": [
            "Please add an undo option for changes made in {product}.",
            "Our team suggests a new integration for {product}.",
            "Could {product} include a way to organize reports into folders?",
            "A configurable dashboard theme would be a useful addition to {product}.",
        ],
        "service_outage": [
            "Every employee is unable to reach {product} from any location.",
            "The entire {product} platform has become unavailable company-wide.",
            "No requests to {product} succeed for any of our users.",
            "Our monitoring confirms a widespread {product} service outage.",
        ],
        "account_cancellation": [
            "Please end our paid membership of {product} and confirm closure.",
            "We have decided to discontinue our {product} subscription.",
            "Do not renew our {product} contract; we wish to cancel it.",
            "Our organization requests permanent closure of its {product} account.",
        ],
    },
    "test": {
        "access_reset": [
            "The verification code for my {product} login is never delivered.",
            "My {product} credentials are refused despite resetting the password.",
            "The authentication challenge prevents me from entering {product}.",
            "Help me regain access to my own {product} profile after a lockout.",
        ],
        "billing_issue": [
            "The {product} charge on our card exceeds the agreed subscription price.",
            "I need a refund because {product} processed a duplicate payment.",
            "Our finance team found an incorrect tax item on the {product} bill.",
            "Please correct the amount we owe for this month's {product} seats.",
        ],
        "bug_report": [
            "Changing a filter in {product} makes the displayed totals incorrect.",
            "The {product} interface crashes whenever I attach a small file.",
            "Records vanish after I press save in the {product} table editor.",
            "An error in the {product} export function creates empty reports.",
        ],
        "feature_request": [
            "Would you consider introducing reusable project templates in {product}?",
            "I propose adding advanced notification controls to {product}.",
            "We would like {product} to support batch updates in a future version.",
            "Can the {product} roadmap include automatic scheduled exports?",
        ],
        "service_outage": [
            "All teams report that {product} has gone offline across our organization.",
            "The {product} servers return service unavailable for every employee.",
            "Nobody can connect to {product}; the platform appears down everywhere.",
            "A company-wide availability incident has interrupted all {product} requests.",
        ],
        "account_cancellation": [
            "We are ending our relationship with {product}; please terminate the plan.",
            "I want our {product} membership cancelled rather than renewed.",
            "Please deactivate the paid {product} subscription after this billing period.",
            "Our company no longer requires {product} and requests account cancellation.",
        ],
    },
    "challenge": {
        "access_reset": [
            "Only my user is locked out of {product}; everyone else can log in.",
            "The identity verification step on {product} keeps rejecting me.",
            "I have lost the means to authenticate to my {product} account.",
        ],
        "billing_issue": [
            "We were invoiced for {product} licenses we never purchased.",
            "Please reconcile a mistaken transaction on our {product} statement.",
            "The {product} renewal deducted an unexpected sum from our card.",
        ],
        "bug_report": [
            "One {product} button triggers a traceback, although the rest works.",
            "The {product} application corrupts a record when I edit it.",
            "Sorting a {product} table produces a reproducible software error.",
        ],
        "feature_request": [
            "A future release of {product} should let us automate repetitive work.",
            "Please put a custom reporting capability on the {product} roadmap.",
            "I am suggesting an enhancement to the {product} user interface.",
        ],
        "service_outage": [
            "This is affecting every {product} user, with no successful connections.",
            "The entire company has lost access because {product} is offline.",
            "Our {product} availability checks fail in every region.",
        ],
        "account_cancellation": [
            "Please wind down our {product} membership and cease future renewal.",
            "We are moving away from {product} and want the subscription terminated.",
            "This is a formal request to close our paid {product} service.",
        ],
    },
}


class TicketDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self, rows: list[Ticket], tokenizer: WordTokenizer, max_seq_len: int
    ) -> None:
        self.items: list[tuple[Tensor, Tensor]] = []

        for row in rows:
            if row["intent"] not in INTENTS or row["priority"] not in PRIORITIES:
                raise ValueError("Unknown ticket label")

            prompt = tokenizer.encode_prompt(row["text"])
            answer = tokenizer.answer_ids(row["intent"], row["priority"])
            ids = prompt + answer

            if len(ids) > max_seq_len:
                raise ValueError(
                    "Input too long; increase context rather than dropping examples"
                )

            self.items.append(
                (
                    torch.tensor(ids[:-1]),
                    torch.tensor([-100] * (len(prompt) - 1) + answer),
                )
            )

        if not self.items:
            raise ValueError("Training and validation datasets must be nonempty")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.items[index]


def batchify(items: list[tuple[Tensor, Tensor]], pad_id: int) -> tuple[Tensor, Tensor]:
    return (
        pad_sequence([x for x, _ in items], batch_first=True, padding_value=pad_id),
        pad_sequence([y for _, y in items], batch_first=True, padding_value=-100),
    )


def prepare_dataset(
    directory: Path, version: str, seed: int = 2026
) -> dict[str, object]:
    import hashlib
    import json

    manifest = directory / "manifest.json"

    if manifest.exists() and json.loads(manifest.read_text())["version"] != version:
        raise ValueError("Use a different directory for each dataset version")

    paths = generate_dataset(directory, seed=seed, version=version)
    result: dict[str, object] = {
        "version": version,
        "seed": seed,
        "origin": "authored synthetic templates; no customer data",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "splits": {
            split: {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "examples": len(read_rows(path)),
                "template_groups": len({row["template_id"] for row in read_rows(path)}),
            }
            for split, path in paths.items()
        },
    }
    manifest.write_text(json.dumps(result, indent=2) + "\n")

    return result
