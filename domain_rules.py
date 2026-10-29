"""Optional business checks that cannot be inferred from JSON tool schemas.

Keep these rules separate from the general source verifier. A new app should
work without adding rules here; add one only for a measured semantic failure.
"""
from __future__ import annotations


def _latest_account_balances(observations):
    balances = {}
    for item in observations:
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        if item["name"] == "banking.get_balance":
            for account in result.get("accounts", []):
                if isinstance(account, dict) and "id" in account and "balance" in account:
                    balances[account["id"]] = account["balance"]
        elif item["name"] == "banking.transfer" and result.get("transaction"):
            transaction = result["transaction"]
            if "account_id" in transaction and "balance" in result:
                balances[transaction["account_id"]] = result["balance"]
        elif item["name"] == "banking.open_savings" and result.get("savings"):
            account_id = item.get("arguments", {}).get("from_account_id")
            if account_id and "balance" in result:
                balances[account_id] = result["balance"]
    return balances


def semantic_issues(name, args, transcript, observations, fold):
    """Return measured app-specific contradictions; default is no issues."""
    issues = []
    if name == "banking.transfer":
        account_id, amount = args.get("from_account_id"), args.get("amount")
        balance = _latest_account_balances(observations).get(account_id)
        if (isinstance(amount, (int, float)) and not isinstance(amount, bool)
                and isinstance(balance, (int, float)) and balance < amount):
            issues.append({
                "code": "insufficient_balance", "field": "from_account_id",
                "message": f"{account_id} có {balance}, không đủ chuyển {amount}.",
                "lookup_tools": [],
            })
    if name == "banking.open_savings":
        request = fold(transcript)
        explicit_goal = any(phrase in request for phrase in (
            "gui tiet kiem", "mo so tiet kiem", "mo tai khoan tiet kiem",
            "gui them vao tiet kiem", "lap so tiet kiem",
        ))
        if not explicit_goal:
            issues.append({
                "code": "wrong_goal", "field": None,
                "message": "open_savings là gửi thêm tiền tiết kiệm; request không yêu cầu mở/gửi tiết kiệm.",
                "lookup_tools": [],
            })
    return issues
