from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")


SUPPORT_TICKETS = [
    {
        "Ticket_ID": "TCK-1001",
        "Customer_Issue": "Customer cannot reset password because the reset link expires immediately after clicking it.",
        "Resolution": "Support cleared the stale reset token, verified the user timezone, and resent a new password reset email. The customer reset successfully.",
        "Category": "Authentication",
    },
    {
        "Ticket_ID": "TCK-1002",
        "Customer_Issue": "Invoice page shows duplicate charges after the customer upgraded from Basic to Pro.",
        "Resolution": "Billing confirmed the first charge was an authorization hold. Support explained the pending hold would disappear within 3 business days.",
        "Category": "Billing",
    },
    {
        "Ticket_ID": "TCK-1003",
        "Customer_Issue": "Webhook events are delayed by several hours for the customer's CRM integration.",
        "Resolution": "Engineering found the endpoint returned HTTP 429. Support advised rate-limit backoff and helped the customer rotate webhook delivery windows.",
        "Category": "Integrations",
    },
    {
        "Ticket_ID": "TCK-1004",
        "Customer_Issue": "User import CSV fails with an unknown column error when uploading employee records.",
        "Resolution": "Support identified an unsupported column named Department Name. Customer renamed it to department and the import completed.",
        "Category": "Data Import",
    },
    {
        "Ticket_ID": "TCK-1005",
        "Customer_Issue": "Admin cannot invite new teammates because invite emails never arrive.",
        "Resolution": "Support found the company domain had strict spam filtering. The customer allowlisted mail.product.example and invites were delivered.",
        "Category": "Account Management",
    },
    {
        "Ticket_ID": "TCK-1006",
        "Customer_Issue": "Dashboard analytics are missing yesterday's data for EU accounts.",
        "Resolution": "The data pipeline had a delayed EU warehouse job. Support shared the incident note and confirmed backfill completed at 14:30 UTC.",
        "Category": "Analytics",
    },
    {
        "Ticket_ID": "TCK-1007",
        "Customer_Issue": "API requests fail with 401 after the customer regenerated their API key.",
        "Resolution": "Support confirmed the old key was still configured in the customer's server environment. Updating the secret fixed authentication.",
        "Category": "API",
    },
    {
        "Ticket_ID": "TCK-1008",
        "Customer_Issue": "Mobile app crashes when opening a project with more than 500 tasks.",
        "Resolution": "Support recommended upgrading to mobile app version 4.8.2, which included a pagination fix for large projects.",
        "Category": "Mobile",
    },
]


KB_ARTICLES = [
    {
        "Article_ID": "KB-001",
        "Title": "Password reset troubleshooting",
        "Body": "Password reset links expire after 30 minutes and can be invalidated by multiple requests. Clear stale tokens, verify user timezone, then send one fresh reset email.",
        "Category": "Authentication",
    },
    {
        "Article_ID": "KB-002",
        "Title": "Understanding upgrade billing holds",
        "Body": "Plan upgrades may create an authorization hold before the final invoice settles. Pending holds normally disappear within 3 business days and are not duplicate payments.",
        "Category": "Billing",
    },
    {
        "Article_ID": "KB-003",
        "Title": "Webhook delivery and retry behavior",
        "Body": "Webhook endpoints should return 2xx quickly. Repeated 429 or 5xx responses trigger retries and delays. Use exponential backoff and monitor endpoint rate limits.",
        "Category": "Integrations",
    },
    {
        "Article_ID": "KB-004",
        "Title": "CSV user import schema",
        "Body": "The user import accepts email, first_name, last_name, role, and department columns. Unknown columns can cause validation errors during upload.",
        "Category": "Data Import",
    },
    {
        "Article_ID": "KB-005",
        "Title": "Email deliverability for invitations",
        "Body": "If invitation emails are not delivered, ask the customer to check spam quarantine and allowlist mail.product.example for their company domain.",
        "Category": "Account Management",
    },
    {
        "Article_ID": "KB-006",
        "Title": "API key rotation",
        "Body": "After regenerating an API key, update every server secret, CI variable, and integration credential that still references the previous key.",
        "Category": "API",
    },
]


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    tickets = pd.DataFrame(SUPPORT_TICKETS)
    articles = pd.DataFrame(KB_ARTICLES)

    tickets.to_csv(DATA_DIR / "support_tickets.csv", index=False)
    articles.to_csv(DATA_DIR / "kb_articles.csv", index=False)

    print(f"Wrote {len(tickets)} support tickets to {DATA_DIR / 'support_tickets.csv'}")
    print(f"Wrote {len(articles)} KB articles to {DATA_DIR / 'kb_articles.csv'}")


if __name__ == "__main__":
    main()
