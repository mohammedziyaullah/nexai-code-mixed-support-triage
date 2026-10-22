"""Unit tests for support queue dashboard data models and state transitions."""

from __future__ import annotations

import unittest
from app import INITIAL_QUEUE_TICKETS


class TestQueueDashboard(unittest.TestCase):
    """Test suite verifying support queue schema, invariants, and operations."""

    def test_initial_queue_tickets_schema(self):
        """Verify initial seed tickets contain all mandatory keys and valid values."""
        self.assertGreaterEqual(len(INITIAL_QUEUE_TICKETS), 5)

        required_keys = {
            "ticket_id",
            "message",
            "language",
            "predicted_intent",
            "predicted_category",
            "confidence",
            "urgency",
            "urgency_reason",
            "triage_verdict",
            "escalation_reason",
            "status",
            "draft_reply",
        }

        for ticket in INITIAL_QUEUE_TICKETS:
            with self.subTest(ticket_id=ticket.get("ticket_id")):
                for key in required_keys:
                    self.assertIn(key, ticket)
                self.assertIn(ticket["urgency"], ["HIGH", "MEDIUM", "LOW"])
                self.assertIn(ticket["language"], ["English", "Hinglish", "Tanglish"])
                self.assertIn(ticket["triage_verdict"], ["ESCALATE_TO_HUMAN", "AUTOMATED_RESOLUTION"])
                self.assertGreaterEqual(ticket["confidence"], 0.0)
                self.assertLessEqual(ticket["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
