from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HumanAnnotationGUIContractTest(unittest.TestCase):
    def test_structured_fields_use_preset_pickers(self) -> None:
        html = (ROOT / "scripts/annotation/web/index.html").read_text(encoding="utf-8")
        script = (ROOT / "scripts/annotation/web/app.js").read_text(encoding="utf-8")

        for picker_id in ("agent-picker", "target-picker", "participants-picker", "preset-picker-modal"):
            self.assertIn(f'id="{picker_id}"', html)

        self.assertNotIn('id="agent-input"', html)
        self.assertNotIn('id="target-input"', html)
        self.assertNotIn('id="participants-input"', html)
        self.assertNotIn('id="custom-flags-input"', html)
        self.assertNotIn('id="action-type-input"', html)
        self.assertIn('id="action-type-select"', html)
        for action_type in ("speak", "action", "other"):
            self.assertIn(f'<option value="{action_type}">{action_type}</option>', html)
        self.assertEqual(html.count('<option value="speak">speak</option>'), 1)
        self.assertIn("option-agent-picker", script)
        self.assertIn("option-target-picker", script)
        self.assertIn("option-participants-picker", script)
        self.assertIn("option-flags-picker", script)

    def test_context_is_large_and_source_alignment_is_on_demand(self) -> None:
        html = (ROOT / "scripts/annotation/web/index.html").read_text(encoding="utf-8")
        script = (ROOT / "scripts/annotation/web/app.js").read_text(encoding="utf-8")

        self.assertNotIn('class="evidence-section"', html)
        self.assertNotIn('id="evidence-state"', html)
        self.assertIn('id="source-alignment-button"', html)
        self.assertIn('id="source-alignment-modal"', html)
        self.assertIn('id="context-window-summary"', html)
        for radius in (8, 16, 32, 64):
            self.assertIn(f'<option value="{radius}"', html)
        self.assertIn("renderSourceAlignment", script)
        self.assertNotIn("renderEvidence", script)


if __name__ == "__main__":
    unittest.main()
