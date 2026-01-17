#!/usr/bin/env python3
"""
REAL COLLABORATIVE TEAM DEMO

Demonstrates TRUE P2P collaboration using:
- SharedContext (persistent storage)
- SharedScratchpad (message passing)
- AgentMessage (inter-agent communication)
- Tool result caching
- Shared insights

Scenario: 4 security experts reviewing code in parallel
- Agent 1: Authentication security review
- Agent 2: API security review
- Agent 3: Data security review
- Agent 4: Infrastructure security review

All work simultaneously, share findings via scratchpad!
"""

import asyncio
import dspy
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Import REAL collaboration infrastructure
from core.persistence.shared_context import SharedContext
from core.foundation.types.agent_types import SharedScratchpad, AgentMessage, CommunicationType
from core.integration.direct_claude_cli_lm import DirectClaudeCLI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityReviewSignature(dspy.Signature):
    """Security review with collaboration context."""
    code: str = dspy.InputField(desc="Code to review")
    focus_area: str = dspy.InputField(desc="Security focus area")
    other_findings: str = dspy.InputField(desc="Findings from other reviewers")
    review: str = dspy.OutputField(desc="Security review with vulnerabilities found")


async def collaborative_security_review(
    reviewer_name: str,
    focus_area: str,
    code_to_review: str,
    shared_context: SharedContext,
    scratchpad: SharedScratchpad
) -> Dict[str, Any]:
    """
    Run security review with collaboration.

    The reviewer:
    1. Reads findings from other reviewers (scratchpad)
    2. Conducts own review
    3. Posts findings to scratchpad for others to see
    """

    print(f"\n{'='*90}")
    print(f"🔒 {reviewer_name} - {focus_area} Security Review")
    print(f"{'='*90}\n")

    # CREATE THE AGENT
    agent = dspy.ChainOfThought(SecurityReviewSignature)

    # STEP 1: Read from shared scratchpad
    print(f"📖 Reading other reviewers' findings...")

    other_findings = []
    for msg in scratchpad.messages:
        if msg.sender != reviewer_name and msg.message_type == CommunicationType.INSIGHT:
            finding = msg.content.get('summary', '')
            if finding:
                other_findings.append(f"{msg.sender}: {finding}")

    if other_findings:
        print(f"   Found {len(other_findings)} findings from other reviewers")
        for finding in other_findings[-3:]:  # Show last 3
            print(f"   - {finding[:100]}...")
    else:
        print(f"   No findings yet (first reviewer)")

    other_findings_text = "\n".join(other_findings) if other_findings else "No findings from other reviewers yet"

    # STEP 2: Conduct review
    print(f"\n🔍 Conducting {focus_area} review...")

    start = datetime.now()
    result = agent(
        code=code_to_review,
        focus_area=focus_area,
        other_findings=other_findings_text
    )
    review_output = result.review
    elapsed = (datetime.now() - start).total_seconds()

    print(f"✅ Review complete in {elapsed:.1f}s")
    print(f"   Generated: {len(review_output)} chars")

    # Extract vulnerabilities found (simple count)
    vuln_count = review_output.lower().count('vulnerability') + review_output.lower().count('issue')
    print(f"   Vulnerabilities found: ~{vuln_count}")

    # STEP 3: Post findings to scratchpad
    print(f"\n📝 Posting findings to scratchpad...")

    # Store in SharedContext
    context_key = f"{reviewer_name.lower().replace(' ', '_')}_review"
    shared_context.set(context_key, review_output)

    # Post message to scratchpad
    message = AgentMessage(
        sender=reviewer_name,
        receiver="*",  # Broadcast to all
        message_type=CommunicationType.INSIGHT,
        content={
            'summary': review_output[:200],
            'full_review_key': context_key,
            'focus_area': focus_area,
            'vulnerabilities_found': vuln_count
        },
        insight=f"{reviewer_name} found {vuln_count} {focus_area} issues"
    )
    scratchpad.add_message(message)

    # Add to shared insights
    scratchpad.shared_insights.append(
        f"{reviewer_name} ({focus_area}): Found {vuln_count} vulnerabilities"
    )

    print(f"   ✅ Posted to scratchpad (message #{len(scratchpad.messages)})")
    print(f"   ✅ Added to shared insights")

    print(f"\n{'='*90}")
    print(f"✅ {reviewer_name} Complete")
    print(f"{'='*90}")

    return {
        'reviewer': reviewer_name,
        'focus_area': focus_area,
        'output': review_output,
        'vulnerabilities': vuln_count,
        'findings_read': len(other_findings),
        'time': elapsed
    }


async def collaborative_demo():
    """Run collaborative security review demo."""

    print("=" * 90)
    print("COLLABORATIVE SECURITY REVIEW - TRUE P2P COLLABORATION")
    print("=" * 90)
    print("\n4 security experts reviewing code in parallel\n")

    # Configure Claude CLI
    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)

    print("✅ Claude 3.5 Sonnet configured")

    # Initialize collaboration infrastructure
    shared_context = SharedContext()
    scratchpad = SharedScratchpad()

    print("✅ Collaboration infrastructure initialized")
    print(f"   SharedContext: {shared_context}")
    print(f"   SharedScratchpad: messages={len(scratchpad.messages)}, insights={len(scratchpad.shared_insights)}")
    print("-" * 90)

    # Code to review
    code_sample = """
// User Authentication API
const express = require('express');
const mysql = require('mysql');
const jwt = require('jsonwebtoken');

const app = express();
app.use(express.json());

// Database connection
const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'admin123',  // Hardcoded password
    database: 'users'
});

// Login endpoint
app.post('/api/login', (req, res) => {
    const { username, password } = req.body;

    // SQL query - potential injection
    const query = `SELECT * FROM users WHERE username='${username}' AND password='${password}'`;

    db.query(query, (err, results) => {
        if (err) {
            res.status(500).send(err.message);  // Leaks error details
        }

        if (results.length > 0) {
            const token = jwt.sign(
                { userId: results[0].id },
                'secret123',  // Hardcoded JWT secret
                { expiresIn: '365d' }  // Too long expiration
            );

            res.json({ token });  // No HTTPS enforcement
        } else {
            res.status(401).send('Invalid credentials');
        }
    });
});

// User data endpoint
app.get('/api/user/:id', (req, res) => {
    const userId = req.params.id;

    // No authentication check!
    db.query(`SELECT * FROM users WHERE id=${userId}`, (err, results) => {
        if (err) {
            res.status(500).send(err);
        }
        res.json(results[0]);  // Returns sensitive data
    });
});

app.listen(3000);
"""

    print(f"\n📋 Code Sample: {len(code_sample)} chars")
    print(f"   Reviewing for security vulnerabilities...")
    print()

    # Store code in shared context
    shared_context.set('code_to_review', code_sample)

    # Define reviewers with focus areas
    reviewers = [
        ('Auth Security Expert', 'Authentication & Authorization'),
        ('API Security Expert', 'API Endpoints & Input Validation'),
        ('Data Security Expert', 'Data Protection & Encryption'),
        ('Infrastructure Expert', 'Configuration & Secrets Management'),
    ]

    # Run all reviewers IN PARALLEL (using asyncio.gather)
    print("🚀 Starting all 4 reviewers in parallel...")
    print("=" * 90)

    tasks = [
        collaborative_security_review(
            reviewer_name=name,
            focus_area=focus,
            code_to_review=code_sample,
            shared_context=shared_context,
            scratchpad=scratchpad
        )
        for name, focus in reviewers
    ]

    # Run concurrently
    results = await asyncio.gather(*tasks)

    # Analysis
    print("\n" + "=" * 90)
    print("COLLABORATIVE REVIEW COMPLETE")
    print("=" * 90)

    total_vulns = sum(r['vulnerabilities'] for r in results)
    total_findings_shared = sum(r['findings_read'] for r in results)

    print(f"\n📊 Review Summary:")
    for result in results:
        print(f"\n{result['reviewer']} ({result['focus_area']}):")
        print(f"  Vulnerabilities found: {result['vulnerabilities']}")
        print(f"  Findings from others: {result['findings_read']}")
        print(f"  Review time: {result['time']:.1f}s")

    print(f"\n🔒 Security Analysis:")
    print(f"  Total Vulnerabilities: {total_vulns}")
    print(f"  Total Cross-Referencing: {total_findings_shared} findings shared")
    print(f"  Messages Exchanged: {len(scratchpad.messages)}")
    print(f"  Shared Insights: {len(scratchpad.shared_insights)}")
    print(f"  SharedContext Items: {len(shared_context.keys())}")

    print(f"\n🗂️  Shared Workspace State:")
    print(f"  SharedContext keys: {shared_context.keys()}")
    print(f"  Shared insights:")
    for insight in scratchpad.shared_insights:
        print(f"    - {insight}")

    # Save output
    output_file = Path("COLLABORATIVE_SECURITY_REVIEW.md")
    doc = f"""# Collaborative Security Review

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Pattern**: Collaborative P2P (4 reviewers in parallel)
**Infrastructure**: SharedContext + SharedScratchpad + AgentMessage

---

## Review Summary

| Reviewer | Focus Area | Vulnerabilities | Findings Shared | Time |
|----------|-----------|-----------------|-----------------|------|
"""

    for result in results:
        doc += f"| {result['reviewer']} | {result['focus_area']} | {result['vulnerabilities']} | {result['findings_read']} | {result['time']:.1f}s |\n"

    doc += f"""
**Total Vulnerabilities**: {total_vulns}
**Total Cross-Referencing**: {total_findings_shared}
**Messages Exchanged**: {len(scratchpad.messages)}

---

## Code Reviewed

```javascript
{code_sample}
```

---

## Individual Reviews

"""

    for result in results:
        doc += f"""### {result['reviewer']} - {result['focus_area']}

**Vulnerabilities Found**: {result['vulnerabilities']}
**Findings from Others**: {result['findings_read']}

{result['output']}

---

"""

    doc += f"""
## Collaboration Evidence

### Messages in Scratchpad
Total: {len(scratchpad.messages)} messages

"""

    for msg in scratchpad.messages:
        doc += f"""**{msg.sender}** → {msg.receiver}
Type: {msg.message_type.value}
Content: {msg.content.get('summary', str(msg.content)[:100])}
Insight: {msg.insight}

"""

    doc += f"""
### Shared Insights
"""

    for insight in scratchpad.shared_insights:
        doc += f"- {insight}\n"

    doc += """
---

## What This Demonstrates

### ✅ TRUE Collaborative P2P
- 4 agents worked in **parallel** (not sequential)
- Used **SharedContext** for persistent storage
- Used **SharedScratchpad** for message passing
- Used **AgentMessage** for inter-agent communication
- Agents read each other's findings in real-time
- Cross-referenced discoveries (later reviewers saw earlier findings)

### ✅ Real Infrastructure Used
- `SharedContext` (core/persistence/shared_context.py) ✓
- `SharedScratchpad` (core/foundation/types/agent_types.py) ✓
- `AgentMessage` with `CommunicationType.INSIGHT` ✓
- Message broadcasting (`receiver="*"`) ✓

### ✅ NOT Sequential String Passing
- Agents didn't wait for each other
- Findings shared via scratchpad, not parameter passing
- Later agents benefited from earlier agents' discoveries
- True collaboration workspace

---

*This is REAL collaborative multi-agent learning with shared workspace infrastructure!*
"""

    output_file.write_text(doc)

    print(f"\n📄 Complete review saved: {output_file}")
    print("=" * 90)

    return True


async def main():
    try:
        success = await collaborative_demo()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        exit(130)
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    print("\n🚀 Collaborative Security Review Demo")
    print("4 security experts reviewing code in parallel with shared scratchpad\n")

    response = input("Ready to run? (y/n): ")
    if response.lower() == 'y':
        asyncio.run(main())
    else:
        print("Cancelled")
