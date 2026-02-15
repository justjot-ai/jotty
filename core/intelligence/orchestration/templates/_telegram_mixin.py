"""Telegram Mixin - Report delivery via Telegram."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from Jotty.core.intelligence.orchestration.templates.swarm_ml_comprehensive import (
        TelegramConfig,
    )

logger = logging.getLogger(__name__)


class TelegramMixin:
    def init_telegram(self, config: "TelegramConfig" = None) -> None:
        """
        Initialize Telegram notifications.

        Args:
            config: Telegram configuration (uses env vars if None)
        """
        from Jotty.core.intelligence.orchestration.templates.swarm_ml_comprehensive import (
            TelegramConfig,
        )

        self._telegram_config = config or TelegramConfig()
        self._telegram_available = False

        if not self._telegram_config.enabled:
            return

        # Load .env file if available
        try:
            from dotenv import load_dotenv

            env_file = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".env")
            load_dotenv(os.path.abspath(env_file), override=False)
        except ImportError:
            pass

        # Get credentials from env if not provided (support both var names)
        if not self._telegram_config.bot_token:
            self._telegram_config.bot_token = os.environ.get(
                "TELEGRAM_BOT_TOKEN", ""
            ) or os.environ.get("TELEGRAM_TOKEN", "")
        if not self._telegram_config.chat_id:
            self._telegram_config.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

        if not self._telegram_config.bot_token or not self._telegram_config.chat_id:
            logger.warning(
                "Telegram credentials not found. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars."
            )
            return

        self._telegram_available = True
        logger.info("Telegram notifications initialized")

    def send_telegram_report(
        self, report_path: str, results: Dict[str, Any] = None, caption: str = None
    ) -> bool:
        """
        Send the PDF report to Telegram.

        Args:
            report_path: Path to the PDF report
            results: Optional results dict for summary message
            caption: Optional custom caption

        Returns:
            True if sent successfully
        """
        if not self._telegram_available:
            logger.warning("Telegram not available")
            return False

        try:
            import requests

            bot_token = self._telegram_config.bot_token
            chat_id = self._telegram_config.chat_id

            # Build caption
            if caption is None:
                caption = self._build_telegram_caption(results)

            # Send document
            if (
                self._telegram_config.send_report_pdf
                and report_path
                and os.path.exists(report_path)
            ):
                url = f"https://api.telegram.org/bot{bot_token}/sendDocument"

                with open(report_path, "rb") as f:
                    files = {"document": f}
                    data = {
                        "chat_id": chat_id,
                        "caption": caption[:1024],  # Telegram caption limit
                        "parse_mode": "HTML",
                    }
                    response = requests.post(url, files=files, data=data, timeout=60)

                if response.status_code == 200:
                    logger.info(f"Report sent to Telegram: {report_path}")
                    return True
                else:
                    logger.error(f"Telegram send failed: {response.text}")
                    return False

            # Send summary message only
            elif self._telegram_config.send_summary_message:
                return self.send_telegram_message(caption)

            return False

        except ImportError:
            logger.warning("requests library not installed for Telegram")
            return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send_telegram_message(self, message: str) -> bool:
        """
        Send a text message to Telegram.

        Args:
            message: Message text (supports HTML)

        Returns:
            True if sent successfully
        """
        if not self._telegram_available:
            return False

        try:
            import requests

            bot_token = self._telegram_config.bot_token
            chat_id = self._telegram_config.chat_id

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": message[:4096],  # Telegram message limit
                "parse_mode": "HTML",
            }
            response = requests.post(url, data=data, timeout=30)

            if response.status_code == 200:
                logger.info("Message sent to Telegram")
                return True
            else:
                logger.error(f"Telegram message failed: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Telegram message failed: {e}")
            return False

    def _build_telegram_caption(self, results: Dict[str, Any] = None) -> str:
        """Build caption for Telegram message."""
        lines = [
            "<b> ML Analysis Report</b>",
            f"<i>Generated by {self.name} v{self.version}</i>",
            "",
        ]

        if results:
            # Add metrics
            if self._telegram_config.include_metrics_in_message:
                score = results.get("final_score", 0)
                model = results.get("best_model", "N/A")

                lines.append(f" <b>Results:</b>")
                lines.append(f"  • Score: <code>{score:.4f}</code>")
                lines.append(f"  • Model: <code>{model}</code>")

                # Add other metrics if available
                for key in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
                    if key in results:
                        lines.append(f"  • {key.title()}: <code>{results[key]:.4f}</code>")

                lines.append("")

            # Add top features
            if self._telegram_config.include_feature_importance:
                importance = results.get("feature_importance", {})
                if importance:
                    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    top_n = self._telegram_config.max_features_in_message

                    lines.append(f" <b>Top {top_n} Features:</b>")
                    for feat, imp in sorted_imp[:top_n]:
                        lines.append(f"  • {feat[:20]}: <code>{imp:.3f}</code>")

        lines.append("")
        lines.append(f" {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        return "\n".join(lines)


# Convenience aliases moved to templates/__init__.py to avoid circular import
