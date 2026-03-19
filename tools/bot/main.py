#!/usr/bin/env python3
"""Telegram bot entrypoint."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from . import config as bot_config
from .handlers import (
    start_cmd,
    help_cmd,
    grades_cmd,
    leaderboard_cmd,
    grade_cmd,
    document_handler,
    text_handler,
)


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent.parent / ".env")
    except ImportError:
        pass

    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "bot.log", encoding="utf-8"),
        ],
    )

    if not bot_config.TELEGRAM_BOT_TOKEN:
        logging.error("TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)

    app = Application.builder().token(bot_config.TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("grades", grades_cmd))
    app.add_handler(CommandHandler("leaderboard", leaderboard_cmd))
    app.add_handler(CommandHandler("grade", grade_cmd))
    app.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
