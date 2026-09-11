from app.shared.clients.mongo_history_utils import (
    clear_history,
    get_recent_messages,
    save_chat_message,
    update_message_item_names,
)


class HistoryRepository:
    def list_recent(self, session_id: str, limit: int = 10) -> list[dict]:
        return get_recent_messages(session_id, limit=limit)

    def save_message(
        self,
        *,
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: list[str] | None = None,
        image_urls: list[str] | None = None,
        message_id: str | None = None,
    ) -> str:
        return save_chat_message(
            session_id=session_id,
            role=role,
            text=text,
            rewritten_query=rewritten_query,
            item_names=item_names,
            image_urls=image_urls,
            message_id=message_id,
        )

    def clear_session(self, session_id: str) -> int:
        return clear_history(session_id)


history_repository = HistoryRepository()